# OCCA

## 概述

本文档涵盖 CMeles 中使用 OCCA 的两方面内容：

1. **设备内存模型与数据初始化**——不同 OCCA 后端的内存空间差异及统一处理方式
2. **核函数开发约定**——OKL 核函数的编写规范、`@tile` 分块策略与跨后端统一原则

与场数据布局、核函数伪代码细节相关的讨论分散在各专项文档中（见[相关文档](#相关文档)），本文档仅聚焦于跨后端共通的设计约定。

---

## 设备内存模型

OCCA 各后端的内存模型存在根本差异，CMeles 需在托管端代码中统一处理。

### 统一内存空间后端（Serial / OpenMP）

Serial 和 OpenMP 后端中，设备内存与主机内存位于 **同一地址空间**。`occa::memory` 本质上是对 `malloc`/`new` 分配的普通 `void*` 的封装，host 端可直接读写该指针（`occa::memory::ptr<T>()`）。

### 分离内存空间后端（CUDA / HIP / OpenCL / Metal）

CUDA、HIP、OpenCL、Metal 等 GPU 后端中，设备内存与主机内存位于 **不同地址空间**。host 端不能直接解引用设备指针，必须通过 `copyTo`/`copyFrom` 显式传输数据。

### 运行时检测

OCCA 提供 `occa::device::hasSeparateMemorySpace()` 方法运行时查询当前设备的内存模型：

```cpp
occa::device device({{"mode", "CUDA"}});
bool separate = device.hasSeparateMemorySpace();  // true 为分离空间
```

`hasSeparateMemorySpace()` 返回值与各后端的对应关系：

| 后端        | `hasSeparateMemorySpace()` |
| :---------- | :------------------------: |
| Serial      | `false`                    |
| OpenMP      | `false`                    |
| CUDA        | `true`                     |
| HIP         | `true`                     |
| OpenCL      | `true`                     |
| Metal       | `true`                     |

---

## 数据初始化策略

基于上述内存模型差异，OCCA 数据对象（`occa::memory`）的创建方式有两种，CMeles 需根据不同后端选择合适的方式。

### 方式一：`wrapMemory`（仅统一内存空间）

`occa::device::wrapMemory()` 将 **已有 host 指针** 包装为 `occa::memory` 对象，不分配新内存、不拷贝数据。适用于 Serial、OpenMP 后端。

```cpp
// Serial / OpenMP: 直接包装 Eigen 向量指针
std::vector<double> host_data(N_total, 0.0);
occa::memory o_data = device.wrapMemory(host_data.data(), N_total);
// 此时 o_data.ptr<double>() == host_data.data()
```

> **注意**：`wrapMemory` 创建的 `occa::memory` 对象**不参与 OCCA 的垃圾回收**（引用计数为 0），需确保底层 host 内存的生命周期覆盖所有 OCCA 核函数调用的完成。

OCCA 的 `wrapMemory` API 定义见 `third_party/occa/docs/api/device/wrapMemory.md`。

### 方式二：`malloc` + `copyTo`/`copyFrom`（所有后端通用）

`occa::device::malloc()` 在设备上分配新内存并返回 `occa::memory` 对象。对于分离内存空间后端（GPU），这是唯一合法的数据创建方式；对于统一内存空间后端（Serial/OpenMP），其效果等价于 `malloc`/`new[]`。

```cpp
// 所有后端通用: 在设备上分配内存
occa::memory o_data = device.malloc<double>(N_total);

// 分离内存空间时，初始化和回读需显式拷贝
o_data.copyFrom(host_src);          // host → device
device.finish();                     // 等待异步拷贝完成
o_data.copyTo(host_dst);            // device → host
```

也可在 `malloc` 时通过 `src` 参数一次性完成分配+初始化：

```cpp
// 分配并同时从 host 指针初始化
occa::memory o_data = device.malloc<double>(N_total, host_src);
```

OCCA 的 `malloc` API 定义见 `third_party/occa/docs/api/device/malloc.md`。

### 方式三：`malloc` + `{"host": true}`（CUDA / HIP 专用优化）

CUDA 和 HIP 后端支持通过分配属性 `{"host": true}` 分配**主机端可访问的 device 内存**（即 CUDA 的 `cudaMallocHost` / pin-memory 或 HIP 的 `hipMallocHost`）：

```cpp
occa::memory o_data = device.malloc<double>(N_total, {"host": true});
// 分离空间后端上，此内存在 host 端也可直接读写
```

此方式适用于需要 host-device 高频交互的场景（如检查点输出、自适应网格的 host 端决策）。当前阶段无此需求，**暂不使用**。

---

## 统一内存管理接口设计

为避免 CMeles 中各处代码根据后端分别写 `if` 分支，设计 `DeviceMemoryManager` 类封装上述差异。

### 职责

`DeviceMemoryManager` 负责：

1. **持有 `occa::device` 引用**，管理设备生命周期
2. **提供统一的 `wrapOrMalloc` 方法**——自动根据 `hasSeparateMemorySpace()` 选择数据创建方式
3. **提供 `copyToHost` / `copyFromHost`**——统一数据同步接口，在统一内存空间上退化为空操作（或 `std::memcpy`）
4. **提供 `hostPinnedMalloc`**——GPU 端特定优化的分配接口，在非 GPU 端退化为普通 `malloc`

### 接口声明

```cpp
/// 统一 OCCA 内存管理接口，屏蔽后端内存模型差异
class DeviceMemoryManager {
public:
    explicit DeviceMemoryManager(occa::device& device);

    /// 返回持有的 occa::device 引用
    occa::device& device() { return device_; }

    /// 统一数据创建：统一空间用 wrapMemory，分离空间用 malloc
    /// @tparam T      数据类型
    /// @param src     host 端源指针（数据将被拷贝到设备，可为 nullptr）
    /// @param entries 元素个数
    occa::memory wrapOrMalloc(const double* src, occa::dim_t entries);

    /// 统一数据创建（零初始化版本）
    occa::memory wrapOrMalloc(occa::dim_t entries);

    /// device → host 同步
    /// 统一空间下退化为空操作（无拷贝），分离空间下执行 copyTo
    void copyToHost(occa::memory& o_data, double* dst, occa::dim_t entries);

    /// host → device 同步
    /// 统一空间下退化（无拷贝），分离空间下执行 copyFrom
    void copyFromHost(occa::memory& o_data, const double* src, occa::dim_t entries);

    /// 当前后端是否使用分离内存空间
    bool hasSeparateMemorySpace() const { return has_separate_; }

private:
    occa::device& device_;
    bool has_separate_;
};
```

### 关键实现

`wrapOrMalloc` 的核心逻辑——仅在分离内存空间后端分配新内存：

```cpp
occa::memory DeviceMemoryManager::wrapOrMalloc(
    const double* src, occa::dim_t entries)
{
    if (has_separate_) {
        // GPU 后端: 分配设备内存，可选择性地用 src 初始化
        return src ? device_.malloc<double>(entries, src)
                   : device_.malloc<double>(entries);
    } else {
        // CPU 后端: 直接包装指针（若无 src 则先分配 host 内存）
        // 调用者需保证 src 生命周期覆盖所有核函数执行
        return device_.wrapMemory<double>(src, entries);
    }
}
```

`copyToHost` 的核心逻辑——统一空间下无实际拷贝：

```cpp
void DeviceMemoryManager::copyToHost(
    occa::memory& o_data, double* dst, occa::dim_t entries)
{
    if (has_separate_) {
        o_data.copyTo(dst, entries * sizeof(double));
    } else {
        // 统一空间: 指针相同，无需拷贝
        // 若需验证可添加 assert(o_data.ptr<double>() == dst);
    }
}
```

### 使用示例

```cpp
// 初始化
occa::device device(occa::json::parse(config.at("occa").as_string()));
DeviceMemoryManager mem_mgr(device);

// 创建场数据——调用者无需关心后端
std::vector<double> u_init(N_total, 0.0);
// ... 填充初始条件 ...
occa::memory o_u = mem_mgr.wrapOrMalloc(u_init.data(), N_total);

// 计算完成后回读结果
std::vector<double> u_result(N_total);
mem_mgr.copyToHost(o_u, u_result.data(), N_total);
```

### 注意事项

- `wrapMemory` 路径下，`occa::memory` 对象的生命周期由 OCCA 的引用计数管理，但底层 host 内存的释放由调用者负责。在 `wrapOrMalloc` 返回后不可释放 `src` 指向的内存。
- `copyToHost` / `copyFromHost` 在统一空间下的"空操作"优化依赖 `hasSeparateMemorySpace()` 的正确性，该值在 `DeviceMemoryManager` 构造时确定，不可在运行时变更。
- 当前设计基于 `double` 类型。后续若需泛型化可增加模板参数。

---

## OCCA/OKL 核函数开发约定

### 统一 `@tile` 分块策略

所有后端（GPU CUDA/HIP、CPU OpenMP/Serial）统一使用 `@tile` 对 elem 循环分块，无需宏定义切换不同代码路径。

- 使用 `@tile(TILE_SIZE, @outer, @inner)` 对 `elem`（单元编号）循环分块。`@tile` 自动拆分为外层 `@outer`（GPU block / OpenMP `#pragma omp parallel for`）和内层 `@inner`（GPU thread / 串行退化）。elem 内全部串行。
- 当前不考虑 `@shared` 共享内存，所有数据读写走全局内存。

> 分块策略的详细描述（目标硬件绑定行为、SIMD 向量化说明、`@outer`/`@inner` 语义、`TILE_SIZE` 选择依据等）见[控制方程与 DG 场](governed_equations_and_DG_field.md#occa-共享内存分块形状)。

### 平台间差异

仅 `TILE_SIZE` 参数根据不同平台调优，OKL 源码本身完全一致：

| 平台 | TILE_SIZE 推荐值 | 说明                     |
| :--- | :--------------: | :----------------------- |
| GPU  |       256        | 匹配 GPU warp/wavefront  |
| CPU  |      按核数      | 通常设为物理核数的整数倍 |

### 关键技术要点

- **SIMD 向量化**：不依赖 OKL 的 `@inner` 标注。实测 OCCA OpenMP 后端对 `@inner` 不生成任何 `#pragma omp simd`。SIMD 向量化由编译器在 `-O3 -march=native` 下自动完成，OKL 源码中无需额外标注。
- **JIT 缓存**：不使用 `@shared` 时，`N_VARS`、`N_MODES`、`N_Q` 等维度作为 `const int` 核函数参数传入（非编译宏），最大化 JIT 缓存命中率。
- **`@shared` 的限制**：若后续启用 `@shared`，数组大小必须为编译时常量，需通过 JIT 编译宏（`occa::kernelBuilder` 的 `addDefine`）传入。
- **禁止使用 C++ 引用传参**：OKL 核函数中**不允许**使用 `const T & value` 形式的 C++ 引用参数。OCCA 的 JIT 代码生成器（`launcher_source.cpp`）在生成 OpenCL kernel source 时可能原样输出引用语法，这在 OpenCL C 中是非法的（OpenCL C 只允许指针，不支持引用），会导致 `CL_BUILD_PROGRAM_FAILURE`。核函数的所有参数应通过**值传递**或**指针传递**，避免引用语法。**注意**：此规则仅适用于 `.okl` 核函数源码；托管端（host）C++ 代码中的 `occa::memory` 等 OCCA 对象正常使用值/引用传递无影响。

### 参考文档

OCCA 的完整文档位于 `third_party/occa/docs/`，CMeles 开发中最常查阅的部分：

- `third_party/occa/docs/guide/okl/introduction.md` — OKL 基本概念（`@outer`/`@inner`/`@shared`/`@exclusive`）
- `third_party/occa/docs/guide/okl/loops-in-depth.md` — OKL 循环规则
- `third_party/occa/docs/guide/okl/attributes.md` — OKL 属性（`@dim`、`@tile` 等）
- `third_party/occa/docs/api/kernel/` — C++ API（`buildKernel`、`run`、`setRunDims` 等）
- `third_party/occa/examples/cpp/` — OKL 核函数示例（了解 `@outer`/`@inner`/`@tile`/`@shared` 的实际语法和用法时优先查阅）
- `third_party/occa/docs/api/device/wrapMemory.md` — `wrapMemory` API
- `third_party/occa/docs/api/device/hasSeparateMemorySpace.md` — `hasSeparateMemorySpace` API
- `third_party/occa/docs/api/device/malloc.md` — `malloc` API（含 `{"host": true}` 共享内存分配）

> **规则**：在处理 OCCA/OKL 相关问题时，必须先阅读上述文档确认语法和限制。关键语法规则：
> - `@outer` 和 `@inner` 是**裸属性**，不用括号或数字：`for (...; @outer)`，**不是** `@outer(0)`
> - OKL **支持嵌套 `@outer`**（多维 block grid，如 `fd2d.okl`），**也支持顺序 `@outer`**（不嵌套的多个 `@outer` 之间顺序执行）
> - 多个 `@inner` 循环**允许存在**但要求**迭代次数必须相同**
> - `@shared` 数组大小必须为编译时常量（可通过 JIT 宏传入）

---

## 相关文档

- [控制方程与 DG 场](governed_equations_and_DG_field.md) — EVM 数据布局、设备端场数据、OKL 分块策略细节、`@outer`/`@inner` 在各后端的映射表
- [输入输出模块](io_module.md) — GPU 数据输出时的 `copyTo` 用法
