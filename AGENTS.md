每次回答都要称呼我为 "Helldiver"。

# CMeles

## 项目概述

CMeles 是一个基于 C++ 的可压缩流动求解器，采用间断 Galerkin（Discontinuous Galerkin, DG）方法进行空间离散化。项目面向学术研究，兼顾计算性能与代码可扩展性。

## 技术栈

| 组件                 | 用途                                                         |
| -------------------- | ------------------------------------------------------------ |
| **OCCA** (`libocca`) | 异构计算后端，支持 CPU 和 GPU，核函数统一用 OKL 编写         |
| **Eigen**            | 数据准备阶段的矩阵/向量运算（如 Vandermonde 矩阵、质量矩阵） |
| **HighFive**         | HDF5 文件 I/O，用于结果输出                                  |
| **toml++**           | TOML 格式配置文件解析                                        |

## 项目结构

- `CMeles.cpp` — 程序入口
- `src/` — 源码目录
- `tests/` — 测试用例（环境验证测试直接放在该目录下，其他测试用例按功能放入对应子目录）
- `docs/` — 项目文档（中文内容，英文文件名）（安装指南、使用说明、代码结构、算法原理等）

## 代码规范

### 编码风格

- 使用 `.clang-format` 定义的代码风格，提交前应格式化代码
- 使用 C++17 或更新标准
- 优先使用现代 C++ 特性（`constexpr`、`std::unique_ptr`/`std::shared_ptr`、lambda 表达式、结构化绑定等）

### 注释

- 使用 LLVM 注释规范，注释语言为英文
- 作为学术开发代码，关键算法步骤和物理含义处应有适当的行内注释
- 注释中的公式采用 LaTeX 语法，使用 `$ ... $` 包裹行内公式，`$$ ... $$` 包裹独立公式；对于VSCode编辑器，安装 Comment Formula 插件可实现公式渲染预览。

### 设计原则

- **多态按边界粒度选择**：粗粒度接口（每物理步/每次求解调用一次的边界，如 `StepperBase::advance()`、solver 控制器）使用虚函数继承——虚调用开销相对阶段内设备内核可忽略，且避免模板实例化膨胀；细粒度热路径（每 DOF/每单元的内层循环）才优先 CRTP 等编译期多态，避免逐次调用的虚开销
- **RAII**：严格遵循 RAII 原则管理资源（内存、文件句柄、设备上下文等），确保异常安全
- **设计模式**：根据场景合理使用工厂模式、单例模式等提升可维护性

### 精度控制

项目使用类型别名 `Real` 统一管理浮点精度，定义在 `src/common/Types.hpp`：

```cpp
#ifdef USE_FLOAT_PRECISION
using Real = float;
constexpr Real RealEpsilon = Real(1e-7);
#else
using Real = double;
constexpr Real RealEpsilon = Real(1e-15);
#endif
```

**使用规则**：

- 所有项目代码应使用 `Real` 代替 `double`/`float` 直接写类型名
- Eigen 类型使用便捷别名：`VectorXr`（向量）、`MatrixXr`（行优先矩阵）、`MatrixXrCol`（列优先矩阵）、`MatrixX2r`（N×2 行优先矩阵）
- 精度敏感的常量（如迭代收敛判据、零值比较）使用 `RealEpsilon` 而非硬编码字面量
- 整数运算中的字面量使用 `Real(...)` 包装以避免精度转换问题（如 `Real(1) / std::sqrt(Real(2))`）

**CMake 选项**：

```bash
cmake -B build -DUSE_FLOAT_PRECISION=ON   # 使用 float（32-bit）
cmake -B build                            # 使用 double（64-bit，默认）
```

`USE_FLOAT_PRECISION` 宏通过 `target_compile_definitions` 传递给所有编译目标（主目标和测试目标），因此所有包含 `Types.hpp` 的文件自动切换精度。

## OCCA/OKL 核函数开发

所有计算核函数定义在 `.okl` 文件中。开发约定详见 `docs/tech_docs/occa.md`，关键规则：

- **所有后端统一**：使用 `@tile(TILE_SIZE, @outer, @inner)` 对 elem 循环分块。`@tile` 自动拆分为外层 `@outer`（GPU → block、OpenMP → `#pragma omp parallel for`）和内层 `@inner`（GPU → thread、OpenMP/Serial → 串行），elem 内全部串行。当前不考虑 `@shared` 共享内存。
- **`@outer` 和 `@inner` 是裸属性**，不用括号或数字：`for (...; @outer)`，**不是** `@outer(0)`
- OKL 支持嵌套 `@outer`（多维 block grid），也支持顺序 `@outer`
- 多个 `@inner` 循环允许存在，但迭代次数必须相同
- `@shared` 数组大小必须为编译时常量（可通过 JIT 宏传入）
- **SIMD 向量化**：不依赖 `@inner` 标注，由编译器在 `-O3 -march=native` 下自动完成
- **设备内存管理**：通过 `DeviceMemoryManager` 封装 `wrapMemory`（统一内存空间后端）和 `malloc`+`copy`（分离内存空间后端）的差异

## 数据输出

使用 HighFive 库将计算结果写入 HDF5 文件，数据采用 SoA（Structure of Arrays）布局直接保存积分点数据。

## Git 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范，commit message 使用英文。格式：

```
<type>(<scope>): <description>

[optional body]
```

常用类型：`feat`、`fix`、`docs`、`refactor`、`test`、`chore`、`perf`。

## 文档

项目文档使用 Markdown 格式，放置在 `docs/` 目录下。文档应涵盖安装方法、使用指南、代码架构和核心算法原理。
