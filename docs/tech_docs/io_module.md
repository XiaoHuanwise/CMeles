# 输入输出模块

本文档描述 CMeles 中程序的输入（TOML 配置文件解析与命令行参数处理）和输出（HDF5 格式流场数据写入）模块的架构设计、数据格式与代码实现方案。

> **实现状态（v1.1）**：HDF5 输出已实现，代码位于 `src/io/`（`HDF5Writer`、`SolutionWriter`、`CheckpointWriter`、`FieldOutput`、`CheckpointConvert`）与 `tools/CMelesConvert.cpp`，测试见 `ctest/io/test_hdf5_output.cpp`。要点与初版设计的差异：
>
> - **计算数据布局**：实际计算状态 `DgField::o_u()` 是模态系数，布局 `[elem][var][mode]`（EVM），而非下文历史版本假设的积分点 AoS `QPoint`。策略 A 输出前在主机端做一次模态→积分点的 Vandermonde 投影 $u(\xi_q) = \sum_m \hat u_m \phi_m(\xi_q)$（即 `DgField::setInitialConditionNodal` L2 投影的逆过程），随后按物理量拆分 SoA 写出。
> - **输出默认关闭**：`[output] enable = false`（默认），关闭时求解器不构造输出器、不建目录、零文件副作用；现有测试均不受影响。
> - **输出变量**：固定写出 7 个数据集——4 个守恒变量 `rho, rho_u, rho_v, E` 与 3 个派生原始变量 `u, v, p`（`format`、`variables` 选择键未实现，预留扩展）。
> - **同步输出**：输出在时间步间同步执行（毫秒级、每 interval 步一次），异步流水线按 profiling 结果另议。
> - **坐标源**：积分点物理坐标由 `quadraturePhysicalCoords`（`src/dg/QuadratureCoords.{hpp,cpp}`）提供，展平顺序 $k = e \cdot N_q^2 + q$。

---

## 一、读入模块：TOML 配置解析

CMeles 使用 [toml++](https://marzer.github.io/tomlplusplus/) 库解析 TOML 格式的配置文件，将用户设定的运行参数读入统一的配置结构体。配置文件的路径通过命令行参数传入。

### 1.1 模块架构

读入模块的设计遵循**单一职责原则**，划分为三个层次：

```
命令行参数        TOML 配置文件
     │                  │
     ▼                  ▼
┌─────────────┐   ┌──────────────┐
│ ArgParser   │   │ ConfigParser │
│ (CLI 解析)  │   │ (TOML 解析)  │
└──────┬──────┘   └──────┬───────┘
       │                 │
       └────────┬────────┘
                ▼
        ┌──────────────┐
        │  Config      │
        │  (配置结构体) │
        └──────────────┘
```

- **ArgParser**：解析命令行参数，提取配置文件路径、输出目录等必要的运行时信息。使用标准库 `std::string_view` 和简单的字符串匹配，避免引入第三方 CLI 框架。
- **ConfigParser**：调用 toml++ 解析 TOML 文件，将键值对映射到 `Config` 结构体的对应字段。包含类型检查、范围校验和默认值回退逻辑。
- **Config**：聚合所有配置数据的纯数据结构（聚合体），不包含任何解析逻辑。所有字段均提供有意义的默认值，确保即便配置文件部分缺失，程序也能以合理参数运行。

### 1.2 命令行参数设计

命令行接口遵循 POSIX 惯例，采用位置参数与可选标志的组合方式：

```
CMeles <config_file> [options]

位置参数:
  config_file         TOML 配置文件的路径（必需）

可选参数:
  -o, --output <dir>  结果输出目录（默认: ./output）
  -v, --verbose       启用详细日志输出
  -h, --help          显示帮助信息并退出
  --version           显示版本信息并退出
```

#### 实现方案

考虑到项目定位为学术研究代码，不引入如 CLI11 等额外的 CLI 解析库。ArgParser 使用简单的循环遍历 `argv`：

```cpp
struct ArgParser {
    std::string config_path;
    std::string output_dir = "./output";
    bool verbose = false;

    static ArgParser parse(int argc, char* argv[]);
};
```

对于 `--help` 和 `--version`，直接输出信息后调用 `std::exit(0)`。解析错误时输出到 `stderr` 并以非零退出码终止。

#### 与配置解析的衔接

`ArgParser` 解析完成后，将 `config_path` 传递给 `load_config()` 完成 TOML 解析，最终在 `main()` 中将两者串联：

```cpp
int main(int argc, char* argv[]) {
    try {
        auto args = ArgParser::parse(argc, argv);
        auto config = load_config(args.config_path);

        // 命令行 -o 覆盖配置文件中的 output.directory
        if (!args.output_dir.empty())
            config.output.directory = args.output_dir;

        // ... 初始化求解器并运行 ...
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
```

**全局异常处理策略**：`main()` 作为统一的异常捕获边界。所有模块（ArgParser、ConfigParser、求解器、输出器）在遇到不可恢复的错误时抛出 `std::exception` 子类异常，由 `main()` 最外层 `catch` 统一格式化输出到 `stderr` 后以 `EXIT_FAILURE` 退出。这避免了各模块各自调用 `exit()` 导致的资源泄露风险，并保证错误信息的格式一致性。

### 1.3 配置文件格式

TOML 配置文件按功能域组织为多个表（table），每个表对应 DG 求解器的一个子系统。以下为顶层结构设计：

```toml
# CMeles 配置文件
# 所有带等号的参数均可选，未指定时使用程序内置默认值

[mesh]
file = "mesh.h5"          # 网格文件路径（HDF5 格式）
type = "quad"             # 单元类型："quad" | "tri"
N_cells = [32, 32]        # 各方向单元数（用于生成结构化网格）

[solver]
order = 3                 # 多项式阶数 p，基函数个数 = (p+1)^d
T_final = 1.0             # 最终物理时间
CFL = 0.5                 # CFL 数
riemann_solver = "HLLC"   # 数值通量格式："HLLC" | "Roe" | "LaxFriedrichs"

[time_marching]
method = "SSPRK3"         # 时间推进方法："Euler" | "SSPRK3"
dt = 1e-4                 # 时间步长（method 为显式定步长时使用）

[physics]
gamma = 1.4               # 比热比
Pr = 0.72                 # 普朗特数
Re = 1000.0               # 雷诺数
Ma = 0.3                  # 参考马赫数

[initial_condition]
type = "uniform"          # 初场类型："uniform" | "vortex" | "shock_tube"
# 各类型对应的参数...
rho = 1.0
u = 0.0
v = 0.0
p = 1.0

[output]
enable = false            # 总开关：false（默认）时不产生任何文件
strategy = "direct"       # 输出策略："direct" (策略 A) | "checkpoint" (策略 B)
interval = 100            # 每隔 N 个时间步输出一次（enable 时须 >= 1）
directory = "./results"   # 输出目录（不存在则自动创建）

[log]
level = 1                 # 日志输出深度：消息深度 <= level 时才输出
                          # 0 = 完全静默；1（默认）= 物理步层（banner、
                          # 进度行、abort、结束摘要）；2 = 内迭代层（双时间
                          # 步伪迭代统计）；更深层级后续按需引入
interval = 100            # 进度行打印间隔（时间步数，>= 1）
```

> 注：`format` 与 `variables` 键未实现（当前仅 HDF5 一种格式、总是全量输出 7 个物理量），预留为未来扩展。`enable = true` 时输出时机为：初始场（step 0）+ 每 `interval` 步 + 终态（与最后一次间隔输出重合时去重）。

### 1.4 配置结构体设计

#### 设计原则

1. **值语义与高效复制**：`Config` 及其所有子结构体应支持高效的复制和移动，便于在初始化阶段传递配置。
2. **默认值即合理值**：所有参数在默认构造后即可使求解器以最基本的设置运行（如低阶、小网格、Euler 方程），降低新用户的调试门槛。
3. **类型安全的分层嵌套**：使用嵌套结构体按功能域组织参数（`Config::Mesh`、`Config::Solver`、`Config::Physics` 等），避免扁平的"大杂烩"结构。

#### 结构体骨架

```cpp
struct Config {
    // ---- 网格参数 ----
    struct Mesh {
        std::string file;           // 外部网格文件路径（空串表示自动生成）
        std::string type = "quad";  // 单元类型
        std::array<int, 2> N_cells = {16, 16};  // 各方向单元数
    } mesh;

    // ---- 求解器参数 ----
    struct Solver {
        int order = 2;                    // 多项式阶数
        double T_final = 1.0;             // 终止时间
        double CFL = 0.5;                 // CFL 数
        std::string riemann_solver = "HLLC";
    } solver;

    // ---- 时间推进参数 ----
    struct TimeMarching {
        std::string method = "SSPRK3";    // 时间推进方法
        double dt = 1e-4;                 // 固定时间步长
    } time_marching;

    // ---- 物理参数 ----
    struct Physics {
        double gamma = 1.4;
        double Pr = 0.72;
        double Re = 1000.0;
        double Ma = 0.3;
    } physics;

    // ---- 初场参数 ----
    struct InitialCondition {
        std::string type = "uniform";
        double rho = 1.0, u = 0.0, v = 0.0, p = 1.0;
    } initial_condition;

    // ---- 输出参数 ----
    struct Output {
        std::string format = "h5";
        std::string strategy = "direct";  // "direct" | "checkpoint"
        int interval = 100;
        std::string directory = "./results";
        std::vector<std::string> variables = {"rho", "u", "v", "p", "E"};
    } output;
};
```

### 1.5 TOML 解析实现

#### 解析流程

```cpp
Config load_config(const std::string& path) {
    // Step 1: 解析 TOML。toml::parse_file 内部处理文件打开和解析，
    // 文件不存在、权限不足、语法错误等情况均抛出 toml::parse_error
    toml::table tbl;
    try {
        tbl = toml::parse_file(path);
    } catch (const toml::parse_error& e) {
        // toml::parse_error 包含 source() 可区分错误来源
        throw std::runtime_error(
            "TOML 解析错误 [" + path + "]:\n" + e.what()
        );
    }

    // Step 2: 逐域提取参数，使用 toml++ 的 type-safe 访问接口
    Config cfg;

    // 使用 visit + lambda 的链式提取，缺省时保留 Config 默认值
    if (auto* mesh_tbl = tbl["mesh"].as_table()) {
        cfg.mesh.file   = mesh_tbl->get("file")->value_or(""sv);
        cfg.mesh.type   = mesh_tbl->get("type")->value_or("quad"sv);
        // 数组类型需单独处理
        if (auto arr = mesh_tbl->get("N_cells")->as_array()) {
            for (size_t i = 0; i < std::min(arr->size(), cfg.mesh.N_cells.size()); ++i)
                cfg.mesh.N_cells[i] = arr->at(i).value_or(cfg.mesh.N_cells[i]);
        }
    }

    if (auto* solver_tbl = tbl["solver"].as_table()) {
        cfg.solver.order   = solver_tbl->get("order")->value_or(2);
        cfg.solver.T_final = solver_tbl->get("T_final")->value_or(1.0);
        cfg.solver.CFL     = solver_tbl->get("CFL")->value_or(0.5);
        cfg.solver.riemann_solver =
            solver_tbl->get("riemann_solver")->value_or("HLLC"sv);
    }

    // ... 其余表同理 ...

    // Step 3: 参数合法性校验
    validate_config(cfg);

    return cfg;
}
```

#### 关键实现细节

**关于 `toml::parse_file` vs `toml::parse`**：当前设计使用 `toml::parse_file`，它内部处理文件打开和解析，通过 `toml::parse_error::source()` 返回 `toml::source_region` 可区分错误来源（文件访问失败 vs 语法错误），但无法对两者设置独立的异常处理路径。备选方案是使用 `std::ifstream` + `toml::parse`，优势在于可以更精细地控制错误报告粒度——在 `ifstream` 打开阶段独立处理 I/O 错误，将文件打开和格式解析完全分离。对于学术代码，`toml::parse_file` 的简约性更符合需求。

**路径处理**：TOML 中的相对路径（如 `mesh.file`）相对于**配置文件所在目录**解析，而非当前工作目录。这通过 `std::filesystem::path(config_path).parent_path() / mesh_file` 实现，确保无论从何处调用程序，配置文件内的相对引用始终有效。

**校验函数**：`validate_config` 在解析完成后执行，检查参数间的逻辑一致性：

| 校验项 | 条件 | 错误信息示例 |
|--------|------|-------------|
| 多项式阶数 | `order >= 1` | `solver.order 必须 ≥ 1，当前值: 0` |
| CFL 数 | `CFL > 0` | `solver.CFL 必须为正数` |
| 比热比 | `gamma > 1.0` | `physics.gamma 必须 > 1` |
| 输出间隔 | `interval >= 0` | `output.interval 不能为负` |
| 日志等级 | `level >= 0`（无上限，更深值合法） | `Config: log.level must be >= 0` |
| 日志间隔 | `interval >= 1` | `Config: log.interval must be >= 1` |
| 时间步长 | `dt > 0`（显式方法时） | `定步长显式方法的 dt 必须为正数` |
| Riemann 求解器 | 必须在支持列表中 | `未知的 riemann_solver，支持: HLLC, Roe, ...` |

校验失败时抛出 `std::invalid_argument` 并附带清晰的错误描述，帮助用户快速定位配置问题。

---

## 二、输出模块：HDF5 数据写入

CMeles 使用 [HighFive](https://github.com/BlueBrain/HighFive) 库将流场计算结果写入 HDF5 文件。HighFive 是 HDF5 C 库的现代 C++ 封装，提供 RAII 资源管理和直观的模板化接口。

### 2.1 模块架构

输出模块的核心职责是：在指定的输出时间步，将 DG 场的计算数据持久化到 HDF5 文件。计算状态是**模态系数**（布局 `[elem][var][mode]`，EVM）：策略 A 在主机端将其投影到积分点并按物理量拆分 SoA 写出；策略 B 直接落盘原始模态系数，转换推迟到离线工具。图 1 描述策略 A 从模态计算格式到 SoA 输出格式的转换关系：

```
计算阶段 (modal, EVM)                 HDF5 文件结构 (SoA)
┌──────────────────────────┐        ┌──────────────────────┐
│ 单元0: [û⁰₀..û⁰ₘ û¹₀....]│        │ /rho   (Dataset)     │
│ 单元1: ...                │ V·û    │ /rho_u (Dataset)     │
│ 单元2: ...                │ ────▶  │ /rho_v (Dataset)     │
│ ...                      │ 投影   │ /E /u /v /p          │
│ [elem][var][mode]        │        │ /time  (Attribute)   │
│                          │        │ /step  (Attribute)   │
│                          │        │ /mesh/ (Group)       │
│                          │        │   /mesh/x            │
└──────────────────────────┘        │   /mesh/y            │
                                    └──────────────────────┘
```

#### 设计要点

- **模态 → 积分点投影 + SoA 拆分**：输出阶段先把设备端模态系数拷回主机，逐单元逐变量计算 $u(\xi_q) = \sum_m \hat u_m \phi_m(\xi_q)$（Vandermonde 矩阵 $V_{qm} = \phi_m(\xi_q)$ 一次 GEMV），再按物理量拆分为独立数组和派生的原始变量（$u = \rho u/\rho$、$p = (\gamma-1)(E - \tfrac12\rho|\vec u|^2)$），写入对应 HDF5 Dataset。
- **元数据以 Attribute 形式附加**：当前物理时间、时间步数等标量信息作为 File 级 Attribute 写入，方便后处理工具读取。
- **OCCA 设备内存**：GPU 计算时需先取回主机。统一使用 `occa::memory::copyTo`（在统一与分离内存后端均有效）；注意 `DeviceMemoryManager::copyToHost` 在统一内存后端是 no-op，不能用于填充输出器自有缓冲。

### 2.2 数据布局：模态计算格式与 SoA 输出格式

CMeles 采用**双格式策略**：计算阶段使用单元主序的模态系数（EVM），输出阶段转换为积分点上的 SoA（Structure of Arrays）以便后处理。

#### 模态系数：计算格式（`[elem][var][mode]`，EVM）

`DgField::o_u()` 按单元主序存储每个单元、每个守恒变量的模态系数 $\hat u^{(m)}_e$（单元 $e$、变量 $m$、模态 $i$），大小 $N_e \cdot N_v \cdot N_{\text{modes}}$，其中 $N_{\text{modes}} = (N+1)(N+2)/2$。该布局与全部 OKL 核函数（体积积分、面通量、装配）一致。

#### SoA：输出格式

后处理工具通常按物理量维度读取（如绘制密度云图时只读取 `rho` 数据集），因此输出时每个物理量写成独立的一维 Dataset：

| 对比维度     | 模态（计算用）            | SoA（输出用）                |
| ------------ | ------------------------- | ---------------------------- |
| 核函数访存   | EVM 连续，kernel 原生布局 | 跨步读取，不用于计算        |
| 访问单物理量 | 跨步读取                  | 连续读取，cache 完全利用     |
| 输出灵活性   | 必须整体转换              | 每个物理量独立数据集         |
| HDF5 存储    | 需要布局知识才能解读      | 简单数组，所有 HDF5 工具通用 |

设网格包含 $N_e$ 个单元、每单元 $N_q^2$ 个体积积分点，$N_{\text{tot}} = N_e \times N_q^2$。投影与拆分在一次主机端遍历中完成：

$$ u^{(m)}_e(\xi_q) = \sum_{i=0}^{N_{\text{modes}}-1} \hat u^{(m)}_{e,i} \, \phi_i(\xi_q) = (V \hat u^{(m)}_e)_q , $$

其中 $V \in \mathbb{R}^{N_q^2 \times N_{\text{modes}}}$ 是 `BasisFunctions2D::vandermonde()`，投影即 `DgField::setInitialConditionNodal`（nodal→modal L2 投影）的逆过程。随后派生原始变量并按 $k = e \cdot N_q^2 + q$ 展平为各物理量的连续数组：

```
rho   : [ρ₀,    ρ₁,    ..., ρ_{Ntot-1}   ]  ← 连续存储，Ntot 个 Real
rho_u : [ρu₀,   ρu₁,   ..., ρu_{Ntot-1}  ]
rho_v : [ρv₀,   ρv₁,   ..., ρv_{Ntot-1}  ]
E     : [E₀,    E₁,    ..., E_{Ntot-1}   ]
u, v, p : 由守恒变量派生的原始变量数组
```

SoA 的展平顺序与积分点坐标数据集（`mesh/x`、`mesh/y`，来自 `quadraturePhysicalCoords`）的行序一一对应。

> **设计取舍**：投影 + SoA 拆分引入 $7 \times N_{\text{tot}}$ 个 `Real` 的额外主机内存与一次全量遍历，对典型问题规模（$10^4 \sim 10^6$ 积分点）在数 MB 至数十 MB 量级，每 interval 步发生一次，完全可接受。

### 2.3 HDF5 文件格式规范

#### 文件命名

输出文件命名为 `solution_<step>.h5`，其中 `<step>` 为零填充的时间步编号（如 `solution_0000100.h5`）。零填充宽度由 $\lfloor\log_{10}(\text{总步数})\rfloor + 1$ 确定，确保文件按名称排序即按时间顺序排序。

#### 顶层结构

```
/                           (Root Group)
├── @time: double           (Attribute) 当前物理时间
├── @step: int              (Attribute) 当前时间步数
├── @order: int             (Attribute) 多项式阶数 p
├── rho:   Dataset<float64> (N_tot,)    密度
├── rho_u: Dataset<float64> (N_tot,)    x-动量
├── rho_v: Dataset<float64> (N_tot,)    y-动量
├── E:     Dataset<float64> (N_tot,)    总能量
├── u:     Dataset<float64> (N_tot,)    x-速度（派生）
├── v:     Dataset<float64> (N_tot,)    y-速度（派生）
├── p:     Dataset<float64> (N_tot,)    压力（派生）
└── mesh/                   (Group)
    ├── x:  Dataset<float64> (N_tot,)   x 坐标
    └── y:  Dataset<float64> (N_tot,)   y 坐标
```

（`float64` 在 `USE_FLOAT_PRECISION` 构建下为 `float32`，即项目类型 `Real`。）

#### 数据集维度说明

每个 Dataset 的形状为 $(N_{\text{tot}},)$，即一维数组，$N_{\text{tot}} = N_e \times N_q^2$，展平索引为：

$$
k = e \cdot N_q^2 + q
$$

其中 $e$ 为单元编号，$q$ 为单元内积分点编号（$q = j N_q + i$，列主序网格序）。守恒变量由模态系数经 $V \hat u$ 投影得到，原始变量在主机端派生；所有数据集与 `/mesh/x`、`/mesh/y` 的坐标序列一一对应。

> **设计决策**：选择一维 Dataset 而非二维 Dataset 的原因在于：（1）避免了 VDS（Virtual Dataset）或 chunking 的复杂性；（2）DG 方法中积分点并非均匀网格节点的简单子集，维护严格的二维拓扑关系在后处理中并不直接有用；（3）简化了写入逻辑，`write` 调用可直接传入连续缓冲，无数据重排开销。

> **注意**：上述文件格式对应策略 A（直接 SoA 输出）。策略 B（检查点 + 后转换）的文件格式见 [2.4 节策略 B](#策略-b检查点--后转换)，其输出采用模态系数二维数据集 $(N_e, N_{\text{modes}})$ 的布局。两种策略输出的 HDF5 结构不同，不可互换。

### 2.4 输出策略

CMeles 提供两种输出策略以适应不同的使用场景，用户可通过配置文件的 `[output]` 表选择。

#### 策略概览

| 维度 | 策略 A：直接 SoA 输出 | 策略 B：检查点 + 后转换 |
|------|----------------------|------------------------|
| **输出内容** | 积分点上的守恒变量 SoA 数组 | 网格信息 + 多项式模态系数 |
| **计算期开销** | AoS → SoA 转换 + 各物理量独立写入 | 直接 dump 原始数据，无转换 |
| **内存峰值** | SoA 所有物理量的临时缓冲区 | 仅拷贝一个模态系数向量 |
| **文件可读性** | 立即可被 ParaView 等工具读取 | 需后处理转换工具 |
| **文件大小** | $N_{\text{vars}} \times N_{\text{tot}}$ 个 `double` | $N_{\text{vars}} \times N_e \times N_{\text{modes}}$ 个 `double` |
| **适用场景** | 小/中规模问题，需要即时可视化 | 大规模/长时间计算，频繁输出检查点 |

> **备注**：$N_{\text{tot}} = N_e \times N_q^2$ 为总积分点数，$N_{\text{modes}} = (N+1)(N+2)/2$ 为每个单元每个变量的模态系数个数。当多项式阶数 $N$ 较高时，积分点数 $N_q^2 \approx (N{+}2)^2$ 显著超过模态数 $N_{\text{modes}}$，策略 B 的存储量仅为策略 A 的 $N_{\text{modes}} / N_q^2 \approx \left(\frac{N+1}{N+2}\right)^2$，高阶下优势突出。

#### 策略 A：直接 SoA 输出

即 2.2–2.3 节所述方案：计算过程中将 AoS 格式的积分点数据转换为 SoA 并写入 HDF5。写入的文件可直接被后处理工具使用，无需额外转换步骤。

##### HDF5 输出器骨架

```cpp
#include <highfive/H5File.hpp>
#include <string>
#include <vector>

class HDF5Writer {
public:
    /// Open (or create) an HDF5 file for writing.
    /// @param filename  Full path to the output .h5 file
    /// @param overwrite If true, overwrite existing file; if false, throw on collision
    explicit HDF5Writer(const std::string& filename, bool overwrite = true)
        : file_(filename, overwrite ? HighFive::File::Overwrite
                                    : HighFive::File::Create)
    {}

    /// Write a single scalar attribute to the root group.
    void write_attribute(const std::string& name, double value) {
        file_.createAttribute(name, value);
    }

    void write_attribute(const std::string& name, int value) {
        file_.createAttribute(name, value);
    }

    /// Write a named dataset from a contiguous buffer (SoA-friendly).
    /// The buffer is written as a 1D dataset of length `count`.
    void write_dataset(const std::string& name,
                       const double* data,
                       size_t count)
    {
        // HighFive automatically infers the datatype from the pointer type.
        // Using the raw pointer avoids an intermediate copy.
        file_.createDataSet<double>(name,
            HighFive::DataSpace({count}))
            .write_raw(data);
    }

    /// Write a named dataset from a std::vector (convenience overload).
    void write_dataset(const std::string& name,
                       const std::vector<double>& vec)
    {
        write_dataset(name, vec.data(), vec.size());
    }

    /// Write a dataset into a subgroup, creating the group if necessary.
    void write_mesh_coordinate(const std::string& coord_name,
                               const std::vector<double>& coords)
    {
        // 惰性创建：分组已存在时直接获取，避免重复 createGroup 抛出异常
        auto mesh_group = file_.exist("mesh") ? file_.getGroup("mesh")
                                              : file_.createGroup("mesh");
        mesh_group.createDataSet<double>(coord_name,
            HighFive::DataSpace({coords.size()}))
            .write_raw(coords.data());
    }

    /// Flush pending writes to disk (RAII also handles this on destruction).
    void flush() { file_.flush(); }

private:
    HighFive::File file_;
};
```

##### 典型输出流程（实现见 `src/io/SolutionWriter.cpp`）

```cpp
void SolutionWriter::write(long step, Real time)
{
    // Step 1: 设备 -> 主机拷贝模态系数（[elem][var][mode]，EVM）。
    //         平 occa::memory::copyTo：统一与分离内存后端均有效
    //         （DeviceMemoryManager::copyToHost 在统一后端是 no-op）。
    field_.o_u().copyTo(modal_.data(), nEvm);

    // Step 2: 逐单元逐变量投影到积分点并拆分 SoA：
    //         nodal = V * u_hat（V 为 Vandermonde，见 2.2 节），
    //         随后派生 u = rho_u/rho、v = rho_v/rho、
    //         p = (gamma-1)(E - rho|u|^2/2)。
    // Step 3: 写元数据属性 time/step/order + 7 个 SoA 数据集
    //         + mesh/x、mesh/y，最后 flush。
}
```

##### 与 OCCA GPU 数据的对接

当数据驻留在 GPU 设备内存（`occa::memory` 对象）中时，输出时先取回主机再投影写入（上节 Step 1）。注意统一使用 `occa::memory::copyTo`：`DeviceMemoryManager::copyToHost` 在统一内存后端（Serial/OpenMP）是 no-op，不能用于填充输出器自有缓冲。

对于多物理量的输出，可并行化拷贝（多个 `occa::memory::copyTo` 调用和 HDF5 写入交替进行，利用异步拷贝和磁盘 I/O 的重叠）。当前实现为同步输出，后续根据 profiling 结果决定是否需要引入异步流水线。

##### RAII 与异常安全

`HDF5Writer` 在构造时打开（或创建）文件，在析构时自动关闭文件。`HighFive::File` 自身遵循 RAII，因此即使在写入过程中抛出异常，文件也会被正确关闭而不会泄露资源。此外，被部分写入的 HDF5 文件在下次以 `Overwrite` 模式打开时会被完整替换，不会残留中间状态。

#### 策略 B：检查点 + 后转换

该策略将输出拆分为两个阶段：

- **计算阶段（在线）**：将网格信息和每个检查点处的 DG 模态系数直接序列化写入 HDF5，不做 AoS → SoA 转换，不计算积分点值，最小化对计算主循环的性能影响。每个检查点处的模态系数仅需一次 `std::memcpy`（或 OCCA `copyTo`）写入文件。
- **转换阶段（离线）**：计算结束后运行独立的 `CMelesConvert` 工具，读取检查点文件中的模态系数，通过 Vandermonde 矩阵投影到积分点，再转换为 SoA 格式写入标准 HDF5 输出文件。

##### 检查点文件格式

一个算例的输出目录结构如下：

```
results/
├── mesh.h5                    (网格信息文件，仅写一次)
├── checkpoint_0000000.h5      (时间步 0 的模态系数)
├── checkpoint_0000100.h5      (时间步 100 的模态系数)
├── checkpoint_0000200.h5      ...
└── ...
```

**网格信息文件 `mesh.h5`**：

```
/                           (Root Group)
├── @dim: int               (Attribute) 空间维数
├── @N_elem: int            (Attribute) 单元总数
├── @order: int             (Attribute) 多项式阶数 p
├── @N_q: int               (Attribute) 每方向积分点数
├── @N_modes: int           (Attribute) 每单元每变量模态数 (p+1)(p+2)/2
├── @element_type: string   (Attribute) "quad" | "tri"
├── x: Dataset<float64>     (N_e * N_q^2,) 所有积分点 x 坐标
└── y: Dataset<float64>     (N_e * N_q^2,) 所有积分点 y 坐标
```

**检查点文件 `checkpoint_<step>.h5`**：

```
/                           (Root Group)
├── @time: double           (Attribute) 当前物理时间
├── @step: int              (Attribute) 当前时间步数
├── @gamma: double          (Attribute) 比热比（离线转换派生原始变量所需）
├── rho:   Dataset<float64> (N_e, N_modes)  ρ 模态系数（按单元排列）
├── rho_u: Dataset<float64> (N_e, N_modes)  ρu 模态系数
├── rho_v: Dataset<float64> (N_e, N_modes)  ρv 模态系数
└── E:     Dataset<float64> (N_e, N_modes)  E 模态系数
```

其中 $N_{\text{modes}} = (N+1)(N+2)/2$ 为每个单元、每个变量的模态系数个数。数据集的形状为二维 $(N_e, N_{\text{modes}})$：第一维为单元索引，第二维为模态索引。计算状态的 `[elem][var][mode]`（EVM）布局在写出前重排为每变量的行主序二维块，`createDataSet` + `write_raw` 一次写入。相比初版设计补充了 `N_modes` 与 `gamma` 属性——离线转换器重建基函数与派生原始变量时所需。

##### 检查点写入器（实现见 `src/io/CheckpointWriter.{hpp,cpp}`）

```cpp
/// Strategy B writer: mesh.h5 (once) + checkpoint_<step>.h5.
class CheckpointWriter : public OutputWriterBase {
public:
    CheckpointWriter(const Config& cfg, const DgField& field);

    /// One-time mesh companion file (metadata + quadrature coordinates).
    void writeMesh();

    /// Dump the raw modal coefficients as (N_e, N_modes) datasets,
    /// plus time/step/gamma attributes; no projection, no conversion.
    void write(long step, Real time) override;
};
```

模态系数在写出前由 `[elem][var][mode]`（EVM）重排为每变量的行主序 $(N_e, N_{\text{modes}})$ 块，`createDataSet` + `write_raw` 一次写入，对计算主循环的影响仅一次设备→主机拷贝。

##### 离线转换器：`CMelesConvert`

`CMelesConvert` 是一个独立的命令行工具，负责将检查点文件转换为标准 SoA HDF5 输出。其工作流程为：

```
mesh.h5 + checkpoint_0000100.h5  →  solution_0000100.h5
```

核心算法：读取检查点中的模态系数 $\mathbf{\hat{u}}_{e}^{(m)}$（单元 $e$，变量 $m$），通过 Vandermonde 矩阵 $V_{qi} = \phi_i(\xi_q)$ 投影到积分点：

$$
\mathbf{u}_e^{(m)}(\xi_q) = \sum_{i=0}^{N_{\text{modes}}-1} \hat{u}_{e,i}^{(m)} \,\phi_i(\xi_q)
$$

其中 $\{\phi_i\}$ 为基函数（详见[基函数文档](basis_functions.md)），$\{\xi_q\}$ 为积分点坐标。对所有单元和所有变量完成投影后，得到积分点上的守恒变量，再派生原始变量并按 SoA 写入标准输出文件。基函数与积分点规则由 `mesh.h5` 的 `order`/`N_q` 属性重建（`BasisFunctions1D(order, N_q)` + `BasisFunctions2D`），全程不涉及 OCCA 设备上下文。

命令行用法（实现见 `tools/CMelesConvert.cpp`，转换核心为 `src/io/CheckpointConvert.cpp` 的 `convertCheckpoint`）：

```
CMelesConvert <directory>                                # 批量：mesh.h5 + checkpoint_*.h5 -> solution_*.h5
CMelesConvert <mesh.h5> <checkpoint.h5> [-o <out.h5>]    # 单个检查点转换
```

批量模式按文件名顺序处理目录中全部 `checkpoint_<step>.h5`，输出同名 `solution_<step>.h5`。

##### 配置项

`[output]` 表通过 `strategy` 字段选择策略（完整键见 1.3 节）：

```toml
[output]
enable = true
strategy = "checkpoint"    # 输出策略："direct" (策略 A) | "checkpoint" (策略 B)
interval = 100
directory = "./results"
```

- `strategy = "direct"`：采用策略 A，输出文件立即可用，适合中小规模计算
- `strategy = "checkpoint"`：采用策略 B，输出原始模态系数，适合大规模/频繁输出的场景

当 `strategy` 未指定时，默认使用 `"direct"`。输出策略的运行时分派由 `FieldOutput`（`src/io/FieldOutput.{hpp,cpp}`）完成：它创建输出目录，按策略持有 `SolutionWriter` 或 `CheckpointWriter`（策略 B 同时立即写出一次 `mesh.h5`），并向求解器主循环暴露统一的 `write(step, time)` 边界（抽象基类 `OutputWriterBase`，粗粒度边界用虚函数）。

### 2.5 未来扩展

| 扩展项 | 说明 |
|--------|------|
| **分片写入（Chunking）** | 对大网格启用 HDF5 chunked dataset 和压缩（如 GZIP level 4），在文件大小和读写性能间取得平衡 |
| **并行 I/O** | 利用 HDF5 的 MPI-IO 后端，支持多节点并行写入同一个输出文件 |
| **XDMF 元文件** | 自动生成 XDMF 轻量级 XML 元文件，使 ParaView 等工具可直接读取时间序列 |
| **Selective Output** | 按 `output.variables` 列表选择性输出物理量，减少不必要数据的磁盘占用 |

---

## 三、相关文档

- [网格与几何](mesh_and_geometry.md) — 网格数据结构与几何映射
- [控制方程与 DG 场](governed_equations_and_DG_field.md) — DGField 数据结构的详细定义
- [时间推进](time_marching.md) — 输出时机由 time_marching 主循环控制

---

*文档版本: 1.1 | 最后更新: 2026-09-10*
