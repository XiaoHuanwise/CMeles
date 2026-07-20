# 输入输出模块

本文档描述 CMeles 中程序的输入（TOML 配置文件解析与命令行参数处理）和输出（HDF5 格式流场数据写入）模块的架构设计、数据格式与代码实现方案。

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
strategy = "direct"       # 输出策略："direct" (策略 A) | "checkpoint" (策略 B)
format = "h5"             # 输出格式（当前仅支持 HDF5）
interval = 100            # 每隔 N 个时间步输出一次
directory = "./results"   # 输出目录（可被命令行 -o 覆盖）
variables = ["rho", "u", "v", "p", "E"]  # 要输出的变量列表
```

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
| 时间步长 | `dt > 0`（显式方法时） | `定步长显式方法的 dt 必须为正数` |
| Riemann 求解器 | 必须在支持列表中 | `未知的 riemann_solver，支持: HLLC, Roe, ...` |

校验失败时抛出 `std::invalid_argument` 并附带清晰的错误描述，帮助用户快速定位配置问题。

---

## 二、输出模块：HDF5 数据写入

CMeles 使用 [HighFive](https://github.com/BlueBrain/HighFive) 库将流场计算结果写入 HDF5 文件。HighFive 是 HDF5 C 库的现代 C++ 封装，提供 RAII 资源管理和直观的模板化接口。

### 2.1 模块架构

输出模块的核心职责是：在指定的输出时间步，将 DG 场的计算数据持久化到 HDF5 文件。根据配置选择的策略，输出内容可以是经过 AoS→SoA 转换的积分点数据（策略 A），也可以是未经转换的模态系数检查点（策略 B）。图 1 描述了策略 A 从 AoS 计算格式到 SoA 输出格式的转换关系：

```
计算阶段 (AoS, elem-major)             HDF5 文件结构 (SoA)
┌──────────────────────────┐        ┌──────────────────────┐
│ 单元0: [q0, q1, ..., qNq]│        │ /rho   (Dataset)     │
│ 单元1: [q0, q1, ..., qNq]│  转换  │ /rho_u (Dataset)     │
│ 单元2: [q0, q1, ..., qNq]│ ────▶  │ /rho_v (Dataset)     │
│ ...                      │        │ /E     (Dataset)     │
│                          │        │ /time  (Attribute)   │
│ 每个 q 包含:             │        │ /step  (Attribute)   │
│  (ρ, ρu, ρv, E)          │        │ /mesh/ (Group)       │
│                          │        │   /mesh/x            │
└──────────────────────────┘        │   /mesh/y            │
                                    └──────────────────────┘
```

#### 设计要点

- **AoS → SoA 转换**：计算阶段采用单元主序的 AoS（Array of Structures）存储 $(e, q) \mapsto (\rho, \rho u, \rho v, E)$，保证 OCCA 核函数对同一积分点上所有变量的合并访问友好。输出阶段将 AoS 转换为 SoA——每个物理量提取为独立的 `std::vector<double>`，再写入对应的 HDF5 Dataset。转换与 HDF5 写入可合并为一次遍历，避免额外的中间缓冲。
- **元数据以 Attribute 形式附加**：将当前物理时间、时间步数等标量信息作为 Group 或 File 级别的 Attribute 写入，方便后处理和可视化工具读取。
- **OCCA 设备内存**：GPU 计算时，需先将 OCCA 设备内存通过 `occa::memory::copyTo` 拷贝到主机端，再执行 AoS → SoA 转换和写入。

### 2.2 数据布局：AoS 计算格式与 SoA 输出格式

CMeles 采用**双格式策略**：计算阶段使用 AoS（Array of Structures）以优化核函数访存，输出阶段转换为 SoA（Structure of Arrays）以便后处理。

#### AoS：计算格式（单元主序，elem-major）

DG 核函数在每个积分点上需要同时访问所有守恒变量 $(\rho, \rho u, \rho v, E)$ 以计算通量。AoS 将同一积分点上的所有变量打包为结构体，同一单元内积分点连续排列：

```
单元 0: [s₀, s₁, ..., s_{Nq-1}]
单元 1: [s₀, s₁, ..., s_{Nq-1}]
...
单元 Ne-1: [s₀, s₁, ..., s_{Nq-1}]

其中 s_q = (ρ_q, ρu_q, ρv_q, E_q) 为单个积分点上的守恒变量结构体
```

积分点 $q$ 在单元 $e$ 中的展平索引为 $k = e \cdot N_q + q$，同一积分点上各变量在内存中相邻，对 OCCA 核函数的合并加载（coalesced access）有利。

```cpp
/// AoS 计算格式：每积分点的守恒变量打包
struct alignas(32) QPoint {
    double rho;
    double rho_u;
    double rho_v;
    double E;
};

/// 所有积分点的 AoS 数组，长度 N_tot = N_e * N_q
std::vector<QPoint> q_aos;
```

#### SoA：输出格式

后处理工具通常按物理量维度读取（如绘制密度云图时只读取 `rho` 数据集），因此输出时转换为 SoA——每个物理量提取为独立的连续数组：

| 对比维度     | AoS（计算用）              | SoA（输出用）                |
| ------------ | -------------------------- | ---------------------------- |
| 核函数访存   | 合并访问，cache 友好       | 跨步读取，效率低             |
| 访问单物理量 | 跨步读取，cache 效率低     | 连续读取，cache 完全利用     |
| 输出灵活性   | 必须写入整个结构体         | 可按需选择性地写入部分物理量 |
| HDF5 存储    | 复合数据类型，工具兼容性差 | 简单数组，所有 HDF5 工具通用 |

设网格包含 $N_e$ 个单元，每个单元有 $N_q$ 个积分点，$N_{\text{tot}} = N_e \times N_q$。转换为 SoA 后，各物理量的内存布局为：

```
rho   : [ρ₀,    ρ₁,    ..., ρ_{Ntot-1}   ]  ← 连续存储，Ntot 个 double
rho_u : [ρu₀,   ρu₁,   ..., ρu_{Ntot-1}  ]  ← 连续存储，Ntot 个 double
rho_v : [ρv₀,   ρv₁,   ..., ρv_{Ntot-1}  ]  ← 连续存储，Ntot 个 double
E     : [E₀,    E₁,    ..., E_{Ntot-1}   ]  ← 连续存储，Ntot 个 double
```

SoA 的序列顺序与计算阶段的展平索引 $k = e \cdot N_q + q$ 保持一致，因此输出的 HDF5 坐标数据集和流场数据集的元素是一一对应的。

#### AoS → SoA 转换实现

转换在 HDF5 写入前完成，与写入遍历合并，一次遍历中同时完成数据重排和写入：

```cpp
/// AoS → SoA 转换并写入 HDF5
void write_aos_as_soa(HDF5Writer& writer, const std::vector<QPoint>& q_aos, size_t N_tot)
{
    std::vector<double> rho(N_tot), rho_u(N_tot), rho_v(N_tot), E(N_tot);

    for (size_t k = 0; k < N_tot; ++k) {
        rho[k]   = q_aos[k].rho;
        rho_u[k] = q_aos[k].rho_u;
        rho_v[k] = q_aos[k].rho_v;
        E[k]     = q_aos[k].E;
    }

    writer.write_dataset("rho",   rho);
    writer.write_dataset("rho_u", rho_u);
    writer.write_dataset("rho_v", rho_v);
    writer.write_dataset("E",     E);
}
```

> **设计取舍**：AoS → SoA 转换引入了 $4 \times N_{\text{tot}} \times 8$ 字节的额外内存和一次全量遍历。对于典型问题规模（$10^4 \sim 10^6$ 积分点），额外开销在数 MB 至数十 MB 量级，完全可接受。若后续 profiling 显示转换成为瓶颈，可优化为分块流水线（chunked pipeline），将 AoS → SoA 转换与 HDF5 异步写入重叠执行。

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
├── p:     Dataset<float64> (N_tot,)    压力（如需要）
└── mesh/                   (Group)
    ├── x:  Dataset<float64> (N_tot,)   x 坐标
    └── y:  Dataset<float64> (N_tot,)   y 坐标
```

#### 数据集维度说明

每个 Dataset 的形状为 $(N_{\text{tot}},)$，即一维数组。对于二维网格，计算阶段采用单元主序（elem-major）的 AoS 形式存储，展平索引为：

$$
k_{\text{AoS}} = e \cdot N_q + q
$$

其中 $e$ 为单元编号，$q$ 为单元内积分点编号。此展平方式保证同一单元内不同积分点的数据在内存中连续，与 OCCA 核函数 `@tile` 的 elem 分块访问模式一致。

输出时，将 AoS 布局**转换为 SoA**（Structure of Arrays）：每个物理量提取为独立的一维 Dataset，总长度为 $N_{\text{tot}}$。AoS → SoA 的转换在输出函数中完成，转换后的数据顺序与计算阶段的展平索引 $k_{\text{AoS}}$ 保持一致，因此 `/mesh/x` 和 `/mesh/y` 数据集中的坐标序列与流场数据的积分点一一对应。

> **设计决策**：选择一维 Dataset 而非二维 Dataset 的原因在于：（1）避免了 VDS（Virtual Dataset）或 chunking 的复杂性；（2）DG 方法中积分点并非均匀网格节点的简单子集，维护严格的二维拓扑关系在后处理中并不直接有用；（3）简化了写入逻辑，`write` 调用可直接传入 `std::vector<T>::data()`，无数据重排开销。

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

> **备注**：$N_{\text{tot}} = N_e \times N_q$ 为总积分点数，$N_{\text{modes}} = (p+1)^d$ 为每个单元每个变量的模态系数个数。当多项式阶数 $p$ 较高时，积分点数 $N_q \approx (p+2)^d$ 显著超过模态数 $N_{\text{modes}}$，策略 B 的存储量仅为策略 A 的 $N_{\text{modes}} / N_q \approx \left(\frac{p+1}{p+2}\right)^d$，高阶下优势突出。

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

##### 典型输出流程

```cpp
void write_solution(
    HDF5Writer& writer,
    const std::vector<QPoint>& q_aos,  // AoS 计算格式
    int step,
    double time,
    int order)
{
    // Step 1: 写入元数据
    writer.write_attribute("time", time);
    writer.write_attribute("step", step);
    writer.write_attribute("order", order);

    // Step 2: AoS → SoA 转换并写入（复用 2.2 节 write_aos_as_soa）
    write_aos_as_soa(writer, q_aos, q_aos.size());

    // Step 3: 刷新缓冲区
    writer.flush();
}
```

##### 与 OCCA GPU 数据的对接

当数据驻留在 GPU 设备内存（`occa::memory` 对象）中时，AoS 格式的 `QPoint` 数组驻留在设备端。输出时需先拷贝到主机，再执行 AoS → SoA 转换并写入：

```cpp
/// 从设备端 AoS 格式输出：设备 → 主机拷贝 → AoS → SoA 转换 → HDF5 写入
void write_from_device(
    HDF5Writer& writer,
    occa::memory o_q_aos,   // 设备端 QPoint 数组（AoS 格式）
    size_t N_tot)
{
    // Step 1: 设备 → 主机拷贝（AoS 格式）
    std::vector<QPoint> host_aos(N_tot);
    o_q_aos.copyTo(host_aos.data());

    // Step 2: AoS → SoA 转换并写入
    write_aos_as_soa(writer, host_aos, N_tot);
}
```

对于多物理量的输出，可并行化拷贝（多个 `occa::memory::copyTo` 调用和 HDF5 写入交替进行，利用异步拷贝和磁盘 I/O 的重叠）。当前阶段优先保证正确性，后续根据 profiling 结果决定是否需要引入异步流水线。

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
├── @element_type: string   (Attribute) "quad" | "tri"
├── x: Dataset<float64>     (N_e * N_q,) 所有积分点 x 坐标
└── y: Dataset<float64>     (N_e * N_q,) 所有积分点 y 坐标
```

**检查点文件 `checkpoint_<step>.h5`**：

```
/                           (Root Group)
├── @time: double           (Attribute) 当前物理时间
├── @step: int              (Attribute) 当前时间步数
├── rho:   Dataset<float64> (N_e, N_modes)  ρ 模态系数（按单元排列）
├── rho_u: Dataset<float64> (N_e, N_modes)  ρu 模态系数
├── rho_v: Dataset<float64> (N_e, N_modes)  ρv 模态系数
└── E:     Dataset<float64> (N_e, N_modes)  E 模态系数
```

其中 $N_{\text{modes}} = (p+1)^d$ 为每个单元、每个变量的模态系数个数。数据集的形状为二维 $(N_e, N_{\text{modes}})$：第一维为单元索引，第二维为模态索引。这种二维结构天然匹配计算阶段的 AoS 内存布局——`u[e][mode]` 可直接通过 `createDataSet` + `write_raw` 零拷贝写入。

##### 检查点写入器

```cpp
/// Lightweight checkpoint writer: dumps modal coefficients directly,
/// no AoS→SoA conversion, minimal overhead for the computation loop.
class CheckpointWriter {
public:
    explicit CheckpointWriter(const std::string& filename)
        : file_(filename, HighFive::File::Overwrite)
    {}

    void write_attributes(double time, int step) {
        file_.createAttribute("time", time);
        file_.createAttribute("step", step);
    }

    /// Write modal coefficients for one variable.
    /// @param name     Variable name (e.g. "rho", "rho_u")
    /// @param data     Pointer to host buffer of shape (N_e, N_modes), row-major
    /// @param N_e      Number of elements
    /// @param N_modes  Number of modes per element per variable
    void write_modal_coeffs(const std::string& name,
                            const double* data,
                            size_t N_e,
                            size_t N_modes)
    {
        file_.createDataSet<double>(name,
            HighFive::DataSpace({N_e, N_modes}))
            .write_raw(data);
    }

    void flush() { file_.flush(); }

private:
    HighFive::File file_;
};
```

##### 离线转换器：`CMelesConvert`

`CMelesConvert` 是一个独立的命令行工具，负责将检查点文件转换为标准 SoA HDF5 输出。其工作流程为：

```
mesh.h5 + checkpoint_0000100.h5  →  solution_0000100.h5
```

核心算法：读取检查点中的模态系数 $\mathbf{\hat{u}}_{e}^{(m)}$（单元 $e$，变量 $m$），通过 Vandermonde 矩阵 $V_{qi} = \phi_i(\xi_q)$ 投影到积分点：

$$
\mathbf{u}_e^{(m)}(\xi_q) = \sum_{i=0}^{N_{\text{modes}}-1} \hat{u}_{e,i}^{(m)} \,\phi_i(\xi_q)
$$

其中 $\{\phi_i\}$ 为基函数（详见[基函数文档](basis_functions.md)），$\{\xi_q\}$ 为积分点坐标。对所有单元和所有变量完成投影后，得到 AoS 格式的积分点值，再执行 AoS → SoA 转换写入标准输出文件。

该工具可在计算完成后对单个检查点文件操作，也可批量处理全部检查点。由于转换不涉及 OCCA 设备上下文，可在无 GPU 的登录节点上独立运行。

##### 配置项扩展

为支持两种策略，`[output]` 表增加 `strategy` 字段：

```toml
[output]
strategy = "checkpoint"    # 输出策略："direct" (策略 A) | "checkpoint" (策略 B)
format = "h5"
interval = 100
directory = "./results"
variables = ["rho", "u", "v", "p", "E"]
```

- `strategy = "direct"`：采用策略 A，输出文件立即可用，适合中小规模计算
- `strategy = "checkpoint"`：采用策略 B，输出原始模态系数，适合大规模/频繁输出的场景

当 `strategy` 未指定时，默认使用 `"direct"` 以保持向后兼容。

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

*文档版本: 1.0 | 最后更新: 2026-07-20*
