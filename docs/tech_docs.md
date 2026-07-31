# 技术文档

本目录包含 CMeles 项目的技术文档。

## 编写准则

技术文档应重点关注以下方面：

1. **算法原理**：详细描述数学公式、物理模型和数值方法的理论基础
2. **代码实现方案**：说明算法如何转化为高效、可维护的代码实现

技术文档应避免过度关注项目管理、使用流程等内容，而应聚焦于核心技术细节。

## 文档索引

- [基函数](tech_docs/basis_functions.md) - 正交多项式基函数、Vandermonde 矩阵、数值求积、坐标变换、求和分解等内容
  - [平行四边形假设](tech_docs/parallelogram_assumption.md) - 常数雅可比行列式与对角质量矩阵的简化假设
- [OCCA 内存模型与核函数开发](tech_docs/occa.md) - 设备内存模型差异（统一 vs 分离内存空间）、`DeviceMemoryManager` 统一接口设计、OKL 核函数开发约定与 `@tile` 分块策略
- [网格与几何](tech_docs/mesh_and_geometry.md) - 一维面单元、二维体单元、法向量、雅可比矩阵、邻接关系、边界处理
- [控制方程与 DG 场](tech_docs/governed_equations_and_DG_field.md) - 可压缩 Navier-Stokes 方程、无量纲化、DG 空间离散、DG 场数据结构和函数设置
  - [黎曼求解器与数值通量](tech_docs/riemann_solver.md) - 数值通量 FDS 方案（LLF/Rusanov、Roe、带熵修正 Roe）与 FVS 方案（Steger-Warming、van Leer），一维 Euler 方程特征结构，统一代数框架与 CMeles 嵌入方式
- [时间推进](tech_docs/time_marching.md) - 定步长显式方法（Euler、SSPRK3）、嵌入式 RK 对自适应步长控制（PI 控制器）、DITR 隐式方法（U2R2/U2R1/U3R1）与双时间步法
- [输入输出模块](tech_docs/io_module.md) - TOML 配置文件解析（读入模块）与 HDF5 流场数据写入（输出模块）的架构设计、数据格式和代码实现
