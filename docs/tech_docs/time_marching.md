# 时间推进

## 子模块导航

- **[显式时间推进](time_marching/explicit_time_marching.md)** — 定步长显式方法
  - [显式欧拉法](time_marching/explicit_time_marching.md#一显式欧拉法explicit-euler-method)
  - [SSPRK3 方法](time_marching/explicit_time_marching.md#二ssprk3-方法strong-stability-preserving-runge-kutta-3rd-order)
- **[显式自适应时间推进](time_marching/explicit_adaptive_time_marching.md)** — 嵌入式 RK 对与 PI 步长控制
  - [显式 RK 一般形式](time_marching/explicit_adaptive_time_marching.md#一显式-rk-方法的一般形式)
  - [嵌入式 RK 对](time_marching/explicit_adaptive_time_marching.md#二嵌入式-rk-对)
  - [PI 步长控制器](time_marching/explicit_adaptive_time_marching.md#三pi-步长控制器)
  - [DG-DITR 双时间步中的应用](time_marching/explicit_adaptive_time_marching.md#四在-dg-ditr-双时间步格式中的应用)
- **[隐式时间推进](time_marching/implicit_time_marching.md)** — DITR 方法与双时间步法
  - [DITR 理论基础](time_marching/implicit_time_marching.md#一ditr-方法理论基础)
  - [DITR U2R2 方法](time_marching/implicit_time_marching.md#二ditr-u2r2-方法)
  - [DITR U2R1 方法](time_marching/implicit_time_marching.md#三ditr-u2r1-方法)
  - [DITR U3R1 方法](time_marching/implicit_time_marching.md#四ditr-u3r1-方法)
  - [后向欧拉法](time_marching/implicit_time_marching.md#五后向欧拉法)
  - [双时间步法](time_marching/implicit_time_marching.md#七双时间步法)

---

## 概述

时间推进模块用于根据半离散控制方程的右端项 $\mathcal{R}(t, \boldsymbol{u})$，对 DG 方法离散得到的常微分方程组进行时间积分：

$$
\frac{\mathrm{d}\boldsymbol{u}}{\mathrm{d}t} = \mathcal{R}(t, \boldsymbol{u}), \quad \boldsymbol{u} \in \mathbb{R}^{N_{\text{elem}} \cdot N_{\text{vars}} \cdot N_{\text{modes}}}
$$

其中 $\boldsymbol{u}$ 是展平后的模态系数向量，$\mathcal{R}$ 为 DG 场模块计算得到的残差向量。时间推进模块将上述 ODE 从初始时刻 $t=0$ 推进到终止时刻 $t=T$，在每个时间步中调用 DG 场模块计算右端项。

## 时间推进方法分类

时间推进方法主要分为显式方法和隐式方法两大类。

### 显式方法

显式方法将当前时刻的解表达为过去时刻解的显式函数，无需求解线性方程组。

**优点**：
- 每时间步计算量小，实现简单
- 内存占用低
- 易于并行化

**缺点**：
- 稳定性受 CFL 条件制约，时间步长受限于最小时空尺度
- 对于粘性占主导或网格高度非均匀的问题，时间步长可能过于严苛
- 不适合低频非定常问题的长时间积分

### 隐式方法

隐式方法将下一时刻的解表达为未知量，通常需要求解（线性化后的）线性方程组。

**优点**：
- 稳定性强，可采用大时间步长（无条件稳定的方法可任意放大步长）
- 适合刚性问题和低频非定常长时间积分
- 对网格质量不敏感

**缺点**：
- 每时间步需求解大规模线性方程组，计算量大
- 实现复杂（涉及 Jacobian 矩阵的构造与存储）
- 对并行计算不友好（全局通信开销大）

## 方法选择策略（AI生成，未审核）

| 应用场景                     | 推荐方法                           | 理由                                                     |
| :--------------------------- | :--------------------------------- | :------------------------------------------------------- |
| 定常问题收敛加速             | 显式方法 + 当地时间步长 / 隐式方法 | 显式方法配合当地时间步长可快速收敛；隐式方法单步推进更快 |
| 非定常问题（高分辨 DNS/LES） | 显式方法（SSPRK3/RK5(4)）          | CFL 限制的物理时间步通常已足够小，显式方法成本更低       |
| 非定常问题 + 自适应步长      | 嵌入式 SSPRK3(3)2 + PI 控制器      | SSP 性质 + 自动误差控制，GPU 友好                        |
| 高度非均匀网格 / 低马赫数    | 隐式 DITR（U2R1 / U3R1）           | L 稳定性允许大时间步长推进                               |
| 高精度非定常长时间积分       | 隐式 DITR（U3R1, 4 阶 + L 稳定）   | 四阶精度与 DG 空间精度匹配                               |
| 强间断 / 激波捕捉            | SSPRK3 / 嵌入式 SSPRK 对           | SSP 性质确保 TVD/TVB 性质在时间离散中保持                |
| GPU 加速隐式求解             | 阶段解耦 DITR + 自适应伪时间步     | 解耦降低系统规模，自适应步长 GPU 友好                    |


---

## 模块实现（src/time/）

时间推进模块已实现，采用 **CRTP 编译期多态**（无虚函数）：

| 文件 | 内容 |
| ---- | ---- |
| `TimeTypes.hpp` | `RhsFunction` / `PositivityLimiter` 类型别名 |
| `StepperBase.hpp` | CRTP 基类：公共上下文（device/mem/rhs/Blas/更新内核/limiter 钩子）与静态接口 `advance(u, t, dt) -> 实际步长` |
| `SimpleExplicitStepper.{hpp,cpp}` | `EulerStepper`（1 阶）、`SspRk3Stepper`（Shu–Osher 3 阶） |
| `ButcherTable.hpp` | 6 张 constexpr Butcher 表（RK32/RK54/SSPRK221/321/332/432，系数取自原型与本文档） |
| `RungeKuttaStepper.{hpp,cpp}` | 通用 embedded RK：Butcher 表为运行期数据（一次上传设备，内核运行期读取），FSAL、RMS 缩放误差、PI 控制器、Hairer 初始步长、`advanceFixed`（定步长伪推进） |
| `ImplicitResidual.{hpp,cpp}` | `BackwardEulerStepper` 与 `DitrStepper`（U2R2/U2R1/U3R1 单类 + 变体系数）：仅构造时间残差 $\mathcal{F}$，含耦合预条件子与解耦每级残差 |
| `DualStepper.hpp` | `DualStepper<PhyStepper>` 模板：伪时间自适应 RK 推进至 $\|\mathcal{F}\|_\infty$ 收敛（REF_STEP = 5 参考范数），耦合/解耦两种模式，内部维护 $u^{n-1}$ 与 $\theta$ |
| `okl/time_update.okl` | 向量更新内核（`explicitEulerUpdate`、`sspConvexCombine`、Butcher 阶段/误差内核、`vecCombine4`、`absInto`） |

要点：

- RK 变体与 DITR 变体的差异是**数据**（Butcher 表 / 重构系数）而非行为，故以单类 + constexpr
  表实现；真正的类型差异（Euler / SSPRK3 / RK / Dual）由模板分派。
- 唯一的运行期→编译期分派点是 `runCompressibleFlowSolver(const Config&)`
  （`src/solver/CompressibleFlowSolver.cpp`），共 5 个求解器实例化。
- 时间步长：`DgField::estimateDt` 按本文档 CFL 公式每步重算（`estimate_dt.okl`），
  `[time_marching] dt > 0` 时使用固定步长。
- 正保持限制器钩子（`setPositivityLimiter`）在每个阶段状态后调用，默认 no-op（移植自原型）。
- 单元测试见 `ctest/time/test_time_stepper.cpp`（ODE $u' = \lambda u$ 收敛阶）与
  `ctest/solver/test_euler_vortex.cpp`（等熵涡整链路验证）。
