# 显式时间推进

## 概述

显式时间推进方法将下一时刻的解表达为当前及之前时刻解的显式函数，无需求解线性方程组。对于 DG 方法半离散化得到的 ODE 系统：

$$
\frac{\mathrm{d}\boldsymbol{u}}{\mathrm{d}t} = \mathcal{R}(t, \boldsymbol{u})
$$

其中 $\boldsymbol{u} \in \mathbb{R}^{N_{\text{elem}} \cdot N_{\text{vars}} \cdot N_{\text{modes}}}$ 为展平的模态系数向量，$\mathcal{R}$ 为 DG 场模块计算得到的残差向量。显式方法以 $\boldsymbol{u}^n$（第 $n$ 时间步的解）为起点，通过若干次右端项求值计算 $\boldsymbol{u}^{n+1}$。

### DG 多项式阶数与 CFL 极限估计

对于 DG 空间离散，显式时间推进的稳定时间步长受 CFL 条件制约。一般形式为：

$$
\Delta t \leq \frac{\text{CFL}}{2N + 1} \cdot \frac{h}{\Lambda_{\text{max}}}
$$

其中各参数含义为：

| 参数                   | 含义                                            |
| :--------------------- | :---------------------------------------------- |
| $N$                    | 多项式阶数                                      |
| $h$                    | 单元特征尺寸                                    |
| $\Lambda_{\text{max}}$ | 通量 Jacobian 矩阵最大特征值（对 Euler 方程为 $ | u | + c$） |
| $\text{CFL}$           | CFL 数（取决于具体时间推进方法）                |

> **关键规律**：时间步长限值与多项式阶数 $N$ 成反比，即 $\Delta t \propto 1/(2N+1)$。高阶多项式虽然提高了空间精度，但显著收紧了对时间步长的限制。因此在实际应用中，显式时间推进通常配合 $N = 1\text{--}3$ 使用，更高阶数需考虑隐式时间推进。

## 一、显式欧拉法（Explicit Euler Method）

### 1.1 算法描述

显式欧拉法是一阶精度的时间推进方法，也是形式最简单的龙格-库塔方法。其更新公式为：

$$
\boldsymbol{u}^{n+1} = \boldsymbol{u}^n + \Delta t \, \mathcal{R}(t^n, \boldsymbol{u}^n)
$$

其中 $\Delta t$ 为时间步长，$\boldsymbol{u}^n = \boldsymbol{u}(t^n)$。

### 1.2 稳定性分析

对于线性标量模型方程 $u' = \lambda u$（$\lambda \in \mathbb{C}$），显式欧拉法的稳定区域为：

$$
|1 + \lambda \Delta t| \leq 1
$$

在复平面上是以 $(-1, 0)$ 为圆心、半径为 $1$ 的圆盘。对于纯对流问题（$\lambda$ 为纯虚数），稳定区域不包含虚轴，因此显式欧拉法本质上不适用于纯对流问题（除非 $\Delta t = 0$），但可配合人工粘性或用于粘性主导的问题（$\lambda$ 为负实数）。

对于 DG 空间离散后的系统，时间步长受 CFL 条件制约：

$$
\Delta t \leq \text{CFL} \cdot \frac{h}{(2N + 1) \, \Lambda_{\text{max}}}
$$

其中 $h$ 为单元特征尺寸，$N$ 为多项式阶数，$\Lambda_{\text{max}}$ 为对流通量 Jacobian 矩阵的最大特征值（对 Euler 方程为 $|u| + c$，$c$ 为声速）。CFL 数对于显式欧拉法通常取 $0.1\text{--}0.5$。

> **备注**：由于显式欧拉法仅一阶精度、稳定区域有限且不包含虚轴，在 CFD 中极少单独使用。其主要用途为：
> - 定常问题的初场迭代推进（配合当地时间步长）
> - 多步方法中的起始步
> - 教学示例

### 1.3 OCCA 实现要点

显式欧拉法的单步更新仅涉及一次 DG 场模块计算和一次向量更新。OCCA 核函数结构如下：

```c
@kernel void explicitEulerUpdate(const int N_dof,
                                  const double dt,
                                  const double *u_n,
                                  const double *res_n,
                                  double *u_np1) {
  for (int dof = 0; dof < N_dof; ++dof; @tile(TILE_SIZE, @outer, @inner)) {
    u_np1[dof] = u_n[dof] + dt * res_n[dof];
  }
}
```

其中 `N_dof = N_elem * N_vars * N_modes` 为总自由度。向量更新核函数以单个自由度为最小粒度并行，每个线程处理一个自由度的 $u^{n+1}$ 计算，与 DG 场模块中的分块策略一致。

## 二、SSPRK3 方法（Strong Stability Preserving Runge-Kutta, 3rd Order）

### 2.1 背景

Strong Stability Preserving Runge-Kutta（SSPRK）方法是一类特殊构造的龙格-库塔方法，其核心性质是：若空间离散算子 $\mathcal{R}$ 在某种半范数下满足向前 Euler 稳定性（即 $\|\boldsymbol{u}^n + \Delta t \mathcal{R}(\boldsymbol{u}^n)\| \leq \|\boldsymbol{u}^n\|$），则 SSPRK 方法能在相同的 CFL 条件下保持该稳定性性质。

> **关键含义**：SSP 性质确保时间离散不会破坏空间离散的稳定性性质（如 TVD、熵稳定等）。对于高精度 DG 方法配合激波捕获格式（如 TVB 限制器、WENO 重构）的问题，SSP 时间推进可以保证空间离散的 TVD/TVB 性质在时间离散中仍然保持。

### 2.2 算法描述

三阶 SSP 龙格-库塔法（SSPRK3，有时写作 SSPRK(3,3)）的更新格式为：

$$
\begin{aligned}
\boldsymbol{u}^{(1)} &= \boldsymbol{u}^n + \Delta t \, \mathcal{R}(t^n, \boldsymbol{u}^n) \\
\boldsymbol{u}^{(2)} &= \frac{3}{4}\boldsymbol{u}^n + \frac{1}{4}\left[\boldsymbol{u}^{(1)} + \Delta t \, \mathcal{R}\left(t^n + \Delta t, \boldsymbol{u}^{(1)}\right)\right] \\
\boldsymbol{u}^{n+1} &= \frac{1}{3}\boldsymbol{u}^n + \frac{2}{3}\left[\boldsymbol{u}^{(2)} + \Delta t \, \mathcal{R}\left(t^n + \frac{\Delta t}{2}, \boldsymbol{u}^{(2)}\right)\right]
\end{aligned}
$$

该格式由 Shu 和 Osher (1988) 提出，是 CFD 中广泛使用的三阶显式时间推进方法。三个子步均可表示为向前 Euler 步与凸组合的形式，这是 SSP 性质的数学基础。

**等价但更紧凑的实现形式**（三阶段三存储）：

$$
\begin{aligned}
\boldsymbol{k}_1 &= \mathcal{R}(t^n, \boldsymbol{u}^n) \\
\boldsymbol{u}^{(1)} &= \boldsymbol{u}^n + \Delta t \, \boldsymbol{k}_1 \\
\boldsymbol{k}_2 &= \mathcal{R}(t^n + \Delta t, \boldsymbol{u}^{(1)}) \\
\boldsymbol{u}^{(2)} &= \frac{3}{4}\boldsymbol{u}^n + \frac{1}{4}\left[\boldsymbol{u}^{(1)} + \Delta t \, \boldsymbol{k}_2\right] \\
\boldsymbol{k}_3 &= \mathcal{R}\left(t^n + \frac{\Delta t}{2}, \boldsymbol{u}^{(2)}\right) \\
\boldsymbol{u}^{n+1} &= \frac{1}{3}\boldsymbol{u}^n + \frac{2}{3}\left[\boldsymbol{u}^{(2)} + \Delta t \, \boldsymbol{k}_3\right]
\end{aligned}
$$

### 2.3 SSP 性质分析

SSPRK3 的 SSP 系数为 $c = 1$，即：

- 若 $\|\boldsymbol{u} + \Delta t \mathcal{R}(\boldsymbol{u})\| \leq \|\boldsymbol{u}\|$ 在 $\Delta t \leq \Delta t_{\text{FE}}$ 时成立
- 则 SSPRK3 在 $\Delta t \leq c \cdot \Delta t_{\text{FE}} = \Delta t_{\text{FE}}$ 时保持该稳定性性质

实际使用中，SSPRK3 的 CFL 限值通常为：

| 空间离散              |      CFL 数       | 备注                   |
| :-------------------- | :---------------: | :--------------------- |
| 一阶迎风 DG           |    $1/(2N+1)$     | $N$ 为多项式阶数       |
| 高阶 DG（$N=3$ 左右） | $0.2\text{--}0.4$ | 经验值，取决于网格质量 |
| 线性对流（均匀网格）  |    $1/(2N+1)$     | 理论 CFL 限值          |

> **备注**：SSPRK3 的 SSP 系数 $c = 1$ 是最优三阶三步 SSP 方法的理论极值。理论上三阶三步 SSP 方法的最大 SSP 系数为 $c_{\text{opt}} = 1$（Gottlieb, Shu & Tadmor, 2001），SSPRK3 恰好达到了此极值。

### 2.4 在 DG 框架中的优势

对于采用 DG 空间离散的可压缩流动求解器，SSPRK3 具有以下优势：

1. **TVB/TVD 保持**：当 DG 方法配合限制器（如 TVB 限制器、WENO 重构）时，SSP 时间离散确保空间离散的 TVB 性质不被时间离散破坏，对于激波问题尤为重要。

2. **三步三存储**：SSPRK3 仅需存储三个中间状态（或两个——若将 $\boldsymbol{u}^{(2)}$ 就地更新到 $\boldsymbol{u}^{(1)}$ 的存储空间），内存开销低于 RK4。

3. **适中的精度与效率平衡**：对于 CFD 中常见的二阶/三阶空间精度配置（$N = 1\text{--}2$），三阶时间精度与空间精度匹配，避免了时间离散误差主导总误差的情况。

4. **稳定的 CFL 门槛**：SSP 性质给出了 CFL 数的严格上界，调试中无需试探性调整 CFL 数来保证稳定性。

### 2.5 OCCA 实现要点

SSPRK3 每时间步需执行三次 DG 场模块右端项求值和两次凸组合向量更新。OCCA 核函数结构如下：

凸组合更新核函数复用同一模板，通过参数 $\alpha$、$\beta$ 控制组合系数：

```c
// 凸组合更新：u_new = α * u_n + β * u_temp
@kernel void sspConvexCombine(const int N_dof,
                               const double alpha, const double beta,
                               const double *u_n,
                               const double *u_temp,
                               double *u_new) {
  for (int dof = 0; dof < N_dof; ++dof; @tile(TILE_SIZE, @outer, @inner)) {
    u_new[dof] = alpha * u_n[dof] + beta * u_temp[dof];
  }
}
```

每时间步的完整流程：

```
1.  RK步:    u1 = u^n + Δt·RHS(u^n)
2. 组合更新: u2 = 0.75·u^n + 0.25·(u1 + Δt·RHS(u1))
3. 组合更新: u^{n+1} = 1/3·u^n + 2/3·(u2 + Δt·RHS(u2))
```

其中 `RHS(·)` 调用 DG 场模块的完整右端项计算链（gradient → volumeIntegral → computeFaceFlux → gatherSurfaceRHS → assembleRHS）。向量更新核函数以总自由度为粒度并行，沿用 `@tile(TILE_SIZE, @outer, @inner)` 分块策略。

---

## 三、时间步长的 CFL 估计

### 3.1 一维对流 CFL

对于一维标量对流方程，DG 方法的显式 CFL 限值遵循：

$$
\Delta t_{\text{max}} = \frac{\text{CFL}}{2N + 1} \cdot \frac{h}{|u| + c}
$$

其中 $N$ 为多项式阶数，$h$ 为单元特征尺寸。CFL 数保守值取 $0.3\text{--}0.5$。

### 3.2 多维推广

对于二维/三维流动，特征尺寸 $h$ 取单元的最小特征尺度：

- 四边形单元：$h = \min(\Delta r, \Delta s)$ 对应的物理尺度
- 三角形单元：$h = \sqrt{2A}$，$A$ 为单元面积

全局稳定时间步长取所有单元中 $\Delta t_{\text{max}}$ 的最小值：

$$
\Delta t = \text{CFL} \cdot \min_{K} \frac{h_K}{(2N + 1) \, \Lambda_{\text{max}, K}}
$$

### 3.3 当地时间步长（定常问题）

对于定常问题收敛加速，可使用当地时间步长：每个单元独立使用其局部 CFL 限值确定 $\Delta t_K$。此时各单元不在同一物理时间推进，但以牺牲时间精度换取收敛速度。适用于显式欧拉法和 SSPRK3 的定常问题求解。

---

## 方法汇总

| 方法       | 精度  | 每步 RHS 次数 |  SSP 性质   | 存储开销 | CFL 限值 | 推荐场景                   |
| :--------- | :---: | :-----------: | :---------: | :------: | :------: | :------------------------- |
| 显式欧拉法 | 一阶  |       1       | 有（$c=1$） |   2 倍   |   最低   | 定常初场                   |
| SSPRK3     | 三阶  |       3       | 有（$c=1$） |   3 倍   |   适中   | **CFD 优选（含间断流动）** |

> **推荐**：对于 CMeles 中的可压缩流动求解（含激波、接触间断等间断），**SSPRK3 为首选显式时间推进方法**。其 SSP 性质确保限制器的 TVB 性质在时间离散中保持，三阶精度在 DG 框架下与空间精度良好匹配，三步存储的开销可接受。

---

## 参考文献

1. Shu, C.-W. & Osher, S. (1988). Efficient implementation of essentially non-oscillatory shock-capturing schemes. *Journal of Computational Physics*, 77(2), 439–471.
2. Gottlieb, S., Shu, C.-W. & Tadmor, E. (2001). Strong stability-preserving high-order time discretization methods. *SIAM Review*, 43(1), 89–112.
3. Cockburn, B. & Shu, C.-W. (2001). Runge–Kutta discontinuous Galerkin methods for convection-dominated problems. *Journal of Scientific Computing*, 16(3), 173–261.
4. Hesthaven, J. S. & Warburton, T. (2008). *Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications*. Springer.
