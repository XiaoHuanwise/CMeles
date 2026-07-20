# 黎曼求解器与数值通量

## 数值通量概述

在 DG 方法的面积分中（见[控制方程与 DG 场文档](governed_equations_and_DG_field.md#三面积分计算)），单元边界上需要计算数值通量 $\hat{\boldsymbol{F}} \cdot \boldsymbol{n}$ 以处理相邻单元间解的不连续性。对于无粘流动（Euler 方程），这等价于求解局部一维 Riemann 问题：给定面两侧状态 $\boldsymbol{q}_L$（左/内侧）和 $\boldsymbol{q}_R$（右/外侧），求沿法向的数值通量 $\hat{F}_n(\boldsymbol{q}_L, \boldsymbol{q}_R)$。

利用 Euler 方程的**旋转不变性**（详见[网格与几何文档 1.3 节](mesh_and_geometry.md#13-法向量)），三维问题可约化为局部坐标系下的一维 Riemann 问题：

$$
\begin{aligned}
\boldsymbol{q}^{\text{local}} &= \mathbf{T} \boldsymbol{q} \\
\hat{F}_n(\boldsymbol{q}_L, \boldsymbol{q}_R, \boldsymbol{n}) &= \mathbf{T}^{-1} \, \hat{F}^{\text{1D}}\big(\mathbf{T}\boldsymbol{q}_L, \mathbf{T}\boldsymbol{q}_R\big)
\end{aligned}
$$

其中 $\mathbf{T}$ 为旋转矩阵。以下各节介绍的一维 Riemann 求解器均在此局部坐标系下工作。

## 一维 Euler 方程与特征结构

在局部法向坐标系下，一维 Euler 方程为：

$$
\frac{\partial \boldsymbol{q}}{\partial t} + \frac{\partial \boldsymbol{f}(\boldsymbol{q})}{\partial x} = 0
$$

守恒变量和通量分别为：

$$
\boldsymbol{q} = \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho w \\ E \end{pmatrix}, \quad
\boldsymbol{f}(\boldsymbol{q}) = \begin{pmatrix} \rho u \\ p + \rho u^2 \\ \rho u v \\ \rho u w \\ u(E + p) \end{pmatrix}
$$

通量 Jacobian 矩阵 $\mathbf{A}(\boldsymbol{q}) = \dfrac{\partial \boldsymbol{f}}{\partial \boldsymbol{q}}$ 为：

$$
\mathbf{A} = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 \\
\frac{\gamma-3}{2}u^2 + \frac{\gamma-1}{2}(v^2+w^2) & (3-\gamma)u & -(\gamma-1)v & -(\gamma-1)w & \gamma-1 \\
-uv & v & u & 0 & 0 \\
-uw & w & 0 & u & 0 \\
u\left[\frac{\gamma-1}{2}(u^2+v^2+w^2) - H\right] & H - (\gamma-1)u^2 & -(\gamma-1)uv & -(\gamma-1)uw & \gamma u
\end{bmatrix}
$$

其中 $\gamma = c_p/c_v$ 为比热比（理想气体 $\gamma = 1.4$），总焓 $H = \dfrac{E + p}{\rho} = \dfrac{a^2}{\gamma-1} + \dfrac{1}{2}(u^2 + v^2 + w^2)$，声速 $a = \sqrt{\gamma p / \rho}$。

$\mathbf{A}$ 的特征值为：

$$
\lambda_1 = u - a,\quad \lambda_2 = \lambda_3 = \lambda_4 = u,\quad \lambda_5 = u + a
$$

**特征值的物理意义**：$\lambda_{1,5}$ 对应声波（沿法向传播的扰动），$\lambda_{2,3,4}$ 对应熵波和涡波（以流体速度对流）。特征值的符号决定了单元边界上信息的传播方向——当特征值为正，信息从左侧传向右侧；特征值为负时反之。

对应的右特征向量矩阵 $\mathbf{R}$（列排列）和左特征向量矩阵 $\mathbf{R}^{-1}$（行排列）如下（以守恒变量 $\boldsymbol{q} = (\rho, \rho u, \rho v, \rho w, E)^T$ 给出）：

$$
\mathbf{R} = \begin{bmatrix}
1 & 1 & 0 & 0 & 1 \\
u-a & u & 0 & 0 & u+a \\
v & v & 1 & 0 & v \\
w & w & 0 & 1 & w \\
H-ua & \frac{u^2+v^2+w^2}{2} & v & w & H+ua
\end{bmatrix},\quad
\mathbf{R}^{-1} = \begin{bmatrix}
\frac{1}{2}\left(b_1 + \frac{u}{a}\right) & -\frac{1}{2}\left(b_2 u + \frac{1}{a}\right) & -\frac{1}{2}b_2 v & -\frac{1}{2}b_2 w & \frac{b_2}{2} \\
1 - b_1 & b_2 u & b_2 v & b_2 w & -b_2 \\
-v & 0 & 1 & 0 & 0 \\
-w & 0 & 0 & 1 & 0 \\
\frac{1}{2}\left(b_1 - \frac{u}{a}\right) & -\frac{1}{2}\left(b_2 u - \frac{1}{a}\right) & -\frac{1}{2}b_2 v & -\frac{1}{2}b_2 w & \frac{b_2}{2}
\end{bmatrix}
$$

其中 $b_1 = \dfrac{\gamma-1}{a^2}\dfrac{u^2+v^2+w^2}{2}$，$b_2 = \dfrac{\gamma-1}{a^2}$。

## 对角化与通量差分分裂（FDS）

通量 Jacobian 矩阵 $\mathbf{A}$ 可对角化为 $\mathbf{A} = \mathbf{R} \boldsymbol{\Lambda} \mathbf{R}^{-1}$，其中 $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \lambda_2, \lambda_3, \lambda_4, \lambda_5)$。基于此分解，可将 $\mathbf{A}$ 按特征值的符号分裂为正部和负部：

$$
\mathbf{A} = \mathbf{A}^+ + \mathbf{A}^-,\quad
\mathbf{A}^{\pm} = \mathbf{R} \boldsymbol{\Lambda}^{\pm} \mathbf{R}^{-1},\quad
\boldsymbol{\Lambda}^{\pm} = \operatorname{diag}(\lambda_k^{\pm})
$$

其中 $\lambda_k^+ = \max(\lambda_k, 0)$，$\lambda_k^- = \min(\lambda_k, 0)$。

通量差分 $\Delta \boldsymbol{f} = \boldsymbol{f}_R - \boldsymbol{f}_L$ 满足 $\Delta \boldsymbol{f} = \tilde{\mathbf{A}} \Delta \boldsymbol{q}$（对于线性双曲系统严格成立；对于非线性系统，Roe 平均 Jacobian $\tilde{\mathbf{A}}$ 满足此等价关系）。从而数值通量可以统一写为：

$$
\boxed{
\hat{F}^{\text{1D}}(\boldsymbol{q}_L, \boldsymbol{q}_R) = \boldsymbol{f}_L + \tilde{\mathbf{A}}^- \Delta \boldsymbol{q} = \boldsymbol{f}_R - \tilde{\mathbf{A}}^+ \Delta \boldsymbol{q}
}
$$

或等价地采用对称形式：

$$
\hat{F}^{\text{1D}} = \frac{1}{2}\big(\boldsymbol{f}_L + \boldsymbol{f}_R\big) - \frac{1}{2}|\tilde{\mathbf{A}}| \Delta \boldsymbol{q}
$$

其中 $|\tilde{\mathbf{A}}| = \mathbf{A}^+ - \mathbf{A}^- = \tilde{\mathbf{R}} |\tilde{\boldsymbol{\Lambda}}| \tilde{\mathbf{R}}^{-1}$，$\tilde{\cdot}$ 表示通过某种平均方式（如 Roe 平均）计算的中间状态。上述通量公式的不同变体取决于如何选择平均状态以及如何定义 $|\tilde{\mathbf{A}}|$，由此导出 LLF、Roe 和带熵修正的 Roe 等不同 Riemann 求解器。

> **对流通量数值耗散**：上式中第二项 $\frac{1}{2}|\tilde{\mathbf{A}}| \Delta \boldsymbol{q}$ 为数值耗散项，作用是在单元界面上引入与特征速度成正比的耗散，以抑制由弱解不唯一引起的数值振荡（如激波处的 Gibbs 振荡、膨胀扇处的非物理解）。

## Local Lax-Friedrichs (LLF) / Rusanov 通量

### 基本公式

LLF 通量（也称 Rusanov 通量）是最简单、最鲁棒的 Riemann 求解器。它不对特征系统进行精细分解，而是用一个全局最大波速来估计数值耗散：

$$
\boxed{
\hat{F}^{\text{LLF}}(\boldsymbol{q}_L, \boldsymbol{q}_R) = \frac{1}{2}\big(\boldsymbol{f}_L + \boldsymbol{f}_R\big) - \frac{\alpha}{2}(\boldsymbol{q}_R - \boldsymbol{q}_L)
}
$$

其中 $\alpha$ 为**局部最大波速**，取两侧声速加权绝对速度的最大值：

$$
\alpha = \max\big(|u_L| + a_L,\; |u_R| + a_R\big)
$$

其中 $a_{L,R} = \sqrt{\gamma p_{L,R} / \rho_{L,R}}$ 为两侧声速，$u_{L,R}$ 为法向速度分量。

### 算法实现（OCCA 核函数伪代码）

```c++
// LLF/Rusanov 通量（对一维 Riemann 问题的一次面积分点计算）
// 输入：qL[5], qR[5] — 面两侧的守恒变量
// 输出：flux[5] — 法向数值通量
void LlfFlux(const double q_l[5], const double q_r[5], double flux[5]) {
    constexpr double gamma = 1.4;
    constexpr double gm1 = gamma - 1.0;

    // 1. 计算两侧的原始变量和通量
    double rho_l = q_l[0], rho_r = q_r[0];
    double u_l = q_l[1] / rho_l, u_r = q_r[1] / rho_r;
    double v_l = q_l[2] / rho_l, v_r = q_r[2] / rho_r;
    double w_l = q_l[3] / rho_l, w_r = q_r[3] / rho_r;
    double e_l = q_l[4], e_r = q_r[4];
    double ke_l = 0.5 * rho_l * (u_l*u_l + v_l*v_l + w_l*w_l);
    double ke_r = 0.5 * rho_r * (u_r*u_r + v_r*v_r + w_r*w_r);
    double p_l = gm1 * (e_l - ke_l);
    double p_r = gm1 * (e_r - ke_r);
    double a_l = std::sqrt(gamma * p_l / rho_l);
    double a_r = std::sqrt(gamma * p_r / rho_r);

    // 2. 计算物理通量
    double f_l[5] = { rho_l * u_l,
                      p_l + rho_l * u_l * u_l,
                      rho_l * u_l * v_l,
                      rho_l * u_l * w_l,
                      u_l * (e_l + p_l) };
    double f_r[5] = { rho_r * u_r,
                      p_r + rho_r * u_r * u_r,
                      rho_r * u_r * v_r,
                      rho_r * u_r * w_r,
                      u_r * (e_r + p_r) };

    // 3. 计算局部最大波速 alpha
    double alpha = std::max(std::abs(u_l) + a_l, std::abs(u_r) + a_r);

    // 4. 组装 LLF 通量：0.5*(fL+fR) - 0.5*alpha*(qR-qL)
    for (int k = 0; k < 5; ++k) {
        flux[k] = 0.5 * (f_l[k] + f_r[k]) - 0.5 * alpha * (q_r[k] - q_l[k]);
    }
}
```

### 特点与适用场景

| 特性     | 评价                                                                                           |
| :------- | :--------------------------------------------------------------------------------------------- |
| 鲁棒性   | **极高**——对激波、接触间断和膨胀扇均有良好表现，无膨胀激波（非物理解）问题                     |
| 精度     | 较粗糙——耗散与 $\alpha |\Delta\boldsymbol{q}|$ 成正比，在接触间断处（理论上无压力-速度跳跃）引入额外耗散 |
| 边界层   | 对剪切层/边界层的耗散较大，可能过度抹平速度剖面                                                 |
| 实现难度 | 简单——无需矩阵运算、无需特征分解                                                               |
| 计算成本 | 低——单次通量计算仅需 2 次 `std::sqrt` + 基本算术运算                                                |
| 适用场景 | 含强激波流动的初始开发/调试阶段；对精度要求不高的快速预览计算；作为更复杂求解器的鲁棒性基准     |

## Roe 通量

### 基本思路

Roe 通量是一种基于通量差分分裂（Flux Difference Splitting, FDS）的 Godunov 型近似 Riemann 求解器。其核心是构造一个满足特定性质的**平均状态**，使得 $\Delta \boldsymbol{f} = \tilde{\mathbf{A}} \Delta \boldsymbol{q}$ 对非线性 Euler 方程精确成立。

### Roe 平均

Roe 平均状态（以 $\tilde{\cdot}$ 表示）通过密度加权平均构造，满足性质：

1. $\tilde{\mathbf{A}}(\boldsymbol{q}_L, \boldsymbol{q}_R) \cdot (\boldsymbol{q}_R - \boldsymbol{q}_L) = \boldsymbol{f}_R - \boldsymbol{f}_L$（守恒性）
2. $\tilde{\mathbf{A}}(\boldsymbol{q}_L, \boldsymbol{q}_R) \to \mathbf{A}(\boldsymbol{q})$ 当 $\boldsymbol{q}_L, \boldsymbol{q}_R \to \boldsymbol{q}$（一致性）
3. $\tilde{\mathbf{A}}$ 可对角化且具有实特征值（双曲性）

对 Euler 方程，Roe 平均变量的具体形式为：

$$
\boxed{
\begin{aligned}
\tilde{\rho} &= \sqrt{\rho_L \rho_R} \\[4pt]
\tilde{u} &= \frac{\sqrt{\rho_L} \, u_L + \sqrt{\rho_R} \, u_R}{\sqrt{\rho_L} + \sqrt{\rho_R}} \\[4pt]
\tilde{v} &= \frac{\sqrt{\rho_L} \, v_L + \sqrt{\rho_R} \, v_R}{\sqrt{\rho_L} + \sqrt{\rho_R}} \\[4pt]
\tilde{w} &= \frac{\sqrt{\rho_L} \, w_L + \sqrt{\rho_R} \, w_R}{\sqrt{\rho_L} + \sqrt{\rho_R}} \\[4pt]
\tilde{H} &= \frac{\sqrt{\rho_L} \, H_L + \sqrt{\rho_R} \, H_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}
\end{aligned}
}
$$

其中 $H_{L,R} = \dfrac{E_{L,R} + p_{L,R}}{\rho_{L,R}}$ 为两侧的总焓。由 $\tilde{H}$ 和平均速度可逆推平均声速：

$$
\tilde{a} = \sqrt{(\gamma - 1)\left(\tilde{H} - \frac{1}{2}(\tilde{u}^2 + \tilde{v}^2 + \tilde{w}^2)\right)}
$$

**Roe 平均的物理意义**：Roe 平均本质上是一种密度加权的 Favre 平均——$\sqrt{\rho}$ 加权确保了守恒性条件 $\Delta\boldsymbol{f} = \tilde{\mathbf{A}}\Delta\boldsymbol{q}$ 对 Euler 方程精确成立。在数学上，若定义参数向量 $\boldsymbol{z} = \sqrt{\rho}(1, u, v, w, H)^T$，则 $\boldsymbol{q}$ 和 $\boldsymbol{f}$ 均为 $\boldsymbol{z}$ 的二次函数，Roe 平均恰好对应 $\boldsymbol{z}$ 的算术平均。

### Roe 通量公式

Roe 通量使用 Roe 平均状态下的 Jacobian 分解：

$$
\boxed{
\hat{F}^{\text{Roe}}(\boldsymbol{q}_L, \boldsymbol{q}_R) = \frac{1}{2}\big(\boldsymbol{f}_L + \boldsymbol{f}_R\big) - \frac{1}{2} \sum_{k=1}^{5} \tilde{\alpha}_k |\tilde{\lambda}_k| \tilde{\boldsymbol{r}}_k
}
$$

其中 $\tilde{\lambda}_k$ 和 $\tilde{\boldsymbol{r}}_k$ 分别为 $\tilde{\mathbf{A}}$ 的特征值和右特征向量，$\tilde{\alpha}_k$ 为 $\Delta \boldsymbol{q} = \boldsymbol{q}_R - \boldsymbol{q}_L$ 在右特征向量基下的展开系数。

**通量公式中各符号的含义**：

- $\frac{1}{2}(\boldsymbol{f}_L + \boldsymbol{f}_R)$：**中心部分**，即两侧物理通量的算术平均；
- $\frac{1}{2} \sum_k \tilde{\alpha}_k |\tilde{\lambda}_k| \tilde{\boldsymbol{r}}_k$：**迎风耗散部分**，$|\tilde{\lambda}_k|$ 表示各特征波的绝对速度，$|\tilde{\alpha}_k|$ 表示该特征波在间断中的跳跃强度。

换言之，Roe 通量 = 中心通量 − 对每个特征波按其绝对速度施加的迎风耗散。当某个特征速度接近零时（声速点、驻点），该波的耗散趋于零，这正是熵修正所要解决的问题。

> **等价表述**：Roe 通量也可写为 $\hat{F}^{\text{Roe}} = \boldsymbol{f}_L + \sum_{\tilde{\lambda}_k < 0} \tilde{\alpha}_k \tilde{\lambda}_k \tilde{\boldsymbol{r}}_k$——即从左侧通量出发，仅加上传入该点的负特征波贡献（Godunov 格式的 Roe 近似）。

### 波强系数 $\tilde{\alpha}_k$ 的展开

将 $\Delta \boldsymbol{q}$ 用原始变量差表示，波强系数为：

$$
\boxed{
\begin{aligned}
\tilde{\alpha}_1 &= \frac{1}{2\tilde{a}^2}\big(\Delta p - \tilde{\rho}\tilde{a} \Delta u\big) \\[4pt]
\tilde{\alpha}_2 &= \Delta\rho - \frac{\Delta p}{\tilde{a}^2} \\[4pt]
\tilde{\alpha}_3 &= \tilde{\rho} \Delta v \\[4pt]
\tilde{\alpha}_4 &= \tilde{\rho} \Delta w \\[4pt]
\tilde{\alpha}_5 &= \frac{1}{2\tilde{a}^2}\big(\Delta p + \tilde{\rho}\tilde{a} \Delta u\big)
\end{aligned}
}
$$

其中 $\Delta(\cdot) = (\cdot)_R - (\cdot)_L$。对应各 $\tilde{\alpha}_k$ 的特征向量 $\tilde{\boldsymbol{r}}_k$（以守恒变量表示）为：

$$
\begin{aligned}
\tilde{\boldsymbol{r}}_1 &= \begin{pmatrix} 1 \\ \tilde{u} - \tilde{a} \\ \tilde{v} \\ \tilde{w} \\ \tilde{H} - \tilde{u}\tilde{a} \end{pmatrix},\quad
\tilde{\boldsymbol{r}}_2 = \begin{pmatrix} 1 \\ \tilde{u} \\ \tilde{v} \\ \tilde{w} \\ \frac{\tilde{u}^2+\tilde{v}^2+\tilde{w}^2}{2} \end{pmatrix},\quad
\tilde{\boldsymbol{r}}_3 = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \\ \tilde{v} \end{pmatrix},\\[10pt]
\tilde{\boldsymbol{r}}_4 &= \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \\ \tilde{w} \end{pmatrix},\quad
\tilde{\boldsymbol{r}}_5 = \begin{pmatrix} 1 \\ \tilde{u} + \tilde{a} \\ \tilde{v} \\ \tilde{w} \\ \tilde{H} + \tilde{u}\tilde{a} \end{pmatrix}
\end{aligned}
$$

则数值耗散项按特征波展开为：

$$
\frac{1}{2} \sum_{k=1}^{5} \tilde{\alpha}_k |\tilde{\lambda}_k| \tilde{\boldsymbol{r}}_k
$$

### 算法实现（OCCA 核函数伪代码）

```c++
// Roe 通量（对一维 Riemann 问题的一次面积分点计算）
// 输入：qL[5], qR[5] — 面两侧的守恒变量
// 输出：flux[5] — 法向数值通量
// eps: 熵修正阈值（见下一节）
void RoeFlux(const double q_l[5], const double q_r[5], double flux[5],
             double eps) {
    constexpr double gamma = 1.4;
    constexpr double gm1 = gamma - 1.0;

    // 1. 提取并计算两侧原始变量
    double rho_l = q_l[0], rho_r = q_r[0];
    double u_l = q_l[1] / rho_l, u_r = q_r[1] / rho_r;
    double v_l = q_l[2] / rho_l, v_r = q_r[2] / rho_r;
    double w_l = q_l[3] / rho_l, w_r = q_r[3] / rho_r;
    double e_l = q_l[4], e_r = q_r[4];
    double ke_l = 0.5 * rho_l * (u_l*u_l + v_l*v_l + w_l*w_l);
    double ke_r = 0.5 * rho_r * (u_r*u_r + v_r*v_r + w_r*w_r);
    double p_l = gm1 * (e_l - ke_l);
    double p_r = gm1 * (e_r - ke_r);
    double h_l = (e_l + p_l) / rho_l;
    double h_r = (e_r + p_r) / rho_r;

    // 2. 计算物理通量 fL, fR
    double f_l[5] = { rho_l * u_l, p_l + rho_l*u_l*u_l, rho_l*u_l*v_l,
                      rho_l*u_l*w_l, u_l*(e_l + p_l) };
    double f_r[5] = { rho_r * u_r, p_r + rho_r*u_r*u_r, rho_r*u_r*v_r,
                      rho_r*u_r*w_r, u_r*(e_r + p_r) };

    // 3. 计算 Roe 平均状态
    double sr_l = std::sqrt(rho_l), sr_r = std::sqrt(rho_r);
    double s_sum_inv = 1.0 / (sr_l + sr_r);
    double rho_tilde = sr_l * sr_r;                    // \tilde{\rho}
    double u_tilde = (sr_l*u_l + sr_r*u_r) * s_sum_inv; // \tilde{u}
    double v_tilde = (sr_l*v_l + sr_r*v_r) * s_sum_inv; // \tilde{v}
    double w_tilde = (sr_l*w_l + sr_r*w_r) * s_sum_inv; // \tilde{w}
    double h_tilde = (sr_l*h_l + sr_r*h_r) * s_sum_inv; // \tilde{H}
    double ke_tilde = 0.5*(u_tilde*u_tilde + v_tilde*v_tilde + w_tilde*w_tilde);
    double a_tilde = std::sqrt(gm1 * (h_tilde - ke_tilde));  // \tilde{a}
    double a2_inv = 1.0 / (a_tilde * a_tilde);

    // 4. 计算原始变量差
    double du = u_r - u_l, dv = v_r - v_l, dw = w_r - w_l;
    double dp = p_r - p_l;

    // 5. 计算波强系数 alpha_k
    double alpha_1 = (dp - rho_tilde * a_tilde * du) * (0.5 * a2_inv);
    double alpha_2 = (rho_r - rho_l) - dp * a2_inv;
    double alpha_3 = rho_tilde * dv;
    double alpha_4 = rho_tilde * dw;
    double alpha_5 = (dp + rho_tilde * a_tilde * du) * (0.5 * a2_inv);

    // 6. 计算 Roe 平均特征值
    double lambda_tilde[5] = {
        u_tilde - a_tilde,   // lambda_1
        u_tilde,             // lambda_2
        u_tilde,             // lambda_3
        u_tilde,             // lambda_4
        u_tilde + a_tilde    // lambda_5
    };

    // 7. 带熵修正的绝对特征值（Roe-E 修正，见下一节详述）
    double abs_lambda[5];
    for (int k = 0; k < 5; ++k) {
        double lam = lambda_tilde[k];
        double abs_lam = std::abs(lam);
        // Harten 型熵修正（平滑过渡）
        if (abs_lam < eps) {
            abs_lam = 0.5 * (lam * lam / eps + eps);
        }
        abs_lambda[k] = abs_lam;
    }

    // 8. 右特征向量 r_k（守恒变量空间）
    double r1[5] = { 1.0, u_tilde - a_tilde, v_tilde, w_tilde,
                     h_tilde - u_tilde * a_tilde };
    double r2[5] = { 1.0, u_tilde, v_tilde, w_tilde, ke_tilde };
    double r3[5] = { 0.0, 0.0, 1.0, 0.0, v_tilde };
    double r4[5] = { 0.0, 0.0, 0.0, 1.0, w_tilde };
    double r5[5] = { 1.0, u_tilde + a_tilde, v_tilde, w_tilde,
                     h_tilde + u_tilde * a_tilde };

    // 9. 组装耗散项 disp = sum_k(alpha_k * |lambda_k| * r_k)
    double disp[5] = {0.0};
    double alpha_arr[5] = {alpha_1, alpha_2, alpha_3, alpha_4, alpha_5};
    const double* r_arr[5] = {r1, r2, r3, r4, r5};
    for (int k = 0; k < 5; ++k) {
        double coeff = alpha_arr[k] * abs_lambda[k];
        for (int m = 0; m < 5; ++m) {
            disp[m] += coeff * r_arr[k][m];
        }
    }

    // 10. 组装 Roe 通量：0.5*(fL+fR) - 0.5*disp
    for (int k = 0; k < 5; ++k) {
        flux[k] = 0.5 * (f_l[k] + f_r[k]) - 0.5 * disp[k];
    }
}
```

### 特点与适用场景

| 特性     | 评价                                                                       |
| :------- | :------------------------------------------------------------------------- |
| 精度     | **高**——对接触间断和剪切层有精确分辨率（$\lambda_{2,3,4}$ 的耗散为零）     |
| 边界层   | 对粘性边界层的分辨率优于 LLF                                               |
| 间断分辨 | 激波分辨率高于 LLF，可精确捕捉孤立接触间断                                  |
| 缺陷     | 在声速点（$|u| = a$）和膨胀扇处产生非物理的膨胀激波（rarefaction shock）   |
| 实现难度 | 较高——需要特征分解、Roe 平均及矩阵-向量运算                                |
| 计算成本 | 中等——单次通量计算涉及 1 次 `std::sqrt`、特征值/向量构造和向量内积               |
| 适用场景 | 粘性流动的边界层区域；不含强膨胀扇的无粘流动（配合熵修正）；对精度要求较高的计算 |

## 带熵修正的 Roe 通量

### 膨胀激波问题

Roe 通量的**根本缺陷**在于：当某条特征曲线的特征值 $\tilde{\lambda}_k$ 过零（即波速改变符号）时，若直接使用 $|\tilde{\lambda}_k|$，则耗散项局部为零，Godunov 格式在此处退化为中心差分，导致膨胀扇中出现非物理解——**膨胀激波**（rarefaction shock）。物理上，膨胀扇应由一系列稀疏波组成（熵增过程），而非压缩间断（熵减过程）。

经典的 Sod 激波管问题中，膨胀扇的头部（$u - a$ 从负变正）若不加熵修正，Roe 通量会在声速点产生非物理解。

### Harten–Hyman 型熵修正（Roe-E）

最常用的修正方案是 Harten 和 Hyman (1983) 提出的熵修正：

$$
\boxed{
|\tilde{\lambda}_k|_{\text{EC}} = \begin{cases}
|\tilde{\lambda}_k|, & |\tilde{\lambda}_k| \ge \delta_k \\[4pt]
\dfrac{\tilde{\lambda}_k^2 + \delta_k^2}{2\delta_k}, & |\tilde{\lambda}_k| < \delta_k
\end{cases}
}
$$

其中 $\delta_k$ 为修正阈值，可取为统一值 $\delta$ 或对每条特征波分别定义。

**修正项的推导**：上述公式等价于用光滑函数 $f(\lambda, \delta) = (\lambda^2+\delta^2)/(2\delta)$ 在 $[-\delta, \delta]$ 区间内连续连接 $|\lambda|$ 的两支。该函数满足 $f(\pm\delta, \delta) = \delta$、$f'(\pm\delta) = \pm 1 = \frac{d}{d\lambda}|\lambda|\big|_{\lambda=\pm\delta}$，因此 $f$ 在 $\pm\delta$ 处 $C^1$ 连续；$f'(0, \delta) = 0$，在过零点处耗散不消失。

阈值 $\delta_k$ 的常见取法：

$$
\begin{aligned}
\delta_k &= \varepsilon \cdot \max(0,\, \lambda_{k,R} - \lambda_{k,L}) \qquad &\text{(Harten 原始方案)} \\[4pt]
\delta_k &= \max(0,\, \lambda_{k,R} - \lambda_{k,L}) \qquad &\text{(激波只在压缩波处修正)} \\[4pt]
\delta_k &= \varepsilon \cdot \big(|u_L| + a_L + |u_R| + a_R\big) \qquad &\text{(全局速度尺度)}
\end{aligned}
$$

其中 $\varepsilon$ 为一个小正数（典型值 0.05–0.2），$\lambda_{k,L}$ 和 $\lambda_{k,R}$ 分别为左右状态下的第 $k$ 个特征值。

> **$\delta_k$ 的选取原则**：取左右特征值之差可以自动识别真正需要修正的位置——若 $\lambda_{k,L} \approx \lambda_{k,R}$（线性波），则 $\delta_k \approx 0$，无需修正；若特征值差异大（如膨胀扇头部，$\lambda_{1,L} < 0 < \lambda_{1,R}$），$\delta_k > 0$ 触发修正。

### 熵修正的物理原理

膨胀扇由一族稀疏波组成，穿过这些波时各特征量（Riemann 不变量）发生连续变化。准确的 Godunov 格式会将膨胀扇内部结构解析为一组中间状态——以声速点为中心，$u-a$ 线性地从左侧负值过渡到右侧正值。

Roe 近似只取一个平均状态 $\tilde{\mathbf{A}}$（即 Roe 平均 Jacobian），将非线性波简化为线性间断。当 $\tilde{\lambda}_k$ 过零时，该波恰好落在跨声速膨胀扇的中心，此时线性化近似失效——精确的通量值取决于 $\tilde{\mathbf{A}}$ 两侧 $u \pm a$ 的详细变化，而非仅仅平均状态的符号。

熵修正通过人为地增大过零特征值的耗散来近似膨胀扇内部结构——Harten–Hyman 修正等价于在 $|\lambda| < \delta$ 区间内引入一个最小耗散水平 $\delta/2$，确保跨声速膨胀扇处耗散不为零，从而抑制非物理的膨胀激波。

> **注意**：熵修正只在膨胀扇/声速点的局部区域（$|\tilde{\lambda}_k| < \delta_k$）激活，对激波和接触间断（$|\tilde{\lambda}_k| \gg \delta_k$）不产生影响。

### 改进的 Roe 通量公式

带熵修正的 Roe 通量公式与标准 Roe 形式相同，仅将 $|\tilde{\lambda}_k|$ 替换为 $|\tilde{\lambda}_k|_{\text{EC}}$：

$$
\boxed{
\hat{F}^{\text{Roe-E}}(\boldsymbol{q}_L, \boldsymbol{q}_R) = \frac{1}{2}\big(\boldsymbol{f}_L + \boldsymbol{f}_R\big) - \frac{1}{2} \sum_{k=1}^{5} \tilde{\alpha}_k |\tilde{\lambda}_k|_{\text{EC}} \, \tilde{\boldsymbol{r}}_k
}
$$

### 算法实现中的熵修正函数

```c++
// 带 Harten 型熵修正的绝对特征值计算
// lam: Roe 平均特征值
// lam_l: 左侧对应特征值
// lam_r: 右侧对应特征值
// eps: 修正系数（典型值 0.1）
inline double EntropyFixAbs(double lam, double lam_l, double lam_r,
                             double eps) {
    double delta = eps * std::max(0.0, lam_r - lam_l);
    double abs_lam = std::abs(lam);
    if (abs_lam < delta) {
        abs_lam = 0.5 * (lam * lam / delta + delta);
    }
    return abs_lam;
}
```

> **简化实现**：若计算效率优先，可使用全局阈值 $\delta = \varepsilon \cdot \max(|u_L|+a_L, |u_R|+a_R)$ 替代逐波特征值差，避免额外计算 $\lambda_{k,L}$ 和 $\lambda_{k,R}$，但会略微过度耗散。

### 特点与适用场景

| 特性     | 评价                                                                     |
| :------- | :----------------------------------------------------------------------- |
| 精度     | 接近 Roe——仅在膨胀扇声速点附近引入额外耗散                               |
| 鲁棒性   | 消除膨胀激波问题；对真空边界仍需额外处理                                 |
| 计算成本 | 略高于标准 Roe——每个特征值多一次分支判断和条件赋值                       |
| 适用场景 | **推荐作为默认的 Roe 求解器变体**——几乎不损失精度，代价可忽略            |

## 通量向量分裂（FVS）

### FVS 基本原理

FDS 方法（Roe）通过近似 Riemann 问题间接构造数值通量——将通量差 $\Delta\boldsymbol{f}$ 投影到特征空间，再正负分裂后重构。**通量向量分裂（Flux Vector Splitting, FVS）** 则换一条路径：利用 Euler 方程通量的齐次性（一阶齐次函数），**直接**将通量向量 $\boldsymbol{f}(\boldsymbol{q})$ 按 Jacobian 特征值的正负分裂为 $\boldsymbol{f}^+$ 和 $\boldsymbol{f}^-$，然后分别从左、右状态迎风取用：

$$
\boxed{
\hat{F}^{\text{FVS}}(\boldsymbol{q}_L, \boldsymbol{q}_R) = \boldsymbol{f}^+(\boldsymbol{q}_L) + \boldsymbol{f}^-(\boldsymbol{q}_R)
}
$$

FVS 不需要求解 Riemann 问题，不需要 Roe 平均——它对左右状态直接计算各自的 $\boldsymbol{f}^{\pm}$，物理意义清晰：**正特征值对应的通量信息从左侧传入面，负特征值对应的通量信息从右侧传入面**。

### 齐次性

对于 Euler 方程，通量向量 $\boldsymbol{f}$ 是守恒变量 $\boldsymbol{q}$ 的一阶齐次函数（$f_i$ 是 $q_j$ 的线性组合），因此满足 Euler 定理：

$$
\boldsymbol{f} = \frac{\partial\boldsymbol{f}}{\partial\boldsymbol{q}} \boldsymbol{q} = \mathbf{A} \boldsymbol{q}
$$

代入 $\mathbf{A} = \mathbf{R}\boldsymbol{\Lambda}\mathbf{R}^{-1}$ 和 $\boldsymbol{\Lambda} = \boldsymbol{\Lambda}^+ + \boldsymbol{\Lambda}^-$，得：

$$
\boldsymbol{f} = \mathbf{A}\boldsymbol{q} = \mathbf{R}(\boldsymbol{\Lambda}^+ + \boldsymbol{\Lambda}^-)\mathbf{R}^{-1}\boldsymbol{q}
= \underbrace{\mathbf{R}\boldsymbol{\Lambda}^+\mathbf{R}^{-1}\boldsymbol{q}}_{\boldsymbol{f}^+} + \underbrace{\mathbf{R}\boldsymbol{\Lambda}^-\mathbf{R}^{-1}\boldsymbol{q}}_{\boldsymbol{f}^-}
$$

因此 $\boldsymbol{f}^{\pm} = \mathbf{A}^{\pm}\boldsymbol{q} = \mathbf{R}\boldsymbol{\Lambda}^{\pm}\mathbf{R}^{-1}\boldsymbol{q}$。不同的 FVS 方案在**如何定义 $\boldsymbol{\Lambda}^{\pm}$**（也即 $\lambda_k^{\pm}$）以及**如何（或是否引入额外处理以保证光滑性）** 上有所不同。

### Steger-Warming 分裂

#### 基本公式

Steger 和 Warming (1981) 提出直接按原始特征值的符号进行分裂：

$$
\boldsymbol{f}^{\pm}(\boldsymbol{q}) = \mathbf{A}^{\pm}(\boldsymbol{q}) \boldsymbol{q} = \mathbf{R}(\boldsymbol{q}) \boldsymbol{\Lambda}^{\pm}(\boldsymbol{q}) \mathbf{R}^{-1}(\boldsymbol{q}) \boldsymbol{q}
$$

其中 $\lambda_k^+ = \frac{\lambda_k + |\lambda_k|}{2} = \max(\lambda_k, 0)$，$\lambda_k^- = \frac{\lambda_k - |\lambda_k|}{2} = \min(\lambda_k, 0)$。

代入 Euler 方程的特征值 $\lambda_1=u-a,\ \lambda_{2,3,4}=u,\ \lambda_5=u+a$，可得 $\boldsymbol{f}^{\pm}$ 的显式表达式。以 $M = u/a$（法向马赫数）为参数：

**情形 1：$M \ge 1$（超音速右行）**——所有 $\lambda_k \ge 0$：
$$
\boldsymbol{f}^+ = \boldsymbol{f},\quad \boldsymbol{f}^- = \boldsymbol{0}
\;\Longrightarrow\; \hat{F} = \boldsymbol{f}(\boldsymbol{q}_L)
$$

**情形 2：$M \le -1$（超音速左行）**——所有 $\lambda_k \le 0$：
$$
\boldsymbol{f}^+ = \boldsymbol{0},\quad \boldsymbol{f}^- = \boldsymbol{f}
\;\Longrightarrow\; \hat{F} = \boldsymbol{f}(\boldsymbol{q}_R)
$$

**情形 3：$|M| < 1$（亚音速）**——$u-a < 0 < u+a$：
$$
\begin{aligned}
\lambda_1^+ &= 0, &\lambda_1^- &= u - a \\
\lambda_{2,3,4}^+ &= \begin{cases} u, & u \ge 0 \\ 0, & u < 0 \end{cases}, &
\lambda_{2,3,4}^- &= \begin{cases} 0, & u \ge 0 \\ u, & u < 0 \end{cases} \\
\lambda_5^+ &= u + a, &\lambda_5^- &= 0
\end{aligned}
$$

通量 $\boldsymbol{f}^+$ 仅含 $\lambda_5$ 波的贡献（当 $u>0$ 时加上 $\lambda_{2,3,4}$ 波），$\boldsymbol{f}^-$ 仅含 $\lambda_1$ 波的贡献（当 $u<0$ 时加上 $\lambda_{2,3,4}$ 波）。

代入 $\mathbf{R}$ 和 $\mathbf{R}^{-1}$ 展开整理，可得亚音速区域 $\boldsymbol{f}^{\pm}$ 的封闭形式（一维，以守恒变量表示）：

$$
\boxed{
\begin{aligned}
\boldsymbol{f}^+ &= \frac{\rho}{2\gamma}
\begin{pmatrix}
2(\gamma-1)\max(u,0) + \max(u-a,0) + \max(u+a,0) \\
2(\gamma-1)\max(u,0)u + \max(u-a,0)(u-a) + \max(u+a,0)(u+a) \\
2(\gamma-1)\max(u,0)v + \max(u-a,0)v + \max(u+a,0)v \\
2(\gamma-1)\max(u,0)w + \max(u-a,0)w + \max(u+a,0)w \\
\text{(能量项——下详)}
\end{pmatrix} \\[10pt]
\boldsymbol{f}^- &= \frac{\rho}{2\gamma}
\begin{pmatrix}
2(\gamma-1)\min(u,0) + \min(u-a,0) + \min(u+a,0) \\
2(\gamma-1)\min(u,0)u + \min(u-a,0)(u-a) + \min(u+a,0)(u+a) \\
2(\gamma-1)\min(u,0)v + \min(u-a,0)v + \min(u+a,0)v \\
2(\gamma-1)\min(u,0)w + \min(u-a,0)w + \min(u+a,0)w \\
\text{(能量项——下详)}
\end{pmatrix}
\end{aligned}
}
$$

其中能量项的 min/max 组合结构类似，对每条特征波使用其对应的能量因子加权。

#### 声速点导数不连续性

Steger-Warming 分裂在**声速点**（$u=0$ 或 $u=\pm a$，即某个 $\lambda_k = 0$）处存在导数不连续——$\max(\lambda_k, 0)$ 和 $\min(\lambda_k, 0)$ 的导数在 $\lambda_k=0$ 处有跳跃。这会在声速点附近产生数值振荡（"glitch"），尤其在亚音速边界层和跨声速膨胀扇中表现明显。

#### 算法实现（OCCA 伪代码——基于直接特征分解的 Steger-Warming）

```c++
// Steger-Warming FVS 通量
// 输入：qL[5], qR[5] — 面两侧守恒变量
// 输出：flux[5] — 法向数值通量
void StegerWarmingFlux(const double q_l[5], const double q_r[5],
                       double flux[5]) {
    double fp[5], fm[5];  // f^+(qL), f^-(qR)

    // f^+ 和 f^- 分别从左、右状态独立计算
    ComputeSplitFluxPositive(q_l, fp);
    ComputeSplitFluxNegative(q_r, fm);

    // FVS 数值通量 = f^+(qL) + f^-(qR)
    for (int k = 0; k < 5; ++k) {
        flux[k] = fp[k] + fm[k];
    }
}

// 对给定状态 q，计算其正分裂通量 f^+
void ComputeSplitFluxPositive(const double q[5], double fp[5]) {
    constexpr double gamma = 1.4;
    double rho = q[0];
    double u = q[1] / rho;
    double v = q[2] / rho;
    double w = q[3] / rho;
    double e = q[4];
    double ke = 0.5 * rho * (u*u + v*v + w*w);
    double p = (gamma - 1.0) * (e - ke);
    double a = std::sqrt(gamma * p / rho);

    // 特征值
    double lam[5] = { u - a, u, u, u, u + a };

    // 特征波投影：alpha = R^{-1} q
    double a2_inv = 1.0 / (a * a);
    double b1 = (gamma - 1.0) * a2_inv * ke / rho;
    double b2 = (gamma - 1.0) * a2_inv;
    double h = (e + p) / rho;

    double alpha[5];
    alpha[0] = 0.5 * ((b1 + u/a) * rho - (b2*u + 1.0/a) * rho*u
                    - b2 * v * rho*v - b2 * w * rho*w + b2 * e);
    alpha[1] = (1.0 - b1) * rho + b2 * u * rho*u + b2 * v * rho*v
             + b2 * w * rho*w - b2 * e;
    alpha[2] = -v * rho + rho * v;   // = 0（一致性检验）
    alpha[3] = -w * rho + rho * w;   // = 0
    alpha[4] = 0.5 * ((b1 - u/a) * rho - (b2*u - 1.0/a) * rho*u
                    - b2 * v * rho*v - b2 * w * rho*w + b2 * e);

    // 右特征向量
    // r1 = (1, u-a, v, w, H-u*a)
    // r2 = (1, u, v, w, (u^2+v^2+w^2)/2)
    // r3 = (0, 0, 1, 0, v)
    // r4 = (0, 0, 0, 1, w)
    // r5 = (1, u+a, v, w, H+u*a)

    for (int m = 0; m < 5; ++m) { fp[m] = 0.0; }
    const double r_val[5][5] = {
        { 1.0, u - a, v, w, h - u*a },
        { 1.0, u,     v, w, ke / rho },
        { 0.0, 0.0,   1.0, 0.0, v },
        { 0.0, 0.0,   0.0, 1.0, w },
        { 1.0, u + a, v, w, h + u*a }
    };
    for (int k = 0; k < 5; ++k) {
        double lam_p = std::max(lam[k], 0.0);  // lambda_k^+
        if (lam_p > 0.0) {
            for (int m = 0; m < 5; ++m) {
                fp[m] += alpha[k] * lam_p * r_val[k][m];
            }
        }
    }
}

// ComputeSplitFluxNegative 结构与 ComputeSplitFluxPositive 对称，
// 仅将 std::max(lam[k], 0.0) 替换为 std::min(lam[k], 0.0)
```

#### 特点

| 特性     | 评价                                                                             |
| :------- | :------------------------------------------------------------------------------- |
| 精度     | 中等——激波分辨率介于 LLF 和 Roe 之间                                             |
| 鲁棒性   | 较好——天然满足迎风特性，不会产生膨胀激波                                         |
| 接触间断 | 抹平较严重——涡波和熵波（$\lambda = u$）内部耗散偏大                             |
| 声速点   | 存在导数不连续导致的数值振荡（"glitch"）                                         |
| 计算成本 | 高——需在**每个状态**（非仅平均状态）做完整的特征分解                             |
| 适用场景 | 超音速/高超声速流动中表现良好；亚音速和跨声速区域不推荐（建议改用 van Leer）     |

### van Leer 分裂

#### 基本思路

van Leer (1982) 针对 Steger-Warming 的两大缺陷——声速点导数不连续和接触间断耗散过大——提出了改进方案。van Leer 分裂的核心要求：

1. **$C^1$ 连续性**：$\boldsymbol{f}^{\pm}$ 在 $M = \pm 1$ 处光滑可导（消除 glitch）
2. **质量通量连续性**：$\rho u = f_1^+ + f_1^-$ 在 $|M|<1$ 时保持连续
3. **对称性**：$\boldsymbol{f}^+(M) = -\boldsymbol{f}^-(-M)$
4. **渐近行为**：$M \to +1$ 时 $\boldsymbol{f}^+ \to \boldsymbol{f}$、$\boldsymbol{f}^- \to 0$；$M \to -1$ 时相反

#### 一维 Euler 方程的 van Leer 分裂

van Leer 通过马赫数多项式插值构造通量分裂。对于一维 Euler 方程（$u$ 为法向速度），定义局部法向马赫数 $M = u/a$：

**超音速区**（$M \ge 1$ 或 $M \le -1$）——与 Steger-Warming 完全一致，纯迎风：

$$
\begin{aligned}
M \ge 1 &: \quad \boldsymbol{f}^+ = \boldsymbol{f},\quad \boldsymbol{f}^- = \boldsymbol{0} \\[4pt]
M \le -1 &: \quad \boldsymbol{f}^+ = \boldsymbol{0},\quad \boldsymbol{f}^- = \boldsymbol{f}
\end{aligned}
$$

**亚音速区**（$|M| < 1$）——采用马赫数多项式的光滑分裂：

$$
\boxed{
\begin{aligned}
f_1^{\pm} &= \pm \rho a \left(\frac{M \pm 1}{2}\right)^2 \\[6pt]
f_2^{\pm} &= f_1^{\pm} \left[\frac{(\gamma-1)u \pm 2a}{\gamma}\right] \\[6pt]
f_3^{\pm} &= f_1^{\pm} v \\[6pt]
f_4^{\pm} &= f_1^{\pm} w \\[6pt]
f_5^{\pm} &= f_1^{\pm} \left[\frac{((\gamma-1)u \pm 2a)^2}{2(\gamma^2-1)} + \frac{v^2+w^2}{2}\right]
\end{aligned}
}
$$

**分裂公式的关键性质**：

- **$C^1$ 连续性**：$(M \pm 1)^2$ 因子在 $M = \mp 1$ 处函数值和一阶导数均为零，保证超音速-亚音速过渡光滑
- **质量通量一致**：$f_1^+ + f_1^- = \frac{\rho a}{4}\left[(M+1)^2 - (1-M)^2\right] = \rho a M = \rho u$（注意符号约定）
- **$f_3^{\pm}$ 和 $f_4^{\pm}$**：切向动量通量与质量通量成正比，精确捕捉剪切层的对流特征
- **对称性**：$f_k^+(M) = -f_k^-(-M)$ 对所有分量成立

> **$C^1$ 连续性的意义**：Steger-Warming 中 $\lambda_k^{\pm} = \frac{1}{2}(\lambda_k \pm |\lambda_k|)$ 在 $\lambda_k=0$ 处导数跳变，导致离散格式在声速点出现伪振荡。van Leer 的 $(M\pm 1)^2$ 是光滑的二次函数，其导数 $\frac{d}{dM}$ 在 $M=\mp 1$ 处连续地趋于零，从根源上消除了 glitch。

#### 算法实现（OCCA 伪代码）

```c++
// van Leer FVS 通量
// 输入：qL[5], qR[5] — 面两侧的守恒变量（已在局部法向坐标系中）
// 输出：flux[5] — 法向数值通量
void VanLeerFlux(const double q_l[5], const double q_r[5], double flux[5]) {
    constexpr double gamma = 1.4;
    constexpr double gm1 = gamma - 1.0;
    constexpr double g2m1 = gamma * gamma - 1.0;  // = (γ-1)(γ+1)

    double fp[5], fm[5];  // f^+(qL), f^-(qR)

    VanLeerSplitPositive(q_l, gamma, gm1, g2m1, fp);
    VanLeerSplitNegative(q_r, gamma, gm1, g2m1, fm);

    // FVS = f^+(qL) + f^-(qR)
    for (int k = 0; k < 5; ++k) {
        flux[k] = fp[k] + fm[k];
    }
}

// 对给定状态 q 计算 van Leer 正分裂通量 f^+
void VanLeerSplitPositive(const double q[5], double gamma, double gm1,
                          double g2m1, double fp[5]) {
    double rho = q[0];
    double u = q[1] / rho;    // 法向速度
    double v = q[2] / rho;    // 切向速度
    double w = q[3] / rho;    // 展向速度
    double e = q[4];
    double ke = 0.5 * rho * (u*u + v*v + w*w);
    double p = gm1 * (e - ke);
    double a = std::sqrt(gamma * p / rho);
    double mach = u / a;

    if (mach >= 1.0) {
        // 超音速右行：f^+ = f
        fp[0] = rho * u;
        fp[1] = p + rho * u * u;
        fp[2] = rho * u * v;
        fp[3] = rho * u * w;
        fp[4] = u * (e + p);
    } else if (mach <= -1.0) {
        // 超音速左行：f^+ = 0
        for (int k = 0; k < 5; ++k) { fp[k] = 0.0; }
    } else {
        // 亚音速 |M| < 1：使用 van Leer 分裂公式
        double mp = 0.25 * (mach + 1.0) * (mach + 1.0);  // ((M+1)/2)^2
        double f1 = rho * a * mp;                         // 质量通量 f_1^+
        double vel_raw = gm1 * u + 2.0 * a;               // (γ-1)u + 2a（未除γ）
        // f_2^+ = f_1^+ * vel_raw / γ；能量项分母用 2(γ²-1)
        fp[0] = f1;
        fp[1] = f1 * vel_raw / gamma;                     // 法向动量通量
        fp[2] = f1 * v;
        fp[3] = f1 * w;
        fp[4] = f1 * (vel_raw * vel_raw / (2.0 * g2m1)   // 能量项
                      + 0.5 * (v*v + w*w));
    }
}

// VanLeerSplitNegative 结构与 VanLeerSplitPositive 对称：
// - M >= 1 → f^- = 0；M <= -1 → f^- = f
// - |M| < 1: Mm = 0.25 * (M-1)^2, f1 = -rho*a*Mm,
//   vel_raw = gm1*u - 2*a, 通量分量公式同正分裂结构
```

> **能量项计算说明**：能量项公式为：
> $$
> f_5^{\pm} = f_1^{\pm} \left[
> \frac{((\gamma-1)u \pm 2a)^2}{2(\gamma^2-1)} + \frac{v^2+w^2}{2}
> \right]
> $$
> 代码中 `vel_raw = gm1*u ± 2*a`，则 `vel_raw/γ` 给出 $(f_2^{\pm}/f_1^{\pm})$，`vel_raw² / (2·g2m1)` 给出能量项的第一部分（$g2m1 \equiv \gamma^2-1$）。

#### 优势分析

| 优势             | 说明                                                                                 |
| :--------------- | :----------------------------------------------------------------------------------- |
| $C^1$ 光滑       | 声速点导数为零，消除 Steger-Warming 的数值振荡                                       |
| 鲁棒性           | 无膨胀激波、无 carbuncle 现象（与 Roe 相比），适用于高马赫数含强激波流动             |
| 接触间断         | 优于 Steger-Warming（van Leer 考虑了切向动量对流），但不如 Roe                        |
| 实现复杂度       | 远低于 Roe / Steger-Warming——无需特征分解、矩阵运算，仅代数四则运算                  |
| 计算效率         | **高于 Roe**——无矩阵运算、无特征分解，单次通量仅 1 次 `sqrt` + 若干乘加运算          |
| 强激波           | 表现优于 Roe（不依赖特征结构的线性近似），在高马赫激波前后更为稳定                   |

> **carbuncle 现象**：Roe 和其他 FDS 格式在计算高马赫数钝头体绕流的弓形激波时，激波驻点区会产生非物理的红外分叉（"carbuncle"），原因是接触/剪切波上数值耗散趋近于零。FVS 对接触间断有一定耗散，天然免疫 carbuncle。

#### 与 Steger-Warming 的声速行为对比

| 工况             | Steger-Warming                                   | van Leer                                       |
| :--------------- | :----------------------------------------------- | :--------------------------------------------- |
| $u \to a^-$      | $\partial \boldsymbol{f}^+/\partial u$ 跳跃      | $\partial \boldsymbol{f}^+/\partial u$ 光滑    |
| $u \to 0$        | $\partial \boldsymbol{f}^+/\partial u$ 跳跃      | $\partial \boldsymbol{f}^+/\partial u$ 光滑    |
| $u \to -a^+$     | $\partial \boldsymbol{f}^-/\partial u$ 跳跃      | $\partial \boldsymbol{f}^-/\partial u$ 光滑    |

### FVS 方法总结

| 编号 | 方法            | 分裂原理                     | 关键机制                       | 首选场景                               |
| :--- | :-------------- | :--------------------------- | :----------------------------- | :------------------------------------- |
| 1    | Steger-Warming  | $\mathbf{A}^{\pm}\boldsymbol{q}$，按 $\lambda_k$ 符号 | 特征分解 → 正负特征值分组     | 高超声速纯超音速流动（无跨声速区）     |
| 2    | van Leer        | 马赫数多项式 ${(M \pm 1)^2}$ | $C^1$ 光滑插值消除 glitch      | **亚/跨/超音速通用**——含强激波的流动   |

> **FVS vs FDS 的统一视角**：两类方法殊途同归。FDS（Roe）从**通量差**入手：$\hat{F} = \frac{1}{2}(f_L+f_R) - \frac{1}{2}|\tilde{\mathbf{A}}|\Delta\boldsymbol{q}$，重在精确捕捉间断结构（$\Delta\boldsymbol{q}$ 的各个特征分量）；FVS 从**通量自身**（Steger-Warming: $f^{\pm} = \mathbf{A}^{\pm}\boldsymbol{q}$, van Leer: 代数分裂）入手：$\hat{F} = f^+(q_L) + f^-(q_R)$，重在光滑性和鲁棒性。FDS 精度高但易出非物理解；FVS 鲁棒但耗散偏大。工程上常将二者混合（如 AUSM 族格式）。

## 五种通量对比总结

| 编号 | 通量类型            | 类别 | 鲁棒性 | 精度 | 计算量 | 典型适用场景                                           |
| :--- | :------------------ | :--: | :----: | :--: | :----: | :----------------------------------------------------- |
| 1    | LLF (Rusanov)       | FDS  |   高   |  低  |   低   | 强激波、开发调试、精度基准                             |
| 2    | Roe                 | FDS  |   低   |  高  |   中   | 不含声速点的流动、接触间断主导的流动                   |
| 3    | Roe-E（带熵修正）   | FDS  |   中   |  高  |   中   | **通用推荐**——粘性流动、含膨胀扇的无粘流动、边界层     |
| 4    | Steger-Warming      | FVS  |   中   |  中  |   高   | 高超声速纯超音速流动（跨声速区会出 glitch）            |
| 5    | van Leer            | FVS  |   高   |  中  |  低–中 | 亚/跨/超音速通用——含强激波流动、避免 carbuncle 的场景  |

> **实际工程建议**：开发初期用 LLF 进行基本验证；对精度要求提升后切换到 Roe-E；含极强激波且需最大鲁棒性的工况（如高马赫钝头体绕流）优先考虑 van Leer。在 CMeles 的核函数实现中，通量类型通过配置文件（TOML）中的参数选择，OCCA 核函数通过 `const int flux_type` 参数（而非编译期宏）进行分支，避免 JIT 重新编译。

## 一维 Riemann 求解器在 CMeles 中的嵌入方式

在 [控制方程与 DG 场文档](governed_equations_and_DG_field.md#三面积分计算)的`computeFaceFlux` 核函数中，每个面单元的每个面积分点 $t_i$ 上调用一维 Riemann 求解器。整体流程为：

1. **局部坐标旋转**：将物理坐标系下的 $\boldsymbol{q}_L$、$\boldsymbol{q}_R$ 和面法向量 $\boldsymbol{n}$ 通过旋转矩阵 $\mathbf{T}$ 转换为局部法向-切向坐标系；
2. **一维 Riemann 求解**：在局部坐标系下，调用 LLF / Roe / Roe-E / Steger-Warming / van Leer 求解器，得到局部坐标系下的法向数值通量 $\hat{F}^{\text{1D}}$；
3. **通量旋转回全局坐标**：将 $\hat{F}^{\text{1D}}$ 通过 $\mathbf{T}^{-1}$ 旋转回物理坐标系，得到 $\hat{\boldsymbol{F}} \cdot \boldsymbol{n}$。

此流程在每个面单元的每个面积分点上独立执行（串行），面单元间通过 `@tile(TILE_SIZE, @outer, @inner)` 并行。

## 参考文献

- Harten, A., Lax, P. D., & van Leer, B. (1983). On upstream differencing and Godunov-type schemes for hyperbolic conservation laws. *SIAM Review*, 25(1), 35–61.
- Roe, P. L. (1981). Approximate Riemann solvers, parameter vectors, and difference schemes. *Journal of Computational Physics*, 43(2), 357–372.
- Harten, A., & Hyman, J. M. (1983). Self-adjusting grid methods for one-dimensional hyperbolic conservation laws. *Journal of Computational Physics*, 50(2), 235–269.
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics* (3rd ed.). Springer.
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
- Steger, J. L., & Warming, R. F. (1981). Flux vector splitting of the inviscid gasdynamic equations with application to finite-difference methods. *Journal of Computational Physics*, 40(2), 263–293.
- van Leer, B. (1982). Flux-vector splitting for the Euler equations. *Proceedings of the 8th International Conference on Numerical Methods in Fluid Dynamics* (pp. 507–512). Springer. Also ICASE Report 82-30, NASA Langley Research Center.
- Liou, M. S., & Steffen, C. J. (1993). A new flux splitting scheme (AUSM). *Journal of Computational Physics*, 107(1), 23–39.
