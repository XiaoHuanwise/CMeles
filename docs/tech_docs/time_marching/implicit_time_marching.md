# 隐式时间推进

## 概述

隐式时间推进方法将下一时刻的解表达为未知量的函数，每时间步需求解非线性方程组。对于刚性问题和高度非均匀网格，隐式方法可采用远大于显式 CFL 限制的时间步长，是实现高效长时间积分的核心技术。

CMeles 中的隐式时间推进以 DITR（Direct Integration with Temporal Reconstruction，时间重构直接积分）方法为核心框架，通过时间重构多项式插值逼近积分点上的状态量，实现高阶隐式时间推进。

> **CMeles 选型**：DITR 方法重构的是状态向量 $\boldsymbol{u}_{\text{f}}(t)$ 而非残差 $\mathcal{R}(t)$，可利用更多信息进行时间重构，相比传统 Adams-Moulton 多步法更紧凑、更灵活，且在适当设计下具有更好的稳定性。

## 一、DITR 方法理论基础

### 1.1 问题描述

经过 DG 空间离散，可压缩 NS 方程化为半离散 ODE：

$$
\boxed{\frac{\mathrm{d}\boldsymbol{u}_{\text{f}}}{\mathrm{d}t} = \mathcal{R}(t, \boldsymbol{u}_{\text{f}})}
$$

其中 $\boldsymbol{u}_{\text{f}} \in \mathbb{R}^{N}$ 为展平的模态系数向量，$t \in [0, \infty]$。直接积分得：

$$
\boldsymbol{u}_{\text{f}}^{n+1} = \boldsymbol{u}_{\text{f}}^{n} + \int_{t^n}^{t^{n+1}} \mathcal{R}(t, \boldsymbol{u}_{\text{f}}) \,\mathrm{d}t
$$

### 1.2 三点插值求积

在区间 $[t^n, t^{n+1}]$ 内，引入三点多项式插值求积公式：

$$
\frac{\boldsymbol{u}_{\text{f}}^{n+1} - \boldsymbol{u}_{\text{f}}^{n}}{\Delta t^n} \approx b_1 \mathcal{R}(t^n, \boldsymbol{u}_{\text{f}}(t^n)) + b_2 \mathcal{R}(t^{n+c_2}, \boldsymbol{u}_{\text{f}}(t^{n+c_2})) + b_3 \mathcal{R}(t^{n+1}, \boldsymbol{u}_{\text{f}}(t^{n+1}))
$$

其中 $t^{n+c_2} = t^n + c_2 \Delta t^n$，$c_2 \in (0, 1)$，$\Delta t^n = t^{n+1} - t^n$。求积权重由二次多项式精确积分条件导出：

$$
\begin{aligned}
b_1 &= \frac{1}{2} - \frac{1}{6c_2} \\[4pt]
b_2 &= \frac{1}{6c_2(1 - c_2)} \\[4pt]
b_3 &= \frac{1}{2} - \frac{1}{6(1 - c_2)}
\end{aligned}
$$

> **精度**：该求积公式具有 2 次代数精度。当 $c_2 = 1/2$ 时退化为三点 Gauss-Lobatto 公式，精度提升至 3 次。

### 1.3 时间重构

式中的 $\boldsymbol{u}_{\text{f}}(t^{n+c_2})$ 为未知量，需要由时间重构（temporal reconstruction）获得。时间重构利用已知时刻的 $\boldsymbol{u}_{\text{f}}$ 和 $\mathcal{R}$ 构造 $\boldsymbol{u}_{\text{f}}(t)$ 的多项式插值：

$$
\begin{aligned}
\boldsymbol{u}_{\text{f}}(t) \approx \; &A_0^n(t)\,\boldsymbol{u}_{\text{f}}^{n-1} + A_1^n(t)\,\boldsymbol{u}_{\text{f}}^{n} + A_2^n(t)\,\boldsymbol{u}_{\text{f}}^{n+1} \\
&+ \Delta t^n D_0^n(t)\,\mathcal{R}^{n-1} + \Delta t^n D_1^n(t)\,\mathcal{R}^{n} + \Delta t^n D_2^n(t)\,\mathcal{R}^{n+1}
\end{aligned}
$$

其中 $A_i^n(t), D_i^n(t),\, i = 0,1,2$ 为多项式基函数。由于求积精度限制，重构多项式最多只需 3 次（对应最多 4 个条件）。

> **与 Adams-Moulton 的对比**：Adams-Moulton 方法同样利用直接积分，但它重构的是 $\mathcal{R}(t)$ 而非 $\boldsymbol{u}_{\text{f}}(t)$。高阶 Adams-Moulton 方法的稳定区域有限，稳定性较差。DITR 方法重构 $\boldsymbol{u}_{\text{f}}$ 可利用更多信息，更加紧凑灵活，能同时获得高阶精度和更好的稳定性。

### 1.4 精度阶数

对光滑问题，设求积公式具有 $m$ 次代数精度、多项式插值次数为 $l$，则 DITR 方法的精度阶数为：

| 量           | 阶数                                        |
| :----------- | :------------------------------------------ |
| 理论精度阶数 | $\min(m, l) + 1$                            |
| 局部截断误差 | $O((\Delta t)^{m+2}) + O((\Delta t)^{l+2})$ |
| 全局截断误差 | $O((\Delta t)^{m+1}) + O((\Delta t)^{l+1})$ |

---

## 二、DITR U2R2 方法

### 2.1 重构方案

选取 $\boldsymbol{u}_{\text{f}}^{n}, \boldsymbol{u}_{\text{f}}^{n+1}, \mathcal{R}^{n}, \mathcal{R}^{n+1}$ 四个条件进行三次 Hermite 插值（U2R2 = 2 个 $\boldsymbol{u}_{\text{f}}$ + 2 个 $\mathcal{R}$）：

$$
\boldsymbol{u}_{\text{f}}^{n+c_2} = a_{1,\text{U2R2}}\,\boldsymbol{u}_{\text{f}}^{n} + a_{2,\text{U2R2}}\,\boldsymbol{u}_{\text{f}}^{n+1} + \Delta t^n d_{1,\text{U2R2}}\,\mathcal{R}^{n} + \Delta t^n d_{2,\text{U2R2}}\,\mathcal{R}^{n+1}
$$

### 2.2 插值系数

$$
\begin{aligned}
a_{1,\text{U2R2}} &= 1 - (3c_2^2 - 2c_2^3) \\
a_{2,\text{U2R2}} &= 3c_2^2 - 2c_2^3 \\
d_{1,\text{U2R2}} &= c_2 - 2c_2^2 + c_2^3 \\
d_{2,\text{U2R2}} &= -c_2^2 + c_2^3
\end{aligned}
$$

### 2.3 精度与稳定性

| 参数       | 条件                           |
| :--------- | :----------------------------- |
| 精度阶数   | $c_2 = 1/2$ 时 4 阶，否则 3 阶 |
| 阶段值精度 | $c_2 = 1/2$ 时 3 次            |
| A-stable   | $c_2 \in [1/2, 1)$             |
| 刚性衰减   | $c_2 > 1/2$ 时更优             |

> **备注**：$c_2 = 1/2$ 时达到最高精度但倾向于保持刚性模态；$c_2 > 1/2$ 时刚性模态衰减更快，适合刚性较强的流动问题。

### 2.4 物理时间残差

U2R2 方法的物理时间残差（Butcher 形式）为：

$$
\begin{aligned}
\mathcal{G}^{n+c_2} &= \frac{a_{1,\text{U2R2}}\,\boldsymbol{u}_{\text{f}}^{n} + a_{2,\text{U2R2}}\,\boldsymbol{u}_{\text{f}}^{n+1} - \boldsymbol{u}_{\text{f}}^{n+c_2}}{\Delta t^n} + d_{1,\text{U2R2}}\,\mathcal{R}^{n} + d_{2,\text{U2R2}}\,\mathcal{R}^{n+1} \\[6pt]
\mathcal{G}^{n+1} &= \frac{\boldsymbol{u}_{\text{f}}^{n} - \boldsymbol{u}_{\text{f}}^{n+1}}{\Delta t^n} + b_1 \mathcal{R}^{n} + b_2 \mathcal{R}^{n+c_2} + b_3 \mathcal{R}^{n+1}
\end{aligned}
$$

### 2.5 OCCA 实现要点

DITR U2R2 每物理时间步需求解两个耦合的非线性系统（$n+c_2$ 和 $n+1$），通过双时间步法迭代。物理解 `uNew` 为张量 `[u_{n+c2}, u_{n+1}]`。预处理矩阵 $\mathbf{P}$ 的解耦形式为：

$$
\mathbf{P} = \begin{bmatrix}
\mathbf{I} & \beta\mathbf{I} \\
0 & \mathbf{I}
\end{bmatrix}
$$

其中 $\beta$ 控制两个阶段的耦合强度，$\beta = 1$ 时傅里叶分析表明稳定性最优。

---

## 三、DITR U2R1 方法

### 3.1 重构方案

选取 $\boldsymbol{u}_{\text{f}}^{n}, \boldsymbol{u}_{\text{f}}^{n+1}, \mathcal{R}^{n+1}$ 三个条件进行二次插值（U2R1 = 2 个 $\boldsymbol{u}_{\text{f}}$ + 1 个 $\mathcal{R}$），减少了对 $\mathcal{R}^n$ 的依赖：

$$
\boldsymbol{u}_{\text{f}}^{n+c_2} = a_{1,\text{U2R1}}\,\boldsymbol{u}_{\text{f}}^{n} + a_{2,\text{U2R1}}\,\boldsymbol{u}_{\text{f}}^{n+1} + \Delta t^n d_{2,\text{U2R1}}\,\mathcal{R}^{n+1}
$$

### 3.2 插值系数

$$
\begin{aligned}
a_{1,\text{U2R1}} &= 1 - (2c_2 - c_2^2) \\
a_{2,\text{U2R1}} &= 2c_2 - c_2^2 \\
d_{2,\text{U2R1}} &= c_2^2 - c_2
\end{aligned}
$$

### 3.3 精度与稳定性

| 参数     | 条件                       |
| :------- | :------------------------- |
| 精度阶数 | 3 阶（任意 $c_2$）         |
| 稳定性   | **L-stable**（任意 $c_2$） |

> **L-stable** 意味着 $z \to -\infty$ 时 $R(z) \to 0$，刚性模态会在一时间步内迅速衰减。对于具有宽特征值谱的 Navier-Stokes 方程，L 稳定性是理想性质。

### 3.4 物理时间残差

U2R1 方法的物理时间残差为：

$$
\begin{aligned}
\mathcal{G}^{n+c_2} &= \frac{a_{1,\text{U2R1}}\,\boldsymbol{u}_{\text{f}}^{n} + a_{2,\text{U2R1}}\,\boldsymbol{u}_{\text{f}}^{n+1} - \boldsymbol{u}_{\text{f}}^{n+c_2}}{\Delta t^n} + d_{2,\text{U2R1}}\,\mathcal{R}^{n+1} \\[6pt]
\mathcal{G}^{n+1} &= \frac{\boldsymbol{u}_{\text{f}}^{n} - \boldsymbol{u}_{\text{f}}^{n+1}}{\Delta t^n} + b_1 \mathcal{R}^{n} + b_2 \mathcal{R}^{n+c_2} + b_3 \mathcal{R}^{n+1}
\end{aligned}
$$

---

## 四、DITR U3R1 方法

### 4.1 重构方案

引入上一时间步的信息 $\boldsymbol{u}_{\text{f}}^{n-1}$，选取 $\boldsymbol{u}_{\text{f}}^{n-1}, \boldsymbol{u}_{\text{f}}^{n}, \boldsymbol{u}_{\text{f}}^{n+1}, \mathcal{R}^{n+1}$ 四个条件（U3R1 = 3 个 $\boldsymbol{u}_{\text{f}}$ + 1 个 $\mathcal{R}$），允许可变时间步长：

$$
\boldsymbol{u}_{\text{f}}^{n+c_2} = a_{0,\text{U3R1}}\,\boldsymbol{u}_{\text{f}}^{n-1} + a_{1,\text{U3R1}}\,\boldsymbol{u}_{\text{f}}^{n} + a_{2,\text{U3R1}}\,\boldsymbol{u}_{\text{f}}^{n+1} + \Delta t^n d_{2,\text{U3R1}}\,\mathcal{R}^{n+1}
$$

### 4.2 插值系数（$\Theta = \Delta t^{n-1}/\Delta t^n$）

$$
\begin{aligned}
a_{0,\text{U3R1}} &= -\frac{c_2 (c_2 - 1)^2}{\Theta (\Theta + 1)^2} \\[6pt]
a_{1,\text{U3R1}} &= \frac{(\Theta + c_2)(c_2 - 1)^2}{\Theta} \\[6pt]
a_{2,\text{U3R1}} &= \frac{c_2\left(-\Theta^2 c_2 + 2\Theta^2 - \Theta c_2^2 + 3\Theta - 2c_2^2 + 3c_2\right)}{(\Theta + 1)^2} \\[6pt]
d_{2,\text{U3R1}} &= \frac{c_2(\Theta + c_2)(c_2 - 1)}{\Theta + 1}
\end{aligned}
$$

### 4.3 精度与稳定性

| 参数     | 条件                                |
| :------- | :---------------------------------- |
| 精度阶数 | $\Theta = 1$ 且 $c_2 = 1/2$ 时 4 阶 |
| 稳定性   | **L-stable**                        |

> **推荐**：DITR U3R1 是 CMeles 中**唯一同时具有 4 阶精度和 L 稳定性的隐式格式**，明显优于基于简单后向差分的隐式方案（BDF2 仅有 L 稳定但仅 2 阶，三阶及更高阶 BDF 方案非无条件稳定）。

### 4.4 物理时间残差

U3R1 方法的物理时间残差为：

$$
\begin{aligned}
\mathcal{G}^{n+c_2} &= \frac{a_{0,\text{U3R1}}\,\boldsymbol{u}_{\text{f}}^{n-1} + a_{1,\text{U3R1}}\,\boldsymbol{u}_{\text{f}}^{n} + a_{2,\text{U3R1}}\,\boldsymbol{u}_{\text{f}}^{n+1} - \boldsymbol{u}_{\text{f}}^{n+c_2}}{\Delta t^n} + d_{2,\text{U3R1}}\,\mathcal{R}^{n+1} \\[6pt]
\mathcal{G}^{n+1} &= \frac{\boldsymbol{u}_{\text{f}}^{n} - \boldsymbol{u}_{\text{f}}^{n+1}}{\Delta t^n} + b_1 \mathcal{R}^{n} + b_2 \mathcal{R}^{n+c_2} + b_3 \mathcal{R}^{n+1}
\end{aligned}
$$

注意 U3R1 的 $\mathcal{G}^{n+c_2}$ 包含上一时间步 $\boldsymbol{u}_{\text{f}}^{n-1}$ 的贡献（系数 $a_{0,\text{U3R1}}$），而 $\mathcal{G}^{n+1}$ 在所有 DITR 格式中保持统一形式。

---

## 五、后向欧拉法

### 5.1 算法描述

后向欧拉法是一阶隐式时间推进方法，作为 CMeles 中最简单的隐式格式，主要用于定常问题求解和教学对比：

$$
\boldsymbol{u}_{\text{f}}^{n+1} = \boldsymbol{u}_{\text{f}}^{n} + \Delta t\,\mathcal{R}(t^{n+1}, \boldsymbol{u}_{\text{f}}^{n+1})
$$

物理时间残差：

$$
\mathcal{G}^{n+1} = \frac{\boldsymbol{u}_{\text{f}}^{n} - \boldsymbol{u}_{\text{f}}^{n+1}}{\Delta t^n} + \mathcal{R}^{n+1}
$$

### 5.2 数值性质

| 参数     | 条件           |
| :------- | :------------- |
| 精度阶数 | 1 阶           |
| 稳定性   | **L-stable**   |
| 每步 RHS | 与迭代次数有关 |

> **备注**：后向欧拉法虽然精度低，但其无条件稳定和强数值耗散特性适合定常问题的快速收敛（可与当地时间步长配合）。

---

## 六、DITR 方法汇总

| 方法       | 重构条件                                                                          | 精度阶数（典型） |          稳定性           | 特点                      |
| :--------- | :-------------------------------------------------------------------------------- | :--------------: | :-----------------------: | :------------------------ |
| 后向欧拉法 | —                                                                                 |        1         |         L-stable          | 最简单，定常收敛          |
| U2R1       | $\boldsymbol{u}^n, \boldsymbol{u}^{n+1}, \mathcal{R}^{n+1}$                       |        3         |         L-stable          | 三阶 + L 稳定，性价比高   |
| U2R2       | $\boldsymbol{u}^n, \boldsymbol{u}^{n+1}, \mathcal{R}^n, \mathcal{R}^{n+1}$        |      3 / 4       | A-stable ($c_2 \geq 1/2$) | 最高 4 阶 A 稳定          |
| U3R1       | $\boldsymbol{u}^{n-1}, \boldsymbol{u}^n, \boldsymbol{u}^{n+1}, \mathcal{R}^{n+1}$ |        4         |         L-stable          | **四阶 + L 稳定（首选）** |

---

## 七、双时间步法

### 7.1 基本原理

DITR 隐式格式每个物理时间步需求解非线性方程组。CMeles 采用双时间步法（Dual Time Stepping），引入伪时间 $\tau$，将非线性求根问题转化为伪时间层上的稳态问题：

$$
\frac{\mathrm{d}\boldsymbol{u}_{\text{f}}}{\mathrm{d}\tau} = \mathcal{F}(\boldsymbol{u}_{\text{f}}) = \mathbf{P}\,\mathcal{G}(\boldsymbol{u}_{\text{f}})
$$

当 $\tau \to \infty$ 时 $\mathcal{F} \to 0$（即 $\mathcal{G} \to 0$），物理时间步的隐式方程得以满足。

### 7.2 预处理矩阵

对于 U2R2/U2R1/U3R1，两个伪时间层（$n+c_2$ 和 $n+1$）耦合为 $2N \times 2N$ 系统：

$$
\begin{bmatrix}
\dfrac{\mathrm{d}\boldsymbol{u}_{\text{f}}^{n+c_2}}{\mathrm{d}\tau} \\[8pt]
\dfrac{\mathrm{d}\boldsymbol{u}_{\text{f}}^{n+1}}{\mathrm{d}\tau}
\end{bmatrix} = \begin{bmatrix}
\mathcal{F}^{n+c_2} \\[4pt]
\mathcal{F}^{n+1}
\end{bmatrix} = \mathbf{P} \begin{bmatrix}
\mathcal{G}^{n+c_2} \\[4pt]
\mathcal{G}^{n+1}
\end{bmatrix}
$$

采用简化预处理矩阵（DITR 原文验证已足够）：

$$
\mathbf{P} = \begin{bmatrix}
\mathbf{I} & \beta\mathbf{I} \\
0 & \mathbf{I}
\end{bmatrix}
$$

其中 $\beta$ 为耦合参数。傅里叶分析表明，对稳定性要求最严格的 U2R2 $c_2 = 1/2$，取 $\beta = 1$ 即可满足 $\max(\mathrm{Re}(\mu_1^{*}), \mathrm{Re}(\mu_2^{*})) < 0$。

> **物理含义**：$\mathbf{P}$ 的下三角结构表明 $n+1$ 方程独立于 $n+c_2$ 方程，而 $n+c_2$ 方程通过 $\beta$ 耦合 $n+1$ 信息，这为阶段解耦求解提供了理论基础。

### 7.3 全耦合 vs. 阶段解耦

| 方案                    | 优点                          | 缺点                                           |
| :---------------------- | :---------------------------- | :--------------------------------------------- |
| 全耦合（fully coupled） | 牛顿收敛快，理论最优          | 需大幅修改代码，`[2N × 2N]` 系统               |
| 阶段解耦（decoupled）   | 两个 `[N × N]` 子系统，易实现 | 收敛稍慢，但 $\Delta\tau \to 0$ 时与全耦合等价 |

CMeles 默认采用阶段解耦方案，通过 `DITRStepper::_trhsNC2()` 和 `DITRStepper::_trhsN1()` 分别求解两个解耦后的伪时间子系统。

### 7.4 收敛判断

伪时间迭代的未收敛条件（当以下两式同时成立时迭代继续）：

$$
\frac{\|\mathcal{F}\|_{\infty}}{\|\mathcal{F}_0\|_{\infty}} > \text{rtol} \quad \text{且} \quad \|\mathcal{F}\|_{\infty} > \text{atol}
$$

其中 $\mathcal{F}_0$ 取迭代前 REF_STEP（5 步）内的最大残差模（防止偶然的小初始残差导致误判收敛）。对应的 C++ 方法签名为 `DualStepper::_notConverged()`。

---

## 八、伪时间傅里叶分析

### 8.1 问题设定

考虑均匀网格上的 1D 线性对流方程 $\partial u/\partial t = -a\,\partial u/\partial x$（$a > 0$）。DG 空间离散后，各单元上的 ODE：

$$
\frac{\mathrm{d}\tilde{\boldsymbol{u}}_j}{\mathrm{d}t} = \frac{a}{\Delta x}\left(\mathbf{C}_{-1}\tilde{\boldsymbol{u}}_{j-1} + \mathbf{C}_0\tilde{\boldsymbol{u}}_j + \mathbf{C}_1\tilde{\boldsymbol{u}}_{j+1}\right)
$$

### 8.2 Fourier 模态分析

假设解为单个 Fourier 模态 $\tilde{\boldsymbol{u}}_j(t) = \hat{\boldsymbol{u}}_j(t)\,e^{\mathrm{i}\varkappa x_j}$：

$$
\mathcal{R}_j = \frac{\mathrm{d}\hat{\boldsymbol{u}}_j}{\mathrm{d}t} = \frac{a}{\Delta x}\,\mathbf{C}'\,\hat{\boldsymbol{u}}_j
$$

其中 $\mathbf{C}' = \mathbf{C}_{-1}e^{-\mathrm{i}\varkappa\Delta x} + \mathbf{C}_0 + \mathbf{C}_1e^{\mathrm{i}\varkappa\Delta x}$。稳定的空间离散要求 $\mathrm{Re}(\lambda_k) \leq 0$（$\lambda_k$ 为 $\mathbf{C}'$ 的特征值）。

### 8.3 DITR 的双时间步特征值

引入预处理矩阵 $\mathbf{P}$，双时间步系统的特征值由下式决定：

$$
\mathbf{C}^{*}_{\text{diag}} = \frac{1}{\text{CFL}_t}\begin{bmatrix}
\mathrm{A}_{11} & \mathrm{A}_{12} \\
\mathrm{A}_{21} & \mathrm{A}_{22}
\end{bmatrix} + \lambda_k\begin{bmatrix}
\mathrm{B}_{11} & \mathrm{B}_{12} \\
\mathrm{B}_{21} & \mathrm{B}_{22}
\end{bmatrix}
$$

对于 DITR + 预处理矩阵 $\mathbf{P}$：

$$
\begin{bmatrix}
\mathrm{A}_{11} & \mathrm{A}_{12} \\
\mathrm{A}_{21} & \mathrm{A}_{22}
\end{bmatrix} = \begin{bmatrix}
-1 & a_2^n - \beta \\
0 & -1
\end{bmatrix}, \quad
\begin{bmatrix}
\mathrm{B}_{11} & \mathrm{B}_{12} \\
\mathrm{B}_{21} & \mathrm{B}_{22}
\end{bmatrix} = \begin{bmatrix}
b_2\beta & b_3\beta + d_2^n \\
b_2 & b_3
\end{bmatrix}
$$

其中 $a_2^n$ 和 $d_2^n$ 为当前 DITR 格式在时间层 $n$ 的插值基函数值（各格式的具体表达式见第二至四节的插值系数），$b_2, b_3$ 为求积公式权重（见 1.2 节）。

稳定性要求 $\mathrm{Re}(\mu_i^{*}(\mu)) \leq 0,\; \forall i \in \{1,2\},\; \mathrm{Re}(\mu) \leq 0$，其中 $\mu = \text{CFL}_t \lambda_k$。

> **结论**：$\beta = 1$ 时各 DITR 格式均满足稳定性条件。详细谱分析见 DITR 原文。

---

## 九、CMeles 实现架构

### 9.1 类层次

```
ImplicitStepper (ABC)
├── BackwardEulerStepper  — 后向欧拉法
│   └── _trhsImpl()      — 物理时间残差
│
└── DITRStepper (ABC)     — DITR 基类
    ├── DITRU2R2Stepper   — U2R2
    ├── DITRU2R1Stepper   — U2R1
    └── DITRU3R1Stepper   — U3R1

DualStepper               — 双时间步调度器
├── _phyStepper:     ImplicitStepper
├── _pseudoStepper:  RungeKuttaStepper
├── step()           — 单物理时间步
└── _isDecoupled     — 全耦合/阶段解耦标志
```

### 9.2 双时间步的典型调用流程

```
1. 初始化猜测: uNew[0] = u, uNew[1] = u
2. 计算伪时间残差: f0 = P * G(uNew)
3. while not converged and cnt < maxPseudoSteps:
     a. 阶段解耦: pseudoRhsC2(·) 与 pseudoRhs(·) 分别推进
     b. 自适应 RK 步: pseudoStepper->step()
     c. 更新 RHS: R_new = phyStepper->rhs(uNew)
     d. cnt++, 检查收敛
4. out = uNew[1]  (即 u^{n+1})
```

### 9.3 伪时间步长策略

| 策略                     | 适用场景                                                   |
| :----------------------- | :--------------------------------------------------------- |
| 自适应（嵌入式 RK + PI） | 通用，GPU 友好；对于一维简单算例固定步长可能更鲁棒    |
| 固定步长（`setSdt`）     | 调试、简单问题                                             |
| 单一步长（`singleDt`）   | 全局统一伪时间步进                                         |

---

## 参考文献

1. He, S., Joo, H., Kim, C. & Yee, K. "A novel direct integration with temporal reconstruction (DITR) approach for unsteady flow simulations", *Journal of Computational Physics*, 202?.
2. E. Hairer, G. Wanner. *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. Springer, 1996.
3. F. D. Witherden, A. M. Farrington, P. E. Vincent. "PyFR: An open source framework for solving advection–diffusion type problems on streaming architectures using the flux reconstruction approach", *Computer Physics Communications*, Vol. 185, pp. 3028–3040, 2014.
4. I. Fekete, S. Conde, J. N. Shadid. "Embedded pairs for optimal explicit strong stability preserving Runge-Kutta methods", *Journal of Computational and Applied Mathematics*, Vol. 412, 2022.
5. J. S. Hesthaven & T. Warburton. *Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications*. Springer, 2008.
