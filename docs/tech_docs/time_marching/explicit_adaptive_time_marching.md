# 显式自适应时间推进

## 概述

显式自适应时间推进方法利用嵌入式 RK 对（Embedded Runge-Kutta Pairs）自动估计每步的局部截断误差，并配合 PI 步长控制器动态调整时间步长，在保证精度的前提下最大化推进效率。对于 DG 方法的双时间步格式（Dual Time Stepping），该方法在伪时间层以显式方式推进，逐自由度的局部自适应步长特别适合 GPU 上的细粒度并行。

$$
\frac{\mathrm{d}\boldsymbol{u}}{\mathrm{d}t} = \mathcal{R}(t, \boldsymbol{u})
$$

> **适用场景**：嵌入式 RK 对的自适应步长控制在双时间步格式的伪时间推进中尤为有效——无需构造单元特征尺度的表达式（这在曲面混合单元网格中难以可靠获得），且允许在同一单元的不同解点和场变量之间使用不同的局部时间步长。

## 一、显式 RK 方法的一般形式

### 1.1 Butcher 表记法

显式龙格-库塔法的一般形式为：

$$
\begin{aligned}
\boldsymbol{u}^{n+1} &= \boldsymbol{u}^{n} + \Delta t \sum_{i=1}^{s} b_i \boldsymbol{k}_i \\[4pt]
\boldsymbol{k}_i &= \mathcal{R}\left(t^n + c_i \Delta t,\; \boldsymbol{u}^n + \Delta t \sum_{j=1}^{i-1} a_{ij} \boldsymbol{k}_j\right)
\end{aligned}
$$

其中 $s$ 为阶段数（stages），$\mathbf{A} = (a_{ij})$ 为 $s \times s$ 严格下三角矩阵，$\mathbf{b} = (b_i)$ 和 $\mathbf{c} = (c_i)$ 为长度为 $s$ 的向量。系数通常以 Butcher 表给出：

$$
\begin{array}{c|c}
\mathbf{c} & \mathbf{A} \\
\hline
& \mathbf{b}^{\mathrm{T}}
\end{array}
$$

对于显式 RK 方法，满足：

$$
c_i = \sum_{j=1}^{s} a_{ij}, \quad \sum_{i=1}^{s} b_i = 1, \quad a_{ij} = 0 \;(i \leq j)
$$

### 1.2 稳定函数

显式 RK 方法的稳定函数由 Butcher 表示为：

$$
R(z) = 1 + z\,\mathbf{b}^{\mathrm{T}}(\mathbf{I} - z\mathbf{A})^{-1}\mathbf{e} = \frac{|\mathbf{I} - z\mathbf{A} + z\mathbf{e}\mathbf{b}^{\mathrm{T}}|}{|\mathbf{I} - z\mathbf{A}|}
$$

其中 $\mathbf{e}$ 为单位向量。稳定区域 $S$ 定义为：

$$
S = \left\{ z \in \mathbb{C} : |R(z)| \leq 1 \right\}
$$

线性稳定条件为 $\Delta \tau \,\lambda^{*} \subseteq S$。

---

## 二、嵌入式 RK 对

### 2.1 基本原理

嵌入式 RK 对是一族时间精度分别为 $q$ 阶和 $q-1$ 阶的 RK 格式，它们共享相同的系数矩阵 $\mathbf{A}$ 和节点向量 $\mathbf{c}$，仅权重向量 $\mathbf{b}$ 不同。其 Butcher 表扩展为：

$$
\begin{array}{c|c}
\mathbf{c} & \mathbf{A} \\
\hline
& \mathbf{b}^{\mathrm{T}} \quad (q\text{ 阶}) \\
& (\mathbf{b}^{*})^{\mathrm{T}} \quad (q-1\text{ 阶})
\end{array}
$$

共享 $\mathbf{A}$ 和 $\mathbf{c}$ 的特性使得两种精度的解可在同一组阶段值 $\boldsymbol{k}_i$ 上计算，无需额外 RHS 求值。

> **Butcher 表约定**：标准记法中扩展行给出低阶权重 $\mathbf{b}^{*}$，局部截断误差由 $\boldsymbol{\xi} = \Delta t \sum_{i=1}^{s} (b_i - b_i^{*}) \boldsymbol{k}_i$ 估计。在 CMeles 代码实现中，为方便直接存储误差向量 $\mathbf{E} = \mathbf{b} - \mathbf{b}^{*}$。**本文档中所有嵌入式 RK 对的 Butcher 表均给出三行：$\mathbf{b}$、$\mathbf{b}^{*}$、$\mathbf{E}$。** 标注 "(AI 计算)" 的值由本文档根据代码推算，非原始代码直接给出。

### 2.2 局部截断误差估计

利用两个不同精度的解，局部截断误差可由下式估计：

$$
\boldsymbol{\xi}(t + \Delta t) = \Delta t \sum_{i=1}^{s} (b_i - b_i^{*})\,\boldsymbol{k}_i
$$

对于 DG 方法，按单元 $l$、变量 $k$、自由度 $j$ 精细化：

$$
\xi_{lkj}(\tau + \Delta \tau_{lkj}) = \Delta \tau_{lkj} \sum_{i=1}^{s} (b_i - b_i^{*})\,k_{i,lkj}
$$

归一化误差定义为：

$$
\sigma_{lkj} = \frac{|\xi_{lkj}(\tau + \Delta \tau_{lkj})|}{\epsilon}
$$

其中 $\epsilon$ 为截断误差容限（tolerance）。步长控制的目标是将 $\sigma_{lkj}$ 保持在 $1$ 附近。

> **备注**：该方法称为"局部外推"（local extrapolation）——低阶解仅用于误差估计，实际推进使用高阶解。

### 2.3 CMeles 中集成的嵌入式 RK 对

CMeles 通过 `RungeKuttaStepper` 基类实现了以下嵌入式 RK 对：

#### RK3(2) — Bogacki-Shampine 对

3 阶推进，2 阶误差估计，4 阶段（FSAL，实际 3 次 RHS 求值）。

$$
\begin{array}{c|cccc}
0 & & & & \\[4pt]
\frac{1}{2} & \frac{1}{2} & & & \\[6pt]
\frac{3}{4} & 0 & \frac{3}{4} & & \\[6pt]
1 & \frac{2}{9} & \frac{1}{3} & \frac{4}{9} & \\[6pt]
\hline
\mathbf{b} & \frac{2}{9} & \frac{1}{3} & \frac{4}{9} & 0 \\[6pt]
\mathbf{b}^{*} & \frac{7}{24} & \frac{1}{4} & \frac{1}{3} & \frac{1}{8} \quad \small(\text{AI 计算})\\[6pt]
\mathbf{E} & -\frac{5}{72} & \frac{1}{12} & \frac{1}{9} & -\frac{1}{8}
\end{array}
$$

> 参考：P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas", *Appl. Math. Lett.* Vol. 2, No. 4, pp. 321-325, 1989.

#### RK5(4) — Dormand-Prince 对

5 阶推进，4 阶误差估计，7 阶段（FSAL，实际 6 次 RHS 求值，因第 7 阶段的 $\boldsymbol{k}$ 与下一时间步的第 1 阶段重合）。

$$
\begin{array}{c|ccccccc}
0 & & & & & & & \\[4pt]
\frac{1}{5} & \frac{1}{5} & & & & & & \\[6pt]
\frac{3}{10} & \frac{3}{40} & \frac{9}{40} & & & & & \\[6pt]
\frac{4}{5} & \frac{44}{45} & -\frac{56}{15} & \frac{32}{9} & & & & \\[6pt]
\frac{8}{9} & \frac{19372}{6561} & -\frac{25360}{2187} & \frac{64448}{6561} & -\frac{212}{729} & & & \\[6pt]
1 & \frac{9017}{3168} & -\frac{355}{33} & \frac{46732}{5247} & \frac{49}{176} & -\frac{5103}{18656} & & \\[6pt]
1 & \frac{35}{384} & 0 & \frac{500}{1113} & \frac{125}{192} & -\frac{2187}{6784} & \frac{11}{84} & \\[6pt]
\hline
\mathbf{b} & \frac{35}{384} & 0 & \frac{500}{1113} & \frac{125}{192} & -\frac{2187}{6784} & \frac{11}{84} & 0 \\[6pt]
\mathbf{b}^{*} & \frac{5179}{57600} & 0 & \frac{7571}{16695} & \frac{393}{640} & -\frac{92097}{339200} & \frac{187}{2100} & \frac{1}{40} \quad \small(\text{AI 计算})\\[6pt]
\mathbf{E} & \frac{71}{57600} & 0 & -\frac{71}{16695} & \frac{71}{1920} & -\frac{17253}{339200} & \frac{22}{525} & -\frac{1}{40}
\end{array}
$$

> 参考：J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta formulae", *Journal of Computational and Applied Mathematics*, Vol. 6, No. 1, pp. 19-26, 1980.

#### SSPRK 嵌入式对

以下 SSP 嵌入式对来自 Fekete, Conde & Shadid (2022)，专为双曲守恒律问题设计，兼具 SSP 性质和自适应能力。

**SSPRK2(2)1**：2 阶推进，1 阶误差估计，2 阶段。

$$
\begin{array}{c|cc}
0 & & \\
1 & 1 & \\
\hline
\mathbf{b} & \frac{1}{2} & \frac{1}{2} & 0 \\[4pt]
\mathbf{b}^{*} & 0.6940\ldots & 0.3060\ldots & 0 \\[4pt]
\mathbf{E} & -0.1940\ldots & 0.1940\ldots & 0 \quad \small(\text{AI 计算})
\end{array}
$$

> $\mathbf{b}^{*}$ 完整数据：$[0.694021459207626,\; 0.305978540792374,\; 0]$

**SSPRK3(2)1**：2 阶推进，1 阶误差估计，3 阶段。

$$
\begin{array}{c|ccc}
0 & & & \\
0.5 & \frac{1}{2} & & \\[6pt]
1 & \frac{1}{2} & \frac{1}{2} & \\[6pt]
\hline
\mathbf{b} & \frac{1}{3} & \frac{1}{3} & \frac{1}{3} & 0 \\[6pt]
\mathbf{b}^{*} & 0.6356\ldots & 0.0335\ldots & 0.3310\ldots & 0 \\[6pt]
\mathbf{E} & -0.3023\ldots & 0.2998\ldots & 0.0024\ldots & 0 \quad \small(\text{AI 计算})
\end{array}
$$

> $\mathbf{b}^{*}$ 完整数据：$[0.635564950337195,\; 0.033488381714827,\; 0.330946667947978,\; 0]$

**SSPRK3(3)2**：3 阶推进，2 阶误差估计，3 阶段。三步三阶 SSP 最优对。

$$
\begin{array}{c|ccc}
0 & & & \\
1 & 1 & & \\[6pt]
\frac{1}{2} & \frac{1}{4} & \frac{1}{4} & \\[6pt]
\hline
\mathbf{b} & \frac{1}{6} & \frac{1}{6} & \frac{2}{3} & 0 \\[6pt]
\mathbf{b}^{*} & 0.2915\ldots & 0.2915\ldots & 0.4170\ldots & 0 \\[6pt]
\mathbf{E} & -0.1248\ldots & -0.1248\ldots & 0.2497\ldots & 0 \quad \small(\text{AI 计算})
\end{array}
$$

> $\mathbf{b}^{*}$ 完整数据：$[0.291485418878409,\; 0.291485418878409,\; 0.417029162243181,\; 0]$

> **推荐**：SSPRK3(3)2 是 CMeles 中含激波流动自适应时间推进的**首选嵌入式对**——兼具三阶精度、SSP 性质和嵌入式误差估计，三步的开销可接受。

**SSPRK4(3)2**：3 阶推进，2 阶误差估计，4 阶段。

$$
\begin{array}{c|cccc}
0 & & & & \\
\frac{1}{2} & \frac{1}{2} & & & \\[6pt]
1 & \frac{1}{2} & \frac{1}{2} & & \\[6pt]
\frac{1}{2} & \frac{1}{6} & \frac{1}{6} & \frac{1}{6} & \\[6pt]
\hline
\mathbf{b} & \frac{1}{6} & \frac{1}{6} & \frac{1}{6} & \frac{1}{2} & 0 \\[6pt]
\mathbf{b}^{*} & 0.1389\ldots & 0.7223\ldots & 0.1389\ldots & 0 & 0 \\[6pt]
\mathbf{E} & 0.0278\ldots & -0.5556\ldots & 0.0278\ldots & \frac{1}{2} & 0 \quad \small(\text{AI 计算})
\end{array}
$$

> $\mathbf{b}^{*}$ 完整数据：$[0.138870252716866,\; 0.722259494566267,\; 0.138870252716866,\; 0,\; 0]$

### 2.4 方法汇总

| 嵌入式对       | 推进阶数 | 误差阶数 | 阶段数 |  SSP 性质  | 推荐场景                                                 |
| :------------- | :------: | :------: | :----: | :--------: | :------------------------------------------------------- |
| RK3(2)         |    3     |    2     |   3    |     无     | 一般非刚性 ODE；伪时间稳态与 SSPRK3(3)2 同档             |
| RK5(4)         |    5     |    4     |  7(6)  |     无     | 高精度光滑问题；伪时间稳态不适用（甜点容差超出实用范围） |
| SSPRK2(2)1     |    2     |    1     |   2    |   SSP 有   | 轻量激波问题（伪时间容差悬崖最陡）                       |
| SSPRK3(2)1     |    2     |    1     |   3    |   SSP 有   | 伪时间稳态优选（低阶估计子 + 大稳定步长）                |
| **SSPRK3(3)2** |  **3**   |  **2**   | **3**  | **SSP 有** | **CFD 优选（激波自适应；默认伪时间格式）**               |
| SSPRK4(3)2     |    3     |    2     |   4    |   SSP 有   | 伪时间稳态次优                                           |

**伪时间稳态甜点容差**（等熵涡容差扫描实验，DITR-U2R1 驱动）：甜点容差随估计子阶数 $p$ 单调变紧（$\Delta\tau \propto \mathrm{tol}^{1/(p+1)}$）——高阶估计子为压低局部误差估计反而限制步长：

| 伪时间格式 | 估计子阶数 | 甜点容差  | 备注                          |
| :--------- | :--------: | :-------: | :---------------------------- |
| SSPRK2(2)1 |     1      | $10^{-4}$ | $10^{-3}$ 挣扎、$10^{-2}$ NaN |
| SSPRK3(2)1 |     1      | $10^{-3}$ | 自适应伪时间最优              |
| SSPRK3(3)2 |     2      | $10^{-6}$ | $10^{-5}$ / $10^{-4}$ 触顶    |
| SSPRK4(3)2 |     2      | $10^{-5}$ | $10^{-4}$ 触顶；次优          |
| RK3(2)     |     2      | $10^{-6}$ | 与 SSPRK3(3)2 同档            |
| RK5(4)     |     4      |     —     | $< 10^{-8}$ 仍触顶，出局      |

> **数据适用范围**：上表及以下要点均在**全局标量控制器**（`pseudo_dt_mode = "global"`，即 RMS 误差范数 + 标量 PI 因子）下测得；局部逐自由度控制器（`pseudo_dt_mode = "local"`，见 3.4 节）的收敛路径不同，尚未做等效扫描，引用本表数据时应注明控制器模式。

要点：

- **容差悬崖陡峭**：甜点右侧一格即触顶挣扎，再松一格 NaN/失败；实用配置取甜点再紧半档。
- **综合排名（wall time）**：SSPRK3(2)1@$10^{-3}$ > SSPRK4(3)2@$10^{-5}$ > RK3(2)@$10^{-6}$ ≈ SSPRK3(3)2@$10^{-6}$ > SSPRK2(2)1@$10^{-4}$ ≫ RK5(4)。
- **最优配置** `ditr_u2r1` + `ssprk321` + `pseudo_rtol = 1e-3`（CFL = 1/5/10 时 39.0/15.6/11.9 s）。注意这是自适应伪时间格式之间的相对排名——固定步长 SSPRK3 基准（4.14~4.30 s）在该算例上仍快 2.8~9.4 倍。
- **精度零损失**：收敛运行的 L2 解逐位一致；伪容差只改收敛路径，解质量由双时间收敛准则（`rtol`）独立把关。
- **PI 控制器参数**以默认值（SAFETY 0.9 / MIN_FACTOR 0.2 / MAX_FACTOR 5 / MAX_GROWTH 10）为准：实验显示激进放大夹持（0.1 / 10 / 100）会使容差悬崖整体收紧约一个量级（甜点容差下也触顶甚至首步 NaN），不可取。

---

## 三、PI 步长控制器

### 3.1 经典 PI 控制器

CMeles 采用 Hairer & Wanner 提出的经典 PI（Proportional-Integral）步长控制器（*Solving Ordinary Differential Equations II*, Sec. IV.2）。步长更新公式为：

$$
\Delta \tau_{lkj}^{\text{new}} = \Delta \tau_{lkj} \cdot \min\!\Big(\alpha_{\max},\; \max\!\big(\alpha_{\min},\; \alpha_{\text{safe}} \cdot \sigma_{lkj}^{-\alpha / q} \cdot (\sigma_{\text{prev}})_{lkj}^{\,\beta / q}\big)\Big)
$$

其中各参数含义：

| 参数                           | 含义                                           |   典型值   |
| :----------------------------- | :--------------------------------------------- | :--------: |
| $\alpha_{\text{safe}}$         | 安全因子（SAFETY）                             |    0.9     |
| $\alpha_{\min}$                | 最小步长缩放因子（MIN_FACTOR）                 |    0.2     |
| $\alpha_{\max}$                | 最大步长缩放因子（MAX_FACTOR）                 |     5      |
| $\gamma_{\max}$                | 步长增长上限（MAX_GROWTH，相对初始步长的倍数） |     10     |
| $\alpha$                       | PI 控制器比例系数（ALPHA）                     |    0.7     |
| $\beta$                        | PI 控制器积分系数（BETA）                      |    0.4     |
| $q$                            | 推进阶数（order）                              | 视格式而定 |
| $\sigma_{lkj}$                 | 当前步归一化误差                               |     —      |
| $(\sigma_{\text{prev}})_{lkj}$ | 上一步归一化误差                               |     —      |

> **备注**：SAFETY / MIN_FACTOR / MAX_FACTOR / MAX_GROWTH 四个控制器常数可在配置文件 `[time_marching]` 中通过 `safety` / `min_factor` / `max_factor` / `max_growth` 键自定义（同一组键同时作用于显式自适应 RK 与双时间步伪时间控制器）；ALPHA 与 BETA 仍为编译期常数（`RungeKuttaStepper::kAlpha` / `kBeta`）。

> **备注**：$\beta$ 项的加入使得 PI 控制器具备了积分（memory）特性，相比纯 P 控制器能有效抑制步长振荡。当 $\beta = 0$ 时退化为纯比例控制器。

### 3.2 步长初始化

初始伪时间步长采用 Hairer & Wanner (*Solving Ordinary Differential Equations I*, Sec. II.4) 中描述的启发式方法（已在 scipy 中实现）：

$$
h_0 = \begin{cases}
10^{-6}, & \text{若 } d_0 < 10^{-5} \text{ 或 } d_1 < 10^{-5} \\[4pt]
0.01 \cdot \dfrac{d_0}{d_1}, & \text{否则}
\end{cases}
$$

其中：

$$
d_0 = \frac{\|\boldsymbol{u}_0\|_{\text{RMS}}}{\text{scale}}, \quad
d_1 = \frac{\|\mathcal{R}(\boldsymbol{u}_0)\|_{\text{RMS}}}{\text{scale}}, \quad
\text{scale} = \text{atol} + \|\boldsymbol{u}_0\|_{\text{RMS}} \cdot \text{rtol}
$$

然后试探一步，利用二阶导数信息细化：

$$
d_2 = \frac{\|\mathcal{R}(\boldsymbol{u}_0 + h_0\mathcal{R}(\boldsymbol{u}_0)) - \mathcal{R}(\boldsymbol{u}_0)\|_{\text{RMS}}}{\text{scale} \cdot h_0}, \quad
h_1 = \min\!\left(100\,h_0,\; \left(\frac{0.01}{\max(d_1, d_2)}\right)^{1/q}\right)
$$

最终 $\Delta \tau_{\text{init}} = \min(h_1, \Delta \tau_{\max})$。

### 3.3 步长接受/拒绝策略

```
若 σ_max < 1.0:
    接受步（step_accepted = true）
    若之前有拒步历史，限制 factor ≤ 1.0
否则:
    拒绝步（step_rejected = true）
    用相同公式计算缩小因子，重新尝试
```

步长更新后施加增长上限（MAX_GROWTH = 10× 初始步长）和上下界约束（min_step, max_step）。

> **备注**：以上为**全局（标量）控制器**的流程，$\sigma$ 是整个解向量的单个 RMS 范数。逐自由度的局部控制器见 3.4 节。

### 3.4 局部（逐自由度）自适应控制器

局部控制器将伪时间步长细化到单个模态系数 $u_{lkj}$（PyFR 风格的逐自由度步长）：每个自由度持有独立的 $\Delta\tau_{lkj}$、归一化误差 $\sigma_{lkj}$ 与 PI 状态 $(\sigma_{\text{prev}})_{lkj}$，RK 各阶段更新与误差估计中的标量乘积全部变为逐自由度乘积。

**逐自由度误差与缩放**：

$$
\xi_{lkj} = \Delta\tau_{lkj} \sum_{i=1}^{s} \left(b_i - b_i^{*}\right) k_{i,lkj}
$$

$$
\sigma_{lkj} = \frac{\left|\xi_{lkj}\right|}{\epsilon_{lkj}}, \quad \epsilon_{lkj} = \mathrm{atol} + \mathrm{rtol} \cdot \max\left(\left|u_{lkj}\right|, \left|u_{lkj}^{\text{new}}\right|\right)
$$

其中缩放 $\epsilon_{lkj}$ 逐自由度构造（全局控制器用的是整个向量的 RMS 范数）；$\sigma_{lkj}$ 逐元素下限 $10^{-14}$。

**接受判据与步长更新**：每个伪时间步内做一次 RK 尝试，计算所有 $\sigma_{lkj}$ 后以**全局最大值**判定接受：

$$
\max_{l,k,j} \sigma_{lkj} < 1 \quad \Rightarrow \quad \text{接受该步}
$$

逐自由度 PI 因子与标量形式相同，但按各自误差独立计算：

$$
f_{lkj} = \alpha_{\text{safe}} \cdot \sigma_{lkj}^{-\alpha/q} \cdot (\sigma_{\text{prev}})_{lkj}^{\,\beta/q}
$$

随后 $\Delta\tau_{lkj} \mathrel{*}= f_{lkj}$，并施加 $\mathrm{MAX\_GROWTH} \cdot \Delta\tau_{\text{init}}$ 增长上限与 $[\mathrm{min\_step}, \mathrm{max\_step}]$ 夹持（$\mathrm{max\_step}$ 为物理步长）。

**与全局控制器的语义差异**（移植自原型，实现有意保留）：

| 方面                        | 全局（标量）控制器                                                    | 局部（逐自由度）控制器                                                         |
| :-------------------------- | :-------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| 误差范数                    | 全向量 RMS：$\sigma = \lVert \xi \rVert_2 / (\sqrt{N}\,\bar\epsilon)$ | 逐自由度 $\sigma_{lkj}$，$\epsilon_{lkj}$ 逐元素构造                           |
| 接受判据                    | $\sigma < 1$                                                          | $\max_{lkj} \sigma_{lkj} < 1$                                                  |
| factor 夹持                 | 拒绝时夹 $\alpha_{\min}$、接受时夹 $\alpha_{\max}$（单侧）            | **每次尝试双侧夹持** $[\alpha_{\min}, \alpha_{\max}]$                          |
| 接受后限增                  | 曾拒步则 factor ≤ 1（标量）                                           | 曾拒步则逐元素 $f_{lkj} \le 1$                                                 |
| $\sigma_{\text{prev}}$ 提交 | 每步末提交                                                            | 仅提交最终（接受/强制接受）尝试的 $\sigma_{lkj}$，中间拒步不进入               |
| 初始化                      | Hairer 标量启发式                                                     | 同一标量启发式**均匀填充**整个 $\Delta\tau_{lkj}$ 数组（分化完全来自 PI 更新） |

**C++ 增补**（原型没有的鲁棒性守护）：拒绝重试模式下，当 $\min_{lkj} \Delta\tau_{lkj} \le \mathrm{min\_step}$ 且仍未接受时强制提交该次尝试——否则触底条目无法再缩小步长，重试将无限重复同一尝试（原型会死循环）。

**实现映射**（`RungeKuttaStepper` 的 `Params::localDt`，经 `[time_marching] pseudo_dt_mode = "local"` 开启，仅作用于双时间步的伪时间步进器）：

- OKL 内核（`src/time/okl/time_update.okl`）：`rkStageCombineLocalDt` / `rkWeightedSumLocalDt` / `rkFinalUpdateLocalDt`（Butcher 递推的逐 DOF dt 变体）、`rkErrorNormLocal`（逐 DOF $\sigma_{lkj}$）、`rkDtUpdateLocal`（逐 DOF PI 更新，含全部夹持）、`fillReal`（均匀初始化）。
- 设备状态：`o_dt_` / `o_sigma_` / `o_sigmaPrev_`（各 $N_{\text{dof}}$）。耦合双时间模式下 $N_{\text{dof}}$ 覆盖堆叠的两个阶段（$n{+}c_2$ 与 $n{+}1$ 行各自持有独立步长，对应原型 `pseudo_dt = empty_like(F)`）；解耦模式下每个阶段步进器各持一份长度 $N$ 的数组。
- 接受判定经 `Blas::amax` 归约 $\max \sigma_{lkj}$；$\min \Delta\tau$ 经"取负 + amax"组合实现（不新增 Blas 原语）。

---

## 四、在 DG-DITR 双时间步格式中的应用

### 4.1 伪时间层上的显式自适应推进

在 DITR 双时间步格式中，每个物理时间步内部以伪时间 $\tau$ 迭代至稳态。伪时间层上采用显式嵌入式 RK 对（如 SSPRK3(3)2）推进。`pseudo_dt_mode = "local"`（见 3.4 节）时，$N_{\text{elem}} \times N_{\text{vars}} \times N_{\text{modes}}$ 个自由度各自持有独立的伪时间步长 $\Delta \tau_{lkj}$；默认的 `"global"` 模式下所有自由度共享单一标量步长。

伪时间残差 $\mathcal{F} = \mathbf{P}\,\mathcal{G}$ 在 $\tau \to \infty$ 时趋于零，物理时间步收敛。自适应步长控制确保每个自由度的截断误差保持在容限内，同时最大化稳定步长。

### 4.2 与 P 多重网格的亲和性

嵌入式 RK 对的自适应步长特别适合与 P 多重网格（p-multigrid）结合：

- **精细局部化**：同一单元内不同模态可有不同步长，高频模态可用较小步长，低频模态可用较大步长
- **投影一致性**：步长信息可随解一同投影到低阶空间
- **CFL 缓解**：低阶表示具有更宽松的 CFL 限值，可配合更大的 $\Delta \tau$ 传播信息

### 4.3 CMeles 实现架构

CMeles 用**单个** `RungeKuttaStepper` 类实现全部六对嵌入式 RK 对（Butcher 表为运行期数据，一次上传设备、内核运行期读取），并通过 `Params::localDt` 提供两种步长控制器：

```
RungeKuttaStepper : StepperBase（单类，六对表共用）
├── Butcher 表: table_ (constexpr 表) + 设备数组 o_A_ / o_B_ / o_E_
├── RK 步:    rkStep()（标量或逐 DOF dt 内核二选一）
├── 全局控制器（默认）
│   ├── 误差: computeErrorNorm()  — RMS 范数 + 标量 scale
│   └── PI:   updateDt()          — 标量 factor
├── 局部控制器（Params::localDt，pseudo_dt_mode = "local"）
│   ├── 误差: computeErrorNormLocal() — 逐 DOF σ，amax 判接受
│   └── PI:   updateDtLocal()        — 逐 DOF factor（rkDtUpdateLocal 内核）
├── 步长初始化: selectInitialStep()（Hairer 启发式；局部模式均匀填充 o_dt_）
└── 接口: advance()（物理时间，恒为全局控制器）/ stepPseudo(allowReject)
         / stepFixed(dt) / setState() / setRhs()
```

`DualStepper` 将自适应显式伪时间步进器与隐式物理时间步进器组合（详见隐式时间推进文档）：

```
DualStepper(residual, pseudoTable, ...)
├── residual_:       DitrResidual / BackwardEulerResidual
└── pseudo_:         RungeKuttaStepper (自适应嵌入式RK，全局或局部控制器)
    └── pseudoC2_:   解耦模式的第二个伪步进器
```

对于解耦 DITR 格式，两个伪时间子系统（$n+c_2$ 和 $n+1$）各自持有独立的 `RungeKuttaStepper` 实例和独立的步长状态；局部模式下即各自独立的逐 DOF 步长数组。控制器模式由 `makeStepperFactory` 经 `[time_marching] pseudo_dt_mode` 装配（`"global"` 默认 / `"local"`），固定伪步长（`pseudo_dt > 0`）优先于两种自适应模式。

---

## 参考文献

1. E. Hairer, S. P. Nørsett, G. Wanner. *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed. Springer, 1993. (Sec. II.4 — 步长初始化)
2. E. Hairer, G. Wanner. *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. Springer, 1996. (Sec. IV.2 — PI 步长控制器)
3. P. Bogacki, L. F. Shampine. "A 3(2) Pair of Runge-Kutta Formulas", *Applied Mathematics Letters*, Vol. 2, No. 4, pp. 321–325, 1989.
4. J. R. Dormand, P. J. Prince. "A family of embedded Runge-Kutta formulae", *Journal of Computational and Applied Mathematics*, Vol. 6, No. 1, pp. 19–26, 1980.
5. I. Fekete, S. Conde, J. N. Shadid. "Embedded pairs for optimal explicit strong stability preserving Runge-Kutta methods", *Journal of Computational and Applied Mathematics*, Vol. 412, 2022.
6. F. D. Witherden, A. M. Farrington, P. E. Vincent. "PyFR: An open source framework for solving advection–diffusion type problems on streaming architectures using the flux reconstruction approach", *Computer Physics Communications*, Vol. 185, pp. 3028–3040, 2014.
