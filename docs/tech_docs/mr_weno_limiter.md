# MR-WENO 限制器

本文档描述 CMeles 中的间断捕获限制器：**多分辨率加权基本无振荡（Multi-Resolution WENO, MR-WENO）限制器**，包括其数学原理、CMeles 网格体系（平行四边形 / 一般四边形 / 三角形混合网格）下的关键修正、问题单元检测方法，以及完整的代码实现方案与验证计划。

## 概述

### 动机

DG 方法以分片 $N$ 次多项式逼近解，光滑问题达到 $N+1$ 阶精度，但解中出现强间断（激波、接触间断）时，高阶多项式不可避免地产生 Gibbs 型伪振荡，并可能导致负密度、负压力等非线性失稳。限制器的任务是在**每个时间步（每个 RK 内 stage）之后**对个别"问题单元"（troubled cell）的多项式进行修改，同时满足：

1. **光滑区保阶**：$N+1$ 阶精度不降低；
2. **间断区非振荡**：伪振荡被抑制，最坏退化到一阶（常数）；
3. **严格守恒**：单元平均不变，不破坏全局守恒；
4. **模板紧凑**：不使用超出 DG 残差计算本身的邻居信息（这对非结构网格实现与并行效率至关重要）。

### MR-WENO 的核心思想

传统 WENO 类限制器在 troubled cell 上用邻居（乃至邻居的邻居）的信息重构候选多项式，模板宽、非结构实现复杂。MR-WENO 的关键观察是：**DG 解的多项式本身就是一个按次数嵌套的候选族**——把本单元多项式逐次截断到低次空间，就免费得到一族候选，无需任何邻居重构：

$$ q_0 \subset q_1 \subset \cdots \subset q_N = u_h $$

间断的伪振荡来自高阶模态，逐级截断等价于逐级降低"分辨率"。WENO 非线性权重自动决定降到哪一级：光滑时各级权重趋于线性权重，完整恢复原 DG 多项式（精度无损）；间断时权重向低级转移，最坏退化到常数（一阶，无振荡）。全部算法中**唯一**用到邻居信息的地方，是最低阶候选的光滑度指示器（借用直接邻居的线性重构，模板与 DG 残差计算相同）。

方法源自 Zhu & Shu 的多分辨率有限体积 WENO 格式，作为 DG 限制器的三篇文献为：

| 文献 | 网格 | 框架 | 阶数 |
| :--- | :--- | :--- | :--- |
| Zhu, Qiu & Shu 2019 (JCP 109105) | 结构四边形 | RKDG | $P_1$–$P_4$ |
| Zhu, Shu & Qiu 2021 (CiCP 29:1030) | 四面体 | RKDG | $P_1$–$P_2$ |
| Lu, Zhu et al. 2023 (CiCP) | 三角形 | LDG (NS 方程) | $P_2$ |

三篇文献的限制器算法完全一致，本文档以其共同算法为准，并按 CMeles 的网格体系（四边形 + 退化四边形三角形，见[网格与几何文档](mesh_and_geometry.md)）给出适配。

### 与其他限制器方案的对比

- **minmod / TVB 斜率限制器**：在光滑极值点误伤导数导致降阶，且含问题相关人工参数；
- **经典 WENO / HWENO 限制器**（Qiu-Shu、Luo 等）：需要邻居乃至邻居的邻居做有限体积式重构，非结构网格上每个候选需解最小二乘/线性系统；非线性权在驻定流场中无法稳定，**稳态残差不能收敛到机器零**；
- **MR-WENO**：模板仅含本单元与直接邻居的线性部分；限制器可平稳退化到一阶，稳态问题的残差可收敛到机器零（Lu 2023 实测验证）——这一性质对 CMeles 的隐式/伪时间步进稳态求解方向（见[时间推进文档](time_marching.md)）尤为重要。

## 记号约定

与[控制方程与 DG 场文档](governed_equations_and_DG_field.md)、[基函数文档](basis_functions.md)保持一致，并补充本方法专用记号：

| 记号 | 含义 |
| :--- | :--- |
| $N$ | 多项式阶数（配置 `order`），$P_N$ 方法精度为 $N+1$ 阶 |
| $N_{\text{modes}}$ | 模态总数，二维 $= \frac{(N+1)(N+2)}{2}$ |
| $N_\ell$ | 次数 $\le \ell$ 的模态数，$N_\ell = \frac{(\ell+1)(\ell+2)}{2}$，$N_N = N_{\text{modes}}$ |
| $\phi_l$ | 归一化 Legendre 张量积正交基（杨辉三角排序 $l = p(p+1)/2 + j$，$p = i+j$ 为总次数） |
| $u^{(l)}$ | 第 $l$ 个模态系数（DG 自由度），存储布局 `u[elem][var][mode]` |
| $u_h \big\|_{\Omega_e}$ | 单元 $e$ 上的 DG 解多项式 $u_h = \sum_{l} u^{(l)} \phi_l$ |
| $\Omega_0$ | 目标单元（正在被限制的 troubled cell），面积 $|\Omega_0|$ |
| $q_\ell$ | $u_h$ 在次数 $\le \ell$ 多项式空间上的 $L^2$ 投影（层级候选） |
| $p_{\ell_1,\ell_2}$ | 第 $\ell_2$ **级**组合中、多项式次数为 $\ell_1$ 的候选（双下标含义见下文） |
| $\gamma_{\ell_1,\ell_2}$、$\omega_{\ell_1,\ell_2}$ | 候选 $p_{\ell_1,\ell_2}$ 的线性权重与非线性权重 |
| $\beta_{\ell_1,\ell_2}$ | 候选 $p_{\ell_1,\ell_2}$ 的光滑度指示器 |
| $\tau_{\ell_2}$ | 第 $\ell_2$ 级两候选光滑度之差的平方 |
| $\varepsilon_w$ | WENO 权重公式的防零小量（$10^{-6}$，勿与非仿射度 $\varepsilon$ 混淆） |
| $\alpha = (\alpha_1, \alpha_2)$ | 导数多重指标，$\|\alpha\| = \alpha_1 + \alpha_2$，$\partial^{\alpha} = \partial^{\|\alpha\|}/\partial x^{\alpha_1} \partial y^{\alpha_2}$ |
| $M$ | 单元质量矩阵 $M_{lm} = \iint \phi_l \phi_m \|\mathbf{J}\| \, dr \, ds$（[基函数文档](basis_functions.md)、[平行四边形假设](parallelogram_assumption.md)） |

> **注**：$\phi_0 = \tilde{P}_0(r)\,\tilde{P}_0(s) = \frac{1}{2}$（归一化 Legendre 的常数基值为 $1/\sqrt{2}$，张量积后为 $1/2$），**不是** $1$。由此，单元平均的一般公式为 $\bar{u} = \dfrac{2\,(M u^{(\cdot)})_0}{|\Omega_0|}$（$u^{(\cdot)}$ 为该单元的模态系数向量）；当且仅当质量矩阵对角（平行四边形单元）时退化为 $\bar{u} = u^{(0)}/2$。实现中**不得**假设 $u^{(0)}$ 等于（或正比于）单元平均，除非已确认仿射单元。

## 方法原理

以下以**标量**情形叙述算法，系统（Euler 方程）情形见[系统情形](#系统情形面法向特征场分解)一节。

### 第一步：层级候选多项式 $q_\ell$

对 troubled cell $\Omega_0$，定义 $q_\ell$（$\ell = 0, \ldots, N$）为 $u_h$ 在次数 $\le \ell$ 多项式空间上的 $L^2$ 投影：

$$
\int_{\Omega_0} \left( q_\ell - u_h \right) \phi_l \, \mathrm{d}\Omega = 0, \qquad l = 0, \ldots, N_\ell - 1
$$

- **仿射单元（平行四边形）快速路径**：质量矩阵对角 $M = \|\mathbf{J}\| I$（见[平行四边形假设](parallelogram_assumption.md)），$L^2$ 投影退化为**模态截断**——直接丢弃总次数 $> \ell$ 的模态系数：

  $$ q_\ell = \sum_{l < N_\ell} u^{(l)} \phi_l $$

  杨辉三角排序保证了 $l < N_\ell$ 恰好对应总次数 $p \le \ell$ 的所有模态。例如 $P_2$（$N_{\text{modes}} = 6$）：$q_0 = u^{(0)}\phi_0$（常数），$q_1$ 保留模态 $0,1,2$（线性），$q_2 = u_h$。**零计算成本。**

- **非仿射单元（三角形、一般四边形）**：质量矩阵满秩（详见[仿射性分类](#仿射性分类cmeles-网格的关键修正)一节），截断不再是 $L^2$ 投影，需按该节的方法求解小规模投影方程。

### 仿射性分类：CMeles 网格的关键修正

CMeles 的几何映射为双线性形式（[网格与几何文档](mesh_and_geometry.md)）：

$$ x = a_0 + a_1 r + a_2 s + a_3 rs, \qquad y = b_0 + b_1 r + b_2 s + b_3 rs $$

**关键代数事实**：展开 $\det \mathbf{J} = (a_1 + a_3 s)(b_2 + b_3 r) - (a_2 + a_3 r)(b_1 + b_3 s)$，两个 $rs$ 项（$a_3 b_3 rs$）恰好相消，故对**任何**双线性映射，

$$ \det \mathbf{J}(r, s) = c_0 + c_1 r + c_2 s \quad \text{（恒为线性）} $$

由此定义**逐单元非仿射度**：

$$ \varepsilon = \frac{|c_1| + |c_2|}{c_0} $$

全部单元类型落在一个由 $\varepsilon$ 刻画的谱上：

| 单元类型 | $\varepsilon$ | 质量矩阵最大非对角元（相对对角元） | 说明 |
| :--- | :---: | :---: | :--- |
| 平行四边形（含矩形） | $0$ | $0$ | $c_1 = c_2 = 0 \Leftrightarrow a_3 = b_3 = 0$，仿射映射，$M = \|\mathbf{J}\| I$ 对角 |
| 一般四边形 | $(0, 1)$ 之间 | $\approx 0.577\,\varepsilon$ | 随偏离平行四边形的程度线性增长 |
| 三角形（退化四边形 $P_4 = P_3$） | 恒为 $1$ | $\approx 0.577$ | 坍缩边（参考坐标 $s = +1$）上 $\det\mathbf{J} \equiv 0$ 强制 $c_1 = 0$、$c_2 = -c_0$，与单元形状无关 |

数值验证数据（$P_2$ 基、单元平均相对误差）：

| 单元 | $\varepsilon$ | $\max \|M_{\text{off}}/M_{\text{diag}}\|$ | $u^{(0)}/2$ 相对单元平均误差 |
| :--- | :---: | :---: | :---: |
| 平行四边形 | 0 | 0 | 0% |
| 梯形 | 0.167 | 0.096 | 2.96% |
| 一般四边形 | 0.214 | 0.098 | 2.49% |
| 直角三角形 | 1.0 | 0.577 | 29.6% |

**对限制器的影响**（非仿射单元上直接使用截断的后果）：

1. **守恒破坏**：单元平均 $\bar{u} = 2(Mu^{(\cdot)})_0 / |\Omega_0|$ 涉及质量矩阵第 0 行的所有耦合项。凸组合保持 $u^{(0)}$ 不再等价于保持 $\bar{u}$；若限制器把解退化到常数候选，单元平均的偏移恰好等于上表第 4 列的误差（三角形为 13%–30% 量级，随解场分布而定）——恰发生在激波旁最需要守恒的地方；
2. **$u^{(0)} \neq$ 平均的 $1/2$ 倍**：特征分解线性化点、KXRCF 归一化等凡用到单元平均处，必须按一般公式 $\bar{u} = 2(Mu^{(\cdot)})_0/|\Omega_0|$ 计算（或直接求积）；
3. **邻居线性多项式的物理梯度不再是常向量**（见光滑度指示器一节的 $\beta_{0,1}$）。

**修正方案（推荐）：troubled cell 内即时 $L^2$ 投影。** 将 $q_\ell$ 改为真投影：求 $c^{(\ell)} \in \mathbb{R}^{N_\ell}$ 满足

$$ M_{\ell\ell}\, c^{(\ell)} = (M u^{(\cdot)})_\ell $$

其中 $M_{\ell\ell}$ 是 $M$ 的前 $N_\ell \times N_\ell$ 分块，$(M u^{(\cdot)})_\ell$ 是 $Mu^{(\cdot)}$ 的前 $N_\ell$ 个分量。两者均可用现有数据在单元内直接求积获得，**无需存储质量矩阵本身**：

$$ (M u^{(\cdot)})_l = \int_{\Omega_0} u_h \phi_l \, \mathrm{d}\Omega = \sum_q \lambda_{WJ}[q]\, \phi_l(q) \left( \sum_m u^{(m)} \phi_m(q) \right), \qquad
M_{\ell\ell} = \mathbf{V}_{\ell}^{T} \operatorname{diag}(\Lambda_{wJ})\, \mathbf{V}_{\ell} $$

$\mathbf{V}_\ell$ 为二维 Vandermonde 矩阵的前 $N_\ell$ 列（[基函数文档 2.3 节](basis_functions.md#23-二维-vandermonde-矩阵)），$\Lambda_{wJ}$ 由几何模块逐积分点提供。每级一个 $N_\ell \times N_\ell$ 稠密 Cholesky（$P_2$ 只有 $3 \times 3$；$q_N = u_h$ 本身无需投影），只在 troubled cell 上执行，代价可忽略。

投影路径恢复论文全部性质：$q_0$ 成为**真单元平均常数**，凸组合链严格保平均；光滑极限链条仍精确还原 $u_h$。**仿射单元上 $M_{\ell\ell} = \|\mathbf{J}\| I$，投影自动退化为截断**——同一套代码按逐单元仿射标志分支即可。

> **实现要求**：仿射判据为逐单元的 $c_1 = c_2 = 0$（等价于现有平行四边形条件 $P_1 + P_3 = P_2 + P_4$，或逐点 $\det \mathbf{J}$ 与首点值一致）。注意 `MeshGeometry::isOrthogonal()` 是**全局**标志（网格中任一单元非平行四边形即为 false，此时全体单元存满阵 $M^{-1}$），限制器**不得**依赖它，需自备逐单元标志。

### 第二步：细节多项式与 telescoping 递推

双下标记号 $p_{\ell_1,\ell_2}$：第二个下标 $\ell_2$ 是**层级**（第几轮"两两对决"），第一个下标 $\ell_1$ 是该候选的多项式次数。第 $\ell_2$ 级比较一个次数 $\ell_2 - 1$ 的候选（低阶候选）与一个次数 $\ell_2$ 的候选（**细节多项式**，detail）。

递推从 $p_{0,1} = q_0$ 开始，对 $\ell = 1, \ldots, N$：

$$ \boxed{\; p_{\ell,\ell} = \frac{1}{\gamma_{\ell,\ell}} \left( q_\ell - \gamma_{\ell-1,\ell}\, p_{\ell-1,\ell} \right) \;}
\qquad \gamma_{\ell-1,\ell} + \gamma_{\ell,\ell} = 1, \quad \gamma_{\ell,\ell} \neq 0 $$

细节多项式的构造保证**无条件恒等式**（对任意正的 $\gamma$、任意的 $p_{\ell-1,\ell}$——哪怕它已被上一级非线性修改——代数上直接成立）：

$$ \gamma_{\ell-1,\ell}\, p_{\ell-1,\ell} + \gamma_{\ell,\ell}\, p_{\ell,\ell} = q_\ell $$

这就是 **telescoping（逐级衔接）结构**：线性权重被吸收进细节多项式的定义，因此 $\gamma$ 可以是**任意正数**（传统 WENO 的线性权重由模板几何唯一确定）。推荐取 $\gamma_{\ell_2-1,\ell_2} = 0.01$（低阶候选）、$\gamma_{\ell_2,\ell_2} = 0.99$（高阶候选）——偏向高阶，光滑区恢复更快。

每级做 WENO 组合（非线性权重见[第四步](#第四步weno-z-非线性权重)），$\ell < N$ 时其输出作为下一级的低阶候选：

$$ p_{\ell,\ell+1} = \omega_{\ell,\ell}\, p_{\ell,\ell} + \omega_{\ell-1,\ell}\, p_{\ell-1,\ell} $$

$\ell = N$ 时的组合即最终重构多项式，覆写 troubled cell 的模态系数。**注意**：每级的细节多项式建立在上一级**非线性结果** $p_{\ell-1,\ell}$ 之上（不是干净的 $q_{\ell-1}$），上一级的修改被本级自动吸收。

### 第三步：光滑度指示器 $\beta$

对每个候选 $p_{\ell_1,\ell_2}$，采用 Jiang-Shu 型 Dirichlet 能量指示器（积分在 troubled cell 上）：

$$ \beta_{\ell_1,\ell_2} = \sum_{\|\alpha\|=1}^{\ell_1} \; \int_{\Omega_0} |\Omega_0|^{\|\alpha\|-1} \left( \partial^{\alpha} p_{\ell_1,\ell_2} \right)^2 \mathrm{d}\Omega $$

- 求和遍历候选多项式**自身次数以内**的所有阶、所有方向的导数（$\|\alpha\| = 1$：$\partial_x, \partial_y$；$\|\alpha\| = 2$：$\partial_{xx}, \partial_{xy}, \partial_{yy}$；以此类推）。文献中亦有将上界写作层级 $\ell_2$ 的记法，两者数值等价（低次候选的高阶导数恒为零）；
- 尺度因子 $|\Omega_0|^{\|\alpha\|-1}$（二维为面积的幂）抵消各阶导数平方积分固有的网格尺度，使各项与解在单元上的变化幅值同量级；光滑场上 $\beta = O(h^2)$（一阶导数项主导）；
- **含义**：$\beta$ 是"振荡能量计"——振荡剧烈的多项式导数能量大，$\beta$ 大；平顺多项式 $\beta$ 小。

**离散实现**：物理导数经链式法则由参考导数与逆雅可比给出（列向量约定，$\text{Jinv}_{11} = \partial r/\partial x$ 等，见[网格与几何文档 2.4 节](mesh_and_geometry.md#24-雅可比矩阵与行列式)）。**阶数 $\le 2$ 的链式法则在任意双线性单元上精确**：二阶修正项（$r_{xx} p_r + s_{xx} p_s$ 等）由 $\partial_r J^{-1} = -J^{-1} (\partial_r J) J^{-1}$ 与常值双线性 Hessian $\partial_r J = [[0, a_3], [0, b_3]]$、$\partial_s J = [[a_3, 0], [b_3, 0]]$ 闭式给出（仿射单元 $a_3 = b_3 = 0$ 时恒为零）；三阶及以上（仅 $P_3/P_4$ 非仿射单元用到）仍为常雅可比多重展开近似。：

$$
\partial_x p = \text{Jinv}_{11}\, \partial_r p + \text{Jinv}_{21}\, \partial_s p, \qquad
\partial_y p = \text{Jinv}_{12}\, \partial_r p + \text{Jinv}_{22}\, \partial_s p
$$

在体积分点上用 $\Lambda_{wJ}$ 加权求和即可（与体积分核函数相同的收缩模式，导数矩阵为 $\mathrm{dV2D}_r$、$\mathrm{dV2D}_s$）。被积函数为多项式，次数 $\le 2N - 2\|\alpha\| + 1$（$\det\mathbf{J}$ 线性贡献 $+1$），张量 Gauss 求积 $N_q \ge N+1$ 即可精确。

**最低阶候选的特例 $\beta_{0,1}$**：常数候选 $q_0$ 的所有导数为零，$\beta_{0,1} \equiv 0$ 会使权重机制失真（系统性偏置常数候选）。补丁：取 $\Omega_0$ 的**直接面邻居**（三角形 3 个、四边形 4 个），对每个邻居 $i$ 取其 $P_1$ 的 $L^2$ 投影 $q_i$（仿射邻居即其模态 $0,1,2$ 的截断），在**目标单元**上积分其物理梯度能量，取最小值：

$$ \beta_{0,1} = \min_i \zeta_i, \qquad \zeta_i = \int_{\Omega_0} \left[ \left(\partial_x q_i\right)^2 + \left(\partial_y q_i\right)^2 \right] \mathrm{d}\Omega $$

取 min 的直觉：只要有一个邻居视角认为流场光滑，就允许权重升阶（保守方向是取更多 troubled 判定）。这是**全部算法唯一用到邻居信息的地方，且只用直接邻居的线性部分**。

计算细节：

- **仿射邻居**（平行四边形）：$\phi_1 = \tilde{P}_1(r)\tilde{P}_0(s)$、$\phi_2 = \tilde{P}_0(r)\tilde{P}_1(s)$ 的物理梯度为**常向量**（$\nabla_x \phi_1 = \mathbf{J}^{-T} (\sqrt{3}/2, 0)^T$，$\nabla_x \phi_2 = \mathbf{J}^{-T} (0, \sqrt{3}/2)^T$，$\mathbf{J}^{-1}$ 逐点常值），故 $\zeta_i = \left| u^{(1)}_i \nabla_x \phi_1 + u^{(2)}_i \nabla_x \phi_2 \right|^2 |\Omega_0|$，闭式计算、无需求积；
- **非仿射邻居**（三角形、一般四边形）：$\mathbf{J}^{-T}$ 随位置变化，梯度非常向量，需在目标单元积分点上求值。由于几何静态，建议在预处理阶段为每个（单元， 邻居）对预计算**邻居梯度表**——目标单元各体积分点处邻居基函数 $\phi_1, \phi_2$ 的物理梯度（每点 4 个实数）——运行期纯查表。首版实现亦可用邻居截断模态（而非真 $P_1$ 投影）近似，$\beta_{0,1}$ 仅是指示器输入，其轻微近似不影响守恒与精度恢复性质。

### 第四步：WENO-Z 非线性权重

第 $\ell_2$ 级两候选 $\left( p_{\ell_2-1,\ell_2},\; p_{\ell_2,\ell_2} \right)$ 的权重采用 WENO-Z 配方：

$$
\tau_{\ell_2} = \left( \beta_{\ell_2,\ell_2} - \beta_{\ell_2-1,\ell_2} \right)^2, \qquad
\bar{\omega}_{\ell_1,\ell_2} = \gamma_{\ell_1,\ell_2} \left( 1 + \frac{\tau_{\ell_2}}{\varepsilon_w + \beta_{\ell_1,\ell_2}} \right), \qquad
\omega_{\ell_1,\ell_2} = \frac{\bar{\omega}_{\ell_1,\ell_2}}{\bar{\omega}_{\ell_2-1,\ell_2} + \bar{\omega}_{\ell_2,\ell_2}}
$$

其中 $\varepsilon_w = 10^{-6}$。两极限行为（实现验证时的关键检查点）：

- **光滑区**：$\beta_{\ell_2,\ell_2} \approx \beta_{\ell_2-1,\ell_2}$，$\tau \to 0$，$\omega \to \gamma$，凸组合退回线性组合——由 telescoping 恒等式沿链条归纳，最终输出精确还原 $q_N = u_h$（对解的修改是 $O(h^2)$ 高阶小量，$\omega - \gamma = O(\tau/(\varepsilon_w + \beta)) = O(h^2)$）；
- **间断区**：高阶候选 $\beta$ 巨大 → $\tau$ 巨大 → $\beta$ 小的候选 $\bar{\omega} \sim \gamma\, \tau / \varepsilon_w$ 被放大约 $10^4$ 倍，权重几乎全部转移给低阶候选——自动降级，逐级退化最坏到常数（一阶）。

### 标量算法总览

以 $P_2$ 为例的完整流程（$N$ 阶方法为 $\ell = 1, \ldots, N$ 的循环）：

```
输入: troubled cell Ω0 的模态系数 u^(0..5)，邻居的线性模态

1. 层级候选（仿射: 截断; 非仿射: 逐级 L2 投影）
     q0 (常数=真单元平均), q1 (线性), q2 = u_h
2. 第 1 级: p_low = q0
     p_11 = (q1 − γ01 · p_low) / γ11            # 模态系数 axpy
     β_low  = min_i ζ_i                          # 邻居梯度能量（第 0 级特例）
     β_high = Σ_{|α|=1} ∫ |Ω0|^{0} (∂^α p_11)²  # 积分点求值 + ΛwJ 收缩
     ω ← WENO-Z(γ01, γ11, β_low, β_high)
     p_low = ω_low · p_low + ω_high · p_11       # 第 1 级输出
3. 第 2 级:
     p_22 = (q2 − γ12 · p_low) / γ22
     β_low  = Σ_{|α|=1} ∫ (∂^α p_low)²          # 注意对非线性结果算 β
     β_high = Σ_{|α|=1..2} ∫ |Ω0|^{|α|−1} (∂^α p_22)²
     ω ← WENO-Z(γ12, γ22, β_low, β_high)
     u_new = ω_low · p_low + ω_high · p_22       # 覆写模态系数
```

主体是纯模态空间的 axpy 运算；需要到积分点上的只有 $\beta$（Vandermonde 表 + 逐点 $\mathbf{J}^{-1}$ 链式求导）与 $\beta_{0,1}$ 的邻居梯度。输出仍是 $N$ 次多项式，自由度向量尺寸不变。

### 系统情形：面法向特征场分解

对 Euler 方程组（$\boldsymbol{u} = (\rho, \rho u, \rho v, E)^T$，$N_{\text{vars}} = 4$），直接在守恒量上执行标量算法会跨特征族污染（激波、接触间断、声波的振荡互相耦合）。处理方式：投影到**局部特征场**逐分量执行标量限制。

**方案选择**：Zhu 2019 在结构网格上用 $x/y$ 两个坐标方向分别分解后取算术平均；三角形与一般四边形网格上面法向是斜向的，坐标方向不再自然。CMeles 统一采用**面法向方案**（Zhu 2021 / Lu 2023）：对 troubled cell 的每条有效边执行，在矩形网格上自动退化为坐标方向方案的加权版本——单一代码路径覆盖全部网格类型。

对 troubled cell $\Omega_0$ 的每条边 $f$（外单位法向 $\boldsymbol{n}_f$，边长 $|e_f|$）：

1. 计算面法向 Jacobian 的左、右特征向量矩阵 $L_f$、$R_f$（$L_f = R_f^{-1}$，$L_f \mathbf{A}_f R_f = \Lambda$ 对角，$\mathbf{A}_f = \mathbf{f}'(\bar{\boldsymbol{u}})\, n_{f,x} + \mathbf{g}'(\bar{\boldsymbol{u}})\, n_{f,y}$），线性化点 $\bar{\boldsymbol{u}}$ 为**真单元平均**（$\bar{u} = 2(Mu^{(\cdot)})_0/|\Omega_0|$，见记号约定）。一维 Euler 特征结构可复用[黎曼求解器文档](riemann_solver.md)的旋转不变性框架：将守恒量旋转到面法向局部坐标系后用一维特征分解；
2. 逐模态投影到特征场：$\tilde{\boldsymbol{u}}^{(l)} = L_f \boldsymbol{u}^{(l)}$（$L_f$ 作用在变量索引上，每个模态一次 $4 \times 4$ 矩阵乘；特征变换与模态截断/投影作用在不同索引上，两者可交换次序）；层级候选、$\beta_{0,1}$ 的邻居线性部分同样在特征场中构造；
3. 对 4 个特征分量各自执行标量 MR-WENO（第 1–4 步），得 $\tilde{\boldsymbol{p}}^{\,f}_{\text{new}}$；
4. 投回物理空间 $\boldsymbol{p}^{\,f}_{\text{new}} = R_f \tilde{\boldsymbol{p}}^{\,f}_{\text{new}}$。

最终重构按**边长加权平均**：

$$ \boldsymbol{u}_{\text{new}} \Big|_{\Omega_0} = \frac{\sum_f |e_f|\, \boldsymbol{p}^{\,f}_{\text{new}}}{\sum_f |e_f|} $$

（`MeshGeometry::faceNormals()` 存储的未归一化法向 $\hat{\boldsymbol{n}}$ 的模即物理边长，可直接用作权重。）

> **相邻 troubled cell 的独立性**：每个 troubled cell 读取其邻居**限制前**的模态系数（用于 $\beta_{0,1}$ 与特征分解），与邻居自身是否被限制无关。因此各单元的限制完全独立、无迭代、无顺序依赖，GPU 上无写冲突——与 DG 残差计算的并行模式一致。

> **粘性流的说明**：特征分解使用的 $\mathbf{A}_f$ 是**无粘** Euler 通量的面法向 Jacobian（在 $\bar{\boldsymbol{u}}$ 处线性化）。粘性项（IPDG / LDG）是抛物型修正，不引入新的双曲特征场，限制器的特征场分解在 NS 方程下**保持不变**（Lu 2023 对 NS 方程同样仅用无粘 Jacobian）。

### 守恒与精度性质小结

| 性质 | 依据 | 验证方法 |
| :--- | :--- | :--- |
| 严格守恒 | 投影路径下所有候选共享真单元平均（$q_0$ 即平均常数），细节多项式保持均值，凸组合保均值 | 限制前后逐单元比较 $2(Mu^{(\cdot)})_0/\|\Omega_0\|$，应逐位一致 |
| 光滑区保阶 | 各级 $\omega \to \gamma$，链条归纳还原 $q_N = u_h$，修改量 $O(h^2)$ | **全场单元强行标记为 troubled** 的精度测试：与无限制器结果对比，$P_3/P_4$ 误差应逐位一致，$P_1/P_2$ 渐近保持 $N+1$ 阶（粗网格误差略增属正常，Zhu 2019 表 4.1/4.2 的行为特征） |
| 间断区非振荡 | 权重自动向低级转移，逐级退化最坏到常数（一阶） | Sod / Lax / 激波密度波 / blast wave 等标准算例 |
| 稳态收敛 | 限制器可平稳退化到一阶，非线性权驻定后稳定 | 定常问题残差历史应收敛到机器零（对比经典 WENO 限制器的残差停滞，Lu 2023） |
| 模板紧凑 | 邻居信息仅 $\beta_{0,1}$ 用到直接邻居的线性模态 | troubled cell 标记的时间-空间分布输出（诊断用） |

## 问题单元检测：修改版 KXRCF

### 判据

将单元边界按流动方向分为入流部分 $\partial\Omega_0^-$（面平均速度与外法向点积为负的边）与出流部分 $\partial\Omega_0^+$。标记 $\Omega_0$ 为 troubled cell 的条件：

$$ \boxed{\;
\frac{\left| \displaystyle\int_{\partial\Omega_0^-} \left( \rho_h \big|_{\Omega_0} - \rho_h \big|_{\Omega_l} \right) \mathrm{d}\Gamma \right|}
{h^{R}\, \left| \partial\Omega_0^- \right| \cdot \left\| \rho_h \right\|_{\partial\Omega_0}}
\; \ge \; C_k \;}
$$

- $\rho_h \big|_{\Omega_0}$、$\rho_h \big|_{\Omega_l}$：同一段边上本单元与邻居 $l$ 的多项式迹（DG 解跨单元间断，两者之差即边界跳跃）；指示变量取**密度**；
- $h$：单元特征长度。三角形用外接圆半径；四边形无外接圆，统一采用**质心到顶点的最大距离**；
- $R = \dfrac{\min(N, 2) + 1}{2}$，即 $P_1$ 取 $1$、$P_2$ 及以上取 $1.5$（Fu & Shu 修正：原版 KXRCF 的 $(N+1)/2$ 在高阶时过度标记）；
- $C_k$：阈值常数，一律取 $1$；
- $\| \rho_h \|_{\partial\Omega_0}$：归一化尺度，Zhu 系列文献取 $\rho_h$ 沿**整条边界的最小值**（min 使指示值更大、标记更保守，更利于鲁棒）。KXRCF 原版无此归一化因子，它是 Fu & Shu 引入的，各文献对 min/max/mean 取法不一——实现中做成可配置项（`shock_norm`），默认 `min`。

**离散实现**：入流/出流按边分类（用 $\Omega_0$ 的单元平均速度在边中点与外法向的点积符号）。分子为入流边上密度跳跃的面求积（复用面通量计算的"先算后 gather"机制：`faceWeights` 表在面积分点上同时求值两侧的 $\rho^\pm$，见[控制方程与 DG 场文档](governed_equations_and_DG_field.md)的面积分一节）；分母的 $\| \rho_h \|_{\partial\Omega_0}$ 在逐单元核函数中求值本单元在各边积分点上的密度迹取 min。

**边界处理**：

- **域边界的入流边**无邻居可比较，跳过该段（不贡献分子与 $\left| \partial\Omega_0^- \right|$）；
- **周期边界面**已由 `Mesh::applyPeriodicPairing` 转为伪内面，按普通内面处理，无需特判；
- 全部边均为出流（如超声速出口附近的单元）时指示值取 $0$，不标记。

### 正值保持的 CFL 上界

**CFL 约定（首先澄清）**：CMeles 配置中的 `cfl` 是**含多项式阶数修正的 DG CFL**，而非经典 FV 单元层 CFL——`estimateDt`（`estimate_dt.okl`）的实现为

$$ \Delta t = \mathrm{CFL} \cdot \min_K \frac{h_K}{(2N+1)\,\Lambda_{\max,K}}, \qquad \Lambda_{\max} = \max_q \big(|\boldsymbol{u}| + c\big), $$

即同一 `cfl` 数值在高阶时允许的物理时间步比 FV 惯例小 $(2N+1)$ 倍（$h_K$：四边形取最短边，三角形取 $\sqrt{2A}$）。以下上界均换算到这一约定下表述。

**一维定理基准**（Zhang & Shu 2010；Gauss–Lobatto 配点求积 + SSP 推进 + 单调通量 + 缩放算子）：前向 Euler 步（经凸组合传播到整个 SSP-RK）逐点保正的条件为

$$ \frac{\alpha\, \Delta t}{h} \;\le\; \tilde\lambda(N) \;=\; \min_m \hat\omega_m \;=\; \frac{1}{N(N+1)} \quad (N \ge 1\text{；}N{=}0 \text{ 时 } \tilde\lambda = 1), $$

其中 $\hat\omega_m$ 为归一化 Gauss–Lobatto 权重（$\sum_m \hat\omega_m = 1$）、$\alpha$ 为最大波速。二维张量积推广（Zhang, Xia & Shu 2012）的条件形如 $\Delta t\,(\alpha_1/h_x + \alpha_2/h_y) \le \tilde\lambda$，各向同性方形网格上等效于 1D 界再除以 2。换算到 $(2N+1)$ 约定并与本项目经验值对照：

| $N$ | 0 | 1 | 2 | 3 | 4 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\tilde\lambda(N) = 1/[N(N{+}1)]$ | 1 | 1/2 | 1/6 | 1/12 | 1/20 |
| 等效 CFL 上界，1D：$(2N{+}1)\tilde\lambda$ | 1.00 | 1.50 | 0.83 | 0.58 | 0.45 |
| 等效 CFL 上界，2D 各向同性（近似减半） | 0.50 | 0.75 | 0.42 | 0.29 | 0.23 |
| 本项目经验值（Zhu 2019） | — | 0.30 | 0.18 | 0.10 | 0.08 |

经验值全部低于 2D 近似上界（1.5–3 倍余量）——**在参考界意义下满足保正条件**；$N = 1$ 时 1D 界（1.5）反而比线性稳定性更松，说明低阶时线性稳定性先行约束。

**与严格定理的差距（强调）**——当前实现是"工程保正"而非"定理保正"：

1. 定理要求 Gauss–Lobatto 型配点（含端点、正对角权重结构使更新可写成正系数凸组合）；CMeles 体积分点为 Gauss 内点、质量矩阵满秩（非仿射单元），凸组合论证不逐字适用，上表应视为**参考界**而非保证；
2. 实现的保障是**事后型**：正性触发检测 + 缩放在违例发生后修复（见"与混合网格及后续发展的衔接"的正值保持条目）；超过参考界运行不会崩溃，而是缩放频繁触发（$\theta \ll 1$ 增多、局部等效降阶、精度下降）；
3. 缩放检查点为体 Gauss + 面 Gauss 点，点间多项式下探在 $N \ge 3$ 时原则上不可控（$P \le 2$ 可由曲率有界）；更高阶需要时可加密检查点集（如含单元角点）；
4. 隐式/双时间推进路径不在 SSP 凸组合论证覆盖内，其稳定性由伪时间步长自适应承担（见[时间推进文档](time_marching.md)）。

若将来要求定理级保证：按所用求积的等效最小正权推导逐单元 $\Delta t$ 上界，并在 `estimateDt` 中与线性稳定性步长取 min 即可（边界清晰的小型后续项）。

## 参数汇总

| 参数 | 记号 | 取值 | 说明 |
| :--- | :---: | :---: | :--- |
| 线性权重（每级相同） | $\gamma_{\ell_2-1,\ell_2} / \gamma_{\ell_2,\ell_2}$ | $0.01\,/\,0.99$ | 任意正数皆可（和为 1）；偏大 $\gamma_{\text{high}}$ 利于光滑区恢复 |
| WENO 防零小量 | $\varepsilon_w$ | $10^{-6}$ | Zhu 2019 用 $10^{-10}$，无实质差异 |
| KXRCF 阈值 | $C_k$ | $1$ | |
| KXRCF 网格尺度幂 | $R$ | $(\min(N,2)+1)/2$ | $P_1$: 1，$P_2$及以上: 1.5 |
| KXRCF 归一化 | — | `min`（可配置） | 见检测一节 |
| 指示变量 | — | 密度 | |
| CFL（含限制器） | — | $P_1$: 0.3，$P_2$: 0.18，$P_3$: 0.1，$P_4$: 0.08 | Zhu 2019 推荐值。约定为 $(2N{+}1)$ 多项式修正 CFL（`estimateDt`），非经典 FV CFL；均低于正值保持定理参考界（见"正值保持的 CFL 上界"节）。精度测试时进一步减小以保证空间误差主导 |
| 应用频率 | — | 每个 RK 内 stage 之后 | 见实现方案的挂载点一节 |

## CMeles 实现方案

### 模块组织

```
src/limiter/
  LimiterBase.hpp          # 抽象接口
  MrWenoLimiter.hpp/.cpp   # MR-WENO 限制器（检测 + 重构）
  okl/
    kxrcf_face_jump.okl    # 核函数 1: 逐面密度跳跃积分
    kxrcf_flag.okl         # 核函数 2: 逐单元判据 → troubled 标志
    mr_weno_limit.okl      # 核函数 3: 逐 troubled cell 重构
```

接口设计（供时间推进模块回调）：

```cpp
// src/limiter/LimiterBase.hpp
class LimiterBase {
public:
    virtual ~LimiterBase() = default;
    /// Apply the limiter to the modal coefficient vector o_u in place.
    /// Called after every RK stage update (see StepperBase hook).
    virtual void apply(occa::memory o_u) = 0;
};
```

`MrWenoLimiter` 持有 `MeshGeometry`、`BasisFunctions2D`、邻接拓扑的设备句柄，负责构建上述三个核函数并按序调用。参数（$C_k$、$\gamma$、$\varepsilon_w$ 等）在构造时从 `Config` 读入，作为核函数运行期参数传入（遵循现有约定：仅 `Real`、`TILE_SIZE`、`NMODES_MAX`、`NQ2_MAX` 等数组容量上限走 JIT 宏，业务维度走 `const int` 参数以最大化 JIT 缓存命中，见[OCCA 文档](occa.md)）。

### OKL 内核设计

三个内核均按现有约定使用 `@tile(TILE_SIZE, @outer, @inner)`（`TILE_SIZE = 256`），单元内全串行、无 `@shared`。

**核函数 1 `kxrcf_face_jump`（逐面）**：模仿 `surface_integral.okl` 的两趟结构——第一趟在面积分点上求值左右单元的密度迹 $\rho^-$、$\rho^+$（模态系数 × `faceWeights` 表），第二趟按积分点加权累加，输出每个面的**带符号**跳跃积分 $\displaystyle \int_{e_f} (\rho_L - \rho_R)\,\mathrm{d}\Gamma$（符号相对左→右法向 $\hat{\boldsymbol{n}}$）。每面独立并行、只写自己的结果，无原子操作。边界面的输出为哑值（由核函数 2 跳过）。

**核函数 2 `kxrcf_flag`（逐单元）**：逐单元遍历其面表（`elemFaces`，四边形 4 条、三角形退化槽位为 $-1$ 跳过）：

1. 由面两侧标志（`faceKL`/`faceKR` 等）确定本单元在该面的朝向，得外法向；
2. 用本单元平均速度 $\bar{\boldsymbol{u}}$（真平均，见记号约定）与外法向点积分类入流/出流；
3. 对入流内面累加 $|\text{jump}_f|$，得分子；累加入流边长得 $\left| \partial\Omega_0^- \right|$；
4. 在本单元各边积分点上求值 $\rho$ 迹取 min（归一化尺度，按配置 min/max/mean）；
5. 计算 $h$（质心最大顶点距离，由 `elem_vertices` 现场算）、比较 $C_k$，写 `troubled[elem]`（int 0/1）。

**核函数 3 `mr_weno_limit`（逐单元）**：非 troubled 单元直接跳过（`if (troubled[elem] == 0) return;` 后仍占线程槽，保持 `@tile` 结构统一）。troubled 单元执行完整算法：

1. 读入本单元 $N_{\text{vars}} \times N_{\text{modes}}$ 模态系数与各邻居的线性模态（按 `elemFaces` + `faceKL`/`faceKR` gather，直接从全局解向量读取，EVM 布局保证邻居的模态段连续）；
2. 逐边做特征场分解（$4 \times 4$；线性化点为真单元平均，用本单元 $M$ 第 0 行加权求出，或一次 $\Lambda_{wJ}$ 求积）；
3. 每个特征分量执行标量算法：层级候选（按逐单元仿射标志走截断或即时投影——投影需现场组装 $M_{\ell\ell} = \mathbf{V}_\ell^T \operatorname{diag}(\Lambda_{wJ}) \mathbf{V}_\ell$ 并做小 Cholesky，$P_4$ 时最大 $10 \times 10$，仍在寄存器/线程局部数组容量内）、telescoping 递推、$\beta$（积分点求值 + $\mathbf{J}^{-1}$ 链式求导 + $\Lambda_{wJ}$ 收缩）、$\beta_{0,1}$（仿射邻居闭式 / 非仿射邻居查梯度表）、WENO-Z 权重；
4. 按边长加权平均 4 个方向的结果，覆写本单元模态系数。

> **逐单元仿射标志**：建议在 `MeshGeometry` 中新增逐单元 int 数组（判据：逐点 $\det\mathbf{J}$ 与首点一致，或逐单元重算 $c_1 = c_2 = 0$），随 `allocateDeviceMemory` 上传。全局标志 `isOrthogonal()` 不适用于混合网格下的逐单元分支。
>
> **邻居梯度表（非仿射邻居的 $\beta_{0,1}$）**：预处理阶段为每个面邻接对预计算目标单元各体积分点处邻居 $\phi_1, \phi_2$ 的物理梯度（每点 4 实数）。涉及把目标积分点物理坐标反解到邻居参考坐标（双线性映射的 Newton 迭代，2–3 步收敛；几何静态故只需离线一次）。首版可先只实现仿射邻居闭式路径 + 非仿射邻居以质心单点梯度近似，观察对算例的影响后再决定是否引入完整查表。

### 挂载点：时间推进模块的 post-stage 回调

限制器必须作用于**每个 RK 内 stage 的状态更新之后、下一次 RHS 之前**（SSP 框架内的非线性稳定性要求）。当前 `StepperBase`（`src/time/StepperBase.hpp`）没有 stage 后钩子（`SimpleExplicitStepper.hpp` 头部注释已预留此设计），需扩展：

```cpp
// src/time/TimeTypes.hpp
/// Post-stage state filter (e.g. limiter), applied in place to the stage
/// state after every Runge-Kutta stage update and to the final output.
using StateFilter = std::function<void(occa::memory o_u)>;

// StepperBase: 新增
void setPostStageFilter(StateFilter filter);   // 允许为空（无限幅）
```

各 stepper 在**每次** stage 状态写回后（`eulerUpdate` / `convexCombine` / `rkStageCombine` 的输出）以及最终输出状态上调用 `filter_`（若非空）：

- `SimpleExplicitStepper`（SSPRK 手写展开）：三个 stage 状态 + 输出，共 4 处；
- `RungeKuttaStepper`（通用 Butcher）：`rkStep` 内每个 stage 的 `o_uStage_` + 最终更新与 FSAL 起点；
- 自适应步长方案：stage 状态必须限制（它要参与 RHS 计算）；被拒步的重试自然重复限制，无需特殊处理。

`CompressibleFlowSolver::run()` 在创建 stepper 后组装 limiter 链（MR-WENO，之后可接保正缩放限制器）：`stepper->setPostStageFilter([solver](occa::memory u) { solver->applyStageFilter(u); })`。限制器对象由 `Config` 的 `[limiter]` 段决定是否创建（`type = "none"` 且 `positivity = false` 时均不创建，保持现有行为）。

**初始场的一次性过滤**（第二个挂载点，`CompressibleFlowSolver` 构造时）：post-stage 过滤器只能看到 RK stage 状态，初始场本身在其覆盖之外——而强间断初始条件（如双马赫问题 $p$ 比 $116:1$ 的入射激波）的模态投影携带 Gibbs 下冲，可使积分点压力为负，第一次 RHS 求值即在 LLF 通量内对负数开方产生 NaN。因此求解器在初始条件投影之后、首个时间步之前调用一次 limiter 链（MR-WENO 抑制振荡；`positivity = true` 时再经 Zhang-Shu 保正缩放保证可容许，见 `positivity_limiter.md`）；该过滤同时让 `riemann_target = "ic"` 的初始场快照引用物理上合理（已过滤）的场。对温和间断该步骤无害。

### 配置项设计

```toml
[limiter]
type = "mr-weno"        # "none"（默认）| "mr-weno"
ckc = 1.0               # KXRCF 阈值 C_k
shock_norm = "min"      # KXRCF 边界值归一化: "min" | "max" | "mean"
gamma_low = 0.01        # 线性权重 γ_{ℓ2−1,ℓ2}
gamma_high = 0.99       # 线性权重 γ_{ℓ2,ℓ2}
epsilon = 1.0e-6        # WENO-Z 防零小量 ε_w
```

指示变量固定为密度；`apply_per_stage` 不设开关（恒为逐 stage，关闭等价于 `type = "none"` 的错误用法）。

### 与混合网格及后续发展的衔接

- **混合网格（四边形 + 三角形 + 一般四边形）**：算法层面完全不区分单元类型，唯一分支是逐单元仿射标志（截断 vs 投影）。当前 `StructuredMeshGenerator` 产出的平行四边形 + 对角切分三角形，以及将来 HDF5 任意网格路径读入的一般四边形，限制器代码无需任何改动；
- **三维扩展**（四面体）：模态数改为 $N_\ell = \frac{(\ell+1)(\ell+2)(\ell+3)}{6}$；邻居 4 个；特征分解沿各面法向（$5 \times 5$）；加权平均的权重改为面面积。算法结构不变；
- **粘性项（NS 方程，IPDG / LDG）**：CMeles 计划以 IPDG（内罚）实现粘性通量。IPDG 是**原始（primal）形式，无辅助变量**——粘性弱形式（体积分中的 $\nabla u_h$、面积分中的平均梯度 $\{\!\{\nabla u\}\!\}$ 与跳跃罚项 $\sigma_f\,[\![u]\!]$）完全由解多项式 $u_h$ 决定。限制器经 post-stage 钩子修改 $u_h$ 后，粘性通量自动使用限制后的多项式，**无条件、无顺序约束**。（对照：若走 LDG 路线，辅助梯度 $\boldsymbol{Q}$ 每个 stage 必须在限制 $\boldsymbol{U}$ **之后**从限制后的 $\boldsymbol{U}$ 重算——Lu 2023 的做法——钩子顺序为 limit $\boldsymbol{U}$ → 重算 $\boldsymbol{Q}$ → 粘性通量。）系统情形的特征分解见上文"粘性流的说明"：仍用无粘 Jacobian，无需修改。粘性流下的补充注意事项：
  1. **KXRCF 在边界层的过标记**：探测器按边界跳跃判别，分辨不足的薄边界层中跳跃幅度偏大，可能把壁面附近大片光滑单元标为 troubled。MR-WENO 的保阶性质使过标记主要损失分辨率与计算效率、不损稳定性；验证时应监控 troubled 标记数——Couette / Poiseuille / 层流平板等纯粘性光滑算例的标记数应接近零，高 Re 薄边界层被大面积标记时可增大 $C_k$；
  2. **粘性时间步长**：显式推进需叠加粘性步长限制（参考 Lu 2023 式 (2.14)，含 $Re$、$Pr$ 的形式）；
  3. **与隐式推进（DITR / 双时间步）的配合**：限制器应作为 **stage 状态的投影**施加于隐式求解**之外**——隐式残差函数以限制后的状态为数据，不在 Newton/GMRES 迭代内部嵌入限制（非线性权切换会使隐式算子非光滑，破坏迭代收敛）。MR-WENO 驻定后非线性权固定、残差可收敛到机器零的性质与该用法配套。
- **正值保持**：MR-WENO 重构本身不保证 $\rho > 0$、$p > 0$，**已叠加 Zhang-Shu 型正值缩放**（仅在 troubled cell 内）：重构多项式在体积分点与本单元各面迹积分点上检查 $\rho$、$p$ 下限（$10^{-8}$），违例时凸缩放 $u_{\text{new}} \leftarrow \theta\, u_{\text{new}} + (1-\theta)\,\bar{u}$（$\theta$ 由各违例点取最小；因 $p$ 沿混合线段为凹函数、$\rho$ 为线性，缩放后端点间逐点 $\ge \min(\text{端点值})$，即严格恢复正值）。均值常数的模态表示为 $(2\bar{u}, 0, \ldots)$（$\phi_0$ 本身即常数 $1/2$），故缩放严格保单元平均、不破坏守恒。**面点必须一并检查**——LLF 通量在面积分点求 $\sqrt{p}$，仅体点正值不能阻止面迹负值的一步 NaN 路径。配套地，检测器增加**正性触发**：体/面点出现非正 $\rho$ 或 $p$ 即强制标记 troubled（KXRCF 对全出流单元分子为零、不标记，而其多项式过冲可能已非物理——实测此为三角形网格强间断算例的失稳根源）；
- **稳态求解**：限制器的可退化性保证驻定后非线性权稳定，与隐式/伪时间步进（见[时间推进文档](time_marching.md)）兼容，残差可收敛到机器零——这是选择 MR-WENO 而非经典 WENO 限制器的核心依据之一。

## 实现路线图与验证计划

### 阶段一：检测器 + 标量限制器（无特征分解）

1. `MeshGeometry` 增加逐单元仿射标志（+ 可选邻居梯度表）；
2. 三个 OKL 内核 + `MrWenoLimiter`（暂以逐变量标量模式运行）+ `StepperBase` 钩子 + `Config` 段；
3. **精度测试（关键验收）**：光滑算例（周期 Burgers / Euler 密度波）将**全部单元强制标记为 troubled**，与无限制器结果对比——$P_3/P_4$ 误差应逐位一致，$P_1/P_2$ 渐近保持 $N+1$ 阶；同时验证限制前后单元平均逐位不变（守恒）与光滑场 $\omega \to \gamma$ 时输出逐位还原 $u_h$；
4. 单元测试：仿射单元上"截断 == 投影"（机器精度）；telescoping 恒等式 $\gamma_{\ell-1,\ell} p_{\ell-1,\ell} + \gamma_{\ell,\ell} p_{\ell,\ell} = q_\ell$；非仿射单元上 $q_0$ == 真单元平均。

### 阶段二：面法向特征分解 + 系统算例

1. 逐边特征场分解（复用旋转不变性 + 一维特征结构）与边长加权平均；
2. 标准算例：Sod、Lax、激波-密度波相互作用（Shu-Osher）、blast wave、Double Mach 反射、前台阶；CFL 按参数表设置；
3. 输出 troubled cell 标记的时间-空间分布（诊断场），检查标记的局部性（激波附近窄带，不蔓延到光滑区）。

### 阶段三：鲁棒性增强与稳态验证

1. 正值守恒缩放（可选）；
2. 稳态算例残差历史验证（对比关闭限制器与仅探测不限制的行为）；
3. 三角形/混合网格算例（`splitTriangles`）复跑阶段一、二的部分算例，确认非仿射路径正确性；
4. 粘性算例（依赖 IPDG 粘性项实现后进行）：Couette / Poiseuille / 层流平板验证 troubled 标记数接近零、限制器不破坏粘性精度；激波-边界层干扰算例验证鲁棒性。

## 参考文献

1. J. Zhu, J. Qiu, C.-W. Shu, *High-order Runge-Kutta discontinuous Galerkin methods with a new type of multi-resolution WENO limiters*, J. Comput. Phys. **407** (2020) 109105.
2. J. Zhu, C.-W. Shu, J. Qiu, *High-order Runge-Kutta discontinuous Galerkin methods with a new type of multi-resolution WENO limiters on tetrahedral meshes*, Commun. Comput. Phys. **29** (2021) 1030–1058.
3. Y. Lu, J. Zhu, S. Cui, Z. Wang, L. Tian, N. Zhao, *High-order local discontinuous Galerkin method with multi-resolution WENO limiter for Navier-Stokes equations on triangular meshes*, Commun. Comput. Phys. (2023).
4. J. Zhu, C.-W. Shu, *A new type of multi-resolution WENO schemes with increasingly higher order of accuracy*, J. Comput. Phys. **375** (2018) 659–683.（有限体积 MR-WENO 原始格式）
5. L. Krivodonova, J. Xin, J.-F. Remacle, N. Chevaugeon, J.E. Flaherty, *Shock detection and limiting with discontinuous Galerkin methods for hyperbolic conservation laws*, Appl. Numer. Math. **48** (2004) 323–338.（KXRCF 探测器）
6. G. Fu, C.-W. Shu, *A new troubled-cell indicator for discontinuous Galerkin methods for hyperbolic conservation laws*, J. Comput. Phys. **347** (2017) 305–327.（KXRCF 的 $h$ 幂次与归一化修正）
7. R. Borges, M. Carmona, B. Costa, W.S. Don, *An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws*, J. Comput. Phys. **227** (2008) 3191–3211.（WENO-Z 权重）
8. G. Jiang, C.-W. Shu, *Efficient implementation of weighted ENO schemes*, J. Comput. Phys. **126** (1996) 202–228.（光滑度指示器）
