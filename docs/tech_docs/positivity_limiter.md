# 保正限制器（Zhang-Shu Positivity-Preserving Scaling Limiter）

## 概述

保正限制器是逐单元的**守恒凸投影**：把单元的高阶模态以单一因子 $\theta \in [0,1]$ 向单元均值缩放，直到该单元所有体积 Gauss 点与面 Gauss 迹点上的密度、内能非负（带相对裕度）。它与 MR-WENO 限制器职责互补：

- **MR-WENO** 管精度：在问题单元上抑制振荡，但**不保证**密度/压力为正（无 $\theta$-缩放机制）；
- **保正限制器** 管可容许性：**保证**喂给任何 RHS 求值的态满足 $\rho > 0$、$e > 0$，从而 LLF 通量中 $\sqrt{\gamma p / \rho}$ 永不遇到负数开方。

与 MR-WENO 的另一本质区别：保正缩放是状态的**连续（分段线性）投影**——$\theta$ 随状态连续变化，没有 WENO 权重/KXRCF 标志那种 $O(1)$ 跳变。因此它**可以**安装在双时间法的伪时间迭代内部（见下文接入语义），而 WENO 类限制器按过滤契约（`StepperBase::setPostStageFilter` 规则 2）必须留在伪迭代之外。

**参考文献**：X. Zhang & C.-W. Shu, "On positivity-preserving high order discontinuous Galerkin schemes for compressible Euler equations on rectangular meshes", J. Comput. Phys. 229 (2010) 8918-8941。

## 算法

对每个单元 $K$，记体积求积精确均值为 $(\bar\rho, \bar m_x, \bar m_y, \bar E)$（非仿射单元也精确的公式 $\bar u = \sum_q \lambda_{wJ,q} u_h(x_q) / \sum_q \lambda_{wJ,q}$），均值内能 $\bar e = \bar E - |\bar m|^2 / (2\bar\rho)$，逐点内能 $e_q = E_q - |m_q|^2 / (2\rho_q)$。

投影作用于点值形式

$$ u_h(x) \leftarrow \bar u + \theta \, (u_h(x) - \bar u), \qquad \theta \in [0,1], $$

在模态层面即：常数模态（均值）不动，全部四个守恒变量的 $m > 0$ 模态同乘 $\theta$——**动量也被缩放**，这是 Zhang-Shu 配方绕过"仅缩放 $E$ 救不了动量过冲"的关键。该操作在仿射单元上严格保持均值（守恒）；因 $\varphi_0$ 为常函数，点值形式的仿射映射与模态缩放精确等价。

$\theta$ 的闭式候选（Zhang-Shu）：

$$ \theta_1 = \min\!\bigl(1,\ \tfrac{(1-\varepsilon)\bar\rho}{\bar\rho - \min_q \rho_q}\bigr), \qquad \theta_2 = \min\!\bigl(1,\ \tfrac{(1-\varepsilon)\bar e}{\bar e - \min_q e_q(\theta_1)}\bigr), \qquad \theta = \theta_1 \theta_2, $$

其中 $\min_q$ 取遍该单元全部体积 Gauss 点与面 Gauss 迹点，$e_q(\theta_1)$ 为 $\theta_1$ 缩放后点态的内能（点值仿射缩放可由原始点值直接算出，无需重新投影模态）。

**验证与回退**：上述闭式在 Zhang-Shu 的 Gauss-Lobatto 点集（体积点与面点重合）下有充分性证明；CMeles 用 Gauss 点求积，闭式仅为起始猜测。内核随后在全部点上**验证**实际判据

$$ \rho_q(\theta) \ge \varepsilon \bar\rho, \qquad e_q(\theta) \ge \varepsilon \bar e, $$

不满足则 $\theta \leftarrow \theta/2$ 重验（上限 60 次）。由于 $\theta \to 0$ 时点值连续收敛到可容许的均值，回退必然终止，因此对任意点集（含 Gauss 点与面迹点的并集）判据都**构造性成立**。

**均值不可容许情形**（$\bar\rho \le 0$ 或 $\bar e \le 0$）：保均值的缩放无法修复均值本身，此类单元被跳过并打标志 2（保持原样、随后 loudly 失败，绝不静默）。显式路径上由 Zhang-Shu 定理保证均值恒可容许；双时间路径无定理，均值破坏会显式暴露。

## 实现

| 组件 | 文件 | 说明 |
| --- | --- | --- |
| 内核 | `src/limiter/okl/positivity_scale.okl` | 单元局部三段式：均值 + $\min\rho_q$ 扫描（复用 `kxrcf_flag.okl` 的 `V2D`/`lambdaWJ`/`Vface` 点值求值与面遍历模式）→ $\theta_1$/$\theta_2$ 闭式 → 验证减半回退 → 缩放 $m>0$ 模态。标准 `@tile(TILE_SIZE)` 全单元循环（无短列表问题），无邻接数据依赖 |
| 宿主 | `src/limiter/PositivityLimiter.{hpp,cpp}` | `LimiterBase` 子类；`apply()` 识别双时间 coupled 模式的堆叠态（长度 $2N$ 时切半分别投影）；`readFlags()` 供诊断/测试读取逐单元动作标志（0 未动 / 1 缩放 / 2 均值不可容许） |
| 配置 | `src/config/Config.{hpp,cpp}` | `[limiter] positivity`（bool，默认 false）、`positivity_epsilon`（相对裕度 $\varepsilon$，默认 1e-10，须在 (0,1)） |

内核 JIT 只需 `Real`/`TILE_SIZE` 定义；$\gamma$ 不需要（判据作用在内能 $e$ 而非 $p$ 上）。

## 接入语义：对所有 stepper 统一生效

`positivity = true` 时（单个 config 开关）：

1. **显式家族**（euler / ssprk3 / rk32 / ssprk221/321/332/432 / rk54）：经既有 `setPostStageFilter` 钩子装入过滤器链 **[MR-WENO（若启用）→ 保正]**，作用于每个将喂给 RHS 的级段态——保正必须放最后，保证下游 RHS 看到的态可容许；
2. **dual-time 家族**（be / ditr-u2r2/u2r1/u3r1）：发布态投影同样走上述链；**伪迭代态**经新增的 `StepperBase::setPseudoStateFilter()` 虚函数通道**只装保正**（`DualStepper` 覆写为安装到 `pseudo_`/`pseudoC2_`，`rkStep` 的级段态/候选态插槽自动生效；coupled 堆叠态由 `apply()` 切半处理）；
3. **初始场过滤**（模态投影后的 Gibbs 欠冲）同样走该链。

`type = "none"` + `positivity = true` 为纯保正模式（无 WENO）。`positivity = false`（默认）时一切现有行为逐位不变。

### 双时间伪迭代内的安全性论证

把 WENO 嵌入伪迭代会产生两个已记录的病理：迭代算子分段不连续（权重/标志 $O(1)$ 跳变）导致残差极限环；投影压低 $\|F\|$ 造成假收敛。保正缩放不受此约束：它是 Lipschitz 连续的投影，$\theta$ 平滑依赖于状态；收敛残差 $\mathcal{F}(u)$ 在投影后的态上求值，收敛目标即"投影后的隐式解"。此外 `DualStepper::notConverged` 增加了 $\|F\|$ 非有限即视为未收敛的护栏，NaN 残差永不再被误判收敛。

### 背景：为何需要它

DMR 强激波上网格扫描阶段 2 的失败矩阵（23 个变体）证明：双时间伪迭代中间态冲出正性域 → `llf.okl` 无下限的 $\sqrt{\gamma p/\rho}$ 产生 NaN → 接受机制（强制接受 / $\sigma$-NaN 盲区 / `amax` 归约跳过 NaN）放行 → 假收敛发布 NaN。保正限制器从上游掐断第一条链（RHS 永远看到可容许态），isfinite 护栏封死最后一条（假收敛）。

## 定理适用范围

- **显式 SSP 路径**：Zhang-Shu 定理完整适用——SSP 时间积分 + 本限制器 + CFL 不超过凸性界（本项目的 $(2N{+}1)$ 约定下 P1: 0.3 / P2: 0.42 / P3: 0.29 量级；实际推荐值更保守）⇒ 逐点保正。$\theta \equiv 1$ 于光滑区域，高阶精度不受影响（光滑场逐位不变，见测试 b）。
- **双时间路径**：无定理；伪迭代可容许性由本限制器**构造性**保证（验证回退），但伪迭代的**收敛效率**是独立问题（`max_pseudo_steps` 上限、`pseudo_rtol` 松紧），见 `time_marching/implicit_time_marching.md`。

## 测试

`ctest/limiter/test_positivity.cpp`：

- (a) 对抗模态场（注入大线性模态使积分点 $\rho<0$、$e<0$）：apply 后全部体积/面点可容许、均值与总质量守恒、注入带单元标志 1、干净单元逐位未动；
- (b) 光滑场：$\theta \equiv 1$，逐位 no-op，标志全 0；
- (c) 堆叠 $2N$ 态：两半分别投影，干净半逐位不变；
- (d) 过滤契约冒烟：euler/ssprk3/rk32/be/ditr-u2r1 全家族带 positivity 完成、质量守恒（覆盖伪迭代通道装配）；
- (e) 双时间强激波（小网格双马赫 IC）：`be + positivity` 存活此前必然 NaN 的启动段，最终态有限。

## 与其他文档的关系

- MR-WENO 限制器与过滤契约总纲：`mr_weno_limiter.md`、`StepperBase::setPostStageFilter` 代码注释；
- 双时间法与伪时间迭代：`time_marching/implicit_time_marching.md`；
- OKL 内核约定（`@tile`、点值求值模式）：`occa.md`。
