# 技术文档

本文档描述 CMeles 项目的数学基础和算法实现。

## 一、一维基函数与积分

本节描述 CMeles 中间断 Galerkin (DG) 方法在一维情况下使用的正交多项式、数值求积及相关矩阵运算。

### 1.1 正则勒让德多项式

#### 定义

正则（归一化）勒让德多项式 $\tilde{P}_n(x)$ 定义在参考区间 $[-1, 1]$ 上。它们由经典勒让德多项式 $P_n(x)$ 乘以归一化因子得到：

$$
\tilde{P}_n(x) = \sqrt{\frac{2n + 1}{2}} \, P_n(x)
$$

#### 正交归一性

正则勒让德多项式在 $[-1, 1]$ 上构成正交归一基：

$$
\int_{-1}^{1} \tilde{P}_m(x) \tilde{P}_n(x) \, dx = \delta_{mn}
$$

其中 $\delta_{mn}$ 是克罗内克符号。此性质确保 DG 离散化中的质量矩阵为单位矩阵，显著简化了计算。

#### 递推关系

经典勒让德多项式满足三项递推关系：

$$
(n + 1) P_{n+1}(x) = (2n + 1) x P_n(x) - n P_{n-1}(x)
$$

初始条件为：

$$
P_0(x) = 1, \quad P_1(x) = x
$$

对于正则多项式，将 $P_n(x) = \sqrt{\frac{2}{2n+1}} \, \tilde{P}_n(x)$ 代入经典递推关系，整理得：

$$
\tilde{P}_{n+1}(x) = \frac{\sqrt{(2n+1)(2n+3)} \cdot x \, \tilde{P}_n(x) - n \sqrt{\frac{2n+3}{2n-1}} \cdot \tilde{P}_{n-1}(x)}{n + 1}
$$

> **注意**：归一化勒让德多项式不能直接套用经典勒让德多项式的递推系数（即保持 $(2n+1)x\tilde{P}_n - n\tilde{P}_{n-1}$ 除以 $n+1$ 的形式），因为归一化因子 $\sqrt{(2n+1)/2}$ 随 $n$ 变化，代入后会产生额外的比例因子。

正则化的初始条件为：

$$
\tilde{P}_0(x) = \frac{1}{\sqrt{2}}, \quad \tilde{P}_1(x) = \sqrt{\frac{3}{2}} \, x
$$

#### 导数递推

经典勒让德多项式的导数满足以下关系：

$$
(2n + 1) P_n(x) = P'_{n+1}(x) - P'_{n-1}(x)
$$

可重排为高效计算导数的形式：

$$
P'_n(x) = \frac{2n - 1}{n} P_{n-1}(x) + P'_{n-2}(x)
$$

对于正则多项式，将 $P_n(x) = \sqrt{\frac{2}{2n+1}} \, \tilde{P}_n(x)$ 代入 $P'_{n+1} = (2n+1)P_n + P'_{n-1}$ 并整理，得到归一化版本的导数递推：

$$
\tilde{P}'_{n+1}(x) = \sqrt{(2n+1)(2n+3)} \, \tilde{P}_n(x) + \sqrt{\frac{2n+3}{2n-1}} \, \tilde{P}'_{n-1}(x)
$$

> **注意**：上式第二项系数为 $\sqrt{\frac{2n+3}{2n-1}}$，仅当 $n = 1$ 时因 $\tilde{P}'_0(x) = 0$ 退化为 $\tilde{P}'_2(x) = \sqrt{15} \, \tilde{P}_1(x)$。对于 $n \ge 2$，该系数 $\neq 1$，不可省略。

#### 前几项多项式

| 阶数 $n$ | $\tilde{P}_n(x)$                                                             |
| :------: | :--------------------------------------------------------------------------- |
|    0     | $\displaystyle\frac{1}{\sqrt{2}}$                                            |
|    1     | $\displaystyle\sqrt{\frac{3}{2}} \, x$                                       |
|    2     | $\displaystyle\sqrt{\frac{5}{2}} \left( \frac{3x^2 - 1}{2} \right)$          |
|    3     | $\displaystyle\sqrt{\frac{7}{2}} \left( \frac{5x^3 - 3x}{2} \right)$         |
|    4     | $\displaystyle\sqrt{\frac{9}{2}} \left( \frac{35x^4 - 30x^2 + 3}{8} \right)$ |

#### 求值算法

在给定点 $x$ 处计算 $\tilde{P}_n(x)$ 可使用以下算法：

```
function evalNormalizedLegendre(x, N):
    P[0] = 1 / sqrt(2)
    if N == 0: return P[0]

    P[1] = sqrt(3/2) * x
    if N == 1: return P[1]

    for n = 1 to N-1:
        P[n+1] = (sqrt((2*n + 1)*(2*n + 3)) * x * P[n]
                  - n * sqrt((2*n + 3)/(2*n - 1)) * P[n-1]) / (n + 1)

    return P[N]
```

该算法计算复杂度为 $O(N)$，对于适中的多项式阶数具有良好的数值稳定性。

#### 导数求值

导数 $\tilde{P}'_n(x)$ 可与多项式求值同时计算：

```
function evalNormalizedLegendreAndDerivative(x, N):
    P[0] = 1 / sqrt(2)
    dP[0] = 0
    if N == 0: return P[0], dP[0]

    P[1] = sqrt(3/2) * x
    dP[1] = sqrt(3/2)
    if N == 1: return P[1], dP[1]

    for n = 1 to N-1:
        P[n+1] = (sqrt((2*n + 1)*(2*n + 3)) * x * P[n]
                  - n * sqrt((2*n + 3)/(2*n - 1)) * P[n-1]) / (n + 1)
        dP[n+1] = sqrt((2*n + 1)*(2*n + 3)) * P[n]
                + sqrt((2*n + 3)/(2*n - 1)) * dP[n-1]

    return P[N], dP[N]
```

### 1.2 Vandermonde 矩阵

在实际计算中，反复使用递推关系求解多项式值效率较低。Vandermonde 矩阵提供了一种高效的方法，通过矩阵-向量乘法直接从多项式系数（模态系数）得到积分点上的值。

#### 定义

给定参考单元上 $N+1$ 个积分点 $\{\xi_0, \xi_1, \ldots, \xi_N\}$（其中 $\xi \in [-1, 1]$），Vandermonde 矩阵 $\mathbf{V}$ 定义为：

$$
\mathbf{V} = \begin{bmatrix}
\tilde{P}_0(\xi_0) & \tilde{P}_1(\xi_0) & \cdots & \tilde{P}_N(\xi_0) \\
\tilde{P}_0(\xi_1) & \tilde{P}_1(\xi_1) & \cdots & \tilde{P}_N(\xi_1) \\
\vdots & \vdots & \ddots & \vdots \\
\tilde{P}_0(\xi_N) & \tilde{P}_1(\xi_N) & \cdots & \tilde{P}_N(\xi_N)
\end{bmatrix}
$$

其中 $\mathbf{V}_{ij} = \tilde{P}_j(\xi_i)$。

#### 模态到节点转换

给定模态系数向量 $\hat{\mathbf{u}} = [\hat{u}_0, \hat{u}_1, \ldots, \hat{u}_N]^T$，节点值向量 $\mathbf{u} = [u_0, u_1, \ldots, u_N]^T$ 可通过矩阵乘法得到：

$$
\mathbf{u} = \mathbf{V} \hat{\mathbf{u}}
$$

即：

$$
u_i = \sum_{j=0}^{N} \hat{u}_j \tilde{P}_j(\xi_i), \quad i = 0, 1, \ldots, N
$$

#### 节点到模态转换：L2 投影

从节点值恢复模态系数需要求解 L2 投影问题。给定节点值 $\mathbf{u}$，求模态系数 $\hat{\mathbf{u}}$ 使得：

$$
\int_{-1}^{1} \left[ u_h(x) - u(x) \right] \phi_i(x) \, dx = 0, \quad i = 0, 1, \ldots, N
$$

其中 $u_h(x) = \sum_{j=0}^{N} \hat{u}_j \phi_j(x)$，$\{\phi_j\}$ 为基函数。展开后得到线性方程组：

$$
\sum_{j=0}^{N} \hat{u}_j \int_{-1}^{1} \phi_j(x) \phi_i(x) \, dx = \int_{-1}^{1} u(x) \phi_i(x) \, dx
$$

定义质量矩阵 $\mathbf{M}$ 和右端项 $\mathbf{f}$：

$$
M_{ij} = \int_{-1}^{1} \phi_i(x) \phi_j(x) \, dx, \quad f_i = \int_{-1}^{1} u(x) \phi_i(x) \, dx
$$

则投影问题简化为：

$$
\mathbf{M} \hat{\mathbf{u}} = \mathbf{f}
$$

**一般情况（非正交基）**：质量矩阵 $\mathbf{M}$ 为满秩稠密矩阵，需要求解线性方程组：

$$
\hat{\mathbf{u}} = \mathbf{M}^{-1} \mathbf{f}
$$

通常采用 LU 分解或 Cholesky 分解求解。对于 Lagrange 插值基等非正交基，质量矩阵非对角，每次投影都需要矩阵求逆，计算开销较大。

**正交归一基的简化**：当基函数 $\{\tilde{P}_j\}$ 正交归一时，质量矩阵为单位矩阵 $\mathbf{M} = \mathbf{I}$，投影简化为：

$$
\hat{\mathbf{u}} = \mathbf{f} = \mathbf{V}^T \mathbf{W} \mathbf{u}
$$

其中 $\mathbf{W}$ 为高斯求积权重构成的对角矩阵。当积分点数量等于模态数量时，$\mathbf{V}^{-1} = \mathbf{V}^T \mathbf{W}$。

### 1.3 坐标变换与雅可比矩阵

在实际计算中，物理单元通常不是标准单元 $[-1, 1]$。坐标变换将物理坐标 $x$ 映射到参考坐标 $\xi \in [-1, 1]$，使得所有计算可在统一的参考域上进行。

#### 一维坐标变换

对于一维单元 $[x_a, x_b]$，线性坐标变换为：

$$
x(\xi) = \frac{x_a + x_b}{2} + \frac{x_b - x_a}{2} \xi
$$

#### 雅可比矩阵与雅可比行列式

雅可比矩阵定义为坐标变换的导数：

$$
\mathbf{J} = \frac{\partial x}{\partial \xi} = \frac{x_b - x_a}{2}
$$

雅可比行列式为 $|\mathbf{J}| = \frac{\Delta x}{2}$，其中 $\Delta x = x_b - x_a$ 为单元长度。

#### 积分变换

物理单元上的积分可变换到参考单元上：

$$
\int_{x_a}^{x_b} f(x) \, dx = \int_{-1}^{1} f(x(\xi)) |\mathbf{J}| \, d\xi = \sum_{i=0}^{N} w_i f(x(\xi_i)) |\mathbf{J}|
$$

#### 导数变换

链式法则给出导数的变换关系：

$$
\frac{\partial u}{\partial x} = \frac{\partial u}{\partial \xi} \frac{\partial \xi}{\partial x} = \frac{\partial u}{\partial \xi} \mathbf{J}^{-1}
$$

在参考单元上使用 Vandermonde 导数矩阵计算导数后，需乘以 $\mathbf{J}^{-1}$ 得到物理坐标下的导数。

### 1.4 Vandermonde 导数矩阵

类似地，Vandermonde 导数矩阵用于高效计算多项式在积分点处的导数值。

#### 定义

Vandermonde 导数矩阵 $\mathbf{D}_V$ 定义在参考单元坐标系下：

$$
\mathbf{D}_V = \begin{bmatrix}
\tilde{P}'_0(\xi_0) & \tilde{P}'_1(\xi_0) & \cdots & \tilde{P}'_N(\xi_0) \\
\tilde{P}'_0(\xi_1) & \tilde{P}'_1(\xi_1) & \cdots & \tilde{P}'_N(\xi_1) \\
\vdots & \vdots & \ddots & \vdots \\
\tilde{P}'_0(\xi_N) & \tilde{P}'_1(\xi_N) & \cdots & \tilde{P}'_N(\xi_N)
\end{bmatrix}
$$

其中 $(\mathbf{D}_V)_{ij} = \tilde{P}'_j(\xi_i)$，导数是相对于参考坐标 $\xi$ 的导数。

#### 导数计算

给定模态系数 $\hat{\mathbf{u}}$，积分点处**参考坐标下的导数**可通过以下矩阵乘法直接得到：

$$
\frac{\partial \mathbf{u}}{\partial \xi} = \mathbf{D}_V \hat{\mathbf{u}}
$$

即：

$$
\left.\frac{\partial u}{\partial \xi}\right|_{\xi_i} = \sum_{j=0}^{N} \hat{u}_j \tilde{P}'_j(\xi_i), \quad i = 0, 1, \ldots, N
$$

#### 物理坐标下的导数

要得到物理坐标 $x$ 下的导数，需要利用链式法则进行坐标变换。

**链式法则推导**：设 $u$ 是物理坐标 $x$ 的函数，而 $x$ 又是参考坐标 $\xi$ 的函数，即 $u = u(x(\xi))$。根据链式法则：

$$
\frac{\partial u}{\partial \xi} = \frac{\partial u}{\partial x} \cdot \frac{\partial x}{\partial \xi}
$$

其中 $\frac{\partial x}{\partial \xi}$ 正是雅可比矩阵 $\mathbf{J}$。因此：

$$
\frac{\partial u}{\partial \xi} = \frac{\partial u}{\partial x} \cdot \mathbf{J}
$$

两边同时乘以 $\mathbf{J}^{-1}$：

$$
\frac{\partial u}{\partial x} = \frac{\partial u}{\partial \xi} \cdot \mathbf{J}^{-1}
$$

**矩阵形式**：将上述关系应用于所有积分点，得到：

$$
\frac{\partial \mathbf{u}}{\partial x} = \frac{\partial \mathbf{u}}{\partial \xi} \mathbf{J}^{-1} = \mathbf{D}_V \hat{\mathbf{u}} \mathbf{J}^{-1}
$$

其中 $\mathbf{J}^{-1}$ 为雅可比矩阵的逆。这一步是必要的，因为 Vandermonde 导数矩阵计算的是参考坐标系下的导数，而实际物理问题需要物理坐标系下的导数。

**注意**：在一维情况下，$\mathbf{J}$ 是标量，左乘与右乘等价。但在高维情况下，梯度向量与雅可比矩阵的乘法顺序取决于布局约定（分子布局或分母布局），需要保持一致性。

#### 与微分矩阵的关系

在谱元方法中，常定义微分矩阵 $\mathbf{D}$ 使得 $\frac{\partial \mathbf{u}}{\partial \xi} = \mathbf{D} \mathbf{u}$。通过 Vandermonde 矩阵，可建立关系：

$$
\mathbf{D} = \mathbf{D}_V \mathbf{V}^{-1}
$$

### 1.5 数值求积

#### 高斯-勒让德求积

高斯-勒让德求积是一种高精度的数值积分方法，对于 $2N-1$ 阶及以下的多项式可精确积分。

给定 $N$ 个积分点 $\{x_i\}$ 和对应的积分权重 $\{w_i\}$，近似积分公式为：

$$
\int_{-1}^{1} f(x) \, dx \approx \sum_{i=0}^{N-1} w_i f(x_i)
$$

#### 积分点的确定

高斯-勒让德积分点是勒让德多项式 $P_N(x)$ 的 $N$ 个零点。对于 $N$ 个积分点，多项式阶数为 $N-1$，可精确积分 $2N-1$ 阶多项式。

积分点满足对称性：若 $x_i$ 是积分点，则 $-x_i$ 也是积分点。

#### 积分权重

积分权重由以下公式确定：

$$
w_i = \frac{2}{(1 - x_i^2) [P'_N(x_i)]^2}
$$

或等价地：

$$
w_i = \frac{2}{N P_{N-1}(x_i) P'_N(x_i)}
$$

#### 前几组积分点和权重

|  $N$  | 积分点 $x_i$                                                                                                         | 权重 $w_i$                                                                             |
| :---: | :------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- |
|   1   | $0$                                                                                                                  | $2$                                                                                    |
|   2   | $\pm \frac{1}{\sqrt{3}} \approx \pm 0.5774$                                                                          | $1$                                                                                    |
|   3   | $0, \pm \sqrt{\frac{3}{5}} \approx \pm 0.7746$                                                                       | $\frac{8}{9}, \frac{5}{9}$                                                             |
|   4   | $\pm \sqrt{\frac{3 - 2\sqrt{6/5}}{7}} \approx \pm 0.3399$, $\pm \sqrt{\frac{3 + 2\sqrt{6/5}}{7}} \approx \pm 0.8611$ | $\frac{18 + \sqrt{30}}{36} \approx 0.6521$, $\frac{18 - \sqrt{30}}{36} \approx 0.3479$ |

#### 求积点的计算

积分点可通过求解勒让德多项式的零点获得。牛顿迭代法是常用方法：

$$
x_i^{(k+1)} = x_i^{(k)} - \frac{P_N(x_i^{(k)})}{P'_N(x_i^{(k)})}
$$

初始猜测可使用 Tricomi 渐近公式给出的近似零点（相比切比雪夫节点更接近真根，收敛更快）：

$$
x_i^{(0)} = -\cos\left(\frac{4i + 3}{4N + 2} \pi\right), \quad i = 0, 1, \ldots, N-1
$$

### 1.6 质量矩阵与 L2 投影

在 DG 方法中，质量矩阵在 L2 投影中起核心作用。对于正交归一基函数，质量矩阵为单位矩阵，投影计算可大幅简化。

#### 一般情况

对于非正交基函数，质量矩阵 $\mathbf{M}$ 非对角，需要求解线性方程组：

$$
\mathbf{M} \hat{\mathbf{u}} = \mathbf{f}
$$

其中 $\mathbf{f}$ 为右端项向量。

#### 正交归一基的优势

使用正则勒让德多项式时，质量矩阵为单位矩阵 $\mathbf{M} = \mathbf{I}$，投影简化为：

$$
\hat{\mathbf{u}} = \mathbf{f}
$$

无需矩阵求逆，计算效率显著提高。这是选择正交归一基函数的重要原因。

---

## 二、二维基函数与积分

本节将一维理论推广至二维。二维基函数由两组一维基函数的完全多项式组合（杨辉三角）产生；二维积分点采用一维高斯-勒让德积分点的张量积构造；通过求和分解（Sum-Factorization）技术将二维求和拆分为两次一维操作，大幅降低计算复杂度。

### 2.1 二维基函数：杨辉三角构造

#### 构造方法

在二维参考单元 $[-1, 1]^2$ 上，设两个方向的参考坐标分别为 $r$ 和 $s$。二维正交基函数由两组一维正则勒让德多项式的乘积构成：

$$
\phi_{ij}(r, s) = \tilde{P}_i(r) \, \tilde{P}_j(s), \quad i, j = 0, 1, \ldots, N
$$

其中 $N$ 为每个方向的多项式阶数。由于正则勒让德多项式的正交归一性，二维基函数同样满足正交归一条件：

$$
\int_{-1}^{1} \int_{-1}^{1} \phi_{ij}(r, s) \, \phi_{kl}(r, s) \, dr \, ds = \delta_{ik} \, \delta_{jl}
$$

#### 杨辉三角排序

为便于存储和索引，二维基函数按总阶数 $p = i + j$ 分层排列，即按"杨辉三角"的行顺序存储：

| 总阶数 $p$ | 基函数集合                                                                                     | 基函数数量 |
| :--------: | :--------------------------------------------------------------------------------------------- | :--------: |
|    $0$     | $\tilde{P}_0(r)\tilde{P}_0(s)$                                                                 |    $1$     |
|    $1$     | $\tilde{P}_1(r)\tilde{P}_0(s)$, $\tilde{P}_0(r)\tilde{P}_1(s)$                                 |    $2$     |
|    $2$     | $\tilde{P}_2(r)\tilde{P}_0(s)$, $\tilde{P}_1(r)\tilde{P}_1(s)$, $\tilde{P}_0(r)\tilde{P}_2(s)$ |    $3$     |
|    $p$     | $\{\tilde{P}_{p-j}(r)\tilde{P}_j(s) \mid j = 0, 1, \ldots, p\}$                                |  $p + 1$   |

对于 $p \leq N$ 的完整多项式空间，基函数总数为：

$$
N_{\text{base}} = \sum_{p=0}^{N} (p + 1) = \frac{(N + 1)(N + 2)}{2}
$$

#### 与张量积空间的关系

上述杨辉三角排序将基函数限制在总阶数 $\leq N$ 的完全多项式空间 $\mathcal{P}_N$ 中。与之对应的另一种常见选择是张量积空间 $\mathcal{Q}_N = \mathcal{P}_N \otimes \mathcal{P}_N$，其中 $i$ 和 $j$ 各自独立地取 $0$ 到 $N$，基函数总数为 $(N+1)^2$。

两种空间的关系为 $\mathcal{P}_N \subset \mathcal{Q}_N$。完全多项式空间 $\mathcal{P}_N$ 的基函数更少（$\frac{(N+1)(N+2)}{2}$ 对比 $(N+1)^2$），自由度与精度之比更高。CMeles 采用完全多项式空间 $\mathcal{P}_N$ 作为基函数空间，基函数按杨辉三角排序存储。

#### 索引映射

杨辉三角排序下，基函数按总阶数 $p = i + j$ 逐层排列，每层内按 $j$ 递增。线性索引 $\ell$ 与二维索引 $(i, j)$ 的映射关系为：

$$
\ell = \frac{p(p + 1)}{2} + j, \quad p = i + j
$$

以 $N = 3$（三阶）为例，共 $\frac{4 \times 5}{2} = 10$ 个基函数，存储布局如下：

```
             j →
             0       1       2       3
         ┌───────┬───────┬───────┬───────┐
  p = 0  │ (0,0) │       │       │       │
  i  ↓   │  ℓ=0  │       │       │       │
         ├───────┼───────┼───────┼───────┤
  p = 1  │ (1,0) │ (0,1) │       │       │
         │  ℓ=1  │  ℓ=2  │       │       │
         ├───────┼───────┼───────┼───────┤
  p = 2  │ (2,0) │ (1,1) │ (0,2) │       │
         │  ℓ=3  │  ℓ=4  │  ℓ=5  │       │
         ├───────┼───────┼───────┼───────┤
  p = 3  │ (3,0) │ (2,1) │ (1,2) │ (0,3) │
         │  ℓ=6  │  ℓ=7  │  ℓ=8  │  ℓ=9  │
         └───────┴───────┴───────┴───────┘
```

每层 $p$ 的起始索引为 $\frac{p(p+1)}{2}$，层内第 $j$ 个元素的索引即 $\ell = \frac{p(p+1)}{2} + j$。

逆映射：给定 $\ell$，先求 $p = \left\lfloor \frac{\sqrt{8\ell + 1} - 1}{2} \right\rfloor$，再得 $j = \ell - \frac{p(p+1)}{2}$，$i = p - j$。

> **关键观察**：每层 $p$ 的起始索引恰好是三角数 $T_p = \frac{p(p+1)}{2}$。
>
> 由于 $0 \le j \le p$，$\ell$ 一定落在区间 $[T_p, \, T_p + p] = [T_p, \, T_{p+1} - 1]$ 内。因此 $p$ 就是满足 $T_p \le \ell$ 的最大整数，即 $\frac{p(p+1)}{2} \le \ell$。
>
> 将其视为关于 $p$ 的不等式：$p^2 + p - 2\ell \le 0$。由二次方程求根公式，方程 $p^2 + p - 2\ell = 0$ 的正根为 $p = \frac{-1 + \sqrt{1 + 8\ell}}{2}$。
>
> 由于 $p$ 必须为整数且取满足不等式的最大值，向下取整即可。

### 2.2 二维积分点：张量积构造

#### 构造方法

二维积分点通过两组一维高斯-勒让德积分点的张量积生成。设一维 $N$ 点高斯-勒让德求积的积分点为 $\{r_i\}_{i=0}^{N-1}$、权重为 $\{w_i^r\}_{i=0}^{N-1}$，则二维积分点和权重为：

$$
(r_i, s_j), \quad w_{ij} = w_i^r \, w_j^s, \quad i, j = 0, 1, \ldots, N-1
$$

对于二维参考单元 $[-1, 1]^2$ 上的积分，有：

$$
\int_{-1}^{1} \int_{-1}^{1} f(r, s) \, dr \, ds \approx \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} w_i^r \, w_j^s \, f(r_i, s_j)
$$

由于两个方向使用相同的一维积分规则（$w_i^r = w_i^s$），二维积分点总数为 $N^2$。

#### 积分精度

张量积求积规则可精确积分的多项式阶数与一维情况相同：$N$ 个积分点在每个方向上可精确积分至 $2N - 1$ 阶。因此二维张量积规则可精确积分如下形式的被积函数：

$$
f(r, s) = \sum_{i=0}^{2N-1} \sum_{j=0}^{2N-1} a_{ij} \, r^i \, s^j
$$

#### 存储布局

积分点按列优先顺序存储，线性索引 $k$ 与二维索引 $(i, j)$ 的映射为：

$$
k = j \cdot N_q + i
$$

其中 $i$ 为 $r$ 方向（列）索引，$j$ 为 $s$ 方向（行）索引。第 $k$ 个积分点坐标为 $(r_i, s_j)$。

> **列优先布局的原因**：Eigen 默认采用列优先（column-major）存储。采用列优先布局后，2D Vandermonde 矩阵的第 $\ell$ 列由两个 1D Vandermonde 列向量做外积再按列优先展平得到（$\text{vec}(\mathbf{v}_{i_\ell} \mathbf{v}_{j_\ell}^T)$ 在列优先约定下直接对应 `V1D.col(i_l) * V1D.col(j_l).transpose()` 的 Eigen 默认展平），无需额外的内存重排。同时，列优先布局下积分点 $(r_i, s_j)$ 与基函数 $\phi_{ij}(r,s) = \tilde{P}_i(r)\tilde{P}_j(s)$ 的自然定义完全一致——i 控制 r、j 控制 s，无需任何 i/j 交换说明。

### 2.3 二维 Vandermonde 矩阵

#### 定义

设二维基函数按杨辉三角排序为 $\{\phi_\ell(r, s)\}_{\ell=0}^{N_{\text{base}}-1}$。由 2.2 节列优先存储布局（$k = j \cdot N_q + i$），第 $k$ 个积分点坐标为 $(r_i, s_j)$。二维 Vandermonde 矩阵 $\mathbf{V}_{2D}$ 的大小为 $N_q^2 \times N_{\text{base}}$，定义为：

$$
(\mathbf{V}_{2D})_{k\ell} = \phi_\ell(r_i, s_j) = \tilde{P}_{i_\ell}(r_i) \, \tilde{P}_{j_\ell}(s_j)
$$

其中 $k = j \cdot N_q + i$ 为积分点的线性索引；$\ell$ 为基函数的杨辉三角索引，$(i_\ell, j_\ell)$ 为第 $\ell$ 个基函数对应的二维阶数（$i_\ell$ 对应 $r$ 方向，$j_\ell$ 对应 $s$ 方向）。

#### 模态到节点转换

$$
\mathbf{u} = \mathbf{V}_{2D} \, \hat{\mathbf{u}}
$$

其中 $\hat{\mathbf{u}}$ 为长度 $N_{\text{base}}$ 的模态系数向量，$\mathbf{u}$ 为长度 $N^2$ 的节点值向量。

> **备注**：模态到节点的求值在数学上等价于对基函数的双重求和，同样可应用求和分解（见 2.5 节）拆分为两次一维操作。但 Vandermonde 矩阵-向量乘法形式 $\mathbf{u} = \mathbf{V}_{2D} \hat{\mathbf{u}}$ 表达更加简洁，且可直接调用高度优化的 BLAS 库，在缓存利用和并行化方面通常优于手写的求和分解循环。因此，求和分解主要应用于数值积分的双重求和（见 2.5 节），而 Vandermonde 求值优先采用矩阵乘法形式。

### 2.4 二维 Vandermonde 导数矩阵

#### 定义

二维基函数对参考坐标 $r$ 和 $s$ 的导数分别构成两个 Vandermonde 导数矩阵（沿用列优先布局，$k = j \cdot N_q + i$）：

$$
(\mathbf{D}_{V,r})_{k\ell} = \frac{\partial \phi_\ell}{\partial r}(r_i, s_j) = \tilde{P}'_{i_\ell}(r_i) \, \tilde{P}_{j_\ell}(s_j)
$$

$$
(\mathbf{D}_{V,s})_{k\ell} = \frac{\partial \phi_\ell}{\partial s}(r_i, s_j) = \tilde{P}_{i_\ell}(r_i) \, \tilde{P}'_{j_\ell}(s_j)
$$

#### 导数计算

给定模态系数 $\hat{\mathbf{u}}$，参考坐标下的偏导数为：

$$
\frac{\partial \mathbf{u}}{\partial r} = \mathbf{D}_{V,r} \, \hat{\mathbf{u}}, \quad \frac{\partial \mathbf{u}}{\partial s} = \mathbf{D}_{V,s} \, \hat{\mathbf{u}}
$$

### 2.5 求和分解（Sum-Factorization）

#### 问题背景

二维张量积数值求积的计算公式为：

$$
\int_{-1}^{1} \int_{-1}^{1} g(r, s) \, dr \, ds \approx \sum_{i=0}^{N_q-1} \sum_{j=0}^{N_q-1} w_i \, w_j \, g(r_i, s_j)
$$

其中 $N_q$ 为每方向积分点数。当被积函数 $g$ 含有可分离的基函数乘积结构时（如 DG 弱形式中的体积分），上述双重求和可通过求和分解拆分为两次一维求和，将复杂度从 $O(N_q^2)$ 降低到 $O(N_q)$（每积分点）。

#### 核心思想

若被积函数中与积分点相关的部分可写成两个方向独立因子的乘积，例如 $g(r_i, s_j) = f(r_i, s_j) \cdot A(r_i) \cdot B(s_j)$，则双重求和可以分解：

$$
\sum_i \sum_j w_i w_j \, f(r_i, s_j) \, A(r_i) \, B(s_j)
= \sum_i w_i \, A(r_i) \underbrace{\left[ \sum_j w_j \, B(s_j) \, f(r_i, s_j) \right]}_{\text{先做 } s \text{ 方向求和}}
$$

内层求和对每个 $i$ 独立进行，结果仅为 $r$ 方向的函数，再由外层求和完成 $r$ 方向积分。

#### 应用示例：DG 体积分

在 DG 弱形式的体积分中，需要对每个测试函数 $\phi_{ij}(r, s) = \tilde{P}_i(r) \tilde{P}_j(s)$ 计算：

$$
R_{ij} = \sum_{a=0}^{N_q-1} \sum_{b=0}^{N_q-1} w_a \, w_b \, \mathbf{F}(r_a, s_b) \cdot \nabla \phi_{ij}(r_a, s_b) \, |J(r_a, s_b)|
$$

以 $r$ 方向的梯度分量为例，$\frac{\partial \phi_{ij}}{\partial r} = \tilde{P}'_i(r) \, \tilde{P}_j(s)$，代入后被积函数中基函数部分可分离：

$$
R_{ij}^{(r)} = \sum_{a=0}^{N_q-1} \sum_{b=0}^{N_q-1} w_a \, w_b \, F_r(r_a, s_b) \, |J(r_a, s_b)| \, \tilde{P}'_i(r_a) \, \tilde{P}_j(s_b)
$$

定义 $G(r_a, s_b) = w_a \, w_b \, F_r(r_a, s_b) \, |J(r_a, s_b)|$（物理量和几何因子，已在积分点处求值），则：

$$
R_{ij}^{(r)} = \sum_{a=0}^{N_q-1} \tilde{P}'_i(r_a) \underbrace{\left[ \sum_{b=0}^{N_q-1} G(r_a, s_b) \, \tilde{P}_j(s_b) \right]}_{\tilde{G}_j(r_a)}
$$

**第一步**（$s$ 方向收缩）：对每个 $a$ 和 $j$，计算

$$
\tilde{G}_j(r_a) = \sum_{b=0}^{N_q-1} G(r_a, s_b) \, \tilde{P}_j(s_b)
$$

复杂度为 $O(N_q^2 \cdot N)$。

**第二步**（$r$ 方向收缩）：对每个 $i$ 和 $j$，计算

$$
R_{ij}^{(r)} = \sum_{a=0}^{N_q-1} \tilde{P}'_i(r_a) \, \tilde{G}_j(r_a)
$$

复杂度为 $O(N \cdot N_q \cdot N)$。

总复杂度为 $O(N_q^2 N + N^2 N_q)$，相比直接双重求和的 $O(N_q^2 N^2)$ 降低了一个量级。

$s$ 方向的梯度分量类似，仅需将第一步中的 $\tilde{P}_j(s_b)$ 替换为 $\tilde{P}'_j(s_b)$，第二步中的 $\tilde{P}'_i(r_a)$ 替换为 $\tilde{P}_i(r_a)$。

#### 在 $\mathcal{P}_N$ 空间中的实现

在完全多项式空间中，基函数满足 $i + j \leq N$，按杨辉三角索引 $\ell$ 展平存储。设第 $\ell$ 个基函数的二维阶数为 $(i_\ell, j_\ell)$。以下介绍两种等价的实现方式。

**方式一：按展平索引 $\ell$ 遍历**

直接遍历 $\ell = 0, 1, \ldots, N_{\text{base}} - 1$，以 $r$ 方向梯度为例：

1. **第一步**（$s$ 方向收缩）：对每个 $\ell$ 和每个积分点 $a$，计算

   $$
   \tilde{G}_\ell(r_a) = \sum_{b=0}^{N_q-1} G(r_a, s_b) \, \tilde{P}_{j_\ell}(s_b)
   $$

   结果存入 $N_{\text{base}} \times N_q$ 的临时数组。

2. **第二步**（$r$ 方向收缩）：对每个 $\ell$，计算

   $$
   R_\ell^{(r)} = \sum_{a=0}^{N_q-1} \tilde{P}'_{i_\ell}(r_a) \, \tilde{G}_\ell(r_a)
   $$

   结果直接写入长度 $N_{\text{base}}$ 的输出数组。

此方式代码简洁，输入输出均为 $\ell$ 索引的展平数组，与存储布局一致。

**方式二：按 $(i, j)$ 分组遍历**

将循环按 $i$ 分层，利用具有相同 $i$ 的基函数共享 $r$ 方向基函数值这一事实：

1. **第一步**（$s$ 方向收缩）：对每个 $i = 0, 1, \ldots, N$，先计算公共的 $s$ 方向收缩结果 $\{\tilde{G}_{i,j}(r_a)\}_{j=0}^{N-i}$，再分别乘以 $\tilde{P}'_i(r_a)$（$r$ 方向梯度）或 $\tilde{P}_i(r_a)$（$s$ 方向梯度），完成第二步。

2. 具体地，$r$ 方向和 $s$ 方向梯度的第一步分别为：

   $$
   \tilde{G}_{i,j}^{(r)}(r_a) = \sum_{b} G(r_a, s_b) \, \tilde{P}_j(s_b), \quad
   \tilde{G}_{i,j}^{(s)}(r_a) = \sum_{b} G(r_a, s_b) \, \tilde{P}'_j(s_b)
   $$

   两者使用不同的基函数（值 vs 导数），**不可复用**，需分别计算。

此方式适合在 $i$ 分组内批量处理多个 $j$，便于利用 SIMD 向量化。

> **备注**：方式一和方式二在数学上完全等价，计算量相同。但在 $\mathcal{P}_N$ 空间中，每层 $i$ 对应的 $j$ 数量不同（$N - i + 1$），方式二的分组循环长度不规则，优化收益有限。CMeles 采用方式一，以展平索引 $\ell$ 为主线进行求和分解。

#### 推广至三维

在三维情况下，三重求和可拆分为三次一维操作（$l + m + n \leq N$）：

$$
R_{lmn} = \sum_a \tilde{P}'_l(r_a) \left[ \sum_b \tilde{P}_m(s_b) \left[ \sum_c G(r_a, s_b, t_c) \, \tilde{P}_n(t_c) \right] \right]
$$

复杂度从 $O(N_q^3 N^3)$ 降低到 $O(N_q^3 N + N_q^2 N^2 + N_q N^3)$。

### 2.6 二维坐标变换与雅可比矩阵

#### 四边形单元的坐标变换

坐标变换的目标是建立参考坐标 $(r, s) \in [-1, 1]^2$ 到物理坐标 $(x, y)$ 的映射。与一维情况不同，二维四边形可能具有弯曲边界，无法仅靠角点坐标简单确定映射。因此需要引入**形函数**（shape function）对几何进行插值：

$$
x(r, s) = \sum_{i=1}^{N_{\text{node}}} \phi_i(r, s) \, x_i, \quad y(r, s) = \sum_{i=1}^{N_{\text{node}}} \phi_i(r, s) \, y_i
$$

其中 $(x_i, y_i)$ 为物理单元的几何节点坐标，$\phi_i(r, s)$ 为 Lagrange 插值形函数。形函数的含义是：参考域上任意一点的物理坐标，是所有几何节点坐标的加权平均，权重即形函数在该点的值。

**直边四边形**（4 个角点）使用双线性形函数即可精确描述：

$$
\phi_1 = \frac{(1-r)(1-s)}{4}, \quad \phi_2 = \frac{(1+r)(1-s)}{4}, \quad \phi_3 = \frac{(1+r)(1+s)}{4}, \quad \phi_4 = \frac{(1-r)(1+s)}{4}
$$

对于具有弯曲边界的高阶几何单元，则需要更多几何节点和更高阶的 Lagrange 形函数。

> **坐标变换与解空间的关系**：在经典有限元方法中，"等参"（isoparametric）指的是几何映射与解的逼近使用同一组形函数。CMeles 作为 DG 求解器，二者采用不同的基函数——几何映射使用 Lagrange 插值形函数描述单元形状，解的逼近使用正则勒让德多项式的模态展开。两者各司其职，坐标变换仅提供几何映射关系，与解空间的选取无关。

#### 雅可比矩阵

二维雅可比矩阵为 $2 \times 2$ 矩阵：

$$
\mathbf{J} = \begin{bmatrix}
\displaystyle \frac{\partial x}{\partial r} & \displaystyle \frac{\partial x}{\partial s} \\[8pt]
\displaystyle \frac{\partial y}{\partial r} & \displaystyle \frac{\partial y}{\partial s}
\end{bmatrix}
$$

雅可比行列式为：

$$
|\mathbf{J}| = \frac{\partial x}{\partial r} \frac{\partial y}{\partial s} - \frac{\partial x}{\partial s} \frac{\partial y}{\partial r}
$$

#### 直边四边形的雅可比行列式

将双线性形函数代入坐标变换，物理坐标可整理为双线性形式：

$$
x(r, s) = a_0 + a_1 r + a_2 s + a_3 rs
$$

其中 $a_0 = \frac{\sum x_i}{4}$，$a_1 = \frac{-x_1 + x_2 + x_3 - x_4}{4}$，$a_2 = \frac{-x_1 - x_2 + x_3 + x_4}{4}$，$a_3 = \frac{x_1 - x_2 + x_3 - x_4}{4}$。$y(r, s)$ 的系数 $b_i$ 定义类似。四个偏导数为：

$$
\frac{\partial x}{\partial r} = a_1 + a_3 s, \quad \frac{\partial x}{\partial s} = a_2 + a_3 r, \quad
\frac{\partial y}{\partial r} = b_1 + b_3 s, \quad \frac{\partial y}{\partial s} = b_2 + b_3 r
$$

代入行列式公式展开并整理各项系数（常数项、$r$、$s$、$rs$），得到雅可比行列式为参考坐标的**双线性函数**：

$$
|\mathbf{J}(r, s)| = \frac{1}{16}\left(\alpha + \beta\,s + \gamma\,r + \delta\,rs\right)
$$

其中四个系数均可表示为边向量叉积。定义四边形四条边的边向量为：

$$
\vec{e}_{21} = \overrightarrow{P_1 P_2}, \quad \vec{e}_{32} = \overrightarrow{P_2 P_3}, \quad \vec{e}_{43} = \overrightarrow{P_3 P_4}, \quad \vec{e}_{14} = \overrightarrow{P_4 P_1}
$$

则各系数的几何含义为：

$$
\alpha = \underbrace{(x_3 - x_1)(y_4 - y_2) - (y_3 - y_1)(x_4 - x_2)}_{\text{对角线向量的叉积 } = \, \overrightarrow{P_1 P_3} \times \overrightarrow{P_2 P_4}}
$$

$$
\beta = \underbrace{(x_3 - x_4)(y_1 - y_2) - (y_3 - y_4)(x_1 - x_2)}_{\overrightarrow{P_4 P_3} \times \overrightarrow{P_2 P_1} \;=\; \vec{e}_{43} \times (-\vec{e}_{21})}
$$

$$
\gamma = \underbrace{(x_2 - x_3)(y_4 - y_1) - (y_2 - y_3)(x_4 - x_1)}_{\overrightarrow{P_3 P_2} \times \overrightarrow{P_1 P_4} \;=\; \vec{e}_{32} \times \vec{e}_{14}}
$$

$$
\delta = \underbrace{(x_2 - x_1)(y_4 - y_3) - (y_2 - y_1)(x_4 - x_3)}_{\overrightarrow{P_1 P_2} \times \overrightarrow{P_3 P_4} \;=\; \vec{e}_{21} \times (-\vec{e}_{43})}
$$

> **补充说明**：若网格中所有单元均为平行四边形，雅可比行列式退化为常数，质量矩阵保持对角结构，无需逐单元存储。详见 [平行四边形假设](parallelogram_assumption.md)。

#### 积分变换

物理单元上的面积分变换到参考单元上：

$$
\iint_\Omega f(x, y) \, dx \, dy = \int_{-1}^{1} \int_{-1}^{1} f(x(r,s), y(r,s)) \, |\mathbf{J}(r, s)| \, dr \, ds
$$

利用张量积求积公式数值计算：

$$
\approx \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} w_i \, w_j \, f(x(r_i, s_j), y(r_i, s_j)) \, |\mathbf{J}(r_i, s_j)|
$$

由于雅可比行列式随参考坐标变化，每个积分点处的 $|\mathbf{J}|$ 需分别计算。

#### 导数变换

物理坐标下的偏导数通过雅可比矩阵的逆变换得到：

$$
\begin{bmatrix}
\displaystyle \frac{\partial u}{\partial x} \\[8pt]
\displaystyle \frac{\partial u}{\partial y}
\end{bmatrix}
= \mathbf{J}^{-1}
\begin{bmatrix}
\displaystyle \frac{\partial u}{\partial r} \\[8pt]
\displaystyle \frac{\partial u}{\partial s}
\end{bmatrix}
$$

其中雅可比矩阵的逆为：

$$
\mathbf{J}^{-1} = \frac{1}{|\mathbf{J}|}
\begin{bmatrix}
\displaystyle \frac{\partial y}{\partial s} & \displaystyle -\frac{\partial x}{\partial s} \\[8pt]
\displaystyle -\frac{\partial y}{\partial r} & \displaystyle \frac{\partial x}{\partial r}
\end{bmatrix}
$$

在参考单元上利用 Vandermonde 导数矩阵和求和分解计算 $\frac{\partial u}{\partial r}$ 和 $\frac{\partial u}{\partial s}$ 后，再乘以 $\mathbf{J}^{-1}$ 得到物理坐标下的导数。由于雅可比矩阵在每个积分点处取值不同，$\mathbf{J}^{-1}$ 需逐积分点计算。

#### 逐单元存储与质量矩阵

在一般四边形单元中，雅可比行列式随参考坐标变化，物理单元上的质量矩阵不再是单位矩阵的标量倍：

$$
M_{\ell\ell'} = \iint_{\text{ref}} \phi_\ell(r, s) \, \phi_{\ell'}(r, s) \, |\mathbf{J}(r, s)| \, dr \, ds
$$

质量矩阵为满秩稠密矩阵，每个单元需单独存储其质量矩阵的逆 $\mathbf{M}^{-1}$，用于 L2 投影：

$$
\hat{\mathbf{u}} = \mathbf{M}^{-1} \mathbf{f}
$$

此外，每个单元需存储其几何信息（顶点坐标或双线性系数），以便在积分点处计算 $\mathbf{J}$ 和 $\mathbf{J}^{-1}$。

#### 三角形作为退化四边形

三角形单元可统一纳入四边形框架：将四边形的两个相邻顶点合并为同一点（如令 $P_3 = P_4$），双线性映射退化为三角形映射。此时 $a_3 = \frac{x_1 - x_2}{4}$（一般非零），雅可比行列式仍为双线性函数，逐单元存储策略与一般四边形完全一致。

> **数值注意事项**：退化四边形的雅可比行列式在被合并顶点所在的参考边上为零（如 $P_3 = P_4$ 时 $|\mathbf{J}| \propto (1 - s)$，在 $s = 1$ 处为零）。由于 Gauss-Legendre 积分点不包含端点，积分计算仍然可行，但靠近退化边的积分点处 $|\mathbf{J}|$ 较小，$\mathbf{J}^{-1}$ 元素量级较大，需关注数值精度。实际使用中应避免过于扁平的退化单元。

> **补充说明**：三角形的坐标变换是仿射变换。因此，若采用在三角域下积分正交的基函数（如 Dubiner 基函数），则雅可比行列式为常数，质量矩阵保持对角结构，无需逐单元存储。但是CMeles目前为了简化开发（与此同时统一混合单元的数据结构），统一采用四边形框架。
