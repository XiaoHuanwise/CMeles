# 控制方程与DG场

## 控制方程（可压缩NS方程）

### 有量纲形式可压缩NS方程

可压缩NS方程守恒形式如下

$$\begin{equation}
\frac{\partial \boldsymbol{q}}{\partial t} + \frac{\partial \boldsymbol{f}}{\partial x} + \frac{\partial \boldsymbol{g}}{\partial y} + \frac{\partial \boldsymbol{h}}{\partial z} = 0
\end{equation}$$

其中 $\boldsymbol{q} = \left(\rho,\rho u, \rho v, \rho w, E \right)^{T}$ ， $\boldsymbol{f} = \boldsymbol{f}_{\text{c}} - \boldsymbol{f}_{\text{v}}$ ， $\boldsymbol{g} = \boldsymbol{g}_{\text{c}} - \boldsymbol{g}_{\text{v}}$ ， $\boldsymbol{h} = \boldsymbol{h}_{\text{c}} - \boldsymbol{h}_{\text{v}}$ 。

$\boldsymbol{f}_{\text{c}}$ ， $\boldsymbol{g}_{\text{c}}$ 和 $\boldsymbol{h}_{\text{c}}$ 是对流通量； $\boldsymbol{f}_{\text{v}}$ ， $\boldsymbol{g}_{\text{v}}$ 和 $\boldsymbol{h}_{\text{v}}$ 是粘性通量。

$\boldsymbol{f}_{\text{c}} = \begin{Bmatrix}\rho u\\ p+\rho u^2\\\rho uv\\\rho uw\\ u(E+p)\end{Bmatrix},
\boldsymbol{g}_{\text{c}} = \begin{Bmatrix}\rho v\\\rho uv\\ p+ \rho v^2\\\rho vw\\ v(E+p)\end{Bmatrix},
\boldsymbol{h}_{\text{c}} = \begin{Bmatrix}\rho w\\\rho uw\\ \rho vw\\ p+ \rho w^2\\ w(E+p)\end{Bmatrix}$

其中 $\rho$ 是密度； $u$ ， $v$ 和 $w$ 是速度分量； $x$ ， $y$ 和 $z$ 是空间坐标； $p$ 是压力； $E$ 是总能。考虑理想气体，则

$E = \frac{p}{\gamma-1} + \frac{1}{2}\rho\left(u^2+v^2+w^2\right)$

其中，对于理想气体， $\gamma = 1.4$ 是比热容比。

$\boldsymbol{f}_{\text{v}} = \begin{Bmatrix}0\\\tau _{xx}\\\tau _{yx}\\\tau _{zx}\\ u_{k}\tau _{kx} + \kappa\frac{\partial T}{\partial x}\end{Bmatrix},
\boldsymbol{g}_{\text{v}} = \begin{Bmatrix}0\\\tau _{xy}\\\tau _{yy}\\\tau _{zy}\\ u_{k}\tau _{ky} + \kappa\frac{\partial T}{\partial y}\end{Bmatrix},
\boldsymbol{h}_{\text{v}} = \begin{Bmatrix}0\\\tau _{xz}\\\tau _{yz}\\\tau _{zz}\\ u_{k}\tau _{kz} + \kappa\frac{\partial T}{\partial z}\end{Bmatrix}$

$\tau _{ij} = 2\mu\left[\frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right) - \frac{1}{3}\delta_{ij}\frac{\partial u_k}{\partial x_k}\right]$ 是偏应力张量， $\mu$ 是粘度， $\kappa$ 是导热系数，温度 $T$ 可由克拉珀龙（Clapeyron）方程 $p = \rho RT$ 求得（对于理想气体，$R = 287.058J\cdot kg^{-1}\cdot K^{-1}$）。

> 将 $\tau _{ij}$ 展开可得（速度变量的下标表示偏导）：
  $\tau _{xx} = 2\mu\left(u_x - \frac{u_x+v_y+w_z}{3}\right)$
  $\tau _{yy} = 2\mu\left(u_y - \frac{u_x+v_y+w_z}{3}\right)$
  $\tau _{zz} = 2\mu\left(u_z - \frac{u_x+v_y+w_z}{3}\right)$
  $\tau _{xy} = \tau _{yx} = \mu\left(v_x + u_y\right)$
  $\tau _{xz} = \tau _{zx} = \mu\left(w_x + u_z\right)$
  $\tau _{yz} = \tau _{zy} = \mu\left(w_y + v_z\right)$

对于动力粘度有 $\mu = \mu_{\text{L}} + \mu_{\text{T}} + \mu_{\text{A}}$ 。其中 $\mu_{\text{L}}$ 是层流粘性， $\mu_{\text{T}}$ 是湍流粘性（湍流模型）， $\mu_{\text{A}}$ 是人工粘性（限制器）。

对于理想气体，层流粘性 $\mu_{\text{L}}$ 遵循 Sutherland's Law （萨瑟兰定律）

$\mu_{\text{L}} = \mu_0\left(\frac{T}{T_0}\right)^{3/2}\frac{T_0 + S}{T + S}$

其中参考温度 $T_0=273.15K$ ，参考温度时的参考粘度 $\mu_0=1.716\times 10^{-5}Pa\cdot s$ ，萨瑟兰温度常数 $S=110.4K$ 。

### 无量纲化

引入参考特征量：特征长度 $L$ ，特征速度 $U_{\infty}$ ，特征时间 $t_{\infty}=\frac{L}{U_{\infty}}$ 等各种物理变量的参考量：$\rho_{\infty}, p_{\infty}=\rho_{\infty}U_{\infty}^2, T_{\infty}, \mu_{\infty} = \mu_0\left(\frac{T_{\infty}}{T_0}\right)^{3/2}\frac{T_0 + S}{T_{\infty} + S}, \kappa _{\infty}$。则各无量纲变量形式如下

$$\begin{equation}
\begin{aligned}
&x^* = \frac{x}{L}, y^* = \frac{y}{L}, z^* = \frac{z}{L}, t^* = \frac{U_{\infty}}{L}t, \rho^* = \frac{\rho}{\rho_{\infty}},\\
&p^* = \frac{p}{\rho_{\infty}U_{\infty}^2}, \boldsymbol{u}^*=\frac{\boldsymbol{u}}{U_{\infty}}, T^* = \frac{T}{T_{\infty}}, \mu^* = \frac{\mu}{\mu_{\infty}}, \kappa ^* = \frac{\kappa}{\kappa _{\infty}}
\end{aligned}
\end{equation}$$

代入 (1) 式整理可得无量纲形式

$$\begin{equation}
\frac{\partial \boldsymbol{q}^*}{\partial t^*} + \frac{\partial \left(\boldsymbol{f}_{\text{c}}^* - \frac{1}{Re}\boldsymbol{f}_{\text{v}}^* \right)}{\partial x^*} + \frac{\partial \left(\boldsymbol{g}_{\text{c}}^* - \frac{1}{Re}\boldsymbol{g}_{\text{v}}^* \right)}{\partial y^*} + \frac{\partial \left(\boldsymbol{h}_{\text{c}}^* - \frac{1}{Re}\boldsymbol{h}_{\text{v}}^* \right)}{\partial z^*} = 0
\end{equation}$$

为简化书写，省略 * 上标，默认所有变量皆为无量纲变量（ref: [知乎](https://zhuanlan.zhihu.com/p/552666237 "流体小白学习笔记（二）——可压缩流体基本方程组的无量纲化")）

$$\begin{equation}\boxed{
\frac{\partial \boldsymbol{q}}{\partial t} + \frac{\partial \left(\boldsymbol{f}_{\text{c}} - \frac{1}{Re}\boldsymbol{f}_{\text{v}} \right)}{\partial x} + \frac{\partial \left(\boldsymbol{g}_{\text{c}} - \frac{1}{Re}\boldsymbol{g}_{\text{v}} \right)}{\partial y} + \frac{\partial \left(\boldsymbol{h}_{\text{c}} - \frac{1}{Re}\boldsymbol{h}_{\text{v}} \right)}{\partial z} = 0
}\end{equation}$$

$\boldsymbol{q} = \begin{Bmatrix}\rho\\\rho u\\\rho v\\\rho w\\ E\end{Bmatrix},
\boldsymbol{f}_{\text{c}} = \begin{Bmatrix}\rho u\\ p+ \rho u^2\\\rho uv\\\rho uw\\ u(E+p)\end{Bmatrix},
\boldsymbol{g}_{\text{c}} = \begin{Bmatrix}\rho v\\\rho uv\\ p+ \rho v^2\\\rho vw\\ v(E+p)\end{Bmatrix},
\boldsymbol{h}_{\text{c}} = \begin{Bmatrix}\rho w\\\rho uw\\\rho vw\\ p+ \rho w^2\\ w(E+p)\end{Bmatrix}$
$\boldsymbol{f}_{\text{v}j} = \begin{Bmatrix}0\\\tau _{xj}\\\tau _{yj}\\\tau _{zj}\\ u_{k}\tau _{kj} + \frac{\kappa}{Pr\left(\gamma -1\right)Ma_{\infty}^{2}}\frac{\partial T}{\partial x_j}\end{Bmatrix}$

对于状态方程有 $p = \frac{RT_{\infty}}{U_{\infty}^2}\rho T = \frac{\rho T}{\frac{\gamma U_{\infty}^2}{\gamma RT_{\infty}}} = \frac{\rho T}{\gamma \frac{U_{\infty}^2}{a_{\infty}^2}} = \frac{\rho T}{\gamma Ma_{\infty}^2}$

对于无量纲萨瑟兰公式有 $\mu_{\text{L}} = T^{\frac{3}{2}}\frac{1 + S/T_{\infty}}{T + S/T_{\infty}}$

对于无量纲的湍流黏性和人工黏性有 $\mu_{\text{T}}^* = Re \cdot\mu_{\text{T}}(\boldsymbol{q}^*), \mu_{\text{A}}^* = Re \cdot\mu_{\text{A}}(\boldsymbol{q}^*)$。即，**将无量纲变量代入有量纲的湍流黏性和人工黏性公式中，得到的结果再乘以雷诺数 $Re$ 即为无量纲湍流黏性和人工黏性**。

> **附加(人工、湍流)黏性无量纲推导**：
>
> 以基于速度散度的激波捕捉人工粘性为例。
>
> **有量纲公式**（二阶人工粘性）：
>
> $$
> \mu_{\text{art}} = C\,\rho\,|\nabla\!\cdot\!\mathbf{u}|\,h^2,
> $$
>
> $C$ 为无量纲常数，$h$ 为当地网格尺寸。
>
> **无量纲推导**：
>
> $$
> \begin{aligned}
> \mu_{\text{art}} &= C\,(\rho_0\rho^*) \left(\frac{U}{L}|\nabla^*\!\cdot\!\mathbf{u}^*|\right) (L\Delta x^*)^2 \\
> &= C\,\rho_0 U L\; \rho^* |\nabla^*\!\cdot\!\mathbf{u}^*| (\Delta x^*)^2.
> \end{aligned}
> $$
>
> 除以参考粘度 $\mu_0 = \dfrac{\rho_0 U L}{Re}$，得无量纲人工粘性系数：
>
> $$
> \boxed{
> \mu_{\text{art}}^* = \frac{\mu_{\text{art}}}{\mu_0}
> = C\,Re\; \rho^* |\nabla^*\!\cdot\!\mathbf{u}^*| (\Delta x^*)^2.
> }
> $$
>
> > **直观总结：无量纲人工粘性公式的形式，相当于把无量纲数直接代入有量纲公式，再把结果乘上雷诺数 $\mathrm{Re}$。**
> > 这正是量纲一致性所要求的：因为有量纲公式给出的量具有动力粘度量纲，而我们在无量纲方程中需要的 $\mu_{\text{art}}^*$ 是以 $\mu_0$ 为尺度的度量，两者之间正好相差一个由参考量组合决定的 $Re$ 因子。
>
> 对于其他常见附加粘性模型，结论完全相同：
> - 声速型：$\mu_{\text{art}} = C \rho c h \;\Longrightarrow\; \mu_{\text{art}}^* = C\,Re\, \rho^* c^* \Delta x^*$;
> - 应变率型（Smagorinsky 型，湍流黏性）：$\mu_{\text{art}} = \rho (C_s \Delta)^2 |S| \;\Longrightarrow\; \mu_{\text{art}}^* = C_s^2\,Re\, \rho^* (\Delta x^*)^2 |S^*|$.

$$\begin{equation}\boxed{
\frac{\partial \boldsymbol{q}}{\partial t} + \frac{\partial \boldsymbol{f}}{\partial x} + \frac{\partial \boldsymbol{g}}{\partial y} + \frac{\partial \boldsymbol{h}}{\partial z} = 0,
\boldsymbol{f}_j = \boldsymbol{f}_{\text{c}j} - \frac{1}{Re}\boldsymbol{f}_{\text{v}j}
}\end{equation}$$

为了便于方程的离散，将方程进一步概括为

$$\begin{equation}
\frac{\partial \boldsymbol{q}}{\partial t} = R\left(\boldsymbol{q}\right) = -\nabla \cdot \boldsymbol{F} = -\nabla \cdot \left[\boldsymbol{F}_{\text{c}}\left(\boldsymbol{q}\right) - \boldsymbol{F}_{\text{v}}\left(\boldsymbol{q},\nabla\boldsymbol{q}\right)\right]
\end{equation}$$

其中对流通量 $\boldsymbol{F}_{\text{c}} = \left[\boldsymbol{f}_{\text{c}},\boldsymbol{g}_{\text{c}},\boldsymbol{h}_{\text{c}}\right]$ ，粘性通量 $\boldsymbol{F}_{\text{v}} = \frac{1}{Re}\left[\boldsymbol{f}_{\text{v}},\boldsymbol{g}_{\text{v}},\boldsymbol{h}_{\text{v}}\right]$。

## 离散化

### 空间离散

将计算域 $\mathit{\Omega}$ 划分为若干个不重叠的单元 $\mathit{\Omega}_e$ 。定义测试函数 $\phi _q$ 并对式 (6) 进行分部积分，得到其DG弱形式

$$\begin{equation}
\int\limits_{\mathit{\Omega}_e} \frac{\partial \boldsymbol{q}_p}{\partial t} \phi _q \, \mathrm{d}\mathit{\Omega}_e = \int\limits_{\mathit{\Omega}_e} \boldsymbol{F} \cdot \nabla\phi _q \, \mathrm{d}\mathit{\Omega}_e - \int\limits_{\mathit{\Gamma}_e} \hat{\boldsymbol{F}}\left(\boldsymbol{q}_p^{-},\boldsymbol{q}_p^{+},\boldsymbol{n}\right) \phi _q \, \mathrm{d}\mathit{\Gamma}_e
\end{equation}$$

其中 $\boldsymbol{q}_p$ 是 $\boldsymbol{q}$ 在分片多项式空间中的近似。 $\boldsymbol{q} \approx \boldsymbol{q}_p = \sum_{j=1}^{o+1}\boldsymbol{u}_j \phi _j$ ，$o$ 是多项式阶数， $\boldsymbol{u}_j$ 是基函数 $\phi _j$ 的系数向量。此外，测试函数 $\phi _q$ 应取遍所有基函数 $\phi _j$ 。 $\mathit{\Gamma}_e$ 是 $\mathit{\Omega}_e$ 的边界， $\hat{\boldsymbol{F}}$ 是用于处理 $\mathit{\Gamma}_e$ 上内变量 $\boldsymbol{q}_p^{-}$ 和外变量 $\boldsymbol{q}_p^{+}$ 之间的不连续的数值通量。 $\boldsymbol{n}$ 是 $\mathit{\Gamma}_e$ 上的外法向量。

数值通量的对流项可以通过黎曼求解器计算法向通量得到。类似的，数值通量的粘性项可以通过局部间断伽辽金法（LDG）或者內罚函数法（IPDG）获得。

黎曼求解器的计算可以应用Euler方程的旋转不变性减少计算量，详见[网格与几何文档 1.3 节](mesh_and_geometry.md#13-法向量)。

考虑测试函数取遍所有基函数（基函数在[基函数文档](basis_functions.md)中有介绍）

$$\begin{equation}
\int\limits_{\mathit{\Omega}_e} \phi _i\phi _j \, \mathrm{d}\mathit{\Omega}_e \frac{\mathrm{d} \boldsymbol{u}_j}{\mathrm{d} t} = \boldsymbol{M} \frac{\mathrm{d} \boldsymbol{u}_j}{\mathrm{d} t} = \int\limits_{\mathit{\Omega}_e} \boldsymbol{F} \cdot \nabla\phi _i \, \mathrm{d}\mathit{\Omega}_e - \int\limits_{\mathit{\Gamma}_e} \hat{\boldsymbol{F}} \phi _i \, \mathrm{d}\mathit{\Gamma}_e
\end{equation}$$

进一步可得ode形式

$$\begin{equation}
\frac{\mathrm{d} \boldsymbol{u}_j}{\mathrm{d} t} = \boldsymbol{M}^{-1} \left[\int\limits_{\mathit{\Omega}_e} \boldsymbol{F} \cdot \nabla\phi _i \, \mathrm{d}\mathit{\Omega}_e - \int\limits_{\mathit{\Gamma}_e} \hat{\boldsymbol{F}} \phi _i \, \mathrm{d}\mathit{\Gamma}_e\right] = \mathcal{R}\left(\boldsymbol{u}_j\right)
\end{equation}$$

对于单个单元， $\boldsymbol{u}_j$ 是一个二阶张量 $u_{kj}$ 。考虑整个求解域中的所有单元， $\boldsymbol{u}_j$ 应当是一个三阶张量 $u_{lkj}$ 。其中 $l$ 是单元编号， $k$ 是变量编号， $j$ 是基函数编号。在实际运算中， $\boldsymbol{u}_j$ 展平（flatten）并存储为列向量 $\boldsymbol{u}_{\text{f}}$ 。

$$\begin{equation}\boxed{
\frac{\mathrm{d} \boldsymbol{u}_{\text{f}}}{\mathrm{d} t} = \mathcal{R}\left(\boldsymbol{u}_{\text{f}}\right)
}\end{equation}$$

### 时间离散

#### 大致思路

考虑到边界条件可能显式地依赖于时间 $t$ ，将式 (10) 进一步改写为更一般的形式

$$\begin{equation}\boxed{
\frac{\mathrm{d} \boldsymbol{u}_{\text{f}}}{\mathrm{d} t} = \mathcal{R}\left(t,\boldsymbol{u}_{\text{f}}\right)
}\end{equation}$$

其中 $t\in\left[0,\infty\right]$ 且 $\boldsymbol{u}_{\text{f}} \in\mathbb{R}^{N}$ 。

这个由DG方法得到的一阶ODE可以被视为一个一般的一阶ODE。

ODE的时间推进部分详见[时间推进文档](time_marching.md)。

## DG场模块

DG场模块用于根据[基函数模块](basis_functions.md)和[网格与几何模块](mesh_and_geometry.md)计算半离散控制方程的右端项。其核心任务是对每个单元计算式 (9) 定义的残差算子 $\mathcal{R}(\boldsymbol{u}_j)$，即体积分与面积分的差值左乘逆质量矩阵。

### 数据结构

#### 场数据的三级张量结构

DG 方法的解以基函数模态系数的方式存储。对于整个计算域，解 $\boldsymbol{u}$ 是一个三级张量 $u_{lkj}$，其中：

| 索引  |     符号     | 含义                              | 取值范围                             |
| :---: | :----------: | :-------------------------------- | :----------------------------------- |
|  $l$  |   单元编号   | 网格中第 $l$ 个单元               | $0, 1, \ldots, N_{\text{elem}} - 1$  |
|  $k$  | 守恒变量编号 | 第 $k$ 个守恒变量                 | $0, 1, \ldots, N_{\text{vars}} - 1$  |
|  $j$  |  基函数编号  | 第 $j$ 个模态系数（杨辉三角索引） | $0, 1, \ldots, N_{\text{modes}} - 1$ |

其中 $N_{\text{elem}}$ 为网格单元总数。对于三维可压缩 NS 方程，守恒变量数 $N_{\text{vars}} = 5$（$\rho, \rho u, \rho v, \rho w, E$）；对于二维情况，$N_{\text{vars}} = 4$（$\rho, \rho u, \rho v, E$）。

基函数模态数 $N_{\text{modes}}$ 由多项式阶数 $N$ 和空间维数决定：

- **二维**（完全多项式空间 $\mathcal{P}_N$）：$N_{\text{modes}} = \dfrac{(N+1)(N+2)}{2}$
- **三维**（完全多项式空间 $\mathcal{P}_N$）：$N_{\text{modes}} = \dfrac{(N+1)(N+2)(N+3)}{6}$

> **约定**：下文统一以 $N_{\text{vars}}$、$N_{\text{modes}}$ 和 $N_q$ 分别表示守恒变量数、基函数模态数和每方向数值积分点数。由于当前不使用 `@shared` 共享内存，这些量在 OCCA 核函数中作为 `const int` 参数传入而非 JIT 编译宏，以最大化 JIT 缓存命中率（详见 [OCCA JIT 编译时的宏定义](#occa-jit-编译时的宏定义)）。

#### Element–Value–Mode 存储布局

在物理存储中，三级张量 $u_{lkj}$ 按 **Element（单元）→ Value（守恒变量）→ Mode（基函数模态系数）** 的层次展平为列向量 $\boldsymbol{u}_{\text{f}} \in \mathbb{R}^{N_{\text{elem}} \cdot N_{\text{vars}} \cdot N_{\text{modes}}}$。

展平顺序为（最内层索引优先）：

```
// 逻辑结构：u[elem][var][mode]
// 展平后的线性索引：
index = elem * (N_vars * N_modes) + var * N_modes + mode
```

直观图示如下（以单个单元为例）：

```
单元 0:
  var 0 (ρ):        [mode₀, mode₁, mode₂, ..., mode_{N_modes-1}]
  var 1 (ρu):       [mode₀, mode₁, mode₂, ..., mode_{N_modes-1}]
  ...
  var N_vars-1 (E): [mode₀, mode₁, mode₂, ..., mode_{N_modes-1}]
单元 1:
  var 0 (ρ):        [mode₀, mode₁, mode₂, ..., mode_{N_modes-1}]
  ...
```

此布局的优点：
1. **单元连续**：同一单元的所有变量和模态系数在内存中连续存储，有利于单元级并行；
2. **变量对齐**：同一变量的所有模态系数连续，便于向量化加载/存储；
3. **与 OCCA 核函数兼容**：展平后的一维数组可直接映射到 OCCA 的 `occa::memory`，在 OKL 核函数中以全局指针形式访问。

> **与公式 (9) 的关系**：在公式 (9) 中，$\boldsymbol{u}_j$（对单个单元）以 $j$（基函数编号）为主索引。但在存储实现中，每个单元内的数据以 `(var, mode)` 顺序排列，即先按守恒变量分层、每层内按模态展开。在 OCCA 核函数中，偏移量由核函数参数 `N_VARS` 和 `N_MODES`（均为 `const int`，托管端传入的值）计算访问 $u[l][k][j]$。

#### OCCA JIT 编译时的宏定义

由于当前不使用 `@shared` 共享内存，`N_VARS`、`N_MODES`、`N_Q` 等维度参数均作为核函数参数传入，不作为编译期宏。这能最大化 JIT 缓存命中率——只要核函数源码不变，即使网格规模和多项式阶数变化也无需重新编译。

> 后续若启用 `@shared`（如将 Vandermonde 矩阵加载到共享内存），则 `@shared` 数组大小必须为编译时常量，届时需要将这些维度通过宏定义传入。OCCA 的 JIT 缓存以源码 + 宏定义组合作为 key，宏参数组合的不同取值会产生独立的缓存条目。

#### 场数据的设备端存储

在 OCCA 设备（GPU 或 CPU）上，场数据使用 `occa::memory` 对象管理。主要存储以下数组：

| 数组  | 长度                                                             | 说明                                          |
| :---- | :--------------------------------------------------------------- | :-------------------------------------------- |
| `u_f` | $N_{\text{elem}} \times N_{\text{vars}} \times N_{\text{modes}}$ | 守恒变量的模态系数（主解向量）                |
| `res` | $N_{\text{elem}} \times N_{\text{vars}} \times N_{\text{modes}}$ | 残差 $\mathcal{R}(\boldsymbol{u}_{\text{f}})$ |

辅助数组（可选驻留于设备端）：

| 数组     | 长度                                                                    | 说明                                   |
| :------- | :---------------------------------------------------------------------- | :------------------------------------- |
| `u_q`    | $N_{\text{elem}} \times N_{\text{vars}} \times N_Q^2$                   | 积分点上的守恒变量节点值               |
| `grad_u` | $N_{\text{elem}} \times N_{\text{vars}} \times \text{dim} \times N_Q^2$ | 积分点上的守恒变量梯度（粘性通量需要） |

#### OCCA 共享内存分块形状

##### OKL 语法规则

OKL 使用 `@outer` / `@inner` 标注循环的并行层次，映射到 GPU 的 block/thread（OpenCL 的 work-group/work-item）：

- `@outer` / `@inner` 是**裸属性**（`for (...; @outer)`），不用括号或数字。可写在 `for` 的尾部（`for (...; @outer)`）或单独一行放在 `for` 前面（`@outer\nfor (...)`），两种风格语义等价（见 `third_party/occa/examples/` 中的 `.okl` 文件）。
- **`@outer` 和 `@inner` 支持嵌套**——嵌套 `@outer` 对应多维 block grid（如 `fd2d.okl`），嵌套 `@inner` 对应多维 thread grid（类似 CUDA 的 `threadIdx.x/y/z`）。CMeles 目前不考虑多维嵌套用法，仅使用单级 `@outer` 和单级 `@inner`。
- **`@outer` 和 `@inner` 之间可包含无标注的普通 `for` 循环**——这些循环在 `@outer` 外由 host 执行（计算边界等），在 `@outer` 内、`@inner` 外由 device 端串行执行（被块内所有线程冗余执行或随 `@inner` 线程分工后各自串行推进）。

##### CMeles 的分块策略

所有后端（GPU CUDA/HIP/OpenCL、CPU OpenMP/Serial）使用统一的 `@tile` 分块策略，无需宏定义切换不同代码路径：

- 对 `elem` 使用 `@tile(TILE_SIZE, @outer, @inner)` 进行分块遍历。`@tile` 将原始的 `for (int elem = 0; elem < N_elem; ++elem)` 循环自动拆分为两级（参考 `third_party/occa/docs/guide/okl/attributes.md`）：
  - 外层 `@outer`：`iTile` 循环，步长为 `TILE_SIZE`，GPU 映射到 block，OpenMP 映射到 `#pragma omp parallel for`
  - 内层 `@inner`：`elem = iTile` 到 `iTile + TILE_SIZE`，GPU 映射到 thread，OpenMP/Serial 退化为普通串行循环
  - 越界保护 `if (elem < N_elem)` 由 `@tile` 默认自动插入
- **一个 elem 内的全部计算为串行**（含 var×mode 遍历、积分点双重循环、通量与梯度基函数的收缩），无 `@shared` 共享内存——所有数据读写走全局内存。
- **SIMD 向量化**：不依赖 OKL 的 `@inner` 标注。实际验证表明，OCCA OpenMP 后端对 `@inner` 不生成任何 `#pragma omp simd` 指令。SIMD 向量化完全由编译器在 `-O3 -march=native` 下自动完成，OKL 源码中无需额外标注。
- `TILE_SIZE` 固定为 256，所有后端统一使用同一值。
- 此设计的优点：实现简单、OKL 源码跨平台通用、elem 间完全独立无同步开销；缺点：未利用 `@shared` 复用通量数据。后续可将 Vandermonde 矩阵等跨 elem 不变的小矩阵加载到 `@shared` 中以减少冗余全局访存（Vandermonde 矩阵规模跨度大——$N=3$ 时 $10 \times 16$，$N=12$ 时 $91 \times 169$，量级 $O(10)$ 到 $O(10^4)$，需谨慎评估 `@shared` 容量），当前优先实现全局内存版本。

对应的 OKL 伪代码结构（所有后端统一）：

```c
// 体积分核函数（全局内存版本，@tile 分块，所有后端通用）
@kernel void volumeIntegral(const int N_elem,
                             const double *u,
                             const double *geoF,
                             double *res) {
  // @tile(TILE_SIZE, @outer, @inner) 自动展开为：
  //   for (int iTile = 0; iTile < N_elem; iTile += TILE_SIZE; @outer)
  //     for (int elem = iTile; elem < iTile + TILE_SIZE; ++elem; @inner)
  //       if (elem < N_elem) { ... }   // 越界保护由 @tile 自动插入
  for (int elem = 0; elem < N_elem; ++elem; @tile(TILE_SIZE, @outer, @inner)) {

    // 一个 elem 内的全部计算，串行执行（无 @shared 共享内存）
    for (int var = 0; var < N_VARS; ++var) {
      for (int mode = 0; mode < N_MODES; ++mode) {
        // 模态→节点求值（Vandermonde 矩阵-向量乘）
        // 物理通量计算（逐积分点）
        // 通量与梯度基函数的收缩（当前为稠密收缩，见下文 Step 3）
        // 结果写入 res[elem * N_VARS * N_MODES + var * N_MODES + mode]
      }
    }
  }
}
```

> **后续优化方向**：
> - 对于规模较小的多项式（$N \le 4$），可将 Vandermonde 矩阵和导数矩阵加载到 `@shared` 中，所有 elem 共享。
> - 对于粘性项路径，可增设 `@shared` 数组缓存 `grad_u` 以减少全局访存。

**面积分核函数同理（computeFaceFlux）**：

```c
@kernel void computeFaceFlux(const int N_face, ...) {
  for (int face = 0; face < N_face; ++face; @tile(TILE_SIZE, @outer, @inner)) {
    // @tile 自动拆分循环层级并插入越界保护（同上）
    // 一个面单元内的全部计算，串行执行
    for (int var = 0; var < N_VARS; ++var) {
      for (int mode = 0; mode < N_MODES; ++mode) {
        // 加载左右两侧节点值
        // Riemann 求解器求数值通量
        // 沿面积分点累加 → face_flux[face][var][mode]
      }
    }
  }
}
```

##### 分块参数总览

| 后端           | `@outer` 绑定                                                         | `@inner` 绑定                                | 共享内存           |
| :------------- | :-------------------------------------------------------------------- | :------------------------------------------- | :----------------- |
| GPU (CUDA/HIP) | `@tile` 拆分后的 elem tile（256/block）                               | `@tile` 拆分后的 elem（1/thread）            | 无（全局内存版本） |
| OpenMP         | `@tile` 拆分后的 elem tile（`@tile` 自动产生的 `@outer`）             | `@tile` 拆分后的 elem（`@inner` 退化为串行） | 无                 |
| Serial         | `@tile` 拆分后的 elem tile（`@tile` 自动产生的 `@outer`，退化为串行） | `@tile` 拆分后的 elem（退化为串行）          | 无                 |

**分块形状的编译时确定**：

当前全局内存版本仅需 `TILE_SIZE = 256` 为编译时常量（`@tile` 的第一个参数必须为编译时常量，因为 `@inner` 的迭代次数需要在编译时确定）。循环边界（`N_ELEM`、`N_VARS`、`N_MODES`、`N_Q`）均通过核函数参数传入，无需 JIT 宏——这最大化 JIT 缓存命中率。

### 函数设置

DG 场模块按功能分为初始化、体积分、面积分和右端项组装四个子模块。

#### （一）初始化：初始条件到模态系数

给定初始条件函数 $\boldsymbol{q}_0(x, y)$（如自由流条件、激波管初始间断等），需将其投影到各单元的模态系数空间中：

1. 对每个单元，在所有积分点 $(r_i, s_j)$ 处求值物理坐标 $(x_{ij}, y_{ij})$；
2. 计算初始条件在积分点处的节点值 $\boldsymbol{q}_0(x_{ij}, y_{ij})$；
3. 通过 L2 投影获得模态系数。物理单元上的质量矩阵为 $M_{\ell\ell'} = \iint_{\text{ref}} \phi_\ell(r,s)\phi_{\ell'}(r,s)|\mathbf{J}(r,s)|\,dr\,ds$。右端项为 $\mathbf{f} = \mathbf{V}_{2D}^T \mathbf{W}_{2D} \mathbf{J}_{2D} \mathbf{q}_k^{\text{(nodal)}}$，其中 $\mathbf{V}_{2D}$ 为二维 Vandermonde 矩阵，$\mathbf{W}_{2D}$ 为求积权重对角矩阵，$\mathbf{J}_{2D}$ 为各积分点处 $|\mathbf{J}(r_i,s_j)|$ 构成的对角矩阵。求解 $\mathbf{M}\hat{\boldsymbol{u}} = \mathbf{f}$ 即得模态系数。

   由于 $|\mathbf{J}(r,s)|$ 随参考坐标变化，一般四边形单元下 $\mathbf{M}$ 为 $N_{\text{modes}} \times N_{\text{modes}}$ 的满秩矩阵（规模通常很小，如 $N=3$ 时仅 $10 \times 10$），其逆可在预处理阶段逐单元预计算并存储。

对于多项式阶数 $N$ 小于初始条件中所含最高频率分量的情况，L2 投影已自动完成滤波。

#### （二）体积分计算

体积分计算 $\displaystyle \int_{\Omega_e} \boldsymbol{F} \cdot \nabla\phi_q \, \mathrm{d}\Omega_e$，对每个单元遍历全部 $N_{\text{vars}} \times N_{\text{modes}}$ 个试函数。核心步骤如下：

**Step 1 — 模态到节点求值**：将模态系数 $\hat{\boldsymbol{u}}$ 通过 Vandermonde 矩阵变换到积分点上的节点值：

$$
\boldsymbol{q}(r_i, s_j) = \sum_{\ell=0}^{N_{\text{modes}}-1} \hat{\boldsymbol{u}}_\ell \, \phi_\ell(r_i, s_j)
$$

这一步可通过 BLAS 矩阵乘法高效实现（见[基函数文档 2.3 节](basis_functions.md#23-二维-vandermonde-矩阵)）。

**Step 2 — 物理通量计算**：在每个积分点上，由节点值 $\boldsymbol{q}$ 和梯度 $\nabla\boldsymbol{q}$ 计算物理通量 $\boldsymbol{F} = \boldsymbol{F}_c(\boldsymbol{q}) - \frac{1}{Re}\boldsymbol{F}_v(\boldsymbol{q}, \nabla\boldsymbol{q})$。

- 对流通量：由守恒变量直接计算（见式 (3)–(5) 定义）；
- 粘性通量：需先在积分点上计算守恒变量的梯度（见下文"粘性项处理"），再代入偏应力张量和热通量公式；
- 梯度计算同样通过 Vandermonde 导数矩阵和链式法则完成：$\frac{\partial \boldsymbol{q}}{\partial x} = \mathbf{D}_{V,r} \hat{\boldsymbol{u}} \cdot \frac{\partial r}{\partial x} + \mathbf{D}_{V,s} \hat{\boldsymbol{u}} \cdot \frac{\partial s}{\partial x}$。

**加权物理通量 $\boldsymbol{H}_r$、$\boldsymbol{H}_s$（Step 2 → Step 3 的衔接）**：分部积分后体积分中的导数转移到试函数上，$R_{\text{vol}}[\ell] = \iint_{\Omega_e} \left( \boldsymbol{f}\,\partial\phi_\ell/\partial x + \boldsymbol{g}\,\partial\phi_\ell/\partial y \right) \mathrm{d}x\,\mathrm{d}y$。基函数定义在参考单元上，物理梯度经链式法则展开（逆雅可比分量 $\text{Jinv}_{11} = \partial r/\partial x$ 等的定义见[网格文档 2.4 节](mesh_and_geometry.md#24-雅可比矩阵与行列式)）：

$$
\frac{\partial\phi}{\partial x} = \text{Jinv}_{11}\,\frac{\partial\phi}{\partial r} + \text{Jinv}_{21}\,\frac{\partial\phi}{\partial s}, \qquad
\frac{\partial\phi}{\partial y} = \text{Jinv}_{12}\,\frac{\partial\phi}{\partial r} + \text{Jinv}_{22}\,\frac{\partial\phi}{\partial s}
$$

代入弱形式并按参考方向归并：

$$
\boldsymbol{f}\,\frac{\partial\phi}{\partial x} + \boldsymbol{g}\,\frac{\partial\phi}{\partial y}
= \underbrace{\left(\boldsymbol{f}\,\text{Jinv}_{11} + \boldsymbol{g}\,\text{Jinv}_{12}\right)}_{\tilde{\boldsymbol{f}}}
\,\frac{\partial\phi}{\partial r}
+ \underbrace{\left(\boldsymbol{f}\,\text{Jinv}_{21} + \boldsymbol{g}\,\text{Jinv}_{22}\right)}_{\tilde{\boldsymbol{g}}}
\,\frac{\partial\phi}{\partial s}
$$

两个括号即 $\mathbf{J}^{-1}\boldsymbol{F}$ 的 $r/s$ 分量——参考坐标系下的通量（$\tilde{\boldsymbol{f}} = \boldsymbol{F}\cdot\nabla r$ 为穿过 $r=\text{const}$ 面的通量密度），只依赖积分点而与试函数无关。将求积权重与雅可比行列式一并折叠（$\lambda_{WJ}[q] = w_q\,|\mathbf{J}|_q$，即 $\boldsymbol{\Lambda}_{wJ}$ 的对角元，由几何模块逐积分点提供），得到**加权物理通量**：

$$
\boldsymbol{H}_r[q] = \lambda_{WJ}[q] \left( \boldsymbol{f}\,\text{Jinv}_{11} + \boldsymbol{g}\,\text{Jinv}_{12} \right)_q, \qquad
\boldsymbol{H}_s[q] = \lambda_{WJ}[q] \left( \boldsymbol{f}\,\text{Jinv}_{21} + \boldsymbol{g}\,\text{Jinv}_{22} \right)_q
$$

于是体积分（$\mathrm{d}x\,\mathrm{d}y = |\mathbf{J}|\,\mathrm{d}r\,\mathrm{d}s$，权重与行列式已折入 $\lambda_{WJ}$）化为纯基函数收缩，直接给出 Step 3 的离散形式：

$$
R_{\text{vol}}[\ell] = \sum_{q=0}^{N_q^2-1} \left[ \boldsymbol{H}_r[q]\, \mathrm{dV2D}_r[q][\ell] + \boldsymbol{H}_s[q]\, \mathrm{dV2D}_s[q][\ell] \right]
$$

这样组织的目的是把所有与 $\ell$ 无关的量（通量、几何、权重）折叠进逐点预计算的 $\boldsymbol{H}_r$、$\boldsymbol{H}_s$，使 mode 循环只剩纯基函数运算。

**Step 3 — 通量与梯度基函数的收缩（当前为稠密收缩）**：当前实现按上式对全部 $N_q^2$ 个积分点直接求和，等价于 $\boldsymbol{H}_r$、$\boldsymbol{H}_s$ 与二维导数 Vandermonde 矩阵的稠密矩阵–向量乘，复杂度为 $O(N_q^2 \cdot N_{\text{modes}})$（每变量每方向）。

由于 $\mathrm{dV2D}_r(q,\ell) = \tilde{P}'_{i_\ell}(r_a)\,\tilde{P}_{j_\ell}(s_b)$（积分点 $q$ 对应 $(r_a, s_b)$，模式 $\ell$ 对应 $(i_\ell, j_\ell)$）具有可分离的张量积结构，上式可进一步用**求和分解**（Sum-Factorization）拆分为两次一维收缩：

$$
R_{ij}^{(r)} = \sum_{a=0}^{N_q-1} \tilde{P}'_i(r_a) \underbrace{\left[ \sum_{b=0}^{N_q-1} G(r_a, s_b) \, \tilde{P}_j(s_b) \right]}_{\text{第一步：} s \text{ 方向收缩}},\quad G(r_a,s_b) = w_a w_b \, F_r(r_a,s_b) \, |J(r_a,s_b)|
$$

总复杂度降为 $O\big((N{+}1)\,N_q^2 + N_{\text{modes}}\,N_q\big)$，实现前提是中间量按一维下标 $j$ 索引并在 $j_\ell = j$ 的各 $\ell$ 间复用（详见[基函数文档 2.5 节](basis_functions.md#25-求和分解sum-factorization)）。对 $s$ 方向梯度的处理同理（$\tilde{P}_j \to \tilde{P}'_j$、$\tilde{P}'_i \to \tilde{P}_i$）。

> **当前选择稠密收缩、求和分解列为后续优化方向**，理由如下：
> 1. **矩阵复用**：稠密收缩直接复用已上传的 $\mathbf{V}_{2D}$、$\mathrm{dV2D}_r$、$\mathrm{dV2D}_s$（Step 1 的模态–节点变换与后续梯度计算同样需要）；求和分解则需向核函数额外传入一维矩阵 $\mathbf{V}_{1D}$、$\mathrm{dV1D}$ 与模态阶数索引 $(i_\ell, j_\ell)$ 表。
> 2. **核函数无需感知基函数结构**：稠密收缩不要求核函数了解张量积结构与杨辉三角索引，代码简单、参数少、JIT 缓存友好。
> 3. **低阶时代价差距有限**：如 $N=3$、$N_q=4$ 时稠密收缩约 160 次乘加（每变量每方向），求和分解约 104 次；$N \lesssim 4$ 时差距不构成瓶颈，收益要到高阶（$N \gtrsim 5$）才显著。

**OCCA 核函数实现**：体积分核函数 `volumeIntegral` 以单元为单位并行（GPU 用 `@tile` 分块，OpenMP 用 `@outer` 并行）。每个单元内的 var×mode 和积分点全部串行计算。汇总得到体积分对残差的贡献 $\boldsymbol{R}_{\text{vol}} \in \mathbb{R}^{N_{\text{elem}} \cdot N_{\text{vars}} \cdot N_{\text{modes}}}$。

**粘性项处理**：

粘性通量 $\boldsymbol{F}_v$ 依赖于 $\nabla\boldsymbol{q}$，因此计算粘性通量前必须先在积分点上构造守恒变量的梯度场。CMeles 采用**两步走**策略：

1. **梯度求值核函数**（`computeGradient`）：对每个单元，在全部 $N_Q^2$ 个积分点上计算 $\nabla\boldsymbol{q}$。使用 Vandermonde 导数矩阵 $\mathbf{D}_{V,r}$ 和 $\mathbf{D}_{V,s}$ 从模态系数求参考坐标下的导数，再通过 $\mathbf{J}^{-T}$ 变换到物理坐标（**列向量约定，注意是 $\mathbf{J}^{-1}$ 的转置**，见[网格与几何文档 2.4 节](mesh_and_geometry.md#逆矩阵)）：
   $$
   \begin{bmatrix} \partial_x \boldsymbol{q} \\ \partial_y \boldsymbol{q} \end{bmatrix}_{(r_i, s_j)}
   = \mathbf{J}^{-T}(r_i, s_j)
   \begin{bmatrix} \sum_{\ell} \hat{\boldsymbol{u}}_\ell \, \partial_r \phi_\ell(r_i, s_j) \\
                    \sum_{\ell} \hat{\boldsymbol{u}}_\ell \, \partial_s \phi_\ell(r_i, s_j) \end{bmatrix}
   $$
   结果存入 `grad_u` 数组（$N_{\text{elem}} \times N_{\text{vars}} \times 2 \times N_Q^2$）。

2. **粘性通量计算**：在体积分核函数中，每个积分点处由 $\boldsymbol{q}$、$\nabla\boldsymbol{q}$ 和几何信息计算粘性通量 $\boldsymbol{F}_v$ 并参与弱形式累加。

> **无粘流动**：对于 Euler 方程（$Re \to \infty$，粘性通量为零），`computeGradient` 核函数可完全跳过，体积分仅计算对流通量，显著减少计算量。是否启用粘性通量由配置文件和编译宏 `VISCOUS_FLUX` 控制。

#### （三）面积分计算

面积分计算 $\displaystyle \int_{\Gamma_e} \hat{\boldsymbol{F}}(\boldsymbol{q}^-, \boldsymbol{q}^+, \boldsymbol{n}) \, \phi_q \, \mathrm{d}\Gamma_e$，对每个面单元遍历全部面积分点。

**Step 1 — 提取面两侧的节点值**：对于每个面积分点 $t_i$，从左右单元（或边界单元）的体单元节点值中插值获得该面点上的 $\boldsymbol{q}^-$ 和 $\boldsymbol{q}^+$。由于体单元的解以模态系数存储，需先在面单元的积分点处求值。对于直边四边形单元，面积分点与体单元积分点坐标可通过双线性映射关系计算，无需重新做完整的模态→节点变换——在体单元积分点采样后沿面方向插值即可。

**Step 2 — 计算数值通量**：

- **对流通量**：使用 Riemann 求解器（如 Rusanov / Local Lax-Friedrichs、Roe、HLLC）沿法向计算。利用 Euler 方程的旋转不变性（见[网格与几何文档 1.3 节](mesh_and_geometry.md#13-法向量)），将物理坐标系下的守恒变量旋转至局部法向-切向坐标系，在局部坐标系下用一维 Riemann 求解器计算通量，再将通量旋转回物理坐标系。

- **粘性通量**：通过 LDG（局部间断 Galerkin）或 IPDG（内罚函数）方法处理。面通量的粘性部分需要面两侧的 $\boldsymbol{q}^-$、$\boldsymbol{q}^+$ 以及两侧的平均梯度。以 LDG 为例：
  $$
  \hat{\boldsymbol{F}}_v = \{\!\{\boldsymbol{F}_v(\boldsymbol{q}, \nabla\boldsymbol{q})\}\!\} \cdot \boldsymbol{n} + \tau \cdot [\![\boldsymbol{q}]\!]
  $$
  其中 $\{\!\{\cdot\}\!\}$ 为平均值算子，$[\![\cdot]\!]$ 为跳跃算子，$\tau$ 为惩罚参数。

**Step 3 — 面积分累加**：对于每个面单元，将数值通量乘以测试函数 $\phi_q(t_i)$、权重 $w_i$ 和面雅可比 $|J_f|$ 后在面积分点上累加：

$$
R_{f,q}^{(k)} = \sum_{i=0}^{N_Q-1} w_i \, \hat{F}_k(t_i) \, \phi_q(t_i) \, |J_f|
$$

其中 $k$ 为守恒变量索引，$q$ 为基函数索引。此结果不直接写入体单元残差，而是存入中间数组 `face_flux[face][k][q]`。随后由 gather 核函数将各面对应的贡献按符号（左单元取正，右单元取负）累加到体单元的 `res` 数组中。

**OCCA 核函数实现**：面积分采用**先计算再 gather**的两步策略，避免写冲突。

1. **面通量计算核函数**（`computeFaceFlux`）：每个面单元独立计算数值通量在面积分点上的加权累加结果（对 $N_Q$ 个积分点做 $\sum_i w_i \hat{F}_k(t_i) \phi_q(t_i) |J_f|$ 收缩），输出为 $N_{\text{vars}} \times N_{\text{modes}}$ 个标量，存入中间数组 `face_flux[f][k][q]`。由于每个面只写自己的结果，面单元间完全独立并行，无需原子操作。

2. **面贡献 gather 核函数**（`gatherSurfaceRHS`）：每个体单元遍历其关联的面（四边形 4 条边），根据该面在体单元中的局部面朝向和符号（左单元 + / 右单元 −），从 `face_flux` 中读取对应的模态系数向量，累加到体单元的残差 `res` 中。此核函数以体单元为单位并行，每个体单元只写自己的残差，同样无写冲突。

两步核函数均使用 `@tile(TILE_SIZE, @outer, @inner)`（GPU）或 `@outer`（OpenMP），并行模式与体积分一致。

此设计避免了 `@atomic` 操作——`@atomic` 仅支持简单标量操作，无法原子更新完整残差向量（$N_{\text{vars}} \times N_{\text{modes}}$ 个 double），且高竞争下性能很差。同时，两步法只需两次核函数启动（一次面、一次体），面核函数间的独立并行性更好，体核函数对残差的写入具有良好缓存局部性。

#### （四）右端项组装

体积分和面积分分别完成后，右端项 $\mathcal{R}(\boldsymbol{u})$ 由两者相减并左乘逆质量矩阵得到。

物理单元上的质量矩阵 $M_{\ell\ell'} = \iint_{\text{ref}} \phi_\ell(r,s)\phi_{\ell'}(r,s)|\mathbf{J}(r,s)|\,dr\,ds$。由于 $|\mathbf{J}(r,s)|$ 为参考坐标的双线性函数，一般四边形下 $\mathbf{M}$ 为 $N_{\text{modes}} \times N_{\text{modes}}$ 的满秩矩阵（规模通常很小，如 $N=3$ 时仅 $10 \times 10$）。DG 场模块的输出——半离散右端项，为求解：

$$
\mathcal{R} = \mathbf{M}^{-1}\bigl(\boldsymbol{R}_{\text{vol}} - \boldsymbol{R}_{\text{surf}}\bigr)
$$

$\mathbf{M}^{-1}$ 可在预处理阶段逐单元预计算并存储，每次右端项组装仅需一次小规模矩阵-向量乘法（$N_{\text{modes}}$ 维），计算量远小于体积分和面积分。

组装核函数 `assembleRHS` 对每个单元执行：先将体积分和面积分贡献按 $(k,j)$ 逐分量相减得到半离散残差 $\boldsymbol{R}_{\text{vol}} - \boldsymbol{R}_{\text{surf}}$，再左乘预计算的 $\mathbf{M}^{-1}$，结果写入 `res` 数组。注意：$\boldsymbol{R}_{\text{surf}}$ 并非独立存储的完整数组——面积分的结果已通过 `gatherSurfaceRHS` 核函数直接累加到 `res` 中（或独立数组中再由此核函数减掉）。具体实现中 `assembleRHS` 可直接在 `res` 上原地完成减法与矩阵乘法。

#### 核函数概览

|      核函数名      |        OKL 文件        | 功能                                                               | 并行粒度 |
| :----------------: | :--------------------: | :----------------------------------------------------------------- | :------: |
|  `initModeCoeffs`  |       `init.okl`       | 初始条件→模态系数的 L2 投影                                        |  每单元  |
| `computeGradient`  |     `gradient.okl`     | 积分点上的守恒变量梯度                                             |  每单元  |
|  `volumeIntegral`  | `volume_integral.okl`  | 体积分：$\int \boldsymbol{F} \cdot \nabla\phi \, \mathrm{d}\Omega$ |  每单元  |
| `computeFaceFlux`  | `surface_integral.okl` | 面通量计算：每个面单元独立计算加权数值通量，输出模态系数向量       | 每面单元 |
| `gatherSurfaceRHS` | `surface_integral.okl` | 面贡献 gather：每个体单元遍历关联面累加符号化通量到残差            |  每单元  |
|   `assembleRHS`    |   `assemble_rhs.okl`   | 体积分 − 面积分，左乘 $\mathbf{M}^{-1}$                            |  每单元  |

> **注**：`computeFaceFlux` 和 `gatherSurfaceRHS` 可放在同一 `.okl` 文件中，以不同 `@kernel` 函数区分（OCCA 一次编译可生成多个 kernel 对象）。

#### DG 场模块的完整调用流程

```
输入: 模态系数数组 u_f

1. [可选] computeGradient    → 积分点上的梯度 grad_u
2. volumeIntegral            → 体积分贡献 R_vol
3. computeFaceFlux           → 面单元模态系数 face_flux
4. gatherSurfaceRHS          → 面贡献 gather 到体单元残差 R_surf
5. assembleRHS               → R = M⁻¹ * (R_vol - R_surf)

输出: 残差数组 res = R(u_f)
```

此残差随后由[时间推进模块](time_marching.md)处理，驱动 $\frac{\mathrm{d}\boldsymbol{u}_{\text{f}}}{\mathrm{d}t} = \mathcal{R}(t, \boldsymbol{u}_{\text{f}})$ 的时间积分。
