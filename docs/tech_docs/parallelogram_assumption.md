# 平行四边形假设与对角质量矩阵

本文档描述一种特殊情况下的简化假设，可作为 [基函数](basis_functions.md) 中一般四边形单元处理的补充说明。

## 动机

对于一般的直边四边形，双线性映射的交叉项使得雅可比行列式随参考坐标变化，物理单元上的质量矩阵不再是单位矩阵的标量倍。若网格质量允许（所有单元均为平行四边形），可采用以下简化假设，使质量矩阵保持对角结构，显著降低计算开销。

## 平行四边形条件

平行四边形的对边平行且等长，即 $\overrightarrow{P_1 P_2} = \overrightarrow{P_4 P_3}$（等价于 $P_1 + P_3 = P_2 + P_4$）。代入双线性系数 $a_3 = \frac{x_1 - x_2 + x_3 - x_4}{4}$，利用 $x_3 = x_2 + x_4 - x_1$ 可得 $a_3 = 0$，同理 $b_3 = 0$。交叉项消失，双线性映射退化为**仿射映射**：

$$
x(r, s) = x_1 + \frac{x_2 - x_1}{2}\, r + \frac{x_4 - x_1}{2}\, s
$$

$$
y(r, s) = y_1 + \frac{y_2 - y_1}{2}\, r + \frac{y_4 - y_1}{2}\, s
$$

## 常数雅可比矩阵

定义从 $P_1$ 出发的两条邻边向量 $\vec{e}_{21} = \overrightarrow{P_1 P_2} = (x_2 - x_1,\, y_2 - y_1)$ 和 $\vec{e}_{41} = \overrightarrow{P_1 P_4} = (x_4 - x_1,\, y_4 - y_1)$，雅可比矩阵为常数矩阵：

$$
\mathbf{J} = \frac{1}{2}
\begin{bmatrix}
x_2 - x_1 & x_4 - x_1 \\
y_2 - y_1 & y_4 - y_1
\end{bmatrix}
= \frac{1}{2}
\begin{bmatrix}
e_{21}^x & e_{41}^x \\
e_{21}^y & e_{41}^y
\end{bmatrix}
$$

雅可比行列式为常数：

$$
|\mathbf{J}| = \frac{1}{4} \left( e_{21}^x \, e_{41}^y - e_{21}^y \, e_{41}^x \right) = \frac{\vec{e}_{21} \times \vec{e}_{41}}{4}
$$

其几何含义为物理平行四边形面积与参考正方形面积之比：$|\mathbf{J}| = \frac{A_{\text{phys}}}{4}$。

## 质量矩阵的对角性

由于雅可比行列式为常数，可从积分中提出：

$$
M_{\ell\ell'} = \iint_{\text{ref}} \phi_\ell(r, s) \, \phi_{\ell'}(r, s) \, |\mathbf{J}| \, dr \, ds = |\mathbf{J}| \iint_{\text{ref}} \phi_\ell(r, s) \, \phi_{\ell'}(r, s) \, dr \, ds = |\mathbf{J}| \, \delta_{\ell\ell'}
$$

物理单元上的质量矩阵为 $\mathbf{M} = |\mathbf{J}| \, \mathbf{I}$，保持对角结构。其逆矩阵为 $\mathbf{M}^{-1} = \frac{1}{|\mathbf{J}|} \, \mathbf{I}$，L2 投影无需矩阵求逆，计算效率与参考单元上完全一致。

## 适用条件

- 网格中所有四边形单元均为平行四边形（对边严格平行）
- 需使用结构化或半结构化网格
- 弯曲边界的高阶几何单元不适用此假设
