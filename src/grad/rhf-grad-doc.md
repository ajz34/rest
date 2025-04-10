# RHF Gradient


## 原理简述


### 大体原理


RHF 梯度包含下述五部分：

- Hamiltonian core
- 库伦
- 交换
- 原子核
- 重叠

上述五项中，从公式推演的角度，较为特别的项是重叠部分。依据当前文档的推导方式，重叠积分本质上不是能量的导数，而更接近 Lagrangian 乘子的导数。

从计算耗时的角度，较为特别的项是交换部分；其次是库伦部分。剩余的部分耗时大体上是可以忽略的。


RHF 梯度的计算如下：

$$
\partial_{A_t} E^\mathsf{SCF} = \partial_{A_t}^\mathrm{S} E^\mathsf{SCF} - S_{\mu \nu}^{A_t} E_{\mu \nu}
$$


其中，记号 $\partial_{A_t}^\mathrm{S}$ 是指 Skeleton 导数，即对核坐标或电子积分的导数、但不对轨道系数作导数。


### 程序实现策略


从程序实现的角度来说，我们将梯度的贡献分为四大类：

- 原子核能量贡献
- Hamiltonian Core 贡献
- 可约化到原子轨道的贡献 (包括重叠积分、部分 J、部分 K 贡献)
- 可约化到辅助基的贡献 (包括部分 J、部分 K 贡献)


$$
\partial_{A_t} E^\textsf{SCF} =
    \partial_{A_t} E^\textsf{nuc}
    + \sum_{\mu \nu} \partial_{A_t} h_{\mu \nu} D_{\mu \nu}
    + \sum_{\mu \in A} \Delta_{t \mu}
    + \sum_{P \in A} \Delta_{t P}
$$


### 程序实现约定


在我们对 RHF 实现中，有如下约定 (convention)：

- 除了最后与 REST 主程序的交互部分外，我们统一使用 col-major。因此，很多张量的定义､以及 broadcasting 规则与 NumPy 是反者来的，在调试程序与使用 Python 作性能初步实现与测评时必须要多留意。
    - 轨道系数 $C_{\mu p}$ 的储存是 $(\mu, p)$，亦是 col-major。这与 REST 和 PySCF 相同。
- 在变量名中，记号简称有
    - `o`: occupied
    - `v`: virtual
    - `b`: basis (atomic orbital)
    - `a`: all (molecular orbital, occ + vir, not used in HF gradient)
    - `u`: auxiliary
    - `tp`: triangular-packed
- 迭代指标变量的记号简称是
    - `i`: occupied
    - `u`: basis
    - `p`: auxiliary
- 电子积分的指标顺序依照 libcint 给出的积分下，不作任何额外内存分配与转置的情况下，col-major 的顺序。这与 PySCF 的记号约定有较大差别 (PySCF 是既非 row 也非 col major)。


## 具体实现


### 原子核贡献


该计算已经在 REST 有所实现。

$$
\partial_{A_t} E^\textsf{SCF} \leftarrow - \sum_M Z_A Z_M r_{AM}^{-3} (A_t - M_t)
$$


### 重叠积分贡献


$$
\begin{align}
E_{\mu \nu} &= C_{\mu i} \varepsilon_i n_i C_{\nu i} \\
\Delta_{t \mu} &\leftarrow 2 (\partial_t \mu | \nu) E_{\mu \nu}
\end{align}
$$


需要注意，这里被求和的指标是 $\nu$，而该指标在 `int1e_ipovlp` (指标顺序 $(t, \nu, \mu)$) 是倒数第二位的，因此需要用 `sum(axis=-2)`。这里 `int1e_ipovlp` 的指标顺序与通常的习惯少许不同。


**函数 `get_dme0`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `mo_coeff` | $C_{\mu p}$ | $(\mu, p)$ | $(n_\mathrm{basis}, n_\mathrm{orb})$ |
| `mo_occ` | $n_p$ | $(p)$ | $(n_\mathrm{orb})$ |
| `mo_energy` | $\varepsilon_p$ | $(p)$ | $(n_\mathrm{orb})$ |
| `dme0` | $E_{\mu \nu}$ | $(\mu, \nu)$ | $(n_\mathrm{basis}, n_\mathrm{basis})$ | sym |


**函数 `get_grad_dao_ovlp`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int1e_ipovlp` | $(\partial_t \mu | \nu)$ | $(\mu, \nu, t)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, 3)$ | anti-sym |
| `dme0` | $E_{\mu \nu}$ | $(\mu, \nu)$ | $(n_\mathrm{basis}, n_\mathrm{basis})$ | sym |
| `dao_ovlp` | $\Delta_{t \mu}$ 贡献项 | $(\mu, t)$ | $(n_\mathrm{basis}, 3)$ |


### Hamiltonian Core 贡献


$$
\partial_{A_t} E^\textsf{SCF} \leftarrow \partial_{A_t} h_{\mu \nu} D_{\mu \nu}
$$


首先，从程序的角度，$\partial_{A_t} h_{\mu \nu}$ 是依赖于原子核 $A$ 的量；由于它不是计算瓶颈问题，浪费一些内存与计算资源并非不可接受。因此，我们可以对每个原子 $A$ 进行迭代，生成其对应的 $\partial_{A_t} h_{\mu \nu}$。这个迭代生成器定义在 `generator_deriv_hcore` 中；该迭代生成器读入分子信息，输出内部的函数；该生成器需要在内部存储一些变量以避免重复计算 (这里是变量 `h1`)。在 Python 中，这可以通过类实现，也可以通过内部函数实现。在 Rust 中，我们就用 closure 实现。


其次，我们需要给出 $\partial_{A_t} h_{\mu \nu}$ 的具体计算过程。首先定义动能与势能算符的表示

$$
\begin{align}
k_{\mu \nu} &= \langle \mu | \hat k | \nu \rangle = \langle \mu | - \frac{1}{2} \nabla^2 | \nu \rangle \\
v_{\mu \nu} &= \langle \mu | \hat v | \nu \rangle = \langle \mu | \sum_A \frac{- Z_A}{|\boldsymbol{r} - \boldsymbol{A}|} | \nu \rangle \\
h_{\mu \nu} &= k_{\mu \nu} + v_{\mu \nu}
\end{align}
$$


其导数则为

$$
\begin{align}
\partial_{A_t} k_{\mu \nu} &= - \langle \partial_t \mu_{\in A} | \hat k | \nu \rangle - \langle \partial_t \mu | \hat k | \partial_t \nu_{\in A} \rangle \\
\partial_{A_t} v_{\mu \nu} &= - \langle \partial_t \mu_{\in A} | \hat v | \nu \rangle - \langle \partial_t \mu | \hat v | \partial_t \nu_{\in A} \rangle + \langle \mu | \partial_t \frac{-Z_A}{|\boldsymbol{r}_{\rightarrow A}|} | \nu \rangle
\end{align}
$$


上式中，$\mu_{\in A}$ 是只有原子轨道中心在 $A$ 原子上的基组 $\mu$ 才被纳入统计。

对于 $\partial_{A_t} v_{\mu \nu}$ 的第三项，由于它只处理原子 $A$ 而不作求和，因此其计算可以通过更换规范原点位置 (gauge origin) 以进行计算；记号 $\boldsymbol{r}_{\rightarrow A}$ 是指将 $\boldsymbol{r}$ 的规范原点变更为 $\boldsymbol{A}$。需要注意，这里我们利用了一次类似于 Stokes 定理的做法，将电子积分作如下变换：

$$
\partial_t \langle \mu | \partial_t \frac{-Z_A}{|\boldsymbol{r}_{\rightarrow A}|} | \nu \rangle \equiv 0
    = \langle \mu | \partial_t \frac{-Z_A}{|\boldsymbol{r}_{\rightarrow A}|} | \nu \rangle
    + \langle \partial_t \mu | \frac{-Z_A}{|\boldsymbol{r}_{\rightarrow A}|} | \nu \rangle
    + \langle \mu | \frac{-Z_A}{|\boldsymbol{r}_{\rightarrow A}|} | \partial_t \nu \rangle
$$

因此，整理上述式子可得

$$
\begin{align}
\texttt{h1}_{\mu \nu}^t &= - \langle \partial_t \mu | \hat k | \nu \rangle - \langle \partial_t \mu | \hat v | \nu \rangle \\
\texttt{int1e\_iprinv}_{\mu \nu}^t &= \langle \partial_t \mu | \frac{1}{|\boldsymbol{r}_{\rightarrow A}|} | \nu \rangle \\
\partial_{A_t} h_{\mu \nu} &= \texttt{h1}_{\mu_{\in A} \nu}^t + \texttt{h1}_{\mu \nu_{\in A}}^t - Z_A \texttt{int1e\_iprinv}_{\mu \nu}^t - Z_A \texttt{int1e\_iprinv}_{\nu \mu}^t
\end{align}
$$


**函数 `generator_deriv_hcore`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int1e_ipkin` | $\langle \partial_t \mu | \hat k | \nu \rangle$ | $(\mu, \nu, t)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, 3)$ | anti-sym |
| `int1e_ipnuc` | $\langle \partial_t \mu | \hat v | \nu \rangle$ | $(\mu, \nu, t)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, 3)$ | non-sym |
| `int1e_iprinv` | $\langle \partial_t \mu | \frac{1}{|\boldsymbol{r}_{\rightarrow A}|} | \nu \rangle$ | $(\mu, \nu, t)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, 3)$ | non-sym, gauge-dependent |
| `h1` | $\texttt{h1}_{\mu \nu}^t$ | $(\mu, \nu, t)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, 3)$ | non-sym |
| `vrinv` | $\partial_{A_t} h_{\mu \nu}$ | $(\mu, \nu, t)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, 3)$ | sym |


### 双电子积分分解矩阵 $\mathbf{J}^{-1/2}$


在目前的 REST 中，双电子积分分解矩阵 $\mathbf{J}^{-1/2}$ 是严格的矩阵分数幂次。

需要指出的是，Cholesky 分解可能在一些缩并问题上有更高的效率。因此如果以后 REST 要更改分解策略，下面的代码和算法需要作一定调整。


| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int2c2e_l_inv` | $(\mathbf{J}^{-1/2})_{PQ}$ | $(P, Q)$ | $(n_\mathrm{aux}, n_\mathrm{aux})$ | sym |


### 密度矩阵的特殊上三角压缩


在我们的算法中，偶尔会出现对称矩阵的乘法与求和计算：

$$
a_t = \sum_{\mu \nu} b_{t, \mu \nu} c_{\mu \nu} = \sum_{\mu \leqslant \nu} (2 - \delta_{\mu \nu}) b_{t, \mu \nu} c_{\mu \nu}
$$

其中，张量 $b_{t, \mu \nu}$ 与矩阵 $c_{\mu \nu}$ 都是关于 $(\mu, \nu)$ 指标对称的。

这种情况下，求和计算不一定要对完整的 $(\mu, \nu)$ 求和，而只对其上三角部分 $(\mu, \nu \geqslant \nu)$ 求和。为此，定义

$$
\begin{align}
b_{t, \mathrm{tp}(\mu \nu)} &\bowtie b_{t, \mu \nu} \\
c_{\mathrm{tp}(\mu \nu)} &\mathop{\tilde{\bowtie}} c_{\mu \nu}
\end{align}
$$

其中，$\mathrm{tp}(\mu \nu)$ 是一维指标，长度 $\frac{1}{2} n_\mathrm{basis} (n_\mathrm{basis} + 1)$。映射关系 $\bowtie$ 与 $\mathop{\tilde{\bowtie}}$ 分别指代的是

$$
\begin{align}
\bowtie &: b_{t, \mathrm{tp}(\mu \nu)} = b_{t, \mu \nu} \\
\mathop{\tilde{\bowtie}} &: c_{\mathrm{tp}(\mu \nu)} = (2 - \delta_{\mu \nu}) c_{\mu \nu}
\end{align}
$$


在当前的梯度计算问题中，我们令密度矩阵的特殊上三角压缩为

$$
D_{\mathrm{tp} (\mu \nu)} \mathop{\tilde{\bowtie}} D_{\mu \nu}
$$


#### 关于到底选用下三角还是上三角


这里涉及到 row-major 与 col-major 其中一个具有根本性分歧的问题。

我们考虑下述对称矩阵

$$
\begin{pmatrix}
0 & 1 & 3 \\
1 & 2 & 4 \\
3 & 4 & 5
\end{pmatrix}
$$

该矩阵是 $3 \times 3 = 9$ 大小的矩阵；它作为演示的示例不大，但在电子结构计算中可能就是内存的存储瓶颈。在这种情况下，储存其三角部分的 6 个元素就可以省出差不多一半内存。

但问题是，这 6 个元素到底要如何存储？

$$
\begin{pmatrix}
0 & * & * \\
1 & 2 & * \\
3 & 4 & 5
\end{pmatrix}
$$

这里 Row-major 与 Col-major 就会产生分歧。
- Row-major 情形下，它将存储为 $(0, 1, 2, 3, 4, 5)$
- Col-major 情形下，它将存储为 $(0, 1, 3, 2, 4, 5)$

但是有意思的是，如果存储上三角矩阵，Col-major 将会给出 $(0, 1, 2, 3, 4, 5)$，与 Row-major 的下三角是相同的：

$$
\begin{pmatrix}
0 & 1 & 3 \\
* & 2 & 4 \\
* & * & 5
\end{pmatrix}
$$


那么我们应该选择下三角、还是上三角？我们指出，在已有程序的使用习惯上，
- Row-major 应选择下三角
    - PySCF 中经常用函数 pack_tril 与 unpack_tril，但你会发现根本没有 pack_triu 与 unpack_triu 函数的定义；
    - NumPy 中，像 Cholesky 分解函数 ([np.linalg.cholesky](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html)) 的默认选项 `upper=False`，也是因为 NumPy 是 row-major，因此默认是下三角；
- Col-major 应选择上三角
    - SciPy 中，Cholesky 分解函数 ([scipy.linalg.cholesky](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky.html)) 的默认选项反而是 `lower=False`，是因为 SciPy 主要使用 [f2py 接口](https://github.com/scipy/scipy/blob/0f1fd4a7268b813fa2b844ca6038e4dfdf90084a/scipy/linalg/flapack_pos_def.pyf.src#L124-L140)，而 f2py 遵循的是 Fortran 习惯使用 col-major，其三角矩阵的定义风格就与 col-major 有所绑定。
- 一些程序通过定义上下三角矩阵的类型，从根源上规避这个问题 (这里是指 Julia；我们的数学库没有 Julia 那样的野心，所以决定将一些使用上的责任让渡给用户)。

我相信 row-major 选择下三角、col-major 选择上三角是有其考虑的。以 row-major 为例，我们可以不需要事先知道矩阵的大小，也能判断下三角矩阵压缩后元素的位置、在压缩前的索引到底是多少，参考下述 Python 程序：

```python
def get_unpacked(idx):
    if idx == 0:
        return (0, 0)
    import numpy as np
    i = int(np.floor(np.sqrt(2 * idx)))
    j = idx - i * (i + 1) // 2
    if j < 0:
        return (i - 1, j + i)
    else:
        return (i, j)
```

```python
for idx in range(6):
    print(get_unpacked(idx))

# output
# (0, 0)
# (1, 0)
# (1, 1)
# (2, 0)
# (2, 1)
# (2, 2)
```

但如果是 row-major 的上三角矩阵，那么就必须事先知道矩阵维度，才能给出压缩前的索引位置。

出于同样的原因，col-major 则倾向于使用上三角，而不使用下三角。


**函数 `pack_triu_tilde`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `dm` | $D_{\mu \nu}$ | $(\mu, \nu)$ | $(n_\mathrm{basis}, n_\mathrm{basis})$ | sym |
| `dm_tp` | $D_{\mathrm{tp} (\mu \nu)}$ | $(\mathrm{tp} (\mu \nu))$ | $(\frac{1}{2} n_\mathrm{basis} (n_\mathrm{basis} + 1))$ | $\mathop{\tilde{\bowtie}}$ |


### J 贡献


以下计算给出的是 J 贡献计算的中间量 (intermediate)

$$
\mathscr{J}_P = (\mathbf{J}^{-1/2})_{QP} Y_{Q, \mu \nu} D_{\mu \nu}
$$


**函数 `get_itm_j`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int2c2e_l_inv` | $(\mathbf{J}^{-1/2})_{PQ}$ | $(P, Q)$ | $(n_\mathrm{aux}, n_\mathrm{aux})$ | sym |
| `ederi_utp` | $Y_{Q, \mathrm{tp} (\mu \nu)}$ | $(\mathrm{tp} (\mu \nu), Q)$ | $(\frac{1}{2} n_\mathrm{basis} (n_\mathrm{basis} + 1), n_\mathrm{aux})$ | $\bowtie$ |
| `dm_tp` | $D_{\mathrm{tp} (\mu \nu)}$ | $(\mathrm{tp} (\mu \nu))$ | $(\frac{1}{2} n_\mathrm{basis} (n_\mathrm{basis} + 1))$ | $\mathop{\tilde{\bowtie}}$ |
| `itm_j` | $\mathscr{J}_P$ | $(P)$ | $(n_\mathrm{aux})$ |


以下是 J 贡献在原子轨道基组导数上的结果：

$$
\Delta_{t \mu} \leftarrow - 2 D_{\mu \nu} (\partial_t \mu \nu | P) \mathscr{J}_P
$$


**函数 `get_grad_dao_j_int3c2e_ip1`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int3c2e_ip1_pubb` | $(\partial_t \mu \nu | P)$ | $(\mu, \nu, P, t)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{aux}, 3)$ | non-sym |
| `dm` | $D_{\mu \nu}$ | $(\mu, \nu)$ | $(n_\mathrm{basis}, n_\mathrm{basis})$ | sym |
| `itm_j` | $\mathscr{J}_P$ | $(P)$ | $(n_\mathrm{aux})$ |
| `dao_j_int3c2e_ip1` | $\Delta_{t \mu}$ 贡献项 | $(\mu, t)$ | $(n_\mathrm{basis}, 3)$ |


以下是 J 贡献在 3c-2e ERI 辅助基组导数上的结果：

$$
\Delta_{t P} \leftarrow - D_{\mu \nu} (\mu \nu | \partial_t P) \mathscr{J}_P
$$


**函数 `get_grad_daux_j_int3c2e_ip2`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int3c2e_ip2_putp` | $(\mu \nu | \partial_t P)$ | $(\mathrm{tp} (\mu \nu), P, t)$ | $(\frac{1}{2} n_\mathrm{basis} (n_\mathrm{basis} + 1), n_\mathrm{aux}, 3)$ | $\bowtie$ |
| `dm_tp` | $D_{\mathrm{tp} (\mu \nu)}$ | $(\mathrm{tp} (\mu \nu))$ | $(\frac{1}{2} n_\mathrm{basis} (n_\mathrm{basis} + 1))$ | $\mathop{\tilde{\bowtie}}$ |
| `itm_j` | $\mathscr{J}_P$ | $(P)$ | $(n_\mathrm{aux})$ |
| `daux_j_int3c2e_ip2` | $\Delta_{t P}$ 贡献项 | $(P, t)$ | $(n_\mathrm{aux}, 3)$ |


以下是 J 贡献在 2c-2e ERI 辅助基组导数上的结果：

$$
\Delta_{tP} \leftarrow \mathscr{J}_P (\partial_t P | Q) \mathscr{J}_Q
$$


**函数 `get_grad_daux_j_int2c2e_ip1`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int2c2e_ip1` | $(\partial_t P | Q)$ | $(P, Q, t)$ | $(n_\mathrm{aux}, n_\mathrm{aux}, 3)$ | anti-sym |
| `itm_j` | $\mathscr{J}_P$ | $(P)$ | $(n_\mathrm{aux})$ |
| `daux_j_int2c2e_ip1` | $\Delta_{t P}$ 贡献项 | $(P, t)$ | $(n_\mathrm{aux}, 3)$ |


### K 贡献


以下计算给出的是 K 贡献计算的中间量 (intermediate)

$$
\mathscr{K}_{P, ij} = (\mathbf{J}^{-1/2})_{QP} Y_{Q, \mu \nu} \tilde C_{\mu i} \tilde C_{\nu j}
$$

需要注意，该中间量是本计算过程唯一需要用到 $O(N^3)$ 大小的存储；其他的计算都可以利用分批方法给出 $O(N^2)$ 大小的存储。但由于两个指标是占据轨道，并且该矩阵是对称的，因此实际上储存空间很小，甚至大幅地小于很多分批计算的电子积分。


同时需要留意的是，与 J 贡献中使用密度矩阵而非轨道系数呼应地，在 K 贡献中，我们使用“加权轨道系数”进行计算：

$$
\tilde C_{\mu i} = C_{\mu i} \sqrt{n_i}
$$


**函数 `get_itm_k_occtp`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int2c2e_l_inv` | $(\mathbf{J}^{-1/2})_{PQ}$ | $(P, Q)$ | $(n_\mathrm{aux}, n_\mathrm{aux})$ | sym |
| `ederi_utp` | $Y_{Q, \mathrm{tp} (\mu \nu)}$ | $(\mathrm{tp} (\mu \nu), Q)$ | $(\frac{1}{2} n_\mathrm{basis} (n_\mathrm{basis} + 1), n_\mathrm{aux})$ | $\bowtie$ |
| `weighted_occ_coeff` | $\tilde C_{\mu i}$ | $(\mu, i)$ | $(n_\mathrm{basis}, n_\mathrm{occ})$ |
| `itm_k_occtp` | $\mathscr{K}_{P, \mathrm{tp} (ij)}$ | $(\mathrm{tp} (ij), P)$ | $(\frac{1}{2} n_\mathrm{occ} (n_\mathrm{occ} + 1), n_\mathrm{aux})$ | $\bowtie$ |


以下计算给出的是 K 贡献计算的中间量 (intermediate)

$$
\mathscr{K}_{P, \mu \nu} = \mathscr{K}_{P, ij} \tilde C_{\mu i} \tilde C_{\nu j}
$$

该贡献中间量的内存需求较大；它尽管是对称的，但我们有意选择不将其压缩，因为后面有一次使用情景是不压缩的。压缩与否对性能的影响其实不大，但代码写起来还是不压缩的比较方便。


**函数 `get_itm_k_ao`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `itm_k_occtp` | $\mathscr{K}_{P, \mathrm{tp} (ij)}$ | $(\mathrm{tp} (ij), P)$ | $(\frac{1}{2} n_\mathrm{occ} (n_\mathrm{occ} + 1), n_\mathrm{aux})$ | $\bowtie$ |
| `weighted_occ_coeff` | $\tilde C_{\mu i}$ | $(\mu, i)$ | $(n_\mathrm{basis}, n_\mathrm{occ})$ |
| `itm_k_ao` | $\mathscr{K}_{P, \mu \nu}$ | $(\mu, \nu, P)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{aux})$ | sym |


以下计算给出的是 K 贡献计算的中间量 (intermediate)

$$
\mathscr{K}_{PQ} = \mathscr{K}_{P, ij} \mathscr{K}_{Q, ij}
$$

需要注意到，在程序实现中，$\mathscr{K}_{P, ij}$ 是以下三角矩阵存储的：

$$
\mathscr{K}_{PQ} = \sum_{i \geqslant j} (2 - \delta_{i j}) \mathscr{K}_{P, ij} \mathscr{K}_{Q, ij} = 2 \sum_{i \geqslant j} \left( \mathscr{K}_{P, ij} \sqrt{1 - \frac{1}{2} \delta_{i j}} \right) \left( \mathscr{K}_{Q, ij} \sqrt{1 - \frac{1}{2} \delta_{i j}} \right)
$$

因此，实际程序实现中，我们需要先对对角元缩放 $\sqrt{0.5}$ 倍，并将乘法的结果缩放 $2$ 倍；计算结束后，对角元缩放 $\sqrt{2}$ 倍。


**函数 `get_itm_k_aux`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `itm_k_occtp` | $\mathscr{K}_{P, \mathrm{tp} (ij)}$ | $(\mathrm{tp} (ij), P)$ | $(\frac{1}{2} n_\mathrm{occ} (n_\mathrm{occ} + 1), n_\mathrm{aux})$ | $\bowtie$ |
| `itm_k_aux` | $\mathscr{K}_{PQ}$ | $(P, Q)$ | $(n_\mathrm{aux}, n_\mathrm{aux})$ | sym |


以下是 K 贡献在原子轨道基组导数上的结果：

$$
\Delta_{t \mu} \leftarrow - 2 (\partial_t \mu \nu | P) \mathscr{K}_{P, \mu \nu}
$$


**函数 h`get_grad_dao_k_int3c2e_ip1`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int3c2e_ip1_pubb` | $(\partial_t \mu \nu | P)$ | $(\mu, \nu, P, t)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{aux}, 3)$ | non-sym |
| `itm_k_ao` | $\mathscr{K}_{P, \mu \nu}$ | $(\mu, \nu, P)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{aux})$ | sym |
| `dao_k_int3c2e_ip1` | $\Delta_{t \mu}$ 贡献项 | $(\mu, t)$ | $(n_\mathrm{basis}, 3)$ |


以下是 K 贡献在 3c-2e ERI 辅助基组导数上的结果：

$$
\Delta_{t P} \leftarrow - (\mu \nu | \partial_t P) \mathscr{K}_{P, \mu \nu}
$$


**函数 `get_grad_daux_k_int3c2e_ip2`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int3c2e_ip2_putp` | $(\mu \nu | \partial_t P)$ | $(\mathrm{tp} (\mu \nu), P, t)$ | $(\frac{1}{2} n_\mathrm{basis} (n_\mathrm{basis} + 1), n_\mathrm{aux}, 3)$ | $\bowtie$ |
| `itm_k_ao` | $\mathscr{K}_{P, \mu \nu}$ | $(\mu, \nu, P)$ | $(n_\mathrm{basis}, n_\mathrm{basis}, n_\mathrm{aux})$ | sym |
| `daux_j_int3c2e_ip2` | $\Delta_{t P}$ 贡献项 | $(P, t)$ | $(n_\mathrm{aux}, 3)$ |


以下是 K 贡献在 2c-2e ERI 辅助基组导数上的结果：

$$
\Delta_{tP} \leftarrow (\partial_t P | Q) \mathscr{K}_{PQ}
$$


**函数 `get_grad_daux_k_int2c2e_ip1`**

| 变量名 | 变量意义 | 指标顺序 | 维度大小 | 其他说明 |
|--|--|--|--|--|
| `int2c2e_ip1` | $(\partial_t P | Q)$ | $(P, Q, t)$ | $(n_\mathrm{aux}, n_\mathrm{aux}, 3)$ | anti-sym |
| `itm_k_aux` | $\mathscr{K}_{PQ}$ | $(P, Q)$ | $(n_\mathrm{aux}, n_\mathrm{aux})$ | sym |
| `daux_k_int2c2e_ip1` | $\Delta_{t P}$ 贡献项 | $(P, t)$ | $(n_\mathrm{aux}, 3)$ |


## 计算需求简要分析


### 内存用量需求


首先，整个计算中唯一一次使用到 $O(N^3)$ 的量是 $\mathscr{K}_{P, \mathrm{tp} (i j)}$。其占用浮点数约是 $\frac{1}{2} n_\mathrm{occ}^2 n_\mathrm{aux}$。大部分情况下，这个量是可以接受的：我们在计算 RI-HF 本身就需要占用大得多的 3c-2e ERI。

其次，其余计算中看起来涉及到 $O(N^3)$ 内存量的部分，都是可以对辅助基进行分解的。这包括

- $(\partial_t \mu \nu | P)$ ($3 \times n_\mathrm{basis}^2 n_\mathrm{aux}$)
- $(\mu \nu | \partial_t P)$ ($1.5 \times n_\mathrm{basis}^2 n_\mathrm{aux}$)
- $\mathscr{K}_{P, \mu \nu}$ ($1 \times n_\mathrm{basis}^2 n_\mathrm{aux}$)

为了保险起见，我们可以设置 $10 \times n_\mathrm{basis}^2$ 作为一个内存分配单元，确定每个分批次中辅助基的数量。


### 计算量需求


该问题比较微妙地，并不是受 $O(N^4)$ 复杂度的计算量影响，而是 $O(N^3)$ 电子积分和内存瓶颈。其中电子积分的计算耗时最大。

目前有尚未解决的疑问。这是关于电子积分耗时的问题。

在每个单步的效率评估时，我发现打印出来的结果有令人疑惑的地方。在对 auxiliary basis batch size 不设上限的前提下 (即 `aux_batch_size` 不设 216 的上限)，仅仅是变更 OpenBLAS 从 OpenBLAS 到 pthreads，就会产生很大的性能影响。

下述计算在 Ryzen 7945HX (16 cores) 下，运行 C12H26/def2-TZVP 的梯度计算耗时情况。最理想的情况是能进 3.4 sec，但这需要在张量 reduce 运算上引入高效的 einsum，而不能仅仅使用 RSTSR 目前实现的、类 NumPy 一般用途的标准算子。


```
OpenBLAS with OpenMP

Detailed time report:
de-jk preparation 1|:    0.432 s for de-jk preparation 1
de-jk preparation power|:    0.336 s for de-jk preparation power
de-jk preparation 2|:    0.130 s for de-jk preparation 2
de-jk batch int|:    1.538 s for de-jk batch int
de-jk batch 1|:    0.149 s for de-jk batch 1
de-jk batch 2|:    0.534 s for de-jk batch 2
de-jk batch 3|:    0.664 s for de-jk batch 3
de-jk batch 4|:    0.098 s for de-jk batch 4
de-jk batch 5|:    0.113 s for de-jk batch 5
Detailed time report:
rhf grad  |:    3.809 s for rhf grad
```


```
OpenBLAS with pthread

Detailed time report:
de-jk preparation 1|:    0.402 s for de-jk preparation 1
de-jk preparation power|:    0.290 s for de-jk preparation power
de-jk preparation 2|:    0.163 s for de-jk preparation 2
de-jk batch int|:    2.863 s for de-jk batch int
de-jk batch 1|:    0.150 s for de-jk batch 1
de-jk batch 2|:    0.546 s for de-jk batch 2
de-jk batch 3|:    0.663 s for de-jk batch 3
de-jk batch 4|:    0.087 s for de-jk batch 4
de-jk batch 5|:    0.115 s for de-jk batch 5
Detailed time report:
rhf grad  |:    5.218 s for rhf grad
```


我们写的 rest-libcint 接口没有真的引入 GEMM 函数或其他 BLAS，但在 `batch int` 即 3c-2e ERI 导数积分的耗时上差距巨大。对于这个耗时结果我确实无法理解。
