# UHF Gradient


## 原理简述


UHF gradient 与 RHF 情形其实是非常相似的。使用的中间变量、函数几乎是一致的。

有区别的地方在于：

- 密度矩阵一般使用 $D_{\mu \nu} = D_{\mu \nu}^\alpha + D_{\mu \nu}^\beta$
- 轨道能加权的密度矩阵 $E_{\mu \nu} = E_{\mu \nu}^\alpha + E_{\mu \nu}^\beta = C_{\mu i}^\alpha \varepsilon_i^\alpha n_i^\alpha C_{\nu i}^\alpha + C_{\mu i}^\beta \varepsilon_i^\beta n_i^\beta C_{\nu i}^\beta$
- 交换积分的辅助量
    $$
    \mathscr{K}_{P, ij}^\sigma = (\mathbf{J}^{-1/2})_{QP} Y_{Q, \mu \nu} C_{\mu i}^\sigma C_{\nu j}^\sigma
    $$
  这里我们就不再使用所谓加权轨道系数 $\tilde C_{\mu i}^\sigma = C_{\mu i}^\sigma \sqrt{n_i^\sigma}$ 了，而且 UHF 的占据数 $n_i^\sigma$ 非 1 即 0，确实没有必要使用加权轨道系数。
- 其他交换积分辅助量 $\mathscr{K}_{PQ}$, $\mathscr{K}_{P, \mu \nu}$ 与密度矩阵相似，都是 $\alpha$ 自旋与 $\beta$ 自旋求和。
- 需要留意，交换积分最后的系数乘以的是 -1，不是 -0.5。


## 计算需求简要分析


### 内存用量需求


首先，整个计算 $\mathscr{K}_{P, \mathrm{tp} (i j)}^{\sigma}$ 的 $\sigma \in \{\alpha, \beta\}$，因此会占用两倍的内存量。但注意到作为 $O(N^3)$ 的内存需求，其有两个指标是占据轨道，因此一般是可以接受的 (如果 3c-2e ERI 都可以存储于内存中)。

其次，其余的内存需求与 RHF 情形几乎完全相同。


### 计算量需求


该问题一般是电子积分瓶颈，基本上不是计算瓶颈。因此我们预期 UHF 梯度与同等大小的 RHF 梯度耗时是非常接近的，不会有两倍的时间差距。
