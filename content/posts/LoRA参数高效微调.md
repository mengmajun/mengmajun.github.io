+++ 
date = '2025-06-30' 
draft = false 
title = 'LoRA参数高效微调' 
categories = ['微调'] 
tags = ['Lora'] 
+++


《LoRA: Low-Rank Adaptation of Large Language Models》该论文由 Edward J. Hu 等人于 2021 年发表，提出了一种参数高效的微调方法，通过向预训练模型的权重矩阵中引入低秩矩阵来实现对大语言模型的轻量级适配

[论文地址](https://arxiv.org/abs/2106.09685)

## 一、大模型权重更新的低秩性

### 定义
在微调大型预训练模型（如Transformer）时，通常的做法是更新整个权重矩阵 $ W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} $。然而，实验发现：**实际有效的参数更新部分 $ \Delta W = W' - W $ 往往具有较低的秩（low-rank）**。

换句话说，虽然权重矩阵本身可能是高维且满秩的，但**其变化量** $ \Delta W $ **往往可以用一个低秩矩阵很好地近似**。

---

### 数学表达

假设原始权重为 $ W $，微调后变为 $ W + \Delta W $。

LoRA 的核心思想就是对这个增量进行低秩建模：

$$
\Delta W \approx A B^T, \quad \text{其中 } A \in \mathbb{R}^{d_{\text{out}} \times r}, B \in \mathbb{R}^{d_{\text{in}} \times r}, \quad r \ll \min(d_{\text{out}}, d_{\text{in}})
$$

也就是说，我们用两个小矩阵 $ A $ 和 $ B $ 相乘得到一个低秩矩阵 $ AB^T $ 来表示权重的变化。

非常好的问题！我们接下来从**线性代数和奇异值分解（SVD）**的角度，深入分析为什么低秩近似能够有效捕捉权重更新矩阵 $ \Delta W $ 的信息。

---


### 奇异值分解（Singular Value Decomposition）

任何实矩阵 $ M \in \mathbb{R}^{m \times n} $ 都可以分解为：

$$
M = U \Sigma V^T
$$

其中：
- $ U \in \mathbb{R}^{m \times m} $ 是正交矩阵，列是左奇异向量；
- $ \Sigma \in \mathbb{R}^{m \times n} $ 是对角矩阵，对角元素为奇异值（非负且降序排列）；
- $ V \in \mathbb{R}^{n \times n} $ 是正交矩阵，列是右奇异向量。

我们可以只保留前 $ r $ 个最大的奇异值及其对应的奇异向量，得到一个**最优的秩 $ r $ 近似矩阵**：

$$
M_r = U_r \Sigma_r V_r^T
$$

这个矩阵 $ M_r $ 在所有秩为 $ r $ 的矩阵中，与原矩阵 $ M $ 的 Frobenius 范数距离最小。

---

### LoRA 与 SVD 的联系

假设我们有一个完整的权重更新矩阵 $ \Delta W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} $，它是由全参数微调训练出来的。

如果我们对其做 SVD：

$$
\Delta W = U \Sigma V^T
$$

我们发现，很多奇异值都非常小，甚至接近于零。这说明：

> **ΔW 的能量集中在前几个奇异值上，也就是说 ΔW 的“信息”主要由前几个主方向决定。**

于是，我们可以通过取前 $ r $ 个奇异值构造一个低秩近似：

$$
\Delta W \approx \Delta W_r = U_r \Sigma_r V_r^T
$$

而 LoRA 所做的就是用两个低秩矩阵 $ A $ 和 $ B $ 来逼近这个 $ \Delta W_r $，即：

$$
A = U_r \Sigma_r^{1/2}, \quad B = V_r \Sigma_r^{1/2} \Rightarrow AB^T = U_r \Sigma_r V_r^T = \Delta W_r
$$

所以，**LoRA 实际上是在学习一个 ΔW 的低秩近似，而这正是 SVD 所给出的最佳低秩逼近**。


**线性代数角度看这种近似的有效性**

1. **信息压缩能力强**
- 即使 $ r \ll d $，也能保留大部分重要信息（前几个奇异值的能量占比很高）。
- 比如，在图像或语言任务中，前几十个奇异值可能已经包含了90%以上的信息。

2. **避免过拟合**
- 高维空间中容易过拟合噪声或不重要的方向；
- 低秩限制迫使模型关注那些真正影响输出的关键方向。

3. **参数效率高**
- 如果 $ W \in \mathbb{R}^{768 \times 768} $，那就有约 59 万个参数；
- 若使用 $ A, B \in \mathbb{R}^{768 \times 8} $，则总参数数只有 $ 2 \times 768 \times 8 = 12,288 $，减少了 97%！


---

## 二、现实数据变化往往具有较低的内在维度

很多实际问题中的数据变化往往具有较低的内在维度，**即现实世界的高维数据往往具有低维结构（low-dimensional manifold）**。这意味着虽然模型本身可能是非常高维的，但与任务相关的特征变化可能只需要在一个相对较小的子空间中捕捉，使用低秩分解能够有效地捕捉这些关键的变化方向，同时忽略那些对于特定任务无关紧样的细节。此外，这种策略还允许我们在不修改原有模型参数的情况下快速地针对不同任务进行调整，提高了灵活性并减少了过拟合的风险

---

### 举个例子

想象你在做文本分类任务，比如判断一句话是正面还是负面评价。

- 输入句子的 embedding 是 768 维的向量。
- 虽然每个词都嵌入到了 768 维空间，但实际上，决定情感正负的关键特征可能只集中在几个方向上，比如：
  - “好” vs “坏”
  - “喜欢” vs “讨厌”

这些关键词所代表的方向，构成了一个低维子空间。因此，**只要在这个低维子空间里做出调整，就能完成任务目标**。

---

### 更正式地讲

设输入数据分布为 $ x \in \mathbb{R}^d $，如果这些数据实际上分布在某个低维流形（manifold）上，那么它们的变化方向就受限于这个流形的维度 $ r \ll d $。

这意味着：

- 数据之间的差异或变化，主要体现在一个低维子空间中；
- 所以，模型在适应这些数据变化时，也只需要在那个低维子空间中调整即可。

---

> LoRA 成功的本质在于它抓住了这样一个事实：  
> **虽然模型是高维的，但任务所需的改变往往是局部、稀疏、低维的。**

通过引入低秩矩阵来捕捉这些关键方向的变化，LoRA 在保持性能的同时大幅减少了参数数量和训练成本。

---


## 三、LoRA代码实现



LoRA 的基本思想是：不直接更新原始模型权重 $ W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} $，而是引入两个低秩矩阵 $ A \in \mathbb{R}^{r \times d_{\text{in}}} $, $ B \in \mathbb{R}^{d_{\text{out}} \times r} $，并计算增量：

$$
\Delta W = B A^T
$$

最终的权重为：

$$
W' = W + \Delta W \cdot \frac{\alpha}{r}
$$

其中：
- $ r $: LoRA 的秩，一般取4、8、16、32
- $ \alpha $: 缩放因子（lora_alpha），一般取值$ 2*r $

---


### 以PEFT库的Linear层说明LoRA的实现：

```
peft/src/peft/tuners/lora/layer.py
```

`class Linear(nn.Module, LoraLayer)` 的作用

这是对 PyTorch 的 `nn.Linear` 层的封装，注入了 LoRA 的逻辑。它继承自 `LoraLayer`，而 `LoraLayer` 是所有 LoRA 模块的基类。

---

**LoRA 矩阵 A 和 B 的定义位置**

这些矩阵是在 `update_layer()` 方法中创建的，该方法负责初始化 LoRA 参数。


```python
def update_layer(
    self,
    adapter_name,
    r,
    lora_alpha,
    lora_dropout,
    init_lora_weights,
    use_rslora,
    use_dora,
    lora_bias
):
    ...
    # 创建 LoRA 的 A 和 B 矩阵
    self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
    self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)

    if use_rslora:
        self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
    else:
        self.scaling[adapter_name] = lora_alpha / r
```



| 变量 | 含义 |
|------|------|
| `self.lora_A[adapter_name]` | 输入 → 低秩空间映射，$ A \in \mathbb{R}^{r \times d_{\text{in}}} $ |
| `self.lora_B[adapter_name]` | 低秩空间 → 输出映射，$ B \in \mathbb{R}^{d_{\text{out}} \times r} $ |
| `self.scaling[adapter_name]` | 缩放因子 $ \frac{\alpha}{r} $ 或 $ \frac{\alpha}{\sqrt{r}} $ |

---

**前向传播（forward）**


```python
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    result = self.base_layer(x, *args, **kwargs)
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        result += lora_B(lora_A(dropout(x))) * scaling
    return result
```

**流程解释**

1. 先经过原始线性层：`result = base_layer(x)`
2. 如果启用了 LoRA，则执行以下操作：
   - 对输入做 dropout
   - 经过低秩映射：`lora_A(dropout(x))`
   - 再映射回输出空间：`lora_B(...)`
   - 最后乘以缩放因子：`* scaling`
3. 将 LoRA 的输出加到原始结果上：`result += ...`
4. 返回最终输出

---

**权重合并（merge）**

在训练完成后，可以将 LoRA 权重合并进原始模型中，以便加速推理。

函数入口

```python
def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
```

合并公式：

```python
delta_weight = self.get_delta_weight(active_adapter)
base_layer.weight.data += delta_weight.to(orig_dtype)
```

实现细节（`get_delta_weight` 函数）：

```python
def get_delta_weight(self, adapter) -> torch.Tensor:
    weight_A = self.lora_A[adapter].weight
    weight_B = self.lora_B[adapter].weight
    output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
    return output_tensor
```

即：

$$
\Delta W = B A^T \cdot \frac{\alpha}{r}
$$

然后将其加到原始权重矩阵中：

$$
W' = W + \Delta W
$$

---
