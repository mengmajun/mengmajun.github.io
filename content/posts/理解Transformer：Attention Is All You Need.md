+++
date = '2025-06-09'
draft = false
title = '理解Transformer：Attention Is All You Need'
categories = ['人工智能', '大模型']
tags = ['经典论文']
+++

# 理解Transformer：Attention Is All You Need

> “Attention is all you need” 是 Google Brain 和多伦多大学于2017年发表的一篇革命性论文，提出了全新的深度学习架构——**Transformer**。该模型彻底抛弃了传统的循环神经网络（RNN）和卷积神经网络（CNN），仅使用注意力机制（Attention Mechanism）来建模序列数据，在机器翻译等领域取得了巨大成功，并启发了后续一系列自然语言处理模型的发展（如BERT、GPT等）。


---

## 一、整体架构概览

![Transformer整体结构图](https://ask.qcloudimg.com/http-save/yehe-2510408/3bf0187d209e2cc79f28d6be94685fc2.png)

Transformer 主要由两部分组成：

- **编码器（Encoder）**：接收输入序列（如源语言句子），提取其语义信息。
- **解码器（Decoder）**：根据编码器输出，逐步生成目标序列（如翻译后的句子）。

整个模型没有使用任何 RNN 或 CNN 层，完全依赖**自注意力机制（Self-Attention）** 和 **前馈网络（Feed-Forward Network）** 完成信息传递和特征提取。

---

## 二、编码器详解

### 1. 编码器的作用

编码器负责将输入序列（如一个英文句子）转换成一组上下文相关的隐藏表示（Hidden States）。这些表示包含了每个词在句子中的全局语义信息。

### 2. 编码器的结构

每一层编码器由两个子层组成：

1. **多头自注意力机制（Multi-Head Self-Attention）**
2. **前馈神经网络（Feed-Forward Network, FFN）**

这两个子层都采用了残差连接（Residual Connection）和层归一化（Layer Normalization）以提高训练稳定性。

### 3. 输出用途

编码器最终输出的所有位置的隐藏状态（Hidden States），会被用作解码器中跨注意力机制的 Key（K）和 Value（V）。

---

## 三、解码器详解

### 1. 解码器的作用

解码器负责根据编码器输出的信息，逐个生成目标序列（如翻译结果）。

### 2. 解码器的结构

每一层解码器包含三个子层：

1. **带掩码的多头自注意力（Masked Multi-Head Self-Attention）**
   - 只能看到当前位置及之前的位置，防止未来信息泄露。
2. **多头跨注意力（Multi-Head Cross-Attention）**
   - Query 来自上一层的输出，Key 和 Value 来自编码器的输出。
   - 实现“关注”源语言中相关部分的能力。
3. **前馈神经网络（FFN）**

同样使用残差连接和层归一化。

---

## 四、Embedding层详解

### 1. 什么是 Embedding？

Embedding 将离散的 token（如单词、字符）映射为连续的向量空间中的表示，使得模型可以更好地捕捉它们之间的语义关系。

### 2. Token Embedding

Token Embedding 是最基本的嵌入形式。例如，给定词汇表大小为 V，我们可以定义一个维度为 $d_{model}$ 的嵌入矩阵，将每个词 ID 映射为一个 d_model 维的向量。

```python
import torch.nn as nn
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=512)
```

### 3. Positional Embedding（位置编码）

由于 Transformer 没有使用 RNN，无法自动捕捉序列顺序信息，因此需要显式加入**位置编码**（Positional Encoding）。

位置编码可以通过学习得到，也可以使用固定的函数生成。原始论文中采用的是正弦和余弦函数组合的形式：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

这样做的好处是模型可以泛化到比训练时更长的序列。

---

## 五、注意力机制详解

### 1. 点积注意力（Scaled Dot-Product Attention）

这是注意力机制的核心公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：
- Q：Query 向量
- K：Key 向量
- V：Value 向量
- $d_k$：Key 向量的维度，用于缩放防止内积过大导致梯度消失。

![Scaled Dot-Product Attention](https://transformers.run/assets/img/attention/attention.png)

### 2. 多头注意力（Multi-Head Attention）

为了增强模型对不同位置、不同表示子空间的关注能力，Transformer 使用了**多头注意力机制**。

基本思想是将 Q、K、V 分别线性变换到多个不同的子空间（称为 Head），然后分别计算注意力，最后拼接并再次线性变换。

![Multi-Head Attention](https://ask.qcloudimg.com/http-save/yehe-2510408/3650a8ae29510a9036270ed8dfd99fdb.png)

数学表达如下：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$
其中：
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

---

## 六、Masked Multi-Head Attention 计算过程

### [图示计算过程](https://excalidraw.com/#json:abc123,def456)

### 1. 为什么需要 Mask？

在解码过程中，我们希望模型只能看到当前时刻之前的输出，不能提前看到未来的词。因此在解码器的第一层注意力中引入了**因果掩码（Causal Mask）**。

### 2. Causal Mask 的实现方式

通过构造一个上三角矩阵（只保留左下角的元素），将未来位置的注意力权重设为负无穷（-∞），从而在 softmax 中忽略这些位置。

```python
def causal_mask(size):
    # 生成上三角矩阵（对角线以上为1，以下为0）
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    # 将1转换为负无穷，0保留
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# 示例：3x3的因果掩码
print(causal_mask(3))

# 输出
tensor([[0., -inf, -inf],
        [0., 0., -inf],
        [0., 0., 0.]])
```

### 3. 为什么要用负无穷而不是 0 或 -1？

- **Softmax 函数特性**：指数会将负无穷变为 0，确保未来位置不会参与计算。
- **数值稳定性**：避免因极小值带来的数值误差或梯度爆炸问题。

---

## 七、训练细节

### 1. 损失函数（目标函数）

Transformer 的主要应用场景是**机器翻译**，因此它使用的是标准的**交叉熵损失函数（Cross-Entropy Loss）**。

#### 目标函数公式：

$$
\mathcal{L} = - \sum_{t=1}^{T_y} \log p(y_t | y_1, ..., y_{t-1}, x)
$$

其中：
- $x$：输入序列（如英文句子）
- $y_t$：目标序列第 $t$ 个词（如法语句子中的一个词）
- 模型输出的是对下一个词的概率分布预测
- 使用交叉熵衡量预测与真实标签之间的差异

为了提高训练效率，通常还会结合 **Label Smoothing** 技术，防止模型对某些类别过于自信。


## 八、总结：Transformer 的优势

| 优势 | 描述 |
|------|------|
| 并行化强 | 不依赖 RNN，可并行处理所有位置，训练效率高 |
| 长程依赖 | 自注意力机制天然适合建模远距离依赖 |
| 结构统一 | 编码器和解码器结构相似，易于扩展和复用 |
| 可解释性强 | 注意力权重可视化有助于理解模型行为 |

---

