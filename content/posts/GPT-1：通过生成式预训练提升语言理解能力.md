+++
date = '2025-05-29'
draft = false
title = 'GPT-1：通过生成式预训练提升语言理解能力'
categories = ['GPT-1', '经典论文']
tags = ['GPT-1', 'Attention机制', '经典论文']
+++
---
##  通过生成式预训练提升语言理解能力

[论文链接：Improving Language Understanding by Generative Pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).

- "Generative Pre-Training" - “生成式预训练”：“生成式” 对应模型基于自回归（Autoregressive）的生成特性（如按序预测下一个 token）；“预训练” 明确体现无监督预训练阶段的核心任务。
- "Improving Language Understanding" - “提升语言理解能力”：虽 GPT-1 以生成任务为核心，但论文中也验证了其在下游理解任务（如问答、文本分类）的迁移能力，“理解” 一词兼顾了模型的双向价值（生成与理解）

---


## 引言

GPT-1（Generative Pre-trained Transformer）是 OpenAI 在 2018 年提出的一种基于 Transformer 的语言模型。它首次系统性地将“**预训练 + 微调**”的范式引入自然语言处理领域，为后续的大规模语言模型（如 GPT-2、GPT-3 和 ChatGPT）奠定了坚实的基础。

本文将从最基础的概率知识讲起，逐步深入解析 GPT-1 的核心思想、模型结构、注意力机制、训练目标及其数学原理，帮助读者全面理解这一开创性工作的技术细节。

---

## 一、基础概率知识回顾

### 1.1 联合概率（Joint Probability）

联合概率描述的是多个事件**同时发生**的概率。

例如，一个句子 $ w_1, w_2, ..., w_n $ 的联合概率表示为：

$$
P(w_1, w_2, \dots, w_n)
$$

在语言模型中，我们希望模型能给出一个句子出现的整体概率。

### 1.2 条件概率（Conditional Probability）

条件概率表示在已知某些事件发生的前提下，另一事件发生的概率：

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
$$

在语言建模任务中，我们更关注每个词在前面词序列下的条件概率：

$$
P(w_i | w_1, w_2, \dots, w_{i-1})
$$

### 1.3 链式法则（Chain Rule）

链式法则是将联合概率分解为一系列条件概率的乘积：

$$
P(w_1, w_2, \dots, w_n) = \prod_{i=1}^n P(w_i | w_1, \dots, w_{i-1})
$$

这是语言模型建模的核心方法。由于直接估计高维联合分布非常困难，通过条件概率逐个预测下一个词成为主流做法。

---

## 二、语言模型的基本概念

语言模型的目标是给定一个词序列 $ w_1, w_2, \dots, w_n $，计算其联合概率：

$$
P(w_1, w_2, \dots, w_n)
$$

通常使用链式法则将其转换为一系列条件概率：

$$
P(w_1, w_2, \dots, w_n) = \sum_{i=1}^n \log P(w_i | w_1, \dots, w_{i-1})
$$

这种形式便于模型建模和优化。GPT 系列属于**自回归语言模型（Autoregressive Language Model）**，即通过前面若干个词来预测下一个词。

---

## 三、GPT-1 的核心思想

GPT-1 的核心贡献在于提出了“**预训练 + 微调**”的范式：

1. **预训练阶段**：在大量无标注语料上训练一个通用的语言模型；
2. **微调阶段**：在具体任务（如分类、问答等）上有监督地微调模型，使其适应特定任务。

这种方式借鉴了计算机视觉中的迁移学习思想，在 NLP 领域取得了突破性进展。

---

## 四、模型架构详解

GPT-1 使用的是 **Transformer 解码器（Decoder-only）** 结构。相比原始的 Transformer 模型，GPT-1 只保留了解码器部分。

### 4.1 Transformer 解码器结构

每个解码器层包括两个主要组件：

1. **掩码多头自注意力机制（Masked Multi-head Self-Attention）**
2. **前馈神经网络（Feed-forward Network）**

由于是自回归模型，只能看到当前位置之前的词，因此在自注意力机制中加入了**掩码（masking）**。

### 4.2 注意力机制详解（重点）

#### 4.2.1 基本公式

标准的注意力机制定义如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中：
- $ Q $：查询向量（Query）
- $ K $：键向量（Key）
- $ V $：值向量（Value）
- $ d_k $：缩放因子，防止点积过大导致 softmax 梯度消失

#### 4.2.2 多头注意力（Multi-head Attention）

GPT-1 使用多头注意力机制，将输入映射到多个不同的子空间中并行处理，最后拼接输出：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O
$$

其中：

$$
head_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
$$

多头机制增强了模型对不同位置关系的捕捉能力。

#### 4.2.3 掩码机制（Masking）

为了保证模型仅依赖于前面的词进行预测，GPT-1 在注意力矩阵中加入了一个**下三角掩码（Lower Triangular Mask）**：

$$
M_{ij} =
\begin{cases}
0 & i \geq j \\
-\infty & i < j
\end{cases}
$$

这样确保了模型在预测第 $ i $ 个词时，只能看到 $ 1 $ 到 $ i-1 $ 个词的信息。

---

## 五、输入表示与位置编码

GPT-1 的输入由以下三部分组成：

1. **词嵌入（Token Embeddings）**：将输入词映射为向量。
2. **位置编码（Positional Encodings）**：表示词的位置信息，使模型能感知顺序。
3. **段落嵌入（Segment Embeddings）**：用于区分不同句子（如问答任务中的问题和答案）。

最终输入表示为三者之和：

$$
\text{Input} = W_e x_i + W_p p_i + W_s s_i
$$

其中：
- $ x_i $ 是第 $ i $ 个词的 one-hot 向量；
- $ p_i $ 是位置索引；
- $ s_i $ 是段落标识。

> 注：与 BERT 不同，GPT-1 使用的是**学习的位置编码**（Learned Positional Embedding），而不是固定的正弦/余弦函数。

---

## 六、训练过程详解

### 6.1 预训练目标：最大似然估计（MLE）

GPT-1 的预训练目标是一个标准的**语言建模任务**，即最大化似然函数：

$$
L_1(T) = \sum_{i=1}^n \log P(w_i | w_1, ..., w_{i-1}; \theta)
$$

其中 $ T = (w_1, ..., w_n) $ 是一段文本，$ \theta $ 是模型参数。

这个目标函数鼓励模型对真实文本中的下一个词做出高概率预测。

### 6.2 Softmax 分布与损失函数

模型最后一层输出是一个 logits 向量，表示各个候选词的得分。然后通过 softmax 函数转化为概率分布：

$$
P(w_i | w_{<i}) = \frac{\exp(h_i^\top e_{w_i})}{\sum_{j} \exp(h_i^\top e_j)}
$$

其中：
- $ h_i $ 是模型最后一层的隐藏状态；
- $ e_j $ 是词 $ j $ 的嵌入向量。

损失函数为负对数似然（Negative Log-Likelihood, NLL）：

$$
\mathcal{L} = -\sum_{i=1}^n \log P(w_i | w_{<i})
$$

通过梯度下降（如 Adam）进行优化。

---

## 七、微调阶段与任务适配

在微调阶段，GPT-1 将预训练的语言模型适配到具体的下游任务，例如：

- 文本分类
- 句子关系判断（如 MNLI）
- 问答系统（如 QNLI）

### 7.1 输入格式与任务适配方式

对于每个任务，GPT-1 在输入中添加一个特殊的开始标记（`<s>`）和结束标记（`<e>`），并在最后添加一个额外的线性层用于分类或回归任务。

例如，在文本分类任务中，模型结构如下：

```
<s> [input sentence] <e> → Transformer → Linear → Softmax
```

此时的损失函数变为任务相关的损失（如交叉熵损失）：

$$
\mathcal{L}_{task} = -\log P(y | x; \theta)
$$

### 7.2 多任务联合训练

除了单独微调，GPT-1 还尝试在微调过程中加入原始语言模型目标作为辅助任务：

$$
\mathcal{L}_{total} = \lambda \cdot \mathcal{L_task} + (1 - \lambda) \cdot \mathcal{L_lm}
$$

这样可以提升模型泛化能力。

---

## 八、总结

GPT-1 是深度学习时代自然语言处理发展的重要里程碑。它不仅引入了强大的 Transformer 架构，还确立了“预训练 + 微调”的标准流程，为后来的大规模语言模型（如 GPT-2、GPT-3、ChatGPT）铺平了道路。

通过本文的讲解，你已经掌握了以下关键知识点：

- 联合概率与条件概率的关系；
- 链式法则在语言模型中的应用；
- 最大似然估计的基本原理；
- Softmax 分布的定义及其在语言模型中的作用；
- GPT-1 的核心思想及其对 NLP 的深远影响；
- 注意力机制的数学推导与实现细节；
- 模型结构、输入表示与训练流程。

