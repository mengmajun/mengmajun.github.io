+++ 
date = '2025-06-17' 
draft = false 
title = '代码阅读：Transformer' 
categories = ['代码阅读'] 
tags = ['代码阅读', 'Transformer'] 
+++

## 维度变化

- Transformer 接受的src源头语言、trg目标语言，维度大小[batch_size, seq_len]
- Encoder 接受输入： src 源头语言是[batch_size, seq_len]，src_mask 填充token mask 是[batch_size, 1, 1, seq_len] 第二个维度是head
  - embedding 维度是 [src_vocab_size, d_model] src_vocab_size源头语言的词表大小，d_model模型词向量的维度
  - position embedding 维度是 [seq_len, d_model] seq_len是序列长度，如果输入的句子没有seq_len也会填充的相同长度，所以需要mask，屏蔽填充位置不计算注意力
  - encoder layer 接受原始输入和上一个encoder layer的输出out，mask
    - multihead attention 接受4个参数，q=k=v=x(输入) mask, 输入x的维度是[batch_size, seq_len, d_model], 经过线性变换后，qkv维度是[batch_size, seq_len, d_model]，经过多头切分后[batch_size, seq_len, head_num, d_model / head_num], 计算完成后[batch_size, seq_len, d_model]
  - layer norm
  - ffn
  - layer norm
- Decoder 接受输入： trg 目标语言是[batch_size, seq_len]，trg_mask 未来位置token mask 是[batch_size, 1, seq_len, seq_len] 第二个维度是head
  - 其余维度变化和encoder一致的


## 词向量

- x 输入维度 [batch_size, seq_len], 输出维度 [batch_size, seq_len, d_model]
- d_model 嵌入维度
  
```python
import torch.nn as nn


class Embedder(nn.Module):
    """
    Embedding class used to embed the inputs
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        According to section 3.4 of Attention is All You Need,
        the embeddings are multiplied by square root of
        d_model

        目的是为了保持嵌入值的尺度与后续的位置编码（positional encoding）相匹配（因为位置编码没有经过缩放）
        """
        input_embeddings = self.embedding(x) * (self.d_model**0.5)
        return input_embeddings


if __name__ == '__main__':
    import math
    import torch

    # =============================
    # ✅ 测试代码开始
    # =============================

    # 设置参数
    vocab_size = 10000   # 词汇表大小
    d_model = 512        # 嵌入维度
    batch_size = 2       # 批次大小
    seq_len = 5          # 序列长度

    # 实例化 Embedder
    embedder = Embedder(vocab_size=vocab_size, d_model=d_model)

    # 构造输入（随机 token ID）
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    # 前向传播
    output_embeddings = embedder(input_ids)


    # 查看结果
    print("Input token IDs shape: ", input_ids.shape)
    print("Output embeddings shape: ", output_embeddings.shape)
    print("\nFirst embedding vector (before scaling):")
    print(embedder.embedding(input_ids)[0][0])  # 第一个样本的第一个 token 的原始嵌入
    print("\nFirst embedding vector (after scaling):")
    print(output_embeddings[0][0])              # 缩放后的结果
    print("\nScaling factor (√d_model):", math.sqrt(d_model))


    """"
    Input token IDs shape:  torch.Size([2, 5])
    Output embeddings shape:  torch.Size([2, 5, 512])
    """"
```

## 位置向量PE

- x 输入维度 [batch_size, seq_len], 输出维度 [seq_len, d_model],  输出的是输入序列长度每个token对应的位置向量
- d_model 嵌入维度

```python
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """
    Implements the positional encodings based
    on section 3.5 in Attention is All you Need
    位置编码的作用是用来区分 token 的顺序位置（token-level），而不是词向量内部元素的位置（dimension-level），每个 token 得到一个唯一的 D 维向量作为其位置编码，用于表示它在序列中的位置。给整个 token 向量加上一个“偏移”或“标识”。
    大模型在学习时候能够像分解电磁波为不同频率的波一样，能够把不同位置的token给分解出来
    """

    def __init__(self, max_seq_len, d_model):
        """
        Args:
            max_len(int): maximum length of the input
            d_model: embedding size
        """
        super().__init__()
        self.positional_encodings = torch.zeros(max_seq_len, d_model)
        # arange创建一个0到max_seq_len-1的数组，unsqueeze 在指定位置增加一个新轴，最后的形状是 (max_seq_len, 1)
        positions = torch.arange(max_seq_len).unsqueeze(1)

        # 分母 生成从 0 开始步长为 2 的序列（偶数索引），例如 [0, 2, 4, ..., d_model-2]
        division_term = 10000 ** (torch.arange(0, d_model, 2) / d_model)
        self.positional_encodings[:, 0::2] = torch.sin(positions / division_term)  # 表示从索引 0 开始每隔两个元素取一个（即所有偶数索引），positions / division_term 是广播运算，结果是一个 (max_seq_len, d_model/2) 的张量。
        self.positional_encodings[:, 1::2] = torch.cos(positions / division_term)  # 奇数位置

    def forward(self, x):
        input_len = x.size()[1]
        return self.positional_encodings[:input_len, :]


# 测试 positional encoding 输出
if __name__ == "__main__":
    max_seq_len = 60
    d_model = 512
    positional_encoder = PositionalEncoder(max_seq_len, d_model)

    # 模拟输入
    x = torch.randn(3, 10, d_model)  # B=3, N=10, D=512
    pe = positional_encoder(x)

    print("Positional Encoding Shape:", pe.shape)  # 应输出: torch.Size([10, 512])  这里的意思是这10个长度序列每个token的位置向量
    print("First Position Vector:\n", pe[0])
```

**使用方式：**
输入x是原始序列[batch_size, seq_len]，经过self.embedding后返回的维度是[batch_size, seq_len, d_model], 然后给每个位置加上位置向量，self.positional_encoding返回的维度是[seq_len, d_model]， 这里会自动广播（从右往左匹配），最后out的维度是[batch_size, seq_len, d_model]

```python
out = self.dropout(self.embedding(x) + self.positional_encoding(x))
```

## 缩放点积注意力

- q k v 输入的原始序列是经过split划分多头后的数据 [batch_szie, seq_len, head_num, d_head], 一般情况下q_len、k_len、v_len都是相同的
- 在encoder中 q k v第二层以后接受的输入都是上一层注意力的输出 [batch_szie, seq_len, head_num, d_head]
- 在decoder中 q第二层以后接受的输入都是上一层注意力的输出 [batch_szie, seq_len, head_num, d_head]， k v是encoder最后一层的输出 [batch_szie, seq_len, head_num, d_head]
- matmul = torch.einsum("bqhd,bkhd->bhqk", [q, k]) 爱因斯坦求和函数，意思是q的现状是bqhd，k的形状是bkhd，计算后的结果形状是bhqk，整个计算过程会自动进行转置

```python

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention as described in
    section 3.2.1 of Attention is All You Need
    缩放点积注意力
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, q, k, v, mask=None):
        # 这里有mask的原因是因为在编码器中不需要mask，在解码器的第一层需要mask实现因果注意力

        # 单头注意力，计算q乘以k的转置，这里使用的是爱因斯坦求和
        # 输入q的形状bqhd， k的形状bkhd，  输出形状 bhqk
        # b=batch size， q=query length  h=head number  d=head dimension  k=key length
        matmul = torch.einsum("bqhd,bkhd->bhqk", [q, k])

        # 等价写法
        # q: (B, Q, H, D)
        # k: (B, K, H, D)

        # 转置成 (B, H, Q, D) 和 (B, H, D, K)
        # q_t = q.transpose(1, 2)  # -> (B, H, Q, D)
        # k_t = k.transpose(1, 2).transpose(2, 3)  # -> (B, H, D, K)

        # 矩阵乘法：(B, H, Q, D) x (B, H, D, K) -> (B, H, Q, K)
        # matmul = torch.matmul(q_t, k_t)

        # 当 d_model 较大时，QK_T的内积可能变得非常大，导致 softmax 的梯度趋近于零，影响训练稳定性
        # scaled_matmul 的结果是一个 shape 为 [batch, heads, seq_len_q, seq_len_k] 的张量
        scaled_matmul = matmul / (self.d_model**0.5)

        if mask is not None:
            # masked_fill 会把 mask 中为 0 的位置替换为极小值（接近负无穷），这样 softmax 后这些位置的概率趋近于 0；
            # 注意这里的mask是一个true/false的布尔张量，mask == 0就表示的是false的位置填充为极小值
            # 这里的意思是每个batchsize，也就是每个句子，会对应一个mask，如果是encoder mask是一个pad位置为false的mask，因为屏蔽掉那些 padding 的位置，避免它们参与注意力权重的计算
            # 如果是decoder的第一层attention mask是一个当前token后续未来位置为false的mask，因为未来位置不需要计算注意力分数
            # 这里会使用pytorch的自动广播机制，从最右边的维度开始对齐，mask被广播为[batch, heads, seq_len_q, seq_len_k]
            scaled_matmul = scaled_matmul.masked_fill(mask == 0, float(1e-20))  # 等价于 masked_fill(mask == False, float(1e-20))

        # 在最后一个维度（特征 维度）上做 softmax，得到注意力权重
        softmax = torch.softmax(scaled_matmul, dim=-1)
        attention = torch.einsum("bhqk, bvhd->bqhd", [softmax, v])

        return attention

```


## 多头注意力MHA

- q k v 输入 [batch_szie, seq_len, d_model], 一般情况下q_len、k_len、v_len都是相同的
- 输出维度 [batch_size, seq_len, d_model]
- 在encode decoder中复用

```python
import torch.nn as nn

from scaled_dot_product_attention import ScaledDotProductAttention


class MultiheadAttention(nn.Module):
    """
    Implements multi-head attention as described in section 3.2.2 of Attenton is All You Need.
    """

    def __init__(self, d_model, heads_num):
        super().__init__()
        # 嵌入维度
        self.d_model = d_model
        # 头的数量
        self.heads_num = heads_num
        # 每个头的维度大小
        self.d_heads = self.d_model // self.heads_num
        assert (
            self.d_heads * self.heads_num == self.d_model
        ), "Embedding size must be divisible by number of heads"

        # 线性变换矩阵
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.attention = ScaledDotProductAttention(d_model)
        self.w_o = nn.Linear(self.heads_num * self.d_heads, self.d_model)

    def split(self, tensor):
        """
        Splits tensor by number of heads, self.heads_num creating an extra dim
        将数据划分为多个头
        Args:
            tensor(nn.tensor): tensor of size [batch_size, tensor_len, d_model]

        Returns:
            tensor(nn.tensor): reshaped input tensor of size [batch_size, tensor_len, heads_num, d_tensor]
        """

        batch_size, tensor_len, tensor_dim = tensor.size()
        return tensor.reshape(
            batch_size, tensor_len, self.heads_num, tensor_dim // self.heads_num
        )

    def concat(self, tensor):
        """
        Concatenates the input tensor, opposite of self.split() by reshaping
        拼接多个头
        Args:
            tensor(nn.tensor): tensor of size [batch_size, tensor_len, heads_num, heads_dim]

        Returns:
            tensort(nn.tensort): reshaped input tensor of size [batch_size, tensor_len, heads_num * heads_dim]
        """

        batch_size, tensor_len, heads_num, heads_dim = tensor.size()
        return tensor.reshape(batch_size, tensor_len, heads_num * heads_dim)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 维度变化 
        # batch_size x q_len x d_model => batch_size x q_len x heads_num x d_heads
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        attention_out = self.attention(q, k, v, mask)
        # 拼接每个头，结果维度为 [batch_size, seq_len, d_model]
        attention_concat = self.concat(attention_out)
        multihead_attenton_out = self.w_o(attention_concat)
        return multihead_attenton_out


if __name__ == '__main__':
    import torch
    import torch.nn as nn

    # =============================
    # 参数设置
    # =============================
    B, L, D = 1, 5, 512  # batch size, seq_len, d_model
    H = 8  # number of heads
    D_HEAD = D // H  # head dimension = 64

    # 随机输入
    x = torch.randn(B, L, D)

    # =============================
    # 方法 A：为每个 head 定义独立线性层
    # =============================
    heads_q = [nn.Linear(D, D_HEAD, bias=False) for _ in range(H)]

    # 手动对每个 head 做线性变换
    q_list = [h(x) for h in heads_q]  # list of (B, L, D_HEAD)

    # 拼接成 multi-head 形式
    q_a = torch.stack(q_list, dim=2)  # shape: (B, L, H, D_HEAD)

    # =============================
    # 方法 B：使用一个大的线性层 + reshape
    # =============================
    w_q = nn.Linear(D, D, bias=False)

    # 把 w_q.weight 初始化成与上面 heads_q 一样的值（横向拼接）
    with torch.no_grad():
        # 初始化大线性层的权重
        for i in range(H):
            # 将每个小线性层的权重复制到大线性层的对应块中
            # 大线性层的形状是 (D, D) = (512, 512)
            # 每个块权重的形状是 (D_HEAD, D) = (64, 512)
            w_q.weight[i * D_HEAD:(i + 1) * D_HEAD] = heads_q[i].weight

    # 做一次线性变换后 reshape
    q_b = w_q(x).reshape(B, L, H, D_HEAD)  # shape: (B, L, H, D_HEAD)

    # =============================
    # 验证结果是否一致
    # =============================
    print("Max diff:", (q_a - q_b).abs().max().item())

    print(torch.allclose(q_a, q_b, atol=1e-6))  # 允许最大 1e-6 的误差
```


## 逐点前馈网络（Point-wise Feed Forward Network）


逐点前馈网络本质上是两层全连接层，对输入的每个位置（token）独立进行相同的变换，因此称为“逐点”（point-wise）。其作用是：
1. **增加模型的表达能力**：通过非线性激活函数引入复杂映射。
2. **维度变换**：先扩展维度，再压缩回原始维度，类似“瓶颈结构”。

```python
import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    """
    Implements the point-wise feed-forward sublayer
    used in the Encoder and Decoder as describe in
    section 3.3 of Attention is All You Need:
    It consists of two linear transformations with a
    ReLU activation in between.
    """

    def __init__(self, d_model, forward_expansion):
        """
        Args:
            d_model(int): embedding size
            forward_expansion(int): the multiple that determines
                                    the inner layers' dim, e.g. 4
                                    according to the paper, 2048 = d_model * 4
        """
        super().__init__()
        self.d_model = d_model
        self.point_wise_ff = nn.Sequential(
            nn.Linear(d_model, d_model * forward_expansion),
            nn.ReLU(),
            nn.Linear(d_model * forward_expansion, d_model),
        )

    def forward(self, x):
        return self.point_wise_ff(x)

```

## 编码器Encoder


```python

import copy
import torch.nn as nn


from sublayers.multihead_attention import MultiheadAttention
from sublayers.point_wise_feed_forward import PointWiseFeedForward
from embeddings.embedder import Embedder
from embeddings.postional_encoder import PositionalEncoder


class EncoderLayer(nn.Module):
    """
    The implementation of a single Encoder layer.
    A stack of these will be used to build
    the encoder portion of the Transformer
    """

    def __init__(self, d_model, heads_num, forward_expansion, dropout):
        super().__init__()
        self.multihead_attention = MultiheadAttention(d_model, heads_num)
        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.point_wise_feed_forward = PointWiseFeedForward(d_model, forward_expansion)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        multihead_attention = self.multihead_attention(q=x, k=x, v=x, mask=mask)
        attention_layer_norm = self.attention_layer_norm(
            x + self.dropout(multihead_attention)
        )
        pwff = self.point_wise_feed_forward(attention_layer_norm)
        out = self.ff_layer_norm(pwff + self.dropout(pwff))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        d_model,
        max_seq_len,
        heads_num,
        forward_expansion,
        dropout,
        layers_num,
    ):
        super().__init__()
        self.embedding = Embedder(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoder(max_seq_len, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, heads_num, forward_expansion, dropout) for _ in range(layers_num)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 词向量+位置向量 [batch_size, seq_len, d_model]
        out = self.dropout(self.embedding(x) + self.positional_encoding(x))
        for encoder_layer in self.encoder_layers:
            out = encoder_layer(out, mask)
        # 输出  [batch_size, seq_len, d_model]
        return out

```


## 解码器Decoder

```python

import copy
import torch
import torch.nn as nn


from sublayers.multihead_attention import MultiheadAttention
from sublayers.point_wise_feed_forward import PointWiseFeedForward
from embeddings.embedder import Embedder
from embeddings.postional_encoder import PositionalEncoder


class DecoderLayer(nn.Module):
    """
    Implements a decoder layer. A stack of these layers
    will be used to build the decoder portion of the transformer
    """

    def __init__(self, d_model, heads_num, forward_expansion, dropout):
        super().__init__()
        self.multihead_attention = MultiheadAttention(d_model, heads_num)
        self.attention_layer_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attention = MultiheadAttention(d_model, heads_num)
        self.enc_dec_att_layer_norm = nn.LayerNorm(d_model)
        self.point_wise_feed_forward = PointWiseFeedForward(d_model, forward_expansion)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out, x, mask):
        # Compute Multi_head attention with masking
        self_attention = self.multihead_attention(q=x, k=x, v=x, mask=mask)

        # Add & Norm
        self_attention_norm = self.attention_layer_norm(x + self.dropout(self_attention))

        # Encoder-Decoder attention
        enc_dec_attention = self.encoder_decoder_attention(q=x, k=enc_out, v=enc_out)

        # Add & Norm
        enc_dec_att_norm = self.attention_layer_norm(
            self_attention_norm + self.dropout(enc_dec_attention)
        )

        # Feed forward
        pwff = self.point_wise_feed_forward(enc_dec_att_norm)

        # Add & Norm
        out = self.ff_layer_norm(pwff + self.dropout(pwff))
        return out


class Decoder(nn.Module):
    """
    Consists of a stack of DecoderLayer()s
    """

    def __init__(
        self,
        trg_vocab_size,
        d_model,
        max_seq_len,
        heads_num,
        forward_expansion,
        dropout,
        layers_num,
    ):
        super().__init__()
        self.embedding = Embedder(trg_vocab_size, d_model)
        self.positional_encoding = PositionalEncoder(max_seq_len, d_model)
        self.decoder = nn.ModuleList(
            [
                copy.deepcopy(
                    DecoderLayer(d_model, heads_num, forward_expansion, dropout)
                )
                for _ in range(layers_num)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, trg_vocab_size)

    def forward(self, enc_out, x, mask):
        out = self.dropout(self.embedding(x) + self.positional_encoding(x))
        for decoder_layer in self.decoder:
            out = decoder_layer(enc_out, out, mask)
        dso = self.linear(out)  # out维度是 batch_size、seq_len、d_model
        out = torch.softmax(dso, dim=-1)  # # out维度是 batch_size、seq_len、trg_vocab_size, 得到的是每个token对应的概率分布
        return out

```

## 模型Transformer

```python
import torch
import torch.nn as nn

from layers.encoder import Encoder
from layers.decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        src_pad_idx,
        trg_vocab_size,
        trg_pad_idx,
        d_model,
        max_seq_len,
        heads_num,
        forward_expansion,
        dropout,
        layers_num,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            max_seq_len,
            heads_num,
            forward_expansion,
            dropout,
            layers_num,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            d_model,
            max_seq_len,
            heads_num,
            forward_expansion,
            dropout,
            layers_num,
        )

    def make_src_mask(self, src):
        # 创建布尔掩码，表示哪些位置不是填充 增加两个维度，变成形状 [batch_size, 1, 1, seq_len]
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.size()
        # 创建下三角矩阵(下三角为1，上三角为0)，确保每个位置只能看到当前及之前的位置（即遮蔽未来词）。
        # torch.tril(torch.ones(3, 3))
        # Out[4]:
        # tensor([[1., 0., 0.],
        #         [1., 1., 0.],
        #         [1., 1., 1.]])
        # 扩展成适合批量数据的形状[batch_size, 1, seq_len, seq_len]。
        return torch.tril(torch.ones(trg_len, trg_len)).expand(batch_size, 1, trg_len, trg_len)

    def forward(self, src, trg):
        # src trg维度 batch_size、seq_len
        src_mask = self.make_src_mask(src)  # [batch_size, 1, 1, seq_len]
        trg_mask = self.make_trg_mask(trg)  # [batch_size, 1, seq_len, seq_len]
        encoder_out = self.encoder(src, src_mask)
        # 解码器第二层的cross-attention 使用encoder_out作为k、v
        decoder_out = self.decoder(encoder_out, trg, trg_mask)  # [batch_size, seq_len, trg_vocab_size]  输出的是目标序列每个位置基于前面所有位置token的概率分布
        return decoder_out


if __name__ == '__main__':
    transformer_model = Transformer(5, 0, 10, 0, 512, 5, 8, 2, 0.1, 6)

    print(transformer_model)

    # 假设经过token化
    src = torch.randint(0, 4, (2, 5))

    trg = torch.randint(0, 9, (2, 5))
    
    # 输出目标语言上的概率分布
    out = transformer_model(src, trg)

    print(out.shape)

    print(out)
```
