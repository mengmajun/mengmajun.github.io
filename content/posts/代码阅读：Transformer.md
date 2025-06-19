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

## 安装德语 英语 分词器

```python
%%capture
# 安装德语 英语 分词器
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
```

## 配置


```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 8
accumulation_steps = 128
d_model = 512
max_seq_len = 128
head_num = 8
forward_expansion = 4
dropout = 0.1
layers_num = 6
init_lr = 1e-5
BETA1 = 0.9
BETA2 = 0.98
adam_eps = 5e-9
epochs = 1000
warmup = 100
weight_decay = 5e-4
clip = 1
factor = 0.9
patience = 10
save_model_dir = r"./model"
model_path = r"D:\project\python\demo\llm_note\model\model_5.163683279037476.pt"

print(device)
```

## 加载space分词器


```python
import spacy

# 加载分词模型
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

# 对文本进行分词
english_text = "A woman with a wallet."
german_text = "Eine Frau mit einer Geldbörse."

print(tokenize_de(german_text))
print(tokenize_en(english_text))
```

## 加载multi30k数据集


```python
from collections import Counter
from datasets import load_dataset

# -----------------------------
# Step 1: 加载数据集与分词器
# -----------------------------
dataset = load_dataset("bentrevett/multi30k")
```


```python
dataset
```

## 统计词频


```python
# -----------------------------
# Step 2: 收集所有 tokens（构建词表用）
# -----------------------------

counter_en = Counter()
counter_de = Counter()

for split in ['train', 'validation', 'test']:
    for example in dataset[split]:
        en = example['en']
        de = example['de']
        
        # 使用 Qwen tokenizer 进行分词
        tokens_en = tokenize_en(en)
        tokens_de = tokenize_de(de)
        
        counter_en.update(tokens_en)
        counter_de.update(tokens_de)
```


```python
counter_en
```


```python
counter_de
```

## 构建自定义词表（加特殊 token）


```python
# -----------------------------
# Step 3: 构建自定义词表（加特殊 token）
# -----------------------------

MIN_FREQ = 2  # 设置最低频次，过滤低频词

pad_token_id=0
sos_token_id=1 
eos_token_id=2
unk_token_id=3

# 英语词表（目标语言）
vocab_en = ['<pad>', '<sos>', '<eos>', '<unk>'] + [token for token, freq in counter_en.items() if freq >= MIN_FREQ]
en2idx = {token: idx for idx, token in enumerate(vocab_en)}
idx2en = {idx: token for token, idx in en2idx.items()}

# 德语词表（源语言）
vocab_de = ['<pad>', '<sos>', '<eos>', '<unk>'] + [token for token, freq in counter_de.items() if freq >= MIN_FREQ]
de2idx = {token: idx for idx, token in enumerate(vocab_de)}
idx2de = {idx: token for token, idx in de2idx.items()}

print("英语词表大小:", len(vocab_en))
print("德语词表大小:", len(vocab_de))
```

## 构建encode、decode


```python
# -----------------------------
# Step 4: 编码函数（将句子转为 token IDs）
# -----------------------------

def encode(sentence, add_special_tokens=True, lang='de'):
    tokenizer = tokenize_de if lang == 'de' else tokenize_en
    vocab_dict = de2idx if lang == 'de' else en2idx

    tokens = tokenizer(sentence)
    ids = [vocab_dict[token] if token in vocab_dict else vocab_dict['<unk>'] for token in tokens]

    if add_special_tokens:
        ids = [vocab_dict['<sos>']] + ids + [vocab_dict['<eos>']]
    
    return ids

# 示例：编码一个句子
en_sentence = "A woman riding a bicycle on the street."
de_sentence = "Eine Frau fährt auf der Straße Fahrrad."

en_ids = encode(en_sentence, lang='en')
de_ids = encode(de_sentence, lang='de')

print("英语句子编码结果:", en_ids)
print("德语句子编码结果:", de_ids)
```


```python
# -----------------------------
# Step 5: 解码函数（将 token IDs 转回句子）
# -----------------------------

def decode(ids, lang='de'):
    idx2word = idx2de if lang == 'de' else idx2en
    return ' '.join([idx2word.get(idx, '<unk>') for idx in ids])

# 示例：解码
print("解码后的英文句子:", decode(en_ids, lang='en'))
print("解码后的德文句子:", decode(de_ids, lang='de'))
```

## 定义Dataset


```python
from torch.utils.data import DataLoader, Dataset

class TranslationDataset(Dataset):
    def __init__(self, dataset_dict, max_length=128, split='train'):
        """
        初始化翻译数据集
        
        参数:
            dataset_dict: 包含训练/测试数据的DatasetDict对象
            tokenizer: 分词器
            max_length: 输入的最大长度
            split: 使用的数据集划分 ('train' 或 'test')
        """
        self.dataset = dataset_dict[split]
        self.max_length = max_length
        self.eos_token_id = en2idx['<eos>']
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取原始文本
        en_text = self.dataset[idx]['en']
        de_text = self.dataset[idx]['de']
        
        # 将每个句子编码为token序列，并添加<sos> <eos>
        en_tokens = encode(en_text, lang='en')
        de_tokens = encode(de_text, lang='de')
        
        # 截断过长的序列
        if len(en_tokens) > self.max_length:
            en_tokens = en_tokens[:self.max_length]
            en_tokens[-1] = self.eos_token_id  # 确保最后一个token是结束token
        
        if len(de_tokens) > self.max_length:
            de_tokens = de_tokens[:self.max_length]
            de_tokens[-1] = self.eos_token_id  # 确保最后一个token是结束token
            
        return {
            'en': en_text,
            'de': de_text,
            'en_input_ids': torch.tensor(en_tokens),
            'de_input_ids': torch.tensor(de_tokens)
        }
    
```


```python
train_dataset = TranslationDataset(dataset_dict=dataset, split='train')
```


```python
train_dataset[0]
```

## 定义Dataloader


```python
def collate_fn(batch, pad_token_id):
    """
    用于DataLoader的collate函数，对批次内的序列进行填充
    
    参数:
        batch: 一批数据
        pad_token_id: 填充token的ID
    """
    # 获取批次内最长序列长度
    max_en_length = max([len(item['en_input_ids']) for item in batch])
    max_de_length = max([len(item['de_input_ids']) for item in batch])
    
    # 创建填充后的张量
    en_input_ids = torch.full((len(batch), max_en_length), pad_token_id, dtype=torch.long)
    de_input_ids = torch.full((len(batch), max_de_length), pad_token_id, dtype=torch.long)
    
    # 填充序列
    for i, item in enumerate(batch):
        en_input_ids[i, :len(item['en_input_ids'])] = item['en_input_ids']
        de_input_ids[i, :len(item['de_input_ids'])] = item['de_input_ids']
    
    return {
        'en': [item['en'] for item in batch],
        'de': [item['de'] for item in batch],
        'en_input_ids': en_input_ids,
        'de_input_ids': de_input_ids,
        'en_attention_mask': en_input_ids != pad_token_id,
        'de_attention_mask': de_input_ids != pad_token_id
    }



# 创建数据集
train_dataset = TranslationDataset(dataset, max_seq_len, 'train')
test_dataset = TranslationDataset(dataset, max_seq_len, 'test')

print(len(train_dataset))
print(len(test_dataset))

# 创建数据加载器
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, pad_token_id)
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: collate_fn(batch, pad_token_id)
)


```


```python
# 打印一批数据的形状
batch = next(iter(train_dataloader))
print("英文输入形状:", batch['en_input_ids'].shape)
print("中文输入形状:", batch['de_input_ids'].shape)
print("英文注意力掩码形状:", batch['en_attention_mask'].shape)
print("中文注意力掩码形状:", batch['de_attention_mask'].shape)
```


```python
batch['en_input_ids'][0].tolist()
```


```python
batch['en_attention_mask'][0]
```


```python
decode(batch['en_input_ids'][0].tolist(), lang='en')
```


```python
decode(batch['de_input_ids'][0].tolist(), lang='de')
```


```python
batch['de'][0]
```


```python

```

## 定义模型

### 词向量


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
```

### 位置向量


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
        self.positional_encodings = torch.zeros(max_seq_len, d_model, device=device)
        # arange创建一个0到max_seq_len-1的数组，unsqueeze 在指定位置增加一个新轴，最后的形状是 (max_seq_len, 1)
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)

        # 分母 生成从 0 开始步长为 2 的序列（偶数索引），例如 [0, 2, 4, ..., d_model-2]
        division_term = 10000 ** (torch.arange(0, d_model, 2, device=device) / d_model)
        self.positional_encodings[:, 0::2] = torch.sin(positions / division_term)  # 表示从索引 0 开始每隔两个元素取一个（即所有偶数索引），positions / division_term 是广播运算，结果是一个 (max_seq_len, d_model/2) 的张量。
        self.positional_encodings[:, 1::2] = torch.cos(positions / division_term)  # 奇数位置

    def forward(self, x):
        input_len = x.size()[1]
        return self.positional_encodings[:input_len, :]
```

### 缩放点积注意力


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

### 多头注意力


```python
import torch.nn as nn


class MultiheadAttention(nn.Module):
    """
    Implements multi-head attention as described in section 3.2.2 of Attenton is All You Need.
    """

    def __init__(self, d_model, heads_num):
        super().__init__()
        self.d_model = d_model
        self.heads_num = heads_num
        self.d_heads = self.d_model // self.heads_num
        assert (
            self.d_heads * self.heads_num == self.d_model
        ), "Embedding size must be divisible by number of heads"

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.attention = ScaledDotProductAttention(d_model)
        self.w_o = nn.Linear(self.heads_num * self.d_heads, self.d_model)

    def split(self, tensor):
        """
        Splits tensor by number of heads, self.heads_num creating an extra dim

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

        Args:
            tensor(nn.tensor): tensor of size [batch_size, tensor_len, heads_num, heads_dim]

        Returns:
            tensort(nn.tensort): reshaped input tensor of size [batch_size, tensor_len, heads_num * heads_dim]
        """

        batch_size, tensor_len, heads_num, heads_dim = tensor.size()
        return tensor.reshape(batch_size, tensor_len, heads_num * heads_dim)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # split q, k, v into heads, i.e. from batch_size x q_len x d_model => batch_size x q_len x heads_num x d_heads
        q = self.split(q)
        k = self.split(k)
        v = self.split(v)

        attention_out = self.attention(q, k, v, mask)
        attention_concat = self.concat(attention_out)
        multihead_attenton_out = self.w_o(attention_concat)
        return multihead_attenton_out
```

### 逐位置FFN

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

### Decoder


```python
import copy
import torch
import torch.nn as nn


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
        self_attention_norm = self.attention_layer_norm(
            x + self.dropout(self_attention)
        )

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
        dso = self.linear(out)  # out维度是 batch_size、seq_len、trg_vocab_size
        # 注意：交叉熵函数会自动进行softmax
        # out = torch.softmax(dso, dim=-1)  # # out维度是 batch_size、seq_len、trg_vocab_size, 得到的是每个token对应的概率分布
        return dso

```

### Encoder


```python
import copy
import torch.nn as nn



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
        out = self.dropout(self.embedding(x) + self.positional_encoding(x))
        for encoder_layer in self.encoder_layers:
            out = encoder_layer(out, mask)
        return out

```

## 创建Transformer模型


```python
import torch
import torch.nn as nn


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
        mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return mask.to(device)

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.size()
        # 创建下三角矩阵(下三角为1，上三角为0)，确保每个位置只能看到当前及之前的位置（即遮蔽未来词）。
        # torch.tril(torch.ones(3, 3))
        # Out[4]:
        # tensor([[1., 0., 0.],
        #         [1., 1., 0.],
        #         [1., 1., 1.]])
        # 扩展成适合批量数据的形状[batch_size, 1, seq_len, seq_len]。
        mask = torch.tril(torch.ones(trg_len, trg_len)).expand(batch_size, 1, trg_len, trg_len)
        return mask.to(device)

    def forward(self, src, trg):
        # src trg维度 batch_size、seq_len
        src_mask = self.make_src_mask(src)  # [batch_size, 1, 1, seq_len]
        trg_mask = self.make_trg_mask(trg)  # [batch_size, 1, seq_len, seq_len]
        encoder_out = self.encoder(src, src_mask)
        # 解码器第二层的cross-attention 使用encoder_out作为k、v
        decoder_out = self.decoder(encoder_out, trg, trg_mask)  # [batch_size, seq_len, trg_vocab_size]  输出的是目标序列每个位置基于前面所有位置token的概率分布
        return decoder_out
    
    def generate(
            self, 
            src, 
            max_length, 
            temperature=0.0,  # 0.0表示贪心搜索，>0.0表示带温度抽样
            sos_token_id=1, 
            eos_token_id=2, 
            pad_token_id=0
        ):
        """
        生成目标序列（贪心搜索或带温度抽样）
        
        参数:
            src: 输入序列 [batch_size, src_seq_len]
            max_length: 最大生成长度
            temperature: 控制随机性 (0.0=贪心搜索，>0.0=抽样)
            sos_token_id: 序列起始符ID
            eos_token_id: 序列终止符ID
            pad_token_id: 填充符ID
        
        返回:
            生成序列 [batch_size, generated_seq_len]
        """
        batch_size = src.size(0)
        
        # 编码源序列
        src_mask = self.make_src_mask(src)
        encoder_out = self.encoder(src, src_mask)
        
        # 初始化目标序列 (起始符)  使用sos_token_id填充张量
        trg = torch.full(
            (batch_size, 1), 
            sos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # 存储完成状态
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 自回归生成
        for _ in range(max_length):
            # 生成当前步的trg_mask
            trg_mask = self.make_trg_mask(trg)
            
            # 解码器前向传播
            decoder_out = self.decoder(encoder_out, trg, trg_mask)
            # 取最后一个token
            next_token_logits = decoder_out[:, -1, :]
            
            # 应用温度控制
            if temperature > 0.0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 更新序列  将next token加入到序列中 重新输入到decoder进行预测
            trg = torch.cat([trg, next_tokens.unsqueeze(1)], dim=1)
            
            # 检查终止符
            finished = finished | (next_tokens == eos_token_id)
            
            # 全部完成则提前终止
            if finished.all():
                break
        
        # 移除起始符并处理填充
        trg = trg[:, 1:]  # 移除<sos>起始符
        
        # 处理EOS之后的token
        eos_positions = (trg == eos_token_id).float().argmax(dim=1)
        for i in range(batch_size):
            if eos_positions[i] > 0:  # 找到EOS
                trg[i, int(eos_positions[i].item())+1:] = pad_token_id
        
        return trg


```

## 实例化模型


```python
# 英译德
src_vocab_size = len(vocab_en)
trg_vocab_size = len(vocab_de)

src_pad_idx = pad_token_id
trg_pad_idx = pad_token_id

model = Transformer(
        src_vocab_size,
        src_pad_idx,
        trg_vocab_size,
        trg_pad_idx,
        d_model,
        max_seq_len,
        head_num,
        forward_expansion,
        dropout,
        layers_num,
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

print(model)
```

## BLUE指标

常用于机器翻译任务，计算目标序列和候选序列的重叠情况，值越大，模型越好


```python
import math
import numpy as np

from collections import Counter


def compute_bleu(reference, candidate):
    """
    Creates 1-4 grams of the reference and candidate
    and appends the precision of the candidate's to
    a list

    Args:
        reference(str): The ground truth phrase
        candidate(str): The candidate phrase

    Returns:
        bleu_score(int): bleu score across n-grams
    """
    precision = []
    reference_words = reference.split()
    candidate_words = candidate.split()
    for n in range(1, 5):
        reference_ngram = Counter(
            [
                " ".join(reference_words[i : i + n])
                for i in range(len(reference_words) + 1 - n)
            ]
        )
        candidate_ngram = Counter(
            [
                " ".join(candidate_words[i : i + n])
                for i in range(len(candidate_words) + 1 - n)
            ]
        )
        if not candidate_ngram or not reference_ngram:
            continue
        overlap = sum((reference_ngram & candidate_ngram).values())
        precision.append(overlap / sum(candidate_ngram.values()))

    brevity_penalty = (
        1
        if len(candidate) >= len(reference)
        else math.exp(1 - len(candidate) / len(reference))
    )

    bleu_score = brevity_penalty * np.mean(precision) * 100
    return bleu_score



reference = "the cat is on the mat"
candidate = "the cat is on a mat"
candidate2 = "the the the the the the"
print(compute_bleu(reference, candidate))
print(compute_bleu(reference, candidate2))
```

## 创建训练函数


```python
import torch
import torch.nn as nn
from tqdm.auto import trange
from tqdm.auto import tqdm


def train(model, data_iterator, device, optimizer, criterion, clip):
    model.train()
    train_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(enumerate(data_iterator), total=len(data_iterator), desc="Training")
    for i, data in progress_bar:
        src = data['en_input_ids'].to(device)
        trg = data['de_input_ids'].to(device)
        
        output = model(src, trg[:, :-1])
        reshaped_output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        # 计算损失并进行反向传播
        loss = criterion(reshaped_output, trg)
        loss = loss / accumulation_steps  # 缩放损失
        loss.backward()  # 反向传播，累积梯度

        # 每隔accumulation_steps步进行一次梯度裁剪和参数更新
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()  # 重置梯度
            print("Iteration: ", i + 1, "with loss: ", loss.item() * accumulation_steps)

        train_loss += loss.item() * accumulation_steps  # 恢复损失的缩放
        
    # 处理剩余的梯度
    if (i + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad()  # 重置梯度
    
    return train_loss / len(data_iterator)

```

## 创建验证函数


```python
def validate(model, data_iterator, device, criterion):
    model.eval()
    val_loss = 0
    all_trg_sentence = []
    all_predicted_sentence = []
    with torch.no_grad():
        for _, data in enumerate(data_iterator):
            src = data['en_input_ids'].to(device)
            trg = data['de_input_ids'].to(device)

            output = model(src, trg[:, :-1])
            reshaped_output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(reshaped_output, trg)
            val_loss += loss.item()

            for j in range(batch_size):
                # 先转移到cpu，否则gpu训练会报错
                trg_tokens = data['de_input_ids'][j].cpu().tolist()
                # 计算blue移除pad token
                trg_tokens = [token for token in trg_tokens if token != pad_token_id]
                trg_sentence = decode(trg_tokens, lang='de')
                
                predicted_tokens = output[j].max(dim=1)[1].cpu().tolist()
                predicted_sentence = decode(predicted_tokens, lang='de')
                
                all_trg_sentence.append(trg_sentence)
                all_predicted_sentence.append(predicted_sentence)
                
                print(f"trg_tokens: {trg_tokens}\npredicted_tokens: {predicted_tokens}\n")
                print(f"trg_sentence: {trg_sentence}\npredicted_sentence: {predicted_sentence}\n-----------------")


            # break # break用于测试
                
    bleu_scores = 0
    for trg_sentence, predicted_sentence in zip(all_trg_sentence, all_predicted_sentence):
        # print(f"trg_sentence: {trg_sentence}\npredicted_sentence: {predicted_sentence}\n-----------------")
        bleu_score = compute_bleu(reference=trg_sentence, candidate=predicted_sentence)
        bleu_scores += bleu_score
        
    return val_loss / len(data_iterator), bleu_scores / len(all_predicted_sentence)
```


```python

```


```python

```

## 创建训练循环


```python
def run_train_loop(
    model,
    train_data_iterator,
    val_data_iterator,
    device,
    optimizer,
    scheduler,
    criterion,
    clip,
):
    train_losses, val_losses, bleu_scores = [], [], []
    best_loss = float("Inf")
    epochs_bar = trange(epochs, desc="Epochs")
    for epoch in epochs_bar:
        train_loss = train(model, train_data_iterator, device, optimizer, criterion, clip)
        val_loss, bleu_score = validate(model, val_data_iterator, device, criterion)
        
        if epoch > warmup:
            # 根据验证损失调整学习率
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_loss:
            torch.save(model.state_dict(), f"{save_model_dir}/model_{val_loss}.pt")
            best_loss = val_loss

        print(f"Train loss at epoch {epoch} is: {train_loss}")
        print(f"Val loss at epoch {epoch} is {val_loss}")
        print(f"Bleu score at epoch {epoch} is {bleu_score}")
```

## 模型初始化、优化器、损失函数


```python
from torch import optim
from torch.optim import Adam

def iniatialize_weights(model):
    if hasattr(model, "weight") and model.weight.dim() > 1:
        nn.init.kaiming_uniform(model.weight.data)

# 模型初始化
model.apply(iniatialize_weights)

# 继续上次的训练，加载预训练权重（覆盖初始化值）
if model_path:
    # 使用 map_location 参数将模型权重映射到 CPU, 如果权重是在GPU上训练的，默认会把模型加载到原来的设备
    model.load_state_dict(torch.load(model_path, map_location=device))

# 将模型移动到指定设备
model = model.to(device)

optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

# 如果 val_loss 在 patience 个 epoch 内未下降 学习率更新：lr = lr * factor
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
# 注意：如果损失函数需要使用特定设备（如处理padding索引），也可以移动到设备上
criterion = criterion.to(device)
```

## 测试验证函数


```python
val_loss, val_blue_score = validate(model, test_dataloader, device, criterion)

print("val_loss: ", val_loss)
print("val_blue_score: ", val_blue_score)
```

## 开始训练


```python
run_train_loop(
    model,
    train_dataloader,
    test_dataloader,
    device,
    optimizer,
    scheduler,
    criterion,
    clip=clip,
)
```

## 测试翻译


```python
batch = next(iter(test_dataloader))
src = batch['en_input_ids']

print(src.shape)

out = model.generate(src, 50, temperature=0, sos_token_id=sos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id)

print(out.shape)

for en, de, ids in zip(batch['en'], batch['de'], out.cpu().tolist()):
    predict = decode(ids, lang='de')
    print(f'en: {en}\nde: {de}\npredict: {predict}\n')
```


```python

```
