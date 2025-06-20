+++ 
date = '2025-06-12' 
draft = false 
title = 'LayerNorm和BatchNorm原理' 
categories = ['基础原理'] 
tags = ['归一化'] 
+++


https://excalidraw.com/#json=afOrWzKTn9l7vR77GOh6v,91anrIaGwxNwo7h09_3FgQ

## LayerNorm

**公式总结**，对于每个 token 的向量 $ x\_i \in \mathbb{R}^{D} $进行归一化：

$$
\mu\_i = \frac{1}{D} \sum\_{d=1}^{D} x\_{i,d}
$$

$$
\sigma\_i^2 = \frac{1}{D} \sum\_{d=1}^{D} (x\_{i,d} - \mu_i)^2
$$

$$
\hat{x}\_{i,d} = \frac{x\_{i,d} - \mu_i}{\sqrt{\sigma\_i^2 + \epsilon}}
$$

$$
y\_{i,d} = \gamma_d \cdot \hat{x}\_{i,d} + \beta\_d
$$

**特点：**
- 对 batch size 不敏感；
- 更适合处理变长序列（如 NLP 中的句子）；
- 在 Transformer 等结构中广泛使用；
- 可以用于 RNN、Transformer 等对 batch 内长度不一致的任务；
- 在小 batch 或动态 batch 场景下更稳定。

**使用场景：**
- 自然语言处理（NLP），如 Transformer；
- 序列建模任务（如机器翻译、文本生成）；
- 小 batch size 或动态 batch size 的情况；
- 大模型（LLM）中的常见选择。

### 手动实现


```python
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        # nn.Parameter 类型，会在反向传播中被自动计算梯度，并在优化器中更新
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

```

### 手动计算验证结果

在 **Transformer 模型**中，每个 token 的表示通常是一个固定维度的向量（比如 `512` 或 `768` 维），称之为 **token embedding** 或 **hidden state**。假设有一个 batch 中有 `batch_size` 个样本，每个样本有 `seq_len` 个 token，每个 token 表示为一个 `d_model` 维向量（如 `d_model=512`）。

那么输入的 shape 通常是：

```
[batch_size，seq_len，d_model]
```


LayerNorm 是对 **每个 token 向量内部的所有维度** 进行归一化。



```python
import torch
import torch.nn as nn

# 设置随机种子以保证可复现性
torch.manual_seed(42)

# 假设 batch_size=2, seq_len=3, hidden_dim=512
x = torch.randn(2, 3, 512)

# 定义 LayerNorm，对最后的 512 维进行归一化
# PyTorch 官方的 nn.LayerNorm 默认包含可学习的 weight（即 gamma）和 bias（即 beta），并且默认是开启的
layer_norm = nn.LayerNorm(512)
# layer_norm = nn.LayerNorm(512, elementwise_affine=False) # 禁用参数学习

custom_layer_norm = LayerNorm(512)

# 应用 LayerNorm
output = layer_norm(x)
custom_output = custom_layer_norm(x)

# 比较差异
diff = (output - custom_output).abs().max()
print(f"最大差异: {diff.item()}")
```

    最大差异: 2.09808349609375e-05
    


```python
# 手动计算

token = x[0, 0, :]  # shape: [512]

# Step 1: 计算均值
mu = token.mean()

# Step 2: 计算标准差（注意不是方差）
std = token.std(unbiased=False)  # 使用有偏估计（PyTorch 默认）

# Step 3: 归一化
epsilon = 1e-5
normalized_token = (token - mu) / torch.sqrt(std.pow(2) + epsilon)

# Step 4: 应用 gamma 和 beta（默认为1和0）
gamma = layer_norm.weight
beta = layer_norm.bias

manual_output = gamma * normalized_token + beta
```


```python
# 取出 PyTorch 的输出
pytorch_output = output[0, 0, :]

# 比较差异
diff = (pytorch_output - manual_output).abs().max()
print(f"最大差异: {diff.item()}")
```

    最大差异: 2.384185791015625e-07
    

## BatchNorm


**原理：**
- **对每个特征通道（channel）在 batch 维度上进行归一化**。
- 即：对于一个 batch 中的所有样本，在同一个通道上计算均值和方差，然后进行标准化。
- 公式如下：

$$
\hat{x}\_i = \frac{x\_i - \mu\_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y\_i = \gamma \hat{x}\_i + \beta
$$

其中：
- $\mu\_B$ 和 $\sigma\_B^2$ 是当前 batch 的均值和方差；
- $\gamma, \beta$ 是可学习参数，用于缩放和平移。

**特点：**
- 对 batch size 敏感：batch 越大效果越好；
- 在卷积神经网络（CNN）中表现很好；
- 不适用于 RNN 或变长序列任务，因为 batch 内长度不一致会导致统计量不稳定；
- 训练时使用 batch 统计信息，推理时使用移动平均。

** 使用场景：**
- CNN 图像分类、目标检测等；
- batch size 较大的任务；
- 固定长度输入任务。

---



BatchNorm 一般用于图像数据

- 输入形状：`[B, C, H, W]`（例如 `[4, 64, 32, 32]`）
- 在卷积层后使用 `BatchNorm2d`
- 每个 channel 上计算均值和方差（不是每个样本或每个像素）

---


| 项目 | 结论 |
|------|------|
| `gamma` 是否会被学习？ | ✅ 是的 |
| `gamma` 是什么类型？ | `nn.Parameter` |
| 是否需要手动添加到优化器？ | ❌ 不需要 |
| 输出是否与 PyTorch 官方一致？ | ✅ 是的（误差极小） |
| BatchNorm2d 的统计维度是？ | `[B, H, W]`（即每个 channel 单独计算） |

---

补充说明

- **BatchNorm2d 的核心思想**是在训练时，对每个 channel 单独计算其在当前 batch 中的均值和方差；
- 这样做的目的是让每层的输入分布更加稳定，加速训练；
- 因此，它非常适用于 CNN，尤其是图像分类、检测等任务。


### 手动实现


```python
import torch
from torch import nn


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum

        # 可学习参数：每个通道一个 gamma 和 beta
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        """
        x: shape [B, C, H, W]
        """
        
        # 计算 batch 内每个 channel 的均值和方差
        # shape [C, ...]
        batch_mean = x.mean(dim=[0, 2, 3])  # 保留 channel 维度
        batch_var = x.var(dim=[0, 2, 3], unbiased=False)

        # 归一化操作
        # 扩展维度以对齐 [C] -> [1, C, 1, 1]
        mean = batch_mean.view(1, -1, 1, 1)
        var = batch_var.view(1, -1, 1, 1)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = gamma * out + beta

        return out
```

### 手动计算验证结果


```python
# 设置随机种子以保证可复现性
torch.manual_seed(42)

# 图像输入：batch_size=4, channels=64, height=32, width=32
x = torch.randn(4, 64, 32, 32)

# 初始化两个 BatchNorm 层
custom_bn = BatchNorm2d(64)
torch_bn = nn.BatchNorm2d(64)

# 复制参数
torch_bn.weight.data = custom_bn.gamma.data.clone().detach()
torch_bn.bias.data = custom_bn.beta.data.clone().detach()

# 前向传播
out_custom = custom_bn(x)
out_torch = torch_bn(x)

# 对比输出差异
print("\n输出最大差异:", (out_custom - out_torch).abs().max().item())
```

    
    输出最大差异: 4.76837158203125e-07
    


```python
x.mean(dim=[0, 2, 3])  # 每个channel的均值
```




    tensor([ 0.0101, -0.0135, -0.0092, -0.0165,  0.0126,  0.0355, -0.0044,  0.0116,
            -0.0034,  0.0064, -0.0173,  0.0234, -0.0011,  0.0131,  0.0207, -0.0021,
             0.0031, -0.0033,  0.0044,  0.0311,  0.0004,  0.0185,  0.0096, -0.0113,
             0.0011, -0.0334,  0.0084,  0.0365,  0.0072, -0.0147,  0.0204, -0.0026,
            -0.0012,  0.0039,  0.0103, -0.0149, -0.0239, -0.0076,  0.0025, -0.0170,
            -0.0031, -0.0155,  0.0092, -0.0040, -0.0129,  0.0217, -0.0156, -0.0007,
             0.0243, -0.0148, -0.0013,  0.0190, -0.0102,  0.0025, -0.0008,  0.0083,
            -0.0131, -0.0284,  0.0029, -0.0304, -0.0088, -0.0117, -0.0247, -0.0007])




```python
x[:, 0, :, :].mean()  # 第0个channel上的均值
```




    tensor(0.0101)




```python
x.var(dim=[0, 2, 3])
```




    tensor([1.0124, 0.9601, 1.0393, 1.0086, 1.0048, 0.9918, 0.9817, 1.0268, 1.0148,
            1.0251, 1.0034, 1.0002, 1.0521, 1.0241, 1.0249, 0.9889, 1.0453, 1.0377,
            0.9842, 0.9838, 1.0301, 1.0398, 0.9660, 1.0187, 1.0066, 0.9683, 1.0045,
            0.9907, 0.9873, 0.9727, 1.0149, 1.0134, 1.0160, 1.0271, 1.0339, 1.0004,
            1.0058, 1.0211, 1.0344, 0.9780, 0.9891, 1.0094, 1.0268, 0.9980, 1.0162,
            1.0130, 0.9945, 1.0454, 0.9770, 1.0016, 0.9693, 0.9896, 0.9611, 1.0004,
            0.9488, 1.0014, 0.9869, 1.0021, 1.0160, 1.0008, 1.0002, 0.9867, 1.0156,
            0.9976])




```python
x[:, 0, :, :].var()
```




    tensor(1.0124)

