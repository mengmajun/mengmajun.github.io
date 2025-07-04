+++ 
date = '2025-07-04' 
draft = false 
title = 'DPO偏好微调' 
categories = ['微调'] 
tags = ['DPO'] 
+++


直接偏好优化：语言模型就是一个奖励模型

[Direct Preference Optimization:Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)

## DPO论文摘要

> While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.

虽然大规模无监督语言模型（LMs）能够学习广泛的世界知识和一定推理能力，但由于其完全无监督的训练特性，要实现对其行为的精确控制十分困难。现有方法通过收集人类对模型生成结果相对质量的标注数据，并基于这些偏好对无监督语言模型进行微调来增强可控性，通常采用人类反馈强化学习（RLHF）技术。然而RLHF流程复杂且往往不稳定：需要先训练反映人类偏好的奖励模型，再通过强化学习微调大型无监督语言模型以最大化预估奖励，同时确保模型参数不会过度偏离原始模型。

本文提出RLHF奖励模型的新参数化方法，可直接解析推导出对应最优策略，从而仅需简单分类损失函数即可解决标准RLHF问题。我们称该算法为直接偏好优化（DPO），其具有稳定性强、性能优异、计算高效的特点，无需在微调期间从语言模型采样或进行大量超参数调优。实验表明，DPO在使语言模型对齐人类偏好方面达到或超越现有方法。值得注意的是，在生成内容的情感控制任务上，DPO微调效果优于基于PPO的RLHF；在文本摘要和单轮对话任务中，DPO在保持响应质量相当或更优的同时，实现和训练过程显著简化。

![DPO vs PLHF](https://github.com/huggingface/trl/assets/49240599/9150fac6-3d88-4ca2-8ec6-2a6f3473216d)


---

## 偏好数据集格式

以下是hugging face trl库的偏好数据格式

```python

# 标准格式
## 推荐的显示说明偏好的格式
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Implicit prompt
preference_example = {"chosen": "The sky is blue.", "rejected": "The sky is green."}

# 对话格式
## 推荐的显示说明偏好的格式
preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                      "chosen": [{"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "assistant", "content": "It is green."}]}
## Implicit prompt
preference_example = {"chosen": [{"role": "user", "content": "What color is the sky?"},
                                 {"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "user", "content": "What color is the sky?"},
                                   {"role": "assistant", "content": "It is green."}]
```

---


## DPO损失函数的直觉理解

$$
\mathcal{L}\_{\text{DPO}}(\pi\_\theta; \pi\_{\text{ref}}) = -\mathbb{E}\_{(x, y\_w, y\_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
\quad 
$$

其中：
- $ \pi\_\theta; \pi\_{\text{ref}} $ 分别是要优化的策略模型、参考模型，参考模型不需要优化，一般是策略模型的初始权重
- $ (x, y\_w, y\_l) $ 表示一个数据pair，分别是输入的prompt，偏好的回答，拒绝的回答
- $ \pi_\theta(y_w|x) $ 表示策略模型给定x生成y的概率，0到1之间
- $  \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} $  表示的是策略模型对偏好回答的对数概率和参考模型的比值，视为一个奖励，如果我们的策略模型生成的偏好回答概率越大，比值越大，相当于奖励越大，如果 $ \pi_\theta(y_w|x) $ 大于 $ \pi_{\text{ref}}(y_w|x) $ 那么比值就大于1，对数值大于0
- $ \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} $ 表示的是策略模型对拒绝回答的对数概率和参考模型比值，视为一个奖励，如果我们的策略模型生成的拒绝回答对数概率越小，比值越小，相当于概率越小，这也是我们希望的，模型对拒绝回答的生成的概率要小.如果 $ \pi_\theta(y_l|x) $ 小于 $\pi_{\text{ref}}(y_l|x) $ 那么比值小于1，对数值小于0
- $ \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$ 这个公式就表示了如果我们模型对偏好回答的奖励越高，对拒绝回答的奖励越低，那么差值越大，通过$ \sigma $ 函数后接近1，整个对数接近0，实现最小化损失优化

---

## DPO伪代码

```python
# 参考模型使用策略模型的初始版本
ref_model = theta_model
for batch in preference_data:
    # 获取批次数据
    prompt = batch["prompt"]          # [batch_size, prompt_len]
    win_response = batch["chosen"]    # [batch_size, response_len]
    loss_response = batch["rejected"] # [batch_size, response_len]

    # 计算参考模型对数概率（无梯度）
    with torch.no_grad():
        # 这里输出的是每个位置next-token的对数概率，因为我们是在训练阶段，直接根据target将对应的概率挑选出来的
        ref_win_logps = ref_model(prompt, win_response).log_prob  # [batch_size, response_len]
        ref_loss_logps = ref_model(prompt, loss_response).log_prob  # [batch_size, response_len]
    
    # 计算策略模型对数概率
    theta_win_logps = theta_model(prompt, win_response).log_prob  # [batch_size,response_len]
    theta_loss_logps = theta_model(prompt, loss_response).log_prob  # [batch_size,response_len]
    
    # 对序列维度求和
    ref_win_logps = ref_win_logps.sum(dim=-1)  # [batch_size]
    ref_loss_logps = ref_loss_logps.sum(dim=-1)  # [batch_size]
    theta_win_logps = theta_win_logps.sum(dim=-1)  # [batch_size]
    theta_loss_logps = theta_loss_logps.sum(dim=-1)  # [batch_size]
    
    # 计算对数概率比（奖励），公式中是softmax后的概率比完了后取对数，等价为先取对数再相减
    log_ratio_win = theta_win_logps - ref_win_logps  # [batch_size]
    log_ratio_loss = theta_loss_logps - ref_loss_logps  # [batch_size]
    
    # 计算DPO损失
    optimizer.zero_grad()
    loss = -F.logsigmoid(beta * (log_ratio_win - log_ratio_loss)).mean()
    
    # 反向传播与优化
    loss.backward()
    optimizer.step()

```

---

## 训练示例代码

```python
# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO")
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()

```

奖励的差距变大的趋势表明模型随着时间正在提高和生成更好的响应，即chosen的答案

![DPO训练奖励差距图](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/dpo-qwen2-reward-margin.png)

---


## DPO公式推导



### 一、背景：RLHF

Reinforcement Learning from Human Feedback 人类反馈强化学习的关键是需要人对模型的回复进行打分

DPO 是对传统 RLHF 流程的一种改进方法。 RLHF 三个主要阶段：

**1. 监督微调（SFT）**
- 使用高质量的人类标注数据对预训练模型进行微调。
- 得到一个初步的策略模型 $\pi^{SFT}$，这个模型已经具备一定的任务能力。

**2. 奖励建模**
- 用 SFT 模型生成多个 response。
- 给人类标注者看，让他们选择更喜欢哪个 response。
- 根据偏好数据训练一个奖励函数 $r_\phi(x, y)$：
  $$
  \mathcal{L}\_R(r_\phi, \mathcal{D}) = -\mathbb{E}\_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
  $$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 sigmoid 函数。
奖励模型的输入是prompt x 和 response y，输出是一个标量值的奖励
注意：这个奖励函数一般是和原始的策略模型一样，只是在最后接了一个线性层

这其实是一个二分类交叉熵损失，目标是让模型学会区分“好”response 和“坏”response。：
- 如果 $y_w$ 被偏好，则我们希望 $r\_\phi(x, y_w) > r\_\phi(x, y_l)$
- 所以 $\sigma(r\_\phi(x, y_w) - r\_\phi(x, y_l)) \approx 1$
- 否则接近 0，损失就大


**3. 强化学习优化**
- 用上面学到的奖励函数来指导语言模型更新。
- 最大化以下目标：
  $$
  \max_{\pi_\theta} \mathbb{E}_{x,y}[r_\phi(x, y)] - \beta \mathbb{D}_{\text{KL}}[\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)]
  $$
  这个 KL 正则项防止模型偏离初始策略太远，避免崩溃或不稳定。

> **总结：RLHF 分三步走，但流程复杂、训练困难、效果依赖奖励模型的质量。**

---

### 二、DPO 的动机

作者指出：**强化学习在语言模型中应用困难重重**，比如 PPO 等算法训练不稳、需要大量调参。于是提出了一种新方法：**DPO**，它的核心思想是：

> ❝ 不再显式地学习奖励函数，而是通过一种巧妙的变换，直接从偏好数据中优化语言模型。❞

换句话说，**DPO 把原本 RLHF 中的奖励建模 + 强化学习两步，合并为一步，直接从偏好数据优化语言模型。**

---

### 三、DPO 的数学推导详解

我们一步步来看：

**1. RLHF 中最优策略的形式**

根据 RLHF 的目标，推导出其**最优策略**形式：


假设：
- 固定 prompt $x$，考虑所有可能的 response $y$
- 可以将该问题视为一个带约束的最大化问题：

$$
\max\_{\pi(y|x)} \sum\_y \pi(y|x) r(x, y) - \beta \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}
$$

这是一个典型的 **最大熵强化学习问题**，可以用拉格朗日乘子法求解。

然后，构造 Lagrangian：

$$
\mathcal{L}(\pi, \lambda) = \sum_y \pi(y|x) r(x, y) - \beta \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi\_{\text{ref}}(y|x)} - \lambda \left( \sum_y \pi(y|x) - 1 \right)
$$

对 $\pi(y|x)$ 求偏导并令其为零：

$$
\frac{\partial \mathcal{L}}{\partial \pi(y|x)} = r(x, y) - \beta \left( \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + 1 \right) - \lambda = 0
$$

整理得：

$$
\log \frac{\pi(y|x)}{\pi\_{\text{ref}}(y|x)} = \frac{1}{\beta} (r(x, y) - \lambda - 1)
$$

两边取指数：

$$
\pi(y|x) = \pi\_{\text{ref}}(y|x) \cdot \exp\left( \frac{1}{\beta} (r(x, y) - \lambda - 1) \right)
$$

归一化后得到：

$$
\pi\_r(y|x) = \frac{1}{Z(x)} \pi\_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)
\quad (式 4)
$$

其中 $Z(x) = \sum\_y \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)$ 是配分函数。

- $\pi\_{\text{ref}}$ 是参考模型（通常是 SFT 模型）。
- $r(x, y)$ 是奖励函数。
- $Z(x)$ 是归一化常数（配分函数），难以计算。

这个公式说明：**语言模型可以表示成参考模型与奖励函数的指数加权组合**。

---

**2. 反向表达奖励函数**

接下来，作者做了一个非常重要的操作：**把奖励函数用策略表示出来**。

对上式两边取对数得：

$$
\log \pi\_r(y|x) = \log \pi\_{\text{ref}}(y|x) + \frac{1}{\beta} r(x, y) - \log Z(x)
$$

移项得：

$$
r(x, y) = \beta \log \frac{\pi\_r(y|x)}{\pi\_{\text{ref}}(y|x)} + \beta \log Z(x)
\quad (式 5)
$$

注意：$\log Z(x)$ 对所有 $y$ 都一样，所以当我们在比较两个 response $y_1, y_2$ 的奖励差异时，这一项会被抵消掉。

---

**3. 应用于 Bradley-Terry 模型**

我们回到 Bradley-Terry 模型：

$$
p^*(y_1 \succ y_2 | x) = \sigma(r^*(x, y_1) - r^*(x, y_2))
$$

代入上面反向得到的奖励函数表达式：

$$
r^*(x, y_1) - r^*(x, y_2) = \beta \log \frac{\pi^*(y_1|x)}{\pi\_{\text{ref}}(y\_1|x)} - \beta \log \frac{\pi^*(y\_2|x)}{\pi\_{\text{ref}}(y\_2|x)}
$$

所以偏好概率变为：

$$
p^*(y\_1 \succ y\_2 | x) = \sigma\left( \beta \log \frac{\pi^*(y_1|x)}{\pi\_{\text{ref}}(y_1|x)} - \beta \log \frac{\pi^*(y\_2|x)}{\pi_{\text{ref}}(y_2|x)} \right)
$$

现在我们把 $\pi^*$ 替换成我们的参数化策略 $\pi_\theta$，就得到了 DPO 的目标函数：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
\quad (式 7)
$$

这就是 DPO 的核心目标函数，也就是说，我们成功地将偏好概率完全用策略 $\pi^*$ 来表达了，而不再需要奖励函数！

---

### 四、最终目标函数：DPO Loss

现在，我们就可以定义我们的目标函数了。我们要最大化偏好数据下模型预测的概率：

$$
\mathcal{L}\_{\text{DPO}}(\pi\_\theta; \pi\_{\text{ref}}) = -\mathbb{E}\_{(x, y\_w, y\_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

这就是 DPO 的目标函数。

---

### 五、梯度分析：DPO 到底做了什么？

我们来看看这个损失函数的梯度：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E}_{(x, y_w, y_l)} \left[ \sigma(\hat{r}(x, y_l) - \hat{r}(x, y_w)) \cdot \left( \nabla_\theta \log \pi(y_w|x) - \nabla_\theta \log \pi(y_l|x) \right) \right]
$$

其中 $\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$

解释：
- 如果 $\hat{r}(x, y_l) > \hat{r}(x, y_w)$，说明当前模型认为非偏好 response 比偏好 response 更好 → **这是一个错误**
- $\sigma(\hat{r}(x, y_l) - \hat{r}(x, y_w))$ 就是对这种错误程度的衡量 → 错误越大，权重越高
- 所以梯度会：
  - 提高 $y_w$ 的 log 概率（鼓励生成偏好 response）
  - 降低 $y_l$ 的 log 概率（惩罚生成非偏好 response）

> **一句话总结：DPO 的梯度方向就是在提升偏好 response、降低非偏好 response 的概率。**

---

### 六、DPO 的优势总结

| 优点 | 描述 |
|------|------|
| 不需要奖励模型 | 直接从偏好数据优化语言模型 |
| 不需要强化学习 | 避免复杂的 PPO 或其他 RL 算法 |
| 损失简单 | 就是一个交叉熵损失，容易实现 |
| 稳定性高 | 实验表明比 RLHF 更稳定 |
| 性能优越 | 在多个任务上表现媲美甚至超过 RLHF |

---

### 七、DPO 的实际使用流程

1. 准备偏好数据集 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}$；
2. 设置参考模型 $\pi_{\text{ref}}$（通常为 SFT 模型）；
3. 定义目标函数 $\mathcal{L}_{\text{DPO}}$；
4. 用标准优化器（如 AdamW）最小化该损失；
5. 得到微调后的语言模型 $\pi_\theta$。

---
