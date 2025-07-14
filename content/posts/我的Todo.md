+++ 
date = '2025-05-09' 
draft = false 
title = '我的Todo' 
categories = ['Todo'] 
tags = ['Todo'] 
+++

## 2025-07
- agentic rag，rag可以作为一个重要的研究点
- GRPO在自然语言转SQL的研究，可以作为自己的一个方向
- 多Agent系统，RL Agent，系统构建，可以梳理出来一个实现综述
- 手动实现GRPO
- 每周学一节强化学习，hugging face的课程用游戏来学习
- 体现我的学习探索能力，基于GRPO的SQL代码生成和结构化图表生成，我想到要生成结构化图片，可不可以先把基本的结构化图表进行原子化，然后让AI来组装呢，就像exceldraw一样

## 2025-06

- 大模型经典论文和代码复现
  - 语言模型：attention》GPT-1》BERT-》GTP2-》GPT3，多模态：VIT》Clip》Siglip，Hugging face：nanoLLM》smolVLM-》smolVLA
  - 代码复现：attention、GPT2、Clip、smolVLM
  - 卡帕西的nanoGPT课程

- 演讲整理
  - Ilya Sutskever的Next Token Prediction 博客
  - Hyung Won Chung的AI研究的主要推动力 [link](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650922111&idx=2&sn=69ae67a08b5ffd782d7bd25d94e6ed7a&chksm=84e41801b393911720f779edaa704cc703a4871b96dd75eff9b599470090e8eb5d50034cebe0&scene=21#wechat_redirect)、[link](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650893355&idx=1&sn=5911ccc05abf5177bb71a47ea5a748c8&chksm=84e4a855b39321434a2a386c9f359979da99dd441e6cf8f88062f1e909ad8f5b9b24198d1edb&scene=21#wechat_redirect)

## Idea

- 现在的RAG技术已经过度到AI自主推理调用知识库，比如阿里的RAG团队的WebDancer、还有一些自主查询RAG论文这些需要重点关注
- 还有就是现在的多Agent系统开发，比如Anthrproic的多agent设计，实质是prompt设计+推理大模型能力，这部分也需要看，最后是复现，这部分可以看看字节的deepflow设计
- AI技术的内容生产流水线，自动博客写作，论文解读等，微博有一个分享
- 项目：RAG论文复现RAFT、自然语言转SQL模型训练基于GRPO、深度研究端到端WebDancer
- 深度学习基础知识
  - CS231n课程、线性代数（矩阵运算、SVD分解、秩）、概率论（条件概率、期望）、高数（微分求导、梯度）
  - 激活函数（SIGMOD，ReLU，tanh， GELU， Swish，GLU， SwishGLU）
  - 炼丹技巧，比如学习率warmup和退火，early-stop，数据增强/清洗
  - 常见NLP任务的损失函数、metric是什么，比如BLEU、ROUGE-L、perplexity
  - 反向传播推导 （面试常考）
  - 梯度爆炸/消失问题及其解决方法
  - BatchNorm / LayerNorm 的区别及数学推导
  - 交叉熵损失函数推导 （分类任务）
  - Softmax交叉熵损失函数怎么计算？
  - BatchNorm 和 LayerNorm 的区别？
Transformer 中的 QKV 矩阵运算推导
- LLM模型架构
  - encode2decoder（T5）、decoder-only（GPT）、encoder-only（BERT）、prefix-decoder(chatGLM)
  - 不同的attention的改进MHA GQA MQA MLA，attention加速FlashAttention、PageAttention了解原理实现, 不同attention机制在推理速度上的差异
  - KV-cache原理和实现
  - MOE原理, MoE 架构中 expert 路由策略 （Top-k routing, load balancing loss）
  - 不同位置编码，正余弦位置编码、可学习位置编码、RoPE位置编码
- LLM训练/微调
  - 预训练：SFT、RLHF、DPO、PPO、GRPO
  - SFT、DPO、PPO、KTO 等算法流程图
  - 微调：LoRA原理和实现、prefix-tuning、p-tuning、adapter
  - LoRA 的矩阵低秩分解原理与参数冻结方式
  - QLoRA 原理简介 （量化 + LoRA 结合）
  - 手动实现LoRA、GRPO
- LLM decoding
  - transformers库的generate函数的所有参数的作用（了解了就知道推理的各种方法了，什么温度，重复性惩罚，top-k, top-p, beam_search, group_beam_search, 避免n-gram重复
  - LLM处理长文本的原理
  - speculative decoding 《A Thorough Examination of Decoding Methods in the Era of LLMs》
  - Beam Search vs Sampling vs Nucleus Sampling 的差异
  - 生成长度控制与惩罚项（如 repetition penalty）
  - 温度调节对分布的影响
- RAG
  - RAFT 特定领域微调
- Aent
  - Agent 架构综述 （如 ReAct, Reflexion, Think-Action-Observe loop）
  - 深度研究端到端WebDancer
- 强化学习
  - 强化学习课程 西湖大学赵世钰老师的《强化学习的数学原理》

## 待读论文清单
instructGPT、lora

