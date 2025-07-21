+++ 
date = '2025-07-21' 
draft = false 
title = 'PresentAgent 基于多Agent生成PPT演示视频' 
categories = ['视频生成', 'Agent开发'] 
tags = ['视频生成', 'Agent开发'] 
+++


[PresentAgent: Multimodal Agent for Presentation Video Generation](https://github.com/AIGeeksGroup/PresentAgent?tab=readme-ov-file)

## 论文工作

论文提取的任务是将长篇的文档转为带叙述的演示视频

过程中会将文档生成不同布局的图片，然后给每个图片写脚本，然后将脚本转换成语音，然后将各个图片语音拼接起来，生成视频

实现方式上多模态模型组成一个流水线，论文提供了代码实现

- 文档处理：将输入文档通过大纲规划分割为语义块。
- 幻灯片生成：为每个语义块生成基于布局的幻灯片视觉内容。
- 旁白与音频合成：将关键信息改写为口语化旁白，并通过文本转语音技术生成音频。
- 音视频同步：将幻灯片与音频合成，生成时间对齐的演示视频。

应用场景：

- 自动将论文转成视频，分享论文研究
- 进一步可以考虑合成由梗图组合成的搞笑短视频等，也适合讲述了视频，科普短视频，时政视频等


## 代码实现

[DeepWiki PresentAgent](https://deepwiki.com/AIGeeksGroup/PresentAgent)
