+++ 
date = '2025-08-14' 
draft = false 
title = 'Vue实现Makrdown论文预览组件' 
categories = ['Vue组件'] 
+++


## 安装依赖

> pnpm install marked highlight.js

## MarkdownPaper.vue

```
<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { marked } from 'marked'
import hljs from 'highlight.js'

// 接收 Markdown 内容
const props = defineProps({
  markdown: { type: String, required: true }
})

// 响应式数据
const contentRef = ref(null)
const toc = ref([]) // 目录 [{ id, text, level }]
const activeId = ref('')

// 生成唯一 ID 的 slug 工具
const slugify = (text) => {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

// 渲染 Markdown 并生成目录
const renderMarkdown = () => {
  const idCount = {}

  // 配置 marked
  marked.setOptions({
    highlight: (code, lang) => {
      const language = hljs.getLanguage(lang) ? lang : 'plaintext'
      return hljs.highlight(code, { language }).value
    },
    langPrefix: 'hljs language-',
    mangle: false,
    headerIds: false
  })

  // 自定义 renderer 处理标题
  const renderer = new marked.Renderer()
  const tocData = []

  renderer.heading = (text, level) => {
  let titleText
  if (typeof text === 'string') {
    titleText = text
  } else if (text.text) {
    titleText = text.text // Token 对象
  } else if (typeof text.tokens?.map === 'function') {
    // 更深层提取文本，比如链接、强调等复合 token
    titleText = text.tokens.map(t => t.text || '').join('')
  } else {
    titleText = String(text)
  }

  const slug = slugify(titleText)
  if (!idCount[slug]) idCount[slug] = 0
  const id = `${slug}-${idCount[slug]++}`

  tocData.push({ id, text: titleText, level })
  return `<h${level} id="${id}" class="scroll-mt-20 font-bold">${titleText}</h${level}>`
}

  toc.value = tocData
  contentRef.value.innerHTML = marked.parse(props.markdown, { renderer })

  // 处理初始 hash
  nextTick(() => {
    const hash = decodeURIComponent(location.hash.slice(1))
    if (hash) {
      const el = document.getElementById(hash)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth' })
        activeId.value = hash
      }
    }
  })
}

// 滚动到指定章节
const scrollToSection = (id) => {
  const el = document.getElementById(id)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth' })
    activeId.value = id
    history.pushState(null, null, '#' + id)
  }
}

// 监听滚动，更新激活的标题
const handleScroll = () => {
  const el = contentRef.value
  if (!el) return

  const scrollPos = el.scrollTop + window.innerHeight * 0.3

  let current = ''
  for (let i = toc.value.length - 1; i >= 0; i--) {
    const { id } = toc.value[i]
    const heading = document.getElementById(id)
    if (heading && heading.offsetTop <= scrollPos) {
      current = id
      break
    }
  }

  if (current !== activeId.value) {
    activeId.value = current
  }
}

// Vue 3 的 nextTick
const nextTick = (fn) => {
  Promise.resolve().then(fn)
}

// 生命周期
onMounted(() => {
  renderMarkdown()
  contentRef.value.addEventListener('scroll', handleScroll)
  handleScroll() // 初始化高亮
})

onBeforeUnmount(() => {
  if (contentRef.value) {
    contentRef.value.removeEventListener('scroll', handleScroll)
  }
})

// 监听 markdown 更新
watch(
  () => props.markdown,
  () => {
    renderMarkdown()
  }
)
</script>

<template>
  <div class="flex h-screen bg-gray-50 font-serif">
    <!-- 左侧目录 -->
    <div
      class="w-64 bg-white border-r border-gray-200 p-6 overflow-y-auto sticky top-0 h-full"
    >
      <h3 class="text-lg font-bold text-gray-800 mb-4">论文目录</h3>
      <ul class="space-y-1 text-sm">
        <li
          v-for="item in toc"
          :key="item.id"
          :class="[
            'px-2 py-1 rounded transition-colors duration-150',
            activeId === item.id
              ? 'bg-black text-white font-medium'
              : 'text-gray-700 hover:bg-gray-100'
          ]"
          :style="{ marginLeft: (item.level - 1) * 16 + 'px' }"
        >
          <a
            href="#"
            @click.prevent="scrollToSection(item.id)"
            class="block"
          >
            {{ item.text }}
          </a>
        </li>
      </ul>
    </div>

    <!-- 右侧论文内容 -->
    <div class="flex-1 overflow-y-auto bg-gray-50">
    <!-- 模拟论文纸张：居中、固定宽度、留白 -->
        <div
            class="w-full max-w-4xl mx-auto bg-white"
            style="padding: 2.5rem 3rem; min-height: 100vh;"
        >
            <!-- 内容渲染容器 -->
            <div ref="contentRef" class="font-serif leading-relaxed"></div>
        </div>
    </div>
  </div>
</template>

<style scoped>
/* 首行缩进：仅对段落生效 */
:deep(p) {
  text-indent: 2em;
  margin: 0.8em 0;
}

/* 取消图片、列表、代码块等的缩进 */
:deep(p img),
:deep(ul),
:deep(ol),
:deep(pre),
:deep(blockquote),
:deep(table),
:deep(figcaption) {
  text-indent: 0 !important;
}

/* 标题样式 */
:deep(h1) { @apply text-2xl mt-10 mb-6 font-bold text-gray-900; }
:deep(h2) { @apply text-xl mt-8 mb-5 font-semibold text-gray-800; }
:deep(h3) { @apply text-lg mt-6 mb-4 font-medium text-gray-700; }

/* 代码块 */
:deep(pre) {
  @apply bg-gray-900 text-green-300 p-4 rounded-md text-sm my-4 overflow-auto;
}

/* 引用块 */
:deep(blockquote) {
  @apply border-l-4 border-blue-500 bg-blue-50 italic py-2 px-4 my-4 text-gray-700;
}

/* 表格 */
:deep(table) {
  @apply w-full my-4 border border-gray-300;
}
:deep(th) {
  @apply bg-gray-100 border-b font-medium text-gray-800 p-2 text-center;
}
:deep(td) {
  @apply border-b border-gray-200 p-2 text-center text-gray-700;
}

/* 图片与图注 */
:deep(figure) {
  @apply my-6 text-center;
}
:deep(figure img) {
  @apply max-w-full h-auto border border-gray-300;
}
:deep(figure figcaption) {
  @apply text-sm text-gray-600 mt-2;
}
</style>

```

## index.vue

```
<template>
  <MarkdownPaper :markdown="paperContent" />
</template>

<script setup>
import MarkdownPaper from './MarkdownPaper.vue'

const paperContent = `
# 基于人脸识别技术的校园考勤管理系统设计与实现

## 摘要

随着人脸识别技术的成熟，传统考勤方式正逐步向智能化转型。然而，当前市场上的考勤系统普遍存在设备成本高、识别速度慢、部署复杂等问题，难以满足中小型教育机构对低成本、高效率考勤管理的需求。为解决上述问题，设计并实现了一种基于人脸识别技术的轻量级校园考勤管理系统。系统采用OpenCV与深度学习框架，结合人脸检测、特征提取与匹配算法，实现了人脸录入、实时识别打卡及考勤数据管理功能。实验结果表明，该系统在保证识别准确率的同时，具备较高的响应速度和良好的可扩展性。该系统的应用不仅有助于提升校园考勤管理的智能化水平，也为相关技术的实践教学提供了有效支撑。

## 正文

### 1. 引言

#### 1.1 研究背景

近年来，人脸识别技术取得了显著进展，尤其在深度学习的推动下，其在教育管理领域的应用日益广泛。例如，基于深度神经网络的图像识别方法，如 ResNet 结构，有效解决了深层网络的梯度消失问题[^1]，为人脸识别提供了更高的准确率和效率。在此背景下，许多教育机构开始尝试将人脸识别应用于考勤管理，以提升管理智能化水平。然而，现有系统普遍存在设备成本高、识别速度慢等问题，限制了其在中小型教育机构中的普及[^1]。因此，设计一种

#### 1.2 问题陈述

当前市场上的考勤系统普遍存在设备成本高、识别速度慢、部署复杂等问题。高昂的硬件和软件投入限制了中小型教育机构的应用普及，而复杂的系统架构也增加了维护难度[^1]。同时，部分系统在人脸识别效率和准确率之间难以取

#### 1.3 研究目标与意义

本研究旨在设计并实现一种基于人脸识别技术的轻量级校园考勤管理系统，以解决当前市场上考勤设备成本高、识别速度慢及部署复杂等问题。通过采用OpenCV与深度学习框架，结合高效的人脸检测与特征匹配算法，系统实现了快速、准确的考勤管理[^1]。该系统不仅提升了中小型教育机构的考勤效率，还降低了硬件投入与维护成本，具有良好的可扩展性与实际应用价值，为智能化校园管理提供了有力支持。


### 2. 相关工作

#### 2.1 国内外研究现状

近年来，人脸识别技术在考勤系统中得到了广泛应用。国外研究中，基于深度学习的方法如卷积神经网络（CNN）被广泛应用于人脸识别，显著提高了识别精度和速度。例如，ResNet 结构的提出解决了深层网络的梯度消失问题，为复杂场景下的人脸识别提供了技术支持[^1]。国内研究方面，众多高校和企业也纷纷推出基于人脸识别的考勤系统，主要集中在人脸检测、特征提取和匹配算法的优化上。这些系统多采用OpenCV与深度学习框架相结合的方式，实现了较高的识别准确率和较快的响应速度。然而，现有系统普遍存在设备成本高、部署复杂等问题，难以满足中小型教育机构对低成本、高效考勤管理的需求。


`
</script>

```
