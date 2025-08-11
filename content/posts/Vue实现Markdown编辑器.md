+++ 
date = '2025-08-11' 
draft = false 
title = 'Vue实现Markdown编辑器' 
categories = ['Vue组件'] 
+++

## 功能

1. 支持预览、编辑

2. 根据参数是否显示编辑/预览


## 安装依赖
> pnpm install @tailwindcss/typography 样式
> pnpm install marked  将markdown渲染为html

## MarkdownEditor.vue
```Vue
<template>
    <div class="max-w-6xl mx-auto p-6 bg-white rounded-xl shadow-lg">
      <!-- 药丸样式模式切换按钮 -->
      <!-- 仅在启用编辑时显示 -->
      <div v-if="editable" class="flex justify-center mb-4">
        <div class="inline-flex rounded-full bg-gray-200 p-1">
          <button
            @click="setMode('preview')"
            class="px-4 py-2 text-sm font-medium rounded-full transition-all duration-200 ease-in-out"
            :class="currentMode === 'preview' 
              ? 'bg-white text-gray-900 shadow-sm' 
              : 'bg-gray-200 hover:text-gray-900'"
            type="button"
          >
            预览
          </button>
          <button
            @click="setMode('edit')"
            class="px-4 py-2 text-sm font-medium rounded-full transition-all duration-200 ease-in-out"
            :class="currentMode === 'edit' 
              ? 'bg-white text-gray-900 shadow-sm' 
              : 'bg-gray-200 hover:text-gray-900'"
            type="button"
          >
            编辑
          </button>
        </div>
      </div>
  
      <!-- 编辑模式：双栏布局 -->
      <!-- 仅在启用编辑且当前为编辑模式时显示 -->
      <div v-if="editable && currentMode === 'edit'" class="flex flex-col md:flex-row gap-6">
        <!-- Markdown 输入框 (左侧) -->
        <div class="w-full md:w-1/2">
          <textarea
            v-model="localMarkdown"
            @input="onMarkdownInput"
            class="w-full h-96 p-4 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:outline-none"
            placeholder="输入 Markdown 内容..."
            :disabled="!editable"
          ></textarea>
        </div>
  
        <!-- 实时预览 (右侧) -->
        <div class="w-full md:w-1/2 p-4 rounded-lg bg-gray-50 overflow-y-auto">
          <div class="prose max-w-none" v-html="previewHtml"></div>
        </div>
      </div>
  
      <!-- 预览模式 或 不可编辑模式：单栏布局 -->
      <div v-else class="p-6 rounded-lg bg-gray-50 min-h-[300px]">
        <div class="prose max-w-none" v-html="previewHtml"></div>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref, computed, watch } from 'vue'
  import { marked } from 'marked'
  
  // 定义 Props, 组件对外的参数
  const props = defineProps({
    markdown: {
      type: String,
      required: true // 父组件必须传递此属性
    },
    // 新增 prop 控制是否可编辑
    editable: {
      type: Boolean,
      default: true // 默认值为 true，即默认启用编辑功能
    }
  })
  
  // 定义 Emits
  const emit = defineEmits(['update'])
  
  // 状态 - 使用 'preview' 或 'edit' 模式
  // 如果不可编辑，则强制为 'preview' 模式
  const currentMode = ref(props.editable ? 'preview' : 'preview')
  
  // 监听 editable prop 的变化，动态调整模式
  watch(() => props.editable, (newVal) => {
    if (!newVal) {
      // 如果设置为不可编辑，强制切换到预览模式
      currentMode.value = 'preview';
    }
    // 如果设置为可编辑，保持当前模式不变或可以重置为 'preview'
    // 这里选择保持 currentMode 不变，让父组件决定是否切换
  });
  
  // 使用一个本地 ref 来管理编辑器中的内容
  const localMarkdown = ref(props.markdown)
  
  // 监听 props.markdown 的变化，当父组件更新它时，同步到本地
  watch(() => props.markdown, (newVal) => {
    // 避免在用户输入时被父组件的更新覆盖
    if (currentMode.value !== 'edit' || localMarkdown.value !== newVal) {
        localMarkdown.value = newVal;
    }
  });
  
  // 设置模式
  const setMode = (mode) => {
    // 只有在启用编辑时才允许切换模式
    if (props.editable) {
      currentMode.value = mode
    }
  }
  
  // 实时渲染 Markdown
  const previewHtml = computed(() => {
    return marked(localMarkdown.value)
  })
  
  // 当文本域内容变化时触发
  const onMarkdownInput = () => {
      // 向父组件发出 'update' 事件，并传递新的 Markdown 内容
      emit('update', localMarkdown.value)
  }
  </script>
  
  <style scoped>
  /* Tailwind 的 prose 类通常来自 @tailwindcss/typography 插件 */
  </style>
```

## index.vue
```Vue
<template>
    <h1>Markdown 编辑器演示, 支持编辑</h1>
    <!-- 传递 markdown prop -->
    <MarkdownEditor 
      :markdown="myMarkdownContent" 
      @update="handleMarkdownUpdate" 
    />
    <div class="mt-4 p-4 bg-gray-100 rounded">
      <h2>父组件感知到的内容：</h2>
      <pre>{{ myMarkdownContent }}</pre>
    </div>
    <h1>Markdown 编辑器演示，不支持编辑</h1>
    <!-- 传递 markdown prop -->
    <MarkdownEditor 
      :markdown="myMarkdownContent" 
      :editable="false"
      @update="handleMarkdownUpdate" 
    />
</template>

<script setup>
import { ref } from 'vue';
import MarkdownEditor from './MarkdownEditor.vue'; 

const myMarkdownContent = ref(`# Hello, Vue!

这是一个 *Markdown* 编辑器。

- 你可以编辑我
- 看到实时预览
`);

// 处理子组件发出的 update 事件
const handleMarkdownUpdate = (newContent) => {
  console.log("父组件收到更新:", newContent);
  // 更新父组件中的数据
  myMarkdownContent.value = newContent; 
  // 你可以在这里执行其他逻辑，比如保存到服务器
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  padding: 20px;
}
</style>
```
