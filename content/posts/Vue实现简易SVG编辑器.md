+++ 
date = '2025-08-14' 
draft = false 
title = 'Vue实现简易SVG编辑器' 
categories = ['Vue组件'] 
+++

## 安装依赖
> pnpm install @svgdotjs/svg.js


## SVGEditor.vue
```Vue

<!-- components/SVGEditor.vue -->
<template>
    <div class="flex flex-col w-full h-full">
      <!-- 工具栏 -->
      <div class="bg-white flex justify-end">
        <button
          @click="exportAsPNG"
          class="bg-blue-600 hover:bg-blue-700 text-white text-sm px-4 py-2 rounded flex items-center gap-2 transition"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V18a2 2 0 01-2 2z" />
          </svg>
        </button>
      </div>
  
      <!-- SVG 画布容器 -->
      <div
        ref="container"
        class="flex-1 flex justify-center items-center bg-white rounded-b-lg overflow-auto"
        @dragstart.prevent
      ></div>
    </div>
</template>
  
  <script setup>
  import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
  import { SVG } from '@svgdotjs/svg.js'
  
  // Props & Emits
  const props = defineProps({
    svgContent: {
      type: String,
      required: true
    }
  })
  
  const emit = defineEmits(['svg-updated'])
  
  // DOM 容器引用
  const container = ref(null)
  
  // 状态
  let draw = null
  let isDragging = false
  let selectedElement = null
  let offset = { x: 0, y: 0 }
  
  // 清理函数
  const cleanup = () => {
    if (draw) draw.off('mousedown')
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
    draw = null
    selectedElement = null
    isDragging = false
  }
  
  // 初始化编辑器
  const initEditor = () => {
    cleanup()
  
    const el = container.value
    if (!el) return
  
    el.innerHTML = props.svgContent
  
    const svgElement = el.querySelector('svg')
    if (!svgElement) return
  
    draw = SVG(svgElement)
  
    const draggableElements = draw.find('rect, circle, ellipse, polygon, path, text, line')
  
    draggableElements.forEach(el => {
      el.node.style.cursor = 'move'
      el.off('mousedown')
      el.on('mousedown', function (e) {
        e.preventDefault()
        e.stopPropagation()
  
        selectedElement = this
        isDragging = true
  
        const bbox = selectedElement.bbox()
        offset.x = e.clientX - bbox.x
        offset.y = e.clientY - bbox.y
  
        document.addEventListener('mousemove', handleMouseMove)
        document.addEventListener('mouseup', handleMouseUp)
      })
    })
  }
  
  // 鼠标移动
  const handleMouseMove = (e) => {
    if (!isDragging || !selectedElement) return
    const x = e.clientX - offset.x
    const y = e.clientY - offset.y
    selectedElement.move(x, y)
  }
  
  // 鼠标释放
  const handleMouseUp = () => {
    if (isDragging && selectedElement && draw) {
      const updatedSVG = draw.svg()
      emit('svg-updated', updatedSVG)
    }
  
    isDragging = false
    selectedElement = null
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
  }
  
  // 导出为 PNG
  const exportAsPNG = () => {
  if (!draw) {
    console.warn('SVG 尚未初始化')
    return
  }

  const svgElement = draw.node

  // 获取实际渲染尺寸（兼容响应式布局）
  const rect = svgElement.getBoundingClientRect()
  const width = rect.width
  const height = rect.height

  // 获取设备像素比，用于高清导出
  const scale = window.devicePixelRatio || 2

  // 创建 canvas 并设置尺寸
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')

  canvas.width = width * scale
  canvas.height = height * scale

  // 设置样式尺寸，避免拉伸
  canvas.style.width = `${width}px`
  canvas.style.height = `${height}px`

  // 缩放 canvas 上下文
  ctx.scale(scale, scale)

  // 序列化 SVG 并确保 xmlns 存在
  const serializer = new XMLSerializer()
  let svgStr = serializer.serializeToString(svgElement)

  if (!svgStr.includes('xmlns="http://www.w3.org/2000/svg"')) {
    svgStr = svgStr.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg" ')
  }

  // 创建 Blob 和 URL
  const blob = new Blob([svgStr], { type: 'image/svg+xml' })
  const url = URL.createObjectURL(blob)

  const img = new Image()
  img.onload = () => {
    // 白底（可选）
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, width, height)

    // 绘制 SVG 到 canvas
    ctx.drawImage(img, 0, 0, width, height)

    // 导出为 PNG
    canvas.toBlob(
      (blob) => {
        const pngUrl = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = pngUrl
        a.download = `diagram-${Date.now()}.png`
        a.click()

        // 清理
        URL.revokeObjectURL(pngUrl)
        URL.revokeObjectURL(url)
      },
      'image/png',
      1.0
    )
  }

  img.onerror = () => {
    console.error('SVG 加载失败')
    URL.revokeObjectURL(url)
  }

  img.src = url
}
  
  // 生命周期
  onMounted(() => {
    initEditor()
  })
  
  watch(
    () => props.svgContent,
    () => {
      nextTick(() => {
        initEditor()
      })
    }
  )
  
  onUnmounted(() => {
    cleanup()
  })
  </script>
  
  <style scoped>
  /* 确保 SVG 元素可交互 */
  [ref='container'] svg {
    max-width: 95%;
    max-height: 95%;
    width: auto;
    height: auto;
    display: block;
    margin: 0 auto;
  }
  
  [ref='container'] svg * {
    pointer-events: all;
  }
  </style>

  ```


## Index.vue测试

```Vue
<!-- ParentComponent.vue -->
<template>
  <div class="min-h-screen bg-gray-100 flex flex-col items-center p-6">
    <h1 class="text-2xl font-bold mb-4 text-gray-800">SVG 拖拽编辑器</h1>

    <!-- 显示当前 SVG 内容（可选调试用） -->
    <div class="mb-4 text-sm text-gray-600">
      <button @click="showCode = !showCode" class="underline">
        {{ showCode ? '隐藏' : '显示' }} SVG 代码
      </button>
    </div>

    <div v-if="showCode" class="bg-gray-900 text-green-400 p-4 rounded text-xs overflow-auto max-h-60 mb-6">
      <pre>{{ formattedSVG }}</pre>
    </div>

    <!-- SVG 编辑器 -->
    <div class="bg-white p-4 rounded shadow w-full max-w-3xl">
      <SVGEditor
        :svg-content="currentSVG"
        @svg-updated="onSVGUpdated"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import SVGEditor from './SVGEditor.vue'

// 初始 SVG 内容
const currentSVG = ref(``)
const showCode = ref(false)

// 模拟从后端获取初始 SVG
const fetchInitialSVG = async () => {
  // 实际项目中可以是 fetch 或 axios
  return `<svg width="600" height="800" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="#000" />
      </marker>
    </defs>
    <rect x="250" y="20" width="100" height="40" rx="10" ry="10" fill="#4CAF50" stroke="#388E3C" />
    <text x="300" y="45" font-family="Arial" font-size="14" fill="white" text-anchor="middle">开始</text>
    <rect x="230" y="90" width="140" height="50" rx="10" ry="10" fill="#2196F3" stroke="#0D47A1" />
    <text x="300" y="115" font-family="Arial" font-size="14" fill="white" text-anchor="middle">输入用户名和密码</text>
    <rect x="250" y="170" width="100" height="40" rx="10" ry="10" fill="#FFC107" stroke="#FF8F00" />
    <text x="300" y="195" font-family="Arial" font-size="14" fill="black" text-anchor="middle">验证</text>
    <rect x="250" y="240" width="100" height="40" rx="10" ry="10" fill="#FF9800" stroke="#E65100" />
    <text x="300" y="265" font-family="Arial" font-size="14" fill="white" text-anchor="middle">登录成功？</text>
    <rect x="100" y="310" width="100" height="40" rx="10" ry="10" fill="#8BC34A" stroke="#33691E" />
    <text x="150" y="335" font-family="Arial" font-size="14" fill="white" text-anchor="middle">登录成功</text>
    <rect x="400" y="310" width="100" height="40" rx="10" ry="10" fill="#F44336" stroke="#B71C1C" />
    <text x="450" y="335" font-family="Arial" font-size="14" fill="white" text-anchor="middle">登录失败</text>
    <rect x="250" y="380" width="100" height="40" rx="10" ry="10" fill="#9E9E9E" stroke="#424242" />
    <text x="300" y="405" font-family="Arial" font-size="14" fill="white" text-anchor="middle">结束</text>
    <line x1="300" y1="60" x2="300" y2="90" stroke="#000" stroke-width="2" marker-end="url(#arrow)" />
    <line x1="300" y1="140" x2="300" y2="170" stroke="#000" stroke-width="2" marker-end="url(#arrow)" />
    <line x1="300" y1="210" x2="300" y2="240" stroke="#000" stroke-width="2" marker-end="url(#arrow)" />
    <line x1="300" y1="280" x2="150" y2="310" stroke="#000" stroke-width="2" marker-end="url(#arrow)" />
    <line x1="300" y1="280" x2="450" y2="310" stroke="#000" stroke-width="2" marker-end="url(#arrow)" />
    <line x1="150" y1="350" x2="300" y2="380" stroke="#000" stroke-width="2" marker-end="url(#arrow)" />
    <line x1="450" y1="350" x2="300" y2="380" stroke="#000" stroke-width="2" marker-end="url(#arrow)" />
  </svg>`
}

// 初始化
fetchInitialSVG().then(svg => {
  currentSVG.value = svg
})

// 处理子组件更新
const onSVGUpdated = (newSVG) => {
  console.log('接收到更新的 SVG:')
  currentSVG.value = newSVG  // 同步更新（可选）
}

// 格式化用于显示
const formattedSVG = computed(() => {
  const parser = new DOMParser()
  const doc = parser.parseFromString(currentSVG.value, 'image/svg+xml')
  return new XMLSerializer().serializeToString(doc)
})
</script>
```
