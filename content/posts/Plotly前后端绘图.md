+++ 
date = '2025-08-11' 
draft = false 
title = 'Plotly前后端绘图' 
categories = ['plotly'] 
+++


## 前端

> pnpm install plotly.js-dist

```Vue
<!-- PlotlyChart.vue -->
<script setup>
import { onMounted, ref, onBeforeUnmount } from 'vue'
import Plotly from 'plotly.js-dist'

// 创建一个模板引用，指向图表容器
const plotContainer = ref(null)

// 图表加载函数
const loadChart = async () => {
  try {
    const response = await fetch('http://localhost:9000/chart-data') // 你的 FastAPI 地址
    if (!response.ok) throw new Error('网络错误：无法获取图表数据')

    const { data, layout, options } = await response.json()

    // 渲染图表
    Plotly.newPlot(plotContainer.value, data, layout, { ...options })
  } catch (error) {
    console.error('图表加载失败:', error)
    plotContainer.value.innerText = '图表加载失败'
  }
}

// 组件挂载后加载图表
onMounted(() => {
  loadChart()
})

// 组件销毁前清理图表（可选）
// 注意：Plotly.purge 是异步的，但通常足够快
onBeforeUnmount(() => {
  if (plotContainer.value) {
    Plotly.purge(plotContainer.value)
  }
})
</script>

<template>
  <div>
    <h2>从 FastAPI 加载 Plotly 图表</h2>
    <!-- 图表容器 -->
    <div ref="plotContainer" style="width: 100%; height: 500px; margin-top: 20px;"></div>
  </div>
</template>

<style scoped>
/* 可选：添加一些样式 */
div[ref="plotContainer"] {
  border: 1px solid #ddd;
  border-radius: 8px;
  background-color: #f9f9f9;
}
</style>
```

## 后端

> uv pip install  plotly

```
from fastapi import FastAPI
import plotly.graph_objects as go
import json
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chart-data")
def get_chart_data():
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    fig.update_layout(title="FastAPI 提供的数据")

    return {
        "data": fig.to_dict()["data"],
        "layout": fig.to_dict()["layout"],
        "options": {}
    }


if __name__ == '__main__':
    uvicorn.run("api_demo:app", host='0.0.0.0', port=9000, reload=False)

```
