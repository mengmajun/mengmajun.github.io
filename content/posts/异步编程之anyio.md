+++ 
date = '2025-07-26' 
draft = false 
title = '异步编程之anyio' 
categories = ['异步编程'] 
tags = ['anyio'] 
+++


在 FastAPI 后端开发中，`anyio` 是一个强大的异步 I/O 库，它提供了跨后端的异步编程支持（如 `asyncio` 和 `trio`），并简化了同步/异步代码的混合调用。以下是 `anyio` 的常用开发实践，结合 FastAPI 场景：

---

### 1. **同步代码转异步（避免阻塞事件循环）**
**场景**：调用同步库（如 `requests`、`pandas`）时，使用 `anyio.to_thread.run_sync` 在后台线程中运行，避免阻塞事件循环。

```python
from fastapi import FastAPI
import anyio.to_thread
import requests  # 同步库

app = FastAPI()

@app.get("/fetch-data")
async def fetch_data(url: str):
    # 将同步的 requests.get 放到线程中运行
    response = await anyio.to_thread.run_sync(requests.get, url)
    return response.json()
```

**优势**：  
- 保持 FastAPI 的异步非阻塞特性。  
- 对比 `asyncio.to_thread`，`anyio` 的版本兼容更多后端（如 `trio`）。

---

### 2. **异步任务并发控制**
**场景**：限制并发任务数量（如爬虫、批量调用外部 API）。

```python
import anyio
from fastapi import FastAPI

app = FastAPI()

async def fetch_item(item_id: int):
    await anyio.sleep(1)  # 模拟异步操作
    return f"result-{item_id}"

@app.get("/process-items")
async def process_items():
    items = list(range(10))
    # 限制最大并发数为 3
    async with anyio.create_task_group() as tg:
        for item in items:
            tg.start_soon(fetch_item, item)
    return {"status": "done"}
```

**关键方法**：  
- `anyio.create_task_group()`：类似 `asyncio.gather`，但提供更精细的任务管理。  
- `anyio.Semaphore`：手动控制并发数（示例见下文）。

---

### 3. **同步/异步互操作**
**场景**：在异步上下文中调用同步代码，或在同步代码中调用异步函数。

#### 异步调用同步代码（常用）：
```python
from fastapi import FastAPI
import anyio.to_thread

app = FastAPI()

def sync_heavy_computation():
    import time
    time.sleep(2)  # 模拟 CPU 密集型任务
    return "done"

@app.get("/compute")
async def compute():
    result = await anyio.to_thread.run_sync(sync_heavy_computation)
    return {"result": result}
```

#### 同步调用异步代码（较少见）：
```python
import anyio
from fastapi import FastAPI

app = FastAPI()

async def async_task():
    await anyio.sleep(1)
    return "async-result"

def sync_wrapper():
    # 在同步函数中运行异步代码
    return anyio.run(async_task)
```

---

### 4. **超时控制**
**场景**：为异步任务设置超时，避免长时间阻塞。

```python
from fastapi import FastAPI
import anyio
from anyio import fail_after, TooSlowError

app = FastAPI()

@app.get("/timeout-test")
async def timeout_test():
    try:
        # 设置 2 秒超时
        with fail_after(2):
            await anyio.sleep(3)  # 会触发超时
    except TooSlowError:
        return {"error": "Timeout"}
    return {"status": "ok"}
```

---

### 5. **文件 I/O 异步化**
**场景**：使用 `anyio` 的异步文件操作替代同步 `open()`。

```python
from fastapi import FastAPI
import anyio

app = FastAPI()

@app.post("/write-log")
async def write_log(message: str):
    async with await anyio.open_file("log.txt", mode="a") as f:
        await f.write(f"{message}\n")
    return {"status": "saved"}
```

**对比**：  
- 比 `aiofiles` 更轻量（`anyio` 内置文件操作）。  
- 支持跨后端统一 API。

---

### 6. **信号量控制并发**
**场景**：限制数据库连接池等共享资源的并发访问。

```python
from fastapi import FastAPI
import anyio

app = FastAPI()
semaphore = anyio.Semaphore(3)  # 最大并发数 3

async def query_database(query: str):
    async with semaphore:
        await anyio.sleep(1)  # 模拟数据库查询
        return f"result-{query}"

@app.get("/search")
async def search(q: str):
    result = await query_database(q)
    return {"result": result}
```

---

### 7. **测试异步代码**
**场景**：使用 `anyio` 的测试工具编写异步测试。

```python
import pytest
import anyio
from fastapi.testclient import TestClient
from myapp import app

client = TestClient(app)

@pytest.mark.anyio  # 使用 anyio 的 pytest 插件
async def test_fetch_data():
    response = await client.get("/fetch-data?url=http://example.com")
    assert response.status_code == 200
```

---

### 最佳实践总结
1. **优先纯异步**：尽量用异步库（如 `httpx` 替代 `requests`）。  
2. **同步代码线程化**：对无法避免的同步调用，用 `anyio.to_thread.run_sync`。  
3. **资源控制**：使用 `Semaphore` 或 `TaskGroup` 管理并发。  
4. **超时必加**：所有外部调用设置超时。  
5. **统一后端**：明确项目使用的异步后端（如 `asyncio`），避免混用。

通过 `anyio`，你可以更安全地在 FastAPI 中混合同步/异步代码，同时保持高性能和可维护性。
