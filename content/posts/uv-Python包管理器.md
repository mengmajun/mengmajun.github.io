+++ 
date = '2025-06-12' 
draft = false 
title = 'uv-Python包管理器' 
categories = ['Python包管理器'] 
+++

# 🚀 `uv` 常用命令速查表

> 💡 前提：项目根目录有合法的 `pyproject.toml`，推荐搭配 `uv.lock` 使用

> pip install uv 通过pip安装uv命令

> COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/ dockerfile中直接复制uv镜像中的文件到当前容器的/bin/目录中


---

## 🧱 1. 初始化 & 安装依赖

| 命令 | 作用 | 说明 |
|------|------|------|
| `uv sync` | 安装依赖 + 生成/更新 `uv.lock` | ✅ 首次安装、协作开发、部署时首选 |
| `uv sync --frozen` | 严格按 `uv.lock` 安装，不更新 | 🛡️ 用于生产环境，确保完全一致 |
| `uv sync --frozen=false` | 忽略 lock，重新解析依赖树 | 🔄 默认行为，用于更新依赖 |

---

## ➕ 2. 管理依赖（添加/删除）

| 命令 | 作用 | 说明 |
|------|------|------|
| `uv add requests` | 添加依赖到 `pyproject.toml` 并更新 `uv.lock` | ✅ 自动写入 `[project.dependencies]` |
| `uv add "requests@latest"` | 升级到最新版本 | 🆙 推荐用于单个包升级 |
| `uv add pytest --group dev` | 添加到可选依赖组（如 dev） | 🧪 用于测试/开发工具 |
| `uv remove requests` | 删除依赖并更新 `uv.lock` | 🗑️ 自动从 `pyproject.toml` 移除 |

> ⚠️ `uv add/remove` 是实验性功能，但已足够稳定用于日常开发

---

## 🔍 3. 检查 & 验证环境

| 命令 | 作用 | 说明 |
|------|------|------|
| `uv pip check` | 检查当前环境依赖是否一致/冲突 | ✅ 部署前或协作时推荐运行 |
| `uv pip list` | 列出已安装包 | 👀 快速查看当前环境 |
| `uv pip show requests` | 查看某个包的详细信息 | ℹ️ 类似 `pip show` |

---

## ♻️ 4. 更新 & 重置依赖

| 命令 | 作用 | 说明 |
|------|------|------|
| `uv add "包名@latest"` | 更新单个包到最新兼容版本 | ✅ 推荐方式 |
| `rm uv.lock && uv sync` | 删除 lock 并重新生成 | 🧹 彻底重置依赖树 |
| `uv pip install -e . --reinstall` | 强制重新安装项目+依赖 | 🔄 有时用于修复环境 |

> ❌ 不推荐：`uv pip install --upgrade --upgrade-package "*"` → 可能破坏兼容性

---

## ▶️ 5. 运行脚本 & 激活环境

| 命令 | 作用 | 说明 |
|------|------|------|
| `uv run python your_script.py` | 在项目环境中运行脚本 | ✅ 无需手动激活虚拟环境 |
| `uv run pytest` | 运行测试（假设 pytest 已安装） | 🧪 |
| `uv run black .` | 格式化代码 | 🎨 |
| `uv venv` | 创建虚拟环境（默认 `.venv`） | 🏗️ 可选，`uv` 默认自动管理环境 |
| `source .venv/bin/activate` (Linux/macOS) | 手动激活虚拟环境 | ⌨️ 传统方式，非必需 |

> 💡 `uv run` 会自动识别并使用项目虚拟环境，非常方便！

---

## 📦 6. 其他实用命令

| 命令 | 作用 |
|------|------|
| `uv cache dir` | 查看缓存目录 |
| `uv cache clean` | 清理下载缓存 |
| `uv python list` | 列出可用 Python 版本（需已安装） |
| `uv python install 3.11` | 安装指定 Python 版本（实验性） |

---

## 📌 附：推荐工作流（现代 Python 项目）

```bash
git clone your-project
cd your-project

# 安装依赖（自动生成/读取 uv.lock）
uv sync

# 开发时添加依赖
uv add black --group dev

# 运行测试
uv run pytest

# 部署前检查
uv pip check

# 部署时严格锁定版本
uv sync --frozen
```

---

## ✅ 一句话总结

> `uv sync` 装依赖，`uv add/remove` 管包，`uv run` 跑命令，`uv pip check` 保平安 —— 速度飞快，标准兼容，现代 Python 开发神器！

---
