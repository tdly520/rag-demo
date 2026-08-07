# RAG Demo — 本地 PDF 问答服务

一个精简的 RAG + Agent Demo：加载本地 PDF，切块后存入 FAISS，Agent 根据问题自主决定是否调用知识库检索工具，再基于证据回答并返回来源。

## 技术选型

| 环节 | 选型 | 原因 |
|:--|:--|:--|
| PDF 加载 | LangChain `PyPDFLoader` | 一行读 PDF，生态完整 |
| 文本切块 | `RecursiveCharacterTextSplitter` | 按段落/句子边界切，保留语义 |
| 向量化 | `sentence-transformers`（all-MiniLM-L6-v2） | 本地离线运行，免费、快 |
| 向量检索 | FAISS | Meta 开源，轻量，适合本地 Demo |
| 大模型 | DeepSeek（OpenAI 兼容接口） | 便宜、中文好，OpenAI SDK 直接调 |
| 服务框架 | FastAPI | 自带 /docs 交互页面，调试方便 |

## 项目结构

```
rag-demo/
├── app.py            # FastAPI 服务（加载 PDF → 建库 → Agent + RAG tool → /ask）
├── requirements.txt  # 依赖清单
├── .env              # API 密钥（不入库）
├── data/
│   └── paper.pdf     # 要检索的 PDF（本地放入，不入库）
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置密钥

复制 `.env.example` 为 `.env`，填入你的 API 密钥：

```bash
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
# 可选：指定本地 sentence-transformers 模型目录
# EMBEDDING_MODEL=C:\path\to\all-MiniLM-L6-v2
```

### 3. 放入 PDF

把你的 PDF 放到 `data/` 目录，命名为 `paper.pdf`（或用环境变量 `PDF_PATH` 指定其他路径）。

### 4. 启动服务

PowerShell（本机已有模型缓存的启动方式）：

```powershell
$env:Path = "C:\Users\ckk\Desktop\留学\week1-llm-api\.venv\Scripts;" + $env:Path
$env:EMBEDDING_MODEL = "C:\Users\ckk\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
Set-Location "C:\Users\ckk\Desktop\留学\rag-demo"
python -m uvicorn app:app --host 127.0.0.1 --port 8001
```

通用启动方式（模型可从 Hugging Face 下载或已由当前 Python 环境正确缓存）：

```bash
uvicorn app:app --reload
```

启动时会加载 embedding 模型、PDF 并构建向量库，看到 `[startup] 已加载 PDF，切成 N 块` 表示就绪。若本地缓存自动发现失败，必须通过 `EMBEDDING_MODEL` 指定模型目录；不要只设置 `HF_HOME`。

### 5. 提问

浏览器打开 http://127.0.0.1:8000/docs，在 `/ask` 接口试试：

```json
{"question": "How much does the token consumption reduce compared to GraphRAG?"}
```

返回：

```json
{
  "answer": "基于论文内容整理的回答...",
  "sources": [{"source": "paper.pdf p.7", "content": "命中片段原文..."}],
  "tool_calls": ["retrieve_knowledge_base"]
}
```

## 三类演示问题

1. 直接回答：`什么是 Python？`，预期 `tool_calls` 为空。
2. 触发检索：`paper.pdf 中 GraphRAG 的 token consumption 是多少？`，预期调用 `retrieve_knowledge_base` 并返回来源。
3. 信息不足：`paper.pdf 是否讨论了量子计算？`，预期先检索，再明确说明现有证据不足，不编造结论。

## API

| 方法 | 路径 | 说明 |
|:--|:--|:--|
| GET | `/health` | 健康检查，返回 `{"status": "ok"}` |
| POST | `/ask` | 提问，Agent 自主决定是否检索，返回 `answer`、`sources` 和 `tool_calls` |

## 调参建议

`app.py` 里可以改三个参数：

- `CHUNK_SIZE`（默认 500）：切块大小。太小信息不完整，太大检索不精准
- `CHUNK_OVERLAP`（默认 50）：相邻块重叠，避免句子在边界被切断
- `RETRIEVE_K`（默认 3）：每次检索返回的片段数，也是实际发给大模型的片段数

## 原理一句话

Agent 先判断问题是否需要文档证据；需要时调用 `retrieve_knowledge_base`：PDF 切块 → 转向量存库 → 问题转向量 → 相似度最高的 K 块 → 将结果连同来源交给大模型组织语言。这样常识问题不必检索，文档问题可追溯。
