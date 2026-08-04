# RAG Demo — 本地 PDF 问答服务

一个精简的 RAG（Retrieval-Augmented Generation）Demo：加载本地 PDF，切块后存入向量库，收到问题时先检索最相关的片段，再让大模型基于这些片段回答，并返回引用来源。

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
├── app.py            # FastAPI 服务（加载 PDF → 建库 → /ask）
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
```

### 3. 放入 PDF

把你的 PDF 放到 `data/` 目录，命名为 `paper.pdf`（或用环境变量 `PDF_PATH` 指定其他路径）。

### 4. 启动服务

```bash
uvicorn app:app --reload
```

启动时会加载 PDF 并构建向量库，看到 `[startup] 已加载 PDF，切成 N 块` 表示就绪。

### 5. 提问

浏览器打开 http://127.0.0.1:8000/docs，在 `/ask` 接口试试：

```json
{"question": "How much does the token consumption reduce compared to GraphRAG?"}
```

返回：

```json
{
  "answer": "Our approach uses 9,580 tokens on average, which is 96.7% lower than GraphRAG's 293,872 tokens.",
  "sources": ["命中片段原文..."]
}
```

## API

| 方法 | 路径 | 说明 |
|:--|:--|:--|
| GET | `/health` | 健康检查，返回 `{"status": "ok"}` |
| POST | `/ask` | 提问，请求 `{"question": "..."}`，返回 `{"answer": "...", "sources": [...]}` |

## 调参建议

`app.py` 里可以改三个参数：

- `CHUNK_SIZE`（默认 500）：切块大小。太小信息不完整，太大检索不精准
- `CHUNK_OVERLAP`（默认 50）：相邻块重叠，避免句子在边界被切断
- `RETRIEVE_K`（默认 3）：每次检索返回的片段数，也是实际发给大模型的片段数

## 原理一句话

检索是向量搜索做的，不是大模型做的：PDF 切块 → 转向量存库 → 问题转向量 → 相似度最高的 K 块 → 只把这 K 块发给大模型组织语言。这样省 token、响应快、回答有依据。
