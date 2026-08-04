# -*- coding: utf-8 -*-
"""
RAG Demo — FastAPI 服务（Day 3 填空版）

把 Day 2 验证过的检索流程包装成 HTTP 接口：
  GET  /health            → 健康检查
  POST /ask               → 接收 {"question": "..."}，返回 {"answer": "...", "sources": [...]}

启动时只加载一次 PDF / 向量库（全局变量），不要每次请求都重建。
运行：uvicorn app:app --reload（在 rag-demo 目录下）
"""

import os
# 模型已缓存，强制离线加载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 清空代理变量：避免本机代理/安全软件拦截对 DeepSeek API 的调用
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from dotenv import load_dotenv
load_dotenv()  # 读取本目录 .env（把 API 密钥放这里）

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


# ===== 配置 =====
# 默认检索 data/paper.pdf，可用环境变量 PDF_PATH 覆盖
PDF_PATH = os.getenv("PDF_PATH", "data/paper.pdf")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVE_K = 3


# ===== 启动时加载一次：PDF → 切块 → 向量库 → 检索器（复用 Day 2 逻辑） =====
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
chunks = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})

llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
)

print(f"[startup] 已加载 PDF，切成 {len(chunks)} 块")


app = FastAPI(title="RAG Demo")


# TODO 1: 定义请求体模型
#   提示：class AskRequest(BaseModel): 里面有一个字段 question: str
class AskRequest(BaseModel):
    question: str


# TODO 2: 健康检查接口
#   提示：@app.get("/health") 装饰一个函数，返回 {"status": "ok"}
@app.get("/health")
def health():
    return {"status": "ok"}



# TODO 3: 提问接口（今天的核心）
#   提示：
#     1. 用 retriever.invoke(req.question) 检索，得到 hits
#     2. 把 hits 拼成 context（"\\n\\n".join(...)）
#     3. 拼 prompt 送 llm.invoke(...)，拿到 answer
#     4. 返回 {"answer": answer.content, "sources": [每块原文]}
#     sources 里建议包含原文片段，方便用户核对回答是不是有依据
@app.post("/ask")
def ask(req: AskRequest):
    hits=retriever.invoke(req.question)
    context="\n\n".join([doc.page_content for doc in hits])
    prompt = f"""基于以下上下文回答问题。如果上下文不足以回答问题，请如实告知。

    上下文：
    {context}

    问题：{req.question}

    回答："""

    # 4. 调用 LLM
    response = llm.invoke(prompt)  # 返回 AIMessage
    answer = response.content

    # 5. 提取来源（每块的原文片段）
    sources = [doc.page_content for doc in hits]

    # 6. 返回结果
    return {
        "answer": answer,
        "sources": sources
    }
