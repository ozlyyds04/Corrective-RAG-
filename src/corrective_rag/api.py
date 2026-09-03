"""Corrective RAG 的 FastAPI 后端（替代原 Streamlit 界面）。

前端为 Vue3 SPA，通过 SSE 实现逐字流式回答与检索证据可视化。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from corrective_rag import jobs
from corrective_rag.core import (
    CorrectiveRAG,
    load_knowledge_bases,
    save_knowledge_bases,
)

load_dotenv()

KB_REGISTRY_PATH = os.getenv(
    "KB_REGISTRY_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "knowledge_bases.json"),
)
REDIS_URL = os.getenv("REDIS_URL", "")
WATCH_DIR = os.getenv("WATCH_DIR", "")
WATCH_KB = os.getenv("WATCH_KB", "默认")


def _get_rag() -> CorrectiveRAG:
    """按当前 .env 构建（并缓存）唯一的 CorrectiveRAG，保证检索器可复用。"""
    if not hasattr(_get_rag, "_instance"):
        _get_rag._instance = CorrectiveRAG(
            openai_api_key=os.getenv("LLM_API_KEY", ""),
            openai_base_url=os.getenv("LLM_BASE_URL", ""),
            model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding"),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            qdrant_url=os.getenv("QDRANT_URL", ""),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        )
    return _get_rag._instance


def _configured() -> bool:
    return bool(
        os.getenv("QDRANT_URL")
        and (os.getenv("LLM_API_KEY") or os.getenv("EMBEDDING_API_KEY"))
    )


def sse(event: str, data: Any) -> str:
    """把数据编码为 Server-Sent Events 帧。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_sources(documents) -> List[Dict[str, Any]]:
    """把检索到的文档转成前端可视化所需的来源结构。"""
    sources: List[Dict[str, Any]] = []
    for doc in documents:
        md = doc.metadata or {}
        sources.append(
            {
                "source": md.get("source", "未知"),
                "title": md.get("title", "无标题"),
                "url": md.get("url", ""),
                "score": md.get("rerank_score"),
                "snippet": doc.page_content[:240],
            }
        )
    return sources


def summarize_steps(steps: Dict[str, Dict[str, Any]]) -> List[str]:
    """只保留各节点名称，避免把文档对象序列化给前端。"""
    return list(steps.keys())


# ---------- 请求模型 ----------


class QueryRequest(BaseModel):
    question: str
    kb_name: str = "默认"


class IngestRequest(BaseModel):
    urls: List[str] = []
    local_paths: List[str] = []
    kb_name: str = "默认"
    max_files: int = 50


class CreateKBRequest(BaseModel):
    name: str


# ---------- 生命周期：启动增量更新的 worker 与文件监听 ----------


@asynccontextmanager
async def lifespan(app: FastAPI):
    if REDIS_URL:
        jobs.start_worker(REDIS_URL, WATCH_KB)
        if WATCH_DIR:
            jobs.start_watcher(REDIS_URL, WATCH_DIR, WATCH_KB)
    yield
    jobs.stop_all()


app = FastAPI(title="Corrective RAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 基础 ----------


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "configured": _configured()}


@app.get("/api/config")
def config() -> Dict[str, Any]:
    return {
        "llm_model": os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding"),
        "qdrant_url": os.getenv("QDRANT_URL", ""),
        "has_llm_key": bool(os.getenv("LLM_API_KEY")),
        "has_embedding_key": bool(os.getenv("EMBEDDING_API_KEY")),
        "has_tavily_key": bool(os.getenv("TAVILY_API_KEY")),
        "watch_dir": WATCH_DIR,
        "watch_enabled": bool(REDIS_URL and WATCH_DIR),
    }


# ---------- 知识库管理 ----------


@app.get("/api/kbs")
def list_kbs() -> Dict[str, Any]:
    return {"kbs": load_knowledge_bases(KB_REGISTRY_PATH)}


@app.post("/api/kbs")
def create_kb(req: CreateKBRequest) -> Dict[str, Any]:
    names = load_knowledge_bases(KB_REGISTRY_PATH)
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "知识库名称不能为空。")
    if name in names:
        raise HTTPException(409, f"知识库「{name}」已存在。")
    names.append(name)
    save_knowledge_bases(KB_REGISTRY_PATH, names)
    return {"ok": True, "kbs": names}


@app.delete("/api/kbs/{name}")
def delete_kb(name: str) -> Dict[str, Any]:
    if name == "默认":
        raise HTTPException(400, "默认知识库不可删除。")
    names = load_knowledge_bases(KB_REGISTRY_PATH)
    if name not in names:
        raise HTTPException(404, "知识库不存在。")
    if os.getenv("QDRANT_URL"):
        _get_rag().delete_knowledge_base(name)
    names.remove(name)
    save_knowledge_bases(KB_REGISTRY_PATH, names)
    return {"ok": True, "kbs": names}


# ---------- 查询（流式 + 非流式） ----------


@app.post("/api/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    """SSE 流式：先发 sources（召回来源 + 重排分数），再逐 token 发 answer。"""

    def gen() -> Iterator[str]:
        rag = _get_rag()
        if rag.retriever is None:
            yield sse("error", {"message": "请先入库文档，再提出问题。"})
            return
        try:
            documents, steps = rag.run_evidence(
                req.question, progress_callback=lambda name, idx: None
            )
            yield sse(
                "sources",
                {
                    "sources": build_sources(steps.get("retrieve", {}).get("documents", [])),
                    "steps": summarize_steps(steps),
                },
            )
            answer_parts: List[str] = []
            for token in rag._generate_stream(req.question, documents):
                answer_parts.append(token)
                yield sse("token", {"text": token})
            yield sse("done", {"answer": "".join(answer_parts)})
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"message": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/query")
def query(req: QueryRequest) -> Dict[str, Any]:
    rag = _get_rag()
    if rag.retriever is None:
        raise HTTPException(400, "请先入库文档，再提出问题。")
    documents, steps = rag.run_evidence(req.question)
    return {
        "answer": rag._generate(req.question, documents),
        "sources": build_sources(steps.get("retrieve", {}).get("documents", [])),
        "steps": summarize_steps(steps),
    }


# ---------- 入库 ----------


@app.post("/api/ingest/stream")
async def ingest_stream(req: IngestRequest) -> StreamingResponse:
    """JSON 来源（URL / 本地路径）入库，SSE 汇报进度。"""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_progress(processed: int, total: int, chunks: int) -> None:
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "progress", "processed": processed, "total": total, "chunks": chunks},
        )

    def work() -> None:
        rag = _get_rag()
        try:
            sources = list(req.urls) + list(req.local_paths)
            chunk_count, errors = rag.ingest_stream(
                sources, kb_name=req.kb_name, progress_callback=on_progress
            )
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "done", "chunk_count": chunk_count, "errors": errors}
            )
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(exc)})

    asyncio.create_task(asyncio.to_thread(work))

    async def gen() -> Iterator[str]:
        yield sse("start", {"kb_name": req.kb_name})
        while True:
            item = await queue.get()
            yield sse(item["type"], item)
            if item["type"] in ("done", "error"):
                break

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/ingest/upload")
def ingest_upload(
    file: List[UploadFile],
    kb_name: str = Form("默认"),
    max_files: int = Form(50),
) -> Dict[str, Any]:
    """文件上传入库（非流式），返回汇总信息。"""
    selected = file[:max_files]
    tmp_paths: List[str] = []
    for uf in selected:
        suffix = os.path.splitext(uf.filename or "")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(uf.file, tmp)
            tmp_paths.append(tmp.name)
    try:
        rag = _get_rag()
        chunk_count, errors = rag.ingest_stream(tmp_paths, kb_name=kb_name)
        return {
            "chunk_count": chunk_count,
            "errors": [{"source": src, "error": err} for src, err in errors],
        }
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------- 增量更新（文件监听 + Redis 队列） ----------


@app.get("/api/watch/status")
def watch_status() -> Dict[str, Any]:
    return jobs.watch_status(REDIS_URL)


@app.post("/api/watch/start")
def watch_start(req: Dict[str, Any]) -> Dict[str, Any]:
    if not REDIS_URL:
        raise HTTPException(400, "未配置 REDIS_URL，无法启用文件监听。")
    directory = req.get("dir") or WATCH_DIR
    kb_name = req.get("kb_name", WATCH_KB)
    if not directory:
        raise HTTPException(400, "缺少监听目录。")
    jobs.start_watcher(REDIS_URL, directory, kb_name)
    return {"ok": True, "watch_status": jobs.watch_status(REDIS_URL)}


@app.post("/api/watch/stop")
def watch_stop() -> Dict[str, Any]:
    jobs.stop_watcher()
    return {"ok": True, "watch_status": jobs.watch_status(REDIS_URL)}


@app.get("/api/watch/stream")
async def watch_stream() -> StreamingResponse:
    """订阅 Redis 事件，SSE 推送增量入库进度。"""

    async def gen() -> Iterator[str]:
        if not REDIS_URL:
            yield sse("error", {"message": "未配置 REDIS_URL。"})
            return
        yield sse("connected", {"watch_dir": WATCH_DIR, "kb_name": WATCH_KB})
        try:
            async for item in jobs.watch_events(REDIS_URL):
                yield sse(item["type"], item)
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"message": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")
