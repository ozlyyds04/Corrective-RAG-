"""Corrective RAG 的 FastAPI 后端（替代原 Streamlit 界面）。

前端为 Vue3 SPA，通过 SSE 实现逐字流式回答与检索证据可视化。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from corrective_rag import jobs
from corrective_rag.core import (
    SUPPORTED_EXTENSIONS,
    CorrectiveRAG,
    load_knowledge_bases,
    save_knowledge_bases,
)
from corrective_rag.memory import MemoryStore

load_dotenv()

KB_REGISTRY_PATH = os.getenv(
    "KB_REGISTRY_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "knowledge_bases.json"),
)
REDIS_URL = os.getenv("REDIS_URL", "")
WATCH_DIR = os.getenv("WATCH_DIR", "")
WATCH_KB = os.getenv("WATCH_KB", "默认")
HOST_MOUNT_PATH = os.getenv("HOST_MOUNT_PATH", "")
CONTAINER_MOUNT_PATH = os.getenv("CONTAINER_MOUNT_PATH", "/host/docs")
API_TOKEN = os.getenv("API_TOKEN", "")
APP_DEBUG = os.getenv("APP_DEBUG", "").lower() in ("1", "true", "yes")
MAX_UPLOAD_FILES = 200


class _Cancelled(Exception):
    """客户端断开时取消长任务的内部标记。"""


def _safe_error(exc: Exception) -> str:
    """回传前端的错误信息：调试模式返回原文，否则只给类型避免泄露细节。"""
    if APP_DEBUG:
        return str(exc)
    return f"{type(exc).__name__}，具体原因见服务日志"


def _containerize_path(p: str) -> str:
    """把宿主机路径映射为容器内的挂载路径。

    仅当配置了 HOST_MOUNT_PATH 且传入路径以其为前缀时生效，否则原样返回。
    例如 HOST_MOUNT_PATH="D:\\Desktop"、CONTAINER_MOUNT_PATH="/host/docs" 时，
    "D:\\Desktop\\Agent八股" -> "/host/docs/Agent八股"。
    """
    if not p or not HOST_MOUNT_PATH:
        return p
    p_norm = p.replace("\\", "/")
    host_norm = HOST_MOUNT_PATH.replace("\\", "/").rstrip("/")
    if p_norm.startswith(host_norm):
        rest = p_norm[len(host_norm):].lstrip("/")
        return f"{CONTAINER_MOUNT_PATH.rstrip('/')}/{rest}" if rest else CONTAINER_MOUNT_PATH.rstrip("/")
    return p


def _get_rag() -> CorrectiveRAG:
    """按当前 .env 构建（并缓存）唯一的 CorrectiveRAG，保证检索器可复用。"""
    if not hasattr(_get_rag, "_instance"):
        _get_rag._instance = CorrectiveRAG(
            openai_api_key=os.getenv("LLM_API_KEY", ""),
            openai_base_url=os.getenv("LLM_BASE_URL", ""),
            model=os.getenv("LLM_MODEL", "deepseek-v4-flash-vision-exp"),
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
                "index": len(sources) + 1,
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
    session_id: str = ""


class IngestRequest(BaseModel):
    urls: List[str] = []
    local_paths: List[str] = []
    kb_name: str = "默认"
    max_files: int = 50


class CreateKBRequest(BaseModel):
    name: str


_memory: Optional[MemoryStore] = None

# 持有后台任务的强引用：事件循环对 task 只保留弱引用，不存下来可能在执行中被 GC
_background_tasks: set = set()


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def get_memory() -> MemoryStore:
    global _memory
    if _memory is None:
        _memory = MemoryStore(_get_rag(), REDIS_URL)
    return _memory


def _memory_context(question: str, session_id: str, kb_name: str) -> tuple[str, str]:
    """读取短期对话历史与情景记忆，返回（对话历史文本, 情景记忆文本）。"""
    mem = get_memory()
    dialogue = ""
    memories = ""
    if session_id:
        dialogue = mem.build_dialogue(mem.get_history(session_id))
        episodes = mem.retrieve_episodic(question, kb_name, 3)
        if episodes:
            memories = "\n".join(f"Q: {e['question']}\nA: {e['answer']}" for e in episodes)
    return dialogue, memories


def _persist_memory(
    session_id: str,
    question: str,
    answer: str,
    kb_name: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    steps: Optional[List[str]] = None,
) -> None:
    """把一轮问答写入短期记忆（Redis）和情景记忆（Qdrant 摘要）。"""
    if not session_id:
        return
    mem = get_memory()
    mem.push_message(session_id, "user", question)
    mem.push_message(
        session_id,
        "assistant",
        answer,
        extra={"sources": sources or [], "steps": steps or []},
    )
    mem.add_episodic(question, answer, kb_name)


# ---------- 生命周期：启动增量更新的 worker 与文件监听 ----------


@asynccontextmanager
async def lifespan(app: FastAPI):
    if REDIS_URL:
        jobs.start_worker(REDIS_URL, WATCH_KB)
        if WATCH_DIR:
            jobs.start_watcher(REDIS_URL, _containerize_path(WATCH_DIR), WATCH_KB)
    yield
    jobs.stop_all()


app = FastAPI(
    title="Corrective RAG API",
    version="0.1.0",
    lifespan=lifespan,
    # 设置 API_TOKEN 时关闭交互文档：/docs、/openapi.json 不带 /api 前缀，
    # 不关的话会绕过鉴权暴露完整接口定义
    docs_url=None if API_TOKEN else "/docs",
    redoc_url=None if API_TOKEN else "/redoc",
    openapi_url=None if API_TOKEN else "/openapi.json",
)

_origins = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8081"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_PATHS = {"/api/health", "/api/config"}


@app.middleware("http")
async def _auth_middleware(request, call_next):
    """若配置了 API_TOKEN 则要求 Bearer / X-API-Token 头；health/config 免鉴权。"""
    if request.method == "OPTIONS":
        return await call_next(request)
    if API_TOKEN and request.url.path.startswith("/api/") and request.url.path not in PUBLIC_PATHS:
        auth = request.headers.get("authorization", "")
        token = request.headers.get("x-api-token", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        if token.startswith("Bearer "):
            token = token[7:].strip()
        if token != API_TOKEN:
            return JSONResponse({"detail": "未授权"}, status_code=401)
    return await call_next(request)


# ---------- 基础 ----------


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "configured": _configured()}


@app.get("/api/config")
def config() -> Dict[str, Any]:
    return {
        "llm_model": os.getenv("LLM_MODEL", "deepseek-v4-flash-vision-exp"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding"),
        "qdrant_url": os.getenv("QDRANT_URL", ""),
        "has_llm_key": bool(os.getenv("LLM_API_KEY")),
        "has_embedding_key": bool(os.getenv("EMBEDDING_API_KEY")),
        "has_tavily_key": bool(os.getenv("TAVILY_API_KEY")),
        "watch_dir": _containerize_path(WATCH_DIR),
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


# ---------- 会话管理（多会话 + 短期记忆） ----------


@app.get("/api/sessions")
def list_sessions() -> Dict[str, Any]:
    return {"sessions": get_memory().list_sessions()}


@app.get("/api/sessions/{sid}/messages")
def session_messages(sid: str) -> Dict[str, Any]:
    return {"messages": get_memory().get_messages(sid)}


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str) -> Dict[str, Any]:
    get_memory().delete_session(sid)
    return {"ok": True}


# ---------- 查询（流式 + 非流式） ----------


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """SSE 流式：先发 sources（召回来源 + 重排分数），再逐 token 发 answer。"""

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop = threading.Event()

    def emit(event: str, data: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))
        except RuntimeError:
            # 事件循环已关闭（服务关停），丢弃即可
            pass

    def work() -> None:
        rag = _get_rag()
        if not rag.prepare_retriever(req.kb_name):
            emit("error", {"message": "请先入库文档，再提出问题。"})
            return
        try:
            documents, steps = rag.run_evidence(req.question, kb_name=req.kb_name)
            documents = rag.prepare_context(documents)
            dialogue, memories = _memory_context(req.question, req.session_id, req.kb_name)
            sources = build_sources(documents)
            step_names = summarize_steps(steps)
            emit("sources", {"sources": sources, "steps": step_names})
            answer_parts: List[str] = []
            for token in rag._generate_stream(
                req.question, documents, dialogue_history=dialogue, memories=memories
            ):
                if stop.is_set():
                    break
                answer_parts.append(token)
                emit("token", {"text": token})
            if stop.is_set():
                return
            answer = "".join(answer_parts)
            if req.session_id:
                _persist_memory(req.session_id, req.question, answer, req.kb_name, sources, step_names)
            emit("done", {"answer": answer})
        except Exception as exc:  # noqa: BLE001
            if not stop.is_set():
                emit("error", {"message": _safe_error(exc)})

    _spawn_background(asyncio.to_thread(work))

    async def gen() -> Iterator[str]:
        try:
            while True:
                event, data = await queue.get()
                yield sse(event, data)
                if event in ("done", "error"):
                    break
        except asyncio.CancelledError:
            stop.set()
            raise
        finally:
            stop.set()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/query")
async def query(req: QueryRequest) -> Dict[str, Any]:
    def _run() -> Dict[str, Any]:
        try:
            rag = _get_rag()
            if not rag.prepare_retriever(req.kb_name):
                raise HTTPException(400, "请先入库文档，再提出问题。")
            documents, steps = rag.run_evidence(req.question, kb_name=req.kb_name)
            documents = rag.prepare_context(documents)
            dialogue, memories = _memory_context(req.question, req.session_id, req.kb_name)
            answer = rag._generate(
                req.question, documents, dialogue_history=dialogue, memories=memories
            )
            sources = build_sources(documents)
            step_names = summarize_steps(steps)
            _persist_memory(req.session_id, req.question, answer, req.kb_name, sources, step_names)
            return {"answer": answer, "sources": sources, "steps": step_names}
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, _safe_error(exc)) from exc

    return await asyncio.to_thread(_run)


# ---------- 入库 ----------


@app.post("/api/ingest/stream")
async def ingest_stream(req: IngestRequest) -> StreamingResponse:
    """JSON 来源（URL / 本地路径）入库，SSE 汇报进度。"""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stop = threading.Event()

    def emit(item: Dict[str, Any]) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            # 事件循环已关闭（服务关停），丢弃即可
            pass

    def on_progress(processed: int, total: int, chunks: int) -> None:
        if stop.is_set():
            raise _Cancelled()
        emit(
            {"type": "progress", "processed": processed, "total": total, "chunks": chunks}
        )

    def work() -> None:
        rag = _get_rag()
        try:
            sources = list(req.urls) + [_containerize_path(p) for p in req.local_paths]
            chunk_count, errors = rag.ingest_stream(
                sources, kb_name=req.kb_name, progress_callback=on_progress
            )
            emit({"type": "done", "chunk_count": chunk_count, "errors": errors})
        except _Cancelled:
            return
        except Exception as exc:  # noqa: BLE001
            if not stop.is_set():
                emit({"type": "error", "message": _safe_error(exc)})

    _spawn_background(asyncio.to_thread(work))

    async def gen() -> Iterator[str]:
        try:
            yield sse("start", {"kb_name": req.kb_name})
            while True:
                item = await queue.get()
                yield sse(item["type"], item)
                if item["type"] in ("done", "error"):
                    break
        except asyncio.CancelledError:
            stop.set()
            raise
        finally:
            stop.set()

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/ingest/upload")
def ingest_upload(
    file: List[UploadFile],
    kb_name: str = Form("默认"),
    max_files: int = Form(50),
) -> Dict[str, Any]:
    """文件上传入库（非流式），返回汇总信息。"""
    max_files = max(1, min(int(max_files), MAX_UPLOAD_FILES))
    supported = [
        uf
        for uf in file
        if os.path.splitext(uf.filename or "")[1].lower() in SUPPORTED_EXTENSIONS
    ]
    skipped_unsupported = len(file) - len(supported)
    selected = supported[:max_files]
    skipped_over_limit = len(supported) - len(selected)
    if not selected:
        raise HTTPException(
            400, "没有可入库的文件：全部类型不受支持，或超过数量上限。"
        )
    tmp_paths: List[str] = []
    for uf in selected:
        suffix = os.path.splitext(uf.filename or "")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(uf.file, tmp)
            tmp_paths.append(tmp.name)
    try:
        try:
            rag = _get_rag()
            chunk_count, errors = rag.ingest_stream(tmp_paths, kb_name=kb_name)
            return {
                "chunk_count": chunk_count,
                "errors": [{"source": src, "error": err} for src, err in errors],
                "skipped_unsupported": skipped_unsupported,
                "skipped_over_limit": skipped_over_limit,
            }
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, _safe_error(exc)) from exc
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
    directory = _containerize_path(req.get("dir") or WATCH_DIR)
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
        yield sse("connected", {"watch_dir": _containerize_path(WATCH_DIR), "kb_name": WATCH_KB})
        try:
            async for item in jobs.watch_events(REDIS_URL):
                yield sse(item["type"], item)
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"message": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")
