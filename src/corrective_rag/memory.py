"""三层记忆：短期（Redis 对话历史）、长期（Qdrant 知识库）、情景（Qdrant 问答摘要）。

短期记忆让模型能引用多轮上下文；长期记忆就是文档向量库；情景记忆把历史问答
摘要向量化，下次提问时做语义检索后注入提示，用于沉淀经验。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import redis
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

MEMORY_COLLECTION = "rag-memory"
HISTORY_KEY = "rag:mem:history:"
SESSION_META_KEY = "rag:mem:session:"
HISTORY_TTL = 60 * 60 * 24  # 一天
HISTORY_TURNS = 8
# Redis 列表保留的消息上限：get_history 只读最后 HISTORY_TURNS 条，
# 但不设上限的话长会话会让列表无界增长（每条还带全量 sources）
HISTORY_MAX_MESSAGES = 100


class MemoryStore:
    """管理短期与情景记忆；长期记忆即文档知识库（由 RAG 检索器承担）。"""

    def __init__(self, rag, redis_url: str = "") -> None:
        self.rag = rag
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._redis_lock = threading.Lock()
        self._qdrant: Optional[QdrantClient] = None
        self._qdrant_lock = threading.Lock()
        # 集合只需确保一次；避免每次写入/检索都 get_collections 往返
        self._collection_ready = False

    @property
    def r(self) -> Optional[redis.Redis]:
        if self._redis is None and self.redis_url:
            with self._redis_lock:
                if self._redis is None:
                    self._redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    # ---------- 短期记忆 ----------

    def push_message(
        self, session_id: str, role: str, content: str, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        if not (self.redis_url and session_id and self.r):
            return
        msg: Dict[str, Any] = {"role": role, "content": content}
        if extra:
            msg.update(extra)
        key = HISTORY_KEY + session_id
        self.r.rpush(key, json.dumps(msg, ensure_ascii=False))
        self.r.ltrim(key, -HISTORY_MAX_MESSAGES, -1)
        self.r.expire(key, HISTORY_TTL)
        meta = SESSION_META_KEY + session_id
        if role == "user" and not self.r.hget(meta, "title"):
            self.r.hset(meta, "title", content.strip()[:40] or "新会话")
        self.r.hset(meta, "updated_at", time.time())
        self.r.hincrby(meta, "count", 1)
        self.r.expire(meta, HISTORY_TTL)

    def get_history(self, session_id: str, n: int = HISTORY_TURNS) -> List[Dict[str, str]]:
        if not (self.redis_url and session_id and self.r):
            return []
        items = self.r.lrange(HISTORY_KEY + session_id, -n, -1)
        out: List[Dict[str, str]] = []
        for item in items:
            try:
                out.append(json.loads(item))
            except (TypeError, ValueError):
                continue
        return out

    def clear_history(self, session_id: str) -> None:
        if session_id and self.r:
            self.r.delete(HISTORY_KEY + session_id)

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        if not (self.redis_url and session_id and self.r):
            return []
        items = self.r.lrange(HISTORY_KEY + session_id, 0, -1)
        out: List[Dict[str, Any]] = []
        for item in items:
            try:
                out.append(json.loads(item))
            except (TypeError, ValueError):
                continue
        return out

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not (self.redis_url and self.r):
            return []
        sessions: List[Dict[str, Any]] = []
        for key in self.r.scan_iter(SESSION_META_KEY + "*", count=200):
            sid = key[len(SESSION_META_KEY):]
            data = self.r.hgetall(key)
            sessions.append(
                {
                    "id": sid,
                    "title": data.get("title", "新会话"),
                    "updated_at": float(data.get("updated_at", 0) or 0),
                    "count": int(data.get("count", 0) or 0),
                }
            )
        sessions.sort(key=lambda s: s["updated_at"], reverse=True)
        return sessions[:limit]

    def delete_session(self, session_id: str) -> None:
        if not (self.redis_url and session_id and self.r):
            return
        self.r.delete(HISTORY_KEY + session_id, SESSION_META_KEY + session_id)

    @staticmethod
    def build_dialogue(history: List[Dict[str, str]], max_chars: int = 1800) -> str:
        parts = []
        for m in history:
            role = "用户" if m.get("role") == "user" else "助手"
            parts.append(f"{role}：{m.get('content', '')}")
        return "\n".join(parts)[-max_chars:]

    # ---------- 情景记忆 ----------

    def _client(self) -> Optional[QdrantClient]:
        if not self.rag.qdrant_url:
            return None
        if self._qdrant is None:
            with self._qdrant_lock:
                if self._qdrant is None:
                    self._qdrant = QdrantClient(
                        url=self.rag.qdrant_url,
                        api_key=self.rag.qdrant_api_key or None,
                        timeout=60,
                    )
        return self._qdrant

    def _ensure_memory_collection(self, client: QdrantClient, embeddings) -> None:
        if self._collection_ready:
            return
        existing = [c.name for c in client.get_collections().collections]
        if MEMORY_COLLECTION not in existing:
            dim = len(embeddings.embed_query("维度探测"))
            client.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._collection_ready = True

    def add_episodic(self, question: str, answer: str, kb_name: str) -> None:
        if not self.rag.qdrant_url:
            return
        try:
            client = self._client()
            embeddings = self.rag._embeddings()
            if client is None:
                return
            self._ensure_memory_collection(client, embeddings)
            text = f"问：{question}\n答：{answer[:600]}"
            vec = embeddings.embed_documents([text])[0]
            client.upsert(
                collection_name=MEMORY_COLLECTION,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={
                            "memory_type": "episodic",
                            "question": question,
                            "answer": answer[:600],
                            "kb": kb_name,
                            "ts": time.time(),
                        },
                    )
                ],
            )
        except Exception:  # noqa: BLE001
            # 情景记忆可选，失败不影响主流程
            pass

    def retrieve_episodic(self, question: str, kb_name: str, k: int = 3) -> List[Dict[str, str]]:
        if not self.rag.qdrant_url:
            return []
        try:
            client = self._client()
            embeddings = self.rag._embeddings()
            if client is None:
                return []
            self._ensure_memory_collection(client, embeddings)
            vec = embeddings.embed_query(question)
            qfilter = Filter(must=[FieldCondition(key="kb", match=MatchValue(value=kb_name))])
            res = client.query_points(
                collection_name=MEMORY_COLLECTION,
                query=vec,
                query_filter=qfilter,
                limit=k,
                with_payload=True,
            )
            out: List[Dict[str, str]] = []
            for p in res.points:
                payload = p.payload or {}
                out.append(
                    {
                        "question": payload.get("question", ""),
                        "answer": payload.get("answer", ""),
                    }
                )
            return out
        except Exception:  # noqa: BLE001
            return []
