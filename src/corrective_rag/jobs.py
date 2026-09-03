"""增量更新：文件监听 + Redis 任务队列 + 后台工人。

监听目录新出现的受支持文档 → 推入 Redis 队列 → worker 消费并向量化入库 → 通过
Redis Pub/Sub 发布进度事件，供 /api/watch/stream 以 SSE 方式推送给前端。
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

import redis

from corrective_rag.core import SUPPORTED_EXTENSIONS

QUEUE_KEY = "rag:ingest:queue"
EVENT_CHANNEL = "rag:ingest:events"
POLL_INTERVAL = 3.0

_lock = threading.Lock()
_worker: Optional[threading.Thread] = None
_watcher: Optional[threading.Thread] = None
_stop_event = threading.Event()
_watch_state: Dict[str, Any] = {"running": False, "directory": "", "kb_name": ""}


def _publish(r: redis.Redis, data: Dict[str, Any]) -> None:
    try:
        r.publish(EVENT_CHANNEL, json.dumps(data, ensure_ascii=False))
    except redis.RedisError:
        pass


def _rmq(redis_url: str) -> redis.Redis:
    return redis.Redis.from_url(redis_url, decode_responses=False, socket_timeout=10)


def _last_modified(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _worker_loop(redis_url: str, kb_name: str) -> None:
    """消费队列：把文件路径向量化入库，并发布进度事件。"""
    r = _rmq(redis_url)
    while not _stop_event.is_set():
        try:
            item = r.blpop(QUEUE_KEY, timeout=POLL_INTERVAL)
        except redis.RedisError:
            time.sleep(POLL_INTERVAL)
            continue
        if item is None:
            continue
        path = item[1].decode("utf-8", errors="replace")
        # 延迟导入，避免与 api 模块循环依赖
        from corrective_rag.api import _get_rag  # noqa: PLC0415

        try:
            _publish(r, {"type": "file_started", "path": path, "kb_name": kb_name})
            rag = _get_rag()
            chunk_count, errors = rag.ingest_stream([path], kb_name=kb_name)
            if errors:
                _publish(
                    r,
                    {
                        "type": "file_error",
                        "path": path,
                        "kb_name": kb_name,
                        "error": errors[0][1],
                    },
                )
            else:
                _publish(
                    r,
                    {
                        "type": "file_done",
                        "path": path,
                        "kb_name": kb_name,
                        "chunks": chunk_count,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            _publish(r, {"type": "file_error", "path": path, "kb_name": kb_name, "error": str(exc)})


def _watcher_loop(redis_url: str, directory: str, kb_name: str) -> None:
    """轮询目录，发现新增/修改文档后入队。"""
    r = _rmq(redis_url)
    snapshot: Dict[str, float] = {}
    while not _stop_event.is_set():
        try:
            current: Dict[str, float] = {}
            for root, _dirs, files in os.walk(directory):
                for name in files:
                    if os.path.splitext(name)[1].lower() not in SUPPORTED_EXTENSIONS:
                        continue
                    full = os.path.join(root, name)
                    current[full] = _last_modified(full)

            for path, mtime in current.items():
                old = snapshot.get(path)
                if old is None or mtime > old:
                    r.rpush(QUEUE_KEY, path)
                    _publish(
                        r,
                        {
                            "type": "file_enqueued",
                            "path": path,
                            "kb_name": kb_name,
                        },
                    )
            snapshot = current
        except Exception:  # noqa: BLE001
            pass
        time.sleep(POLL_INTERVAL)


def start_worker(redis_url: str, kb_name: str = "默认") -> None:
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(
            target=_worker_loop, args=(redis_url, kb_name), daemon=True
        )
        _worker.start()


def start_watcher(redis_url: str, directory: str, kb_name: str = "默认") -> None:
    global _watcher
    with _lock:
        _watch_state.update(running=True, directory=directory, kb_name=kb_name)
        if _watcher is not None and _watcher.is_alive():
            return
        _watcher = threading.Thread(
            target=_watcher_loop, args=(redis_url, directory, kb_name), daemon=True
        )
        _watcher.start()


def stop_watcher() -> None:
    with _lock:
        _watch_state["running"] = False


def stop_all() -> None:
    _stop_event.set()
    for t in (_worker, _watcher):
        if t is not None and t.is_alive():
            t.join(timeout=2)


def watch_status(redis_url: str = "") -> Dict[str, Any]:
    queue_size = 0
    if redis_url:
        try:
            queue_size = _rmq(redis_url).llen(QUEUE_KEY)
        except redis.RedisError:
            queue_size = -1
    return {**_watch_state, "queue_size": queue_size}


async def watch_events(redis_url: str) -> Iterator[Dict[str, Any]]:
    """订阅事件频道，逐条产出增量入库进度（异步生成器）。"""
    client = redis.asyncio.Redis.from_url(redis_url)
    pubsub = client.pubsub()
    await pubsub.subscribe(EVENT_CHANNEL)
    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                data = json.loads(message["data"])
            except (TypeError, ValueError):
                continue
            yield data
    finally:
        await pubsub.unsubscribe(EVENT_CHANNEL)
        await client.aclose()
