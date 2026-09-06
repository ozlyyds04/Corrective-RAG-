"""增量更新：文件监听 + Redis 任务队列 + 后台工人。

监听目录新出现的受支持文档 → 推入 Redis 队列 → worker 消费并向量化入库 → 通过
Redis Pub/Sub 发布进度事件，供 /api/watch/stream 以 SSE 方式推送给前端。
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import redis

from corrective_rag.core import SUPPORTED_EXTENSIONS

QUEUE_KEY = "rag:ingest:queue"
EVENT_CHANNEL = "rag:ingest:events"
POLL_INTERVAL = 3.0
LEADER_TTL = 30
# 已入队记录（路径 -> mtime 字符串）：跨重启持久化，区分"真的新文件"和"已入库文件"
SEEN_KEY = "rag:watch:seen"
# 锁值用进程内唯一的实例 ID 而非 PID：不同容器有独立的 PID 命名空间，PID 会碰撞，
# 两个实例可能同时通过 r.get(key) == 锁值 的自检，锁就失效了
_INSTANCE_ID = uuid.uuid4().hex.encode()

_lock = threading.Lock()
_worker: Optional[threading.Thread] = None
_watcher: Optional[threading.Thread] = None
_worker_stop = threading.Event()
_watcher_stop = threading.Event()
_watch_state: Dict[str, Any] = {"running": False, "directory": "", "kb_name": ""}


def _publish(r: redis.Redis, data: Dict[str, Any]) -> None:
    try:
        r.publish(EVENT_CHANNEL, json.dumps(data, ensure_ascii=False))
    except redis.RedisError:
        pass


def _rmq(redis_url: str) -> redis.Redis:
    return redis.Redis.from_url(redis_url, decode_responses=False, socket_timeout=10)


def _ensure_leader(r: redis.Redis, key: str) -> bool:
    """让同一集群里只有一个进程持有锁（leader），避免多 worker 重复消费/监听。"""
    try:
        if r.set(key, _INSTANCE_ID, nx=True, ex=LEADER_TTL):
            return True
        if r.get(key) == _INSTANCE_ID:
            r.expire(key, LEADER_TTL)
            return True
        return False
    except redis.RedisError:
        # Redis 不可用时降级为允许运行，保证单机模式可用
        return True


@contextmanager
def _leader_lease(r: redis.Redis, key: str):
    """持锁期间后台续期。

    TTL 只在循环顶部续期的话，单个长任务（如大文件分批向量化）期间锁就会过期，
    另一个实例会拿到 leadership 并发消费队列。这里在任务执行期间启动心跳线程
    定期续期；若已失去 leadership（锁值被别人覆盖）则不再续。
    """
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(max(1.0, LEADER_TTL / 3)):
            try:
                if r.get(key) == _INSTANCE_ID:
                    r.expire(key, LEADER_TTL)
            except redis.RedisError:
                pass

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def _last_modified(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _worker_loop(redis_url: str, kb_name: str) -> None:
    """消费队列：把文件路径向量化入库，并发布进度事件。"""
    r = _rmq(redis_url)
    while not _worker_stop.is_set():
        if not _ensure_leader(r, "rag:worker:leader"):
            _worker_stop.wait(POLL_INTERVAL)
            continue
        try:
            item = r.blpop(QUEUE_KEY, timeout=POLL_INTERVAL)
        except redis.RedisError:
            _worker_stop.wait(POLL_INTERVAL)
            continue
        if item is None:
            continue
        path = item[1].decode("utf-8", errors="replace")
        # 延迟导入，避免与 api 模块循环依赖
        from corrective_rag.api import _get_rag  # noqa: PLC0415

        try:
            _publish(r, {"type": "file_started", "path": path, "kb_name": kb_name})
            with _leader_lease(r, "rag:worker:leader"):
                rag = _get_rag()
                chunk_count, errors = rag.ingest_stream(
                    [path], kb_name=kb_name, clear_existing=False
                )
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


def _load_seen(r: redis.Redis) -> Dict[str, str]:
    """读取已入队记录（路径 -> mtime 字符串）。"""
    try:
        raw = r.hgetall(SEEN_KEY)
    except redis.RedisError:
        return {}
    return {
        (k.decode() if isinstance(k, bytes) else str(k)): (
            v.decode() if isinstance(v, bytes) else str(v)
        )
        for k, v in raw.items()
    }


def _watcher_loop(redis_url: str, directory: str, kb_name: str) -> None:
    """轮询目录，发现新增/修改文档后入队。

    已入队记录持久化在 Redis（路径 -> mtime）：重启后只对新增或变更过的文件入队，
    首次见到的文件（含监听启动前的历史文件、停机期间新增的文件）也会正常入库。
    """
    r = _rmq(redis_url)
    while not _watcher_stop.is_set():
        if not _ensure_leader(r, "rag:watcher:leader"):
            _watcher_stop.wait(POLL_INTERVAL)
            continue
        try:
            with _leader_lease(r, "rag:watcher:leader"):
                seen = _load_seen(r)
                changed: Dict[str, str] = {}
                for root, _dirs, files in os.walk(directory):
                    for name in files:
                        if os.path.splitext(name)[1].lower() not in SUPPORTED_EXTENSIONS:
                            continue
                        full = os.path.join(root, name)
                        stamp = f"{_last_modified(full):.3f}"
                        if seen.get(full) == stamp:
                            continue
                        r.rpush(QUEUE_KEY, full)
                        _publish(
                            r,
                            {
                                "type": "file_enqueued",
                                "path": full,
                                "kb_name": kb_name,
                            },
                        )
                        seen[full] = stamp
                        changed[full] = stamp
                if changed:
                    r.hset(SEEN_KEY, mapping=changed)
        except Exception:  # noqa: BLE001
            pass
        _watcher_stop.wait(POLL_INTERVAL)


def start_worker(redis_url: str, kb_name: str = "默认") -> None:
    global _worker
    with _lock:
        _worker_stop.clear()
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(
            target=_worker_loop, args=(redis_url, kb_name), daemon=True
        )
        _worker.start()


def start_watcher(redis_url: str, directory: str, kb_name: str = "默认") -> None:
    global _watcher
    with _lock:
        _watcher_stop.clear()
        _watch_state.update(running=True, directory=directory, kb_name=kb_name)
        if _watcher is not None and _watcher.is_alive():
            return
        _watcher = threading.Thread(
            target=_watcher_loop, args=(redis_url, directory, kb_name), daemon=True
        )
        _watcher.start()


def stop_watcher() -> None:
    with _lock:
        _watcher_stop.set()
        _watch_state["running"] = False


def stop_all() -> None:
    _worker_stop.set()
    _watcher_stop.set()
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
