"""jobs 停止事件与 leader 选举的测试（无需 Redis，用 mock）。"""

import os
from unittest.mock import MagicMock

import redis

from corrective_rag import jobs


def test_stop_watcher_only_sets_watcher_event():
    jobs._watcher_stop.clear()
    jobs._worker_stop.clear()
    jobs.stop_watcher()
    assert jobs._watcher_stop.is_set()
    assert not jobs._worker_stop.is_set()


def test_ensure_leader_acquires_when_key_free():
    r = MagicMock()
    r.set.return_value = True
    assert jobs._ensure_leader(r, "rag:worker:leader") is True
    r.set.assert_called_once()


def test_ensure_leader_renews_when_still_owner():
    r = MagicMock()
    r.set.return_value = None  # nx 未命中，key 已被占用
    r.get.return_value = jobs._INSTANCE_ID
    assert jobs._ensure_leader(r, "rag:worker:leader") is True
    r.expire.assert_called_once()


def test_ensure_leader_rejects_when_other_instance_holds():
    r = MagicMock()
    r.set.return_value = None
    r.get.return_value = b"other-instance"
    assert jobs._ensure_leader(r, "rag:worker:leader") is False
    r.expire.assert_not_called()


def test_ensure_leader_degrades_when_redis_unavailable():
    r = MagicMock()
    r.set.side_effect = redis.RedisError("down")
    assert jobs._ensure_leader(r, "rag:worker:leader") is True


def test_leader_identity_is_not_pid():
    # 锁值不能是 PID：不同容器有独立 PID 命名空间，PID 碰撞会让两个实例同时自认 leader
    assert jobs._INSTANCE_ID != str(os.getpid()).encode()
