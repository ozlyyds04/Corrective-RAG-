"""FastAPI 后端冒烟测试（不触发真实入库/网络请求）。"""

import os
import uuid

from fastapi.testclient import TestClient

from corrective_rag.api import app


def make_client():
    return TestClient(app)


def test_health():
    with make_client() as client:
        res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "configured" in body


def test_config():
    with make_client() as client:
        res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert "llm_model" in body
    assert "embedding_model" in body
    assert "qdrant_url" in body


def test_list_kbs_has_default():
    with make_client() as client:
        res = client.get("/api/kbs")
    assert res.status_code == 200
    assert "默认" in res.json()["kbs"]


def test_create_and_delete_kb(monkeypatch):
    # 避免真实删除触发云端 Qdrant 网络调用
    monkeypatch.setenv("QDRANT_URL", "")
    name = f"测试库-{uuid.uuid4().hex[:6]}"
    with make_client() as client:
        res = client.post("/api/kbs", json={"name": name})
    assert res.status_code == 200
    assert name in res.json()["kbs"]

    with make_client() as client:
        res = client.delete(f"/api/kbs/{name}")
    assert res.status_code == 200
    assert name not in res.json()["kbs"]


def test_default_kb_cannot_be_deleted():
    with make_client() as client:
        res = client.delete("/api/kbs/默认")
    assert res.status_code == 400


def test_query_without_retriever_returns_400(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "")
    with make_client() as client:
        res = client.post("/api/query", json={"question": "你好", "kb_name": "默认"})
    assert res.status_code == 400


def test_query_stream_without_retriever_reports_error(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "")
    with make_client() as client:
        res = client.post("/api/query/stream", json={"question": "你好", "kb_name": "默认"})
    assert res.status_code == 200
    assert "请先入库文档" in res.text
