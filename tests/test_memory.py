"""记忆模块测试（无需网络/Redis）。"""

from corrective_rag.memory import MemoryStore


def test_build_dialogue_formats_roles():
    history = [
        {"role": "user", "content": "什么是 RAG？"},
        {"role": "assistant", "content": "RAG 是检索增强生成。"},
    ]
    text = MemoryStore.build_dialogue(history)
    assert "用户：什么是 RAG？" in text
    assert "助手：RAG 是检索增强生成。" in text


def test_build_dialogue_truncates():
    history = [{"role": "user", "content": "A" * 2000}]
    text = MemoryStore.build_dialogue(history)
    assert len(text) <= 1800


def test_empty_redis_returns_empty_history():
    store = MemoryStore(rag=None, redis_url="")
    assert store.get_history("any") == []
    assert store.build_dialogue([]) == ""
