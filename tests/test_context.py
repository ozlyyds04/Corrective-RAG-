"""上下文压缩（token 预算 + 择优保留）测试。"""

from corrective_rag.core import CorrectiveRAG, Document, _count_tokens


def test_prepare_context_keeps_highest_score_and_truncates():
    rag = CorrectiveRAG(openai_api_key="x")
    rag.max_context_tokens = 500
    big = Document(page_content="这是一个很长很长的内容。" * 200, metadata={"rerank_score": 9.0})
    low = Document(page_content="低分内容，不应被选中", metadata={"rerank_score": 1.0})

    out = rag.prepare_context([big, low])

    assert len(out) == 1
    assert out[0].metadata["rerank_score"] == 9.0
    assert _count_tokens(out[0].page_content) <= 500


def test_prepare_context_keeps_all_when_within_budget():
    rag = CorrectiveRAG(openai_api_key="x")
    rag.max_context_tokens = 2000
    docs = [
        Document(page_content="短片段1", metadata={"rerank_score": 8.0}),
        Document(page_content="短片段2", metadata={"rerank_score": 7.0}),
    ]

    out = rag.prepare_context(docs)

    assert len(out) == 2


def test_prepare_context_no_per_chunk_truncation_within_budget():
    # 回归测试：总量在预算内时，不应按"单段配额"再截一刀丢掉最相关内容
    rag = CorrectiveRAG(openai_api_key="x")
    rag.max_context_tokens = 2600
    docs = [
        Document(page_content="甲" * 1500, metadata={"rerank_score": 9.0}),
        Document(page_content="乙" * 500, metadata={"rerank_score": 8.0}),
        Document(page_content="丙" * 500, metadata={"rerank_score": 7.0}),
        Document(page_content="丁" * 100, metadata={"rerank_score": 6.0}),
    ]

    out = rag.prepare_context(docs)

    assert [d.page_content for d in out] == [d.page_content for d in docs]


def test_prepare_context_orders_by_score():
    rag = CorrectiveRAG(openai_api_key="x")
    rag.max_context_tokens = 2000
    docs = [
        Document(page_content="低分", metadata={"rerank_score": 1.0}),
        Document(page_content="高分", metadata={"rerank_score": 9.0}),
    ]

    out = rag.prepare_context(docs)

    assert out[0].page_content == "高分"
    assert out[1].page_content == "低分"


def test_get_retriever_miss_cooldown():
    # 构建失败后的冷却期内不再重复尝试（避免反复新建 Qdrant 连接）
    rag = CorrectiveRAG(openai_api_key="x", qdrant_url="http://qdrant.invalid")
    rag._retriever_misses["默认"] = float("inf")
    assert rag._get_retriever("默认") is None
