"""纠错工作流的路由与执行测试（全部使用假 LLM / 假检索器，不发真实请求）。"""

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from corrective_rag.core import CorrectiveRAG, _is_retryable


def test_retryable_error_classification():
    # 网络中断 / 超时类错误应该重试
    assert _is_retryable(ConnectionAbortedError("WinError 10053"))
    assert _is_retryable(httpx.ConnectError("connection refused"))
    assert _is_retryable(TimeoutError("timed out"))
    # 参数 / 逻辑类错误不应该重试
    assert not _is_retryable(ValueError("batch size is invalid"))


def test_run_reports_step_progress():
    rag = make_rag([make_doc("相关文档")], generation="答案")
    seen = []
    rag.run("问题", progress_callback=lambda name, idx: seen.append((name, idx)))
    assert [name for name, _ in seen] == ["retrieve", "grade_documents", "generate"]
    assert [idx for _, idx in seen] == [1, 2, 3]


def test_rerank_reorders_by_scores():
    rag = CorrectiveRAG(openai_api_key="sk-test")
    rag._llm = lambda: RunnableLambda(
        lambda x: AIMessage(content='{"scores": [1, 10, 5]}')
    )
    docs = [make_doc(f"文档{i}") for i in range(3)]
    out = rag._rerank("问题", docs)
    assert out[0].page_content == "文档1"
    assert out[1].page_content == "文档2"
    assert out[2].page_content == "文档0"


def test_rerank_fallback_on_bad_response():
    rag = CorrectiveRAG(openai_api_key="sk-test")
    rag._llm = lambda: RunnableLambda(
        lambda x: AIMessage(content="这不是 JSON 输出")
    )
    docs = [make_doc("a"), make_doc("b")]
    assert rag._rerank("问题", docs) == docs


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, question):
        return list(self.docs)


class FakeLLM:
    """按提示词内容返回预设响应的假 LLM。"""

    def __init__(self, generation: str = "生成的答案"):
        self.generation = generation

    def __call__(self, prompt_value):
        text = getattr(prompt_value, "to_string", lambda: str(prompt_value))()
        if "grading the relevance" in text:
            if "不相关" in text:
                return AIMessage(content='{"score": "no"}')
            return AIMessage(content='{"score": "yes"}')
        if "search-optimized version" in text:
            return AIMessage(content="改写后的问题")
        return AIMessage(content=self.generation)


def make_rag(docs, generation="生成的答案"):
    rag = CorrectiveRAG(
        openai_api_key="sk-test",
        tavily_api_key="tvly-test",
        retriever=FakeRetriever(docs),
    )
    rag._llm = lambda: RunnableLambda(FakeLLM(generation))
    rag._tavily_search = lambda query: [
        Document(
            page_content="来自网络搜索的内容",
            metadata={"source": "tavily_search", "query": query},
        )
    ]
    return rag


def make_doc(content):
    return Document(page_content=content, metadata={"source": "test"})


def test_all_relevant_routes_directly_to_generate():
    docs = [make_doc("相关文档A"), make_doc("相关文档B")]
    rag = make_rag(docs, generation="完整答案")
    steps, final = rag.run("测试问题")

    assert list(steps.keys()) == ["retrieve", "grade_documents", "generate"]
    assert final["generation"] == "完整答案"
    assert steps["grade_documents"]["run_web_search"] == "No"


def test_irrelevant_document_triggers_rewrite_and_web_search():
    # 本地片段全部被判定为不相关时才触发查询改写 + 网络搜索
    docs = [make_doc("完全不相关的内容A"), make_doc("完全不相关的内容B")]
    rag = make_rag(docs, generation="联网后的答案")
    steps, final = rag.run("测试问题")

    assert list(steps.keys()) == [
        "retrieve",
        "grade_documents",
        "transform_query",
        "web_search",
        "generate",
    ]
    assert final["generation"] == "联网后的答案"
    assert any(d.metadata.get("source") == "tavily_search" for d in final["documents"])


def test_web_results_are_filtered_before_generation():
    docs = [make_doc("完全不相关的内容A")]
    rag = make_rag(docs, generation="过滤后的答案")
    rag._tavily_search = lambda query: [
        Document(
            page_content="与问题相关的网络内容",
            metadata={"source": "tavily_search", "url": "a.com"},
        ),
        Document(
            page_content="完全不相关的网络垃圾内容",
            metadata={"source": "tavily_search", "url": "b.com"},
        ),
    ]
    steps, final = rag.run("测试问题")
    web_docs = [d for d in final["documents"] if d.metadata.get("source") == "tavily_search"]
    assert len(web_docs) == 1
    assert "相关" in web_docs[0].page_content
    assert "垃圾" not in web_docs[0].page_content


def test_grade_documents_filters_irrelevant():
    docs = [make_doc("相关文档"), make_doc("完全不相关的内容")]
    rag = make_rag(docs)
    filtered, search = rag._grade_documents("测试问题", docs)
    assert len(filtered) == 1
    assert "相关文档" in filtered[0].page_content
    # 仍保留至少一个相关文档时，不触发网络搜索
    assert search == "No"


def test_grade_documents_all_filtered_triggers_search():
    docs = [make_doc("完全不相关的内容A"), make_doc("完全不相关的内容B")]
    rag = make_rag(docs)
    filtered, search = rag._grade_documents("测试问题", docs)
    assert filtered == []
    assert search == "Yes"


def test_baseline_keeps_all_documents_and_skips_grading():
    docs = [make_doc("相关文档"), make_doc("完全不相关的内容")]
    rag = make_rag(docs, generation="基线答案")
    result = rag.run_baseline("测试问题")

    assert result["generation"] == "基线答案"
    assert len(result["documents"]) == 2


def test_no_retriever_returns_empty_documents():
    rag = CorrectiveRAG(openai_api_key="sk-test")
    rag._llm = lambda: RunnableLambda(FakeLLM())
    rag._tavily_search = lambda query: []
    steps, final = rag.run("没有入库的问题")
    assert list(steps.keys()) == [
        "retrieve",
        "grade_documents",
        "transform_query",
        "web_search",
        "generate",
    ]
    assert final["documents"] == []


def test_ingest_requires_documents():
    rag = CorrectiveRAG(openai_api_key="sk-test")
    with pytest.raises(ValueError, match="没有可入库"):
        rag.ingest([])
