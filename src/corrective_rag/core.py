"""纠正式 RAG 核心逻辑（不依赖 Streamlit，便于测试与离线评估）。

包含：文档加载器、Qdrant 入库、LangGraph 纠错工作流、基线流程。
"""

from __future__ import annotations

import json
import os
import re
import uuid
import zlib
from typing import Any, Dict, Iterator, List, Optional, TypedDict
from urllib.parse import urlparse

import httpx
import jieba
import openai
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tavily import TavilyClient
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".html", ".csv"}
DEFAULT_COLLECTION = "rag-qdrant"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TOP_K = 5
SPARSE_FIELD = "sparse"
DENSE_FIELD = "dense"


def tokenize_for_sparse(text: str) -> List[str]:
    """分词：中文用 jieba，英文/数字按词拆分，用于构建稀疏向量。"""
    tokens: List[str] = []
    for token in jieba.cut(text):
        token = token.strip().lower()
        if not token:
            continue
        if re.fullmatch(r"[a-z0-9_.\-+]+", token):
            tokens.append(token)
        else:
            tokens.extend(re.findall(r"[\u4e00-\u9fff]+", token))
    return tokens


def build_sparse_vector(text: str) -> SparseVector:
    """把文本转成词频稀疏向量（Qdrant sparse vector，索引为 token 的稳定哈希）。"""
    counter: Dict[int, int] = {}
    for token in tokenize_for_sparse(text):
        index = zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF
        counter[index] = counter.get(index, 0) + 1
    indices = sorted(counter.keys())
    values = [float(counter[i]) for i in indices]
    return SparseVector(indices=indices, values=values)


def _is_retryable(exc: BaseException) -> bool:
    """连接中断/超时类错误可重试；参数或鉴权类错误不重试。"""
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError, OSError)):
        return True
    return isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError))


_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception(_is_retryable),
)


@_RETRY_DECORATOR
def _embed_query_with_retry(embeddings, text: str) -> List[float]:
    return embeddings.embed_query(text)


@_RETRY_DECORATOR
def _embed_documents_with_retry(embeddings, texts: List[str]) -> List[List[float]]:
    return embeddings.embed_documents(texts)


@_RETRY_DECORATOR
def _upsert_with_retry(client, collection_name: str, points) -> None:
    client.upsert(collection_name=collection_name, points=points)


@_RETRY_DECORATOR
def _delete_points_with_retry(client, collection_name: str, kb_name: str) -> None:
    query_filter = Filter(
        must=[FieldCondition(key="kb", match=MatchValue(value=kb_name))]
    )
    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(filter=query_filter),
    )


@_RETRY_DECORATOR
def _search_with_retry(
    client,
    collection_name: str,
    query,
    top_k: int,
    query_filter: Optional[Filter] = None,
    prefetch: Optional[List[Prefetch]] = None,
):
    return client.query_points(
        collection_name=collection_name,
        query=query,
        prefetch=prefetch or None,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    )


def load_documents(file_or_url: str, is_url: bool = True) -> List[Document]:
    """加载单个文档来源（URL 或本地文件），失败时抛出异常。"""
    if is_url:
        # .pdf 链接必须按 PDF 解析；WebBaseLoader 会把 HTML 解析器跑在二进制内容上，
        # 把解码后的乱码数据写入向量库。
        if urlparse(file_or_url).path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_or_url)
        else:
            loader = WebBaseLoader(file_or_url)
            loader.requests_per_second = 1
    else:
        file_extension = os.path.splitext(file_or_url)[1].lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_or_url)
        elif file_extension in (".txt", ".md"):
            # 优先按 UTF-8 解码，失败时回退到 GBK，兼容常见中文文本文件
            for encoding in ("utf-8", "gbk"):
                try:
                    with open(file_or_url, encoding=encoding) as f:
                        text = f.read()
                    return [Document(page_content=text, metadata={"source": file_or_url})]
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Cannot decode text file: {file_or_url}")
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_or_url)
        elif file_extension == ".html":
            # 优先按 UTF-8 解析，失败时回退到 GBK
            for encoding in ("utf-8", "gbk"):
                try:
                    return BSHTMLLoader(
                        file_or_url,
                        open_encoding=encoding,
                        bs_kwargs={"features": "html.parser"},
                    ).load()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Cannot decode html file: {file_or_url}")
        elif file_extension == ".csv":
            loader = CSVLoader(file_or_url)
        else:
            raise ValueError(f"不支持的文档类型：{file_extension}")

    return loader.load()


def load_local_path(path: str) -> List[Document]:
    """加载本地文件或文件夹中的全部受支持文档。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在：{path}")

    if os.path.isfile(path):
        return load_documents(path, is_url=False)

    # 递归遍历文件夹，收集所有受支持格式的文档
    all_docs: List[Document] = []
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                all_docs.extend(load_documents(os.path.join(root, name), is_url=False))

    if not all_docs:
        raise ValueError("文件夹中没有找到受支持的文档。")
    return all_docs


def expand_document_sources(sources: List[str]) -> List[str]:
    """把文件夹展开成文件列表；URL / 单文件原样保留。"""
    expanded: List[str] = []
    for src in sources:
        if src.startswith(("http://", "https://")):
            expanded.append(src)
        elif os.path.isfile(src):
            expanded.append(src)
        elif os.path.isdir(src):
            found = False
            for root, _dirs, files in os.walk(src):
                for name in sorted(files):
                    if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                        expanded.append(os.path.join(root, name))
                        found = True
            if not found:
                raise ValueError(f"文件夹中没有找到受支持的文档：{src}")
        else:
            raise FileNotFoundError(f"路径不存在：{src}")
    return expanded


def load_knowledge_bases(path: str) -> List[str]:
    """读取知识库名称列表（保证至少包含一个默认库）。"""
    names: List[str] = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            names = [n for n in data if isinstance(n, str) and n.strip()]
        except Exception:
            names = []
    if "默认" not in names:
        names = ["默认"] + names
    return names


def save_knowledge_bases(path: str, names: List[str]) -> None:
    """保存知识库名称列表。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def execute_tavily_search(client, query: str) -> Dict[str, Any]:
    """带重试逻辑的 Tavily 搜索。"""
    return client.search(query=query, max_results=TOP_K, search_depth="advanced")


class GraphState(TypedDict):
    keys: Dict[str, Any]


class QdrantRetriever:
    """基于 qdrant-client 新 API 的轻量检索器（不依赖已弃用的 langchain_qdrant）。"""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings,
        kb_name: str = "默认",
        top_k: int = TOP_K,
        reranker=None,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.kb_name = kb_name
        self.top_k = top_k
        self.reranker = reranker

    def invoke(self, query: str) -> List[Document]:
        # 多路召回：稠密向量 + 关键词稀疏向量，RRF 融合
        dense_vector = self.embeddings.embed_query(query)
        sparse_vector = build_sparse_vector(query)
        kb_filter = Filter(
            must=[FieldCondition(key="kb", match=MatchValue(value=self.kb_name))]
        )
        response = _search_with_retry(
            self.client,
            self.collection_name,
            FusionQuery(fusion=Fusion.RRF),
            self.top_k * 2 + 2,
            query_filter=kb_filter,
            prefetch=[
                Prefetch(query=dense_vector, using=DENSE_FIELD, limit=self.top_k * 2),
                Prefetch(query=sparse_vector, using=SPARSE_FIELD, limit=self.top_k * 2),
            ],
        )
        documents: List[Document] = []
        for point in response.points:
            payload = point.payload or {}
            documents.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}) or {},
                )
            )
        # 检索重排：用 LLM 对候选重新打分排序，取 Top-K
        if self.reranker is not None and documents:
            documents = self.reranker(query, documents)
        return documents[: self.top_k]


class CorrectiveRAG:
    """纠正式 RAG：检索 → 相关性评分 → 查询改写 → 网络搜索兜底 → 生成。"""

    def __init__(
        self,
        openai_api_key: str = "",
        tavily_api_key: str = "",
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        openai_base_url: str = "",
        model: str = "deepseek-v4-flash",
        embedding_model: str = "qwen3.7-text-embedding",
        embedding_api_key: str = "",
        embedding_base_url: str = "",
        embedding_dim: Optional[int] = None,
        retriever=None,
    ) -> None:
        self.openai_api_key = openai_api_key
        self.tavily_api_key = tavily_api_key
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.openai_base_url = openai_base_url
        self.model = model
        self.embedding_model = embedding_model
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        # 向量维度：None 表示入库时自动探测
        self.embedding_dim = embedding_dim
        self.retriever = retriever
        # 可选的状态回调，供 UI 展示进度（例如 Streamlit 的占位提示）
        self.status_callback: Optional[callable] = None

    # ---------- 基础设施 ----------

    def _status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message)

    def _llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model,
            api_key=self.openai_api_key,
            base_url=self.openai_base_url or None,
            temperature=0,
            max_tokens=1000,
        )

    def _embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=self.embedding_api_key or self.openai_api_key,
            base_url=self.embedding_base_url or self.openai_base_url or None,
            # DashScope 兼容接口不接受 token 数组，只接受原始文本；
            # 关闭长度安全 token 化路径，直接发送字符串（切块后文本足够短）。
            check_embedding_ctx_length=False,
            # DashScope 单次请求最多 20 条，分批发送
            chunk_size=20,
        )

    def _tavily_search(self, query: str) -> List[Document]:
        """执行 Tavily 网络搜索并包装为文档；未配置密钥时返回空列表。"""
        if not self.tavily_api_key:
            return []
        client = TavilyClient(api_key=self.tavily_api_key)
        results = execute_tavily_search(client, query)
        items = results.get("results", []) if isinstance(results, dict) else results
        # 每条结果单独作为一个文档，便于后续逐条过滤
        return [
            Document(
                page_content=(
                    f"标题：{item.get('title', '无标题')}\n"
                    f"内容：{item.get('content', '无内容')}\n"
                ),
                metadata={
                    "source": "tavily_search",
                    "query": query,
                    "url": item.get("url", ""),
                },
            )
            for item in items
        ]

    # ---------- 文档入库 ----------

    def _ensure_collection(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings,
    ) -> None:
        """确保集合存在且包含稠密+稀疏双向量配置；旧结构集合自动重建。"""
        existing = [c.name for c in client.get_collections().collections]
        needs_create = collection_name not in existing
        if not needs_create:
            params = client.get_collection(collection_name).config.params
            dense_ok = isinstance(params.vectors, dict) and DENSE_FIELD in params.vectors
            sparse_ok = bool(
                getattr(params, "sparse_vectors", None)
                and SPARSE_FIELD in params.sparse_vectors
            )
            if not (dense_ok and sparse_ok):
                client.delete_collection(collection_name)
                needs_create = True
        if needs_create:
            dimension = self.embedding_dim
            if dimension is None:
                dimension = len(_embed_query_with_retry(embeddings, "向量维度探测"))
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    DENSE_FIELD: VectorParams(
                        size=dimension, distance=Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    SPARSE_FIELD: SparseVectorParams(index=SparseIndexParams()),
                },
            )
        # 过滤字段需要索引（keyword 类型），保证按知识库过滤的检索/删除可用
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="kb",
                field_schema="keyword",
            )
        except Exception:
            pass

    def ingest(
        self,
        docs: List[Document],
        collection_name: str = DEFAULT_COLLECTION,
        kb_name: str = "默认",
    ) -> None:
        """切块、写入 Qdrant，并设置检索器。"""
        if not docs:
            raise ValueError("没有可入库的文档。")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        all_splits = text_splitter.split_documents(docs)

        client = QdrantClient(
            url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=120
        )
        embeddings = self._embeddings()
        self._ensure_collection(client, collection_name, embeddings)
        # 只清理当前知识库的数据，保留其他知识库
        _delete_points_with_retry(client, collection_name, kb_name)

        # 直接写入 Qdrant（langchain_qdrant 已弃用，且与新版 qdrant-client 不兼容）
        # DashScope 单次请求最多 20 条，这里显式分批，避免依赖 langchain 内部实现
        vectors: List[List[float]] = []
        batch: List[str] = []
        for chunk in all_splits:
            batch.append(chunk.page_content)
            if len(batch) >= 20:
                vectors.extend(_embed_documents_with_retry(embeddings, batch))
                batch = []
        if batch:
            vectors.extend(_embed_documents_with_retry(embeddings, batch))
        if len(vectors) != len(all_splits):
            raise ValueError("向量数量与文档片段数量不一致，请检查 embedding 服务。")

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_FIELD: vector,
                    SPARSE_FIELD: build_sparse_vector(chunk.page_content),
                },
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                    "kb": kb_name,
                },
            )
            for chunk, vector in zip(all_splits, vectors)
        ]
        _upsert_with_retry(client, collection_name, points)
        self.retriever = QdrantRetriever(
            client, collection_name, embeddings, kb_name=kb_name, reranker=self._rerank
        )

    def ingest_stream(
        self,
        sources: List[str],
        kb_name: str = "默认",
        collection_name: str = DEFAULT_COLLECTION,
        progress_callback=None,
        batch_size: int = 20,
    ) -> tuple[int, List[tuple[str, str]]]:
        """边加载边分批入库（每批最多 batch_size 条向量化后立即写入）。

        kb_name 用于区分知识库：数据统一存在一个集合里，通过 payload 标签过滤。
        progress_callback(processed_files, total_files, chunk_count) 用于展示进度。
        返回（向量化片段总数，[(来源, 错误信息), ...]）。
        """
        expanded = expand_document_sources(sources)
        if not expanded:
            raise ValueError("没有可入库的文档来源。")
        total = len(expanded)

        client = QdrantClient(
            url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=120
        )
        embeddings = self._embeddings()
        self._ensure_collection(client, collection_name, embeddings)

        # 只清理当前知识库的数据，保留其他知识库
        _delete_points_with_retry(client, collection_name, kb_name)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        errors: List[tuple[str, str]] = []
        chunk_batch: List[Document] = []
        processed_files = 0
        chunk_count = 0

        def flush() -> None:
            nonlocal chunk_count
            if not chunk_batch:
                return
            texts = [c.page_content for c in chunk_batch]
            vectors = _embed_documents_with_retry(embeddings, texts)
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        DENSE_FIELD: vector,
                        SPARSE_FIELD: build_sparse_vector(c.page_content),
                    },
                    payload={
                        "page_content": c.page_content,
                        "metadata": c.metadata,
                        "kb": kb_name,
                    },
                )
                for c, vector in zip(chunk_batch, vectors)
            ]
            _upsert_with_retry(client, collection_name, points)
            chunk_count += len(chunk_batch)
            chunk_batch.clear()

        for src in expanded:
            try:
                docs = load_documents(src, is_url=src.startswith(("http://", "https://")))
                for doc in docs:
                    chunk_batch.extend(text_splitter.split_documents([doc]))
                    if len(chunk_batch) >= batch_size:
                        flush()
            except Exception as e:  # noqa: BLE001
                errors.append((src, str(e)))
            finally:
                processed_files += 1
                if progress_callback is not None:
                    progress_callback(processed_files, total, chunk_count)
        flush()
        # 最后一批写入完成后，再回调一次，保证进度显示准确的片段总数
        if progress_callback is not None:
            progress_callback(total, total, chunk_count)

        self.retriever = QdrantRetriever(
            client, collection_name, embeddings, kb_name=kb_name, reranker=self._rerank
        )
        return chunk_count, errors

    def delete_knowledge_base(
        self, kb_name: str, collection_name: str = DEFAULT_COLLECTION
    ) -> None:
        """删除指定知识库的全部数据（其他知识库不受影响）。"""
        client = QdrantClient(
            url=self.qdrant_url, api_key=self.qdrant_api_key, timeout=120
        )
        existing = [c.name for c in client.get_collections().collections]
        if collection_name in existing:
            _delete_points_with_retry(client, collection_name, kb_name)

    # ---------- 单步能力 ----------

    def _retrieve(self, question: str) -> List[Document]:
        if self.retriever is None:
            return []
        return self.retriever.invoke(question)

    def _generate(self, question: str, documents: List[Document]) -> str:
        prompt = PromptTemplate(
            template="""You are an assistant that answers questions strictly based on the provided context.
            Rules:
            - Answer ONLY using the information in the context.
            - Do NOT add general knowledge, speculation, examples, or content from outside the context.
            - If the context does not contain enough information to answer the question,
              reply with "根据现有资料无法回答该问题" and do not guess.
            Context: {context}
            Question: {question}
            Answer:""",
            input_variables=["context", "question"],
        )
        llm = self._llm()
        context = "\n\n".join(doc.page_content for doc in documents)
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain.invoke({})

    def _grade_documents(self, question: str, documents: List[Document]) -> tuple[List[Document], str]:
        """逐条判断文档相关性，返回（保留的文档，是否需要网络搜索）。"""
        llm = self._llm()
        prompt = PromptTemplate(
            template="""You are grading the relevance of a retrieved document to a user question.
            Return ONLY a JSON object with a "score" field that is either "yes" or "no".
            Do not include any other text or explanation.

            Document: {context}
            Question: {question}

            Rules:
            - Check for related keywords or semantic meaning
            - Use lenient grading to only filter clear mismatches
            - Return exactly like this example: {{"score": "yes"}} or {{"score": "no"}}""",
            input_variables=["context", "question"],
        )
        chain = prompt | llm | StrOutputParser()

        import json
        import re

        filtered_docs: List[Document] = []
        for d in documents:
            try:
                response = chain.invoke({"question": question, "context": d.page_content})
                json_match = re.search(r"\{.*\}", response)
                if json_match:
                    response = json_match.group()
                score = json.loads(response)
                if score.get("score") == "yes":
                    filtered_docs.append(d)
            except Exception:
                # 出错时保留该文档，确保安全
                filtered_docs.append(d)
        # 本地文档仍有关联内容就不触发网络搜索；全部被过滤或没检索到内容时才兜底
        search = "Yes" if not filtered_docs else "No"
        return filtered_docs, search

    def _rerank(
        self, question: str, documents: List[Document]
    ) -> List[Document]:
        """LLM 重排：让模型对候选片段按相关性打分，按分数降序返回。"""
        if not documents:
            return documents
        llm = self._llm()
        numbered = "\n\n".join(
            f"[{index}] {doc.page_content[:300]}" for index, doc in enumerate(documents, start=1)
        )
        prompt = f"""根据与问题的相关性，给每个候选片段打分（0 到 10 分）。
        只输出 JSON，格式为 {{"scores": [数字列表，顺序与候选片段一致]}}，不要输出其他内容。
        问题：{question}
        候选片段：
        {numbered}"""
        try:
            response = llm.invoke(prompt)
            match = re.search(r"\{.*\}", response.content, re.S)
            if not match:
                return documents
            scores = json.loads(match.group()).get("scores", [])
            if len(scores) != len(documents):
                return documents
            ranked = sorted(
                zip(scores, documents), key=lambda item: -float(item[0])
            )
            result: List[Document] = []
            for score, doc in ranked:
                # 把重排分数挂到文档元数据上，供检索可视化展示
                doc.metadata["rerank_score"] = float(score)
                result.append(doc)
            return result
        except Exception:
            return documents

    def _transform_query(self, question: str) -> str:
        prompt = PromptTemplate(
            template="""Generate a search-optimized version of this question by
            analyzing its core semantic meaning and intent.
            \n ------- \n
            {question}
            \n ------- \n
            Return only the improved question with no additional text:""",
            input_variables=["question"],
        )
        llm = self._llm()
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question})

    # ---------- LangGraph 节点 ----------

    def _node_retrieve(self, state: GraphState) -> Dict[str, Any]:
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = self._retrieve(question)
        return {"keys": {"documents": documents, "question": question}}

    def _node_generate(self, state: GraphState) -> Dict[str, Any]:
        state_dict = state["keys"]
        question, documents = state_dict["question"], state_dict["documents"]
        generation = self._generate(question, documents)
        return {
            "keys": {
                "documents": documents,
                "question": question,
                "generation": generation,
            }
        }

    def _node_grade_documents(self, state: GraphState) -> Dict[str, Any]:
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        filtered_docs, search = self._grade_documents(question, documents)
        return {
            "keys": {
                "documents": filtered_docs,
                "question": question,
                "run_web_search": search,
            }
        }

    def _node_transform_query(self, state: GraphState) -> Dict[str, Any]:
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        better_question = self._transform_query(question)
        return {"keys": {"documents": documents, "question": better_question}}

    def _node_web_search(self, state: GraphState) -> Dict[str, Any]:
        state_dict = state["keys"]
        question = state_dict["question"]
        documents = state_dict["documents"]
        self._status("正在发起网络搜索...")
        web_documents = self._tavily_search(question)
        # 网络结果先做相关性过滤，避免主题漂移内容进入生成上下文
        if web_documents:
            relevant_web, _ = self._grade_documents(question, web_documents)
            documents = documents + relevant_web
        return {"keys": {"documents": documents, "question": question}}

    def _decide_to_generate(self, state: GraphState) -> str:
        state_dict = state["keys"]
        search = state_dict.get("run_web_search", "No")
        if search == "Yes":
            return "transform_query"
        return "generate"

    # ---------- 图与执行 ----------

    def build(self):
        """构建并编译纠错工作流图。"""
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self._node_retrieve)
        workflow.add_node("grade_documents", self._node_grade_documents)
        workflow.add_node("generate", self._node_generate)
        workflow.add_node("transform_query", self._node_transform_query)
        workflow.add_node("web_search", self._node_web_search)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {"transform_query": "transform_query", "generate": "generate"},
        )
        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def run(
        self,
        question: str,
        progress_callback=None,
    ) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """运行纠错工作流，返回（各步骤状态，最终状态）。

        progress_callback(node_name, node_index) 用于展示回答进度。
        """
        app = self.build()
        steps: Dict[str, Dict[str, Any]] = {}
        final_state: Dict[str, Any] = {}
        node_index = 0
        for output in app.stream({"keys": {"question": question}}):
            for key, value in output.items():
                node_index += 1
                if progress_callback is not None:
                    progress_callback(key, node_index)
                steps[key] = value["keys"]
                final_state = value["keys"]
        return steps, final_state

    def run_baseline(self, question: str) -> Dict[str, Any]:
        """基线流程：检索后直接生成，不做相关性评分与网络搜索兜底。"""
        documents = self._retrieve(question)
        generation = self._generate(question, documents)
        return {"documents": documents, "question": question, "generation": generation}

    # ---------- 流式生成与证据解析（供 FastAPI 界面复用） ----------

    def _generate_stream(self, question: str, documents: List[Document]) -> Iterator[str]:
        """逐 token 流式生成最终回答（与 _generate 使用相同的提示词）。"""
        prompt = PromptTemplate(
            template="""You are an assistant that answers questions strictly based on the provided context.
            Rules:
            - Answer ONLY using the information in the context.
            - Do NOT add general knowledge, speculation, examples, or content from outside the context.
            - If the context does not contain enough information to answer the question,
              reply with "根据现有资料无法回答该问题" and do not guess.
            Context: {context}
            Question: {question}
            Answer:""",
            input_variables=["context", "question"],
        )
        llm = self._llm()
        context = "\n\n".join(doc.page_content for doc in documents)
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: question}
            | prompt
            | llm
            | StrOutputParser()
        )
        for chunk in rag_chain.stream({}):
            yield chunk

    def run_evidence(
        self, question: str, progress_callback=None
    ) -> tuple[List[Document], Dict[str, Dict[str, Any]]]:
        """执行纠错检索链（检索 → 评分 → 必要时改写查询 + 网络搜索），但不生成回答。

        返回（用于生成最终回答的文档列表，各步骤状态）。生成交给 _generate_stream 流式完成，
        以便前端展示检索到的片段与重排分数。
        """
        steps: Dict[str, Dict[str, Any]] = {}
        documents = self._retrieve(question)
        if progress_callback is not None:
            progress_callback("retrieve", 0)
        steps["retrieve"] = {"documents": documents, "question": question}

        self._status("正在评估检索结果相关性...")
        filtered, search = self._grade_documents(question, documents)
        if progress_callback is not None:
            progress_callback("grade_documents", 1)
        steps["grade_documents"] = {
            "documents": filtered,
            "run_web_search": search,
            "question": question,
        }

        if search == "Yes":
            self._status("检索不足，正在改写查询并搜索网络...")
            better_question = self._transform_query(question)
            if progress_callback is not None:
                progress_callback("transform_query", 2)
            steps["transform_query"] = {
                "question": better_question,
                "documents": filtered,
            }
            web_documents = self._tavily_search(better_question)
            final_documents = filtered
            if web_documents:
                relevant_web, _ = self._grade_documents(better_question, web_documents)
                final_documents = filtered + relevant_web
            if progress_callback is not None:
                progress_callback("web_search", 3)
            steps["web_search"] = {
                "documents": final_documents,
                "question": better_question,
            }
        else:
            if progress_callback is not None:
                progress_callback("generate", 4)
            final_documents = filtered
            steps["generate"] = {
                "documents": final_documents,
                "question": question,
            }
        return final_documents, steps
