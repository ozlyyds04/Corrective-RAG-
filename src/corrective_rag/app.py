"""纠正式 RAG Agent 的 Streamlit 界面（核心逻辑见 rag_core.py）。"""

import os
import pprint
import tempfile

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from corrective_rag.core import CorrectiveRAG, load_knowledge_bases, save_knowledge_bases

KB_REGISTRY_PATH = os.getenv(
    "KB_REGISTRY_PATH",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "knowledge_bases.json"
    ),
)


def initialize_session_state():
    """初始化会话状态变量（优先读取 .env）。"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.openai_api_key = os.getenv("LLM_API_KEY", "")
        st.session_state.openai_base_url = os.getenv("LLM_BASE_URL", "")
        st.session_state.model = os.getenv("LLM_MODEL", "deepseek-v4-flash")
        st.session_state.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "qwen3.7-text-embedding"
        )
        st.session_state.embedding_api_key = os.getenv("EMBEDDING_API_KEY", "")
        st.session_state.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "")
        st.session_state.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        st.session_state.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        st.session_state.qdrant_url = os.getenv("QDRANT_URL", "")
        st.session_state.doc_url = os.getenv("DOC_URL", "")
        st.session_state.local_path = ""
        st.session_state.upload_limit = 50
        st.session_state.initialized = True


def get_rag() -> CorrectiveRAG:
    """根据当前会话配置构建 CorrectiveRAG，并保留已入库的检索器。"""
    previous = st.session_state.get("rag")
    rag = CorrectiveRAG(
        openai_api_key=st.session_state.openai_api_key,
        openai_base_url=st.session_state.openai_base_url,
        model=st.session_state.model,
        embedding_model=st.session_state.embedding_model,
        embedding_api_key=st.session_state.embedding_api_key,
        embedding_base_url=st.session_state.embedding_base_url,
        tavily_api_key=st.session_state.tavily_api_key,
        qdrant_url=st.session_state.qdrant_url,
        qdrant_api_key=st.session_state.qdrant_api_key,
        retriever=previous.retriever if previous is not None else None,
    )
    st.session_state.rag = rag
    return rag


def format_document(doc) -> str:
    return f"""
    来源：{doc.metadata.get('source', '未知')}
    标题：{doc.metadata.get('title', '无标题')}
    内容：{doc.page_content[:200]}...
    """


def format_state(state: dict) -> dict:
    formatted = {}
    for key, value in state.items():
        if key == "documents":
            formatted[key] = [format_document(doc) for doc in value]
        else:
            formatted[key] = value
    return formatted


initialize_session_state()

st.subheader("知识库管理")
kb_names = load_knowledge_bases(KB_REGISTRY_PATH)
new_kb_name = st.text_input("新建知识库名称：")
if new_kb_name.strip() and new_kb_name.strip() not in kb_names:
    st.caption("输入新名称后，点击下方「新建知识库」按钮创建。")
if st.button("新建知识库") and new_kb_name.strip():
    name = new_kb_name.strip()
    if name in kb_names:
        st.warning(f"知识库「{name}」已存在。")
    else:
        try:
            kb_names.append(name)
            save_knowledge_bases(KB_REGISTRY_PATH, kb_names)
            st.session_state.current_kb = name
            st.rerun()
        except Exception as e:
            st.error(f"创建知识库失败：{str(e)}")

current_kb_default = st.session_state.get("current_kb", "默认")
current_kb = st.selectbox(
    "当前知识库",
    kb_names,
    index=kb_names.index(current_kb_default)
    if current_kb_default in kb_names
    else 0,
)
st.session_state.current_kb = current_kb

if st.button("删除当前知识库"):
    if current_kb == "默认":
        st.warning("默认知识库不可删除。")
    else:
        try:
            rag = get_rag()
            rag.delete_knowledge_base(current_kb)
        except Exception as e:
            st.error(f"删除知识库失败：{str(e)}")
        else:
            kb_names.remove(current_kb)
            save_knowledge_bases(KB_REGISTRY_PATH, kb_names)
            st.session_state.current_kb = kb_names[0] if kb_names else "默认"
            st.rerun()

st.subheader("文档输入（可多选，合并入库）")
use_url = st.checkbox("URL", value=st.session_state.get("use_url", False))
use_local = st.checkbox(
    "本地路径/文件夹", value=st.session_state.get("use_local", False)
)
use_upload = st.checkbox(
    "文件上传", value=st.session_state.get("use_upload", False)
)

sources = []
tmp_files = []

if use_url:
    url_text = st.text_area(
        "输入 URL（每行一个）：", value=st.session_state.get("url_text", ""), height=100
    )
    st.session_state.url_text = url_text
    sources += [u.strip() for u in url_text.splitlines() if u.strip()]

if use_local:
    local_path = st.text_input(
        "输入本地文件或文件夹路径：", value=st.session_state.local_path
    )
    st.session_state.local_path = local_path
    if local_path.strip():
        sources.append(local_path.strip())

if use_upload:
    st.session_state.upload_limit = st.number_input(
        "单次最多上传文件数",
        min_value=1,
        max_value=500,
        value=st.session_state.upload_limit,
        step=1,
    )
    uploaded_files = st.file_uploader(
        "上传文档",
        type=["pdf", "txt", "md", "docx", "html", "csv"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        if len(uploaded_files) > st.session_state.upload_limit:
            st.warning(
                f"一次最多上传 {st.session_state.upload_limit} 个文件，"
                f"多余的 {len(uploaded_files) - st.session_state.upload_limit} 个文件将被忽略。"
            )
            uploaded_files = uploaded_files[: st.session_state.upload_limit]
        # 写入临时文件，入库完成后清理
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
            tmp_files.append(tmp_file.name)
        sources += tmp_files

# 去重（保留顺序）
sources = list(dict.fromkeys(sources))

if not sources:
    st.info("请至少选择一种输入方式（URL / 本地路径 / 文件上传），可多选合并。")

# 来源标识：任一来源变化都会触发重新入库
source_key = "|".join(sorted(sources)) if sources else None

try:
    if sources and st.session_state.get("ingested_source") != source_key:
        rag = get_rag()
        if not (rag.qdrant_url and (rag.openai_api_key or rag.embedding_api_key)):
            st.info("请先在 .env 中配置 API 密钥和 Qdrant 地址，再进行入库。")
        else:
            try:
                progress = st.progress(0.0, text="准备入库...")

                def on_progress(processed, total, chunks):
                    ratio = processed / total if total else 1.0
                    progress.progress(
                        min(ratio, 1.0),
                        text=f"正在入库：{processed}/{total} 个来源，已向量化 {chunks} 个片段",
                    )

                chunk_count, errors = rag.ingest_stream(
                    sources, kb_name=current_kb, progress_callback=on_progress
                )
                progress.empty()
                st.session_state.ingested_source = source_key
                st.success(f"入库完成：共向量化 {chunk_count} 个片段。")
                for src, err in errors:
                    display = src if src.startswith("http") else os.path.basename(src)
                    st.warning(f"跳过 {display}：{err}")
            except Exception as e:
                st.error(f"文档入库失败：{str(e)}")
finally:
    # 清理临时上传文件
    for tmp_file in tmp_files:
        try:
            os.unlink(tmp_file)
        except OSError:
            pass

st.title("🔄 纠正式 RAG Agent")

st.text("示例问题：这篇论文中的实验结果和消融研究是什么？")

# 用户输入
user_question = st.text_input("请输入你的问题：")

if user_question:
    rag = get_rag()
    if rag.retriever is None:
        st.warning("请先在上方加载并入库文档，再提出问题。")
    else:
        status = st.empty()
        rag.status_callback = lambda msg: status.info(msg)
        answer_progress = st.progress(0.0, text="准备回答...")

        def on_answer_step(name, idx):
            answer_progress.progress(
                min(idx / 5.0, 1.0), text=f"正在执行步骤「{name}」..."
            )

        steps = {}
        final_state = {}
        try:
            steps, final_state = rag.run(
                user_question, progress_callback=on_answer_step
            )
        except Exception as e:
            st.error(f"运行出错：{str(e)}")
        finally:
            rag.status_callback = None
            answer_progress.empty()
            status.empty()

        for key, value in steps.items():
            with st.expander(f"步骤「{key}」："):
                st.text(pprint.pformat(format_state(value), indent=2, width=80))

        final_generation = final_state.get("generation", "未生成最终回答。")
        st.subheader("最终回答：")
        st.write(final_generation)
