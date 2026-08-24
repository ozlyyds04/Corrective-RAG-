"""文档加载器单元测试。"""

import pytest

from corrective_rag.core import (
    SUPPORTED_EXTENSIONS,
    build_sparse_vector,
    expand_document_sources,
    load_documents,
    load_knowledge_bases,
    load_local_path,
    save_knowledge_bases,
    tokenize_for_sparse,
)


def write_text(tmp_path, name, content, encoding="utf-8"):
    path = tmp_path / name
    if isinstance(content, str):
        path.write_text(content, encoding=encoding)
    else:
        path.write_bytes(content)
    return str(path)


def test_txt_utf8(tmp_path):
    path = write_text(tmp_path, "a.txt", "你好，Corrective RAG 测试。")
    docs = load_documents(path, is_url=False)
    assert len(docs) == 1
    assert "Corrective RAG" in docs[0].page_content


def test_txt_gbk_fallback(tmp_path):
    path = tmp_path / "gbk.txt"
    path.write_bytes("中文编码测试内容".encode("gbk"))
    docs = load_documents(str(path), is_url=False)
    assert "中文编码测试内容" in docs[0].page_content


def test_markdown(tmp_path):
    path = write_text(tmp_path, "note.md", "# 标题\n正文内容")
    docs = load_documents(path, is_url=False)
    assert "正文内容" in docs[0].page_content


def test_csv(tmp_path):
    path = write_text(tmp_path, "data.csv", "name,age\nAlice,30\nBob,25")
    docs = load_documents(path, is_url=False)
    contents = "\n".join(d.page_content for d in docs)
    assert "Alice" in contents


def test_html(tmp_path):
    path = write_text(
        tmp_path, "page.html", "<html><body><h1>标题</h1><p>正文内容</p></body></html>"
    )
    docs = load_documents(path, is_url=False)
    assert any("正文内容" in d.page_content for d in docs)


def test_docx(docx_factory):
    path = docx_factory()
    docs = load_documents(path, is_url=False)
    assert any("Docx content test" in d.page_content for d in docs)


def test_unsupported_extension(tmp_path):
    path = write_text(tmp_path, "notes.xyz", "hello")
    with pytest.raises(ValueError, match="不支持的文档类型"):
        load_documents(path, is_url=False)


def test_local_path_single_file(tmp_path):
    path = write_text(tmp_path, "single.txt", "单文件内容")
    docs = load_local_path(path)
    assert len(docs) == 1
    assert "单文件内容" in docs[0].page_content


def test_local_path_folder_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    write_text(tmp_path, "a.txt", "文件A")
    write_text(tmp_path / "sub", "b.md", "# 文件B")
    docs = load_local_path(str(tmp_path))
    assert len(docs) == 2


def test_local_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="路径不存在"):
        load_local_path(str(tmp_path / "nope"))


def test_local_path_no_supported_files(tmp_path):
    write_text(tmp_path, "x.xyz", "x")
    with pytest.raises(ValueError, match="受支持"):
        load_local_path(str(tmp_path))


def test_supported_extensions():
    assert SUPPORTED_EXTENSIONS == {".pdf", ".txt", ".md", ".docx", ".html", ".csv"}


def test_expand_sources_folder_and_url(tmp_path):
    (tmp_path / "sub").mkdir()
    write_text(tmp_path, "a.txt", "A")
    write_text(tmp_path / "sub", "b.md", "B")
    expanded = expand_document_sources([str(tmp_path), "https://example.com/x.pdf"])
    assert len(expanded) == 3
    assert any(s.endswith("a.txt") for s in expanded)
    assert any(s.endswith("b.md") for s in expanded)
    assert "https://example.com/x.pdf" in expanded


def test_expand_sources_missing_path():
    with pytest.raises(FileNotFoundError, match="路径不存在"):
        expand_document_sources(["Z:/no/such/path"])


def test_expand_sources_empty_folder(tmp_path):
    with pytest.raises(ValueError, match="受支持"):
        expand_document_sources([str(tmp_path)])


def test_knowledge_bases_registry(tmp_path):
    path = str(tmp_path / "kb.json")
    assert load_knowledge_bases(path) == ["默认"]
    save_knowledge_bases(path, ["默认", "面试库"])
    assert load_knowledge_bases(path) == ["默认", "面试库"]
    # 损坏的注册表文件应回退到默认
    (tmp_path / "kb.json").write_text("{bad json", encoding="utf-8")
    assert load_knowledge_bases(path) == ["默认"]


def test_sparse_vector_builder_invariants():
    sv = build_sparse_vector("向量数据库 混合检索 测试")
    assert len(sv.indices) > 0
    assert len(sv.indices) == len(sv.values)
    assert sv.indices == sorted(sv.indices)
    assert all(v > 0 for v in sv.values)


def test_sparse_vector_deterministic():
    text = "混合检索 向量 测试 检索"
    sv1 = build_sparse_vector(text)
    sv2 = build_sparse_vector(text)
    assert sv1.indices == sv2.indices
    assert sv1.values == sv2.values


def test_tokenize_contains_chinese_and_english():
    tokens = tokenize_for_sparse("向量 database 混合检索")
    joined = " ".join(tokens)
    assert "database" in joined
    assert any("\u4e00" <= ch <= "\u9fff" for ch in joined)
