"""Streamlit 界面冒烟测试（AppTest，不触发真实入库/网络请求）。"""

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "src" / "corrective_rag" / "app.py"


def make_app(registry_path: Path, **session_overrides):
    os.environ["KB_REGISTRY_PATH"] = str(registry_path / "kb.json")
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.session_state["initialized"] = True
    at.session_state["openai_api_key"] = "sk-test"
    at.session_state["openai_base_url"] = ""
    at.session_state["model"] = "deepseek-v4-flash"
    at.session_state["embedding_model"] = "text-embedding-3-small"
    at.session_state["embedding_api_key"] = ""
    at.session_state["embedding_base_url"] = ""
    at.session_state["qdrant_url"] = ""
    at.session_state["qdrant_api_key"] = ""
    at.session_state["tavily_api_key"] = "tvly-test"
    at.session_state["doc_url"] = ""
    at.session_state["local_path"] = ""
    at.session_state["upload_limit"] = 50
    for key, value in session_overrides.items():
        at.session_state[key] = value
    at.run()
    return at


def checkbox_by_label(at, label):
    return [c for c in at.checkbox if c.label == label][0]


def test_app_renders_without_keys():
    import tempfile

    os.environ["KB_REGISTRY_PATH"] = os.path.join(tempfile.gettempdir(), "kb_test.json")
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception
    assert [c.label for c in at.checkbox] == ["URL", "本地路径/文件夹", "文件上传"]
    assert "纠正式" in at.title[0].value
    assert any("当前知识库" in s.label for s in at.selectbox)


def test_local_folder_mode_loads_documents(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("本地文件内容", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("# 标题", encoding="utf-8")

    at = make_app(tmp_path, local_path=str(tmp_path), ingested_source=str(tmp_path))
    checkbox_by_label(at, "本地路径/文件夹").set_value(True)
    at.run()

    assert not at.exception
    assert not at.error


def test_multi_url_textarea_accepts_multiple_urls(tmp_path):
    source_key = "https://example.com/a.pdf|https://example.com/b.pdf"
    at = make_app(tmp_path, ingested_source=source_key)
    checkbox_by_label(at, "URL").set_value(True)
    at.run()
    at.text_area[0].set_value("https://example.com/a.pdf\nhttps://example.com/b.pdf")
    at.run()
    assert not at.exception
    assert "example.com/a.pdf" in at.text_area[0].value


def test_upload_limit_warns_and_truncates(tmp_path):
    at = make_app(tmp_path)
    checkbox_by_label(at, "文件上传").set_value(True)
    at.run()
    limit = [n for n in at.number_input if "最多上传" in n.label][0]
    limit.set_value(2)
    at.run()
    at.file_uploader[0].set_value(
        [
            ("a.txt", b"a", "text/plain"),
            ("b.txt", b"b", "text/plain"),
            ("c.txt", b"c", "text/plain"),
        ]
    )
    at.run()
    assert not at.exception
    assert any("最多上传 2" in w.value for w in at.warning)


def test_create_knowledge_base(tmp_path):
    at = make_app(tmp_path)
    name_input = [t for t in at.text_input if "新建知识库" in t.label][0]
    name_input.set_value("面试库")
    at.run()
    create_btn = [b for b in at.button if b.label == "新建知识库"][0]
    create_btn.set_value(True)
    at.run()
    assert not at.exception
    selector = [s for s in at.selectbox if "当前知识库" in s.label][0]
    assert "面试库" in selector.options
