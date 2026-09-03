"""pytest 公共夹具。"""

import os
import tempfile
import zipfile

import pytest

# 在导入 api 模块前固定知识库注册表路径，避免污染仓库根目录的 knowledge_bases.json
os.environ["KB_REGISTRY_PATH"] = os.path.join(tempfile.gettempdir(), "kb_test.json")
if os.path.exists(os.environ["KB_REGISTRY_PATH"]):
    os.remove(os.environ["KB_REGISTRY_PATH"])


def make_minimal_docx(path: str, text: str = "Docx content test") -> None:
    """创建一个最小可解析的 .docx 文件（纯 OOXML，无需 python-docx）。"""
    parts = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>"
        ),
        "word/document.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
        ),
    }
    with zipfile.ZipFile(path, "w") as z:
        for name, data in parts.items():
            z.writestr(name, data)


@pytest.fixture
def docx_factory(tmp_path):
    def _make(name: str = "sample.docx", text: str = "Docx content test") -> str:
        path = tmp_path / name
        make_minimal_docx(str(path), text)
        return str(path)

    return _make
