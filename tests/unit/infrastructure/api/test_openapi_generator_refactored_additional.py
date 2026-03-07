import json
from pathlib import Path

import pytest

from src.infrastructure.api.openapi_generator_refactored import (
    DocumentationAssembler,
    DocumentationExportConfig,
    RQAApiDocumentationGenerator,
)


class _GeneratorUnderTest(RQAApiDocumentationGenerator):
    """便于注入假数据的子类"""

    def __init__(self):
        super().__init__()
        # 覆盖默认 schema 以便控制输出
        self.api_schema.title = "Injected API"
        self.api_schema.version = "2.0.0"
        self.api_schema.description = "测试描述"
        self.api_schema.servers = [{"url": "http://test"}]

    def _prepare_document_assembler(self):
        super()._prepare_document_assembler()
        # 添加额外标签，用于验证 include_statistics 流程
        self._doc_assembler.add_tags([{"name": "Custom", "description": "自定义"}])


def test_generate_documentation_with_config_object(tmp_path: Path, monkeypatch):
    generator = _GeneratorUnderTest()

    # 打补丁捕获统计输出
    outputs = []

    def fake_print(*args, **kwargs):
        outputs.append(" ".join(map(str, args)))

    monkeypatch.setattr("builtins.print", fake_print)

    config = DocumentationExportConfig(
        output_dir=str(tmp_path),
        format_types=["json", "yaml"],
        include_statistics=True,
    )

    result = generator.generate_documentation(config)

    assert (tmp_path / "openapi.json").exists()
    assert (tmp_path / "openapi.yaml").exists()
    assert result["json"].endswith("openapi.json")
    assert any("API文档统计" in line for line in outputs)


def test_generate_documentation_with_string_path(tmp_path: Path):
    generator = _GeneratorUnderTest()
    output_dir = tmp_path / "string-mode"

    result = generator.generate_documentation(str(output_dir))

    assert (output_dir / "openapi.json").exists()
    assert (output_dir / "openapi.yaml").exists()
    assert "json" in result
    assert "yaml" in result


def test_get_statistics_returns_counts():
    generator = _GeneratorUnderTest()
    stats = generator.get_statistics()

    assert stats["endpoints"]["total"] >= 1
    assert stats["schemas"] >= 1
    assert stats["servers"] == len(generator.api_schema.servers)
    assert stats["security_schemes"] == len(generator.api_schema.security_schemes)

