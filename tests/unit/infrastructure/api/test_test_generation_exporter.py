import json
from pathlib import Path

import pytest

from src.infrastructure.api.test_generation.exporter import TestSuiteExporter
from src.infrastructure.api.test_generation.models import TestCase, TestScenario, TestSuite


def _build_sample_suite() -> TestSuite:
    case = TestCase(
        id="tc-1",
        title="验证响应码",
        description="响应应为200",
        priority="high",
        category="functional",
        preconditions=["服务可用"],
        test_steps=[{"step": "调用接口"}],
        expected_results=["返回200"],
        status="passed",
        tags=["smoke"],
    )
    scenario = TestScenario(
        id="sc-1",
        name="获取用户信息",
        description="GET /users/{id}",
        endpoint="/users/{id}",
        method="GET",
        test_cases=[case],
        setup_steps=["启动服务"],
        teardown_steps=["清理环境"],
    )
    return TestSuite(
        id="suite-1",
        name="用户接口套件",
        description="覆盖用户接口测试",
        scenarios=[scenario],
    )


def test_export_json_creates_file_with_content(tmp_path):
    suite = _build_sample_suite()
    exporter = TestSuiteExporter()

    exporter.export(test_suites={"suite-1": suite}, format_type="json", output_dir=str(tmp_path))

    json_path = tmp_path / "test_suites.json"
    assert json_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["suite-1"]["name"] == "用户接口套件"
    assert data["suite-1"]["scenarios"][0]["test_cases"][0]["title"] == "验证响应码"


def test_export_html_and_markdown_contains_suite(tmp_path):
    suite = _build_sample_suite()
    exporter = TestSuiteExporter()

    exporter.export(test_suites={"suite-1": suite}, format_type="html", output_dir=str(tmp_path))
    exporter.export(test_suites={"suite-1": suite}, format_type="markdown", output_dir=str(tmp_path))

    html_content = (tmp_path / "test_suites.html").read_text(encoding="utf-8")
    md_content = (tmp_path / "test_suites.md").read_text(encoding="utf-8")

    assert "用户接口套件" in html_content
    assert "覆盖用户接口测试" in md_content


def test_export_yaml_when_available(tmp_path):
    yaml = pytest.importorskip("yaml")
    suite = _build_sample_suite()
    exporter = TestSuiteExporter()

    exporter.export(test_suites={"suite-1": suite}, format_type="yaml", output_dir=str(tmp_path))

    yaml_path = tmp_path / "test_suites.yaml"
    assert yaml_path.exists()
    content = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert content["suite-1"]["scenarios"][0]["method"] == "GET"


def test_legacy_export_uses_output_path(tmp_path):
    suite = _build_sample_suite()
    exporter = TestSuiteExporter()
    output_file = tmp_path / "legacy.json"

    exporter.export(test_suite=suite, format="json", output_path=str(output_file))

    assert output_file.exists()
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert "combined" in data
    assert data["combined"]["scenarios"][0]["endpoint"] == "/users/{id}"


def test_export_unsupported_format_raises(tmp_path):
    suite = _build_sample_suite()
    exporter = TestSuiteExporter()

    with pytest.raises(ValueError):
        exporter.export(test_suites={"suite-1": suite}, format_type="pdf", output_dir=str(tmp_path))

