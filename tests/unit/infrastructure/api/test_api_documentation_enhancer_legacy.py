from copy import deepcopy

import pytest

from src.infrastructure.api.api_documentation_enhancer import APIDocumentationEnhancer


def test_add_enhancement_and_enhance_documentation():
    enhancer = APIDocumentationEnhancer()

    def add_version(docs):
        updated = deepcopy(docs)
        updated["info"]["version"] = "2.0"
        return updated

    enhancer.add_enhancement("add_version", add_version)

    docs = {"info": {"title": "API", "version": "1.0"}}
    enhanced = enhancer.enhance_documentation(docs)

    # 原对象不应被修改
    assert docs["info"]["version"] == "1.0"
    assert enhanced["info"]["version"] == "2.0"
    assert enhancer.get_enhancement_count() == 1


def test_enhancement_failure_is_caught_and_continues(capfd):
    enhancer = APIDocumentationEnhancer()

    def broken_enhancement(_docs):
        raise ValueError("boom")

    def second_enhancement(docs):
        updated = deepcopy(docs)
        updated["flag"] = True
        return updated

    enhancer.add_enhancement("broken", broken_enhancement)
    enhancer.add_enhancement("second", second_enhancement)

    result = enhancer.enhance_documentation({})
    captured = capfd.readouterr().out

    assert result["flag"] is True
    assert "应用增强 broken 时出错" in captured


def test_remove_enhancement_and_apply_template():
    enhancer = APIDocumentationEnhancer()

    enhancer.add_enhancement("noop", lambda docs: docs)
    assert enhancer.remove_enhancement("noop") is True
    assert enhancer.remove_enhancement("noop") is False

    enhancer.add_template("with_meta", lambda docs: {**docs, "meta": {"author": "qa"}})
    templated = enhancer.apply_template({"info": {}}, "with_meta")
    assert templated["meta"]["author"] == "qa"
    assert enhancer.get_template_count() == 1

    unchanged = enhancer.apply_template({"info": {}}, "missing")
    assert unchanged == {"info": {}}

