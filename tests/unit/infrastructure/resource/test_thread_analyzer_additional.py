from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

import src.infrastructure.resource.utils.thread_analyzer as analyzer_module
from src.infrastructure.resource.utils.thread_analyzer import ThreadAnalyzer


@dataclass
class DummyThread:
    name: str
    ident: int
    daemon: bool
    _alive: bool

    def is_alive(self) -> bool:
        return self._alive


class DummyLogger:
    def __init__(self):
        self.errors: List[str] = []

    def error(self, message: str, *args, **kwargs):
        self.errors.append(message)

    def log_error(self, message: str, *args, **kwargs):
        self.error(message, *args, **kwargs)


class DummyErrorHandler:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        self.calls.append({"error": error, "context": context or {}})


@pytest.fixture
def analyzer():
    logger = DummyLogger()
    error_handler = DummyErrorHandler()
    return ThreadAnalyzer(logger=logger, error_handler=error_handler), logger, error_handler


def test_analyze_threads_with_stacks(monkeypatch, analyzer):
    analyzer_instance, _, _ = analyzer
    main_thread = DummyThread("MainThread", 1, False, True)
    worker_thread = DummyThread("Worker", 2, True, False)

    monkeypatch.setattr(analyzer_module.threading, "current_thread", lambda: main_thread)
    monkeypatch.setattr(analyzer_module.threading, "enumerate", lambda: [main_thread, worker_thread])

    result = analyzer_instance.analyze_threads(include_stacks=True)

    assert result["current_thread"]["name"] == "MainThread"
    assert result["thread_count"] == 2
    assert result["statistics"]["dead_threads"] == 1
    stacks = [thread_data.get("stack") for thread_data in result["threads"]]
    assert any(isinstance(stack, list) and stack for stack in stacks)
    assert any(stack == ["Thread Worker (ID: 2)"] for stack in stacks)


def test_analyze_threads_handles_failure(monkeypatch, analyzer):
    analyzer_instance, _, error_handler = analyzer
    monkeypatch.setattr(analyzer_module.threading, "current_thread", lambda: DummyThread("Main", 1, False, True))
    def raise_enum():
        raise RuntimeError("boom")

    monkeypatch.setattr(analyzer_module.threading, "enumerate", raise_enum)

    result = analyzer_instance.analyze_threads(include_stacks=False)
    assert "error" in result
    assert error_handler.calls and error_handler.calls[0]["context"]["context"] == "分析线程状态失败"


def test_detect_thread_issues(monkeypatch, analyzer):
    analyzer_instance, _, _ = analyzer
    threads = [DummyThread(f"T{i}", i, daemon=(i % 2 == 0), _alive=(i % 5 != 0)) for i in range(60)]
    monkeypatch.setattr(analyzer_module.threading, "enumerate", lambda: threads)

    issues = analyzer_instance.detect_thread_issues()
    assert any(problem["type"] == "high_thread_count" for problem in issues["problems"]) or any(
        warning["type"] == "moderate_thread_count" for warning in issues["warnings"]
    )
    assert any(warning["type"] == "dead_threads" for warning in issues["warnings"])


def test_analyze_thread_stacks(monkeypatch, analyzer):
    analyzer_instance, _, _ = analyzer
    threads = [DummyThread("Main", 1, False, True), DummyThread("Worker", 2, True, True)]

    monkeypatch.setattr(analyzer_module.threading, "enumerate", lambda: threads)

    result = analyzer_instance.analyze_thread_stacks()
    assert result["thread_count"] == 2
    for info in result["thread_stacks"]:
        assert "stack" in info
        assert "stack_lines" in info


def test_deadlock_risk_assessment(monkeypatch, analyzer):
    analyzer_instance, _, _ = analyzer
    threads = [DummyThread(f"T{i}", i, daemon=False, _alive=i % 3 != 0) for i in range(120)]
    monkeypatch.setattr(analyzer_module.threading, "enumerate", lambda: threads)

    result = analyzer_instance.get_deadlock_risk()
    assert result["risk_level"] in {"medium", "high"}
    assert result["total_risk_score"] >= 0
    assert result["risk_factors"]


def test_thread_info_and_count(monkeypatch, analyzer):
    analyzer_instance, _, _ = analyzer
    main_thread = DummyThread("MainThread", 1, False, True)
    threads = [main_thread, DummyThread("Worker", 2, True, True)]

    monkeypatch.setattr(analyzer_module.threading, "current_thread", lambda: main_thread)
    monkeypatch.setattr(analyzer_module.threading, "enumerate", lambda: threads)

    assert analyzer_instance.get_thread_count() == 2

    info = analyzer_instance.get_thread_info()
    assert info["total_threads"] == 2
    assert info["current_thread"]["name"] == "MainThread"
    assert len(info["thread_list"]) == 2

