import builtins
from types import SimpleNamespace

import pytest

import src.ml.deep_learning.core.integration_tests as integration_module


@pytest.fixture(autouse=True)
def stub_print(monkeypatch):
    outputs = []
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: outputs.append(args))
    return outputs


def test_main_success(monkeypatch):
    results = {
        "total_tests": 10,
        "passed_tests": 10,
        "failed_tests": 0,
        "skipped_tests": 0,
        "pass_rate": 0.98,
    }
    generated = []

    monkeypatch.setattr(
        integration_module,
        "TestSuite",
        lambda: SimpleNamespace(run_all_tests=lambda: results),
    )
    monkeypatch.setattr(
        integration_module,
        "generate_test_report",
        lambda res, path: generated.append((res, path)),
    )

    assert integration_module.main() is True
    assert generated
    assert generated[0][0]["pass_rate"] == 0.98


def test_main_failure(monkeypatch):
    results = {
        "total_tests": 4,
        "passed_tests": 2,
        "failed_tests": 2,
        "skipped_tests": 0,
        "pass_rate": 0.5,
    }

    monkeypatch.setattr(
        integration_module,
        "TestSuite",
        lambda: SimpleNamespace(run_all_tests=lambda: results),
    )
    monkeypatch.setattr(integration_module, "generate_test_report", lambda *args, **kwargs: None)

    assert integration_module.main() is False

