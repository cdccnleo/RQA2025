import json

import pandas as pd
import pytest

import src.ml.deep_learning.core.integration_tests as integration_module


def test_test_suite_run_all_tests(monkeypatch):
    class DummyLoader:
        def loadTestsFromTestCase(self, cls):
            return [cls.__name__]

    class DummySuite:
        def __init__(self):
            self.tests = []

        def addTests(self, tests):
            self.tests.extend(tests)

    class DummyResult:
        testsRun = 4
        failures = []
        errors = [("case", AssertionError("boom"))]
        skipped = [("case", "skip")]

    class DummyRunner:
        def __init__(self, verbosity):
            self.verbosity = verbosity

        def run(self, suite):
            return DummyResult()

    monkeypatch.setattr(integration_module.unittest, "TestLoader", lambda: DummyLoader())
    monkeypatch.setattr(integration_module.unittest, "TestSuite", lambda: DummySuite())
    monkeypatch.setattr(integration_module.unittest, "TextTestRunner", lambda verbosity: DummyRunner(verbosity))

    suite = integration_module.TestSuite()
    results = suite.run_all_tests()

    assert results["total_tests"] == 4
    assert results["failed_tests"] == 1
    assert results["skipped_tests"] == 1
    assert results["passed_tests"] == 3


def test_generate_test_report_creates_file(tmp_path):
    results = {
        "total_tests": 5,
        "passed_tests": 3,
        "failed_tests": 1,
        "skipped_tests": 1,
    }
    output = tmp_path / "report.json"
    integration_module.generate_test_report(results, str(output))

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["test_results"]["failed_tests"] == 1
    assert "修复失败的测试用例" in data["recommendations"]
    assert data["summary"]["total_tests"] == 5


def test_create_test_financial_data_structure():
    df = integration_module.create_test_financial_data(50)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    for column in ["open", "high", "low", "close", "volume"]:
        assert column in df.columns

