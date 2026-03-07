import json
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ml.deep_learning.core import integration_tests
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def _ensure_logger():
    integration_tests.logger = logging.getLogger("integration_tests.unit")


def test_create_test_financial_data_generates_expected_columns():
    _ensure_logger()
    df = integration_tests.create_test_financial_data(48)
    assert len(df) == 48
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        assert col in df.columns
    # ensure NaN injection happens but not 100%
    nan_count = df["close"].isna().sum()
    assert 0 < nan_count < 48


def test_generate_test_report_writes_json(tmp_path):
    _ensure_logger()
    results = {
        "total_tests": 5,
        "passed_tests": 4,
        "failed_tests": 1,
        "skipped_tests": 0,
        "test_details": [],
    }
    report_path = tmp_path / "reports" / "integration_report.json"
    integration_tests.generate_test_report(results, str(report_path))

    content = json.loads(report_path.read_text(encoding="utf-8"))
    assert content["test_results"]["total_tests"] == 5
    assert content["summary"]["pass_rate"] == pytest.approx(0.8)
    assert "generated_at" in content


def test_test_suite_run_all_tests_with_mocks(monkeypatch):
    _ensure_logger()

    class FakeLoader:
        def loadTestsFromTestCase(self, test_class):
            return [f"case-{test_class.__name__}"]

    class FakeSuite:
        def __init__(self):
            self.tests = []

        def addTests(self, tests):
            self.tests.extend(tests)

    class FakeRunner:
        def __init__(self, verbosity=1):
            pass

        def run(self, suite):
            return SimpleNamespace(
                testsRun=len(suite.tests), failures=[], errors=[], skipped=[]
            )

    monkeypatch.setattr(integration_tests.unittest, "TestLoader", lambda: FakeLoader())
    monkeypatch.setattr(integration_tests.unittest, "TestSuite", lambda: FakeSuite())
    monkeypatch.setattr(
        integration_tests.unittest, "TextTestRunner", lambda verbosity=2: FakeRunner()
    )

    suite = integration_tests.TestSuite()
    results = suite.run_all_tests()

    assert results["total_tests"] == 1
    assert results["failed_tests"] == 0
    assert results["skipped_tests"] == 0
    assert results["passed_tests"] == 1

