#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disaster_tester 额外单测，覆盖依赖缺失与异常分支。
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.infrastructure.utils.components import disaster_tester


@pytest.fixture(autouse=True)
def _patch_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """避免真实 docker.from_env 调用。"""

    class _DummyDocker:
        pass

    monkeypatch.setattr(disaster_tester.docker, "from_env", lambda: _DummyDocker())


class _FailoverMonitor:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.calls = 0

    def get_status(self) -> Dict[str, Dict[str, bool]]:
        self.calls += 1
        # 第一次返回 primary=False 以满足故障检测；后续保持 False
        return {"health_status": {"primary": False, "secondary": True}}


class _RecoveryMonitor:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.calls = 0

    def get_status(self) -> Dict[str, Dict[str, bool]]:
        self.calls += 1
        if self.calls < 2:
            return {"health_status": {"primary": False, "secondary": False}}
        return {"health_status": {"primary": True, "secondary": True}}


class _DummyOrderManager:
    def get_daily_stats(self, *_: Any) -> Dict[str, int]:
        return {"executed_orders": 3}

    def get_weekly_stats(self, *_: Any) -> Dict[str, int]:
        return {"total_volume": 10}

    def get_monthly_stats(self, *_: Any) -> Dict[str, int]:
        return {"total_trades": 20, "top_symbols": ["AAA"]}

    def get_orders_by_event(self, *_: Any) -> list[Dict[str, Any]]:
        return [{"order_id": "O1", "symbol": "XYZ", "quantity": 100}]


class _DummyRiskController:
    def get_daily_risk_stats(self, *_: Any) -> Dict[str, int]:
        return {"total_checks": 1}

    def get_weekly_risk_stats(self, *_: Any) -> Dict[str, Any]:
        return {"violation_rate": 0.1}

    def get_monthly_risk_stats(self, *_: Any) -> Dict[str, Any]:
        return {"total_checks": 5, "violation_trends": {"increasing": False}, "metrics": {}}

    def get_event_actions(self, *_: Any) -> Dict[str, Any]:
        return {"actions": ["freeze"], "follow_up": "done"}


class _DummyDataAdapter:
    def __init__(self, event: Dict[str, Any]) -> None:
        self._event = event

    def get_monitoring_data(self, *_: Any) -> Dict[str, Any]:
        return {"large_orders": [1]}

    def get_account_activity(self, *_: Any) -> Dict[str, Any]:
        return {"changes": ["delta"], "positions": [], "margin_usage": {}}

    def get_regulatory_data(self, *_: Any) -> Dict[str, Any]:
        return {"large_trades": ["LT"], "positions": [], "short_selling": {}}

    def get_exception_event(self, event_id: str) -> Dict[str, Any]:
        return self._event


def test_error_handler_collects_messages(caplog: pytest.LogCaptureFixture) -> None:
    handler = disaster_tester.ErrorHandler()
    handler.handle(RuntimeError("boom"))
    assert handler.errors == ["boom"]
    assert any("boom" in record.message for record in caplog.records)


def test_start_and_stop_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    started: Dict[str, bool] = {"start": False, "joined": False}

    class _DummyThread:
        def __init__(self, target, daemon):
            self._target = target

        def start(self):
            started["start"] = True
            self._target()

        def join(self, timeout):
            started["joined"] = True

    monkeypatch.setattr(disaster_tester.threading, "Thread", _DummyThread)
    tester = disaster_tester.DisasterTester()
    tester.test_cases = []  # 避免运行真实用例
    tester.start_test_suite()
    tester.stop_test_suite()
    assert started["start"] and started["joined"]


def test_run_test_suite_handles_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    tester = disaster_tester.DisasterTester()
    tester.test_cases = [{"name": "boom", "type": "failover", "params": {}}]

    def _raising_case(_):
        raise RuntimeError("fail")

    tester._run_test_case = _raising_case  # type: ignore[assignment]
    captured: list[str] = []

    def _handle(exc: Exception) -> None:
        captured.append(str(exc))

    tester.error_handler.handle = _handle  # type: ignore[assignment]
    tester.running = True
    tester._run_test_suite()
    assert captured == ["fail"]
    tester.running = False


def test_run_test_case_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    tester = disaster_tester.DisasterTester()
    monkeypatch.setattr(tester, "_test_failover", lambda **_: True)
    monkeypatch.setattr(tester, "_test_data_sync", lambda **_: True)
    monkeypatch.setattr(tester, "_test_recovery", lambda **_: False)
    monkeypatch.setattr(tester, "_test_performance", lambda **_: True)

    assert tester._run_test_case({"name": "f1", "type": "failover", "params": {}})["passed"] is True
    assert tester._run_test_case({"name": "f2", "type": "data_sync", "params": {}})["passed"] is True
    assert tester._run_test_case({"name": "f3", "type": "recovery", "params": {}})["passed"] is False
    assert tester._run_test_case({"name": "f4", "type": "performance", "params": {}})["passed"] is True


def test_test_failover_without_monitor_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disaster_tester, "DisasterMonitor", None)
    tester = disaster_tester.DisasterTester(config={"monitoring": {}})
    assert tester._test_failover("svc", 1.0, "service_status") is False


def test_test_failover_with_monitor_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disaster_tester, "DisasterMonitor", _FailoverMonitor)
    tester = disaster_tester.DisasterTester(config={"monitoring": {}})
    assert tester._test_failover("svc", 1.0, "service_status") is True


def test_test_data_sync_polices_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    tester = disaster_tester.DisasterTester()
    monkeypatch.setattr(tester, "_sync_data", lambda data: None)
    monkeypatch.setattr(tester, "_verify_sync_result", lambda expected: True)
    assert tester._test_data_sync(data_size=0, expected_time=0.001, tolerance=0.5) is True


def test_test_recovery_with_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disaster_tester, "DisasterMonitor", _RecoveryMonitor)
    tester = disaster_tester.DisasterTester(config={"monitoring": {}})
    assert tester._test_recovery("primary", expected_time=1.0) is True


def test_generate_report_summary() -> None:
    tester = disaster_tester.DisasterTester()
    tester.test_results = [
        {"name": "ok", "passed": True},
        {"name": "fail", "passed": False},
    ]
    summary = tester.generate_report()
    assert summary["summary"]["passed"] == 1
    assert summary["summary"]["failed"] == 1
    assert summary["summary"]["success_rate"] == 0.5

