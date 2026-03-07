#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_generator 额外单测，覆盖缺省依赖与异常路径。
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator


class _RaisingComponent:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def boom(self, *args: Any, **kwargs: Any) -> None:
        raise self._exc


class _DummyDataAdapter:
    def __init__(self, event: Dict[str, Any]) -> None:
        self._event = event

    def get_exception_event(self, event_id: str) -> Dict[str, Any]:
        return self._event

    def get_monitoring_data(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"large_orders": [1]}

    def get_account_activity(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"changes": ["delta"]}

    def get_regulatory_data(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"large_trades": ["LT"]}


class _DummyOrderManager:
    def get_daily_stats(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"executed_orders": 5}

    def get_weekly_stats(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"total_volume": 10}

    def get_monthly_stats(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"total_trades": 50}

    def get_orders_by_event(self, *_: Any, **__: Any) -> list[Dict[str, Any]]:
        return [{"order_id": "A1", "symbol": "XYZ", "quantity": 100}]


class _DummyRiskController:
    def get_daily_risk_stats(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"total_checks": 3}

    def get_weekly_risk_stats(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"violation_rate": 0.1}

    def get_monthly_risk_stats(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"total_checks": 20}

    def get_event_actions(self, *_: Any, **__: Any) -> Dict[str, Any]:
        return {"actions": ["freeze"]}


def test_call_or_default_handles_missing_and_exception(caplog: pytest.LogCaptureFixture) -> None:
    generator = ComplianceReportGenerator()

    # 组件缺失返回默认值
    assert generator._call_or_default(None, "method", default={"fallback": True}) == {"fallback": True}

    # 方法缺失返回默认值
    class _Obj:
        pass

    assert generator._call_or_default(_Obj(), "missing", default=0) == 0

    # 方法抛错返回默认值并记录日志
    generator.order_manager = _RaisingComponent(RuntimeError("boom"))
    assert generator._call_or_default(generator.order_manager, "boom", default=[], log_context="测试") == []
    assert any("测试" in record.message for record in caplog.records)


def test_generate_reports_without_dependencies_use_defaults() -> None:
    generator = ComplianceReportGenerator()

    daily = generator.generate_daily_report()
    assert daily["metadata"]["report_type"] == "daily"
    assert daily["sections"]  # 模板仍然存在

    weekly = generator.generate_weekly_report()
    assert weekly["metadata"]["report_type"] == "weekly"

    monthly = generator.generate_monthly_report(year=2024, month=1)
    assert monthly["metadata"]["report_type"] == "monthly"


def test_generate_exception_report_missing_event_raises() -> None:
    generator = ComplianceReportGenerator()
    generator.data_adapter = _DummyDataAdapter(event={})
    generator.order_manager = _DummyOrderManager()
    generator.risk_controller = _DummyRiskController()

    with pytest.raises(ValueError):
        generator.generate_exception_report("event-1")


def test_generate_exception_report_success() -> None:
    generator = ComplianceReportGenerator()
    generator.data_adapter = _DummyDataAdapter(
        {"type": "violation", "detected_at": "2024-01-01", "severity": "high"}
    )
    generator.order_manager = _DummyOrderManager()
    generator.risk_controller = _DummyRiskController()

    report = generator.generate_exception_report("event-2")
    assert report["metadata"]["report_type"] == "exception"
    related_section = next(sec for sec in report["sections"] if sec["name"] == "相关订单")
    assert related_section["data"]["symbols"] == ["XYZ"]
    actions_section = next(sec for sec in report["sections"] if sec["name"] == "处理措施")
    assert actions_section["data"]["actions_taken"] == ["freeze"]

