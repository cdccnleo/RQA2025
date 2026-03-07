#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
market_aware_retry 模块单测，覆盖市场阶段判定与重试策略。
"""

from __future__ import annotations

from datetime import datetime, time, timedelta

import pytest

from src.infrastructure.utils.tools.market_aware_retry import (
    MarketAwareRetryHandler,
    MarketPhase,
    SmartOrderRetry,
)


@pytest.fixture
def trading_hours() -> dict:
    return {
        "pre_open": time(9, 15),
        "morning_open": time(9, 30),
        "morning_close": time(11, 30),
        "afternoon_open": time(13, 0),
        "afternoon_close": time(15, 0),
    }


@pytest.fixture
def handler(trading_hours: dict) -> MarketAwareRetryHandler:
    holidays = [datetime(2025, 1, 1)]
    return MarketAwareRetryHandler(
        trading_hours=trading_hours,
        holidays_calendar=holidays,
        timezone="Asia/Shanghai",
    )


def localized(handler: MarketAwareRetryHandler, dt: datetime) -> datetime:
    return handler.timezone.localize(dt)


def test_get_market_phase_covers_all(handler: MarketAwareRetryHandler, trading_hours: dict):
    tztime = handler.timezone

    assert handler.get_market_phase(localized(handler, datetime(2025, 1, 4, 10, 0))) == MarketPhase.CLOSED  # 周六
    assert handler.get_market_phase(localized(handler, datetime(2025, 1, 1, 10, 0))) == MarketPhase.CLOSED  # 节假日
    assert handler.get_market_phase(localized(handler, datetime(2025, 1, 2, 9, 0))) == MarketPhase.PRE_OPEN
    assert handler.get_market_phase(localized(handler, datetime(2025, 1, 2, 10, 0))) == MarketPhase.MORNING
    assert handler.get_market_phase(localized(handler, datetime(2025, 1, 2, 12, 0))) == MarketPhase.LUNCH_BREAK
    assert handler.get_market_phase(localized(handler, datetime(2025, 1, 2, 14, 0))) == MarketPhase.AFTERNOON
    assert handler.get_market_phase(localized(handler, datetime(2025, 1, 2, 15, 30))) == MarketPhase.CLOSED

    assert handler.is_market_open(localized(handler, datetime(2025, 1, 2, 10, 0))) is True
    assert handler.is_market_open(localized(handler, datetime(2025, 1, 2, 12, 10))) is False


def test_next_market_open_time_transitions(handler: MarketAwareRetryHandler):
    # 上午 -> 下午开盘
    morning = localized(handler, datetime(2025, 1, 2, 10, 0))
    next_open = handler.next_market_open_time(morning)
    assert next_open.hour == 13 and next_open.minute == 0

    # 收盘后 -> 下一个工作日早上
    friday_close = localized(handler, datetime(2025, 1, 3, 16, 0))
    monday_open = handler.next_market_open_time(friday_close)
    assert monday_open.weekday() == 0  # 周一
    assert monday_open.hour == 9 and monday_open.minute == 30


def test_retry_strategy(handler: MarketAwareRetryHandler):
    handler.max_retry_attempts = 2
    assert handler.should_retry() is True
    assert handler.should_retry() is True
    assert handler.should_retry() is False  # 超过最大次数

    handler.reset_attempts()
    pre_open = localized(handler, datetime(2025, 1, 2, 9, 5))
    assert handler.get_retry_delay(pre_open) == 0.0

    handler.reset_attempts()
    handler.should_retry()  # current_attempt = 1
    in_session = localized(handler, datetime(2025, 1, 2, 10, 0))
    assert handler.get_retry_delay(in_session) == pytest.approx(handler.base_retry_interval)

    handler.should_retry()  # current_attempt = 2
    assert handler.get_retry_delay(in_session) == pytest.approx(min(handler.base_retry_interval * 2, 60))

    after_close = localized(handler, datetime(2025, 1, 2, 18, 0))
    delay = handler.get_retry_delay(after_close)
    assert delay > 0
    next_open = handler.next_market_open_time(after_close)
    assert delay == pytest.approx((next_open - after_close).total_seconds())


def test_register_holiday_and_update_hours(handler: MarketAwareRetryHandler):
    new_holiday = datetime(2025, 2, 10)
    handler.register_holiday(new_holiday)
    assert new_holiday.date() in handler.holidays

    handler.update_trading_hours({"morning_open": time(9, 45)})
    assert handler.trading_hours["morning_open"] == time(9, 45)


def test_smart_order_retry_flow(handler: MarketAwareRetryHandler, monkeypatch):
    handler.reset_attempts()
    manager = SmartOrderRetry(handler)
    order = {"order_id": "ORD-1"}

    attempts = {"count": 0}

    def fake_execute(self, order_dict):
        attempts["count"] += 1
        return {"success": attempts["count"] >= 2, "order_id": order_dict.get("order_id")}

    monkeypatch.setattr(SmartOrderRetry, "_try_execute", fake_execute, raising=False)

    retry_calls = {"count": 0}

    def fake_should_retry(last_failure_time=None):
        retry_calls["count"] += 1
        return retry_calls["count"] <= 2

    monkeypatch.setattr(handler, "should_retry", fake_should_retry, raising=False)
    monkeypatch.setattr(handler, "get_retry_delay", lambda last_failure_time=None: 0.0, raising=False)

    result = manager.submit_order(order)
    assert result["success"] is False
    assert order["order_id"] in manager.pending_orders

    # 第二次执行应成功并清理 pending
    now = datetime.now()
    manager.check_pending_orders(now=now)
    assert order["order_id"] not in manager.pending_orders

