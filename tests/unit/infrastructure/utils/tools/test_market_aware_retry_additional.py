import pytz
import pytest

from datetime import datetime, time, timedelta
from typing import Dict

from src.infrastructure.utils.tools.market_aware_retry import (
    MarketAwareRetryHandler,
    MarketPhase,
    SmartOrderRetry,
)


@pytest.fixture
def handler():
    return MarketAwareRetryHandler(holidays_calendar=[], timezone="Asia/Shanghai")


def localized(tz, year, month, day, hour, minute=0):
    return tz.localize(datetime(year, month, day, hour, minute))


def test_market_phase_transitions(handler):
    tz = handler.timezone
    assert handler.get_market_phase(localized(tz, 2024, 4, 3, 9, 10)) == MarketPhase.PRE_OPEN
    assert handler.get_market_phase(localized(tz, 2024, 4, 3, 10, 0)) == MarketPhase.MORNING
    assert handler.get_market_phase(localized(tz, 2024, 4, 3, 12, 15)) == MarketPhase.LUNCH_BREAK
    assert handler.get_market_phase(localized(tz, 2024, 4, 3, 14, 0)) == MarketPhase.AFTERNOON
    assert handler.get_market_phase(localized(tz, 2024, 4, 3, 16, 30)) == MarketPhase.CLOSED
    assert handler.get_market_phase(localized(tz, 2024, 4, 6, 10, 0)) == MarketPhase.CLOSED


def test_is_market_open(handler):
    tz = handler.timezone
    assert handler.is_market_open(localized(tz, 2024, 4, 3, 10, 0))
    assert not handler.is_market_open(localized(tz, 2024, 4, 3, 12, 30))
    assert not handler.is_market_open(localized(tz, 2024, 4, 6, 10, 0))


def test_next_market_open_time(handler):
    tz = handler.timezone
    lunch_time = localized(tz, 2024, 4, 3, 12, 10)
    assert handler.next_market_open_time(lunch_time) == localized(
        tz, 2024, 4, 3, handler.trading_hours["afternoon_open"].hour, handler.trading_hours["afternoon_open"].minute
    )

    friday_close = localized(tz, 2024, 4, 5, 16, 0)
    next_open = handler.next_market_open_time(friday_close)
    assert next_open.weekday() == 0  # Monday
    assert next_open.hour == handler.trading_hours["morning_open"].hour
    assert next_open.minute == handler.trading_hours["morning_open"].minute


def test_get_retry_delay_during_open(handler):
    tz = handler.timezone
    market_time = localized(tz, 2024, 4, 3, 10, 0)
    handler.reset_attempts()
    handler.should_retry()
    assert handler.get_retry_delay(market_time) == pytest.approx(handler.base_retry_interval)
    handler.should_retry()
    assert handler.get_retry_delay(market_time) == pytest.approx(min(handler.base_retry_interval * 2, 60))
    handler.reset_attempts()
    handler.max_retry_attempts = 10
    for _ in range(6):
        handler.should_retry()
    assert handler.get_retry_delay(market_time) == pytest.approx(60.0)


def test_get_retry_delay_when_market_closed(handler):
    tz = handler.timezone
    after_close = localized(tz, 2024, 4, 3, 16, 0)
    handler.reset_attempts()
    expected = (handler.next_market_open_time(after_close) - after_close).total_seconds()
    assert handler.get_retry_delay(after_close) == pytest.approx(expected)


def test_register_holiday_and_phase(handler):
    tz = handler.timezone
    holiday = localized(tz, 2024, 4, 4, 10, 0)
    handler.register_holiday(holiday)
    assert holiday.date() in handler.holidays
    assert handler.get_market_phase(holiday) == MarketPhase.CLOSED


def test_update_trading_hours(handler):
    handler.update_trading_hours({"morning_open": time(9, 0)})
    assert handler.trading_hours["morning_open"] == time(9, 0)


class ToggleOrderRetry(SmartOrderRetry):
    def __init__(self, retry_handler):
        super().__init__(retry_handler)
        self._attempts = {}

    def _try_execute(self, order: Dict) -> Dict:
        order_id = order["order_id"]
        count = self._attempts.get(order_id, 0)
        self._attempts[order_id] = count + 1
        return {"success": count >= 1, "order_id": order_id}


def test_smart_order_retry_flow(handler):
    handler.reset_attempts()
    manager = ToggleOrderRetry(handler)
    order = {"order_id": "ORD-1"}

    result = manager.submit_order(order)
    assert result["success"] is False
    assert "ORD-1" in manager.pending_orders

    target_time = manager.pending_orders["ORD-1"]["next_retry"] + timedelta(seconds=1)
    manager.check_pending_orders(now=target_time)
    assert "ORD-1" not in manager.pending_orders
import pytz
import pytest

from datetime import datetime, time, timedelta
from typing import Dict

from src.infrastructure.utils.tools.market_aware_retry import (
    MarketAwareRetryHandler,
    MarketPhase,
    SmartOrderRetry,
)


@pytest.fixture
def handler():
    return MarketAwareRetryHandler(holidays_calendar=[], timezone="Asia/Shanghai")


def localized(tz, year, month, day, hour, minute=0):
    return tz.localize(datetime(year, month, day, hour, minute))


def test_market_phase_transitions(handler):
    tz = handler.timezone
    assert handler.get_market_phase(localized(tz, 2024, 4, 3, 9, 10)) == MarketPhase.PRE_OPEN
    assert handler.get_market_phase(localized(tz, 2024, 4, 3, 10, 0)) == MarketPhase.MORNING
    assert handler.get_market_phase(localized(tz, 2024, 4, 3, 12, 15)) == MarketPhase.LUNCH_BREAK
    assert handler.get_market_phase(localized(tz, 2024, 4, 3, 14, 0)) == MarketPhase.AFTERNOON
    assert handler.get_market_phase(localized(tz, 2024, 4, 3, 16, 30)) == MarketPhase.CLOSED
    assert handler.get_market_phase(localized(tz, 2024, 4, 6, 10, 0)) == MarketPhase.CLOSED


def test_is_market_open(handler):
    tz = handler.timezone
    assert handler.is_market_open(localized(tz, 2024, 4, 3, 10, 0))
    assert not handler.is_market_open(localized(tz, 2024, 4, 3, 12, 30))
    assert not handler.is_market_open(localized(tz, 2024, 4, 6, 10, 0))


def test_next_market_open_time(handler):
    tz = handler.timezone
    lunch_time = localized(tz, 2024, 4, 3, 12, 10)
    assert handler.next_market_open_time(lunch_time) == localized(
        tz, 2024, 4, 3, handler.trading_hours["afternoon_open"].hour, handler.trading_hours["afternoon_open"].minute
    )

    friday_close = localized(tz, 2024, 4, 5, 16, 0)
    next_open = handler.next_market_open_time(friday_close)
    assert next_open.weekday() == 0  # Monday
    assert next_open.hour == handler.trading_hours["morning_open"].hour
    assert next_open.minute == handler.trading_hours["morning_open"].minute


def test_get_retry_delay_during_open(handler):
    tz = handler.timezone
    market_time = localized(tz, 2024, 4, 3, 10, 0)
    handler.reset_attempts()
    handler.should_retry()
    assert handler.get_retry_delay(market_time) == pytest.approx(handler.base_retry_interval)
    handler.should_retry()
    assert handler.get_retry_delay(market_time) == pytest.approx(min(handler.base_retry_interval * 2, 60))
    handler.reset_attempts()
    handler.max_retry_attempts = 10
    for _ in range(6):
        handler.should_retry()
    assert handler.get_retry_delay(market_time) == pytest.approx(60.0)


def test_get_retry_delay_when_market_closed(handler):
    tz = handler.timezone
    after_close = localized(tz, 2024, 4, 3, 16, 0)
    handler.reset_attempts()
    expected = (handler.next_market_open_time(after_close) - after_close).total_seconds()
    assert handler.get_retry_delay(after_close) == pytest.approx(expected)


def test_register_holiday_and_phase(handler):
    tz = handler.timezone
    holiday = localized(tz, 2024, 4, 4, 10, 0)
    handler.register_holiday(holiday)
    assert holiday.date() in handler.holidays
    assert handler.get_market_phase(holiday) == MarketPhase.CLOSED


def test_update_trading_hours(handler):
    handler.update_trading_hours({"morning_open": time(9, 0)})
    assert handler.trading_hours["morning_open"] == time(9, 0)


class ToggleOrderRetry(SmartOrderRetry):
    def __init__(self, retry_handler):
        super().__init__(retry_handler)
        self._attempts = {}

    def _try_execute(self, order: Dict) -> Dict:
        order_id = order["order_id"]
        count = self._attempts.get(order_id, 0)
        self._attempts[order_id] = count + 1
        return {"success": count >= 1, "order_id": order_id}


def test_smart_order_retry_flow(handler):
    handler.reset_attempts()
    manager = ToggleOrderRetry(handler)
    order = {"order_id": "ORD-1"}

    result = manager.submit_order(order)
    assert result["success"] is False
    assert "ORD-1" in manager.pending_orders

    target_time = manager.pending_orders["ORD-1"]["next_retry"] + timedelta(seconds=1)
    manager.check_pending_orders(now=target_time)
    assert "ORD-1" not in manager.pending_orders

