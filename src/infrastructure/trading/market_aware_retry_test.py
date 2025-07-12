import pytest
from datetime import datetime, time, timedelta
from .market_aware_retry import MarketAwareRetryHandler, MarketPhase
from unittest.mock import patch

class TestMarketAwareRetryHandler:
    @pytest.fixture
    def handler(self):
        return MarketAwareRetryHandler()

    @pytest.mark.parametrize("test_time,expected_phase", [
        ("2024-01-01 00:00:00", MarketPhase.CLOSED),  # 元旦假期
        ("2024-01-02 09:00:00", MarketPhase.PRE_OPEN),  # 周二开盘前
        ("2024-01-02 09:30:00", MarketPhase.MORNING),  # 上午交易
        ("2024-01-02 11:30:00", MarketPhase.LUNCH_BREAK),  # 午间休市
        ("2024-01-02 13:00:00", MarketPhase.AFTERNOON),  # 下午交易
        ("2024-01-02 15:00:00", MarketPhase.CLOSED),  # 收盘后
        ("2024-01-06 10:00:00", MarketPhase.CLOSED),  # 周六
    ])
    def test_market_phases(self, handler, test_time, expected_phase):
        test_dt = datetime.strptime(test_time, "%Y-%m-%d %H:%M:%S")
        assert handler.get_market_phase(test_dt) == expected_phase

    def test_holiday_handling(self):
        # 自定义节假日
        holidays = [datetime(2024, 5, 1)]  # 劳动节
        handler = MarketAwareRetryHandler(holidays_calendar=holidays)

        test_dt = datetime(2024, 5, 1, 10, 0)  # 劳动节上午
        assert handler.get_market_phase(test_dt) == MarketPhase.CLOSED

    @pytest.mark.parametrize("test_time,expected_delay", [
        ("2024-01-02 09:00:00", 0),  # 开盘前立即重试
        ("2024-01-02 11:31:00", 89 * 60),  # 午间休市(89分钟)
        ("2024-01-02 15:01:00", 18*3600 + 29*60),  # 收盘后(次日9:30)
        ("2024-01-05 15:01:00", 3*24*3600 + 29*60),  # 周五收盘后(下周二9:30)
    ])
    def test_retry_delay_calculation(self, handler, test_time, expected_delay):
        test_dt = datetime.strptime(test_time, "%Y-%m-%d %H:%M:%S")
        delay = handler.get_retry_delay(test_dt)
        assert abs(delay - expected_delay) < 1  # 允许1秒误差

    def test_max_retry_attempts(self, handler):
        handler.max_retry_attempts = 2
        assert handler.should_retry()  # 第一次
        assert handler.should_retry()  # 第二次
        assert not handler.should_retry()  # 第三次

    def test_custom_trading_hours(self):
        custom_hours = {
            "pre_open": time(9, 0),
            "morning_open": time(9, 30),
            "morning_close": time(11, 30),
            "afternoon_open": time(13, 30),  # 下午1:30开盘
            "afternoon_close": time(15, 0)
        }
        handler = MarketAwareRetryHandler(trading_hours=custom_hours)

        test_dt = datetime(2024, 1, 2, 13, 0)
        assert handler.get_market_phase(test_dt) == MarketPhase.LUNCH_BREAK

class TestSmartOrderRetry:
    @pytest.fixture
    def retry_manager(self):
        handler = MarketAwareRetryHandler()
        return SmartOrderRetry(handler)

    def test_order_retry_flow(self, retry_manager):
        # 模拟首次失败
        with patch.object(retry_manager, '_try_execute', return_value={"success": False}):
            order = {"order_id": "test123"}
            result = retry_manager.submit_order(order)
            assert not result["success"]
            assert "test123" in retry_manager.pending_orders

        # 模拟重试成功
        with patch.object(retry_manager, '_try_execute', return_value={"success": True}):
            retry_manager.check_pending_orders()
            assert "test123" not in retry_manager.pending_orders

    def test_market_closed_handling(self, retry_manager):
        # 设置非交易时段
        retry_manager.retry_handler.get_market_phase = lambda dt: MarketPhase.CLOSED

        with patch.object(retry_manager, '_try_execute', return_value={"success": False}):
            order = {"order_id": "test456"}
            result = retry_manager.submit_order(order)
            assert not result["success"]
            assert "test456" in retry_manager.pending_orders
            # 验证延迟时间应大于0
            assert retry_manager.pending_orders["test456"]["next_retry"] > datetime.now()
