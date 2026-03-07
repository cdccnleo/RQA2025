#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层市场感知重试处理器组件综合测试

测试目标：提升utils/tools/market_aware_retry.py的真实覆盖率
实际导入和使用src.infrastructure.utils.tools.market_aware_retry模块
"""

import pytest
from datetime import datetime, time, timedelta
from unittest.mock import patch


class TestMarketPhase:
    """测试市场阶段枚举"""
    
    def test_market_phase_values(self):
        """测试市场阶段枚举值"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketPhase
        
        assert MarketPhase.PRE_OPEN is not None
        assert MarketPhase.MORNING is not None
        assert MarketPhase.LUNCH_BREAK is not None
        assert MarketPhase.AFTERNOON is not None
        assert MarketPhase.CLOSED is not None


class TestMarketAwareRetryHandler:
    """测试市场感知重试处理器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        
        assert handler.trading_hours is not None
        assert handler.base_retry_interval == 5
        assert handler.max_retry_attempts == 3
        assert handler.current_attempt == 0
    
    def test_init_custom_trading_hours(self):
        """测试使用自定义交易时间初始化"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        custom_hours = {
            "morning_open": time(9, 0),
            "morning_close": time(12, 0),
            "afternoon_open": time(13, 0),
            "afternoon_close": time(15, 0),
            "pre_open": time(8, 45)
        }
        
        handler = MarketAwareRetryHandler(trading_hours=custom_hours)
        
        assert handler.trading_hours == custom_hours
    
    def test_init_custom_holidays(self):
        """测试使用自定义节假日初始化"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        custom_holidays = [datetime(2024, 1, 1), datetime(2024, 10, 1)]
        
        handler = MarketAwareRetryHandler(holidays_calendar=custom_holidays)
        
        assert len(handler.holidays) == 2
    
    def test_get_market_phase_pre_open(self):
        """测试获取开盘前阶段"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler, MarketPhase
        
        handler = MarketAwareRetryHandler()
        
        # 模拟开盘前时间
        test_time = datetime(2024, 1, 2, 8, 0)  # 周二早上8点
        phase = handler.get_market_phase(test_time)
        
        assert phase == MarketPhase.PRE_OPEN
    
    def test_get_market_phase_morning(self):
        """测试获取上午交易阶段"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler, MarketPhase
        
        handler = MarketAwareRetryHandler()
        
        # 模拟上午交易时间
        test_time = datetime(2024, 1, 2, 10, 0)  # 周二上午10点
        phase = handler.get_market_phase(test_time)
        
        assert phase == MarketPhase.MORNING
    
    def test_get_market_phase_lunch_break(self):
        """测试获取午间休市阶段"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler, MarketPhase
        
        handler = MarketAwareRetryHandler()
        
        # 模拟午间休市时间
        test_time = datetime(2024, 1, 2, 12, 0)  # 周二中午12点
        phase = handler.get_market_phase(test_time)
        
        assert phase == MarketPhase.LUNCH_BREAK
    
    def test_get_market_phase_afternoon(self):
        """测试获取下午交易阶段"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler, MarketPhase
        
        handler = MarketAwareRetryHandler()
        
        # 模拟下午交易时间
        test_time = datetime(2024, 1, 2, 14, 0)  # 周二下午2点
        phase = handler.get_market_phase(test_time)
        
        assert phase == MarketPhase.AFTERNOON
    
    def test_get_market_phase_closed_weekend(self):
        """测试获取周末收盘阶段"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler, MarketPhase
        
        handler = MarketAwareRetryHandler()
        
        # 模拟周末时间
        test_time = datetime(2024, 1, 6, 10, 0)  # 周六上午10点
        phase = handler.get_market_phase(test_time)
        
        assert phase == MarketPhase.CLOSED
    
    def test_get_market_phase_closed_holiday(self):
        """测试获取节假日收盘阶段"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler, MarketPhase
        
        # 使用自定义节假日列表，确保测试日期是节假日
        custom_holidays = [datetime(2024, 1, 1).date()]  # 元旦
        handler = MarketAwareRetryHandler(holidays_calendar=[datetime(2024, 1, 1)])
        
        # 模拟节假日（元旦）
        test_time = datetime(2024, 1, 1, 10, 0)  # 元旦上午10点
        phase = handler.get_market_phase(test_time)
        
        assert phase == MarketPhase.CLOSED
    
    def test_get_market_phase_current_time(self):
        """测试获取当前时间的市场阶段"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        
        phase = handler.get_market_phase()
        
        assert phase is not None
    
    def test_is_market_open(self):
        """测试检查市场是否开市"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        
        # 测试上午交易时间
        test_time = datetime(2024, 1, 2, 10, 0)  # 周二上午10点
        assert handler.is_market_open(test_time) is True
        
        # 测试收盘后时间
        test_time = datetime(2024, 1, 2, 16, 0)  # 周二下午4点
        assert handler.is_market_open(test_time) is False
    
    def test_next_market_open_time(self):
        """测试计算下一个开市时间"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        
        # 测试开盘前
        test_time = datetime(2024, 1, 2, 8, 0)  # 周二早上8点
        next_open = handler.next_market_open_time(test_time)
        
        assert isinstance(next_open, datetime)
        assert next_open.hour == 9
        assert next_open.minute == 30
    
    def test_should_retry(self):
        """测试判断是否应该继续重试"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        
        assert handler.should_retry() is True
        assert handler.current_attempt == 1
        
        assert handler.should_retry() is True
        assert handler.current_attempt == 2
        
        assert handler.should_retry() is True
        assert handler.current_attempt == 3
        
        # 超过最大重试次数
        assert handler.should_retry() is False
    
    def test_get_retry_delay(self):
        """测试获取重试延迟时间"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        
        # 测试交易时间内的延迟
        test_time = datetime(2024, 1, 2, 10, 0)  # 周二上午10点
        delay = handler.get_retry_delay(test_time)
        
        assert isinstance(delay, float)
        assert delay >= 0
    
    def test_reset_attempts(self):
        """测试重置重试计数器"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        handler.current_attempt = 3
        
        handler.reset_attempts()
        
        assert handler.current_attempt == 0
    
    def test_register_holiday(self):
        """测试注册节假日"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        holiday = datetime(2024, 12, 25)
        
        handler.register_holiday(holiday)
        
        assert holiday.date() in handler.holidays
    
    def test_update_trading_hours(self):
        """测试更新交易时间配置"""
        from src.infrastructure.utils.tools.market_aware_retry import MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        new_hours = {"morning_open": time(9, 0)}
        
        handler.update_trading_hours(new_hours)
        
        assert handler.trading_hours["morning_open"] == time(9, 0)


class TestSmartOrderRetry:
    """测试智能订单重试管理器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.tools.market_aware_retry import SmartOrderRetry, MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        retry_manager = SmartOrderRetry(handler)
        
        assert retry_manager.retry_handler == handler
        assert isinstance(retry_manager.pending_orders, dict)
    
    def test_submit_order(self):
        """测试提交订单"""
        from src.infrastructure.utils.tools.market_aware_retry import SmartOrderRetry, MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        retry_manager = SmartOrderRetry(handler)
        
        order = {"order_id": "test123", "symbol": "AAPL", "quantity": 100}
        result = retry_manager.submit_order(order)
        
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_check_pending_orders(self):
        """测试检查待重试订单"""
        from src.infrastructure.utils.tools.market_aware_retry import SmartOrderRetry, MarketAwareRetryHandler
        
        handler = MarketAwareRetryHandler()
        retry_manager = SmartOrderRetry(handler)
        
        # 检查空订单列表
        retry_manager.check_pending_orders()
        
        assert True  # 如果没有抛出异常，说明功能正常

