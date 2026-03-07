#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层市场数据日志器组件测试

测试目标：提升utils/monitoring/market_data_logger.py的真实覆盖率
实际导入和使用src.infrastructure.utils.monitoring.market_data_logger模块
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch


class TestMarketDataDeduplicator:
    """测试市场数据去重处理器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator(window_size=5)
        
        assert deduplicator.window_size == 5
        assert isinstance(deduplicator.last_hashes, dict)
    
    def test_generate_hash(self):
        """测试生成行情数据指纹"""
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator()
        tick_data = {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000
        }
        
        hash1 = deduplicator._generate_hash(tick_data)
        hash2 = deduplicator._generate_hash(tick_data)
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 produces 64 hex characters
    
    def test_generate_hash_missing_fields(self):
        """测试缺少必要字段时生成哈希"""
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator()
        tick_data = {"incomplete": "data"}
        
        hash_value = deduplicator._generate_hash(tick_data)
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64
    
    def test_is_duplicate_same_data(self):
        """测试相同数据判断为重复"""
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator(window_size=10)
        tick_data = {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000
        }
        
        # 第一次不是重复
        assert deduplicator.is_duplicate(tick_data) is False
        
        # 立即再次检查应该是重复
        assert deduplicator.is_duplicate(tick_data) is True
    
    def test_is_duplicate_different_data(self):
        """测试不同数据判断为不重复"""
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator()
        tick_data1 = {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000
        }
        tick_data2 = {
            "symbol": "AAPL",
            "price": 151.0,
            "volume": 1000
        }
        
        assert deduplicator.is_duplicate(tick_data1) is False
        assert deduplicator.is_duplicate(tick_data2) is False
    
    def test_is_duplicate_expired_window(self):
        """测试时间窗口过期后不重复"""
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator(window_size=1)
        tick_data = {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000
        }
        
        deduplicator.is_duplicate(tick_data)
        
        # 模拟时间窗口过期
        with patch('src.infrastructure.utils.monitoring.market_data_logger.datetime') as mock_datetime:
            mock_now = datetime.now() + timedelta(seconds=2)
            mock_datetime.now.return_value = mock_now
            
            assert deduplicator.is_duplicate(tick_data) is False
    
    def test_is_duplicate_missing_symbol(self):
        """测试缺少symbol字段"""
        from src.infrastructure.utils.monitoring.market_data_logger import MarketDataDeduplicator
        
        deduplicator = MarketDataDeduplicator()
        tick_data = {"price": 150.0}
        
        result = deduplicator.is_duplicate(tick_data)
        
        assert result is False


class TestTradingHoursAwareCircuitBreaker:
    """测试交易时段感知熔断器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.monitoring.market_data_logger import TradingHoursAwareCircuitBreaker
        
        schedule = {
            'morning': {'start': '09:30', 'end': '11:30', 'threshold': 0.8},
            'afternoon': {'start': '13:00', 'end': '15:00', 'threshold': 0.7}
        }
        
        breaker = TradingHoursAwareCircuitBreaker(schedule)
        
        assert breaker.schedule == schedule
        assert breaker.current_threshold == 0.9
    
    def test_get_current_period(self):
        """测试获取当前交易时段"""
        from src.infrastructure.utils.monitoring.market_data_logger import TradingHoursAwareCircuitBreaker
        
        schedule = {
            'morning': {'start': '09:30', 'end': '11:30', 'threshold': 0.8}
        }
        
        breaker = TradingHoursAwareCircuitBreaker(schedule)
        
        period = breaker._get_current_period()
        
        # 结果取决于当前时间，但应该返回None或有效的时段名称
        assert period is None or period in schedule.keys()
    
    def test_should_trigger(self):
        """测试判断是否触发熔断"""
        from src.infrastructure.utils.monitoring.market_data_logger import TradingHoursAwareCircuitBreaker
        
        schedule = {
            'morning': {'start': '09:30', 'end': '11:30', 'threshold': 0.8}
        }
        
        breaker = TradingHoursAwareCircuitBreaker(schedule)
        
        # 测试高负载
        assert breaker.should_trigger(0.95) is True
        
        # 测试低负载
        assert breaker.should_trigger(0.5) is False

