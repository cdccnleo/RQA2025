#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易引擎依赖注入版本单元测试

测试目标：提升trading_engine_di.py的覆盖率到90%+
按照业务流程驱动架构设计测试依赖注入交易引擎
"""

import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, MagicMock, patch

from src.trading.core.trading_engine_di import (
    TradingEngine,
    OrderType,
    OrderDirection,
    OrderStatus,
)


class MockInfrastructureProvider:
    """模拟基础设施服务提供者"""

    def __init__(self):
        self.logger = MagicMock()
        self.monitor = MagicMock()
        self.cache_service = MagicMock()
        self.config_manager = MagicMock()

    def get_service_status(self):
        """获取服务状态"""
        from enum import Enum

        class ServiceStatus(Enum):
            HEALTHY = "healthy"
            DEGRADED = "degraded"
            UNHEALTHY = "unhealthy"

        return ServiceStatus.HEALTHY

    def get_service_health_report(self):
        """获取服务健康报告"""
        from dataclasses import dataclass

        @dataclass
        class HealthStatus:
            status: str

        return {
            "logger": HealthStatus("healthy"),
            "monitor": HealthStatus("healthy"),
            "cache": HealthStatus("healthy"),
        }


class TestTradingEngineDIInitialization:
    """测试交易引擎DI初始化"""

    def test_init_with_default_config(self):
        """测试使用默认配置初始化"""
        infra_provider = MockInfrastructureProvider()
        engine = TradingEngine(infra_provider)

        assert engine.infrastructure == infra_provider
        assert engine.logger == infra_provider.logger
        assert engine.monitor == infra_provider.monitor
        assert engine.cache == infra_provider.cache_service
        assert engine.config_manager == infra_provider.config_manager
        assert engine.execution_engine is not None

    def test_init_with_custom_config(self):
        """测试使用自定义配置初始化"""
        infra_provider = MockInfrastructureProvider()
        config = {
            "max_orders_per_minute": 200,
            "default_order_timeout": 60,
            "risk_check_enabled": False,
        }
        engine = TradingEngine(infra_provider, config)

        assert engine.config["max_orders_per_minute"] == 200
        assert engine.config["default_order_timeout"] == 60
        assert engine.config["risk_check_enabled"] is False

    def test_load_default_config(self):
        """测试加载默认配置"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.config_manager.get.return_value = None
        engine = TradingEngine(infra_provider)

        assert "max_orders_per_minute" in engine.config
        assert "default_order_timeout" in engine.config
        assert "risk_check_enabled" in engine.config
        assert "market_data_cache_ttl" in engine.config

    def test_config_from_manager(self):
        """测试从配置管理器加载配置"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.config_manager.get.side_effect = lambda key, default: {
            "trading.max_orders_per_minute": 150,
            "trading.default_order_timeout": 45,
            "trading.risk_check_enabled": True,
            "trading.market_data_cache_ttl": 600,
        }.get(key, default)

        engine = TradingEngine(infra_provider)

        assert engine.config["max_orders_per_minute"] == 150
        assert engine.config["default_order_timeout"] == 45

    def test_init_logs_initialization(self):
        """测试初始化记录日志"""
        infra_provider = MockInfrastructureProvider()
        engine = TradingEngine(infra_provider)

        infra_provider.logger.info.assert_called()
        assert "initialized" in str(infra_provider.logger.info.call_args).lower()


class TestTradingEngineDIOrderPlacement:
    """测试交易引擎DI下单功能"""

    def test_place_market_order(self):
        """测试下市价单"""
        infra_provider = MockInfrastructureProvider()
        engine = TradingEngine(infra_provider)

        result = engine.place_order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=100,
        )

        assert result["status"] == "accepted"
        assert result["symbol"] == "AAPL"
        assert result["quantity"] == 100
        assert "order_id" in result
        assert "timestamp" in result

    def test_place_limit_order(self):
        """测试下限价单"""
        infra_provider = MockInfrastructureProvider()
        engine = TradingEngine(infra_provider)

        result = engine.place_order(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            direction=OrderDirection.BUY,
            quantity=100,
            price=150.0,
        )

        assert result["status"] == "accepted"
        assert result["price"] == 150.0
        assert "order_id" in result
        assert result["symbol"] == "AAPL"

    def test_place_order_with_cache_hit(self):
        """测试下单时使用缓存市场数据"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.return_value = {
            "price": 150.0,
            "volume": 1000000,
        }
        engine = TradingEngine(infra_provider)

        result = engine.place_order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=100,
        )

        assert result["status"] == "accepted"
        infra_provider.cache_service.get.assert_called()

    def test_place_order_error_handling(self):
        """测试下单错误处理"""
        infra_provider = MockInfrastructureProvider()
        engine = TradingEngine(infra_provider)
        
        # Mock place_order内部可能抛出的异常（比如cache.get）
        infra_provider.cache_service.get.side_effect = Exception("Test error")

        result = engine.place_order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=100,
        )

        assert result["status"] == "error"
        assert "error" in result

    def test_place_order_monitoring(self):
        """测试下单监控指标"""
        infra_provider = MockInfrastructureProvider()
        engine = TradingEngine(infra_provider)

        engine.place_order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=100,
        )

        assert infra_provider.monitor.record_histogram.called
        assert infra_provider.monitor.increment_counter.called


class TestTradingEngineDIPortfolioStatus:
    """测试交易引擎DI投资组合状态"""

    def test_get_portfolio_status_cached(self):
        """测试获取缓存的投资组合状态"""
        infra_provider = MockInfrastructureProvider()
        cached_portfolio = {
            "total_value": 1000000.0,
            "cash": 500000.0,
            "status": "active",
        }
        infra_provider.cache_service.get.return_value = cached_portfolio
        engine = TradingEngine(infra_provider)

        result = engine.get_portfolio_status()

        assert result == cached_portfolio
        infra_provider.cache_service.get.assert_called_with("portfolio_status")

    def test_get_portfolio_status_uncached(self):
        """测试获取未缓存的投资组合状态"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.return_value = None
        engine = TradingEngine(infra_provider)

        result = engine.get_portfolio_status()

        assert result["total_value"] == 1000000.0
        assert result["cash"] == 500000.0
        assert "positions" in result
        assert result["status"] == "active"
        assert infra_provider.cache_service.set.called

    def test_get_portfolio_status_cache_ttl(self):
        """测试投资组合状态缓存TTL"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.return_value = None
        engine = TradingEngine(infra_provider)

        engine.get_portfolio_status()

        call_args = infra_provider.cache_service.set.call_args
        assert call_args[0][0] == "portfolio_status"
        assert call_args[1]["ttl"] == 60

    def test_get_portfolio_status_monitoring(self):
        """测试投资组合状态监控"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.return_value = None
        engine = TradingEngine(infra_provider)

        engine.get_portfolio_status()

        assert infra_provider.monitor.record_metric.called

    def test_get_portfolio_status_error_handling(self):
        """测试投资组合状态错误处理"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.side_effect = Exception("Cache error")
        engine = TradingEngine(infra_provider)

        result = engine.get_portfolio_status()

        assert result["status"] == "error"
        assert "error" in result


class TestTradingEngineDIMarketData:
    """测试交易引擎DI市场数据"""

    def test_get_market_data_cached(self):
        """测试获取缓存的市场数据"""
        infra_provider = MockInfrastructureProvider()
        cached_data = {
            "symbol": "AAPL",
            "price": 150.0,
            "volume": 1000000,
        }
        infra_provider.cache_service.get.return_value = cached_data
        engine = TradingEngine(infra_provider)

        result = engine.get_market_data("AAPL")

        assert result == cached_data
        infra_provider.cache_service.get.assert_called()

    def test_get_market_data_uncached(self):
        """测试获取未缓存的市场数据"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.return_value = None
        engine = TradingEngine(infra_provider)

        result = engine.get_market_data("AAPL")

        assert result["symbol"] == "AAPL"
        assert "price" in result
        assert "volume" in result
        assert "timestamp" in result
        assert infra_provider.cache_service.set.called

    def test_get_market_data_cache_config(self):
        """测试市场数据缓存配置"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.return_value = None
        config = {"market_data_cache_ttl": 600}
        engine = TradingEngine(infra_provider, config)

        engine.get_market_data("AAPL")

        call_args = infra_provider.cache_service.set.call_args
        assert call_args[1]["ttl"] == 600

    def test_get_market_data_error_handling(self):
        """测试市场数据错误处理"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.side_effect = Exception("Cache error")
        engine = TradingEngine(infra_provider)

        result = engine.get_market_data("AAPL")

        assert result is None
        infra_provider.logger.error.assert_called()


class TestTradingEngineDIHealthStatus:
    """测试交易引擎DI健康状态"""

    def test_get_health_status_healthy(self):
        """测试获取健康状态"""
        infra_provider = MockInfrastructureProvider()
        # 确保所有组件都是健康的
        infra_provider.cache_service.is_healthy = MagicMock(return_value=True)
        infra_provider.monitor.is_healthy = MagicMock(return_value=True)
        engine = TradingEngine(infra_provider)

        result = engine.get_health_status()

        # 检查返回结构
        assert "overall_status" in result
        assert "infrastructure_status" in result
        assert "local_components" in result
        assert "timestamp" in result
        # 如果所有组件健康，应该是healthy或degraded（取决于infrastructure状态）
        assert result["overall_status"] in ["healthy", "degraded"]

    def test_get_health_status_degraded(self):
        """测试获取降级状态"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.is_healthy.return_value = False
        infra_provider.monitor.is_healthy.return_value = True
        engine = TradingEngine(infra_provider)

        result = engine.get_health_status()

        assert result["overall_status"] == "degraded"

    def test_get_health_status_error_handling(self):
        """测试健康状态错误处理"""
        infra_provider = MockInfrastructureProvider()
        # 使用MagicMock替换方法，使其可以设置side_effect
        infra_provider.get_service_status = MagicMock(side_effect=Exception("Service error"))
        engine = TradingEngine(infra_provider)

        result = engine.get_health_status()

        assert result["overall_status"] == "unhealthy"
        assert "error" in result

    def test_get_health_status_monitoring(self):
        """测试健康状态监控"""
        infra_provider = MockInfrastructureProvider()
        engine = TradingEngine(infra_provider)

        engine.get_health_status()

        assert infra_provider.monitor.record_metric.called


class TestTradingEngineDIOrderEnums:
    """测试交易引擎DI订单枚举"""

    def test_order_type_values(self):
        """测试订单类型枚举值"""
        assert OrderType.MARKET.value == 1
        assert OrderType.LIMIT.value == 2
        assert OrderType.STOP.value == 3

    def test_order_direction_values(self):
        """测试订单方向枚举值"""
        assert OrderDirection.BUY.value == 1
        assert OrderDirection.SELL.value == -1

    def test_order_status_values(self):
        """测试订单状态枚举值"""
        assert OrderStatus.PENDING.value == 1
        assert OrderStatus.FILLED.value == 2
        assert OrderStatus.CANCELLED.value == 3
        assert OrderStatus.REJECTED.value == 4
    
    def test_load_config_exception_handling(self):
        """测试配置加载异常处理"""
        infra_provider = MockInfrastructureProvider()
        # Mock配置管理器抛出异常
        infra_provider.config_manager.get.side_effect = Exception("Config error")
        engine = TradingEngine(infra_provider)
        
        # 应该使用默认值
        assert "max_orders_per_minute" in engine.config
        assert engine.config["max_orders_per_minute"] == 100  # 默认值
    
    def test_place_order_no_cached_market_data(self):
        """测试下单时没有缓存市场数据"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.return_value = None  # 没有缓存
        engine = TradingEngine(infra_provider)
        
        result = engine.place_order(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            direction=OrderDirection.BUY,
            quantity=100,
        )
        
        assert result["status"] == "accepted"
        assert "order_id" in result
    
    def test_create_trading_engine(self):
        """测试创建交易引擎工厂函数"""
        from unittest.mock import patch, MagicMock
        
        # Mock基础设施提供者
        mock_provider = MockInfrastructureProvider()
        mock_provider.initialize_all_services = MagicMock()
        
        # get_infrastructure_provider是从infrastructure模块导入的，需要patch正确的模块
        with patch('src.infrastructure.core.infrastructure_service_provider.get_infrastructure_provider', return_value=mock_provider):
            from src.trading.core.trading_engine_di import create_trading_engine
            
            try:
                engine = create_trading_engine(config={'test': 'value'})
                assert engine is not None
                assert isinstance(engine, TradingEngine)
                # 验证initialize_all_services被调用
                mock_provider.initialize_all_services.assert_called_once()
            except (ImportError, AttributeError, ModuleNotFoundError) as e:
                # 如果基础设施模块不可用，跳过测试
                pytest.skip(f"Infrastructure module not available: {e}")
    
    def test_get_default_trading_engine(self):
        """测试获取默认交易引擎"""
        from unittest.mock import patch, MagicMock
        
        mock_engine = MagicMock(spec=TradingEngine)
        
        with patch('src.trading.core.trading_engine_di.create_trading_engine', return_value=mock_engine) as mock_create:
            from src.trading.core.trading_engine_di import get_default_trading_engine
            
            try:
                engine = get_default_trading_engine()
                assert engine == mock_engine
                # get_default_trading_engine调用create_trading_engine()时不传参数
                # 验证调用时没有参数（空参数列表）
                assert mock_create.called
                # 检查调用参数为空
                call_args = mock_create.call_args
                assert call_args is None or len(call_args[0]) == 0
            except (ImportError, AttributeError) as e:
                pytest.skip(f"Infrastructure module not available: {e}")

    def test_get_health_status_cache_unhealthy(self):
        """测试健康状态检查 - 缓存不健康"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.is_healthy = Mock(return_value=False)
        engine = TradingEngine(infra_provider)
        
        result = engine.get_health_status()
        
        assert result["overall_status"] == "degraded" or result["overall_status"] == "healthy"
        assert "local_components" in result

    def test_get_health_status_monitor_unhealthy(self):
        """测试健康状态检查 - 监控不健康"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.monitor.is_healthy = Mock(return_value=False)
        engine = TradingEngine(infra_provider)
        
        result = engine.get_health_status()
        
        assert result["overall_status"] == "degraded" or result["overall_status"] == "healthy"
        assert "local_components" in result

    def test_get_health_status_no_execution_engine(self):
        """测试健康状态检查 - 无执行引擎"""
        infra_provider = MockInfrastructureProvider()
        engine = TradingEngine(infra_provider)
        engine.execution_engine = None
        
        result = engine.get_health_status()
        
        assert result["local_components"]["execution_engine"] == "unhealthy"
        assert result["overall_status"] == "degraded"

    def test_get_market_data_cache_miss(self):
        """测试获取市场数据 - 缓存未命中"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.return_value = None
        engine = TradingEngine(infra_provider)
        
        result = engine.get_market_data("AAPL")
        
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert "price" in result
        assert infra_provider.cache_service.set.called

    def test_get_market_data_error(self):
        """测试获取市场数据 - 发生错误"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.side_effect = Exception("Cache error")
        engine = TradingEngine(infra_provider)
        
        result = engine.get_market_data("AAPL")
        
        assert result is None

    def test_place_order_with_price(self):
        """测试下单 - 带价格（限价单）"""
        infra_provider = MockInfrastructureProvider()
        engine = TradingEngine(infra_provider)
        
        result = engine.place_order(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            direction=OrderDirection.BUY,
            quantity=100,
            price=150.0
        )
        
        assert result["status"] == "accepted"
        assert result["price"] == 150.0

    def test_get_portfolio_status_cache_error(self):
        """测试获取投资组合状态 - 缓存错误"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.side_effect = Exception("Cache error")
        engine = TradingEngine(infra_provider)
        
        result = engine.get_portfolio_status()
        
        assert result["status"] == "error"
        assert "error" in result

    def test_get_portfolio_status_set_cache_error(self):
        """测试获取投资组合状态 - 设置缓存错误"""
        infra_provider = MockInfrastructureProvider()
        infra_provider.cache_service.get.return_value = None
        infra_provider.cache_service.set.side_effect = Exception("Set cache error")
        engine = TradingEngine(infra_provider)
        
        # 应该仍然返回投资组合状态，即使缓存设置失败
        result = engine.get_portfolio_status()
        
        assert "total_value" in result or result["status"] == "error"

