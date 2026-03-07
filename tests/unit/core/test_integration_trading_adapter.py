#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试交易层适配器

测试目标：提升integration/adapters/trading_adapter.py的覆盖率到100%
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from src.core.integration.adapters.trading_adapter import (
    TradingLayerAdapter,
    TradingInfrastructureBridge
)
from src.core.integration.adapters.business_adapters import BusinessLayerType


class TestTradingInfrastructureBridge:
    """测试交易基础设施桥接器"""

    @pytest.fixture
    def bridge(self):
        """创建基础设施桥接器实例"""
        return TradingInfrastructureBridge()

    def test_bridge_initialization(self, bridge):
        """测试桥接器初始化"""
        assert hasattr(bridge, '_bridge_id')
        assert hasattr(bridge, '_created_at')
        assert hasattr(bridge, '_services')
        assert isinstance(bridge._services, dict)

    def test_get_service(self, bridge):
        """测试获取服务"""
        # 默认情况下服务字典为空
        result = bridge.get_service("nonexistent_service")
        assert result is None

    def test_register_service(self, bridge):
        """测试注册服务"""
        mock_service = Mock()
        bridge.register_service("test_service", mock_service)

        assert "test_service" in bridge._services
        assert bridge._services["test_service"] == mock_service

    def test_get_registered_service(self, bridge):
        """测试获取已注册的服务"""
        mock_service = Mock()
        service_name = "registered_service"

        bridge.register_service(service_name, mock_service)
        result = bridge.get_service(service_name)

        assert result == mock_service

    def test_get_service_info(self, bridge):
        """测试获取服务信息"""
        info = bridge.get_service_info()

        assert isinstance(info, dict)
        assert "bridge_id" in info
        assert "created_at" in info
        assert "service_count" in info
        assert "services" in info

    def test_bridge_health_check(self, bridge):
        """测试桥接器健康检查"""
        health = bridge.health_check()

        assert isinstance(health, dict)
        assert "status" in health
        assert "services" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_bridge_metrics(self, bridge):
        """测试桥接器指标"""
        metrics = bridge.get_metrics()

        assert isinstance(metrics, dict)
        assert "total_services" in metrics
        assert "active_services" in metrics
        assert "bridge_uptime" in metrics


class TestTradingLayerAdapter:
    """测试交易层适配器"""

    @pytest.fixture
    def adapter(self):
        """创建交易层适配器实例"""
        return TradingLayerAdapter()

    def test_adapter_initialization(self, adapter):
        """测试适配器初始化"""
        assert adapter.layer_type == BusinessLayerType.TRADING
        assert hasattr(adapter, '_service_bridges')
        assert isinstance(adapter._service_bridges, dict)

    def test_trading_bridge_creation(self, adapter):
        """测试交易桥接器创建"""
        assert "trading_infrastructure_bridge" in adapter._service_bridges
        bridge = adapter._service_bridges["trading_infrastructure_bridge"]
        assert isinstance(bridge, TradingInfrastructureBridge)

    @patch('src.core.integration.adapters.trading_adapter.TradingInfrastructureBridge')
    def test_trading_bridge_creation_failure_handling(self, mock_bridge_class, adapter):
        """测试交易桥接器创建失败处理"""
        mock_bridge_class.side_effect = Exception("Bridge creation failed")

        # 重新初始化以触发桥接器创建
        adapter._init_trading_specific_services()

        # 即使桥接器创建失败，服务桥接器字典也应该存在
        assert hasattr(adapter, '_service_bridges')

    def test_get_trading_engine(self, adapter):
        """测试获取交易引擎"""
        with patch.object(adapter, '_get_service_from_bridge') as mock_get_service:
            mock_engine = Mock()
            mock_get_service.return_value = mock_engine

            result = adapter.get_trading_engine()

            assert result == mock_engine
            mock_get_service.assert_called_with("trading_engine")

    def test_get_trading_engine_fallback(self, adapter):
        """测试交易引擎获取失败时的回退"""
        with patch.object(adapter, '_get_service_from_bridge', return_value=None):
            with patch.object(adapter, '_create_fallback_trading_engine') as mock_fallback:
                mock_fallback_engine = Mock()
                mock_fallback.return_value = mock_fallback_engine

                result = adapter.get_trading_engine()

                assert result == mock_fallback_engine
                mock_fallback.assert_called_once()

    def test_get_order_service(self, adapter):
        """测试获取订单服务"""
        with patch.object(adapter, '_get_service_from_bridge') as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service

            result = adapter.get_order_service()

            assert result == mock_service
            mock_get_service.assert_called_with("order_service")

    def test_get_portfolio_service(self, adapter):
        """测试获取投资组合服务"""
        with patch.object(adapter, '_get_service_from_bridge') as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service

            result = adapter.get_portfolio_service()

            assert result == mock_service
            mock_get_service.assert_called_with("portfolio_service")

    def test_get_risk_service(self, adapter):
        """测试获取风险服务"""
        with patch.object(adapter, '_get_service_from_bridge') as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service

            result = adapter.get_risk_service()

            assert result == mock_service
            mock_get_service.assert_called_with("risk_service")

    def test_get_market_data_service(self, adapter):
        """测试获取市场数据服务"""
        with patch.object(adapter, '_get_service_from_bridge') as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service

            result = adapter.get_market_data_service()

            assert result == mock_service
            mock_get_service.assert_called_with("market_data_service")

    def test_get_strategy_service(self, adapter):
        """测试获取策略服务"""
        with patch.object(adapter, '_get_service_from_bridge') as mock_get_service:
            mock_service = Mock()
            mock_get_service.return_value = mock_service

            result = adapter.get_strategy_service()

            assert result == mock_service
            mock_get_service.assert_called_with("strategy_service")

    def test_execute_trade(self, adapter):
        """测试执行交易"""
        trade_params = {
            "symbol": "000001",
            "quantity": 100,
            "price": 10.5,
            "direction": "BUY"
        }

        with patch.object(adapter, 'get_trading_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.execute_trade.return_value = {"order_id": "12345", "status": "success"}
            mock_get_engine.return_value = mock_engine

            result = adapter.execute_trade(trade_params)

            assert result["order_id"] == "12345"
            assert result["status"] == "success"
            mock_engine.execute_trade.assert_called_once_with(trade_params)

    def test_execute_trade_engine_failure(self, adapter):
        """测试交易引擎失败的情况"""
        trade_params = {"symbol": "000001", "quantity": 100}

        with patch.object(adapter, 'get_trading_engine', return_value=None):
            result = adapter.execute_trade(trade_params)

            assert result["status"] == "error"
            assert "error" in result

    def test_get_portfolio_status(self, adapter):
        """测试获取投资组合状态"""
        with patch.object(adapter, 'get_portfolio_service') as mock_get_service:
            mock_service = Mock()
            mock_service.get_portfolio_status.return_value = {"total_value": 100000, "pnl": 1500}
            mock_get_service.return_value = mock_service

            result = adapter.get_portfolio_status()

            assert result["total_value"] == 100000
            assert result["pnl"] == 1500
            mock_service.get_portfolio_status.assert_called_once()

    def test_get_portfolio_status_service_failure(self, adapter):
        """测试投资组合服务失败的情况"""
        with patch.object(adapter, 'get_portfolio_service', return_value=None):
            result = adapter.get_portfolio_status()

            assert result["status"] == "error"
            assert "error" in result

    def test_get_market_data(self, adapter):
        """测试获取市场数据"""
        symbol = "000001"

        with patch.object(adapter, 'get_market_data_service') as mock_get_service:
            mock_service = Mock()
            mock_service.get_market_data.return_value = {"price": 10.5, "volume": 1000}
            mock_get_service.return_value = mock_service

            result = adapter.get_market_data(symbol)

            assert result["price"] == 10.5
            assert result["volume"] == 1000
            mock_service.get_market_data.assert_called_once_with(symbol)

    def test_get_market_data_service_failure(self, adapter):
        """测试市场数据服务失败的情况"""
        symbol = "000001"

        with patch.object(adapter, 'get_market_data_service', return_value=None):
            result = adapter.get_market_data(symbol)

            assert result["status"] == "error"
            assert "error" in result

    def test_get_risk_metrics(self, adapter):
        """测试获取风险指标"""
        with patch.object(adapter, 'get_risk_service') as mock_get_service:
            mock_service = Mock()
            mock_service.get_risk_metrics.return_value = {"var": 0.05, "sharpe_ratio": 1.2}
            mock_get_service.return_value = mock_service

            result = adapter.get_risk_metrics()

            assert result["var"] == 0.05
            assert result["sharpe_ratio"] == 1.2
            mock_service.get_risk_metrics.assert_called_once()

    def test_get_risk_metrics_service_failure(self, adapter):
        """测试风险服务失败的情况"""
        with patch.object(adapter, 'get_risk_service', return_value=None):
            result = adapter.get_risk_metrics()

            assert result["status"] == "error"
            assert "error" in result

    def test_run_strategy(self, adapter):
        """测试运行策略"""
        strategy_config = {"strategy_type": "momentum", "parameters": {"period": 20}}

        with patch.object(adapter, 'get_strategy_service') as mock_get_service:
            mock_service = Mock()
            mock_service.run_strategy.return_value = {"strategy_id": "strat_001", "status": "running"}
            mock_get_service.return_value = mock_service

            result = adapter.run_strategy(strategy_config)

            assert result["strategy_id"] == "strat_001"
            assert result["status"] == "running"
            mock_service.run_strategy.assert_called_once_with(strategy_config)

    def test_run_strategy_service_failure(self, adapter):
        """测试策略服务失败的情况"""
        strategy_config = {"strategy_type": "momentum"}

        with patch.object(adapter, 'get_strategy_service', return_value=None):
            result = adapter.run_strategy(strategy_config)

            assert result["status"] == "error"
            assert "error" in result

    def test_adapter_health_check(self, adapter):
        """测试适配器健康检查"""
        health = adapter.health_check()

        assert isinstance(health, dict)
        assert "status" in health
        assert "layer_type" in health
        assert "services" in health
        assert health["layer_type"] == BusinessLayerType.TRADING

    def test_adapter_get_metrics(self, adapter):
        """测试适配器指标获取"""
        metrics = adapter.get_metrics()

        assert isinstance(metrics, dict)
        assert "layer_type" in metrics
        assert "service_count" in metrics
        assert "bridge_count" in metrics
        assert metrics["layer_type"] == BusinessLayerType.TRADING

    def test_get_service_from_bridge(self, adapter):
        """测试从桥接器获取服务"""
        service_name = "test_service"
        mock_service = Mock()

        # 手动设置桥接器中的服务
        adapter._service_bridges["trading_infrastructure_bridge"].register_service(service_name, mock_service)

        result = adapter._get_service_from_bridge(service_name)

        assert result == mock_service

    def test_get_service_from_bridge_not_found(self, adapter):
        """测试从桥接器获取不存在的服务"""
        result = adapter._get_service_from_bridge("nonexistent_service")

        assert result is None

    def test_create_fallback_trading_engine(self, adapter):
        """测试创建回退交易引擎"""
        fallback_engine = adapter._create_fallback_trading_engine()

        assert fallback_engine is not None
        assert hasattr(fallback_engine, 'execute_trade')

    def test_create_fallback_services(self, adapter):
        """测试创建回退服务"""
        # 测试各种回退服务创建
        services = ["order", "portfolio", "risk", "market_data", "strategy"]

        for service_type in services:
            method_name = f"_create_fallback_{service_type}_service"
            if hasattr(adapter, method_name):
                fallback_service = getattr(adapter, method_name)()
                assert fallback_service is not None


class TestTradingAdapterIntegration:
    """测试交易适配器集成场景"""

    @pytest.fixture
    def adapter(self):
        """创建完整的交易适配器"""
        return TradingLayerAdapter()

    def test_complete_trading_workflow(self, adapter):
        """测试完整的交易工作流程"""
        # 1. 执行交易
        trade_params = {
            "symbol": "000001",
            "quantity": 100,
            "price": 10.5,
            "direction": "BUY"
        }

        with patch.object(adapter, 'get_trading_engine') as mock_get_engine:
            mock_engine = Mock()
            mock_engine.execute_trade.return_value = {
                "order_id": "order_001",
                "status": "filled",
                "executed_quantity": 100,
                "executed_price": 10.5
            }
            mock_get_engine.return_value = mock_engine

            trade_result = adapter.execute_trade(trade_params)

            assert trade_result["order_id"] == "order_001"
            assert trade_result["status"] == "filled"

        # 2. 获取投资组合状态
        with patch.object(adapter, 'get_portfolio_service') as mock_get_portfolio:
            mock_portfolio = Mock()
            mock_portfolio.get_portfolio_status.return_value = {
                "total_value": 105000,
                "cash": 95000,
                "positions": {"000001": {"quantity": 100, "avg_price": 10.5}}
            }
            mock_get_portfolio.return_value = mock_portfolio

            portfolio_result = adapter.get_portfolio_status()

            assert portfolio_result["total_value"] == 105000
            assert "000001" in portfolio_result["positions"]

        # 3. 获取市场数据
        with patch.object(adapter, 'get_market_data_service') as mock_get_market_data:
            mock_market_data = Mock()
            mock_market_data.get_market_data.return_value = {
                "symbol": "000001",
                "price": 10.8,
                "change": 0.3,
                "volume": 50000
            }
            mock_get_market_data.return_value = mock_market_data

            market_result = adapter.get_market_data("000001")

            assert market_result["symbol"] == "000001"
            assert market_result["price"] == 10.8

        # 4. 获取风险指标
        with patch.object(adapter, 'get_risk_service') as mock_get_risk:
            mock_risk = Mock()
            mock_risk.get_risk_metrics.return_value = {
                "portfolio_var": 0.03,
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.08
            }
            mock_get_risk.return_value = mock_risk

            risk_result = adapter.get_risk_metrics()

            assert risk_result["portfolio_var"] == 0.03
            assert risk_result["sharpe_ratio"] == 1.5

    def test_error_handling_and_recovery(self, adapter):
        """测试错误处理和恢复"""
        # 1. 模拟服务不可用的情况
        with patch.object(adapter, '_get_service_from_bridge', return_value=None):
            # 所有服务调用都应该返回错误但不抛出异常
            trade_result = adapter.execute_trade({"symbol": "000001"})
            assert trade_result["status"] == "error"

            portfolio_result = adapter.get_portfolio_status()
            assert portfolio_result["status"] == "error"

            market_result = adapter.get_market_data("000001")
            assert market_result["status"] == "error"

            risk_result = adapter.get_risk_metrics()
            assert risk_result["status"] == "error"

            strategy_result = adapter.run_strategy({"strategy_type": "test"})
            assert strategy_result["status"] == "error"

        # 2. 验证适配器仍然健康
        health = adapter.health_check()
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

        # 3. 验证指标仍然可用
        metrics = adapter.get_metrics()
        assert isinstance(metrics, dict)
        assert "service_count" in metrics

    def test_concurrent_service_access(self, adapter):
        """测试并发服务访问"""
        import threading
        import time

        results = []
        errors = []

        def access_service(service_method, *args):
            try:
                result = getattr(adapter, service_method)(*args)
                results.append(f"{service_method}_success")
            except Exception as e:
                errors.append(f"{service_method}_error: {str(e)}")

        # 创建多个线程并发访问不同服务
        threads = []
        service_calls = [
            ("get_portfolio_status",),
            ("get_risk_metrics",),
            ("get_market_data", "000001"),
            ("execute_trade", {"symbol": "000002", "quantity": 50}),
            ("run_strategy", {"strategy_type": "test"})
        ]

        for service_call in service_calls:
            thread = threading.Thread(target=access_service, args=service_call)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=5)

        # 验证没有发生错误（即使服务不可用，也应该优雅处理）
        assert len(errors) == 0

        # 验证所有调用都返回了结果
        assert len(results) == len(service_calls)

    def test_service_bridge_management(self, adapter):
        """测试服务桥接器管理"""
        # 验证桥接器存在
        assert "trading_infrastructure_bridge" in adapter._service_bridges

        bridge = adapter._service_bridges["trading_infrastructure_bridge"]

        # 注册一些测试服务
        test_services = {
            "cache_service": Mock(),
            "logging_service": Mock(),
            "monitoring_service": Mock()
        }

        for service_name, service_instance in test_services.items():
            bridge.register_service(service_name, service_instance)

        # 验证服务已注册
        for service_name in test_services.keys():
            retrieved_service = adapter._get_service_from_bridge(service_name)
            assert retrieved_service is not None
            assert retrieved_service == test_services[service_name]

        # 验证桥接器信息
        bridge_info = bridge.get_service_info()
        assert bridge_info["service_count"] == len(test_services)

    def test_adapter_performance_monitoring(self, adapter):
        """测试适配器性能监控"""
        import time

        start_time = time.time()

        # 执行多个服务调用
        operations = [
            lambda: adapter.get_portfolio_status(),
            lambda: adapter.get_risk_metrics(),
            lambda: adapter.get_market_data("000001"),
            lambda: adapter.execute_trade({"symbol": "000001", "quantity": 100}),
            lambda: adapter.run_strategy({"strategy_type": "momentum"})
        ]

        for operation in operations:
            operation()

        end_time = time.time()
        total_time = end_time - start_time

        # 验证在合理时间内完成
        assert total_time < 5.0  # 应该在5秒内完成所有操作

        # 验证健康检查仍然正常
        health = adapter.health_check()
        assert isinstance(health, dict)

        # 验证指标数据可用
        metrics = adapter.get_metrics()
        assert isinstance(metrics, dict)
        assert "service_count" in metrics

    def test_service_fallback_mechanisms(self, adapter):
        """测试服务回退机制"""
        # 1. 测试交易引擎回退
        fallback_engine = adapter._create_fallback_trading_engine()
        assert fallback_engine is not None

        # 验证回退引擎的基本功能
        result = fallback_engine.execute_trade({"symbol": "000001", "quantity": 100})
        assert isinstance(result, dict)
        assert "status" in result

        # 2. 测试其他服务的回退创建
        fallback_services = []

        # 尝试创建各种回退服务
        service_types = ["order", "portfolio", "risk", "market_data", "strategy"]
        for service_type in service_types:
            method_name = f"_create_fallback_{service_type}_service"
            if hasattr(adapter, method_name):
                service = getattr(adapter, method_name)()
                fallback_services.append(service)
                assert service is not None

        # 验证至少创建了一些回退服务
        assert len(fallback_services) > 0

    def test_adapter_resource_cleanup(self, adapter):
        """测试适配器资源清理"""
        # 注册一些服务到桥接器
        bridge = adapter._service_bridges["trading_infrastructure_bridge"]

        mock_services = [Mock() for _ in range(5)]
        for i, service in enumerate(mock_services):
            bridge.register_service(f"service_{i}", service)

        # 验证服务已注册
        assert bridge.get_service_info()["service_count"] == 5

        # 注意：当前实现可能没有显式的清理方法
        # 但适配器应该保持稳定状态
        health = adapter.health_check()
        assert isinstance(health, dict)

        metrics = adapter.get_metrics()
        assert isinstance(metrics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
