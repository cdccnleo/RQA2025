# -*- coding: utf-8 -*-
"""
统一异常处理框架测试
测试RQA2025系统的异常分类、处理和监控功能
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# 直接从模块导入，避免复杂的__init__.py依赖
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from src.core.foundation.exceptions.unified_exceptions import (
    RQA2025Exception,
    BusinessException,
    ValidationError,
    BusinessLogicError,
    WorkflowError,
    TradingError,
    RiskError,
    StrategyError,
    InfrastructureException,
    ConfigurationError,
    CacheError,
    LoggingError,
    MonitoringError,
    DatabaseError,
    NetworkError,
    ResourceError,
    FileSystemError,
    HealthCheckError,
    SystemException,
    SecurityError,
    PerformanceError,
    ConcurrencyError,
    AsyncError,
    ExternalServiceException,
    ThirdPartyAPIError,
    DataSourceError,
    ExceptionHandler,
    ExceptionHandlingStrategy,
    ExceptionMonitor,
    ExceptionLogger,
    ExceptionConfiguration,
    ExceptionStatistics
)


class TestRQA2025Exception:
    """RQA2025基础异常类测试"""

    def test_base_exception_creation(self):
        """测试基础异常创建"""
        exc = RQA2025Exception(
            message="Test error message",
            error_code=1001,
            error_type="TEST_ERROR",
            context={"user_id": 123, "operation": "test"},
            severity="ERROR"
        )

        assert str(exc) == "Test error message"
        assert exc.message == "Test error message"
        assert exc.error_code == 1001
        assert exc.error_type == "TEST_ERROR"
        assert exc.context == {"user_id": 123, "operation": "test"}
        assert exc.severity == "ERROR"
        assert isinstance(exc.timestamp, str)
        assert exc.stack_trace is not None

    def test_base_exception_defaults(self):
        """测试基础异常默认值"""
        exc = RQA2025Exception("Simple error")

        assert str(exc) == "Simple error"
        assert exc.error_code == -1
        assert exc.error_type == "UNKNOWN"
        assert exc.context == {}
        assert exc.severity == "ERROR"
        assert isinstance(exc.timestamp, str)

    def test_exception_to_dict(self):
        """测试异常转换为字典"""
        exc = RQA2025Exception(
            message="Dict test",
            error_code=2001,
            error_type="DICT_TEST",
            context={"key": "value"},
            severity="WARNING"
        )

        exc_dict = exc.to_dict()
        assert exc_dict["message"] == "Dict test"
        assert exc_dict["error_code"] == 2001
        assert exc_dict["error_type"] == "DICT_TEST"
        assert exc_dict["severity"] == "WARNING"
        assert "timestamp" in exc_dict
        assert "stack_trace" in exc_dict

    def test_exception_json_serialization(self):
        """测试异常JSON序列化"""
        exc = RQA2025Exception("JSON test", error_code=3001)

        json_str = exc.to_json()
        parsed = json.loads(json_str)

        assert parsed["message"] == "JSON test"
        assert parsed["error_code"] == 3001
        assert "timestamp" in parsed


class TestBusinessExceptions:
    """业务异常测试"""

    def test_validation_error(self):
        """测试验证错误"""
        exc = ValidationError(
            message="Invalid input data",
            field="email",
            value="invalid-email",
            validation_rule="email_format"
        )

        assert exc.error_type == "VALIDATION_ERROR"
        assert exc.field == "email"
        assert exc.value == "invalid-email"
        assert exc.validation_rule == "email_format"

    def test_business_logic_error(self):
        """测试业务逻辑错误"""
        exc = BusinessLogicError(
            message="Business rule violation",
            rule_name="max_trade_limit",
            current_value=100000,
            limit_value=50000
        )

        assert exc.error_type == "BUSINESS_LOGIC_ERROR"
        assert exc.rule_name == "max_trade_limit"
        assert exc.current_value == 100000
        assert exc.limit_value == 50000

    def test_trading_error(self):
        """测试交易错误"""
        exc = TradingError(
            message="Order execution failed",
            order_id="ORD_001",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            reason="INSUFFICIENT_FUNDS"
        )

        assert exc.error_type == "TRADING_ERROR"
        assert exc.order_id == "ORD_001"
        assert exc.symbol == "AAPL"
        assert exc.side == "BUY"
        assert exc.quantity == 100

    def test_risk_error(self):
        """测试风险错误"""
        exc = RiskError(
            message="Risk limit exceeded",
            risk_type="POSITION_SIZE",
            current_value=150000,
            limit_value=100000,
            threshold_percentage=50.0
        )

        assert exc.error_type == "RISK_ERROR"
        assert exc.risk_type == "POSITION_SIZE"
        assert exc.current_value == 150000
        assert exc.limit_value == 100000

    def test_strategy_error(self):
        """测试策略错误"""
        exc = StrategyError(
            message="Strategy execution error",
            strategy_id="STRAT_001",
            strategy_name="MeanReversion",
            error_phase="SIGNAL_GENERATION",
            retry_count=2
        )

        assert exc.error_type == "STRATEGY_ERROR"
        assert exc.strategy_id == "STRAT_001"
        assert exc.strategy_name == "MeanReversion"
        assert exc.error_phase == "SIGNAL_GENERATION"


class TestInfrastructureExceptions:
    """基础设施异常测试"""

    def test_configuration_error(self):
        """测试配置错误"""
        exc = ConfigurationError(
            message="Configuration file not found",
            config_file="/etc/config.json",
            config_key="database.host",
            expected_type="string"
        )

        assert exc.error_type == "CONFIGURATION_ERROR"
        assert exc.config_file == "/etc/config.json"
        assert exc.config_key == "database.host"

    def test_cache_error(self):
        """测试缓存错误"""
        exc = CacheError(
            message="Cache connection failed",
            cache_type="REDIS",
            operation="GET",
            key="user:123",
            error_details="Connection timeout"
        )

        assert exc.error_type == "CACHE_ERROR"
        assert exc.cache_type == "REDIS"
        assert exc.operation == "GET"
        assert exc.key == "user:123"

    def test_database_error(self):
        """测试数据库错误"""
        exc = DatabaseError(
            message="Query execution failed",
            database_type="POSTGRESQL",
            operation="SELECT",
            table_name="trades",
            query="SELECT * FROM trades WHERE id = $1",
            error_code="23505"
        )

        assert exc.error_type == "DATABASE_ERROR"
        assert exc.database_type == "POSTGRESQL"
        assert exc.operation == "SELECT"
        assert exc.table_name == "trades"

    def test_network_error(self):
        """测试网络错误"""
        exc = NetworkError(
            message="API call failed",
            url="https://api.example.com/trades",
            method="POST",
            status_code=500,
            response_time=30.5,
            timeout=10.0
        )

        assert exc.error_type == "NETWORK_ERROR"
        assert exc.url == "https://api.example.com/trades"
        assert exc.method == "POST"
        assert exc.status_code == 500

    def test_resource_error(self):
        """测试资源错误"""
        exc = ResourceError(
            message="Memory limit exceeded",
            resource_type="MEMORY",
            current_usage=90.5,
            limit=85.0,
            unit="PERCENT"
        )

        assert exc.error_type == "RESOURCE_ERROR"
        assert exc.resource_type == "MEMORY"
        assert exc.current_usage == 90.5
        assert exc.limit == 85.0


class TestSystemExceptions:
    """系统异常测试"""

    def test_security_error(self):
        """测试安全错误"""
        exc = SecurityError(
            message="Unauthorized access attempt",
            user_id="user123",
            resource="admin_panel",
            action="DELETE",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0"
        )

        assert exc.error_type == "SECURITY_ERROR"
        assert exc.user_id == "user123"
        assert exc.resource == "admin_panel"
        assert exc.action == "DELETE"

    def test_performance_error(self):
        """测试性能错误"""
        exc = PerformanceError(
            message="Response time exceeded threshold",
            operation="get_user_portfolio",
            response_time=5.2,
            threshold=2.0,
            concurrent_users=150,
            system_load=85.0
        )

        assert exc.error_type == "PERFORMANCE_ERROR"
        assert exc.operation == "get_user_portfolio"
        assert exc.response_time == 5.2
        assert exc.threshold == 2.0

    def test_concurrency_error(self):
        """测试并发错误"""
        exc = ConcurrencyError(
            message="Deadlock detected",
            operation="transfer_funds",
            lock_type="DATABASE_ROW_LOCK",
            wait_time=30.0,
            transaction_id="TXN_12345"
        )

        assert exc.error_type == "CONCURRENCY_ERROR"
        assert exc.operation == "transfer_funds"
        assert exc.lock_type == "DATABASE_ROW_LOCK"
        assert exc.wait_time == 30.0


class TestExternalServiceExceptions:
    """外部服务异常测试"""

    def test_third_party_api_error(self):
        """测试第三方API错误"""
        exc = ThirdPartyAPIError(
            message="External API rate limit exceeded",
            service_name="AlphaVantage",
            endpoint="/query",
            api_key_masked="AV_****_****_****_KEY",
            rate_limit=5,
            retry_after=60
        )

        assert exc.error_type == "THIRD_PARTY_API_ERROR"
        assert exc.service_name == "AlphaVantage"
        assert exc.endpoint == "/query"
        assert exc.rate_limit == 5
        assert exc.retry_after == 60

    def test_data_source_error(self):
        """测试数据源错误"""
        exc = DataSourceError(
            message="Market data feed disconnected",
            data_source="Bloomberg",
            feed_type="REAL_TIME",
            symbol="AAPL",
            last_update="2023-10-09T14:30:00Z"
        )

        assert exc.error_type == "DATA_SOURCE_ERROR"
        assert exc.data_source == "Bloomberg"
        assert exc.feed_type == "REAL_TIME"
        assert exc.symbol == "AAPL"


class TestExceptionHandler:
    """异常处理器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.handler = ExceptionHandler()

    def test_handler_initialization(self):
        """测试处理器初始化"""
        assert isinstance(self.handler.strategies, dict)
        assert len(self.handler.strategies) == 0

    def test_register_strategy(self):
        """测试注册策略"""
        strategy = Mock(spec=ExceptionHandlingStrategy)
        strategy.name = "test_strategy"

        self.handler.register_strategy("test", strategy)
        assert "test" in self.handler.strategies
        assert self.handler.strategies["test"] == strategy

    def test_handle_exception_with_strategy(self):
        """测试使用策略处理异常"""
        strategy = Mock(spec=ExceptionHandlingStrategy)
        strategy.can_handle.return_value = True
        strategy.handle.return_value = "handled"

        self.handler.register_strategy("test", strategy)

        exc = ValueError("Test error")
        result = self.handler.handle_exception(exc, {"context": "test"})

        assert result == "handled"
        strategy.can_handle.assert_called_once_with(exc)
        strategy.handle.assert_called_once_with(exc, {"context": "test"})

    def test_handle_exception_no_strategy(self):
        """测试无策略处理异常"""
        exc = ValueError("Test error")
        result = self.handler.handle_exception(exc, {})

        assert result is None


class TestExceptionHandlingStrategy:
    """异常处理策略测试"""

    def test_strategy_interface(self):
        """测试策略接口"""
        strategy = ExceptionHandlingStrategy("test_strategy")

        assert strategy.name == "test_strategy"
        assert hasattr(strategy, 'can_handle')
        assert hasattr(strategy, 'handle')

    def test_can_handle_default(self):
        """测试默认can_handle方法"""
        strategy = ExceptionHandlingStrategy("test")

        # 默认实现应该返回False
        assert strategy.can_handle(ValueError("test")) is False

    def test_handle_default(self):
        """测试默认handle方法"""
        strategy = ExceptionHandlingStrategy("test")

        # 默认实现应该抛出异常
        with pytest.raises(NotImplementedError):
            strategy.handle(ValueError("test"), {})


class TestExceptionMonitor:
    """异常监控器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.monitor = ExceptionMonitor()

    def test_monitor_initialization(self):
        """测试监控器初始化"""
        assert isinstance(self.monitor.statistics, ExceptionStatistics)
        assert isinstance(self.monitor.logger, ExceptionLogger)
        assert self.monitor.enabled is True

    def test_record_exception(self):
        """测试记录异常"""
        exc = ValidationError("Test validation error", field="email")

        initial_count = self.monitor.statistics.get_exception_count("ValidationError")
        self.monitor.record_exception(exc)

        new_count = self.monitor.statistics.get_exception_count("ValidationError")
        assert new_count == initial_count + 1

    def test_get_exception_summary(self):
        """测试获取异常摘要"""
        exc1 = ValidationError("Error 1", field="field1")
        exc2 = DatabaseError("Error 2", operation="SELECT")

        self.monitor.record_exception(exc1)
        self.monitor.record_exception(exc2)

        summary = self.monitor.get_exception_summary()
        assert "ValidationError" in summary
        assert "DatabaseError" in summary
        assert summary["ValidationError"] >= 1
        assert summary["DatabaseError"] >= 1

    @patch('src.core.foundation.exceptions.unified_exceptions.logger')
    def test_alert_on_threshold(self, mock_logger):
        """测试阈值告警"""
        # 配置告警阈值
        self.monitor.alert_thresholds = {"ValidationError": 2}

        # 记录异常直到触发告警
        for i in range(3):
            exc = ValidationError(f"Error {i}", field="test")
            self.monitor.record_exception(exc)

        # 验证告警被触发
        mock_logger.warning.assert_called()


class TestExceptionStatistics:
    """异常统计测试"""

    def setup_method(self):
        """设置测试方法"""
        self.stats = ExceptionStatistics()

    def test_statistics_initialization(self):
        """测试统计初始化"""
        assert isinstance(self.stats.exception_counts, dict)
        assert isinstance(self.stats.exception_types, set)
        assert isinstance(self.stats.timestamps, list)

    def test_record_exception(self):
        """测试记录异常"""
        exc = RQA2025Exception("Test error", error_type="TEST_TYPE")

        self.stats.record_exception(exc)

        assert self.stats.exception_counts["TEST_TYPE"] == 1
        assert "TEST_TYPE" in self.stats.exception_types
        assert len(self.stats.timestamps) == 1

    def test_get_exception_count(self):
        """测试获取异常计数"""
        exc1 = ValidationError("Error 1", field="field1")
        exc2 = ValidationError("Error 2", field="field2")

        self.stats.record_exception(exc1)
        self.stats.record_exception(exc2)

        count = self.stats.get_exception_count("ValidationError")
        assert count == 2

        # 测试不存在的异常类型
        count_none = self.stats.get_exception_count("NonExistent")
        assert count_none == 0

    def test_get_top_exceptions(self):
        """测试获取最常见异常"""
        # 记录不同类型的异常
        exceptions = [
            ValidationError("V1", field="f1"),
            ValidationError("V2", field="f2"),
            DatabaseError("D1", operation="SELECT"),
            DatabaseError("D2", operation="INSERT"),
            DatabaseError("D3", operation="UPDATE"),
            NetworkError("N1", url="http://test.com")
        ]

        for exc in exceptions:
            self.stats.record_exception(exc)

        top_exceptions = self.stats.get_top_exceptions(2)

        assert len(top_exceptions) == 2
        # DatabaseError应该排在第一位（3次）
        assert top_exceptions[0][0] == "DatabaseError"
        assert top_exceptions[0][1] == 3
        # ValidationError应该排在第二位（2次）
        assert top_exceptions[1][0] == "ValidationError"
        assert top_exceptions[1][1] == 2

    def test_clear_statistics(self):
        """测试清除统计"""
        exc = RQA2025Exception("Test", error_type="TEST")
        self.stats.record_exception(exc)

        assert self.stats.exception_counts["TEST"] == 1

        self.stats.clear()
        assert len(self.stats.exception_counts) == 0
        assert len(self.stats.exception_types) == 0
        assert len(self.stats.timestamps) == 0


class TestExceptionIntegration:
    """异常处理集成测试"""

    def test_complete_exception_workflow(self):
        """测试完整的异常处理工作流"""
        # 1. 创建异常
        exc = TradingError(
            message="Trade failed",
            order_id="ORD_001",
            symbol="AAPL",
            side="BUY",
            quantity=100
        )

        # 2. 创建处理器和策略
        handler = ExceptionHandler()
        strategy = Mock(spec=ExceptionHandlingStrategy)
        strategy.can_handle.return_value = True
        strategy.handle.return_value = "RECOVERED"

        handler.register_strategy("trading", strategy)

        # 3. 处理异常
        result = handler.handle_exception(exc, {"retry": True})

        assert result == "RECOVERED"
        strategy.can_handle.assert_called_once_with(exc)

    def test_exception_monitoring_workflow(self):
        """测试异常监控工作流"""
        monitor = ExceptionMonitor()

        # 记录多个异常
        exceptions = [
            ValidationError("Invalid email", field="email"),
            DatabaseError("Connection failed", operation="CONNECT"),
            NetworkError("Timeout", url="http://api.test.com"),
            ValidationError("Invalid phone", field="phone")
        ]

        for exc in exceptions:
            monitor.record_exception(exc)

        # 验证统计
        summary = monitor.get_exception_summary()
        assert summary["ValidationError"] == 2
        assert summary["DatabaseError"] == 1
        assert summary["NetworkError"] == 1

        # 验证详细统计
        stats = monitor.statistics
        assert stats.get_exception_count("ValidationError") == 2
        assert len(stats.exception_types) == 3

    def test_exception_serialization_workflow(self):
        """测试异常序列化工作流"""
        # 创建复杂异常
        exc = RiskError(
            message="Risk limit breached",
            risk_type="VAR_LIMIT",
            current_value=100000,
            limit_value=50000,
            threshold_percentage=100.0
        )

        # 序列化为JSON
        json_str = exc.to_json()
        parsed_data = json.loads(json_str)

        # 验证关键字段
        assert parsed_data["message"] == "Risk limit breached"
        assert parsed_data["error_type"] == "RISK_ERROR"
        assert parsed_data["risk_type"] == "VAR_LIMIT"
        assert parsed_data["current_value"] == 100000
        assert parsed_data["limit_value"] == 50000

        # 验证时间戳存在
        assert "timestamp" in parsed_data
        assert "stack_trace" in parsed_data

    def test_exception_inheritance_hierarchy(self):
        """测试异常继承层次"""
        # 测试不同层级的异常
        base_exc = RQA2025Exception("Base error")
        business_exc = BusinessException("Business error")
        trading_exc = TradingError("Trading error", order_id="ORD_001")
        infra_exc = InfrastructureException("Infra error")
        config_exc = ConfigurationError("Config error", config_key="db.host")

        # 验证继承关系
        assert isinstance(base_exc, Exception)
        assert isinstance(business_exc, RQA2025Exception)
        assert isinstance(trading_exc, BusinessException)
        assert isinstance(infra_exc, RQA2025Exception)
        assert isinstance(config_exc, InfrastructureException)

        # 验证错误类型
        assert base_exc.error_type == "UNKNOWN"
        assert business_exc.error_type == "BUSINESS_ERROR"
        assert trading_exc.error_type == "TRADING_ERROR"
        assert infra_exc.error_type == "INFRASTRUCTURE_ERROR"
        assert config_exc.error_type == "CONFIGURATION_ERROR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
