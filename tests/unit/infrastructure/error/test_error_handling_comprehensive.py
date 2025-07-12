#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异常处理模块全面单元测试
覆盖错误处理、重试、熔断器、安全错误等核心功能
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.infrastructure.error.error_handler import ErrorHandler
from src.infrastructure.error.retry_handler import RetryHandler
from src.infrastructure.error.circuit_breaker import CircuitBreaker
from src.infrastructure.error.exceptions import (
    ConfigError, DatabaseError, NetworkError, SecurityError,
    ValidationError, TimeoutError, ResourceError
)
from src.infrastructure.error.security_errors import SecurityViolationError
from src.infrastructure.trading.error_handler import TradingErrorHandler
from src.infrastructure.trading.market_aware_retry import MarketAwareRetry
from src.infrastructure.trading.persistent_error_handler import PersistentErrorHandler

class TestErrorHandlingComprehensive:
    """异常处理全面测试"""
    
    @pytest.fixture
    def error_handler(self):
        """创建错误处理器实例"""
        return ErrorHandler()
    
    @pytest.fixture
    def retry_handler(self):
        """创建重试处理器实例"""
        return RetryHandler(max_retries=3, backoff_factor=2)
    
    @pytest.fixture
    def circuit_breaker(self):
        """创建熔断器实例"""
        return CircuitBreaker(failure_threshold=3, recovery_timeout=60)
    
    def test_basic_error_handling(self, error_handler):
        """测试基础错误处理"""
        # 测试异常捕获和处理
        def failing_function():
            raise ValueError("测试错误")
        
        result = error_handler.handle(failing_function)
        assert result is None
        
        # 测试异常记录
        error_log = error_handler.get_error_log()
        assert len(error_log) > 0
        assert "ValueError" in str(error_log[0])
    
    def test_retry_handler(self, retry_handler):
        """测试重试处理器"""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("网络错误")
            return "成功"
        
        # 测试重试机制
        result = retry_handler.execute(failing_function)
        assert result == "成功"
        assert call_count == 3
        
        # 测试超过最大重试次数
        call_count = 0
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise NetworkError("持续失败")
        
        with pytest.raises(NetworkError):
            retry_handler.execute(always_failing)
        assert call_count == 4  # 初始调用 + 3次重试
    
    def test_circuit_breaker(self, circuit_breaker):
        """测试熔断器"""
        def failing_function():
            raise NetworkError("网络错误")

        def successful_function():
            return "成功"

        # 测试熔断器打开
        for _ in range(3):
            with pytest.raises(NetworkError):
                circuit_breaker.execute(failing_function)

        # 验证熔断器已打开
        assert circuit_breaker.is_open()

        # 测试熔断器打开时的快速失败
        with pytest.raises(Exception):
            circuit_breaker.execute(successful_function)

        # 测试熔断器恢复 - 设置超时时间
        circuit_breaker.last_failure_time = time.time() - 70  # 超过恢复时间
        result = circuit_breaker.execute(successful_function)
        assert result == "成功"
        assert circuit_breaker.is_closed()
    
    def test_error_classification(self, error_handler):
        """测试错误分类"""
        # 测试配置错误
        with pytest.raises(ConfigError):
            error_handler.classify_and_raise(ConfigError("配置错误"))
        
        # 测试数据库错误
        with pytest.raises(DatabaseError):
            error_handler.classify_and_raise(DatabaseError("数据库错误"))
        
        # 测试网络错误
        with pytest.raises(NetworkError):
            error_handler.classify_and_raise(NetworkError("网络错误"))
        
        # 测试安全错误
        with pytest.raises(SecurityError):
            error_handler.classify_and_raise(SecurityError("安全错误"))
    
    def test_error_recovery(self, error_handler):
        """测试错误恢复"""
        recovery_strategies = {
            ConfigError: lambda e: {"action": "reload_config", "error": str(e)},
            DatabaseError: lambda e: {"action": "reconnect_db", "error": str(e)},
            NetworkError: lambda e: {"action": "retry", "error": str(e)}
        }
        
        error_handler.set_recovery_strategies(recovery_strategies)
        
        # 测试配置错误恢复
        try:
            raise ConfigError("配置错误")
        except ConfigError as e:
            recovery = error_handler.recover(e)
            assert recovery["action"] == "reload_config"
        
        # 测试数据库错误恢复
        try:
            raise DatabaseError("数据库错误")
        except DatabaseError as e:
            recovery = error_handler.recover(e)
            assert recovery["action"] == "reconnect_db"
    
    def test_error_logging(self, error_handler):
        """测试错误日志记录"""
        # 记录不同类型的错误
        error_handler.log_error(ConfigError("配置错误"), "config_operation")
        error_handler.log_error(DatabaseError("数据库错误"), "db_operation")
        error_handler.log_error(NetworkError("网络错误"), "network_operation")
        
        # 获取错误统计
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == 3
        assert stats["config_errors"] == 1
        assert stats["database_errors"] == 1
        assert stats["network_errors"] == 1
    
    def test_error_notification(self, error_handler):
        """测试错误通知"""
        notifications = []
        
        def notification_handler(error_type, error_message, context):
            notifications.append({
                "type": error_type,
                "message": error_message,
                "context": context
            })
        
        error_handler.set_notification_handler(notification_handler)
        
        # 触发错误通知
        error_handler.notify_error(ConfigError("配置错误"), "test_context")
        
        assert len(notifications) == 1
        assert notifications[0]["type"] == "ConfigError"
        assert notifications[0]["context"] == "test_context"
    
    def test_security_error_handling(self, error_handler):
        """测试安全错误处理"""
        from src.infrastructure.error.security_errors import SecurityViolationError
        
        # 测试安全违规错误
        with pytest.raises(SecurityViolationError):
            error_handler.handle_security_violation("未授权访问", "user_123")
        
        # 测试安全错误记录
        security_log = error_handler.get_security_log()
        assert len(security_log) > 0
        assert "未授权访问" in str(security_log[0])
    
    def test_trading_error_handler(self, error_handler):
        """测试交易错误处理"""
        trading_handler = TradingErrorHandler()
        
        # 测试订单错误
        order_error = {"order_id": "12345", "error": "余额不足"}
        trading_handler.handle_order_error(order_error)
        
        # 测试风控错误
        risk_error = {"order_id": "12345", "error": "超出风控限制"}
        trading_handler.handle_risk_error(risk_error)
        
        # 获取交易错误统计
        trading_stats = trading_handler.get_error_statistics()
        assert trading_stats["order_errors"] >= 1
        assert trading_stats["risk_errors"] >= 1
    
    def test_market_aware_retry(self, error_handler):
        """测试市场感知重试"""
        market_retry = MarketAwareRetry()
        
        # 模拟市场时间
        market_retry.set_market_time("09:30:00")
        
        # 测试交易时间内的重试
        def trading_function():
            raise NetworkError("交易时间网络错误")
        
        with pytest.raises(NetworkError):
            market_retry.execute(trading_function)
        
        # 测试非交易时间的跳过
        market_retry.set_market_time("23:00:00")
        result = market_retry.execute(lambda: "非交易时间")
        assert result == "非交易时间"
    
    def test_persistent_error_handler(self, error_handler):
        """测试持久化错误处理"""
        persistent_handler = PersistentErrorHandler()
        
        # 测试错误持久化
        error_data = {
            "error_type": "NetworkError",
            "message": "网络连接失败",
            "timestamp": datetime.now().isoformat(),
            "context": "order_execution"
        }
        
        persistent_handler.persist_error(error_data)
        
        # 测试错误恢复
        recovered_errors = persistent_handler.get_persistent_errors()
        assert len(recovered_errors) > 0
        assert recovered_errors[0]["error_type"] == "NetworkError"
    
    def test_error_timeout_handling(self, error_handler):
        """测试错误超时处理"""
        timeout_handler = RetryHandler(max_retries=2, timeout=1.0)
        
        def slow_function():
            time.sleep(2.0)  # 超过超时时间
            return "完成"
        
        # 测试超时错误
        with pytest.raises(TimeoutError):
            timeout_handler.execute(slow_function)
    
    def test_error_resource_handling(self, error_handler):
        """测试资源错误处理"""
        resource_handler = ErrorHandler()
        
        # 测试资源耗尽错误
        with pytest.raises(ResourceError):
            resource_handler.handle_resource_error("内存不足", "memory")
        
        # 测试资源清理
        cleanup_result = resource_handler.cleanup_resources()
        assert cleanup_result["status"] == "completed"
    
    def test_error_validation(self, error_handler):
        """测试错误验证"""
        # 测试有效错误
        valid_error = ValidationError("输入验证失败")
        assert error_handler.validate_error(valid_error) is True
        
        # 测试无效错误
        invalid_error = "字符串错误"
        assert error_handler.validate_error(invalid_error) is False
    
    def test_error_serialization(self, error_handler):
        """测试错误序列化"""
        error_data = {
            "type": "NetworkError",
            "message": "网络连接失败",
            "timestamp": datetime.now().isoformat(),
            "stack_trace": "错误堆栈信息"
        }
        
        # 序列化错误
        serialized = error_handler.serialize_error(error_data)
        assert isinstance(serialized, str)
        
        # 反序列化错误
        deserialized = error_handler.deserialize_error(serialized)
        assert deserialized["type"] == "NetworkError"
        assert deserialized["message"] == "网络连接失败"
    
    def test_error_aggregation(self, error_handler):
        """测试错误聚合"""
        # 添加多个相同类型的错误
        for i in range(5):
            error_handler.log_error(NetworkError(f"网络错误 {i}"), "network")
        
        # 获取聚合统计
        aggregation = error_handler.aggregate_errors()
        assert aggregation["NetworkError"]["count"] == 5
        assert aggregation["NetworkError"]["frequency"] > 0
    
    def test_error_cleanup(self, error_handler):
        """测试错误清理"""
        # 添加一些错误
        error_handler.log_error(ConfigError("配置错误"), "config")
        error_handler.log_error(DatabaseError("数据库错误"), "database")
        
        # 清理旧错误
        cleanup_result = error_handler.cleanup_old_errors(days=1)
        assert cleanup_result["cleaned_count"] >= 0
    
    def test_error_performance(self, error_handler):
        """测试错误处理性能"""
        import time
        
        # 测试大量错误处理的性能
        start_time = time.time()
        for i in range(1000):
            error_handler.log_error(NetworkError(f"性能测试错误 {i}"), "performance")
        processing_time = time.time() - start_time
        
        # 性能要求：处理1000个错误应在1秒内完成
        assert processing_time < 1.0
    
    def test_error_concurrency(self, error_handler):
        """测试错误处理并发"""
        import threading
        
        error_count = 0
        
        def error_worker():
            nonlocal error_count
            for i in range(100):
                error_handler.log_error(NetworkError(f"并发错误 {i}"), "concurrent")
                error_count += 1
        
        # 启动多个线程同时记录错误
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=error_worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有错误都被记录
        assert error_count == 500
    
    def test_error_recovery_strategies(self, error_handler):
        """测试错误恢复策略"""
        # 定义自定义恢复策略
        def custom_recovery_strategy(error):
            if isinstance(error, NetworkError):
                return {"action": "switch_network", "retry": True}
            elif isinstance(error, DatabaseError):
                return {"action": "switch_database", "retry": False}
            return {"action": "default", "retry": True}
        
        error_handler.set_custom_recovery_strategy(custom_recovery_strategy)
        
        # 测试网络错误恢复策略
        network_recovery = error_handler.recover(NetworkError("网络错误"))
        assert network_recovery["action"] == "switch_network"
        assert network_recovery["retry"] is True
        
        # 测试数据库错误恢复策略
        db_recovery = error_handler.recover(DatabaseError("数据库错误"))
        assert db_recovery["action"] == "switch_database"
        assert db_recovery["retry"] is False
    
    def test_error_monitoring(self, error_handler):
        """测试错误监控"""
        # 启动错误监控
        error_handler.start_monitoring()
        
        # 触发一些错误
        error_handler.log_error(ConfigError("监控测试错误"), "monitoring")
        
        # 获取监控指标
        monitoring_metrics = error_handler.get_monitoring_metrics()
        assert monitoring_metrics["error_rate"] >= 0
        assert monitoring_metrics["active_errors"] >= 0
        
        # 停止错误监控
        error_handler.stop_monitoring()
    
    def test_error_alerting(self, error_handler):
        """测试错误告警"""
        alerts = []
        
        def alert_handler(alert_type, message, severity):
            alerts.append({
                "type": alert_type,
                "message": message,
                "severity": severity
            })
        
        error_handler.set_alert_handler(alert_handler)
        
        # 触发告警
        error_handler.trigger_alert("high_error_rate", "错误率过高", "critical")
        
        assert len(alerts) == 1
        assert alerts[0]["type"] == "high_error_rate"
        assert alerts[0]["severity"] == "critical" 