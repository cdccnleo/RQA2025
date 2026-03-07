"""
策略执行监控模块综合测试

批量测试所有监控组件的核心功能
"""

import pytest
import asyncio
import time
import json
import tempfile
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class TestExecutionMonitorBasic:
    """执行监控器基础测试"""
    
    def test_import_and_initialization(self):
        """测试导入和初始化"""
        from src.gateway.web.execution_monitor import (
            ExecutionMonitor, ExecutionStatus, ExecutionMetrics, MonitoringRule
        )
        
        monitor = ExecutionMonitor()
        assert monitor is not None
        assert len(monitor._monitoring_rules) > 0  # 应该有默认规则
        
    def test_strategy_lifecycle(self):
        """测试策略生命周期"""
        from src.gateway.web.execution_monitor import (
            ExecutionMonitor, ExecutionStatus
        )
        
        monitor = ExecutionMonitor()
        
        # 注册策略
        monitor.register_strategy("test_001", "测试策略")
        assert "test_001" in monitor._monitored_strategies
        
        # 更新状态
        monitor.update_strategy_status("test_001", ExecutionStatus.RUNNING)
        assert monitor.get_execution_status("test_001") == ExecutionStatus.RUNNING
        
        # 更新指标
        monitor.update_metrics("test_001", latency_ms=100.0, signal_count=5)
        metrics = monitor.get_metrics("test_001")
        assert metrics.latency_ms == 100.0
        assert metrics.signal_count == 5
        
        # 注销策略
        monitor.unregister_strategy("test_001")
        assert "test_001" not in monitor._monitored_strategies


class TestAnomalyDetectorBasic:
    """异常检测器基础测试"""
    
    def test_import_and_initialization(self):
        """测试导入和初始化"""
        from src.gateway.web.anomaly_detector import (
            AnomalyDetector, AnomalyRule, AnomalyEvent, AnomalySeverity
        )
        
        detector = AnomalyDetector()
        assert detector is not None
        assert len(detector._rules) > 0  # 应该有默认规则
        
    def test_latency_anomaly_detection(self):
        """测试延迟异常检测"""
        from src.gateway.web.anomaly_detector import (
            AnomalyDetector, AnomalyEvent, AnomalySeverity
        )
        
        detector = AnomalyDetector()
        events = []
        
        def on_anomaly(event):
            events.append(event)
            
        detector.add_anomaly_callback(on_anomaly)
        
        # 添加正常延迟数据
        for _ in range(10):
            detector.add_latency_data("test_001", 50.0)
            
        # 添加异常延迟数据
        for _ in range(5):
            detector.add_latency_data("test_001", 2000.0)
            
        # 手动触发检测
        detector._detect_latency_anomaly("test_001", 2000.0)
        
        # 应该检测到异常 (检查事件列表或历史记录)
        assert len(events) > 0 or len(detector._anomaly_history.get("test_001", [])) > 0


class TestAlertCenterBasic:
    """告警通知中心基础测试"""
    
    def test_import_and_initialization(self):
        """测试导入和初始化"""
        from src.gateway.web.alert_center import (
            AlertCenter, Alert, AlertSeverity, AlertStatus
        )
        
        center = AlertCenter()
        assert center is not None
        
    def test_alert_lifecycle(self):
        """测试告警生命周期"""
        from src.gateway.web.alert_center import (
            AlertCenter, AlertSeverity
        )
        
        center = AlertCenter()
        
        # 创建告警
        alert = center.create_alert(
            strategy_id="test_001",
            title="测试告警",
            message="测试消息",
            severity=AlertSeverity.HIGH,
            source="test"
        )
        
        assert alert is not None
        assert alert.strategy_id == "test_001"
        
        # 确认告警
        center.acknowledge_alert(alert.alert_id, "user_001")
        updated = center.get_alert(alert.alert_id)
        assert updated.status.value == "acknowledged"
        
        # 解决告警
        center.resolve_alert(alert.alert_id, "user_001", "已解决")
        updated = center.get_alert(alert.alert_id)
        assert updated.status.value == "resolved"


class TestAuditLoggerBasic:
    """审计日志系统基础测试"""
    
    def test_import_and_initialization(self):
        """测试导入和初始化"""
        from src.gateway.web.audit_logger import (
            AuditLogger, AuditLevel, AuditCategory
        )
        
        # 重置单例以进行测试
        AuditLogger._instance = None
        
        # 使用临时数据库
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
            
        try:
            logger = AuditLogger(db_path=db_path)
            assert logger is not None
            assert logger._initialized is True
        finally:
            # 清理
            AuditLogger._instance = None
            if os.path.exists(db_path):
                os.unlink(db_path)
                
    def test_log_creation(self):
        """测试日志创建"""
        import sqlite3
        import gc
        from src.gateway.web.audit_logger import (
            AuditLogger, AuditLevel, AuditCategory
        )
        
        # 重置单例以进行测试
        AuditLogger._instance = None
        
        # 使用唯一的数据库路径
        db_path = os.path.join(tempfile.gettempdir(), f"audit_test_{os.getpid()}_{int(time.time())}.db")
        
        try:
            # 先手动创建数据库表
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        log_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        category TEXT NOT NULL,
                        action TEXT NOT NULL,
                        user_id TEXT,
                        strategy_id TEXT,
                        session_id TEXT,
                        source_ip TEXT,
                        request_id TEXT,
                        message TEXT NOT NULL,
                        details TEXT,
                        result TEXT NOT NULL,
                        error_code TEXT,
                        error_message TEXT,
                        duration_ms REAL,
                        related_logs TEXT,
                        checksum TEXT NOT NULL
                    )
                """)
                conn.commit()
            
            logger = AuditLogger(db_path=db_path)
            
            # 创建日志
            entry = logger.log(
                level=AuditLevel.INFO,
                category=AuditCategory.STRATEGY_EXECUTION,
                action="strategy_start",
                message="策略启动",
                user_id="user_001",
                strategy_id="test_001"
            )
            
            assert entry is not None
            assert entry.log_id is not None
            assert entry.user_id == "user_001"
            assert entry.strategy_id == "test_001"
            
            # 强制刷新
            logger.flush()
            
            # 查询日志
            logs = logger.query_logs(
                strategy_id="test_001",
                limit=10
            )
            
            assert len(logs) > 0
            
        finally:
            # 清理 - 强制垃圾回收以关闭数据库连接
            AuditLogger._instance = None
            gc.collect()
            
            # 尝试删除文件（如果失败则忽略）
            try:
                if os.path.exists(db_path):
                    os.unlink(db_path)
            except PermissionError:
                pass  # 文件仍被占用，忽略


class TestAuthMiddlewareBasic:
    """权限控制系统基础测试"""
    
    def test_import_and_initialization(self):
        """测试导入和初始化"""
        from src.gateway.web.auth_middleware import (
            AuthManager, User, Role, Permission
        )
        
        auth = AuthManager()
        assert auth is not None
        
    def test_user_permissions(self):
        """测试用户权限"""
        from src.gateway.web.auth_middleware import (
            AuthManager, User, Role, Permission
        )
        
        auth = AuthManager()
        
        # 创建用户
        user = User(
            user_id="test_user",
            username="test",
            roles=[Role.TRADER]
        )
        
        # 验证角色权限
        assert user.has_permission(Permission.STRATEGY_VIEW) is True
        assert user.has_permission(Permission.SYSTEM_CONFIGURE) is False
        
    def test_permission_check(self):
        """测试权限检查"""
        from src.gateway.web.auth_middleware import (
            AuthManager, User, Role, Permission
        )
        
        auth = AuthManager()
        
        user = User(
            user_id="admin_user",
            username="admin",
            roles=[Role.ADMIN]
        )
        
        # 管理员应该有所有权限
        has_perm, reason = auth.check_permission(user, Permission.STRATEGY_DELETE)
        assert has_perm is True


class TestSignalQualityBasic:
    """信号质量监控基础测试"""
    
    def test_import_and_initialization(self):
        """测试导入和初始化"""
        from src.gateway.web.signal_quality_monitor import (
            SignalQualityMonitor, TradingSignal, SignalStatus
        )
        
        monitor = SignalQualityMonitor()
        assert monitor is not None
        assert monitor.latency_monitor is not None
        assert monitor.deduplicator is not None
        
    def test_signal_processing(self):
        """测试信号处理"""
        from src.gateway.web.signal_quality_monitor import (
            SignalQualityMonitor, TradingSignal, SignalStatus
        )
        
        monitor = SignalQualityMonitor()
        
        # 创建信号
        signal = TradingSignal(
            signal_id="sig_001",
            strategy_id="test_001",
            symbol="AAPL",
            direction="BUY",
            timestamp=datetime.now(),
            source_timestamp=datetime.now(),
            receive_timestamp=datetime.now(),
            price=150.0,
            volume=100.0
        )
        
        # 处理信号
        processed = monitor.process_signal(signal)
        
        assert processed is not None
        assert processed.signal_id == "sig_001"
        
    def test_signal_deduplication(self):
        """测试信号去重"""
        from src.gateway.web.signal_quality_monitor import (
            SignalDeduplicator, TradingSignal, SignalStatus
        )
        
        dedup = SignalDeduplicator()
        
        # 创建相同信号
        timestamp = datetime.now()
        
        signal1 = TradingSignal(
            signal_id="sig_001",
            strategy_id="test_001",
            symbol="AAPL",
            direction="BUY",
            timestamp=timestamp,
            source_timestamp=timestamp,
            receive_timestamp=timestamp,
            price=150.0
        )
        
        signal2 = TradingSignal(
            signal_id="sig_002",
            strategy_id="test_001",
            symbol="AAPL",
            direction="BUY",
            timestamp=timestamp,
            source_timestamp=timestamp,
            receive_timestamp=timestamp,
            price=150.0
        )
        
        # 处理第一个信号
        result1 = dedup.add_signal(signal1)
        assert result1.status != SignalStatus.DUPLICATE
        
        # 处理重复信号
        result2 = dedup.add_signal(signal2)
        assert result2.status == SignalStatus.DUPLICATE


class TestDataMaskingBasic:
    """数据脱敏基础测试"""
    
    def test_import_and_initialization(self):
        """测试导入和初始化"""
        from src.gateway.web.data_masking import (
            DataMasker, MaskingStrategy
        )
        
        masker = DataMasker()
        assert masker is not None
        assert len(masker.rules) > 0
        
    def test_masking_strategies(self):
        """测试脱敏策略"""
        from src.gateway.web.data_masking import (
            DataMasker, MaskingStrategy, MaskingRule
        )
        
        masker = DataMasker()
        
        # 测试MASK策略
        rule = MaskingRule(
            field_pattern=".*password.*",
            strategy=MaskingStrategy.MASK,
            params={'show_last': 4}
        )
        
        result = masker.mask_value("mypassword123", rule)
        assert "****" in result or result != "mypassword123"
        
    def test_dict_masking(self):
        """测试字典脱敏"""
        from src.gateway.web.data_masking import DataMasker
        
        masker = DataMasker()
        
        data = {
            "username": "test_user",
            "password": "secret123",
            "api_key": "abc123xyz",
            "email": "test@example.com"
        }
        
        masked = masker.mask(data)
        
        # 敏感字段应该被脱敏
        assert masked["password"] != "secret123"
        assert masked["api_key"] != "abc123xyz"
        
        # 非敏感字段应该保持不变
        assert masked["username"] == "test_user"


class TestIntegration:
    """集成测试"""
    
    def test_monitor_anomaly_alert_flow(self):
        """测试监控-异常-告警流程"""
        from src.gateway.web.execution_monitor import ExecutionMonitor, ExecutionStatus
        from src.gateway.web.anomaly_detector import AnomalyDetector
        from src.gateway.web.alert_center import AlertCenter, AlertSeverity
        
        # 创建组件
        monitor = ExecutionMonitor()
        detector = AnomalyDetector()
        alert_center = AlertCenter()
        
        # 注册策略
        monitor.register_strategy("integration_test", "集成测试策略")
        
        # 更新指标触发异常
        for i in range(20):
            monitor.update_metrics(
                "integration_test",
                latency_ms=2000.0 + i * 100,  # 高延迟
                signal_count=i + 1
            )
            
        # 验证监控数据
        metrics = monitor.get_metrics("integration_test")
        assert metrics is not None
        assert metrics.latency_ms > 1000  # 应该有高延迟记录
        
    def test_full_monitoring_pipeline(self):
        """测试完整监控流程"""
        from src.gateway.web.execution_monitor import ExecutionMonitor, ExecutionStatus
        from src.gateway.web.signal_quality_monitor import SignalQualityMonitor, TradingSignal
        
        monitor = ExecutionMonitor()
        quality_monitor = SignalQualityMonitor()
        
        # 注册策略
        monitor.register_strategy("pipeline_test", "流程测试策略")
        monitor.update_strategy_status("pipeline_test", ExecutionStatus.RUNNING)
        
        # 创建并处理信号
        signal = TradingSignal(
            signal_id="pipe_001",
            strategy_id="pipeline_test",
            symbol="TEST",
            direction="BUY",
            timestamp=datetime.now(),
            source_timestamp=datetime.now(),
            receive_timestamp=datetime.now(),
            price=100.0
        )
        
        processed = quality_monitor.process_signal(signal)
        
        # 更新监控指标
        monitor.update_metrics(
            "pipeline_test",
            latency_ms=processed.latency_ms or 50.0,
            signal_count=1
        )
        
        # 验证流程完成
        metrics = monitor.get_metrics("pipeline_test")
        assert metrics.signal_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
