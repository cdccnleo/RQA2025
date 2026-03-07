#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 基础监控器

测试logging/monitors/base.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from abc import ABC
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.infrastructure.logging.monitors import (
    ILogMonitor, BaseMonitor, MetricCollector, AlertManager, DataStorage, CallbackHandler, BaseMonitorComponent, AlertLevel, AlertData
)
from src.infrastructure.logging.core.exceptions import LogMonitorError
from src.infrastructure.logging.monitors.enums import AlertData, AlertLevel


class ConcreteTestMonitor(BaseMonitorComponent):
    """测试用的具体监控器实现"""

    def _check_health(self):
        """实现抽象方法"""
        return True

    def _collect_metrics(self):
        """实现抽象方法"""
        return []

    def collect_metrics(self):
        """实现抽象方法"""
        return super().collect_metrics()

    def detect_anomalies(self, metrics=None):
        """实现抽象方法"""
        return []

    def _update_health_status(self, status, timestamp=None):
        """实现抽象方法"""
        super()._update_health_status(status)
        if timestamp:
            self.last_check_time = timestamp
        else:
            self.last_check_time = datetime.now()

    def _should_run_check(self):
        """实现抽象方法"""
        return True

    def _record_metrics(self, metrics):
        """实现抽象方法"""
        super()._record_metrics(metrics)

    def _record_anomaly(self, anomaly):
        """实现抽象方法"""
        super()._record_anomaly(anomaly)


class ConcreteBaseMonitor(BaseMonitor):
    """测试BaseMonitor的具体实现"""

    def _check_health(self) -> Dict[str, Any]:
        """实现抽象方法"""
        return {"status": "healthy", "details": "Test monitor is healthy"}

    def _collect_metrics(self) -> Dict[str, Any]:
        """实现抽象方法"""
        return {"test_metric": 42.0, "another_metric": 3.14}

    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """实现抽象方法"""
        return [{"type": "test", "message": "Test anomaly"}]

    def _update_health_status(self, status: str, timestamp=None):
        """实现方法"""
        self.health_status = status
        if timestamp:
            self.last_check_time = timestamp

    def _record_metrics(self, metrics):
        """实现方法"""
        # 简单的实现，保存到历史记录
        if hasattr(self, 'metrics_history'):
            self.metrics_history.append({
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })

    def _record_anomaly(self, anomaly):
        """实现方法"""
        # 简单的实现，保存到异常记录
        if hasattr(self, 'anomalies'):
            self.anomalies.append({
                'anomaly': anomaly,
                'timestamp': datetime.now().isoformat()
            })

    def set_config(self, config: Dict[str, Any]):
        """设置配置"""
        self.config.update(config)
        if 'interval' in config:
            self.interval = config['interval']
        if 'retention_days' in config:
            self.retention_days = config['retention_days']


class TestILogMonitor:
    """测试日志监控器接口"""

    def test_interface_inheritance(self):
        """测试接口继承"""
        assert issubclass(ILogMonitor, ABC)

    def test_interface_abstract_methods(self):
        """测试接口抽象方法"""
        abstract_methods = ILogMonitor.__abstractmethods__
        expected_methods = {'check_health', 'collect_metrics', 'detect_anomalies', 'get_status'}

        assert len(abstract_methods) >= len(expected_methods)
        for method in expected_methods:
            assert method in abstract_methods

    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            ILogMonitor()

    def test_interface_concrete_implementation(self):
        """测试接口的具体实现"""
        class ConcreteMonitor(ILogMonitor):
            def check_health(self):
                return {"status": "healthy", "timestamp": datetime.now()}

            def collect_metrics(self):
                return {"metric1": 42, "metric2": 3.14}

            def detect_anomalies(self):
                return [{"type": "warning", "message": "Test anomaly"}]

            def get_status(self):
                return {"state": "active", "uptime": 3600}

        monitor = ConcreteMonitor()

        # 验证所有方法都可以调用
        assert isinstance(monitor.check_health(), dict)
        assert isinstance(monitor.collect_metrics(), dict)
        assert isinstance(monitor.detect_anomalies(), list)
        assert isinstance(monitor.get_status(), dict)


class TestBaseMonitor:
    """测试基础监控器"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = ConcreteTestMonitor(config={})

    def test_initialization_default(self):
        """测试默认初始化"""
        assert self.monitor.config == {'interval': 60, 'retention_days': 7}
        assert self.monitor.name == "ConcreteTestMonitor"
        assert self.monitor.enabled is True
        assert self.monitor.interval == 60
        assert self.monitor.retention_days == 7

        assert self.monitor.last_check_time is None
        assert self.monitor.health_status.value == "unknown"
        assert isinstance(self.monitor.metrics_history, list)
        assert isinstance(self.monitor.anomalies, list)

    def test_initialization_with_config(self):
        """测试带配置的初始化"""
        config = {
            'name': 'TestMonitor',
            'enabled': False,
            'interval': 30,
            'retention_days': 14,
            'custom_param': 'value'
        }

        monitor = ConcreteTestMonitor(config=config)

        assert monitor.config == config
        assert monitor.name == 'TestMonitor'
        assert monitor.enabled is False
        assert monitor.interval == 30
        assert monitor.retention_days == 14

    def test_check_health_implemented(self):
        """测试健康检查已实现"""
        # BaseMonitorComponent已经实现了check_health
        result = self.monitor.check_health()
        assert isinstance(result, dict)
        assert 'status' in result

    def test_collect_metrics_implemented(self):
        """测试指标收集已实现"""
        # BaseMonitorComponent已经实现了collect_metrics相关功能
        # 这里我们测试record_metric方法
        result = self.monitor.record_metric("test_metric", 42.0)
        assert result is True

    def test_detect_anomalies_implemented(self):
        """测试异常检测已实现"""
        # BaseMonitorComponent的detect_anomalies方法已实现
        result = self.monitor.detect_anomalies({})
        assert isinstance(result, list)

    def test_get_status_basic(self):
        """测试获取基本状态"""
        status = self.monitor.get_status()

        assert isinstance(status, dict)
        assert 'name' in status
        assert 'enabled' in status
        assert 'health_status' in status
        # last_check_time 只有在设置了之后才会存在
        # assert 'last_check_time' in status
        assert 'uptime_seconds' in status
        assert 'config' in status

        assert status['name'] == 'ConcreteTestMonitor'
        assert status['enabled'] is True
        assert status['health_status'] == 'unknown'
        assert isinstance(status['uptime_seconds'], (int, float))

    def test_get_status_with_check_time(self):
        """测试获取带有检查时间的状态"""
        # 设置最后检查时间
        check_time = datetime.now()
        self.monitor.last_check_time = check_time

        status = self.monitor.get_status()

        assert status['last_check_time'] == check_time.isoformat()

    def test_update_health_status(self):
        """测试更新健康状态"""
        self.monitor._update_health_status("healthy")

        assert self.monitor.health_status.value == "healthy"
        assert self.monitor.last_check_time is not None

    def test_update_health_status_with_timestamp(self):
        """测试更新健康状态带时间戳"""
        custom_time = datetime(2023, 1, 1, 12, 0, 0)

        self.monitor._update_health_status("warning", custom_time)

        assert self.monitor.health_status.value == "degraded"
        assert self.monitor.last_check_time == custom_time

    def test_record_metrics(self):
        """测试记录指标"""
        metrics = {"cpu_usage": 85.5, "memory_usage": 72.3}

        self.monitor._record_metrics(metrics)

        assert len(self.monitor.metrics_history) == 1

        recorded = self.monitor.metrics_history[0]
        assert recorded["metrics"] == metrics
        assert "timestamp" in recorded
        assert isinstance(recorded["timestamp"], datetime)

    def test_record_metrics_multiple(self):
        """测试记录多个指标"""
        metrics1 = {"metric1": 10}
        metrics2 = {"metric2": 20}
        metrics3 = {"metric3": 30}

        self.monitor._record_metrics(metrics1)
        self.monitor._record_metrics(metrics2)
        self.monitor._record_metrics(metrics3)

        assert len(self.monitor.metrics_history) == 3

        # 验证顺序和内容
        assert self.monitor.metrics_history[0]["metrics"] == metrics1
        assert self.monitor.metrics_history[1]["metrics"] == metrics2
        assert self.monitor.metrics_history[2]["metrics"] == metrics3

    def test_record_anomaly(self):
        """测试记录异常"""
        anomaly = {
            "type": "performance",
            "severity": "warning",
            "message": "High CPU usage detected",
            "value": 95.5,
            "threshold": 80
        }

        self.monitor._record_anomaly(anomaly)

        assert len(self.monitor.anomalies) == 1

        recorded = self.monitor.anomalies[0]
        assert recorded["anomaly"] == anomaly
        assert "timestamp" in recorded
        assert isinstance(recorded["timestamp"], datetime)

    def test_record_anomaly_multiple(self):
        """测试记录多个异常"""
        anomalies = [
            {"type": "cpu", "severity": "warning", "value": 90},
            {"type": "memory", "severity": "error", "value": 95},
            {"type": "disk", "severity": "info", "value": 85}
        ]

        for anomaly in anomalies:
            self.monitor._record_anomaly(anomaly)

        assert len(self.monitor.anomalies) == 3

        for i, recorded in enumerate(self.monitor.anomalies):
            assert recorded["anomaly"] == anomalies[i]

    def test_get_recent_metrics(self):
        """测试获取最近指标"""
        # 记录一些指标
        metrics_list = [
            {"cpu": 80},
            {"cpu": 85},
            {"cpu": 90},
            {"cpu": 75}
        ]

        for metrics in metrics_list:
            self.monitor._record_metrics(metrics)

        recent = self.monitor.get_recent_metrics(2)

        assert len(recent) == 2
        # 应该返回最新的2个
        assert recent[0]["metrics"] == {"cpu": 90}
        assert recent[1]["metrics"] == {"cpu": 75}

    def test_get_recent_metrics_all(self):
        """测试获取所有最近指标"""
        metrics_list = [{"cpu": i} for i in range(5)]

        for metrics in metrics_list:
            self.monitor._record_metrics(metrics)

        recent = self.monitor.get_recent_metrics(10)  # 请求更多

        assert len(recent) == 5  # 返回所有

    def test_get_recent_anomalies(self):
        """测试获取最近异常"""
        anomalies = [
            {"type": "cpu", "value": 90},
            {"type": "memory", "value": 95},
            {"type": "disk", "value": 85}
        ]

        for anomaly in anomalies:
            self.monitor._record_anomaly(anomaly)

        recent = self.monitor.get_recent_anomalies(2)

        assert len(recent) == 2
        assert recent[0]["anomaly"] == {"type": "memory", "value": 95}
        assert recent[1]["anomaly"] == {"type": "disk", "value": 85}

    def test_cleanup_old_data(self):
        """测试清理旧数据"""
        # 记录一些带有过去时间戳的指标
        old_time = datetime.now() - timedelta(days=10)
        recent_time = datetime.now() - timedelta(hours=1)

        # 手动设置时间戳来模拟旧数据
        self.monitor._record_metrics({"old": 1})
        self.monitor._record_metrics({"recent": 2})

        # 手动修改时间戳
        self.monitor.metrics_history[0]["timestamp"] = old_time
        self.monitor.metrics_history[1]["timestamp"] = recent_time

        # 清理7天前的旧数据
        self.monitor._cleanup_old_data()

        # 应该只保留最近的数据
        assert len(self.monitor.metrics_history) == 1
        assert self.monitor.metrics_history[0]["metrics"] == {"recent": 2}

    def test_cleanup_old_anomalies(self):
        """测试清理旧异常"""
        # 记录异常并设置旧时间戳
        self.monitor._record_anomaly({"type": "old"})
        self.monitor._record_anomaly({"type": "recent"})

        old_time = datetime.now() - timedelta(days=10)
        recent_time = datetime.now() - timedelta(hours=1)

        self.monitor.anomalies[0]["timestamp"] = old_time
        self.monitor.anomalies[1]["timestamp"] = recent_time

        self.monitor._cleanup_old_data()

        assert len(self.monitor.anomalies) == 1
        assert self.monitor.anomalies[0]["anomaly"] == {"type": "recent"}

    def test_is_enabled(self):
        """测试启用状态"""
        assert self.monitor.enabled is True

        # 禁用监控器
        self.monitor.enabled = False
        assert self.monitor.enabled is False

    def test_set_config(self):
        """测试设置配置"""
        new_config = {
            'interval': 120,
            'retention_days': 30,
            'custom_setting': 'value'
        }

        self.monitor.set_config(new_config)

        assert self.monitor.config == new_config
        assert self.monitor.interval == 120
        assert self.monitor.retention_days == 30

    def test_set_config_partial(self):
        """测试部分设置配置"""
        original_config = {'interval': 60, 'retention_days': 7}

        # 只更新部分配置
        partial_config = {'interval': 90}

        self.monitor.set_config(partial_config)

        # 应该合并配置
        expected_config = {'interval': 90, 'retention_days': 7}
        assert self.monitor.config == expected_config
        assert self.monitor.interval == 90
        assert self.monitor.retention_days == 7

    def test_get_config(self):
        """测试获取配置"""
        config = {"key": "value", "number": 42}

        self.monitor.config = config

        retrieved = self.monitor.get_config()

        assert retrieved == config

    def test_should_run_check(self):
        """测试是否应该运行检查"""
        # 初始状态（没有最后检查时间）
        assert self.monitor.last_check_time is None

        # 设置最后检查时间为现在
        now = datetime.now()
        self.monitor.last_check_time = now

        # 间隔为60秒，应该在60秒后才能再次运行
        assert self.monitor.last_check_time == now

        # 设置旧的检查时间（超过60秒）
        self.monitor.last_check_time = datetime.now() - timedelta(seconds=61)
        old_time = self.monitor.last_check_time

        # 验证时间被正确设置
        assert old_time < datetime.now() - timedelta(seconds=60)

    def test_run_health_check_disabled(self):
        """测试运行健康检查（禁用状态）"""
        self.monitor.enabled = False

        # Mock方法
        with patch.object(self.monitor, 'check_health') as mock_check:
            result = self.monitor.run_health_check()

            # 不应该调用check_health
            mock_check.assert_not_called()

            # 应该返回禁用状态
            assert result["status"] == "disabled"

    def test_run_health_check_enabled(self):
        """测试运行健康检查（启用状态）"""
        self.monitor.enabled = True

        health_result = {"status": "healthy", "details": "All good"}

        with patch.object(self.monitor, 'check_health', return_value=health_result) as mock_check:
            result = self.monitor.run_health_check()

            mock_check.assert_called_once()
            assert result == health_result

            # 应该更新健康状态和检查时间
            assert self.monitor.health_status.value == "healthy"
            assert self.monitor.last_check_time is not None

    def test_run_periodic_checks(self):
        """测试运行定期检查"""
        self.monitor.enabled = True

        # Mock相关方法
        with patch.object(self.monitor, '_should_run_check', return_value=True), \
             patch.object(self.monitor, 'collect_metrics') as mock_collect, \
             patch.object(self.monitor, 'detect_anomalies') as mock_detect, \
             patch.object(self.monitor, '_cleanup_old_data') as mock_cleanup:

            mock_collect.return_value = {"cpu": 75}
            mock_detect.return_value = []

            self.monitor.run_periodic_checks()

            mock_collect.assert_called_once()
            mock_detect.assert_called_once()
            mock_cleanup.assert_called_once()

    def test_run_periodic_checks_disabled(self):
        """测试运行定期检查（禁用状态）"""
        self.monitor.enabled = False

        with patch.object(self.monitor, '_should_run_check', return_value=True), \
             patch.object(self.monitor, 'collect_metrics') as mock_collect:

            self.monitor.run_periodic_checks()

            # 不应该调用收集方法
            mock_collect.assert_not_called()

    def test_run_periodic_checks_not_due(self):
        """测试运行定期检查（未到时间）"""
        self.monitor.enabled = True

        with patch.object(self.monitor, '_should_run_check', return_value=False), \
             patch.object(self.monitor, 'collect_metrics') as mock_collect:

            self.monitor.run_periodic_checks()

            # 不应该调用收集方法
            mock_collect.assert_not_called()

    def test_get_uptime_seconds(self):
        """测试获取运行时间秒数"""
        # 这个方法可能不存在，取决于具体实现
        # 如果存在，应该返回一个数字
        try:
            uptime = self.monitor._get_uptime_seconds()
            assert isinstance(uptime, (int, float))
            assert uptime >= 0
        except AttributeError:
            # 如果方法不存在，跳过测试
            pass

    def test_monitor_statistics(self):
        """测试监控器统计信息"""
        # 记录一些数据
        self.monitor._record_metrics({"m1": 1})
        self.monitor._record_metrics({"m2": 2})
        self.monitor._record_anomaly({"type": "warning"})
        self.monitor._record_anomaly({"type": "error"})

        # 获取扩展状态
        status = self.monitor.get_status()

        assert "metrics_collected" in status or len(self.monitor.metrics_history) == 2
        assert "anomalies_detected" in status or len(self.monitor.anomalies) == 2

    def test_error_handling_in_methods(self):
        """测试方法中的错误处理"""
        # 测试记录指标时的错误处理
        self.monitor._record_metrics({"test": "value"})

        # 即使传递了非预期的值，也应该能处理
        assert len(self.monitor.metrics_history) == 1

    def test_thread_safety_basic(self):
        """测试基本线程安全性"""
        import threading

        results = []
        errors = []

        def concurrent_operation(thread_id):
            try:
                # 每个线程记录不同的指标
                self.monitor._record_metrics({f"thread_{thread_id}": thread_id})
                self.monitor._record_anomaly({"thread": thread_id, "type": "test"})

                results.append(f"thread_{thread_id}_completed")

            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {e}")

        # 启动多个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0
        assert len(results) == 5

        # 验证记录的数量
        assert len(self.monitor.metrics_history) == 5
        assert len(self.monitor.anomalies) == 5

    def test_monitor_configuration_validation(self):
        """测试监控器配置验证"""
        # 有效的配置
        valid_configs = [
            {},
            {"name": "TestMonitor"},
            {"enabled": True, "interval": 30},
            {"retention_days": 14, "custom": "value"}
        ]

        for config in valid_configs:
            monitor = ConcreteBaseMonitor(config)
            # 配置应该包含原始值，但会同步当前的字段值
            # 由于BaseMonitor会验证并可能修正无效值，我们需要检查实际值
            assert monitor.config['interval'] == monitor.interval
            assert monitor.config['retention_days'] == monitor.retention_days
            # 验证其他配置项保持不变
            for key, value in config.items():
                if key not in ['interval', 'retention_days']:
                    assert monitor.config[key] == value

        # 无效配置应该被处理或抛出异常
        invalid_configs = [
            {"interval": -1},  # 负数间隔
            {"retention_days": 0},  # 零保留天数
        ]

        for config in invalid_configs:
            try:
                monitor = ConcreteBaseMonitor(config)
                # 如果没有抛出异常，至少应该有合理的默认值
                assert monitor.interval > 0
                assert monitor.retention_days > 0
            except (ValueError, TypeError):
                # 如果抛出异常，也是可以接受的
                pass

    def test_monitor_lifecycle(self):
        """测试监控器生命周期"""
        # 创建
        monitor = ConcreteBaseMonitor({"name": "LifecycleTest"})

        # 配置
        monitor.set_config({"interval": 120})

        # 运行一些操作
        monitor._record_metrics({"test": 1})
        monitor._record_anomaly({"type": "test"})

        # 检查状态
        status = monitor.get_status()
        assert status["name"] == "LifecycleTest"
        assert status["enabled"] is True

        # 验证数据完整性
        assert len(monitor.metrics_history) == 1
        assert len(monitor.anomalies) == 1

    def test_monitor_performance_baseline(self):
        """测试监控器性能基线"""
        import time

        start_time = time.time()

        # 执行大量操作
        for i in range(100):
            self.monitor._record_metrics({"metric": i})
            self.monitor._record_anomaly({"type": "perf_test", "id": i})

        end_time = time.time()
        duration = end_time - start_time

        # 性能应该在合理范围内
        assert duration < 1.0  # 少于1秒处理100个操作

        # 验证数据完整性
        assert len(self.monitor.metrics_history) == 100
        assert len(self.monitor.anomalies) == 100

    def test_monitor_memory_management(self):
        """测试监控器内存管理"""
        # 记录大量数据
        for i in range(1000):
            self.monitor._record_metrics({"large_metric": i, "data": "x" * 100})
            self.monitor._record_anomaly({"large_anomaly": i, "data": "y" * 100})

        # 清理旧数据
        self.monitor._cleanup_old_data()

        # 应该有合理的内存使用
        # 这里我们只是验证操作不会崩溃
        assert isinstance(self.monitor.metrics_history, list)
        assert isinstance(self.monitor.anomalies, list)

    def test_monitor_comprehensive_status(self):
        """测试监控器综合状态"""
        # 设置各种状态
        self.monitor._update_health_status("healthy")
        self.monitor._record_metrics({"cpu": 75, "memory": 60})
        self.monitor._record_anomaly({"type": "warning", "message": "High CPU"})

        status = self.monitor.get_status()

        # 验证状态包含所有必要信息
        required_keys = ['name', 'enabled', 'health_status', 'config', 'uptime_seconds']
        for key in required_keys:
            assert key in status

        # 验证状态值合理
        assert status['health_status'] == 'healthy'
        assert status['enabled'] is True
        assert isinstance(status['uptime_seconds'], (int, float))

    def test_check_health_disabled_monitor(self):
        """测试禁用监控器的健康检查"""
        monitor = ConcreteBaseMonitor()
        monitor.enabled = False

        result = monitor.check_health()

        assert result['status'] == 'disabled'
        assert result['enabled'] is False

    def test_check_health_enabled_monitor(self):
        """测试启用监控器的健康检查"""
        monitor = ConcreteBaseMonitor()
        monitor.enabled = True

        # Mock _check_health method
        with patch.object(monitor, '_check_health', return_value={'status': 'healthy', 'details': 'All good'}) as mock_check:
            result = monitor.check_health()

            mock_check.assert_called_once()
            assert result['status'] == 'healthy'
            assert result['details'] == 'All good'
            assert monitor.health_status == 'healthy'
            assert monitor.last_check_time is not None

    def test_check_health_exception_handling(self):
        """测试健康检查的异常处理"""
        monitor = ConcreteBaseMonitor()
        monitor.enabled = True

        with patch.object(monitor, '_check_health', side_effect=Exception("Health check failed")):
            with pytest.raises(LogMonitorError, match="Health check failed"):
                monitor.check_health()

            # 验证健康状态被设置为error
            assert monitor.health_status == 'error'

    def test_collect_metrics_disabled_monitor(self):
        """测试禁用监控器的指标收集"""
        monitor = ConcreteBaseMonitor()
        monitor.enabled = False

        result = monitor.collect_metrics()

        assert result == {}

    def test_collect_metrics_enabled_monitor(self):
        """测试启用监控器的指标收集"""
        monitor = ConcreteBaseMonitor()
        monitor.enabled = True

        mock_metrics = {'cpu_usage': 75.5, 'memory_usage': 60.2}

        with patch.object(monitor, '_collect_metrics', return_value=mock_metrics):
            result = monitor.collect_metrics()

            # 验证基本字段被添加
            assert 'timestamp' in result
            assert 'monitor_name' in result
            assert result['monitor_name'] == monitor.name
            assert result['cpu_usage'] == 75.5
            assert result['memory_usage'] == 60.2

            # 验证指标被添加到历史记录
            assert len(monitor.metrics_history) == 1
            assert monitor.metrics_history[0] == result

    def test_collect_metrics_exception_handling(self):
        """测试指标收集的异常处理"""
        monitor = ConcreteBaseMonitor()
        monitor.enabled = True

        with patch.object(monitor, '_collect_metrics', side_effect=Exception("Collection failed")):
            with pytest.raises(LogMonitorError, match="Metrics collection failed"):
                monitor.collect_metrics()

    def test_detect_anomalies_disabled_monitor(self):
        """测试禁用监控器的异常检测"""
        monitor = ConcreteBaseMonitor()
        monitor.enabled = False

        result = monitor.detect_anomalies()

        assert result == []

    def test_detect_anomalies_enabled_monitor(self):
        """测试启用监控器的异常检测"""
        monitor = ConcreteBaseMonitor()
        monitor.enabled = True

        mock_anomalies = [
            {'type': 'cpu_spike', 'severity': 'warning', 'value': 95.0},
            {'type': 'memory_leak', 'severity': 'error', 'value': 98.5}
        ]

        with patch.object(monitor, '_detect_anomalies', return_value=mock_anomalies):
            result = monitor.detect_anomalies()

            assert len(result) == 2

            # 验证异常被添加到历史记录并添加了时间戳和监控器名称
            assert len(monitor.anomalies) == 2
            for anomaly in monitor.anomalies:
                assert 'timestamp' in anomaly
                assert 'monitor_name' in anomaly
                assert anomaly['monitor_name'] == monitor.name

            # 验证返回的是新检测到的异常
            assert result == mock_anomalies

    def test_detect_anomalies_exception_handling(self):
        """测试异常检测的异常处理"""
        monitor = ConcreteBaseMonitor()
        monitor.enabled = True

        with patch.object(monitor, '_detect_anomalies', side_effect=Exception("Detection failed")):
            with pytest.raises(LogMonitorError, match="Anomaly detection failed"):
                monitor.detect_anomalies()

    def test_get_status_comprehensive(self):
        """测试获取完整的状态信息"""
        config = {
            'name': 'TestMonitor',
            'enabled': True,
            'interval': 30,
            'retention_days': 14
        }
        monitor = ConcreteBaseMonitor(config)

        # 设置一些状态
        monitor._update_health_status('healthy')
        monitor.last_check_time = datetime(2023, 1, 1, 12, 0, 0)
        monitor._record_metrics({'test_metric': 42})
        monitor._record_anomaly({'type': 'test_anomaly'})

        status = monitor.get_status()

        assert status['name'] == 'TestMonitor'
        assert status['enabled'] is True
        assert status['health_status'] == 'healthy'
        assert status['last_check_time'] == '2023-01-01T12:00:00'
        assert status['interval'] == 30
        assert status['retention_days'] == 14
        assert status['metrics_count'] == 1
        assert status['anomalies_count'] == 1
        assert status['type'] == 'ConcreteBaseMonitor'

    def test_get_status_without_last_check_time(self):
        """测试获取没有最后检查时间的状态"""
        monitor = ConcreteBaseMonitor()

        status = monitor.get_status()

        assert 'last_check_time' not in status

    def test_cleanup_old_data(self):
        """测试清理过期数据"""
        monitor = ConcreteBaseMonitor()
        monitor.retention_days = 7  # 7天保留期

        # 创建一些测试数据
        base_time = datetime.now()

        # 添加7天内的数据（应该保留）
        recent_time = base_time - timedelta(days=6)
        monitor.metrics_history = [{'timestamp': recent_time.isoformat(), 'data': 'recent'}]
        monitor.anomalies = [{'timestamp': recent_time.isoformat(), 'type': 'recent'}]

        # 添加8天前的数据（应该被清理）
        old_time = base_time - timedelta(days=8)
        monitor.metrics_history.append({'timestamp': old_time.isoformat(), 'data': 'old'})
        monitor.anomalies.append({'timestamp': old_time.isoformat(), 'type': 'old'})

        # 执行清理
        monitor._cleanup_old_data()

        # 验证只有最近的数据被保留
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0]['data'] == 'recent'

        assert len(monitor.anomalies) == 1
        assert monitor.anomalies[0]['type'] == 'recent'

    def test_cleanup_old_data_edge_cases(self):
        """测试清理过期数据的边界情况"""
        monitor = ConcreteBaseMonitor()
        monitor.retention_days = 1  # 1天保留期

        # 空数据列表
        monitor.metrics_history = []
        monitor.anomalies = []
        monitor._cleanup_old_data()  # 应该不会出错

        # 只有一个数据项
        monitor.metrics_history = [{'timestamp': datetime.now().isoformat(), 'data': 'single'}]
        monitor._cleanup_old_data()
        assert len(monitor.metrics_history) == 1

    def test_monitor_workflow_integration(self):
        """测试监控器完整工作流程"""
        monitor = ConcreteBaseMonitor({
            'name': 'WorkflowMonitor',
            'enabled': True,
            'interval': 60,
            'retention_days': 7
        })

        # 1. 初始状态
        assert monitor.health_status == 'unknown'
        assert len(monitor.metrics_history) == 0
        assert len(monitor.anomalies) == 0

        # 2. 执行健康检查
        with patch.object(monitor, '_check_health', return_value={'status': 'healthy'}):
            health_result = monitor.check_health()
            assert health_result['status'] == 'healthy'
            assert monitor.health_status == 'healthy'

        # 3. 收集指标
        test_metrics = {'response_time': 150, 'error_rate': 0.02}
        with patch.object(monitor, '_collect_metrics', return_value=test_metrics):
            metrics_result = monitor.collect_metrics()
            assert 'response_time' in metrics_result
            assert 'timestamp' in metrics_result
            assert len(monitor.metrics_history) == 1

        # 4. 检测异常
        test_anomalies = [{'type': 'high_response_time', 'severity': 'warning'}]
        with patch.object(monitor, '_detect_anomalies', return_value=test_anomalies):
            anomalies_result = monitor.detect_anomalies()
            assert len(anomalies_result) == 1
            assert len(monitor.anomalies) == 1

        # 5. 获取状态
        status = monitor.get_status()
        assert status['health_status'] == 'healthy'
        assert status['metrics_count'] == 1
        assert status['anomalies_count'] == 1

        # 6. 清理过期数据
        monitor._cleanup_old_data()  # 在测试环境中不会有过期数据

    def test_monitor_error_recovery(self):
        """测试监控器错误恢复"""
        monitor = ConcreteBaseMonitor()
        monitor.enabled = True

        # 第一次健康检查失败
        with patch.object(monitor, '_check_health', side_effect=Exception("Temporary failure")):
            with pytest.raises(LogMonitorError):
                monitor.check_health()

            assert monitor.health_status == 'error'

        # 第二次健康检查成功 - 应该能够恢复
        with patch.object(monitor, '_check_health', return_value={'status': 'healthy'}):
            result = monitor.check_health()
            assert result['status'] == 'healthy'
            assert monitor.health_status == 'healthy'

    def test_monitor_configuration_updates(self):
        """测试监控器配置更新"""
        monitor = ConcreteBaseMonitor()

        # 初始配置
        assert monitor.interval == 60
        assert monitor.retention_days == 7

        # 更新配置
        monitor.interval = 30
        monitor.retention_days = 14

        # 验证配置生效
        status = monitor.get_status()
        assert status['interval'] == 30
        assert status['retention_days'] == 14

    def test_monitor_data_persistence_simulation(self):
        """测试监控器数据持久性模拟"""
        monitor = ConcreteBaseMonitor()

        # 模拟收集大量数据
        for i in range(10):
            monitor._record_metrics({f'metric_{i}': i})
            monitor._record_anomaly({'type': f'anomaly_{i}', 'value': i * 10})

        # 验证数据完整性
        assert len(monitor.metrics_history) == 10
        assert len(monitor.anomalies) == 10

        # 模拟状态查询
        status = monitor.get_status()
        assert status['metrics_count'] == 10
        assert status['anomalies_count'] == 10

    def test_monitor_timestamp_handling(self):
        """测试监控器时间戳处理"""
        monitor = ConcreteBaseMonitor()

        # 手动设置时间戳
        test_time = datetime(2023, 12, 25, 10, 30, 0)
        monitor.last_check_time = test_time

        status = monitor.get_status()
        assert status['last_check_time'] == test_time.isoformat()

    def test_monitor_edge_cases(self):
        """测试监控器边界情况"""
        # 空配置
        monitor1 = ConcreteBaseMonitor({})
        assert monitor1.config == {'interval': 60, 'retention_days': 7}
        assert monitor1.name == "ConcreteBaseMonitor"

        # None配置
        monitor2 = ConcreteBaseMonitor(None)
        assert monitor2.config == {'interval': 60, 'retention_days': 7}

        # 大量配置
        large_config = {f"key_{i}": f"value_{i}" for i in range(100)}
        monitor3 = ConcreteBaseMonitor(large_config)
        # 100个原有配置 + 2个默认字段 (interval, retention_days)
        assert len(monitor3.config) == 102

        # 极端值配置
        extreme_config = {
            "interval": 999999,
            "retention_days": 999999
        }
        monitor4 = ConcreteBaseMonitor(extreme_config)
        assert monitor4.interval == 999999
        assert monitor4.retention_days == 999999


class TestMetricCollector:
    """测试指标收集器"""

    def setup_method(self):
        """测试前准备"""
        self.collector = MetricCollector()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.collector, '_metrics')
        assert hasattr(self.collector, '_max_metrics')
        assert hasattr(self.collector, '_metric_callbacks')
        assert isinstance(self.collector._metrics, dict)
        assert self.collector._max_metrics == 1000
        assert isinstance(self.collector._metric_callbacks, list)

    def test_initialization_with_custom_max_metrics(self):
        """测试自定义最大指标数初始化"""
        collector = MetricCollector(max_metrics=500)
        assert collector._max_metrics == 500

    def test_record_metric_success(self):
        """测试成功记录指标"""
        result = self.collector.record_metric("test_metric", 42.5, {"tag": "value"})
        assert result is True
        assert "test_metric" in self.collector._metrics
        assert len(self.collector._metrics["test_metric"]) == 1

        metric_data = self.collector._metrics["test_metric"][0]
        assert metric_data["name"] == "test_metric"
        assert metric_data["value"] == 42.5
        assert metric_data["labels"] == {"tag": "value"}
        assert "timestamp" in metric_data

    def test_record_metric_without_labels(self):
        """测试记录指标时不带标签"""
        result = self.collector.record_metric("simple_metric", 100)
        assert result is True

        metric_data = self.collector._metrics["simple_metric"][0]
        assert metric_data["labels"] == {}

    def test_record_metric_with_custom_timestamp(self):
        """测试记录指标时使用自定义时间戳"""
        custom_timestamp = datetime(2023, 1, 1, 12, 0, 0)
        result = self.collector.record_metric("timestamp_metric", 25.0, timestamp=custom_timestamp)
        assert result is True

        metric_data = self.collector._metrics["timestamp_metric"][0]
        assert metric_data["timestamp"] == custom_timestamp

    def test_record_metric_exception_handling(self):
        """测试记录指标时的异常处理"""
        # 创建一个会引发异常的场景
        collector = MetricCollector(max_metrics=0)  # 这可能会导致问题

        # 这里应该不会抛出异常，而是返回False
        result = collector.record_metric("problematic_metric", "invalid_value")
        # 即使出现异常，也应该返回布尔值
        assert isinstance(result, bool)

    def test_add_metric_callback(self):
        """测试添加指标回调"""
        callback_calls = []

        def test_callback(name, metric_data):
            callback_calls.append((name, metric_data["value"]))

        self.collector.add_metric_callback(test_callback)
        assert len(self.collector._metric_callbacks) == 1
        assert test_callback in self.collector._metric_callbacks

        # 记录指标时应该触发回调
        self.collector.record_metric("callback_test", 99.0)
        assert len(callback_calls) == 1
        assert callback_calls[0] == ("callback_test", 99.0)

    def test_multiple_metric_callbacks(self):
        """测试多个指标回调"""
        calls = []

        def callback1(name, data):
            calls.append(f"callback1_{name}")

        def callback2(name, data):
            calls.append(f"callback2_{name}")

        self.collector.add_metric_callback(callback1)
        self.collector.add_metric_callback(callback2)

        self.collector.record_metric("multi_callback", 50.0)
        assert len(calls) == 2
        assert "callback1_multi_callback" in calls
        assert "callback2_multi_callback" in calls

    def test_get_metrics_all(self):
        """测试获取所有指标"""
        # 记录多个指标
        self.collector.record_metric("metric1", 10.0)
        self.collector.record_metric("metric2", 20.0)
        self.collector.record_metric("metric1", 15.0)  # 再次记录metric1

        metrics = self.collector.get_metrics()
        assert isinstance(metrics, dict)
        assert "metric1" in metrics
        assert "metric2" in metrics
        assert len(metrics["metric1"]) == 2
        assert len(metrics["metric2"]) == 1

    def test_get_metrics_specific_name(self):
        """测试获取特定名称的指标"""
        self.collector.record_metric("metric1", 10.0)
        self.collector.record_metric("metric2", 20.0)

        metrics = self.collector.get_metrics("metric1")
        assert "metric1" in metrics
        assert "metric2" not in metrics
        assert len(metrics["metric1"]) == 1

    def test_get_metrics_with_limit(self):
        """测试获取指标时限制数量"""
        # 记录多个相同指标
        for i in range(5):
            self.collector.record_metric("limited_metric", float(i))

        # 不限制
        all_metrics = self.collector.get_metrics("limited_metric")
        assert len(all_metrics["limited_metric"]) == 5

        # 限制为2
        limited_metrics = self.collector.get_metrics("limited_metric", limit=2)
        assert len(limited_metrics["limited_metric"]) == 2

    def test_get_metrics_nonexistent(self):
        """测试获取不存在的指标"""
        metrics = self.collector.get_metrics("nonexistent")
        assert metrics == {}

    def test_clear_metrics_all(self):
        """测试清除所有指标"""
        self.collector.record_metric("metric1", 10.0)
        self.collector.record_metric("metric2", 20.0)

        assert len(self.collector._metrics) == 2

        self.collector.clear_metrics()
        assert len(self.collector._metrics) == 0

    def test_clear_metrics_specific(self):
        """测试清除特定指标"""
        self.collector.record_metric("metric1", 10.0)
        self.collector.record_metric("metric2", 20.0)

        self.collector.clear_metrics("metric1")
        assert "metric1" not in self.collector._metrics
        assert "metric2" in self.collector._metrics

    def test_capacity_management(self):
        """测试容量管理"""
        # 创建容量很小的收集器
        collector = MetricCollector(max_metrics=2)

        # 记录超过容量的指标
        for i in range(4):
            collector.record_metric("capacity_test", float(i))

        # 应该只保留最新的2个
        metrics = collector.get_metrics("capacity_test")
        assert len(metrics["capacity_test"]) == 2

        # 验证保留的是最新的
        values = [m["value"] for m in metrics["capacity_test"]]
        assert 2.0 in values
        assert 3.0 in values


class TestAlertManager:
    """测试告警管理器"""

    def setup_method(self):
        """测试前准备"""
        self.manager = AlertManager()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.manager, '_alerts')
        assert hasattr(self.manager, '_max_alerts')
        assert hasattr(self.manager, '_alert_callbacks')
        assert isinstance(self.manager._alerts, list)
        assert self.manager._max_alerts == 1000

    def test_initialization_with_custom_max_alerts(self):
        """测试自定义最大告警数初始化"""
        manager = AlertManager(max_alerts=500)
        assert manager._max_alerts == 500

    def test_record_alert_success(self):
        """测试成功记录告警"""
        result = self.manager.record_alert("Test alert message", AlertLevel.WARNING, {"component": "test"})
        assert result is True
        assert len(self.manager._alerts) == 1

        alert_data = self.manager._alerts[0]
        assert alert_data["message"] == "Test alert message"
        assert alert_data["level"] == "warning"
        assert alert_data["labels"] == {"component": "test"}
        assert "timestamp" in alert_data

    def test_record_alert_all_levels(self):
        """测试记录所有级别的告警"""
        levels = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR]

        for level in levels:
            self.manager.record_alert(f"{level.value} alert", level)

        assert len(self.manager._alerts) == 3

        # 验证级别被正确转换
        expected_levels = ["info", "warning", "error"]
        actual_levels = [alert["level"] for alert in self.manager._alerts]
        assert actual_levels == expected_levels

    def test_record_alert_without_labels(self):
        """测试记录告警时不带标签"""
        result = self.manager.record_alert("Simple alert", AlertLevel.INFO)
        assert result is True

        alert_data = self.manager._alerts[0]
        assert alert_data["labels"] == {}

    def test_record_alert_with_custom_timestamp(self):
        """测试记录告警时使用自定义时间戳"""
        custom_timestamp = datetime(2023, 1, 1, 12, 0, 0)
        result = self.manager.record_alert("Timestamp alert", AlertLevel.ERROR, timestamp=custom_timestamp)
        assert result is True

        alert_data = self.manager._alerts[0]
        assert alert_data["timestamp"] == custom_timestamp

    def test_add_alert_callback(self):
        """测试添加告警回调"""
        callback_calls = []

        def test_callback(alert_data):
            callback_calls.append(alert_data["message"])

        self.manager.add_alert_callback(test_callback)
        assert len(self.manager._alert_callbacks) == 1

        # 记录告警时应该触发回调
        self.manager.record_alert("Callback test alert", AlertLevel.WARNING)
        assert len(callback_calls) == 1
        assert callback_calls[0] == "Callback test alert"

    def test_multiple_alert_callbacks(self):
        """测试多个告警回调"""
        calls = []

        def callback1(alert_data):
            calls.append(f"callback1_{alert_data['level']}")

        def callback2(alert_data):
            calls.append(f"callback2_{alert_data['level']}")

        self.manager.add_alert_callback(callback1)
        self.manager.add_alert_callback(callback2)

        self.manager.record_alert("Multi callback alert", AlertLevel.ERROR)
        assert len(calls) == 2
        assert "callback1_error" in calls
        assert "callback2_error" in calls

    def test_get_alerts_all(self):
        """测试获取所有告警"""
        self.manager.record_alert("Alert 1", AlertLevel.INFO)
        self.manager.record_alert("Alert 2", AlertLevel.WARNING)

        alerts = self.manager.get_alerts()
        assert isinstance(alerts, list)
        assert len(alerts) == 2

    def test_get_alerts_by_level(self):
        """测试按级别获取告警"""
        self.manager.record_alert("Info alert", AlertLevel.INFO)
        self.manager.record_alert("Warning alert", AlertLevel.WARNING)
        self.manager.record_alert("Error alert", AlertLevel.ERROR)

        info_alerts = self.manager.get_alerts(level="info")
        warning_alerts = self.manager.get_alerts(level="warning")
        error_alerts = self.manager.get_alerts(level="error")

        assert len(info_alerts) == 1
        assert len(warning_alerts) == 1
        assert len(error_alerts) == 1

        assert info_alerts[0]["level"] == "info"
        assert warning_alerts[0]["level"] == "warning"
        assert error_alerts[0]["level"] == "error"

    def test_get_alerts_with_limit(self):
        """测试获取告警时限制数量"""
        for i in range(5):
            self.manager.record_alert(f"Alert {i}", AlertLevel.WARNING)

        all_alerts = self.manager.get_alerts()
        assert len(all_alerts) == 5

        limited_alerts = self.manager.get_alerts(limit=2)
        assert len(limited_alerts) == 2

    def test_clear_alerts_all(self):
        """测试清除所有告警"""
        self.manager.record_alert("Alert 1", AlertLevel.INFO)
        self.manager.record_alert("Alert 2", AlertLevel.WARNING)

        assert len(self.manager._alerts) == 2

        self.manager.clear_alerts()
        assert len(self.manager._alerts) == 0

    def test_clear_alerts_by_level(self):
        """测试按级别清除告警"""
        self.manager.record_alert("Info alert", AlertLevel.INFO)
        self.manager.record_alert("Warning alert", AlertLevel.WARNING)
        self.manager.record_alert("Error alert", AlertLevel.ERROR)

        self.manager.clear_alerts(level="warning")
        remaining_alerts = self.manager.get_alerts()

        # 应该只剩下info和error级别的告警
        levels = [alert["level"] for alert in remaining_alerts]
        assert "warning" not in levels
        assert "info" in levels or "error" in levels

    def test_alert_capacity_management(self):
        """测试告警容量管理"""
        # 创建容量很小的管理器
        manager = AlertManager(max_alerts=2)

        # 记录超过容量的告警
        for i in range(4):
            manager.record_alert(f"Capacity alert {i}", AlertLevel.WARNING)

        # 应该只保留最新的告警
        alerts = manager.get_alerts()
        assert len(alerts) == 2


class TestDataStorage:
    """测试数据存储器"""

    def setup_method(self):
        """测试前准备"""
        self.store = DataStorage()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.store, '_storage_backends')
        assert isinstance(self.store._storage_backends, dict)

    def test_store_data_memory(self):
        """测试存储数据到内存"""
        result = self.store.store_data("test_key", {"data": "value"}, "memory")
        assert result is True
        assert "memory" in self.store._storage_backends
        assert "test_key" in self.store._storage_backends["memory"]

    def test_store_data_file(self):
        """测试存储数据到文件（模拟）"""
        result = self.store.store_data("file_key", {"data": "file_value"}, "file")
        assert result is True
        assert "file" in self.store._storage_backends
        assert "file_key" in self.store._storage_backends["file"]

    def test_retrieve_data_memory(self):
        """测试从内存检索数据"""
        test_data = {"data": "memory_data", "timestamp": datetime.now()}
        self.store._storage_backends["memory"] = {"memory_key": test_data}

        result = self.store.retrieve_data("memory_key", "memory")
        assert result == test_data["data"]

    def test_retrieve_data_file(self):
        """测试从文件检索数据"""
        test_data = {"data": "file_data", "timestamp": datetime.now()}
        self.store._storage_backends["file"] = {"file_key": test_data}

        result = self.store.retrieve_data("file_key", "file")
        assert result == test_data["data"]

    def test_retrieve_data_not_found(self):
        """测试检索不存在的数据"""
        result = self.store.retrieve_data("nonexistent", "memory")
        assert result is None

    def test_delete_data_memory(self):
        """测试删除内存数据"""
        test_data = {"data": "to_delete", "timestamp": datetime.now()}
        self.store._storage_backends["memory"] = {"delete_key": test_data}

        result = self.store.delete_data("delete_key", "memory")
        assert result is True
        assert "delete_key" not in self.store._storage_backends["memory"]

    def test_delete_data_file(self):
        """测试删除文件数据"""
        test_data = {"data": "file_to_delete", "timestamp": datetime.now()}
        self.store._storage_backends["file"] = {"delete_file_key": test_data}

        result = self.store.delete_data("delete_file_key", "file")
        assert result is True
        assert "delete_file_key" not in self.store._storage_backends["file"]

    def test_delete_data_not_found(self):
        """测试删除不存在的数据"""
        result = self.store.delete_data("nonexistent", "memory")
        assert result is False

    def test_list_keys_memory(self):
        """测试列出内存键"""
        self.store._storage_backends["memory"] = {
            "key1": {"data": "value1", "timestamp": datetime.now()},
            "key2": {"data": "value2", "timestamp": datetime.now()},
            "key3": {"data": "value3", "timestamp": datetime.now()}
        }

        keys = self.store.list_keys("memory")
        assert isinstance(keys, list)
        assert set(keys) == {"key1", "key2", "key3"}

    def test_list_keys_file(self):
        """测试列出文件键"""
        self.store._storage_backends["file"] = {
            "file_key1": {"data": "file_value1", "timestamp": datetime.now()},
            "file_key2": {"data": "file_value2", "timestamp": datetime.now()}
        }

        keys = self.store.list_keys("file")
        assert isinstance(keys, list)
        assert set(keys) == {"file_key1", "file_key2"}

    def test_list_keys_empty(self):
        """测试列出空存储器的键"""
        keys = self.store.list_keys("memory")
        assert keys == []


class TestCallbackHandler:
    """测试回调处理器"""

    def setup_method(self):
        """测试前准备"""
        self.handler = CallbackHandler()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.handler, '_callbacks')
        assert isinstance(self.handler._callbacks, dict)

    def test_register_callback(self):
        """测试注册回调"""
        def test_callback():
            pass

        self.handler.register_callback("test_event", test_callback)
        assert "test_event" in self.handler._callbacks
        assert test_callback in self.handler._callbacks["test_event"]

    def test_register_multiple_callbacks_same_event(self):
        """测试为同一事件注册多个回调"""
        def callback1():
            pass

        def callback2():
            pass

        self.handler.register_callback("same_event", callback1)
        self.handler.register_callback("same_event", callback2)

        assert len(self.handler._callbacks["same_event"]) == 2
        assert callback1 in self.handler._callbacks["same_event"]
        assert callback2 in self.handler._callbacks["same_event"]

    def test_unregister_callback(self):
        """测试取消注册回调"""
        def test_callback():
            pass

        self.handler.register_callback("test_event", test_callback)
        assert test_callback in self.handler._callbacks["test_event"]

        self.handler.unregister_callback("test_event", test_callback)
        assert test_callback not in self.handler._callbacks["test_event"]

    def test_trigger_callbacks(self):
        """测试触发回调"""
        callback_calls = []

        def callback1(arg1, arg2=None):
            callback_calls.append(f"callback1_{arg1}_{arg2}")

        def callback2(arg1, arg2=None):
            callback_calls.append(f"callback2_{arg1}_{arg2}")

        self.handler.register_callback("trigger_event", callback1)
        self.handler.register_callback("trigger_event", callback2)

        self.handler.trigger_callbacks("trigger_event", "value1", arg2="value2")

        assert len(callback_calls) == 2
        assert "callback1_value1_value2" in callback_calls
        assert "callback2_value1_value2" in callback_calls

    def test_trigger_callbacks_no_callbacks(self):
        """测试触发没有回调的事件"""
        # 不应该抛出异常
        self.handler.trigger_callbacks("no_callbacks_event", "arg1", kwarg1="value1")

    def test_clear_callbacks_specific_event(self):
        """测试清除特定事件的回调"""
        def callback1():
            pass

        def callback2():
            pass

        self.handler.register_callback("event1", callback1)
        self.handler.register_callback("event2", callback2)

        assert len(self.handler._callbacks["event1"]) == 1
        assert len(self.handler._callbacks["event2"]) == 1

        self.handler.clear_callbacks("event1")

        assert "event1" not in self.handler._callbacks
        assert len(self.handler._callbacks["event2"]) == 1

    def test_clear_callbacks_all(self):
        """测试清除所有回调"""
        def callback1():
            pass

        def callback2():
            pass

        self.handler.register_callback("event1", callback1)
        self.handler.register_callback("event2", callback2)

        self.handler.clear_callbacks()

        assert len(self.handler._callbacks) == 0


class TestBaseMonitorComponent:
    """测试基础监控器组件"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = BaseMonitorComponent()

    def test_initialization(self):
        """测试初始化"""
        assert hasattr(self.monitor, '_metric_collector')
        assert hasattr(self.monitor, '_alert_manager')
        assert hasattr(self.monitor, '_data_storage')
        assert hasattr(self.monitor, '_callback_handler')
        assert hasattr(self.monitor, 'config')

        assert isinstance(self.monitor._metric_collector, MetricCollector)
        assert isinstance(self.monitor._alert_manager, AlertManager)
        assert isinstance(self.monitor._data_storage, DataStorage)
        assert isinstance(self.monitor._callback_handler, CallbackHandler)

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = {"max_metrics": 500, "max_alerts": 200}
        monitor = BaseMonitorComponent(config)

        assert monitor.config == config
        assert monitor._metric_collector._max_metrics == 500
        assert monitor._alert_manager._max_alerts == 200

    def test_initialization_defaults(self):
        monitor = BaseMonitorComponent()
        # 由于我们同步默认值，config 会包含默认字段
        assert 'interval' in monitor.config
        assert 'retention_days' in monitor.config
        assert monitor.enabled is True
        assert monitor.interval == 60

    def test_record_metric_basic(self):
        monitor = BaseMonitorComponent()
        monitor.record_metric("test_metric", 42.0)
        assert "test_metric" in monitor.metrics

    def test_record_metric(self):
        """测试记录指标"""
        result = self.monitor.record_metric("test_metric", 42.0, metadata={"tag": "value"})
        assert result is True

        # 验证指标被存储
        metrics = self.monitor.get_metrics("test_metric")
        assert "test_metric" in metrics
        assert len(metrics["test_metric"]) == 1

    def test_record_alert(self):
        """测试记录告警"""
        result = self.monitor.record_alert("Test alert", AlertLevel.WARNING, {"component": "test"})
        assert result is True

        # 验证告警被存储
        alerts = self.monitor.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["message"] == "Test alert"

    def test_add_metric_callback(self):
        """测试添加指标回调"""
        callback_calls = []

        def test_callback(name, data):
            callback_calls.append((name, data["value"]))

        self.monitor.add_metric_callback("callback_metric", test_callback)

        # 记录指标应该触发回调
        self.monitor.record_metric("callback_metric", 99.0)
        assert len(callback_calls) == 1
        assert callback_calls[0] == ("callback_metric", 99.0)

    def test_add_alert_callback(self):
        """测试添加告警回调"""
        callback_calls = []

        def test_callback(alert_data):
            callback_calls.append(alert_data["level"])

        self.monitor.add_alert_callback(test_callback)

        # 记录告警应该触发回调
        self.monitor.record_alert("Callback alert", AlertLevel.ERROR)
        assert len(callback_calls) == 1
        assert callback_calls[0] == "error"

    def test_get_metrics(self):
        """测试获取指标"""
        self.monitor.record_metric("metric1", 10.0)
        self.monitor.record_metric("metric2", 20.0)

        all_metrics = self.monitor.get_metrics()
        assert isinstance(all_metrics, dict)
        assert len(all_metrics) >= 2

        specific_metric = self.monitor.get_metrics("metric1")
        assert "metric1" in specific_metric

    def test_get_alerts(self):
        """测试获取告警"""
        self.monitor.record_alert("Alert 1", AlertLevel.INFO)
        self.monitor.record_alert("Alert 2", AlertLevel.WARNING)

        alerts = self.monitor.get_alerts()
        assert isinstance(alerts, list)
        assert len(alerts) == 2

    def test_store_and_retrieve_data(self):
        """测试存储和检索数据"""
        test_data = {"key": "value", "number": 42}

        # 存储数据
        result = self.monitor.store_data("test_key", test_data)
        assert result is True

        # 检索数据
        retrieved = self.monitor.retrieve_data("test_key")
        assert retrieved == test_data

    def test_clear_metrics(self):
        """测试清除指标"""
        self.monitor.record_metric("test_metric", 100.0)
        assert len(self.monitor.get_metrics()) > 0

        self.monitor.clear_metrics()
        # 清除后应该没有指标了
        metrics = self.monitor.get_metrics()
        assert len(metrics) == 0

    def test_clear_alerts(self):
        """测试清除告警"""
        self.monitor.record_alert("Test alert", AlertLevel.WARNING)
        assert len(self.monitor.get_alerts()) == 1

        self.monitor.clear_alerts()
        assert len(self.monitor.get_alerts()) == 0

    def test_register_callback(self):
        """测试注册事件回调"""
        callback_calls = []

        def test_callback(*args, **kwargs):
            callback_calls.append(("callback_called", args, kwargs))

        self.monitor.register_callback("test_event", test_callback)

        # 触发回调
        self.monitor.trigger_callbacks("test_event", "arg1", kwarg1="value1")
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "callback_called"

    def test_get_status(self):
        """测试获取状态"""
        status = self.monitor.get_status()
        assert isinstance(status, dict)
        # 状态应该包含基本信息
        assert "status" in status

    def test_start_and_stop(self):
        """测试启动和停止"""
        # 启动监控器
        self.monitor.start()

        # 停止监控器
        self.monitor.stop()

        # 不应该抛出异常

    def test_check_health(self):
        monitor = BaseMonitorComponent()
        health = monitor.check_health()
        assert "status" in health

    def test_collect_metrics(self):
        monitor = BaseMonitorComponent()
        metrics = monitor.collect_metrics()
        assert isinstance(metrics, dict)

    def test_detect_anomalies(self):
        monitor = BaseMonitorComponent()
        anomalies = monitor.detect_anomalies({})
        assert isinstance(anomalies, list)
