#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控系统简化独立测试

测试基本的监控功能，不依赖复杂的导入链
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from datetime import datetime


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """告警状态枚举"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class Alert:
    """告警类"""

    def __init__(self,
                 alert_id: str,
                 name: str,
                 level: AlertLevel,
                 message: str,
                 source: str = "",
                 details: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[float] = None):
        self.alert_id = alert_id
        self.name = name
        self.level = level
        self.message = message
        self.source = source
        self.details = details or {}
        self.timestamp = timestamp or time.time()
        self.status = AlertStatus.ACTIVE

    def resolve(self):
        """解决告警"""
        self.status = AlertStatus.RESOLVED

    def acknowledge(self):
        """确认告警"""
        if self.status == AlertStatus.ACTIVE:
            self.status = AlertStatus.ACKNOWLEDGED

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "details": self.details,
            "timestamp": self.timestamp,
            "status": self.status.value
        }


class Metric:
    """指标类"""

    def __init__(self,
                 name: str,
                 metric_type: MetricType,
                 value: Any = 0,
                 labels: Optional[Dict[str, str]] = None,
                 description: str = "",
                 timestamp: Optional[float] = None):
        self.name = name
        self.metric_type = metric_type
        self.value = value
        self.labels = labels or {}
        self.description = description
        self.timestamp = timestamp or time.time()

    def update(self, value: Any):
        """更新指标值"""
        self.value = value
        self.timestamp = time.time()

    def increment(self, amount: float = 1.0):
        """增加计数器值"""
        if self.metric_type == MetricType.COUNTER:
            self.value += amount
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "labels": self.labels,
            "description": self.description,
            "timestamp": self.timestamp
        }


class SimpleMonitoringSystem:
    """简化的监控系统"""

    def __init__(self, name: str = "monitoring_system"):
        self.name = name
        self._alerts: Dict[str, Alert] = {}
        self._metrics: Dict[str, Metric] = {}
        self._alert_handlers: List[Callable[[Alert], None]] = []
        # 使用可重入锁避免死锁（修复：get_system_status中调用get_active_alerts的嵌套锁问题）
        self._lock = threading.RLock()

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """添加告警处理器"""
        self._alert_handlers.append(handler)

    def remove_alert_handler(self, handler: Callable[[Alert], None]):
        """移除告警处理器"""
        if handler in self._alert_handlers:
            self._alert_handlers.remove(handler)

    def create_alert(self,
                     name: str,
                     level: AlertLevel,
                     message: str,
                     source: str = "",
                     details: Optional[Dict[str, Any]] = None) -> Alert:
        """创建告警"""
        alert_id = f"{name}_{int(time.time() * 1000)}"
        alert = Alert(alert_id, name, level, message, source, details)

        with self._lock:
            self._alerts[alert_id] = alert

        # 触发告警处理器
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception:
                # 处理器异常不应影响系统运行
                pass

        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self._lock:
            if alert_id in self._alerts:
                self._alerts[alert_id].resolve()
                return True
        return False

    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        with self._lock:
            if alert_id in self._alerts:
                self._alerts[alert_id].acknowledge()
                return True
        return False

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """获取告警"""
        with self._lock:
            return self._alerts.get(alert_id)

    def get_all_alerts(self) -> Dict[str, Alert]:
        """获取所有告警"""
        with self._lock:
            return self._alerts.copy()

    def get_active_alerts(self) -> Dict[str, Alert]:
        """获取活跃告警"""
        with self._lock:
            return {aid: alert for aid, alert in self._alerts.items()
                   if alert.status == AlertStatus.ACTIVE}

    def create_metric(self,
                      name: str,
                      metric_type: MetricType,
                      initial_value: Any = 0,
                      labels: Optional[Dict[str, str]] = None,
                      description: str = "") -> Metric:
        """创建指标"""
        metric = Metric(name, metric_type, initial_value, labels, description)

        with self._lock:
            self._metrics[name] = metric

        return metric

    def update_metric(self, name: str, value: Any) -> bool:
        """更新指标"""
        with self._lock:
            if name in self._metrics:
                self._metrics[name].update(value)
                return True
        return False

    def increment_metric(self, name: str, amount: float = 1.0) -> bool:
        """增加指标值"""
        with self._lock:
            if name in self._metrics:
                self._metrics[name].increment(amount)
                return True
        return False

    def get_metric(self, name: str) -> Optional[Metric]:
        """获取指标"""
        with self._lock:
            return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Metric]:
        """获取所有指标"""
        with self._lock:
            return self._metrics.copy()

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self._lock:
            active_alerts = len(self.get_active_alerts())
            total_alerts = len(self._alerts)
            total_metrics = len(self._metrics)

            # 计算告警严重程度
            alert_levels = {}
            for alert in self._alerts.values():
                level = alert.level.value
                alert_levels[level] = alert_levels.get(level, 0) + 1

            return {
                "system_name": self.name,
                "timestamp": time.time(),
                "alerts": {
                    "total": total_alerts,
                    "active": active_alerts,
                    "by_level": alert_levels
                },
                "metrics": {
                    "total": total_metrics,
                    "types": {m.metric_type.value: list(self._metrics.keys()) for m in self._metrics.values()}
                },
                "health_score": max(0, 100 - (active_alerts * 10))  # 简单健康评分
            }


class TestSimpleMonitoringSystem:
    """测试简化监控系统"""

    def setup_method(self, method):
        """测试前准备"""
        self.monitoring = SimpleMonitoringSystem("test_monitoring")

    def teardown_method(self, method):
        """测试后清理"""
        self.monitoring._alerts.clear()
        self.monitoring._metrics.clear()
        self.monitoring._alert_handlers.clear()

    def test_initialization(self):
        """测试初始化"""
        assert self.monitoring.name == "test_monitoring"
        assert self.monitoring._alerts == {}
        assert self.monitoring._metrics == {}
        assert self.monitoring._alert_handlers == []

    def test_alert_creation_and_management(self):
        """测试告警创建和管理"""
        # 创建告警
        alert = self.monitoring.create_alert(
            name="test_alert",
            level=AlertLevel.WARNING,
            message="Test alert message",
            source="test_component",
            details={"error_code": 500}
        )

        assert alert.alert_id.startswith("test_alert_")
        assert alert.name == "test_alert"
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test alert message"
        assert alert.source == "test_component"
        assert alert.status == AlertStatus.ACTIVE
        assert alert.details["error_code"] == 500

        # 验证告警已存储
        stored_alert = self.monitoring.get_alert(alert.alert_id)
        assert stored_alert is not None
        assert stored_alert.name == alert.name

        # 确认告警
        result = self.monitoring.acknowledge_alert(alert.alert_id)
        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED

        # 解决告警
        result = self.monitoring.resolve_alert(alert.alert_id)
        assert result is True
        assert alert.status == AlertStatus.RESOLVED

    def test_alert_handlers(self):
        """测试告警处理器"""
        handler_calls = []

        def test_handler(alert):
            handler_calls.append(alert)

        # 添加处理器
        self.monitoring.add_alert_handler(test_handler)
        assert len(self.monitoring._alert_handlers) == 1

        # 创建告警，触发处理器
        alert = self.monitoring.create_alert(
            name="handler_test",
            level=AlertLevel.ERROR,
            message="Handler test"
        )

        # 验证处理器被调用
        assert len(handler_calls) == 1
        assert handler_calls[0] == alert

        # 移除处理器
        self.monitoring.remove_alert_handler(test_handler)
        assert len(self.monitoring._alert_handlers) == 0

    def test_multiple_alerts_management(self):
        """测试多告警管理"""
        # 创建多个告警
        alerts = []
        for i in range(5):
            alert = self.monitoring.create_alert(
                name=f"multi_alert_{i}",
                level=AlertLevel.ERROR if i % 2 == 0 else AlertLevel.WARNING,
                message=f"Multi alert {i}",
                source=f"component_{i}"
            )
            alerts.append(alert)

        # 验证所有告警都存在
        all_alerts = self.monitoring.get_all_alerts()
        assert len(all_alerts) == 5

        # 验证活跃告警
        active_alerts = self.monitoring.get_active_alerts()
        assert len(active_alerts) == 5

        # 解决一些告警
        for i in range(3):
            self.monitoring.resolve_alert(alerts[i].alert_id)

        # 验证活跃告警数量减少
        active_alerts = self.monitoring.get_active_alerts()
        assert len(active_alerts) == 2

    def test_metric_creation_and_management(self):
        """测试指标创建和管理"""
        # 创建计数器指标
        counter = self.monitoring.create_metric(
            name="test_counter",
            metric_type=MetricType.COUNTER,
            initial_value=0,
            description="Test counter metric"
        )

        assert counter.name == "test_counter"
        assert counter.metric_type == MetricType.COUNTER
        assert counter.value == 0
        assert counter.description == "Test counter metric"

        # 创建量规指标
        gauge = self.monitoring.create_metric(
            name="test_gauge",
            metric_type=MetricType.GAUGE,
            initial_value=100.5,
            labels={"unit": "percentage"}
        )

        assert gauge.name == "test_gauge"
        assert gauge.metric_type == MetricType.GAUGE
        assert gauge.value == 100.5
        assert gauge.labels["unit"] == "percentage"

        # 更新指标
        result = self.monitoring.update_metric("test_gauge", 95.2)
        assert result is True

        stored_gauge = self.monitoring.get_metric("test_gauge")
        assert stored_gauge.value == 95.2

        # 增加计数器
        result = self.monitoring.increment_metric("test_counter", 5)
        assert result is True

        stored_counter = self.monitoring.get_metric("test_counter")
        assert stored_counter.value == 5

        # 再次增加
        result = self.monitoring.increment_metric("test_counter")
        assert result is True
        assert stored_counter.value == 6

    def test_metric_types_and_operations(self):
        """测试不同指标类型和操作"""
        metrics = {}

        # 测试计数器
        metrics["counter"] = self.monitoring.create_metric(
            "counter_metric", MetricType.COUNTER, 0
        )

        # 测试量规
        metrics["gauge"] = self.monitoring.create_metric(
            "gauge_metric", MetricType.GAUGE, 50
        )

        # 测试直方图（模拟）
        metrics["histogram"] = self.monitoring.create_metric(
            "histogram_metric", MetricType.HISTOGRAM, [1.0, 2.0, 3.0]
        )

        # 测试汇总（模拟）
        metrics["summary"] = self.monitoring.create_metric(
            "summary_metric", MetricType.SUMMARY, {"count": 10, "sum": 50.5}
        )

        # 验证所有指标都创建成功
        all_metrics = self.monitoring.get_all_metrics()
        assert len(all_metrics) == 4

        for metric in metrics.values():
            stored = self.monitoring.get_metric(metric.name)
            assert stored is not None
            assert stored.metric_type == metric.metric_type

    def test_system_status_reporting(self):
        """测试系统状态报告"""
        # 创建一些告警和指标
        self.monitoring.create_alert("status_test_1", AlertLevel.WARNING, "Test warning")
        self.monitoring.create_alert("status_test_2", AlertLevel.ERROR, "Test error")

        self.monitoring.create_metric("status_metric_1", MetricType.COUNTER)
        self.monitoring.create_metric("status_metric_2", MetricType.GAUGE)

        # 获取系统状态
        status = self.monitoring.get_system_status()

        assert status["system_name"] == "test_monitoring"
        assert isinstance(status["timestamp"], float)
        assert status["alerts"]["total"] == 2
        assert status["alerts"]["active"] == 2
        assert status["alerts"]["by_level"]["warning"] == 1
        assert status["alerts"]["by_level"]["error"] == 1
        assert status["metrics"]["total"] == 2
        assert isinstance(status["health_score"], (int, float))

    def test_concurrent_alert_handling(self):
        """测试并发告警处理"""
        import concurrent.futures

        alert_count = 50
        created_alerts = []

        def create_alert_worker(worker_id):
            """告警创建工作线程"""
            alerts = []
            for i in range(10):
                alert = self.monitoring.create_alert(
                    name=f"concurrent_alert_{worker_id}_{i}",
                    level=AlertLevel.INFO,
                    message=f"Concurrent alert {worker_id}-{i}"
                )
                alerts.append(alert.alert_id)
            return alerts

        # 并发创建告警
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_alert_worker, i) for i in range(5)]
            for future in concurrent.futures.as_completed(futures):
                created_alerts.extend(future.result())

        # 验证所有告警都创建成功
        assert len(created_alerts) == alert_count
        all_alerts = self.monitoring.get_all_alerts()
        assert len(all_alerts) == alert_count

        # 验证每个告警都可以访问
        for alert_id in created_alerts:
            alert = self.monitoring.get_alert(alert_id)
            assert alert is not None
            assert alert.status == AlertStatus.ACTIVE

    def test_metric_thread_safety(self):
        """测试指标线程安全性"""
        import concurrent.futures

        metric_name = "thread_safety_counter"
        self.monitoring.create_metric(metric_name, MetricType.COUNTER, 0)

        increment_count = 100
        thread_count = 10

        def increment_worker():
            """指标增加工作线程"""
            for _ in range(increment_count):
                self.monitoring.increment_metric(metric_name)

        # 并发增加指标
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [executor.submit(increment_worker) for _ in range(thread_count)]
            concurrent.futures.wait(futures)

        # 验证最终值
        metric = self.monitoring.get_metric(metric_name)
        expected_value = increment_count * thread_count
        assert metric.value == expected_value, \
            f"Expected {expected_value}, got {metric.value}"

    def test_alert_lifecycle_management(self):
        """测试告警生命周期管理"""
        # 创建告警
        alert = self.monitoring.create_alert(
            "lifecycle_test", AlertLevel.CRITICAL, "Lifecycle test"
        )

        # 初始状态应该是活跃的
        assert alert.status == AlertStatus.ACTIVE

        # 确认告警
        self.monitoring.acknowledge_alert(alert.alert_id)
        alert = self.monitoring.get_alert(alert.alert_id)
        assert alert.status == AlertStatus.ACKNOWLEDGED

        # 解决告警
        self.monitoring.resolve_alert(alert.alert_id)
        alert = self.monitoring.get_alert(alert.alert_id)
        assert alert.status == AlertStatus.RESOLVED

        # 验证活跃告警列表中不再包含已解决的告警
        active_alerts = self.monitoring.get_active_alerts()
        assert alert.alert_id not in active_alerts

    def test_metric_data_integrity(self):
        """测试指标数据完整性"""
        # 创建各种类型的指标
        metrics_data = [
            ("string_metric", MetricType.GAUGE, "test_value"),
            ("int_metric", MetricType.COUNTER, 42),
            ("float_metric", MetricType.GAUGE, 3.14159),
            ("bool_metric", MetricType.GAUGE, True),
            ("list_metric", MetricType.GAUGE, [1, 2, 3]),
            ("dict_metric", MetricType.GAUGE, {"key": "value"})
        ]

        for name, mtype, value in metrics_data:
            self.monitoring.create_metric(name, mtype, value)

            # 验证存储和检索
            stored = self.monitoring.get_metric(name)
            assert stored is not None
            assert stored.value == value
            assert stored.metric_type == mtype

            # 验证字典序列化
            data = stored.to_dict()
            assert data["name"] == name
            assert data["type"] == mtype.value
            assert data["value"] == value

    def test_performance_under_load(self):
        """测试负载下的性能"""
        import time

        # 创建大量告警和指标
        alert_count = 100
        metric_count = 50

        # 批量创建告警
        start_time = time.time()
        for i in range(alert_count):
            self.monitoring.create_alert(
                f"perf_alert_{i}",
                AlertLevel.INFO,
                f"Performance alert {i}"
            )
        alert_creation_time = time.time() - start_time

        # 批量创建指标
        start_time = time.time()
        for i in range(metric_count):
            self.monitoring.create_metric(
                f"perf_metric_{i}",
                MetricType.COUNTER,
                initial_value=i
            )
        metric_creation_time = time.time() - start_time

        # 测试查询性能
        start_time = time.time()
        all_alerts = self.monitoring.get_all_alerts()
        all_metrics = self.monitoring.get_all_metrics()
        status = self.monitoring.get_system_status()
        query_time = time.time() - start_time

        # 验证结果
        assert len(all_alerts) == alert_count
        assert len(all_metrics) == metric_count
        assert status["alerts"]["total"] == alert_count
        assert status["metrics"]["total"] == metric_count

        # 性能断言（宽松的限制）
        assert alert_creation_time < 2.0, f"Alert creation too slow: {alert_creation_time}s"
        assert metric_creation_time < 1.0, f"Metric creation too slow: {metric_creation_time}s"
        assert query_time < 0.5, f"Query too slow: {query_time}s"

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        # 测试异常的告警处理器
        def failing_handler(alert):
            raise Exception("Handler failed")

        self.monitoring.add_alert_handler(failing_handler)

        # 创建告警，即使处理器失败也不应影响系统
        alert = self.monitoring.create_alert(
            "error_test", AlertLevel.ERROR, "Error handling test"
        )

        # 验证告警仍然创建成功
        assert alert is not None
        stored = self.monitoring.get_alert(alert.alert_id)
        assert stored is not None

        # 移除失败的处理器
        self.monitoring.remove_alert_handler(failing_handler)

        # 验证系统仍然正常工作
        normal_alert = self.monitoring.create_alert(
            "recovery_test", AlertLevel.INFO, "Recovery test"
        )
        assert normal_alert is not None

    def test_alert_and_metric_integration(self):
        """测试告警和指标的集成"""
        # 创建基于指标的告警逻辑
        response_time_metric = self.monitoring.create_metric(
            "response_time", MetricType.GAUGE, 100, {"endpoint": "/api/test"}
        )

        # 模拟响应时间增加
        self.monitoring.update_metric("response_time", 250)  # 超过阈值

        # 创建告警
        alert = self.monitoring.create_alert(
            "high_response_time",
            AlertLevel.WARNING,
            "Response time is too high",
            details={"current_time": 250, "threshold": 200}
        )

        # 验证告警和指标都正确创建
        assert alert.level == AlertLevel.WARNING
        assert "250" in alert.message or "high" in alert.message

        metric = self.monitoring.get_metric("response_time")
        assert metric.value == 250

        # 验证系统状态包含两者
        status = self.monitoring.get_system_status()
        assert status["alerts"]["total"] >= 1
        assert status["metrics"]["total"] >= 1


if __name__ == '__main__':
    pytest.main([__file__])

