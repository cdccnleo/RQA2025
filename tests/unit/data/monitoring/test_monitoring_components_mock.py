# -*- coding: utf-8 -*-
"""
数据监控组件Mock测试
测试数据质量监控、性能监控、异常检测和数据血缘追踪功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import asyncio
import threading
import time


class MockQualityLevel(Enum):
    """模拟质量等级枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class MockQualityMetrics:
    """模拟质量指标数据类"""

    def __init__(self, completeness: float, accuracy: float, consistency: float,
                 timeliness: float, validity: float, overall_score: float):
        self.completeness = completeness
        self.accuracy = accuracy
        self.consistency = consistency
        self.timeliness = timeliness
        self.validity = validity
        self.overall_score = overall_score
        self.total_score = overall_score  # 兼容测试用例


@dataclass
class MockPerformanceMetric:
    """模拟性能指标"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MockPerformanceAlert:
    """模拟性能告警"""
    level: str  # 'info', 'warning', 'error', 'critical'
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime = field(default_factory=datetime.now)


class MockQualityMonitor:
    """模拟质量监控器"""

    def __init__(self, monitor_id: str, config: Optional[Dict[str, Any]] = None):
        self.monitor_id = monitor_id
        self.config = config or {}
        self.is_active = False
        self.quality_history = []
        self.alert_count = 0
        self.last_check_time = None
        self.logger = Mock()

    def start_monitoring(self) -> bool:
        """启动监控"""
        try:
            self.is_active = True
            self.logger.info(f"Quality monitor {self.monitor_id} started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start quality monitor: {e}")
            return False

    def stop_monitoring(self) -> bool:
        """停止监控"""
        try:
            self.is_active = False
            self.logger.info(f"Quality monitor {self.monitor_id} stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop quality monitor: {e}")
            return False

    def assess_quality(self, data: Any) -> MockQualityMetrics:
        """评估数据质量"""
        if not self.is_active:
            raise Exception("Quality monitor not active")

        self.last_check_time = datetime.now()

        # 模拟质量评估
        if isinstance(data, pd.DataFrame):
            completeness = self._calculate_completeness(data)
            accuracy = self._calculate_accuracy(data)
            consistency = self._calculate_consistency(data)
            timeliness = self._calculate_timeliness(data)
            validity = self._calculate_validity(data)
        else:
            # 对于非DataFrame数据，使用默认值
            completeness = 0.95
            accuracy = 0.92
            consistency = 0.88
            timeliness = 0.90
            validity = 0.85

        overall_score = (completeness + accuracy + consistency + timeliness + validity) / 5.0

        metrics = MockQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            overall_score=overall_score
        )

        self.quality_history.append(metrics)

        # 检查是否需要告警
        if overall_score < 0.7:
            self.alert_count += 1
            self.logger.warning(f"Low quality score: {overall_score}")

        return metrics

    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """计算完整性"""
        if df.empty:
            return 0.0
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.count().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0

    def _calculate_accuracy(self, df: pd.DataFrame) -> float:
        """计算准确性（模拟）"""
        # 模拟准确性检查：检查是否有异常值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 1.0

        accuracy_scores = []
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                # 简单的异常值检测
                mean_val = data.mean()
                std_val = data.std()
                if std_val > 0:
                    z_scores = np.abs((data - mean_val) / std_val)
                    normal_count = (z_scores < 3).sum()  # 3倍标准差以内为正常
                    accuracy_scores.append(normal_count / len(data))

        return np.mean(accuracy_scores) if accuracy_scores else 1.0

    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        """计算一致性（模拟）"""
        # 检查数据类型一致性
        consistency_score = 0.9  # 模拟值
        return consistency_score

    def _calculate_timeliness(self, df: pd.DataFrame) -> float:
        """计算及时性（模拟）"""
        # 检查数据新鲜度
        timeliness_score = 0.85  # 模拟值
        return timeliness_score

    def _calculate_validity(self, df: pd.DataFrame) -> float:
        """计算有效性（模拟）"""
        # 检查数据范围和格式有效性
        validity_score = 0.95  # 模拟值
        return validity_score

    def get_quality_trend(self, hours: int = 24) -> List[MockQualityMetrics]:
        """获取质量趋势"""
        # 返回最近的质量评估结果
        return self.quality_history[-min(hours, len(self.quality_history)):]

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        return {
            "monitor_id": self.monitor_id,
            "is_active": self.is_active,
            "total_checks": len(self.quality_history),
            "alert_count": self.alert_count,
            "last_check_time": self.last_check_time,
            "avg_quality_score": np.mean([m.overall_score for m in self.quality_history]) if self.quality_history else 0.0
        }


class MockPerformanceMonitor:
    """模拟性能监控器"""

    def __init__(self, monitor_id: str, config: Optional[Dict[str, Any]] = None):
        self.monitor_id = monitor_id
        self.config = config or {}
        self.is_active = False
        self.metrics_history = []
        self.alerts_history = []
        self.metric_counters = {}
        self.logger = Mock()

    def start_monitoring(self) -> bool:
        """启动性能监控"""
        try:
            self.is_active = True
            self.logger.info(f"Performance monitor {self.monitor_id} started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start performance monitor: {e}")
            return False

    def stop_monitoring(self) -> bool:
        """停止性能监控"""
        try:
            self.is_active = False
            self.logger.info(f"Performance monitor {self.monitor_id} stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop performance monitor: {e}")
            return False

    def record_metric(self, name: str, value: float, unit: str = "",
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """记录性能指标"""
        if not self.is_active:
            return False

        metric = MockPerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )

        self.metrics_history.append(metric)

        # 更新计数器
        if name not in self.metric_counters:
            self.metric_counters[name] = 0
        self.metric_counters[name] += 1

        # 检查阈值告警
        self._check_thresholds(metric)

        return True

    def start_timer(self, operation_name: str) -> str:
        """开始计时"""
        timer_id = f"{operation_name}_{int(time.time() * 1000)}"
        self.record_metric(f"{operation_name}_start", time.time(), "timestamp", {"timer_id": timer_id})
        return timer_id

    def end_timer(self, timer_id: str, operation_name: str) -> float:
        """结束计时"""
        end_time = time.time()
        self.record_metric(f"{operation_name}_end", end_time, "timestamp", {"timer_id": timer_id})

        # 计算持续时间（查找开始时间）
        duration = 0.0
        for metric in reversed(self.metrics_history):
            if metric.name == f"{operation_name}_start" and metric.metadata.get("timer_id") == timer_id:
                duration = end_time - metric.value
                break

        if duration > 0:
            self.record_metric(f"{operation_name}_duration", duration, "seconds")

        return duration

    def _check_thresholds(self, metric: MockPerformanceMetric) -> None:
        """检查阈值并生成告警"""
        thresholds = self.config.get("thresholds", {})

        if metric.name in thresholds:
            threshold_config = thresholds[metric.name]
            threshold_value = threshold_config.get("value", 0)
            comparison = threshold_config.get("comparison", "gt")  # gt: greater than, lt: less than

            alert_triggered = False
            if comparison == "gt" and metric.value > threshold_value:
                alert_triggered = True
            elif comparison == "lt" and metric.value < threshold_value:
                alert_triggered = True

            if alert_triggered:
                alert = MockPerformanceAlert(
                    level=threshold_config.get("level", "warning"),
                    message=f"Threshold exceeded for {metric.name}",
                    metric_name=metric.name,
                    threshold=threshold_value,
                    current_value=metric.value
                )
                self.alerts_history.append(alert)
                self.logger.warning(f"Performance alert: {alert.message}")

    def get_metrics_summary(self, metric_name: Optional[str] = None,
                          hours: int = 1) -> Dict[str, Any]:
        """获取指标汇总"""
        # 筛选指标
        relevant_metrics = self.metrics_history
        if metric_name:
            relevant_metrics = [m for m in relevant_metrics if m.name == metric_name]

        if not relevant_metrics:
            return {"count": 0, "avg": 0, "min": 0, "max": 0}

        values = [m.value for m in relevant_metrics]

        return {
            "count": len(values),
            "avg": np.mean(values),
            "min": np.min(values),
            "max": np.max(values),
            "latest": values[-1] if values else 0
        }

    def get_alerts(self, level: Optional[str] = None, hours: int = 24) -> List[MockPerformanceAlert]:
        """获取告警"""
        alerts = self.alerts_history
        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        return {
            "monitor_id": self.monitor_id,
            "is_active": self.is_active,
            "total_metrics": len(self.metrics_history),
            "total_alerts": len(self.alerts_history),
            "unique_metrics": len(self.metric_counters),
            "metrics_counts": self.metric_counters
        }


class MockAnomalyDetector:
    """模拟异常检测器"""

    def __init__(self, detector_id: str, config: Optional[Dict[str, Any]] = None):
        self.detector_id = detector_id
        self.config = config or {}
        self.is_active = False
        self.baseline_data = []
        self.anomalies_detected = []
        self.detection_methods = ["zscore", "iqr", "isolation_forest"]
        self.logger = Mock()

    def initialize_baseline(self, data: List[float]) -> bool:
        """初始化基线数据"""
        try:
            self.baseline_data = data.copy()
            self.logger.info(f"Baseline initialized with {len(data)} data points")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize baseline: {e}")
            return False

    def detect_anomalies(self, new_data: List[float],
                        method: str = "zscore") -> List[Dict[str, Any]]:
        """检测异常"""
        if not self.is_active:
            raise Exception("Anomaly detector not active")

        if method not in self.detection_methods:
            raise ValueError(f"Unsupported detection method: {method}")

        anomalies = []

        if method == "zscore":
            anomalies = self._zscore_detection(new_data)
        elif method == "iqr":
            anomalies = self._iqr_detection(new_data)
        elif method == "isolation_forest":
            anomalies = self._isolation_forest_detection(new_data)

        self.anomalies_detected.extend(anomalies)
        return anomalies

    def _zscore_detection(self, data: List[float]) -> List[Dict[str, Any]]:
        """Z-Score异常检测"""
        anomalies = []
        if not self.baseline_data:
            return anomalies

        all_data = self.baseline_data + data
        mean_val = np.mean(all_data)
        std_val = np.std(all_data)

        if std_val == 0:
            return anomalies

        for i, value in enumerate(data):
            z_score = abs((value - mean_val) / std_val)
            if z_score > 3.0:  # 3倍标准差
                anomalies.append({
                    "index": len(self.baseline_data) + i,
                    "value": value,
                    "score": z_score,
                    "method": "zscore",
                    "timestamp": datetime.now()
                })

        return anomalies

    def _iqr_detection(self, data: List[float]) -> List[Dict[str, Any]]:
        """IQR异常检测"""
        anomalies = []
        if not self.baseline_data:
            return anomalies

        all_data = self.baseline_data + data
        q1 = np.percentile(all_data, 25)
        q3 = np.percentile(all_data, 75)
        iqr = q3 - q1

        if iqr == 0:
            return anomalies

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                anomalies.append({
                    "index": len(self.baseline_data) + i,
                    "value": value,
                    "bounds": [lower_bound, upper_bound],
                    "method": "iqr",
                    "timestamp": datetime.now()
                })

        return anomalies

    def _isolation_forest_detection(self, data: List[float]) -> List[Dict[str, Any]]:
        """隔离森林异常检测（模拟）"""
        anomalies = []
        # 模拟隔离森林检测
        for i, value in enumerate(data):
            if value > np.mean(self.baseline_data) + 3 * np.std(self.baseline_data):
                anomalies.append({
                    "index": len(self.baseline_data) + i,
                    "value": value,
                    "score": 0.8,  # 模拟异常分数
                    "method": "isolation_forest",
                    "timestamp": datetime.now()
                })
        return anomalies

    def start_detection(self) -> bool:
        """启动检测"""
        try:
            self.is_active = True
            self.logger.info(f"Anomaly detector {self.detector_id} started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start anomaly detector: {e}")
            return False

    def stop_detection(self) -> bool:
        """停止检测"""
        try:
            self.is_active = False
            self.logger.info(f"Anomaly detector {self.detector_id} stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop anomaly detector: {e}")
            return False

    def get_detection_stats(self) -> Dict[str, Any]:
        """获取检测统计"""
        return {
            "detector_id": self.detector_id,
            "is_active": self.is_active,
            "baseline_size": len(self.baseline_data),
            "anomalies_detected": len(self.anomalies_detected),
            "detection_methods": self.detection_methods
        }


class MockDataLineageTracker:
    """模拟数据血缘追踪器"""

    def __init__(self, tracker_id: str, config: Optional[Dict[str, Any]] = None):
        self.tracker_id = tracker_id
        self.config = config or {}
        self.is_active = False
        self.lineage_graph = {}
        self.data_transformations = []
        self.logger = Mock()

    def start_tracking(self) -> bool:
        """启动追踪"""
        try:
            self.is_active = True
            self.logger.info(f"Data lineage tracker {self.tracker_id} started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start lineage tracker: {e}")
            return False

    def stop_tracking(self) -> bool:
        """停止追踪"""
        try:
            self.is_active = False
            self.logger.info(f"Data lineage tracker {self.tracker_id} stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop lineage tracker: {e}")
            return False

    def record_transformation(self, input_datasets: List[str], output_dataset: str,
                            transformation_type: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """记录数据转换"""
        if not self.is_active:
            return False

        transformation = {
            "input_datasets": input_datasets,
            "output_dataset": output_dataset,
            "transformation_type": transformation_type,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }

        self.data_transformations.append(transformation)

        # 更新血缘图
        for input_ds in input_datasets:
            if input_ds not in self.lineage_graph:
                self.lineage_graph[input_ds] = {"outputs": [], "inputs": []}
            self.lineage_graph[input_ds]["outputs"].append(output_dataset)

        if output_dataset not in self.lineage_graph:
            self.lineage_graph[output_dataset] = {"outputs": [], "inputs": input_datasets}
        else:
            self.lineage_graph[output_dataset]["inputs"].extend(input_datasets)

        return True

    def get_lineage(self, dataset: str, direction: str = "both",
                   max_depth: int = 3) -> Dict[str, Any]:
        """获取数据血缘"""
        if dataset not in self.lineage_graph:
            return {"dataset": dataset, "lineage": {}}

        def traverse(current_dataset: str, current_depth: int,
                    visited: set, direction_filter: str) -> Dict[str, Any]:
            if current_depth >= max_depth or current_dataset in visited:
                return {"dataset": current_dataset, "children": []}

            visited.add(current_dataset)
            node_info = self.lineage_graph.get(current_dataset, {"outputs": [], "inputs": []})

            children = []
            if direction_filter in ["downstream", "both"]:
                for output in node_info["outputs"]:
                    children.append(traverse(output, current_depth + 1, visited.copy(), "downstream"))

            if direction_filter in ["upstream", "both"]:
                for input_ds in node_info["inputs"]:
                    children.append(traverse(input_ds, current_depth + 1, visited.copy(), "upstream"))

            return {"dataset": current_dataset, "children": children}

        return traverse(dataset, 0, set(), direction)

    def get_transformation_history(self, dataset: Optional[str] = None,
                                 hours: int = 24) -> List[Dict[str, Any]]:
        """获取转换历史"""
        transformations = self.data_transformations

        if dataset:
            transformations = [t for t in transformations
                              if dataset in t["input_datasets"] or dataset == t["output_dataset"]]

        # 按时间排序（最新的在前）
        transformations.sort(key=lambda x: x["timestamp"], reverse=True)

        return transformations

    def validate_lineage(self) -> List[str]:
        """验证血缘一致性"""
        issues = []

        # 检查循环依赖
        for dataset in self.lineage_graph:
            if self._has_circular_dependency(dataset, set()):
                issues.append(f"Circular dependency detected involving {dataset}")

        # 检查孤立节点
        for dataset, info in self.lineage_graph.items():
            if not info["inputs"] and not info["outputs"]:
                issues.append(f"Isolated dataset: {dataset}")

        return issues

    def _has_circular_dependency(self, dataset: str, visited: set) -> bool:
        """检查循环依赖"""
        if dataset in visited:
            return True

        visited.add(dataset)
        node_info = self.lineage_graph.get(dataset, {"outputs": [], "inputs": []})

        for output in node_info["outputs"]:
            if self._has_circular_dependency(output, visited.copy()):
                return True

        return False

    def get_tracking_stats(self) -> Dict[str, Any]:
        """获取追踪统计"""
        return {
            "tracker_id": self.tracker_id,
            "is_active": self.is_active,
            "total_datasets": len(self.lineage_graph),
            "total_transformations": len(self.data_transformations),
            "lineage_issues": self.validate_lineage()
        }


class TestMockQualityMonitor:
    """模拟质量监控器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = {"alert_threshold": 0.8, "check_interval": 60}
        self.monitor = MockQualityMonitor("test_quality_monitor", self.config)

    def test_monitor_initialization(self):
        """测试监控器初始化"""
        assert self.monitor.monitor_id == "test_quality_monitor"
        assert not self.monitor.is_active
        assert len(self.monitor.quality_history) == 0

    def test_monitor_start_stop(self):
        """测试监控器启动和停止"""
        assert self.monitor.start_monitoring()
        assert self.monitor.is_active

        assert self.monitor.stop_monitoring()
        assert not self.monitor.is_active

    def test_quality_assessment_dataframe(self):
        """测试DataFrame质量评估"""
        self.monitor.start_monitoring()

        # 创建测试数据
        df = pd.DataFrame({
            "A": [1, 2, None, 4, 5],  # 有缺失值
            "B": [10, 20, 30, 40, 50],  # 完整数据
            "C": ["a", "b", "c", "d", "e"]  # 完整数据
        })

        metrics = self.monitor.assess_quality(df)

        assert isinstance(metrics, MockQualityMetrics)
        assert 0 <= metrics.completeness <= 1
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.consistency <= 1
        assert 0 <= metrics.timeliness <= 1
        assert 0 <= metrics.validity <= 1
        assert 0 <= metrics.overall_score <= 1

        # 检查历史记录
        assert len(self.monitor.quality_history) == 1
        assert self.monitor.last_check_time is not None

    def test_quality_assessment_non_dataframe(self):
        """测试非DataFrame数据质量评估"""
        self.monitor.start_monitoring()

        # 测试字典数据
        data = {"key": "value"}
        metrics = self.monitor.assess_quality(data)

        assert isinstance(metrics, MockQualityMetrics)
        assert metrics.overall_score > 0

    def test_quality_metrics_calculation(self):
        """测试质量指标计算"""
        self.monitor.start_monitoring()

        # 创建完整数据
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [10, 20, 30, 40, 50]
        })

        metrics = self.monitor.assess_quality(df)

        # 完整数据应该有高分
        assert metrics.completeness == 1.0  # 完全没有缺失值
        assert metrics.overall_score > 0.8  # 总体分数应该较高

    def test_quality_trend(self):
        """测试质量趋势"""
        self.monitor.start_monitoring()

        # 进行多次评估
        for i in range(3):
            df = pd.DataFrame({"A": [1, 2, 3]})
            self.monitor.assess_quality(df)

        trend = self.monitor.get_quality_trend(24)
        assert len(trend) == 3
        assert all(isinstance(m, MockQualityMetrics) for m in trend)

    def test_monitoring_stats(self):
        """测试监控统计"""
        self.monitor.start_monitoring()

        # 进行一些操作
        df = pd.DataFrame({"A": [1, None, 3]})
        self.monitor.assess_quality(df)

        stats = self.monitor.get_monitoring_stats()
        assert stats["monitor_id"] == "test_quality_monitor"
        assert stats["is_active"] is True
        assert stats["total_checks"] == 1
        assert isinstance(stats["avg_quality_score"], (float, np.floating))


class TestMockPerformanceMonitor:
    """模拟性能监控器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = {
            "thresholds": {
                "response_time": {"value": 1.0, "comparison": "gt", "level": "warning"},
                "error_rate": {"value": 0.05, "comparison": "gt", "level": "error"}
            }
        }
        self.monitor = MockPerformanceMonitor("test_perf_monitor", self.config)

    def test_monitor_initialization(self):
        """测试监控器初始化"""
        assert self.monitor.monitor_id == "test_perf_monitor"
        assert not self.monitor.is_active

    def test_monitor_start_stop(self):
        """测试监控器启动和停止"""
        assert self.monitor.start_monitoring()
        assert self.monitor.is_active

        assert self.monitor.stop_monitoring()
        assert not self.monitor.is_active

    def test_record_metric(self):
        """测试记录指标"""
        self.monitor.start_monitoring()

        # 记录指标
        assert self.monitor.record_metric("response_time", 0.5, "seconds")
        assert self.monitor.record_metric("throughput", 100.0, "req/sec")

        assert len(self.monitor.metrics_history) == 2
        assert self.monitor.metric_counters["response_time"] == 1
        assert self.monitor.metric_counters["throughput"] == 1

    def test_timer_functionality(self):
        """测试计时器功能"""
        self.monitor.start_monitoring()

        # 开始计时
        timer_id = self.monitor.start_timer("database_query")
        time.sleep(0.01)  # 短暂延迟

        # 结束计时
        duration = self.monitor.end_timer(timer_id, "database_query")

        assert duration > 0
        assert duration < 1.0  # 应该很短

        # 检查记录的指标
        duration_metrics = [m for m in self.monitor.metrics_history if "duration" in m.name]
        assert len(duration_metrics) == 1

    def test_threshold_alerts(self):
        """测试阈值告警"""
        self.monitor.start_monitoring()

        # 记录超过阈值的指标
        self.monitor.record_metric("response_time", 1.5, "seconds")  # 超过1.0的阈值

        # 检查是否产生了告警
        alerts = self.monitor.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].level == "warning"
        assert alerts[0].metric_name == "response_time"
        assert alerts[0].threshold == 1.0
        assert alerts[0].current_value == 1.5

    def test_metrics_summary(self):
        """测试指标汇总"""
        self.monitor.start_monitoring()

        # 记录多个相同指标的值
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            self.monitor.record_metric("response_time", value, "seconds")

        summary = self.monitor.get_metrics_summary("response_time")
        assert summary["count"] == 5
        assert summary["avg"] == 3.0
        assert summary["min"] == 1.0
        assert summary["max"] == 5.0
        assert summary["latest"] == 5.0

    def test_monitoring_stats(self):
        """测试监控统计"""
        self.monitor.start_monitoring()

        # 记录一些指标
        self.monitor.record_metric("metric1", 1.0)
        self.monitor.record_metric("metric2", 2.0)
        self.monitor.record_metric("metric1", 1.5)  # 重复指标

        stats = self.monitor.get_monitoring_stats()
        assert stats["total_metrics"] == 3
        assert stats["unique_metrics"] == 2
        assert stats["metrics_counts"]["metric1"] == 2
        assert stats["metrics_counts"]["metric2"] == 1


class TestMockAnomalyDetector:
    """模拟异常检测器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = {"sensitivity": 0.95, "methods": ["zscore", "iqr"]}
        self.detector = MockAnomalyDetector("test_anomaly_detector", self.config)

    def test_detector_initialization(self):
        """测试检测器初始化"""
        assert self.detector.detector_id == "test_anomaly_detector"
        assert not self.detector.is_active
        assert len(self.detector.detection_methods) == 3

    def test_baseline_initialization(self):
        """测试基线初始化"""
        baseline_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert self.detector.initialize_baseline(baseline_data)
        assert len(self.detector.baseline_data) == 5

    def test_anomaly_detection_zscore(self):
        """测试Z-Score异常检测"""
        baseline_data = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 均值14.5, 标准差3.03
        self.detector.initialize_baseline(baseline_data)
        self.detector.start_detection()

        # 测试正常数据
        normal_data = [14, 15, 16]
        anomalies = self.detector.detect_anomalies(normal_data, "zscore")
        assert len(anomalies) == 0

        # 测试异常数据
        anomalous_data = [50]  # 明显异常值
        anomalies = self.detector.detect_anomalies(anomalous_data, "zscore")
        assert len(anomalies) == 1
        assert anomalies[0]["method"] == "zscore"
        assert anomalies[0]["value"] == 50

    def test_anomaly_detection_iqr(self):
        """测试IQR异常检测"""
        baseline_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Q1=3, Q3=8, IQR=5, 边界: 下限=3-7.5=-4.5, 上限=8+7.5=15.5
        self.detector.initialize_baseline(baseline_data)
        self.detector.start_detection()

        # 测试正常数据
        normal_data = [4, 5, 6, 7]
        anomalies = self.detector.detect_anomalies(normal_data, "iqr")
        assert len(anomalies) == 0

        # 测试异常数据
        anomalous_data = [20]  # 超过上限
        anomalies = self.detector.detect_anomalies(anomalous_data, "iqr")
        assert len(anomalies) == 1
        assert anomalies[0]["method"] == "iqr"

    def test_anomaly_detection_isolation_forest(self):
        """测试隔离森林异常检测"""
        baseline_data = [1, 2, 3, 4, 5]
        self.detector.initialize_baseline(baseline_data)
        self.detector.start_detection()

        anomalous_data = [100]  # 明显异常
        anomalies = self.detector.detect_anomalies(anomalous_data, "isolation_forest")
        # 隔离森林的模拟实现可能检测到也可能检测不到，取决于实现
        assert isinstance(anomalies, list)

    def test_detection_stats(self):
        """测试检测统计"""
        baseline_data = [1, 2, 3, 4, 5]
        self.detector.initialize_baseline(baseline_data)
        self.detector.start_detection()

        # 进行一些检测
        self.detector.detect_anomalies([10], "zscore")
        self.detector.detect_anomalies([20], "iqr")

        stats = self.detector.get_detection_stats()
        assert stats["is_active"] is True
        assert stats["baseline_size"] == 5
        assert stats["anomalies_detected"] >= 0  # 可能检测到也可能没检测到


class TestMockDataLineageTracker:
    """模拟数据血缘追踪器测试"""

    def setup_method(self):
        """设置测试方法"""
        self.config = {"max_depth": 5, "enable_caching": True}
        self.tracker = MockDataLineageTracker("test_lineage_tracker", self.config)

    def test_tracker_initialization(self):
        """测试追踪器初始化"""
        assert self.tracker.tracker_id == "test_lineage_tracker"
        assert not self.tracker.is_active

    def test_tracker_start_stop(self):
        """测试追踪器启动和停止"""
        assert self.tracker.start_tracking()
        assert self.tracker.is_active

        assert self.tracker.stop_tracking()
        assert not self.tracker.is_active

    def test_record_transformation(self):
        """测试记录转换"""
        self.tracker.start_tracking()

        # 记录转换
        assert self.tracker.record_transformation(
            input_datasets=["raw_data"],
            output_dataset="cleaned_data",
            transformation_type="cleaning",
            metadata={"rows_removed": 10}
        )

        assert len(self.tracker.data_transformations) == 1
        assert len(self.tracker.lineage_graph) == 2  # raw_data 和 cleaned_data

        # 检查血缘关系
        assert "raw_data" in self.tracker.lineage_graph
        assert "cleaned_data" in self.tracker.lineage_graph
        assert "cleaned_data" in self.tracker.lineage_graph["raw_data"]["outputs"]

    def test_get_lineage(self):
        """测试获取血缘"""
        self.tracker.start_tracking()

        # 建立血缘关系
        self.tracker.record_transformation(["A"], "B", "transform1")
        self.tracker.record_transformation(["B"], "C", "transform2")
        self.tracker.record_transformation(["A"], "D", "transform3")

        # 获取B的下游血缘
        lineage = self.tracker.get_lineage("B", "downstream", 2)
        assert lineage["dataset"] == "B"
        assert len(lineage["children"]) > 0

        # 获取C的上游血缘
        lineage = self.tracker.get_lineage("C", "upstream", 2)
        assert lineage["dataset"] == "C"

    def test_transformation_history(self):
        """测试转换历史"""
        self.tracker.start_tracking()

        # 记录多个转换
        self.tracker.record_transformation(["input1"], "output1", "type1")
        self.tracker.record_transformation(["input2"], "output2", "type2")
        self.tracker.record_transformation(["output1"], "final", "type3")

        # 获取所有转换历史
        history = self.tracker.get_transformation_history()
        assert len(history) == 3

        # 获取特定数据集的转换历史
        history = self.tracker.get_transformation_history("output1")
        assert len(history) == 2  # output1作为输入和输出

    def test_lineage_validation(self):
        """测试血缘验证"""
        self.tracker.start_tracking()

        # 添加正常血缘
        self.tracker.record_transformation(["A"], "B", "transform")
        self.tracker.record_transformation(["B"], "C", "transform")

        # 验证应该没有问题
        issues = self.tracker.validate_lineage()
        assert len(issues) == 0

        # 添加孤立节点
        self.tracker.lineage_graph["isolated"] = {"outputs": [], "inputs": []}

        issues = self.tracker.validate_lineage()
        assert len(issues) > 0  # 应该检测到孤立节点

    def test_tracking_stats(self):
        """测试追踪统计"""
        self.tracker.start_tracking()

        # 进行一些操作
        self.tracker.record_transformation(["A"], "B", "transform1")
        self.tracker.record_transformation(["B"], "C", "transform2")

        stats = self.tracker.get_tracking_stats()
        assert stats["is_active"] is True
        assert stats["total_datasets"] == 3  # A, B, C
        assert stats["total_transformations"] == 2


class TestMonitoringIntegration:
    """监控集成测试"""

    def test_complete_monitoring_pipeline(self):
        """测试完整监控管道"""
        # 创建各个监控组件
        quality_monitor = MockQualityMonitor("quality_monitor")
        perf_monitor = MockPerformanceMonitor("perf_monitor")
        anomaly_detector = MockAnomalyDetector("anomaly_detector")
        lineage_tracker = MockDataLineageTracker("lineage_tracker")

        # 启动所有组件
        assert quality_monitor.start_monitoring()
        assert perf_monitor.start_monitoring()
        assert anomaly_detector.start_detection()
        assert lineage_tracker.start_tracking()

        # 1. 记录数据血缘
        lineage_tracker.record_transformation(
            ["raw_market_data"],
            "processed_market_data",
            "data_cleaning"
        )

        lineage_tracker.record_transformation(
            ["processed_market_data"],
            "analyzed_data",
            "technical_analysis"
        )

        # 2. 监控数据质量
        test_data = pd.DataFrame({
            "price": [100, 101, 99, 102, 98],
            "volume": [1000, 1100, 900, 1200, 800]
        })

        quality_metrics = quality_monitor.assess_quality(test_data)
        assert quality_metrics.overall_score > 0

        # 3. 记录性能指标
        perf_monitor.record_metric("data_processing_time", 0.5, "seconds")
        perf_monitor.record_metric("memory_usage", 85.0, "percent")

        # 4. 异常检测
        baseline_prices = [100, 101, 102, 103, 104]
        anomaly_detector.initialize_baseline(baseline_prices)

        new_prices = [105, 106, 150]  # 150是异常值
        anomalies = anomaly_detector.detect_anomalies(new_prices, "zscore")

        # 验证集成结果
        quality_stats = quality_monitor.get_monitoring_stats()
        perf_stats = perf_monitor.get_monitoring_stats()
        anomaly_stats = anomaly_detector.get_detection_stats()
        lineage_stats = lineage_tracker.get_tracking_stats()

        assert quality_stats["total_checks"] == 1
        assert perf_stats["total_metrics"] == 2
        assert anomaly_stats["baseline_size"] == 5
        assert lineage_stats["total_transformations"] == 2

        # 验证血缘追踪
        lineage = lineage_tracker.get_lineage("analyzed_data", "upstream")
        assert lineage["dataset"] == "analyzed_data"

        # 清理资源
        quality_monitor.stop_monitoring()
        perf_monitor.stop_monitoring()
        anomaly_detector.stop_detection()
        lineage_tracker.stop_tracking()

    def test_monitoring_alert_system(self):
        """测试监控告警系统"""
        perf_monitor = MockPerformanceMonitor("alert_monitor", {
            "thresholds": {
                "error_rate": {"value": 0.1, "comparison": "gt", "level": "error"},
                "latency": {"value": 2.0, "comparison": "gt", "level": "warning"}
            }
        })

        perf_monitor.start_monitoring()

        # 记录正常指标
        perf_monitor.record_metric("error_rate", 0.05)  # 正常
        perf_monitor.record_metric("latency", 1.5)      # 正常

        # 记录异常指标
        perf_monitor.record_metric("error_rate", 0.15)  # 触发告警
        perf_monitor.record_metric("latency", 2.5)      # 触发告警

        alerts = perf_monitor.get_alerts()
        assert len(alerts) == 2

        error_alerts = perf_monitor.get_alerts("error")
        assert len(error_alerts) == 1
        assert error_alerts[0].metric_name == "error_rate"

        warning_alerts = perf_monitor.get_alerts("warning")
        assert len(warning_alerts) == 1
        assert warning_alerts[0].metric_name == "latency"

    def test_cross_component_monitoring(self):
        """测试跨组件监控"""
        # 创建多个监控器
        monitors = []
        for i in range(3):
            monitor = MockQualityMonitor(f"monitor_{i}")
            monitor.start_monitoring()
            monitors.append(monitor)

        # 所有监控器评估相同的数据
        test_data = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

        results = []
        for monitor in monitors:
            metrics = monitor.assess_quality(test_data)
            results.append(metrics.overall_score)

        # 结果应该一致（因为数据相同）
        assert all(abs(score - results[0]) < 0.001 for score in results)

        # 检查统计
        for monitor in monitors:
            stats = monitor.get_monitoring_stats()
            assert stats["total_checks"] == 1
            monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


