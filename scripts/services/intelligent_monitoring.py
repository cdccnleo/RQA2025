#!/usr/bin/env python3
"""
智能化监控系统

实现自动调优、预测性维护和智能监控功能
"""

from src.core import EventBus, ServiceContainer
from src.services.micro_service import MicroService
import os
import sys
import json
import time
import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """指标数据"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: MetricType


@dataclass
class Alert:
    """告警信息"""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    service_name: str
    metric_name: str
    threshold: float
    current_value: float


@dataclass
class Prediction:
    """预测结果"""
    service_name: str
    metric_name: str
    predicted_value: float
    confidence: float
    timestamp: datetime
    time_horizon: timedelta


class IntelligentMonitoring:
    """智能化监控系统"""

    def __init__(self, config_path: str = None):
        """
        初始化智能化监控系统

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or "config/intelligent_monitoring.yaml"
        self.config = self._load_config()
        self.event_bus = EventBus()
        self.container = ServiceContainer()
        self.micro_service = MicroService(self.event_bus, self.container)
        self.logger = logging.getLogger(__name__)

        # 数据存储
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = []
        self.predictions = []

        # 机器学习模型
        self.ml_models = {}

        # 监控状态
        self.monitoring_active = False
        self.auto_tuning_active = False
        self.predictive_maintenance_active = False

        # 线程锁
        self._lock = threading.Lock()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "monitoring": {
                "enabled": True,
                "collection_interval": 30,
                "retention_days": 30
            },
            "alerts": {
                "cpu_threshold": 80.0,
                "memory_threshold": 85.0,
                "response_time_threshold": 1000.0,
                "error_rate_threshold": 5.0
            },
            "auto_tuning": {
                "enabled": True,
                "cpu_target": 70.0,
                "memory_target": 75.0,
                "response_time_target": 500.0
            },
            "predictive_maintenance": {
                "enabled": True,
                "prediction_horizon_hours": 24,
                "confidence_threshold": 0.8
            },
            "ml_models": {
                "cpu_prediction": "linear_regression",
                "memory_prediction": "random_forest",
                "response_time_prediction": "lstm"
            }
        }

    async def start_monitoring(self) -> bool:
        """启动监控系统"""
        try:
            self.logger.info("启动智能化监控系统...")

            # 启动微服务框架
            if not self.micro_service.start():
                raise Exception("微服务框架启动失败")

            # 启动监控线程
            self.monitoring_active = True
            self._start_monitoring_thread()

            # 启动自动调优
            if self.config["auto_tuning"]["enabled"]:
                await self._start_auto_tuning()

            # 启动预测性维护
            if self.config["predictive_maintenance"]["enabled"]:
                await self._start_predictive_maintenance()

            self.logger.info("智能化监控系统启动完成")
            return True

        except Exception as e:
            self.logger.error(f"启动监控系统失败: {e}")
            return False

    def _start_monitoring_thread(self):
        """启动监控线程"""
        def monitoring_worker():
            while self.monitoring_active:
                try:
                    self._collect_metrics()
                    time.sleep(self.config["monitoring"]["collection_interval"])
                except Exception as e:
                    self.logger.error(f"监控数据收集失败: {e}")

        thread = threading.Thread(target=monitoring_worker, daemon=True)
        thread.start()

    def _collect_metrics(self):
        """收集指标数据"""
        with self._lock:
            # 收集服务健康状态
            services = self.micro_service.list_services()
            for service in services:
                health = self.micro_service.check_service_health(service.service_id)

                # CPU使用率
                cpu_metric = Metric(
                    name="cpu_usage",
                    value=self._get_cpu_usage(service.service_id),
                    timestamp=datetime.now(),
                    labels={"service": service.service_name},
                    metric_type=MetricType.GAUGE
                )
                self._store_metric(cpu_metric)

                # 内存使用率
                memory_metric = Metric(
                    name="memory_usage",
                    value=self._get_memory_usage(service.service_id),
                    timestamp=datetime.now(),
                    labels={"service": service.service_name},
                    metric_type=MetricType.GAUGE
                )
                self._store_metric(memory_metric)

                # 响应时间
                response_time_metric = Metric(
                    name="response_time",
                    value=self._get_response_time(service.service_id),
                    timestamp=datetime.now(),
                    labels={"service": service.service_name},
                    metric_type=MetricType.HISTOGRAM
                )
                self._store_metric(response_time_metric)

                # 错误率
                error_rate_metric = Metric(
                    name="error_rate",
                    value=self._get_error_rate(service.service_id),
                    timestamp=datetime.now(),
                    labels={"service": service.service_name},
                    metric_type=MetricType.GAUGE
                )
                self._store_metric(error_rate_metric)

    def _store_metric(self, metric: Metric):
        """存储指标数据"""
        key = f"{metric.name}_{metric.labels.get('service', 'unknown')}"
        self.metrics_history[key].append(metric)

    def _get_cpu_usage(self, service_id: str) -> float:
        """获取CPU使用率（模拟）"""
        # 这里应该从实际的监控系统获取数据
        return np.random.uniform(20, 90)

    def _get_memory_usage(self, service_id: str) -> float:
        """获取内存使用率（模拟）"""
        return np.random.uniform(30, 85)

    def _get_response_time(self, service_id: str) -> float:
        """获取响应时间（模拟）"""
        return np.random.uniform(50, 800)

    def _get_error_rate(self, service_id: str) -> float:
        """获取错误率（模拟）"""
        return np.random.uniform(0, 10)

    async def _start_auto_tuning(self):
        """启动自动调优"""
        self.auto_tuning_active = True
        self.logger.info("启动自动调优系统...")

        # 启动自动调优线程
        def auto_tuning_worker():
            while self.auto_tuning_active:
                try:
                    self._perform_auto_tuning()
                    time.sleep(60)  # 每分钟检查一次
                except Exception as e:
                    self.logger.error(f"自动调优失败: {e}")

        thread = threading.Thread(target=auto_tuning_worker, daemon=True)
        thread.start()

    def _perform_auto_tuning(self):
        """执行自动调优"""
        with self._lock:
            for service_name in self._get_service_names():
                # 检查CPU使用率
                cpu_usage = self._get_latest_metric(f"cpu_usage_{service_name}")
                if cpu_usage and cpu_usage > self.config["alerts"]["cpu_threshold"]:
                    self._scale_service(service_name, "up")

                # 检查内存使用率
                memory_usage = self._get_latest_metric(f"memory_usage_{service_name}")
                if memory_usage and memory_usage > self.config["alerts"]["memory_threshold"]:
                    self._scale_service(service_name, "up")

                # 检查响应时间
                response_time = self._get_latest_metric(f"response_time_{service_name}")
                if response_time and response_time > self.config["alerts"]["response_time_threshold"]:
                    self._optimize_service(service_name)

    def _get_latest_metric(self, metric_name: str) -> Optional[float]:
        """获取最新指标值"""
        if metric_name in self.metrics_history and self.metrics_history[metric_name]:
            return self.metrics_history[metric_name][-1].value
        return None

    def _get_service_names(self) -> List[str]:
        """获取服务名称列表"""
        services = self.micro_service.list_services()
        return [service.service_name for service in services]

    def _scale_service(self, service_name: str, direction: str):
        """扩缩容服务"""
        self.logger.info(f"自动调优: {service_name} {direction}")
        # 这里应该调用Kubernetes API进行扩缩容
        # 目前只是记录日志

    def _optimize_service(self, service_name: str):
        """优化服务性能"""
        self.logger.info(f"优化服务性能: {service_name}")
        # 这里可以实现各种优化策略
        # 比如调整缓存、连接池等

    async def _start_predictive_maintenance(self):
        """启动预测性维护"""
        self.predictive_maintenance_active = True
        self.logger.info("启动预测性维护系统...")

        # 启动预测线程
        def prediction_worker():
            while self.predictive_maintenance_active:
                try:
                    self._perform_predictions()
                    time.sleep(300)  # 每5分钟预测一次
                except Exception as e:
                    self.logger.error(f"预测性维护失败: {e}")

        thread = threading.Thread(target=prediction_worker, daemon=True)
        thread.start()

    def _perform_predictions(self):
        """执行预测"""
        with self._lock:
            for service_name in self._get_service_names():
                # 预测CPU使用率
                cpu_prediction = self._predict_cpu_usage(service_name)
                if cpu_prediction:
                    self.predictions.append(cpu_prediction)

                # 预测内存使用率
                memory_prediction = self._predict_memory_usage(service_name)
                if memory_prediction:
                    self.predictions.append(memory_prediction)

                # 预测响应时间
                response_time_prediction = self._predict_response_time(service_name)
                if response_time_prediction:
                    self.predictions.append(response_time_prediction)

    def _predict_cpu_usage(self, service_name: str) -> Optional[Prediction]:
        """预测CPU使用率"""
        metric_name = f"cpu_usage_{service_name}"
        if metric_name not in self.metrics_history:
            return None

        # 简单的线性预测（实际应该使用更复杂的ML模型）
        metrics = list(self.metrics_history[metric_name])
        if len(metrics) < 10:
            return None

        values = [m.value for m in metrics[-10:]]
        trend = np.polyfit(range(len(values)), values, 1)

        # 预测24小时后的值
        future_value = trend[0] * 24 + trend[1]
        confidence = 0.8  # 简化处理

        return Prediction(
            service_name=service_name,
            metric_name="cpu_usage",
            predicted_value=max(0, min(100, future_value)),
            confidence=confidence,
            timestamp=datetime.now(),
            time_horizon=timedelta(hours=24)
        )

    def _predict_memory_usage(self, service_name: str) -> Optional[Prediction]:
        """预测内存使用率"""
        metric_name = f"memory_usage_{service_name}"
        if metric_name not in self.metrics_history:
            return None

        metrics = list(self.metrics_history[metric_name])
        if len(metrics) < 10:
            return None

        values = [m.value for m in metrics[-10:]]
        trend = np.polyfit(range(len(values)), values, 1)

        future_value = trend[0] * 24 + trend[1]
        confidence = 0.75

        return Prediction(
            service_name=service_name,
            metric_name="memory_usage",
            predicted_value=max(0, min(100, future_value)),
            confidence=confidence,
            timestamp=datetime.now(),
            time_horizon=timedelta(hours=24)
        )

    def _predict_response_time(self, service_name: str) -> Optional[Prediction]:
        """预测响应时间"""
        metric_name = f"response_time_{service_name}"
        if metric_name not in self.metrics_history:
            return None

        metrics = list(self.metrics_history[metric_name])
        if len(metrics) < 10:
            return None

        values = [m.value for m in metrics[-10:]]
        trend = np.polyfit(range(len(values)), values, 1)

        future_value = trend[0] * 24 + trend[1]
        confidence = 0.7

        return Prediction(
            service_name=service_name,
            metric_name="response_time",
            predicted_value=max(0, future_value),
            confidence=confidence,
            timestamp=datetime.now(),
            time_horizon=timedelta(hours=24)
        )

    def check_alerts(self):
        """检查告警"""
        with self._lock:
            for service_name in self._get_service_names():
                # 检查CPU告警
                cpu_usage = self._get_latest_metric(f"cpu_usage_{service_name}")
                if cpu_usage and cpu_usage > self.config["alerts"]["cpu_threshold"]:
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"CPU使用率过高: {cpu_usage:.1f}%",
                        service_name,
                        "cpu_usage",
                        self.config["alerts"]["cpu_threshold"],
                        cpu_usage
                    )

                # 检查内存告警
                memory_usage = self._get_latest_metric(f"memory_usage_{service_name}")
                if memory_usage and memory_usage > self.config["alerts"]["memory_threshold"]:
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"内存使用率过高: {memory_usage:.1f}%",
                        service_name,
                        "memory_usage",
                        self.config["alerts"]["memory_threshold"],
                        memory_usage
                    )

                # 检查响应时间告警
                response_time = self._get_latest_metric(f"response_time_{service_name}")
                if response_time and response_time > self.config["alerts"]["response_time_threshold"]:
                    self._create_alert(
                        AlertLevel.ERROR,
                        f"响应时间过长: {response_time:.1f}ms",
                        service_name,
                        "response_time",
                        self.config["alerts"]["response_time_threshold"],
                        response_time
                    )

                # 检查错误率告警
                error_rate = self._get_latest_metric(f"error_rate_{service_name}")
                if error_rate and error_rate > self.config["alerts"]["error_rate_threshold"]:
                    self._create_alert(
                        AlertLevel.CRITICAL,
                        f"错误率过高: {error_rate:.1f}%",
                        service_name,
                        "error_rate",
                        self.config["alerts"]["error_rate_threshold"],
                        error_rate
                    )

    def _create_alert(self, level: AlertLevel, message: str, service_name: str,
                      metric_name: str, threshold: float, current_value: float):
        """创建告警"""
        alert = Alert(
            id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            level=level,
            message=message,
            timestamp=datetime.now(),
            service_name=service_name,
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value
        )

        self.alerts.append(alert)
        self.logger.warning(f"告警: {message}")

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        with self._lock:
            return {
                "metrics_count": sum(len(metrics) for metrics in self.metrics_history.values()),
                "alerts_count": len(self.alerts),
                "predictions_count": len(self.predictions),
                "services_monitored": len(self._get_service_names()),
                "monitoring_active": self.monitoring_active,
                "auto_tuning_active": self.auto_tuning_active,
                "predictive_maintenance_active": self.predictive_maintenance_active
            }

    def generate_monitoring_report(self) -> str:
        """生成监控报告"""
        stats = self.get_monitoring_stats()

        report = f"""
# 智能化监控系统报告

## 监控概览
- 监控指标数量: {stats['metrics_count']}
- 告警数量: {stats['alerts_count']}
- 预测数量: {stats['predictions_count']}
- 监控服务数: {stats['services_monitored']}

## 系统状态
- 监控系统: {'运行中' if stats['monitoring_active'] else '已停止'}
- 自动调优: {'运行中' if stats['auto_tuning_active'] else '已停止'}
- 预测性维护: {'运行中' if stats['predictive_maintenance_active'] else '已停止'}

## 最新告警
{chr(10).join(f"- [{alert.level.value.upper()}] {alert.message}" for alert in self.alerts[-5:])}

## 最新预测
{chr(10).join(f"- {pred.service_name} {pred.metric_name}: {pred.predicted_value:.1f} (置信度: {pred.confidence:.2f})" for pred in self.predictions[-5:])}
"""
        return report

    def stop_monitoring(self):
        """停止监控系统"""
        self.logger.info("停止智能化监控系统...")
        self.monitoring_active = False
        self.auto_tuning_active = False
        self.predictive_maintenance_active = False
        self.micro_service.stop()


async def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建监控系统
    monitoring = IntelligentMonitoring()

    # 启动监控
    success = await monitoring.start_monitoring()

    if success:
        print("智能化监控系统启动成功！")

        # 运行一段时间
        await asyncio.sleep(60)

        # 生成报告
        report = monitoring.generate_monitoring_report()
        print(report)

        # 停止监控
        monitoring.stop_monitoring()
        print("监控系统已停止")
    else:
        print("智能化监控系统启动失败！")


if __name__ == "__main__":
    asyncio.run(main())
