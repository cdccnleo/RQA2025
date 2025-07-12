import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum, auto
import threading
import queue
import json
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()

@dataclass
class MetricLabel:
    name: str
    value: str

class Metric:
    """监控指标基类"""
    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._lock = threading.Lock()

class Counter(Metric):
    """计数器指标"""
    def __init__(self, name: str, description: str, labels: List[str] = None):
        super().__init__(name, description, labels)
        self._values = {}  # {(label_values): value}

    def inc(self, labels: Dict[str, str] = None, value: float = 1):
        """增加计数器值"""
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] = self._values.get(label_key, 0) + value

    def get(self, labels: Dict[str, str] = None) -> float:
        """获取计数器值"""
        label_key = self._get_label_key(labels)
        return self._values.get(label_key, 0)

    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """生成标签键"""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

class Gauge(Metric):
    """仪表盘指标"""
    def __init__(self, name: str, description: str, labels: List[str] = None):
        super().__init__(name, description, labels)
        self._values = {}  # {(label_values): value}

    def set(self, value: float, labels: Dict[str, str] = None):
        """设置仪表盘值"""
        label_key = self._get_label_key(labels)
        with self._lock:
            self._values[label_key] = value

    def get(self, labels: Dict[str, str] = None) -> float:
        """获取仪表盘值"""
        label_key = self._get_label_key(labels)
        return self._values.get(label_key, 0)

    def _get_label_key(self, labels: Dict[str, str]) -> str:
        """生成标签键"""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

class AlertChannel(ABC):
    """告警通道抽象基类"""
    @abstractmethod
    def send(self, alert: Dict[str, Any]) -> bool:
        """发送告警"""
        pass

class DingTalkChannel(AlertChannel):
    """钉钉告警通道"""
    def __init__(self, webhook: str):
        self.webhook = webhook

    def send(self, alert: Dict[str, Any]) -> bool:
        try:
            resp = requests.post(
                self.webhook,
                json={
                    "msgtype": "markdown",
                    "markdown": {
                        "title": f"{alert['level']}告警: {alert['name']}",
                        "text": f"### {alert['name']}告警\n" +
                                f"**级别**: {alert['level']}\n" +
                                f"**时间**: {alert['timestamp']}\n" +
                                f"**详情**: {json.dumps(alert['data'], indent=2)}"
                    }
                },
                timeout=5
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"钉钉告警发送失败: {e}")
            return False

class WeComChannel(AlertChannel):
    """企业微信告警通道"""
    def __init__(self, webhook: str):
        self.webhook = webhook

    def send(self, alert: Dict[str, Any]) -> bool:
        try:
            resp = requests.post(
                self.webhook,
                json={
                    "msgtype": "markdown",
                    "markdown": {
                        "content": f"**{alert['name']}告警**\n" +
                                   f">级别: {alert['level']}\n" +
                                   f">时间: {alert['timestamp']}\n" +
                                   f">详情: {json.dumps(alert['data'], indent=2)}"
                    }
                },
                timeout=5
            )
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"企业微信告警发送失败: {e}")
            return False

class MonitoringSystem:
    """一体化监控告警系统"""

    def __init__(self, prometheus_port: int = 9090):
        self.metrics: Dict[str, Metric] = {}
        self.alert_channels: Dict[str, AlertChannel] = {}
        self.alert_rules: Dict[str, Dict] = {}
        self.alert_queue = queue.Queue(maxsize=1000)
        self._alert_thread = threading.Thread(target=self._alert_worker)
        self._alert_thread.daemon = True
        self._alert_thread.start()

        # Prometheus exporter
        self.prometheus_port = prometheus_port
        self._start_prometheus_exporter()

    def register_metric(self, metric: Metric) -> bool:
        """注册监控指标"""
        if metric.name in self.metrics:
            logger.warning(f"指标已存在: {metric.name}")
            return False
        self.metrics[metric.name] = metric
        return True

    def register_counter(self, name: str, description: str, labels: List[str] = None) -> Counter:
        """注册计数器指标"""
        counter = Counter(name, description, labels)
        self.register_metric(counter)
        return counter

    def register_gauge(self, name: str, description: str, labels: List[str] = None) -> Gauge:
        """注册仪表盘指标"""
        gauge = Gauge(name, description, labels)
        self.register_metric(gauge)
        return gauge

    def add_alert_channel(self, name: str, channel: AlertChannel) -> None:
        """添加告警通道"""
        self.alert_channels[name] = channel

    def add_alert_rule(self, name: str, condition: str, channels: List[str], level: AlertLevel) -> None:
        """添加告警规则"""
        self.alert_rules[name] = {
            "condition": condition,
            "channels": channels,
            "level": level
        }

    def record_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """记录错误事件"""
        error_type = type(error).__name__

        # 记录错误指标
        if "trading_errors_total" in self.metrics:
            counter = self.metrics["trading_errors_total"]
            if isinstance(counter, Counter):
                counter.inc(labels={"error_type": error_type})

        # 检查是否需要告警
        if self._should_alert(error):
            self.trigger_alert(
                name="TRADING_ERROR",
                level=AlertLevel.CRITICAL if self._is_critical(error) else AlertLevel.WARNING,
                data={
                    "error_type": error_type,
                    "message": str(error),
                    "context": context
                }
            )

    def trigger_alert(self, name: str, level: AlertLevel, data: Dict[str, Any]) -> None:
        """触发告警"""
        alert = {
            "name": name,
            "level": level.name,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        try:
            self.alert_queue.put_nowait(alert)
        except queue.Full:
            logger.error("告警队列已满，丢弃告警")

    def _alert_worker(self) -> None:
        """告警工作线程"""
        while True:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._dispatch_alert(alert)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"告警处理异常: {e}")

    def _dispatch_alert(self, alert: Dict[str, Any]) -> None:
        """分发告警到各个通道"""
        rule_name = alert.get("name")
        if rule_name not in self.alert_rules:
            return

        rule = self.alert_rules[rule_name]
        for channel_name in rule["channels"]:
            if channel_name in self.alert_channels:
                self.alert_channels[channel_name].send(alert)

    def _should_alert(self, error: Exception) -> bool:
        """判断是否需要告警"""
        # 这里可以添加更复杂的判断逻辑
        return True

    def _is_critical(self, error: Exception) -> bool:
        """判断是否关键错误"""
        # 这里可以添加更复杂的判断逻辑
        return False

    def _start_prometheus_exporter(self) -> None:
        """启动Prometheus exporter"""
        from prometheus_client import start_http_server, Counter as PromCounter, Gauge as PromGauge

        # 启动Prometheus HTTP服务器
        start_http_server(self.prometheus_port)

        # 创建Prometheus指标映射
        self.prom_metrics = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, Counter):
                self.prom_metrics[name] = PromCounter(
                    name, metric.description, metric.labels
                )
            elif isinstance(metric, Gauge):
                self.prom_metrics[name] = PromGauge(
                    name, metric.description, metric.labels
                )

    def get_prometheus_metrics(self) -> str:
        """获取Prometheus格式的指标数据"""
        from prometheus_client import generate_latest
        return generate_latest()

# 全局监控系统实例
monitor = MonitoringSystem()
monitor.register_counter(
    "trading_errors_total",
    "Total trading errors",
    ["error_type"]
)

# 示例告警通道配置
monitor.add_alert_channel(
    "dingtalk",
    DingTalkChannel("https://oapi.dingtalk.com/robot/send?access_token=xxx")
)
monitor.add_alert_rule(
    "TRADING_ERROR",
    condition="error_type in ['OrderRejectedError', 'ConnectionError']",
    channels=["dingtalk"],
    level=AlertLevel.CRITICAL
)
