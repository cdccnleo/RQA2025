"""
prometheus_monitor 模块

提供 prometheus_monitor 相关功能和接口。
"""

import logging
import time

# -*- coding: utf-8 -*-

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, delete_from_gateway
from typing import Dict, Optional, Any, List, Union
"""
基础设施层 - 日志系统组件

prometheus_monitor 模块

日志系统相关的文件
提供日志系统相关的功能实现。
"""

#!/usr / bin / env python
"""
prometheus_monitor - 日志系统

职责说明：
负责系统日志记录、日志格式化、日志存储和日志分析

核心职责：
- 日志记录和格式化
- 日志级别管理
- 日志存储和轮转
- 日志分析和监控
- 日志搜索和过滤
- 日志性能优化

相关接口：
- ILoggingComponent
- ILogger
- ILogHandler
""" """
Prometheus监控集成
提供与Prometheus监控系统的集成功能
"""

from .enums import AlertData

logger = logging.getLogger(__name__)


class MetricsRegistry:
    """
    指标注册器 - 专门负责Prometheus指标的注册和管理

    单一职责：创建和管理Prometheus指标对象
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self._registry = registry or CollectorRegistry()
        self._gauges: Dict[str, Any] = {}
        self._counters: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}
        self._logger = logger  # 兼容性属性

    def create_gauge(self, name: str, description: str, labels: Optional[List[str]] = None) -> Any:
        """
        创建Gauge指标

        Args:
            name: 指标名称
            description: 指标描述
            labels: 标签列表

        Returns:
            Gauge对象
        """
        if name in self._gauges:
            return self._gauges[name]

        gauge = Gauge(name, description, labels or [], registry=self._registry)
        self._gauges[name] = gauge
        return gauge

    def create_counter(self, name: str, description: str, labels: Optional[List[str]] = None) -> Any:
        """
        创建Counter指标

        Args:
            name: 指标名称
            description: 指标描述
            labels: 标签列表

        Returns:
            Counter对象
        """
        from prometheus_client import Counter

        if name in self._counters:
            return self._counters[name]

        counter = Counter(name, description, labels or [], registry=self._registry)
        self._counters[name] = counter
        return counter

    def create_histogram(self, name: str, description: str, labels: Optional[List[str]] = None, buckets: Optional[List[float]] = None) -> Any:
        """
        创建Histogram指标

        Args:
            name: 指标名称
            description: 指标描述
            labels: 标签列表
            buckets: 桶列表

        Returns:
            Histogram对象
        """
        from prometheus_client import Histogram

        if name in self._histograms:
            return self._histograms[name]

        # 使用默认的 buckets 如果没有提供
        if buckets is None:
            buckets = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]

        histogram = Histogram(name, description, labels or [], buckets=buckets, registry=self._registry)
        self._histograms[name] = histogram
        return histogram

    def get_gauge(self, name: str) -> Optional[Any]:
        """获取Gauge指标"""
        return self._gauges.get(name)

    def get_counter(self, name: str) -> Optional[Any]:
        """获取Counter指标"""
        return self._counters.get(name)

    def get_histogram(self, name: str) -> Optional[Any]:
        """获取Histogram指标"""
        return self._histograms.get(name)

    def get_metric(self, name: str) -> Optional[Any]:
        """获取指标（通用方法）"""
        # 按优先级查找：gauge -> counter -> histogram
        metric = self.get_gauge(name)
        if metric:
            return metric

        metric = self.get_counter(name)
        if metric:
            return metric

        return self.get_histogram(name)

    def remove_gauge(self, name: str) -> bool:
        """移除Gauge指标"""
        if name in self._gauges:
            del self._gauges[name]
            return True
        return False

    def remove_counter(self, name: str) -> bool:
        """移除Counter指标"""
        if name in self._counters:
            del self._counters[name]
            return True
        return False

    def remove_histogram(self, name: str) -> bool:
        """移除Histogram指标"""
        if name in self._histograms:
            del self._histograms[name]
            return True
        return False

    def clear_all_metrics(self):
        """清除所有指标"""
        # 清除所有指标，包括默认指标
        self._gauges.clear()
        self._counters.clear()
        self._histograms.clear()

    def get_registry(self) -> CollectorRegistry:
        """获取注册表"""
        return self._registry


class MetricsExporter:
    """
    指标导出器 - 专门负责指标数据的导出和推送

    单一职责：处理指标数据的导出操作
    """

    def __init__(self, registry: MetricsRegistry, job_name: str = "logging_system"):
        self._registry = registry
        self._job_name = job_name
        self._gateway_url: Optional[str] = None
        self._push_gateway_url: Optional[str] = None  # 兼容性属性
        self._logger = logger  # 兼容性属性

    def set_gateway_url(self, url: str):
        """设置Push Gateway URL"""
        self._gateway_url = url
        self._push_gateway_url = url  # 保持兼容性

    def get_registry(self) -> MetricsRegistry:
        """获取注册表"""
        return self._registry

    def update_push_gateway_url(self, url: str):
        """更新推送网关URL"""
        self.set_gateway_url(url)

    def update_job_name(self, job_name: str):
        """更新作业名称"""
        self._job_name = job_name

    def get_metrics_summary(self) -> Dict[str, int]:
        """获取指标摘要"""
        return {
            "gauges": len(self._registry._gauges),
            "counters": len(self._registry._counters),
            "histograms": len(self._registry._histograms)
        }

    def push_metrics(self, grouping_key: Optional[Dict[str, str]] = None) -> bool:
        """
        推送指标到Push Gateway

        Args:
            grouping_key: 分组键

        Returns:
            是否推送成功
        """
        if not self._gateway_url:
            logger.warning("Push Gateway URL未设置")
            return False

        try:
            push_to_gateway(
                self._gateway_url,
                job=self._job_name,
                registry=self._registry.get_registry(),
                grouping_key=grouping_key or {}
            )
            logger.info(f"指标已推送到Push Gateway: {self._gateway_url}")
            return True
        except Exception as e:
            logger.error(f"推送指标失败: {e}")
            return False

    def export_metrics_to_gateway(self, grouping_key: Optional[Dict[str, str]] = None) -> bool:
        """导出指标到Push Gateway（兼容性方法）"""
        return self.push_metrics(grouping_key)

    def delete_metrics_from_gateway(self) -> bool:
        """从Push Gateway删除指标（兼容性方法）"""
        return self.delete_metrics()

    def delete_metrics(self, grouping_key: Optional[Dict[str, str]] = None) -> bool:
        """
        从Push Gateway删除指标

        Args:
            grouping_key: 分组键

        Returns:
            是否删除成功
        """
        if not self._gateway_url:
            logger.warning("Push Gateway URL未设置")
            return False

        try:
            delete_from_gateway(
                self._gateway_url,
                job=self._job_name,
                grouping_key=grouping_key or {}
            )
            logger.info(f"指标已从Push Gateway删除: {self._gateway_url}")
            return True
        except Exception as e:
            logger.error(f"删除指标失败: {e}")
            return False

    def send_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> bool:
        """
        发送单个指标值

        Args:
            name: 指标名称
            value: 指标值
            labels: 标签字典

        Returns:
            是否发送成功
        """
        try:
            metric = self._registry.get_metric(name)
            if not metric:
                logger.warning(f"指标未注册: {name}")
                return False

            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)

            logger.debug(f"指标已更新: {name} = {value}")
            return True
        except Exception as e:
            logger.error(f"发送指标失败 {name}: {e}")
            return False


class AlertHandler:
    """
    告警处理器 - 专门负责告警的处理和发送

    单一职责：处理告警逻辑和告警数据格式化
    """

    def __init__(self, metrics_exporter: MetricsExporter):
        self._exporter = metrics_exporter
        self._alert_metrics: Dict[str, Any] = {}
        self._alert_states: Dict[str, Dict[str, Any]] = {}
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._logger = logger  # 兼容性属性

    def register_alert_metric(self, name: str, description: str) -> bool:
        """
        注册告警指标

        Args:
            name: 告警指标名称
            description: 指标描述

        Returns:
            是否注册成功
        """
        try:
            # 为告警创建计数器指标
            alert_metric = self._exporter._registry.create_gauge(
                f"{name}_alerts",
                f"Number of {description}",
                ["severity", "source"]
            )
            self._alert_metrics[name] = alert_metric
            return True
        except Exception as e:
            logger.error(f"注册告警指标失败 {name}: {e}")
            return False

    def handle_alert(self, alert_name: str, message: str, severity: str = "warning",
                     source: str = "logging_system", extra_labels: Optional[Dict[str, str]] = None) -> bool:
        """
        处理告警

        Args:
            alert_name: 告警名称
            message: 告警消息
            severity: 告警级别
            source: 告警来源
            extra_labels: 额外标签

        Returns:
            是否处理成功
        """
        try:
            if alert_name not in self._alert_metrics:
                if not self.register_alert_metric(alert_name, f"Alerts for {alert_name}"):
                    return False

            metric = self._alert_metrics[alert_name]

            # 构建标签
            labels = {
                "severity": severity,
                "source": source
            }
            if extra_labels:
                labels.update(extra_labels)

            # 增加告警计数
            try:
                labeled_metric = metric.labels(**labels)
                current_value = labeled_metric._value or 0
                labeled_metric.set(current_value + 1)
            except Exception:
                # 如果标签不存在，创建新的标签并设置值为1
                metric.labels(**labels).set(1)

            logger.info(f"告警已处理: {alert_name} - {message} (severity: {severity})")
            return True
        except Exception as e:
            logger.error(f"处理告警失败 {alert_name}: {e}")
            return False

    def reset_alert(self, alert_name: str, severity: str = "warning",
                    source: str = "logging_system", extra_labels: Optional[Dict[str, str]] = None) -> bool:
        """
        重置告警计数

        Args:
            alert_name: 告警名称
            severity: 告警级别
            source: 告警来源
            extra_labels: 额外标签

        Returns:
            是否重置成功
        """
        try:
            if alert_name not in self._alert_metrics:
                return False

            metric = self._alert_metrics[alert_name]

            # 构建标签
            labels = {
                "severity": severity,
                "source": source
            }
            if extra_labels:
                labels.update(extra_labels)

            # 重置为0
            metric.labels(**labels).set(0)

            logger.info(f"告警已重置: {alert_name} (severity: {severity})")
            return True
        except Exception as e:
            logger.error(f"重置告警失败 {alert_name}: {e}")
            return False

    def add_alert_rule(self, name: str, config: Dict[str, Any]) -> bool:
        """添加告警规则"""
        try:
            self._alert_rules[name] = config
            logger.info(f"告警规则已添加: {name}")
            return True
        except Exception as e:
            logger.error(f"添加告警规则失败 {name}: {e}")
            return False

    def remove_alert_rule(self, name: str) -> bool:
        """移除告警规则"""
        try:
            if name in self._alert_rules:
                del self._alert_rules[name]
                logger.info(f"告警规则已移除: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"移除告警规则失败 {name}: {e}")
            return False

    def get_alert_rules(self) -> Dict[str, Any]:
        """获取告警规则"""
        return self._alert_rules.copy()

    def evaluate_alert_condition(self, metric_name: str, condition: str) -> bool:
        """评估告警条件"""
        return self._evaluate_condition(metric_name, condition)

    def _evaluate_condition(self, metric_name: str, condition: str) -> bool:
        """内部方法：评估告警条件"""
        # 简单的实现，实际应该解析条件表达式
        try:
            # 这里可以实现更复杂的条件评估逻辑
            return False
        except Exception as e:
            logger.error(f"评估告警条件失败 {metric_name} {condition}: {e}")
            return False

    def trigger_alert(self, alert_data: Union[str, Dict[str, Any]], severity: str = "warning", message: str = "Alert triggered") -> bool:
        """触发告警"""
        if isinstance(alert_data, dict):
            name = alert_data.get("name", "unknown_alert")
            severity = alert_data.get("severity", severity)
            message = alert_data.get("message", message)
        else:
            name = alert_data

        # 更新告警状态
        self._alert_states[name] = {
            "name": name,
            "severity": severity,
            "message": message,
            "timestamp": time.time(),
            "active": True
        }

        return self.handle_alert(name, message, severity)

    def resolve_alert(self, name: str, severity: str = "warning") -> bool:
        """解决告警"""
        if name in self._alert_states:
            self._alert_states[name]["active"] = False
            self._alert_states[name]["resolved_at"] = time.time()

        return self.reset_alert(name, severity)

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return [alert for alert in self._alert_states.values() if alert.get("active", False)]

    def clear_all_alerts(self) -> bool:
        """清除所有告警"""
        self._alert_states.clear()
        logger.info("所有告警已清除")
        return True


class PrometheusMonitor:
    """
    Prometheus监控集成类（门面类）

    协调各个Prometheus处理组件，提供统一的监控接口
    遵循门面模式和组合优于继承原则
    """

    def __init__(
        self,
        gateway_url_or_config: Union[str, Dict[str, Any], None] = None,
        registry: Optional[CollectorRegistry] = None,
        *,
        gateway_url: Optional[str] = None,
    ):
        """
        初始化Prometheus监控器

        Args:
            gateway_url_or_config: Prometheus PushGateway地址或配置字典
            registry: Prometheus CollectorRegistry（可选，测试隔离用）
        """
        # 解析配置
        config: Dict[str, Any] = {}
        if isinstance(gateway_url_or_config, dict):
            config.update(gateway_url_or_config)
        elif gateway_url_or_config is not None:
            config["push_gateway_url"] = gateway_url_or_config
        if gateway_url is not None:
            config["push_gateway_url"] = gateway_url

        resolved_gateway_url = config.get("push_gateway_url", "http://localhost:9091")
        job_name = config.get("job_name", "logging_system")
        alert_rules = config.get("alert_rules", {})

        # 组合各个组件
        self._registry = MetricsRegistry(registry)
        self._exporter = MetricsExporter(self._registry, job_name=job_name)
        self._alert_handler = AlertHandler(self._exporter)

        # 设置Push Gateway URL
        self._exporter.set_gateway_url(resolved_gateway_url)

        # 兼容性属性
        self.gateway_url = resolved_gateway_url
        self.registry = self._registry.get_registry()

        # 添加配置的告警规则
        for rule_name, rule_config in alert_rules.items():
            self._alert_handler.add_alert_rule(rule_name, rule_config)

        # 兼容性属性
        self._logger = logger

        # 注册常用指标
        self._register_default_metrics()

    # 门面方法 - 委托给各个组件

    def send_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> bool:
        """发送指标"""
        return self._exporter.send_metric(name, value, labels)

    def alert(self, message: str, severity: str = "warning", labels: Optional[Dict[str, str]] = None) -> bool:
        """发送告警"""
        alert_name = "logging_alert"
        extra_labels = labels or {}
        # 过滤标签，只保留有效的标签名称
        valid_labels = {k: v for k, v in extra_labels.items() if k.replace('_', '').isalnum()}
        return self._alert_handler.handle_alert(
            alert_name, message, severity, "logging_system", valid_labels
        )

    def push_metrics(self, grouping_key: Optional[Dict[str, str]] = None) -> bool:
        """推送指标"""
        return self._exporter.push_metrics(grouping_key)

    def cleanup(self) -> bool:
        """清理指标"""
        return self._exporter.delete_metrics()

    def create_gauge_metric(self, name: str, description: str, labels: Optional[List[str]] = None) -> Optional[Any]:
        """创建Gauge指标"""
        try:
            return self._registry.create_gauge(name, description, labels)
        except Exception as e:
            logger.error(f"创建Gauge指标失败 {name}: {e}")
            return None

    def create_counter_metric(self, name: str, description: str, labels: Optional[List[str]] = None) -> Optional[Any]:
        """创建Counter指标"""
        try:
            return self._registry.create_counter(name, description, labels)
        except Exception as e:
            logger.error(f"创建Counter指标失败 {name}: {e}")
            return None

    def create_histogram_metric(self, name: str, description: str, labels: Optional[List[str]] = None, buckets: Optional[List[float]] = None) -> Optional[Any]:
        """创建Histogram指标"""
        try:
            return self._registry.create_histogram(name, description, labels, buckets)
        except Exception as e:
            logger.error(f"创建Histogram指标失败 {name}: {e}")
            return None

    def record_metric_value(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> bool:
        """记录指标值"""
        try:
            return self._exporter.send_metric(name, value, labels)
        except Exception as e:
            logger.error(f"记录指标值失败 {name}: {e}")
            return False

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        amount: Optional[float] = None,
    ) -> bool:
        """递增计数器"""
        try:
            increment_value = amount if amount is not None else value
            counter = self._registry.get_counter(name)
            if counter:
                if labels:
                    counter.labels(**labels).inc(increment_value)
                else:
                    counter.inc(increment_value)
                return True
            return False
        except Exception as e:
            logger.error(f"递增计数器失败 {name}: {e}")
            return False

    def record_histogram_value(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> bool:
        """记录直方图值"""
        try:
            histogram = self._registry.get_histogram(name)
            if histogram:
                if labels:
                    histogram.labels(**labels).observe(value)
                else:
                    histogram.observe(value)
                return True
            return False
        except Exception as e:
            logger.error(f"记录直方图值失败 {name}: {e}")
            return False

    def add_alert_rule(self, name: str, config: Dict[str, Any]) -> bool:
        """添加告警规则"""
        try:
            return self._alert_handler.add_alert_rule(name, config)
        except Exception as e:
            logger.error(f"添加告警规则失败 {name}: {e}")
            return False

    def remove_alert_rule(self, name: str) -> bool:
        """移除告警规则"""
        try:
            return self._alert_handler.remove_alert_rule(name)
        except Exception as e:
            logger.error(f"移除告警规则失败 {name}: {e}")
            return False

    def get_alert_rules(self) -> Dict[str, Any]:
        """获取告警规则"""
        try:
            return self._alert_handler.get_alert_rules()
        except Exception as e:
            logger.error(f"获取告警规则失败: {e}")
            return {}

    def trigger_alert(self, alert_data: Dict[str, Any]) -> bool:
        """触发告警"""
        try:
            # 直接使用 AlertHandler 的 trigger_alert 方法
            return self._alert_handler.trigger_alert(alert_data)
        except Exception as e:
            logger.error(f"触发告警失败: {e}")
            return False

    def resolve_alert(self, alert_name: str) -> bool:
        """解决告警"""
        try:
            return self.reset_alert(alert_name)
        except Exception as e:
            logger.error(f"解决告警失败 {alert_name}: {e}")
            return False

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        try:
            return self._alert_handler.get_active_alerts()
        except Exception as e:
            logger.error(f"获取活跃告警失败: {e}")
            return []

    def export_metrics(self) -> bool:
        """导出指标"""
        try:
            return self._exporter.push_metrics()
        except Exception as e:
            logger.error(f"导出指标失败: {e}")
            return False

    def clear_metrics(self) -> bool:
        """清除指标"""
        try:
            # 清除所有指标
            self._registry.clear_all_metrics()
            return True
        except Exception as e:
            logger.error(f"清除指标失败: {e}")
            return False

    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        try:
            return {
                'gauges': len(self._registry._gauges),
                'counters': len(self._registry._counters),
                'histograms': len(self._registry._histograms),
                'total_metrics': len(self._registry._gauges) + len(self._registry._counters) + len(self._registry._histograms)
            }
        except Exception as e:
            logger.error(f"获取指标摘要失败: {e}")
            return {}

    def register_gauge(self, name: str, description: str, labels: Optional[List[str]] = None) -> Optional[Gauge]:
        """注册Gauge指标"""
        try:
            return self._registry.create_gauge(name, description, labels)
        except Exception as e:
            logger.error(f"注册指标失败 {name}: {e}")
            return None

    def get_metric(self, name: str) -> Optional[Any]:
        """获取指标"""
        return self._registry.get_metric(name)

    def reset_alert(self, alert_name: str = "logging_alert", severity: str = "warning",
                    source: str = "logging_system", extra_labels: Optional[Dict[str, str]] = None) -> bool:
        """重置告警"""
        return self._alert_handler.reset_alert(alert_name, severity, source, extra_labels)

    # 私有方法

    def _register_default_metrics(self):
        """注册默认指标"""
        try:
            # 注册日志相关的默认指标
            self.register_gauge(
                "log_messages_total",
                "Total number of log messages",
                ["level", "logger"]
            )

            self.register_gauge(
                "log_errors_total",
                "Total number of error log messages",
                ["error_type", "component"]
            )

            self.register_gauge(
                "log_processing_time",
                "Time spent processing logs",
                ["operation"]
            )

            logger.info("默认Prometheus指标已注册")
        except Exception as e:
            logger.error(f"注册默认指标失败: {e}")

    # 兼容性方法（如果需要保持向后兼容）

    def _create_prometheus_client(self) -> Dict[str, Any]:
        """创建Prometheus客户端（兼容性方法）"""
        return {
            'registry': self.registry,
            'gateway_url': self.gateway_url,
            'exporter': self._exporter,
            'alert_handler': self._alert_handler
        }

    def _register_metrics(self) -> None:
        """注册指标（兼容性方法）"""
        self._register_default_metrics()
