#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prometheus监控集成
提供与Prometheus监控系统的集成功能
"""

import time
from typing import Dict, Optional, Any
from datetime import datetime
import logging
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)

class PrometheusMonitor:
    """Prometheus监控集成类"""

    def __init__(self, gateway_url: str = "http://localhost:9091"):
        """
        初始化Prometheus监控器

        Args:
            gateway_url: Prometheus PushGateway地址
        """
        self.gateway_url = gateway_url
        self.registry = CollectorRegistry()
        self.client = self._create_prometheus_client()

        # 注册常用指标
        self._register_metrics()

    def _create_prometheus_client(self) -> Any:
        """创建Prometheus客户端连接"""
        # 这里可以扩展为直接连接Prometheus服务器的客户端
        # 目前使用PushGateway模式
        return {
            'registry': self.registry,
            'gateway_url': self.gateway_url
        }

    def _register_metrics(self) -> None:
        """注册常用指标"""
        self.alert_count = Gauge(
            'alerts_total',
            'Total number of alerts',
            ['severity', 'source'],
            registry=self.registry
        )

        self.experiment_duration = Gauge(
            'chaos_experiment_duration_seconds',
            'Duration of chaos experiments',
            ['experiment'],
            registry=self.registry
        )

    def alert(self, source: str, message: str, severity: str = "warning") -> None:
        """
        发送告警到Prometheus

        Args:
            source: 告警来源
            message: 告警消息
            severity: 告警级别(warning/critical)
        """
        logger.warning(f"Prometheus alert - {severity}: {source} - {message}")

        # 记录告警指标
        self.alert_count.labels(severity=severity, source=source).inc()

        # 推送指标到Gateway
        self.push_metrics()

    def send_metric(self, name: str, value: float, labels: Optional[Dict] = None) -> None:
        """
        发送自定义指标到Prometheus

        Args:
            name: 指标名称
            value: 指标值
            labels: 指标标签
        """
        labels = labels or {}

        # 动态创建或获取指标
        if name not in self.registry._names_to_collectors:
            metric = Gauge(
                name,
                f'Custom metric {name}',
                list(labels.keys()),
                registry=self.registry
            )
        else:
            metric = self.registry._names_to_collectors[name]

        # 设置指标值
        metric.labels(**labels).set(value)

    def push_metrics(self) -> None:
        """推送所有指标到Prometheus PushGateway"""
        try:
            push_to_gateway(
                self.gateway_url,
                job='rqa_monitoring',
                registry=self.registry
            )
            logger.debug("Metrics pushed to Prometheus successfully")
        except Exception as e:
            logger.error(f"Failed to push metrics to Prometheus: {e}")

    def cleanup(self) -> None:
        """清理PushGateway中的指标"""
        try:
            from prometheus_client import delete_from_gateway
            delete_from_gateway(self.gateway_url, job='rqa_monitoring')
        except ImportError:
            logger.warning("Prometheus client does not support delete_from_gateway")
        except Exception as e:
            logger.error(f"Failed to cleanup Prometheus metrics: {e}")

