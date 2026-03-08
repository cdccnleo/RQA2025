#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
"""
监控服务实现
Monitoring Service Implementation

提供策略运行监控、性能跟踪、告警管理和健康检查功能。
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

from strategy.interfaces.monitoring_interfaces import (
    IMonitoringService, MetricData, Alert,
    AlertRule, MetricType
)
from src.infrastructure.integration.business_adapters import get_unified_adapter_factory

# 增强的持久化支持
try:
    from ...features.monitoring.metrics_persistence import get_enhanced_persistence_manager
    ENHANCED_PERSISTENCE_AVAILABLE = True
except ImportError:
    ENHANCED_PERSISTENCE_AVAILABLE = False
    logger.warning("增强的持久化功能不可用，使用标准模式")


@dataclass
class MonitoringConfig:

    """监控配置"""
    monitoring_id: str
    strategy_id: str
    metrics_interval: int = 60  # 指标收集间隔（秒）
    alert_check_interval: int = 30  # 告警检查间隔（秒）
    max_metrics_history: int = 1000  # 最大指标历史数量
    enabled: bool = True


class MonitoringService(IMonitoringService):

    """
    监控服务
    Monitoring Service

    提供策略运行监控、性能跟踪和健康检查功能。
    """

    def __init__(self):

        # 初始化增强的持久化管理器
        if ENHANCED_PERSISTENCE_AVAILABLE:
            persistence_config = {
                'path': './monitoring_data_enhanced',
                'primary_backend': 'sqlite',
                'compression': 'lz4',
                'batch_size': 200,
                'batch_timeout': 1.0,
                'archive': {
                    'hot_data_days': 7,
                    'warm_data_days': 30,
                    'cold_data_days': 365
                }
            }
            self.enhanced_persistence = get_enhanced_persistence_manager(persistence_config)
            logger.info("增强的持久化管理器已初始化")
        else:
            self.enhanced_persistence = None

        """初始化监控服务"""
        self.adapter_factory = get_unified_adapter_factory()

        # 监控配置
        self.monitoring_configs: Dict[str, MonitoringConfig] = {}

        # 指标数据存储
        self.metrics_data: Dict[str, List[MetricData]] = {}

        # 监控任务
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}

        # 告警规则
        self.alert_rules: Dict[str, AlertRule] = {}

        # 活跃告警
        self.active_alerts: List[Alert] = []

        logger.info("监控服务初始化完成")

    async def start_monitoring(self, strategy_id: str) -> bool:
        """
        开始监控策略

        Args:
            strategy_id: 策略ID

        Returns:
            bool: 启动是否成功
        """
        try:
            if strategy_id in self.monitoring_configs:
                logger.warning(f"策略 {strategy_id} 已在监控中")
                return False

            # 创建监控配置
            config = MonitoringConfig(
                monitoring_id=f"monitor_{strategy_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                strategy_id=strategy_id
            )

            self.monitoring_configs[strategy_id] = config
            self.metrics_data[strategy_id] = []

            # 启动监控任务
            task = asyncio.create_task(self._monitoring_loop(strategy_id))
            self.monitoring_tasks[strategy_id] = task

            # 发布事件
            await self._publish_event("monitoring_started", {
                "strategy_id": strategy_id,
                "monitoring_id": config.monitoring_id,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"策略 {strategy_id} 监控已启动")
            return True

        except Exception as e:
            logger.error(f"启动策略监控失败: {e}")
            return False

    async def stop_monitoring(self, strategy_id: str) -> bool:
        """
        停止监控策略

        Args:
            strategy_id: 策略ID

        Returns:
            bool: 停止是否成功
        """
        try:
            if strategy_id not in self.monitoring_configs:
                logger.warning(f"策略 {strategy_id} 不在监控中")
                return False

            # 取消监控任务
            if strategy_id in self.monitoring_tasks:
                self.monitoring_tasks[strategy_id].cancel()
                del self.monitoring_tasks[strategy_id]

            # 清理数据
            if strategy_id in self.metrics_data:
                del self.metrics_data[strategy_id]

            del self.monitoring_configs[strategy_id]

            # 发布事件
            await self._publish_event("monitoring_stopped", {
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"策略 {strategy_id} 监控已停止")
            return True

        except Exception as e:
            logger.error(f"停止策略监控失败: {e}")
            return False

    async def _monitoring_loop(self, strategy_id: str):
        """
        监控循环

        Args:
            strategy_id: 策略ID
        """
        config = self.monitoring_configs[strategy_id]

        while strategy_id in self.monitoring_configs:
            try:
                # 收集指标
                await self._collect_metrics(strategy_id)

                # 检查告警
                await self._check_alerts(strategy_id)

                # 等待下一个收集周期
                await asyncio.sleep(config.metrics_interval)

            except asyncio.CancelledError:
                logger.info(f"策略 {strategy_id} 监控循环被取消")
                break
            except Exception as e:
                logger.error(f"策略 {strategy_id} 监控循环异常: {e}")
                await asyncio.sleep(5)  # 出错后等待5秒再继续

    async def _collect_metrics(self, strategy_id: str):
        """
        收集指标

        Args:
            strategy_id: 策略ID
        """
        try:
            # 这里简化实现，实际应该从策略服务获取实时指标
            # 暂时生成模拟指标数据

            import secrets
            np.random.seed(hash(f"{strategy_id}_{datetime.now()}") % 2 ** 32)

            metrics = {
                'cpu_usage': secrets.uniform(10, 80),
                'memory_usage': secrets.uniform(20, 90),
                'response_time': secrets.uniform(1, 50),
                'throughput': secrets.uniform(50, 200),
                'error_rate': secrets.uniform(0, 0.05),
                'strategy_return': secrets.uniform(-0.02, 0.03),
                'strategy_sharpe': secrets.uniform(-1, 2),
                'strategy_drawdown': secrets.uniform(0, 0.1)
            }

            # 记录指标
            for metric_name, value in metrics.items():
                metric_data = MetricData(
                    metric_name=metric_name,
                    value=value,
                    timestamp=datetime.now(),
                    strategy_id=strategy_id,
                    metric_type=MetricType.PERFORMANCE
                )

                await self.record_metric(metric_data)

        except Exception as e:
            logger.error(f"收集策略 {strategy_id} 指标失败: {e}")

    async def _check_alerts(self, strategy_id: str):
        """
        检查告警

        Args:
            strategy_id: 策略ID
        """
        try:
            # 获取相关的告警规则
            strategy_rules = [
                rule for rule in self.alert_rules.values()
                if rule.strategy_id == strategy_id
            ]

            for rule in strategy_rules:
                await self._evaluate_alert_rule(rule)

        except Exception as e:
            logger.error(f"检查策略 {strategy_id} 告警失败: {e}")

    async def _evaluate_alert_rule(self, rule: AlertRule):
        """
        评估告警规则

        Args:
            rule: 告警规则
        """
        try:
            # 获取最新指标值
            latest_value = await self._get_latest_metric_value(
                rule.strategy_id, rule.metric_name
            )

            if latest_value is None:
                return

            # 评估条件
            triggered = self._evaluate_condition(latest_value, rule.threshold, rule.condition)

            if triggered:
                # 创建告警
                alert = Alert(
                    alert_id=f"alert_{rule.rule_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                    strategy_id=rule.strategy_id,
                    rule_id=rule.rule_id,
                    level=rule.level,
                    message=f"指标 {rule.metric_name} {rule.condition} {rule.threshold}，当前值: {latest_value}",
                    metric_value=latest_value,
                    threshold=rule.threshold,
                    timestamp=datetime.now()
                )

                # 检查是否已有相同活跃告警
                existing_alert = next(
                    (a for a in self.active_alerts
                     if a.rule_id == rule.rule_id and not a.resolved),
                    None
                )

                if not existing_alert:
                    self.active_alerts.append(alert)

                    # 发布告警事件
                    await self._publish_event("alert_triggered", {
                        "alert_id": alert.alert_id,
                        "strategy_id": rule.strategy_id,
                        "rule_id": rule.rule_id,
                        "level": rule.level.value,
                        "message": alert.message,
                        "timestamp": datetime.now().isoformat()
                    })

                    logger.warning(f"告警触发: {alert.message}")

        except Exception as e:
            logger.error(f"评估告警规则 {rule.rule_id} 失败: {e}")

    def _evaluate_condition(self, value: float, threshold: float, condition: str) -> bool:
        """
        评估条件

        Args:
            value: 实际值
            threshold: 阈值
            condition: 条件操作符

        Returns:
            bool: 条件是否满足
        """
        if condition == '>':
            return value > threshold
        elif condition == '<':
            return value < threshold
        elif condition == '>=':
            return value >= threshold
        elif condition == '<=':
            return value <= threshold
        elif condition == '==':
            return value == threshold
        elif condition == '!=':
            return value != threshold
        else:
            return False

    async def _get_latest_metric_value(self, strategy_id: str, metric_name: str) -> Optional[float]:
        """
        获取最新指标值

        Args:
            strategy_id: 策略ID
            metric_name: 指标名称

        Returns:
            Optional[float]: 最新指标值
        """
        try:
            if strategy_id not in self.metrics_data:
                return None

            # 查找最新的指标值
            metrics = self.metrics_data[strategy_id]
            relevant_metrics = [
                m for m in metrics
                if m.metric_name == metric_name
            ]

            if not relevant_metrics:
                return None

            # 返回最新的指标值
            latest_metric = max(relevant_metrics, key=lambda m: m.timestamp)
            return latest_metric.value

        except Exception as e:
            logger.error(f"获取最新指标值失败: {e}")
            return None

    def get_current_metrics(self, strategy_id: str,


                            metric_types: Optional[List[MetricType]] = None) -> Dict[str, MetricData]:
        """
        获取当前指标

        Args:
            strategy_id: 策略ID
            metric_types: 指标类型过滤器

        Returns:
            Dict[str, MetricData]: 指标数据字典
        """
        try:
            if strategy_id not in self.metrics_data:
                return {}

            metrics = self.metrics_data[strategy_id]
            if not metrics:
                return {}

            # 获取每个指标的最新值
            latest_metrics = {}
            metric_names = set(m.metric_name for m in metrics)

            for metric_name in metric_names:
                relevant_metrics = [m for m in metrics if m.metric_name == metric_name]
                if relevant_metrics:
                    latest_metric = max(relevant_metrics, key=lambda m: m.timestamp)

                    # 应用类型过滤
                    if metric_types is None or latest_metric.metric_type in metric_types:
                        latest_metrics[metric_name] = latest_metric

            return latest_metrics

        except Exception as e:
            logger.error(f"获取当前指标失败: {e}")
            return {}

    def get_metric_history(self, strategy_id: str, metric_name: str,


                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[MetricData]:
        """
        获取指标历史

        Args:
            strategy_id: 策略ID
            metric_name: 指标名称
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[MetricData]: 指标历史数据列表
        """
        try:
            if strategy_id not in self.metrics_data:
                return []

            metrics = self.metrics_data[strategy_id]

            # 过滤指标名称
            filtered_metrics = [
                m for m in metrics
                if m.metric_name == metric_name
            ]

            # 过滤时间范围
            if start_time:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if m.timestamp >= start_time
                ]

            if end_time:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if m.timestamp <= end_time
                ]

            # 按时间排序
            filtered_metrics.sort(key=lambda m: m.timestamp)

            return filtered_metrics

        except Exception as e:
            logger.error(f"获取指标历史失败: {e}")
            return []

    async def record_metric(self, metric_data: MetricData) -> bool:
        """
        记录指标

        Args:
            metric_data: 指标数据

        Returns:
            bool: 记录是否成功
        """
        try:
            strategy_id = metric_data.strategy_id

            if strategy_id not in self.metrics_data:
                self.metrics_data[strategy_id] = []

            # 添加指标数据
            self.metrics_data[strategy_id].append(metric_data)

            # 限制历史数据数量
            config = self.monitoring_configs.get(strategy_id)
            if config and len(self.metrics_data[strategy_id]) > config.max_metrics_history:
                # 保留最新的数据
                self.metrics_data[strategy_id] = self.metrics_data[strategy_id][-config.max_metrics_history:]

            logger.debug(f"指标已记录: {strategy_id} - {metric_data.metric_name} = {metric_data.value}")
            return True

        except Exception as e:
            logger.error(f"记录指标失败: {e}")
            return False

    async def _publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        发布事件

        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        try:
            event_bus_adapter = self.adapter_factory.get_adapter("event_bus")
            await event_bus_adapter.publish_event({
                "event_type": f"monitoring_{event_type}",
                "data": event_data,
                "source": "monitoring_service",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"事件发布异常: {e}")


# 导出类
__all__ = [
    'MonitoringService',
    'MonitoringConfig'
]

