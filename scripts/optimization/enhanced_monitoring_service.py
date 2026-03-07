#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的监控服务实现

集成新的持久化管理器，提供高性能的监控数据存储和检索功能。
包括实时数据流处理、智能缓存和数据生命周期管理。
"""

from scripts.optimization.monitoring_persistence_enhancer import (
    EnhancedMetricsPersistenceManager, MetricRecord
)
from src.strategy.interfaces.monitoring_interfaces import (
    IMonitoringService, MetricData, Alert, AlertRule, MetricType
)
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# from src.core.integration.business_adapters import get_unified_adapter_factory

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMonitoringConfig:
    """增强的监控配置"""
    monitoring_id: str
    strategy_id: str
    metrics_interval: int = 60
    alert_check_interval: int = 30
    max_metrics_history: int = 1000
    enabled: bool = True

    # 持久化配置
    persistence_enabled: bool = True
    storage_backend: str = "sqlite"
    compression_enabled: bool = True
    archive_enabled: bool = True


class EnhancedMonitoringService(IMonitoringService):
    """
    增强的监控服务

    集成了高性能持久化管理器，提供完整的监控数据生命周期管理
    """

    def __init__(self, config: Optional[Dict] = None):
        """初始化增强的监控服务"""
        self.config = config or {}
        # self.adapter_factory = get_unified_adapter_factory()
        self.adapter_factory = None  # 临时占位符

        # 监控配置
        self.monitoring_configs: Dict[str, EnhancedMonitoringConfig] = {}

        # 内存指标数据（用于快速访问）
        self.metrics_data: Dict[str, List[MetricData]] = {}

        # 监控任务
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}

        # 告警规则和活跃告警
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: List[Alert] = []

        # 初始化持久化管理器
        persistence_config = {
            'path': self.config.get('storage_path', './monitoring_data_enhanced'),
            'primary_backend': self.config.get('storage_backend', 'sqlite'),
            'compression': self.config.get('compression', 'lz4'),
            'batch_size': self.config.get('batch_size', 500),
            'batch_timeout': self.config.get('batch_timeout', 2.0),
            'max_workers': self.config.get('max_workers', 4),
            'archive': {
                'hot_data_days': self.config.get('hot_data_days', 7),
                'warm_data_days': self.config.get('warm_data_days', 30),
                'cold_data_days': self.config.get('cold_data_days', 365)
            }
        }

        self.persistence_manager = EnhancedMetricsPersistenceManager(persistence_config)

        # 注册数据流处理器
        self._setup_stream_processors()

        logger.info("增强的监控服务初始化完成")

    def _setup_stream_processors(self):
        """设置数据流处理器"""
        # 实时告警处理器
        async def alert_processor(record: MetricRecord):
            await self._process_real_time_alerts(record)

        # 性能分析处理器
        def performance_processor(record: MetricRecord):
            self._update_performance_metrics(record)

        # 异常检测处理器
        async def anomaly_processor(record: MetricRecord):
            await self._detect_anomalies(record)

        self.persistence_manager.stream_processors.extend([
            alert_processor,
            performance_processor,
            anomaly_processor
        ])

    def start_monitoring(self, strategy_id: str) -> bool:
        """开始监控策略"""
        try:
            if strategy_id in self.monitoring_configs:
                logger.warning(f"策略 {strategy_id} 已在监控中")
                return False

            # 创建增强的监控配置
            config = EnhancedMonitoringConfig(
                monitoring_id=f"enhanced_monitor_{strategy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy_id=strategy_id,
                persistence_enabled=self.config.get('persistence_enabled', True),
                storage_backend=self.config.get('storage_backend', 'sqlite'),
                compression_enabled=self.config.get('compression_enabled', True),
                archive_enabled=self.config.get('archive_enabled', True)
            )

            self.monitoring_configs[strategy_id] = config
            self.metrics_data[strategy_id] = []

            # 发布事件 (同步版本)
            logger.info(f"增强监控已启动 - 策略: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"启动增强监控失败: {e}")
            return False

    def stop_monitoring(self, strategy_id: str) -> bool:
        """停止监控策略"""
        try:
            if strategy_id not in self.monitoring_configs:
                logger.warning(f"策略 {strategy_id} 不在监控中")
                return False

            # 取消监控任务
            if strategy_id in self.monitoring_tasks:
                self.monitoring_tasks[strategy_id].cancel()
                del self.monitoring_tasks[strategy_id]

            # 清理内存数据
            if strategy_id in self.metrics_data:
                del self.metrics_data[strategy_id]

            del self.monitoring_configs[strategy_id]

            logger.info(f"增强监控已停止 - 策略: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"停止增强监控失败: {e}")
            return False

    async def _enhanced_monitoring_loop(self, strategy_id: str):
        """增强的监控循环"""
        config = self.monitoring_configs[strategy_id]

        while strategy_id in self.monitoring_configs:
            try:
                # 收集指标
                await self._collect_enhanced_metrics(strategy_id)

                # 检查告警
                # await self._check_alerts(strategy_id)  # 临时注释

                # 更新性能报告
                await self._update_performance_reports(strategy_id)

                # 等待下一个收集周期
                await asyncio.sleep(config.metrics_interval)

            except asyncio.CancelledError:
                logger.info(f"策略 {strategy_id} 增强监控循环被取消")
                break
            except Exception as e:
                logger.error(f"策略 {strategy_id} 增强监控循环异常: {e}")
                await asyncio.sleep(5)

    async def _collect_enhanced_metrics(self, strategy_id: str):
        """收集增强的指标"""
        try:
            # 生成模拟指标数据（实际应用中从真实源获取）
            import random
            random.seed(hash(f"{strategy_id}_{datetime.now()}") % 2**32)

            metrics = {
                'cpu_usage': random.uniform(10, 80),
                'memory_usage': random.uniform(20, 90),
                'disk_io': random.uniform(0, 100),
                'network_io': random.uniform(0, 50),
                'response_time': random.uniform(1, 50),
                'throughput': random.uniform(50, 200),
                'error_rate': random.uniform(0, 0.05),
                'strategy_return': random.uniform(-0.02, 0.03),
                'strategy_sharpe': random.uniform(-1, 2),
                'strategy_drawdown': random.uniform(0, 0.1),
                'trade_count': random.randint(0, 10),
                'profit_factor': random.uniform(0.8, 2.5)
            }

            # 记录指标到内存和持久化存储
            for metric_name, value in metrics.items():
                # 创建指标数据
                metric_data = MetricData(
                    metric_name=metric_name,
                    value=value,
                    timestamp=datetime.now(),
                    strategy_id=strategy_id,
                    metric_type=self._classify_metric_type(metric_name)
                )

                # 记录到内存
                await self.record_metric(metric_data)

                # 异步持久化存储
                await self.persistence_manager.store_metric_async(
                    component_name="strategy_monitoring",
                    metric_name=metric_name,
                    metric_value=value,
                    metric_type=metric_data.metric_type.value,
                    labels={
                        "strategy_id": strategy_id,
                        "component": "monitoring_service"
                    },
                    priority=self._get_metric_priority(metric_name)
                )

        except Exception as e:
            logger.error(f"收集增强指标失败 - 策略: {strategy_id}, 错误: {e}")

    def _classify_metric_type(self, metric_name: str) -> MetricType:
        """分类指标类型"""
        if metric_name in ['cpu_usage', 'memory_usage', 'disk_io', 'network_io']:
            return MetricType.SYSTEM
        elif metric_name in ['response_time', 'throughput', 'error_rate']:
            return MetricType.PERFORMANCE
        elif metric_name.startswith('strategy_'):
            return MetricType.CUSTOM  # 使用CUSTOM替代BUSINESS
        else:
            return MetricType.CUSTOM

    def _get_metric_priority(self, metric_name: str) -> int:
        """获取指标优先级"""
        high_priority_metrics = ['error_rate', 'strategy_drawdown', 'cpu_usage']
        medium_priority_metrics = ['strategy_return', 'throughput', 'memory_usage']

        if metric_name in high_priority_metrics:
            return 3
        elif metric_name in medium_priority_metrics:
            return 2
        else:
            return 1

    async def _process_real_time_alerts(self, record: MetricRecord):
        """处理实时告警"""
        try:
            # 简化的告警处理逻辑
            if record.metric_name == 'error_rate' and record.metric_value > 0.05:
                logger.warning(f"高错误率告警: {record.metric_value}")
            elif record.metric_name == 'cpu_usage' and record.metric_value > 80:
                logger.warning(f"高CPU使用率告警: {record.metric_value}")

        except Exception as e:
            logger.error(f"实时告警处理失败: {e}")

    def _update_performance_metrics(self, record: MetricRecord):
        """更新性能指标"""
        try:
            # 更新内存中的性能统计
            metric_key = f"{record.component_name}:{record.metric_name}"

            # 这里可以添加更复杂的性能分析逻辑
            # 例如：计算移动平均、标准差、趋势分析等

        except Exception as e:
            logger.error(f"更新性能指标失败: {e}")

    async def _detect_anomalies(self, record: MetricRecord):
        """检测异常"""
        try:
            # 异常检测逻辑
            # 可以使用统计方法、机器学习模型等

            # 简单的阈值检测示例
            if record.metric_name == 'error_rate' and record.metric_value > 0.1:
                logger.warning(f"检测到高错误率异常: {record.metric_value}")

            elif record.metric_name == 'cpu_usage' and record.metric_value > 90:
                logger.warning(f"检测到高CPU使用率异常: {record.metric_value}")

        except Exception as e:
            logger.error(f"异常检测失败: {e}")

    async def _update_performance_reports(self, strategy_id: str):
        """更新性能报告"""
        try:
            # 获取最近的指标数据
            if strategy_id in self.metrics_data:
                metrics = self.metrics_data[strategy_id]

                if metrics:
                    # 简化的性能报告
                    logger.debug(f"策略 {strategy_id} 性能报告: 收集了 {len(metrics)} 条指标")

        except Exception as e:
            logger.error(f"更新性能报告失败: {e}")

    def _calculate_average_metric(self, metrics: List[MetricData], metric_name: str) -> float:
        """计算指标平均值"""
        relevant_metrics = [m for m in metrics if m.metric_name == metric_name]
        if relevant_metrics:
            return sum(m.value for m in relevant_metrics) / len(relevant_metrics)
        return 0.0

    async def record_metric(self, metric_data: MetricData) -> bool:
        """记录指标（继承原有方法）"""
        try:
            strategy_id = metric_data.strategy_id

            if strategy_id not in self.metrics_data:
                self.metrics_data[strategy_id] = []

            # 添加指标数据
            self.metrics_data[strategy_id].append(metric_data)

            # 限制历史数据数量
            config = self.monitoring_configs.get(strategy_id)
            if config and len(self.metrics_data[strategy_id]) > config.max_metrics_history:
                self.metrics_data[strategy_id] = self.metrics_data[strategy_id][-config.max_metrics_history:]

            logger.debug(f"指标已记录: {strategy_id} - {metric_data.metric_name} = {metric_data.value}")
            return True

        except Exception as e:
            logger.error(f"记录指标失败: {e}")
            return False

    async def _publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """发布事件"""
        try:
            # 简化的事件发布
            logger.debug(f"事件发布: {event_type} - {event_data}")
        except Exception as e:
            logger.error(f"事件发布异常: {e}")

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

    def shutdown(self):
        """关闭监控服务"""
        logger.info("正在关闭增强的监控服务...")

        # 停止所有监控任务
        for strategy_id in list(self.monitoring_configs.keys()):
            self.stop_monitoring(strategy_id)

        # 停止持久化管理器
        self.persistence_manager.stop()

        logger.info("增强的监控服务已关闭")


def create_enhanced_monitoring_service(config: Optional[Dict] = None) -> EnhancedMonitoringService:
    """创建增强的监控服务实例"""
    return EnhancedMonitoringService(config)
