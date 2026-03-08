#!/usr/bin/env python3
"""
RQA2025 AI驱动性能优化器

基于机器学习和深度学习的智能性能监控和优化系统
支持实时性能分析、预测性优化、自动参数调优
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Tuple, Protocol
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

try:
    import pandas as pd
except ImportError:
    pd = None

from src.core.constants import (
    MAX_QUEUE_SIZE, SECONDS_PER_MINUTE, MAX_RETRIES,
    DEFAULT_BATCH_SIZE, MAX_RECORDS, DEFAULT_TIMEOUT,
    DEFAULT_TEST_TIMEOUT, MINUTES_PER_HOUR, DEFAULT_PAGE_SIZE
)

from ..monitoring.deep_learning_predictor import get_deep_learning_predictor

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):

    """优化模式"""
    REACTIVE = "reactive"      # 反应式优化
    PREDICTIVE = "predictive"  # 预测性优化
    PROACTIVE = "proactive"    # 主动式优化
    ADAPTIVE = "adaptive"      # 自适应优化


class PerformanceMetric(Enum):

    """性能指标"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_LENGTH = "queue_length"
    CONNECTION_COUNT = "connection_count"


@dataclass
class PerformanceData:

    """性能数据"""
    timestamp: datetime
    metrics: Dict[str, float]
    context: Dict[str, Any] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationAction:

    """优化动作"""
    action_id: str
    action_type: str
    target_component: str
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"


@dataclass
class PerformanceInsight:

    """性能洞察"""
    insight_id: str
    insight_type: str
    description: str
    severity: str
    confidence: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class PerformancePredictor:

    """
    性能预测器

    使用机器学习模型预测系统性能趋势
    """

    def __init__(self):

        self.dl_predictor = get_deep_learning_predictor()
        self.performance_history = deque(maxlen=MAX_QUEUE_SIZE)
        self.prediction_models = {}
        self.prediction_cache = {}
        self.is_trained = False

        logger.info("性能预测器初始化完成")

    async def collect_performance_data(self, metrics: Dict[str, Any]) -> None:
        """收集性能数据"""
        data_point = PerformanceData(
            timestamp=datetime.now(),
            metrics=metrics.copy()
        )
        self.performance_history.append(data_point)

    async def predict_performance_trend(self, metric_name: str,
                                        prediction_horizon: int = SECONDS_PER_MINUTE) -> Dict[str, Any]:
        """
        预测性能趋势

        Args:
            metric_name: 指标名称
            prediction_horizon: 预测时间范围（秒）

        Returns:
            预测结果
        """
        try:
            # 检查是否有足够的训练数据
            if len(self.performance_history) < MAX_RETRIES:
                return {
                    'status': 'insufficient_data',
                    'message': '需要更多性能数据进行预测'
                }

            # 准备训练数据
            data_points = list(self.performance_history)
            df = self._prepare_prediction_data(data_points, metric_name)

            if df.empty:
                return {
                    'status': 'no_data',
                    'message': f'没有找到指标 {metric_name} 的数据'
                }

            # 使用深度学习模型进行预测
            model_name = f"performance_{metric_name}_predictor"
            prediction_result = self.dl_predictor.get_optimized_prediction(
                model_name, df, steps=prediction_horizon // DEFAULT_BATCH_SIZE  # 每10秒预测一步
            )

            if prediction_result['status'] == 'success':
                predictions = prediction_result.get('predictions', [])
                confidence_intervals = prediction_result.get('confidence_intervals', [])

                # 分析预测趋势
                trend_analysis = await self._analyze_prediction_trend(
                    predictions, confidence_intervals
                )

                return {
                    'status': 'success',
                    'predictions': predictions,
                    'confidence_intervals': confidence_intervals,
                    'trend_analysis': trend_analysis,
                    'prediction_horizon': prediction_horizon,
                    'timestamp': datetime.now().isoformat()
                }

            else:
                return {
                    'status': 'prediction_failed',
                    'message': prediction_result.get('message', '预测失败')
                }

        except Exception as e:
            logger.error(f"性能趋势预测失败: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _prepare_prediction_data(self, data_points: List[PerformanceData],


                                 metric_name: str) -> pd.DataFrame:
        """准备预测数据"""
        timestamps = []
        values = []

        for point in data_points:
            if metric_name in point.metrics:
                timestamps.append(point.timestamp)
                values.append(point.metrics[metric_name])

        if not timestamps:
            return pd.DataFrame()

        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        }).set_index('timestamp')

        return df

    async def _analyze_prediction_trend(self, predictions: List[float],
                                        confidence_intervals: List[Tuple[float, float]]) -> Dict[str, Any]:
        """分析预测趋势"""
        if not predictions:
            return {'trend': 'unknown'}

        # 计算趋势方向
        if len(predictions) >= 2:
            start_value = predictions[0]
            end_value = predictions[-1]
            trend_change = (end_value - start_value) / max(abs(start_value), 0.001)

            if trend_change > 0.05:
                trend = 'increasing'
            elif trend_change < -0.05:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'

        # 计算波动性
        if len(predictions) > 1:
            volatility = np.std(predictions) / max(np.mean(predictions), 0.001)
        else:
            volatility = 0.0

        # 计算置信度
        avg_confidence = 0.0
        if confidence_intervals:
            confidence_widths = [
                (upper - lower) / max(abs((upper + lower) / 2), 0.001)
                for lower, upper in confidence_intervals
            ]
            avg_confidence = 1.0 - min(np.mean(confidence_widths), 1.0)

        return {
            'trend': trend,
            'trend_change_percent': trend_change * MAX_RETRIES if 'trend_change' in locals() else 0,
            'volatility': volatility,
            'avg_confidence': avg_confidence,
            'prediction_range': {
                'min': min(predictions),
                'max': max(predictions),
                'avg': np.mean(predictions)
            }
        }

    async def detect_performance_anomalies(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测性能异常"""
        anomalies = []

        try:
            for metric_name, current_value in current_metrics.items():
                # 获取历史数据
                historical_values = [
                    point.metrics.get(metric_name, 0)
                    for point in self.performance_history
                    if metric_name in point.metrics
                ]

                if len(historical_values) < 50:  # 需要足够的样本
                    continue

                # 计算统计特征
                mean_value = np.mean(historical_values)
                std_value = np.std(historical_values)

                if std_value > 0:
                    z_score = abs(current_value - mean_value) / std_value

                    # 检测异常（Z - score > 3）
                    if z_score > 3:
                        anomaly = {
                            'metric': metric_name,
                            'current_value': current_value,
                            'expected_value': mean_value,
                            'z_score': z_score,
                            'severity': 'high' if z_score > 5 else 'medium',
                            'timestamp': datetime.now().isoformat()
                        }
                        anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            return []


# 性能优化相关协议
class OptimizationController(Protocol):
    """优化控制器协议"""
    async def start_optimization(self, mode: OptimizationMode): ...
    async def stop_optimization(self): ...


class OptimizationExecutor(Protocol):
    """优化执行器协议"""
    async def optimize_performance(self, current_metrics: Dict[str, Any]) -> List[OptimizationAction]: ...


class OptimizationMonitor(Protocol):
    """优化监控器协议"""
    def get_optimization_status(self) -> Dict[str, Any]: ...
    def get_optimization_history(self) -> List[Dict[str, Any]]: ...


class PerformancePredictor(Protocol):
    """性能预测器协议"""
    async def collect_performance_data(self, metrics: Dict[str, Any]): ...
    def predict_performance_trends(self) -> Dict[str, Any]: ...


@dataclass
class OptimizationConfig:
    """优化配置"""
    mode: OptimizationMode = OptimizationMode.ADAPTIVE
    max_concurrent_optimizations: int = 5
    optimization_interval: int = SECONDS_PER_MINUTE
    monitoring_interval: int = DEFAULT_TIMEOUT
    enable_auto_optimization: bool = True


class OptimizationControllerImpl:
    """优化控制器实现 - 职责：控制优化过程"""

    def __init__(self, config: OptimizationConfig, executor: OptimizationExecutor, monitor: OptimizationMonitor):
        self.config = config
        self.executor = executor
        self.monitor = monitor
        self.is_running = False
        self.optimization_task = None
        self.monitoring_task = None

    async def start_optimization(self, mode: OptimizationMode):
        """启动性能优化"""
        if self.is_running:
            return

        self.is_running = True
        self.config.mode = mode

        logger.info(f"启动AI性能优化，模式: {mode.value}")

        # 启动优化循环
        self.optimization_task = asyncio.create_task(self._optimization_loop())

        # 启动监控循环
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_optimization(self):
        """停止性能优化"""
        if not self.is_running:
            return

        self.is_running = False

        # 取消任务
        if self.optimization_task:
            self.optimization_task.cancel()
        if self.monitoring_task:
            self.monitoring_task.cancel()

        logger.info("AI性能优化已停止")

    async def _optimization_loop(self):
        """优化循环"""
        while self.is_running:
            try:
                # 这里可以实现定期的优化检查逻辑
                await asyncio.sleep(self.config.optimization_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"优化循环异常: {e}")

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 这里可以实现定期的监控逻辑
                await asyncio.sleep(self.config.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")


class OptimizationStrategiesImpl:
    """优化策略实现 - 职责：执行各种优化策略"""

    def __init__(self, config: OptimizationConfig, predictor: PerformancePredictor):
        self.config = config
        self.predictor = predictor
        self.optimization_strategies = {
            'cpu_optimization': self._optimize_cpu_usage,
            'memory_optimization': self._optimize_memory_usage,
            'io_optimization': self._optimize_io_performance,
            'network_optimization': self._optimize_network_performance,
            'concurrency_optimization': self._optimize_concurrency
        }

    async def execute_optimization(self, strategy_name: str, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """执行特定优化策略"""
        if strategy_name in self.optimization_strategies:
            return await self.optimization_strategies[strategy_name](metrics)
        return []

    async def _optimize_cpu_usage(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """优化CPU使用率"""
        actions = []
        cpu_usage = metrics.get('cpu_usage', 0)

        if cpu_usage > 90:
            actions.append(OptimizationAction(
                action_type=OptimizationActionType.SCALE_UP,
                target_resource="cpu",
                parameters={"scale_factor": 1.5},
                priority=OptimizationPriority.HIGH
            ))

        return actions

    async def _optimize_memory_usage(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """优化内存使用率"""
        actions = []
        memory_usage = metrics.get('memory_usage', 0)

        if memory_usage > 85:
            actions.append(OptimizationAction(
                action_type=OptimizationActionType.SCALE_MEMORY,
                target_resource="memory",
                parameters={"additional_gb": 2},
                priority=OptimizationPriority.HIGH
            ))

        return actions

    async def _optimize_io_performance(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """优化IO性能"""
        actions = []
        io_latency = metrics.get('io_latency', 0)

        if io_latency > MAX_RETRIES:  # 100ms
            actions.append(OptimizationAction(
                action_type=OptimizationActionType.OPTIMIZE_IO,
                target_resource="storage",
                parameters={"cache_size": "2GB"},
                priority=OptimizationPriority.MEDIUM
            ))

        return actions

    async def _optimize_network_performance(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """优化网络性能"""
        actions = []
        network_latency = metrics.get('network_latency', 0)

        if network_latency > 50:  # 50ms
            actions.append(OptimizationAction(
                action_type=OptimizationActionType.OPTIMIZE_NETWORK,
                target_resource="network",
                parameters={"connection_pool_size": MAX_RETRIES},
                priority=OptimizationPriority.MEDIUM
            ))

        return actions

    async def _optimize_concurrency(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """优化并发性能"""
        actions = []
        active_connections = metrics.get('active_connections', 0)

        if active_connections > MAX_RECORDS:
            actions.append(OptimizationAction(
                action_type=OptimizationActionType.SCALE_CONCURRENCY,
                target_resource="concurrency",
                parameters={"max_workers": 50},
                priority=OptimizationPriority.MEDIUM
            ))

        return actions


class OptimizationMonitorImpl:
    """优化监控器实现 - 职责：监控优化状态和历史"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimization_actions = deque(maxlen=MAX_RECORDS)
        self.performance_insights = deque(maxlen=DEFAULT_PAGE_SIZE * 5)  # 500

    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        return {
            "is_running": False,  # 这里需要连接到控制器
            "total_actions": len(self.optimization_actions),
            "recent_actions": list(self.optimization_actions)[-DEFAULT_BATCH_SIZE:],
            "insights_count": len(self.performance_insights)
        }

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return list(self.optimization_actions)

    def record_optimization_action(self, action: OptimizationAction):
        """记录优化动作"""
        self.optimization_actions.append({
            "timestamp": time.time(),
            "action": action.dict() if hasattr(action, 'dict') else str(action)
        })

    def record_performance_insight(self, insight: Dict[str, Any]):
        """记录性能洞察"""
        insight_with_timestamp = {
            "timestamp": time.time(),
            **insight
        }
        self.performance_insights.append(insight_with_timestamp)


class OptimizationExecutorImpl:
    """优化执行器实现 - 职责：执行优化逻辑"""

    def __init__(self, config: OptimizationConfig, strategies: OptimizationStrategiesImpl,
                 predictor: PerformancePredictor, monitor: OptimizationMonitorImpl):
        self.config = config
        self.strategies = strategies
        self.predictor = predictor
        self.monitor = monitor

    async def optimize_performance(self, current_metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """执行性能优化"""
        try:
            actions = []

            # 收集性能数据用于预测
            await self.predictor.collect_performance_data(current_metrics)

            # 根据优化模式选择策略
            if self.config.mode == OptimizationMode.REACTIVE:
                actions.extend(await self._reactive_optimization(current_metrics))
            elif self.config.mode == OptimizationMode.PREDICTIVE:
                actions.extend(await self._predictive_optimization(current_metrics))
            elif self.config.mode == OptimizationMode.PROACTIVE:
                actions.extend(await self._proactive_optimization(current_metrics))
            elif self.config.mode == OptimizationMode.ADAPTIVE:
                actions.extend(await self._adaptive_optimization(current_metrics))

            # 记录优化动作
            for action in actions:
                self.monitor.record_optimization_action(action)

            return actions

        except Exception as e:
            logger.error(f"性能优化执行失败: {e}")
            return []

    async def _reactive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """被动优化 - 响应当前性能问题"""
        actions = []

        # CPU优化
        actions.extend(await self.strategies._optimize_cpu_usage(metrics))

        # 内存优化
        actions.extend(await self.strategies._optimize_memory_usage(metrics))

        # IO优化
        actions.extend(await self.strategies._optimize_io_performance(metrics))

        # 网络优化
        actions.extend(await self.strategies._optimize_network_performance(metrics))

        return actions

    async def _predictive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """预测性优化 - 基于预测提前优化"""
        actions = []

        # 获取性能预测
        predictions = self.predictor.predict_performance_trends()

        # 基于预测结果优化
        predicted_load = predictions.get('predicted_load', 1.0)
        if predicted_load > 1.5:  # 预计负载增加50%
            actions.append(OptimizationAction(
                action_type=OptimizationActionType.PRE_SCALE,
                target_resource="general",
                parameters={"scale_factor": predicted_load},
                priority=OptimizationPriority.MEDIUM
            ))

        return actions

    async def _proactive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """主动优化 - 持续优化以提高性能"""
        actions = []

        # 并发优化
        actions.extend(await self.strategies._optimize_concurrency(metrics))

        # 定期优化建议
        actions.append(OptimizationAction(
            action_type=OptimizationActionType.HEALTH_CHECK,
            target_resource="system",
            parameters={},
            priority=OptimizationPriority.LOW
        ))

        return actions

    async def _adaptive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """自适应优化 - 结合多种策略的智能优化"""
        actions = []

        # 结合被动和预测性优化
        reactive_actions = await self._reactive_optimization(metrics)
        predictive_actions = await self._predictive_optimization(metrics)

        actions.extend(reactive_actions)
        actions.extend(predictive_actions)

        # 自适应调整
        if len(actions) > self.config.max_concurrent_optimizations:
            # 按优先级排序，只保留最重要的
            actions.sort(key=lambda x: x.priority.value, reverse=True)
            actions = actions[:self.config.max_concurrent_optimizations]

        return actions


class PerformanceOptimizer:

    """
    性能优化器 - 重构版：组合模式

    基于AI分析自动优化系统性能
    """

    def __init__(self):
        # 初始化配置
        self.config = OptimizationConfig()

        # 初始化预测器
        self.predictor = PerformancePredictor()

        # 初始化监控器
        self.monitor = OptimizationMonitorImpl(self.config)

        # 初始化策略
        self.strategies = OptimizationStrategiesImpl(self.config, self.predictor)

        # 初始化执行器
        self.executor = OptimizationExecutorImpl(self.config, self.strategies, self.predictor, self.monitor)

        # 初始化控制器
        self.controller = OptimizationControllerImpl(self.config, self.executor, self.monitor)

        # 兼容性属性
        self.optimization_actions = self.monitor.optimization_actions
        self.performance_insights = self.monitor.performance_insights
        self.optimization_mode = self.config.mode
        self.is_running = False

        logger.info("重构后的性能优化器初始化完成")

    # 代理方法到专门的组件
    async def start_optimization(self, mode: OptimizationMode = OptimizationMode.ADAPTIVE):
        """启动性能优化 - 代理到控制器"""
        self.is_running = True
        self.optimization_mode = mode
        return await self.controller.start_optimization(mode)

    async def stop_optimization(self):
        """停止性能优化 - 代理到控制器"""
        self.is_running = False
        return await self.controller.stop_optimization()

    async def optimize_performance(self, current_metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """执行性能优化 - 代理到执行器"""
        return await self.executor.optimize_performance(current_metrics)

    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态 - 代理到监控器"""
        status = self.monitor.get_optimization_status()
        status["is_running"] = self.is_running
        return status

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史 - 代理到监控器"""
        return self.monitor.get_optimization_history()

    def get_performance_insights(self) -> List[Dict[str, Any]]:
        """获取性能洞察 - 代理到监控器"""
        return list(self.monitor.performance_insights)

    # 保持向后兼容性
    async def _optimization_loop(self):
        """优化循环（向后兼容）"""
        return await self.controller._optimization_loop()

    async def _monitoring_loop(self):
        """监控循环（向后兼容）"""
        return await self.controller._monitoring_loop()

    async def _reactive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """被动优化（向后兼容）"""
        return await self.executor._reactive_optimization(metrics)

    async def _predictive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """预测性优化（向后兼容）"""
        return await self.executor._predictive_optimization(metrics)

    async def _proactive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """主动优化（向后兼容）"""
        return await self.executor._proactive_optimization(metrics)

    async def _adaptive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """自适应优化（向后兼容）"""
        return await self.executor._adaptive_optimization(metrics)


class IntelligentPerformanceMonitor:

    """
    智能性能监控器

    结合AI预测和实时监控的智能性能管理系统
    """

    def __init__(self):

        self.predictor = PerformancePredictor()
        self.optimizer = PerformanceOptimizer()
        self.monitoring_tasks = {}
        self.alert_thresholds = {}
        self.is_running = False

        logger.info("智能性能监控器初始化完成")

    async def start_monitoring(self, optimization_mode: OptimizationMode = OptimizationMode.ADAPTIVE):
        """启动智能监控"""
        if self.is_running:
            return

        self.is_running = True

        # 启动性能优化器
        await self.optimizer.start_optimization(optimization_mode)

        # 启动监控任务
        await self._start_monitoring_tasks()

        logger.info("智能性能监控启动完成")

    async def stop_monitoring(self):
        """停止智能监控"""
        if not self.is_running:
            return

        self.is_running = False

        # 停止优化器
        await self.optimizer.stop_optimization()

        # 停止监控任务
        await self._stop_monitoring_tasks()

        logger.info("智能性能监控已停止")

    async def _start_monitoring_tasks(self):
        """启动监控任务"""
        # 实时性能监控任务
        self.monitoring_tasks['realtime'] = asyncio.create_task(self._realtime_monitoring_task())

        # 预测监控任务
        self.monitoring_tasks['predictive'] = asyncio.create_task(
            self._predictive_monitoring_task())

        # 健康检查任务
        self.monitoring_tasks['health'] = asyncio.create_task(self._health_check_task())

    async def _stop_monitoring_tasks(self):
        """停止监控任务"""
        for task in self.monitoring_tasks.values():
            task.cancel()

        self.monitoring_tasks.clear()

    async def _realtime_monitoring_task(self):
        """实时监控任务"""
        while self.is_running:
            try:
                # 获取实时指标
                metrics = await self.optimizer._get_current_performance_metrics()

                # 实时优化
                await self.optimizer.optimize_performance(metrics)

                # 检查告警阈值
                await self._check_alert_thresholds(metrics)

                await asyncio.sleep(DEFAULT_BATCH_SIZE)  # 每10秒监控一次

            except Exception as e:
                logger.error(f"实时监控任务错误: {e}")
                await asyncio.sleep(DEFAULT_BATCH_SIZE)

    async def _predictive_monitoring_task(self):
        """预测监控任务"""
        while self.is_running:
            try:
                # 预测关键指标
                predictions = {}
                key_metrics = ['cpu_usage', 'memory_usage', 'response_time']

                for metric in key_metrics:
                    # 10分钟预测
                    prediction = await self.predictor.predict_performance_trend(metric, SECONDS_PER_MINUTE * MINUTES_PER_HOUR)
                    if prediction['status'] == 'success':
                        predictions[metric] = prediction

                # 基于预测结果进行预防性优化
                await self._predictive_optimization(predictions)

                await asyncio.sleep(DEFAULT_TEST_TIMEOUT)  # 每5分钟预测一次

            except Exception as e:
                logger.error(f"预测监控任务错误: {e}")
                await asyncio.sleep(DEFAULT_TEST_TIMEOUT)

    async def _health_check_task(self):
        """健康检查任务"""
        while self.is_running:
            try:
                # 执行系统健康检查
                health_status = await self._perform_health_check()

                # 根据健康状态调整优化策略
                await self._adjust_optimization_strategy(health_status)

                await asyncio.sleep(SECONDS_PER_MINUTE)  # 每分钟检查一次

            except Exception as e:
                logger.error(f"健康检查任务错误: {e}")
                await asyncio.sleep(SECONDS_PER_MINUTE)

    async def _check_alert_thresholds(self, metrics: Dict[str, Any]):
        """检查告警阈值"""
        try:
            alerts = []

            for metric_name, value in metrics.items():
                if metric_name in self.alert_thresholds:
                    threshold = self.alert_thresholds[metric_name]

                    if value > threshold.get('critical', float('inf')):
                        alerts.append({
                            'level': 'critical',
                            'metric': metric_name,
                            'value': value,
                            'threshold': threshold['critical']
                        })
                    elif value > threshold.get('warning', float('inf')):
                        alerts.append({
                            'level': 'warning',
                            'metric': metric_name,
                            'value': value,
                            'threshold': threshold['warning']
                        })

            # 发送告警
            for alert in alerts:
                await self._send_performance_alert(alert)

        except Exception as e:
            logger.warning(f"检查告警阈值失败: {e}")

    async def _predictive_optimization(self, predictions: Dict[str, Any]):
        """预测性优化"""
        try:
            for metric, prediction in predictions.items():
                trend_analysis = prediction.get('trend_analysis', {})

                # 如果预测会出现性能问题，提前优化
                if trend_analysis.get('trend') == 'increasing':
                    predicted_max = trend_analysis.get('prediction_range', {}).get('max', 0)

                    if metric == 'cpu_usage' and predicted_max > 75:
                        await self.optimizer._create_optimization_action(
                            'cpu_optimization', 'system', {'predictive': True}
                        )
                    elif metric == 'memory_usage' and predicted_max > 80:
                        await self.optimizer._create_optimization_action(
                            'memory_optimization', 'system', {'predictive': True}
                        )

        except Exception as e:
            logger.warning(f"预测性优化失败: {e}")

    async def _perform_health_check(self) -> Dict[str, Any]:
        """执行健康检查"""
        try:
            # 获取系统状态
            system_status = await self.optimizer._get_current_performance_metrics()

            # 计算健康评分
            health_score = await self._calculate_health_score(system_status)

            return {
                'health_score': health_score,
                'system_status': system_status,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {'health_score': 0, 'error': str(e)}

    async def _calculate_health_score(self, system_status: Dict[str, Any]) -> float:
        """计算健康评分"""
        try:
            score = MAX_RETRIES  # 100.0

            # CPU健康评分
            cpu_usage = system_status.get('cpu_usage', 0)
            if cpu_usage > 80:
                score -= (cpu_usage - 80) * 0.5

            # 内存健康评分
            memory_usage = system_status.get('memory_usage', 0)
            if memory_usage > 85:
                score -= (memory_usage - 85) * 0.5

            # 响应时间健康评分
            response_time = system_status.get('response_time', 0)
            if response_time > MAX_RECORDS:
                score -= min((response_time - MAX_RECORDS) / MAX_RETRIES, 20)

            return max(0, min(MAX_RETRIES, score))

        except Exception:
            return 50.0  # 默认中等健康评分

    async def _adjust_optimization_strategy(self, health_status: Dict[str, Any]):
        """调整优化策略"""
        try:
            health_score = health_status.get('health_score', 50)

            # 根据健康评分调整优化模式
            if health_score < DEFAULT_TIMEOUT:
                self.optimizer.optimization_mode = OptimizationMode.REACTIVE
                logger.info("系统健康评分低，切换到反应式优化模式")
            elif health_score < SECONDS_PER_MINUTE:
                self.optimizer.optimization_mode = OptimizationMode.PREDICTIVE
                logger.info("系统健康评分中等，切换到预测性优化模式")
            else:
                self.optimizer.optimization_mode = OptimizationMode.ADAPTIVE
                logger.info("系统健康评分良好，使用自适应优化模式")

        except Exception as e:
            logger.warning(f"调整优化策略失败: {e}")

    async def _send_performance_alert(self, alert: Dict[str, Any]):
        """发送性能告警"""
        try:
            logger.warning(f"性能告警: {alert}")

            # 这里可以集成告警系统发送通知
            # 例如：邮件、短信、Slack等

        except Exception as e:
            logger.error(f"发送性能告警失败: {e}")

    def set_alert_threshold(self, metric_name: str, warning: float, critical: float):
        """设置告警阈值"""
        self.alert_thresholds[metric_name] = {
            'warning': warning,
            'critical': critical
        }
        logger.info(f"告警阈值已设置: {metric_name} (警告: {warning}, 严重: {critical})")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            'is_running': self.is_running,
            'optimizer_status': self.optimizer.get_optimization_status(),
            'alert_thresholds': self.alert_thresholds.copy(),
            'active_tasks': len(self.monitoring_tasks)
        }


# 全局实例
_performance_optimizer_instance = None
_intelligent_monitor_instance = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """获取性能优化器实例"""
    global _performance_optimizer_instance
    if _performance_optimizer_instance is None:
        _performance_optimizer_instance = PerformanceOptimizer()
    return _performance_optimizer_instance


def get_intelligent_performance_monitor() -> IntelligentPerformanceMonitor:
    """获取智能性能监控器实例"""
import numpy as np
    global _intelligent_monitor_instance
    if _intelligent_monitor_instance is None:
        _intelligent_monitor_instance = IntelligentPerformanceMonitor()
    return _intelligent_monitor_instance


if __name__ == "__main__":
    # 测试代码
    print("AI性能优化器测试")

    async def test_performance_optimizer():
        # 获取优化器实例
        optimizer = get_performance_optimizer()
        monitor = get_intelligent_performance_monitor()

        # 设置告警阈值
        monitor.set_alert_threshold('cpu_usage', 70, 90)
        monitor.set_alert_threshold('memory_usage', 75, 90)

        # 启动智能监控
        await monitor.start_monitoring(OptimizationMode.ADAPTIVE)

        try:
            # 运行10分钟测试
            await asyncio.sleep(SECONDS_PER_MINUTE * MINUTES_PER_HOUR)

        finally:
            # 停止监控
            await monitor.stop_monitoring()

        # 获取优化历史
        history = optimizer.get_optimization_history()
        insights = optimizer.get_performance_insights()

        print("测试结果:")
        print(f"  优化动作总数: {len(history)}")
        print(f"  性能洞察总数: {len(insights)}")

        # 显示最近的洞察
        for insight in insights[-5:]:
            print(f"  洞察: {insight.description} (严重程度: {insight.severity})")

    # 运行测试
    asyncio.run(test_performance_optimizer())
    print("AI性能优化器测试完成")


# 导出和别名
AIPerformanceOptimizer = PerformanceOptimizer

__all__ = ['PerformanceOptimizer', 'AIPerformanceOptimizer', 'OptimizationMode', 'PerformanceMetric']
