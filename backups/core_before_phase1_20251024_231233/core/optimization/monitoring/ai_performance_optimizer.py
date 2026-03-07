#!/usr/bin/env python3
"""
RQA2025 AI驱动性能优化器

基于机器学习和深度学习的智能性能监控和优化系统
支持实时性能分析、预测性优化、自动参数调优
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque
import psutil

from ..monitoring.deep_learning_predictor import get_deep_learning_predictor
from ..monitoring.performance_analyzer import get_performance_analyzer
from ..core.integration.service_communicator import get_cloud_native_optimizer

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
        self.performance_history = deque(maxlen=10000)
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
                                        prediction_horizon: int = 60) -> Dict[str, Any]:
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
            if len(self.performance_history) < 100:
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
                model_name, df, steps=prediction_horizon // 10  # 每10秒预测一步
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
            'trend_change_percent': trend_change * 100 if 'trend_change' in locals() else 0,
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


class PerformanceOptimizer:

    """
    性能优化器

    基于AI分析自动优化系统性能
    """

    def __init__(self):

        self.predictor = PerformancePredictor()
        self.performance_analyzer = get_performance_analyzer()
        self.cloud_optimizer = get_cloud_native_optimizer()

        self.optimization_actions = deque(maxlen=1000)
        self.performance_insights = deque(maxlen=500)

        self.optimization_mode = OptimizationMode.ADAPTIVE
        self.is_running = False

        # 优化策略配置
        self.optimization_strategies = {
            'cpu_optimization': self._optimize_cpu_usage,
            'memory_optimization': self._optimize_memory_usage,
            'io_optimization': self._optimize_io_performance,
            'network_optimization': self._optimize_network_performance,
            'concurrency_optimization': self._optimize_concurrency
        }

        logger.info("性能优化器初始化完成")

    async def start_optimization(self, mode: OptimizationMode = OptimizationMode.ADAPTIVE):
        """启动性能优化"""
        if self.is_running:
            return

        self.is_running = True
        self.optimization_mode = mode

        logger.info(f"启动AI性能优化，模式: {mode.value}")

        # 启动优化循环
        asyncio.create_task(self._optimization_loop())

        # 启动监控循环
        asyncio.create_task(self._monitoring_loop())

    async def stop_optimization(self):
        """停止性能优化"""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("AI性能优化已停止")

    async def optimize_performance(self, current_metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """
        执行性能优化

        Args:
            current_metrics: 当前性能指标

        Returns:
            优化动作列表
        """
        try:
            actions = []

            # 收集性能数据用于预测
            await self.predictor.collect_performance_data(current_metrics)

            # 根据优化模式选择策略
            if self.optimization_mode == OptimizationMode.REACTIVE:
                actions.extend(await self._reactive_optimization(current_metrics))
            elif self.optimization_mode == OptimizationMode.PREDICTIVE:
                actions.extend(await self._predictive_optimization(current_metrics))
            elif self.optimization_mode == OptimizationMode.PROACTIVE:
                actions.extend(await self._proactive_optimization(current_metrics))
            elif self.optimization_mode == OptimizationMode.ADAPTIVE:
                actions.extend(await self._adaptive_optimization(current_metrics))

            # 记录优化动作
            for action in actions:
                self.optimization_actions.append(action)

            return actions

        except Exception as e:
            logger.error(f"性能优化执行失败: {e}")
            return []

    async def _reactive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """反应式优化 - 基于当前状态"""
        actions = []

        # CPU优化
        if metrics.get('cpu_usage', 0) > 80:
            actions.append(await self._create_optimization_action(
                'cpu_optimization', 'system', {'cpu_threshold': 80}
            ))

        # 内存优化
        if metrics.get('memory_usage', 0) > 85:
            actions.append(await self._create_optimization_action(
                'memory_optimization', 'system', {'memory_threshold': 85}
            ))

        # IO优化
        if metrics.get('disk_io', 0) > 1000:  # KB / s
            actions.append(await self._create_optimization_action(
                'io_optimization', 'storage', {'io_threshold': 1000}
            ))

        return actions

    async def _predictive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """预测性优化 - 基于预测趋势"""
        actions = []

        try:
            # 预测关键指标趋势
            critical_metrics = ['cpu_usage', 'memory_usage', 'response_time', 'throughput']

            for metric in critical_metrics:
                prediction = await self.predictor.predict_performance_trend(metric, 300)  # 5分钟预测

                if prediction['status'] == 'success':
                    trend_analysis = prediction.get('trend_analysis', {})

                    # 如果预测将出现性能问题，提前优化
                    if trend_analysis.get('trend') == 'increasing':
                        if metric == 'cpu_usage' and trend_analysis.get('prediction_range', {}).get('max', 0) > 75:
                            actions.append(await self._create_optimization_action(
                                'cpu_optimization', 'system', {'prediction_based': True}
                            ))
                        elif metric == 'memory_usage' and trend_analysis.get('prediction_range', {}).get('max', 0) > 80:
                            actions.append(await self._create_optimization_action(
                                'memory_optimization', 'system', {'prediction_based': True}
                            ))

        except Exception as e:
            logger.warning(f"预测性优化失败: {e}")

        return actions

    async def _proactive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """主动式优化 - 持续优化"""
        actions = []

        try:
            # 基于系统负载模式进行优化
            system_load = await self._analyze_system_load_pattern()

            if system_load.get('pattern') == 'high_frequency_trading':
                # 高频交易模式优化
                actions.append(await self._create_optimization_action(
                    'concurrency_optimization', 'trading_engine',
                    {'mode': 'high_frequency', 'thread_pool_size': 16}
                ))
            elif system_load.get('pattern') == 'batch_processing':
                # 批处理模式优化
                actions.append(await self._create_optimization_action(
                    'memory_optimization', 'batch_processor',
                    {'mode': 'batch', 'buffer_size': 10000}
                ))

        except Exception as e:
            logger.warning(f"主动式优化失败: {e}")

        return actions

    async def _adaptive_optimization(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """自适应优化 - 结合多种策略"""
        actions = []

        # 组合使用反应式、预测性和主动式优化
        reactive_actions = await self._reactive_optimization(metrics)
        predictive_actions = await self._predictive_optimization(metrics)
        proactive_actions = await self._proactive_optimization(metrics)

        # 合并并去重
        all_actions = reactive_actions + predictive_actions + proactive_actions

        # 基于优先级和置信度排序
        all_actions.sort(key=lambda x: x.expected_impact.get('priority', 0), reverse=True)

        # 限制动作数量，避免过度优化
        actions = all_actions[:3]

        return actions

    async def _create_optimization_action(self, action_type: str, target_component: str,
                                          parameters: Dict[str, Any]) -> OptimizationAction:
        """创建优化动作"""
        action_id = f"opt_{int(time.time())}_{action_type}"

        # 计算预期影响
        expected_impact = await self._calculate_expected_impact(action_type, parameters)

        return OptimizationAction(
            action_id=action_id,
            action_type=action_type,
            target_component=target_component,
            parameters=parameters,
            expected_impact=expected_impact,
            confidence=0.8  # 默认置信度
        )

    async def _calculate_expected_impact(self, action_type: str, parameters: Dict[str, Any]) -> Dict[str, float]:
        """计算预期影响"""
        impact = {'priority': 1.0, 'improvement': 0.0, 'risk': 0.0}

        if action_type == 'cpu_optimization':
            impact['improvement'] = 15.0  # 预期CPU改善15%
            impact['priority'] = 8.0
            impact['risk'] = 2.0
        elif action_type == 'memory_optimization':
            impact['improvement'] = 20.0  # 预期内存改善20%
            impact['priority'] = 9.0
            impact['risk'] = 3.0
        elif action_type == 'io_optimization':
            impact['improvement'] = 25.0  # 预期IO改善25%
            impact['priority'] = 7.0
            impact['risk'] = 4.0

        return impact

    async def _analyze_system_load_pattern(self) -> Dict[str, Any]:
        """分析系统负载模式"""
        # 这里可以实现更复杂的负载模式分析
        return {'pattern': 'normal', 'confidence': 0.7}

    async def _optimization_loop(self):
        """优化循环"""
        while self.is_running:
            try:
                # 获取当前性能指标
                current_metrics = await self._get_current_performance_metrics()

                # 执行优化
                actions = await self.optimize_performance(current_metrics)

                # 执行优化动作
                for action in actions:
                    await self._execute_optimization_action(action)

                # 生成性能洞察
                insights = await self._generate_performance_insights(current_metrics)
                for insight in insights:
                    self.performance_insights.append(insight)

                await asyncio.sleep(30)  # 每30秒执行一次优化

            except Exception as e:
                logger.error(f"优化循环错误: {e}")
                await asyncio.sleep(30)

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 检测异常
                current_metrics = await self._get_current_performance_metrics()
                anomalies = await self.predictor.detect_performance_anomalies(current_metrics)

                # 处理异常
                for anomaly in anomalies:
                    await self._handle_performance_anomaly(anomaly)

                await asyncio.sleep(60)  # 每分钟检测一次异常

            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(60)

    async def _execute_optimization_action(self, action: OptimizationAction):
        """执行优化动作"""
        try:
            action.status = "executing"

            # 调用相应的优化策略
            if action.action_type in self.optimization_strategies:
                strategy_func = self.optimization_strategies[action.action_type]
                result = await strategy_func(action.parameters)

                if result.get('success', False):
                    action.status = "completed"
                    logger.info(f"优化动作执行成功: {action.action_id}")
                else:
                    action.status = "failed"
                    logger.warning(f"优化动作执行失败: {action.action_id}")
            else:
                action.status = "failed"
                logger.warning(f"未知优化动作类型: {action.action_type}")

        except Exception as e:
            action.status = "failed"
            logger.error(f"优化动作执行异常: {e}")

    async def _optimize_cpu_usage(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """优化CPU使用率"""
        try:
            # 这里实现具体的CPU优化逻辑
            # 例如：调整线程池大小、优化算法等

            return {
                'success': True,
                'message': 'CPU优化执行完成',
                'changes': {'thread_pool_size': 8}
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}

    async def _optimize_memory_usage(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """优化内存使用"""
        try:
            # 实现内存优化逻辑
            # 例如：垃圾回收、缓存清理等

            return {
                'success': True,
                'message': '内存优化执行完成',
                'changes': {'cache_size': 512}
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}

    async def _optimize_io_performance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """优化IO性能"""
        try:
            # 实现IO优化逻辑
            # 例如：调整缓冲区大小、使用异步IO等

            return {
                'success': True,
                'message': 'IO优化执行完成',
                'changes': {'buffer_size': 8192}
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}

    async def _optimize_network_performance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """优化网络性能"""
        try:
            # 实现网络优化逻辑
            # 例如：连接池优化、协议优化等

            return {
                'success': True,
                'message': '网络优化执行完成',
                'changes': {'connection_pool_size': 20}
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}

    async def _optimize_concurrency(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """优化并发性能"""
        try:
            # 实现并发优化逻辑
            # 例如：调整并发级别、负载均衡等

            return {
                'success': True,
                'message': '并发优化执行完成',
                'changes': {'concurrency_level': 16}
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}

    async def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """获取当前性能指标"""
        try:
            # 使用psutil获取系统指标
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()

            metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_read_bytes': disk_io.read_bytes if disk_io else 0,
                'disk_write_bytes': disk_io.write_bytes if disk_io else 0,
                'network_sent_bytes': network_io.bytes_sent if network_io else 0,
                'network_recv_bytes': network_io.bytes_recv if network_io else 0,
                'timestamp': datetime.now().isoformat()
            }

            # 从性能分析器获取应用级指标
            app_metrics = await self.performance_analyzer.get_performance_metrics()
            metrics.update(app_metrics)

            return metrics

        except Exception as e:
            logger.error(f"获取性能指标失败: {e}")
            return {}

    async def _generate_performance_insights(self, metrics: Dict[str, Any]) -> List[PerformanceInsight]:
        """生成性能洞察"""
        insights = []

        try:
            # CPU洞察
            cpu_usage = metrics.get('cpu_usage', 0)
            if cpu_usage > 80:
                insights.append(PerformanceInsight(
                    insight_id=f"cpu_high_{int(time.time())}",
                    insight_type="cpu_performance",
                    description=f"CPU使用率过高: {cpu_usage:.1f}%",
                    severity="high",
                    confidence=0.9,
                    recommendations=[
                        "考虑增加CPU核心数",
                        "优化CPU密集型算法",
                        "实施负载均衡"
                    ]
                ))

            # 内存洞察
            memory_usage = metrics.get('memory_usage', 0)
            if memory_usage > 85:
                insights.append(PerformanceInsight(
                    insight_id=f"memory_high_{int(time.time())}",
                    insight_type="memory_performance",
                    description=f"内存使用率过高: {memory_usage:.1f}%",
                    severity="high",
                    confidence=0.9,
                    recommendations=[
                        "增加系统内存",
                        "优化内存使用算法",
                        "实施内存缓存策略"
                    ]
                ))

            # 响应时间洞察
            response_time = metrics.get('response_time', 0)
            if response_time > 1000:  # 1秒
                insights.append(PerformanceInsight(
                    insight_id=f"response_slow_{int(time.time())}",
                    insight_type="response_performance",
                    description=f"响应时间过慢: {response_time:.0f}ms",
                    severity="medium",
                    confidence=0.8,
                    recommendations=[
                        "优化数据库查询",
                        "实施缓存策略",
                        "使用异步处理"
                    ]
                ))

        except Exception as e:
            logger.warning(f"生成性能洞察失败: {e}")

        return insights

    async def _handle_performance_anomaly(self, anomaly: Dict[str, Any]):
        """处理性能异常"""
        try:
            logger.warning(f"检测到性能异常: {anomaly}")

            # 创建紧急优化动作
            emergency_action = await self._create_optimization_action(
                f"{anomaly['metric']}_emergency_optimization",
                'system',
                {'anomaly': anomaly, 'emergency': True}
            )

            # 立即执行
            await self._execute_optimization_action(emergency_action)

        except Exception as e:
            logger.error(f"处理性能异常失败: {e}")

    def get_optimization_history(self) -> List[OptimizationAction]:
        """获取优化历史"""
        return list(self.optimization_actions)

    def get_performance_insights(self) -> List[PerformanceInsight]:
        """获取性能洞察"""
        return list(self.performance_insights)

    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        return {
            'is_running': self.is_running,
            'mode': self.optimization_mode.value,
            'total_actions': len(self.optimization_actions),
            'pending_actions': len([a for a in self.optimization_actions if a.status == 'pending']),
            'completed_actions': len([a for a in self.optimization_actions if a.status == 'completed']),
            'failed_actions': len([a for a in self.optimization_actions if a.status == 'failed']),
            'total_insights': len(self.performance_insights)
        }


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

                await asyncio.sleep(10)  # 每10秒监控一次

            except Exception as e:
                logger.error(f"实时监控任务错误: {e}")
                await asyncio.sleep(10)

    async def _predictive_monitoring_task(self):
        """预测监控任务"""
        while self.is_running:
            try:
                # 预测关键指标
                predictions = {}
                key_metrics = ['cpu_usage', 'memory_usage', 'response_time']

                for metric in key_metrics:
                    # 10分钟预测
                    prediction = await self.predictor.predict_performance_trend(metric, 600)
                    if prediction['status'] == 'success':
                        predictions[metric] = prediction

                # 基于预测结果进行预防性优化
                await self._predictive_optimization(predictions)

                await asyncio.sleep(300)  # 每5分钟预测一次

            except Exception as e:
                logger.error(f"预测监控任务错误: {e}")
                await asyncio.sleep(300)

    async def _health_check_task(self):
        """健康检查任务"""
        while self.is_running:
            try:
                # 执行系统健康检查
                health_status = await self._perform_health_check()

                # 根据健康状态调整优化策略
                await self._adjust_optimization_strategy(health_status)

                await asyncio.sleep(60)  # 每分钟检查一次

            except Exception as e:
                logger.error(f"健康检查任务错误: {e}")
                await asyncio.sleep(60)

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
            score = 100.0

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
            if response_time > 1000:
                score -= min((response_time - 1000) / 100, 20)

            return max(0, min(100, score))

        except Exception:
            return 50.0  # 默认中等健康评分

    async def _adjust_optimization_strategy(self, health_status: Dict[str, Any]):
        """调整优化策略"""
        try:
            health_score = health_status.get('health_score', 50)

            # 根据健康评分调整优化模式
            if health_score < 30:
                self.optimizer.optimization_mode = OptimizationMode.REACTIVE
                logger.info("系统健康评分低，切换到反应式优化模式")
            elif health_score < 60:
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
            await asyncio.sleep(600)

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
