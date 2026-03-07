#!/usr/bin/env python3
"""
RQA2025 实时性能监控和分析工具

提供全面的系统性能监控、实时分析和瓶颈识别功能
"""

import time
import threading
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict, deque
import asyncio
import warnings

try:
    from ...ai.deep_learning_predictor import get_deep_learning_predictor
except ImportError:
    def get_deep_learning_predictor():
        return None

try:
    from ..core.integration.service_communicator import get_cloud_native_optimizer
except ImportError:
    def get_cloud_native_optimizer():
        return None

logger = logging.getLogger(__name__)

# 过滤警告
warnings.filterwarnings('ignore')


class PerformanceMetric(Enum):

    """性能指标枚举"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    SYSTEM_LOAD = "system_load"
    PROCESS_COUNT = "process_count"
    THREAD_COUNT = "thread_count"
    CONTEXT_SWITCHES = "context_switches"
    PAGE_FAULTS = "page_faults"


class AnalysisMode(Enum):

    """分析模式枚举"""
    REALTIME = "realtime"        # 实时分析
    HISTORICAL = "historical"    # 历史分析
    COMPARATIVE = "comparative"  # 对比分析
    TREND = "trend"             # 趋势分析
    ANOMALY = "anomaly"         # 异常分析


@dataclass
class PerformanceData:

    """性能数据结构"""
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckAnalysis:

    """瓶颈分析结果"""
    component: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    recommendations: List[str]
    impact_score: float
    confidence: float


@dataclass
class PerformanceReport:

    """性能报告"""
    analysis_period: Tuple[datetime, datetime]
    summary: Dict[str, Any]
    bottlenecks: List[BottleneckAnalysis]
    trends: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime


class PerformanceAnalyzer:

    """实时性能监控和分析工具"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 配置参数
        self.collection_interval = self.config.get('collection_interval', 1.0)  # 采集间隔(秒)
        self.history_size = self.config.get('history_size', 3600)  # 历史数据大小(秒)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 2.0)  # 异常阈值(标准差倍数)

        # 数据存储
        self.performance_history = defaultdict(lambda: deque(maxlen=self.history_size))
        self.baseline_stats = {}  # 基线统计
        self.anomaly_history = deque(maxlen=1000)

        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.baseline_calculated = False

        # 回调函数
        self.metric_callbacks: List[Callable[[PerformanceData], None]] = []
        self.anomaly_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.bottleneck_callbacks: List[Callable[[BottleneckAnalysis], None]] = []

        # 初始化深度学习预测器
        self.dl_predictor = get_deep_learning_predictor()

        # 初始化云原生优化器
        self.cloud_optimizer = get_cloud_native_optimizer()

        # 深度学习模型状态
        self.ml_models_trained = {}  # 记录已训练的模型
        self.prediction_enabled = self.config.get('prediction_enabled', True)
        self.anomaly_detection_enabled = self.config.get('anomaly_detection_enabled', True)

        # 增强监控功能
        self.service_monitoring_enabled = self.config.get('service_monitoring_enabled', True)
        self.auto_optimization_enabled = self.config.get('auto_optimization_enabled', False)
        self.real_time_alerts_enabled = self.config.get('real_time_alerts_enabled', True)

        # 服务监控数据
        self.service_health_scores = {}
        self.service_performance_history = defaultdict(lambda: deque(maxlen=100))
        self.optimization_recommendations = deque(maxlen=50)

        # 初始化基线数据
        self._initialize_baseline()

        logger.info("性能分析器初始化完成，深度学习功能已启用")

    def _initialize_baseline(self):
        """初始化基线数据"""
        logger.info("正在初始化性能基线数据...")

        # 收集初始数据建立基线
        baseline_samples = []
        for _ in range(60):  # 收集60个样本
            sample = self._collect_system_metrics()
            baseline_samples.append(sample)
            time.sleep(0.1)

        # 计算基线统计
        if baseline_samples:
            self.baseline_stats = self._calculate_baseline_stats(baseline_samples)
            self.baseline_calculated = True
            logger.info("性能基线数据初始化完成")

    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统性能指标"""
        metrics = {}

        try:
            # CPU指标
            metrics['cpu_usage'] = psutil.cpu_percent(interval=None)
            metrics['cpu_count'] = psutil.cpu_count()
            metrics['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else 0

            # 内存指标
            memory = psutil.virtual_memory()
            metrics['memory_usage'] = memory.percent
            metrics['memory_used'] = memory.used / (1024 ** 3)  # GB
            metrics['memory_available'] = memory.available / (1024 ** 3)  # GB

            # 磁盘指标
            disk = psutil.disk_usage('/')
            metrics['disk_usage'] = disk.percent
            metrics['disk_free'] = disk.free / (1024 ** 3)  # GB

            # 网络指标
            network = psutil.net_io_counters()
            metrics['network_bytes_sent'] = network.bytes_sent / (1024 ** 2)  # MB
            metrics['network_bytes_recv'] = network.bytes_recv / (1024 ** 2)  # MB

            # 系统负载
            load = psutil.getloadavg()
            metrics['system_load_1'] = load[0]
            metrics['system_load_5'] = load[1]
            metrics['system_load_15'] = load[2]

            # 进程信息
            metrics['process_count'] = len(psutil.pids())

        except Exception as e:
            logger.warning(f"收集系统指标失败: {e}")

        return metrics

    def _calculate_baseline_stats(self, samples: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """计算基线统计"""
        baseline_stats = {}

        if not samples:
            return baseline_stats

        # 转换为numpy数组便于计算
        sample_arrays = {}
        for metric in samples[0].keys():
            values = [sample.get(metric, 0) for sample in samples if metric in sample]
            if values:
                sample_arrays[metric] = np.array(values)

        # 计算统计指标
        for metric, values in sample_arrays.items():
            baseline_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'percentile_25': float(np.percentile(values, 25)),
                'percentile_75': float(np.percentile(values, 75)),
                'percentile_95': float(np.percentile(values, 95))
            }

        return baseline_stats

    def start_monitoring(self):
        """启动性能监控"""
        if self.is_monitoring:
            logger.warning("性能监控已在运行")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("性能监控已启动")

    def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        logger.info("性能监控已停止")

    def _monitoring_loop(self):
        """监控主循环"""
        while self.is_monitoring:
            try:
                # 收集性能数据
                metrics = self._collect_system_metrics()
                timestamp = datetime.now()

                # 创建性能数据对象
                performance_data = PerformanceData(
                    timestamp=timestamp,
                    metrics=metrics,
                    context={'source': 'system_monitor'}
                )

                # 存储历史数据
                for metric_name, value in metrics.items():
                    self.performance_history[metric_name].append({
                        'timestamp': timestamp,
                        'value': value
                    })

                # 触发回调
                self._trigger_metric_callbacks(performance_data)

                # 实时异常检测
                self._detect_realtime_anomalies(metrics, timestamp)

                # 定期瓶颈分析
                if int(time.time()) % 30 == 0:  # 每30秒分析一次
                    self._analyze_bottlenecks()

            except Exception as e:
                logger.error(f"性能监控循环错误: {e}")

            time.sleep(self.collection_interval)

    def _trigger_metric_callbacks(self, data: PerformanceData):
        """触发指标回调"""
        for callback in self.metric_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"指标回调执行失败: {e}")

    def _detect_realtime_anomalies(self, metrics: Dict[str, float], timestamp: datetime):
        """实时异常检测"""
        if not self.baseline_calculated:
            return

        anomalies = []

        for metric_name, value in metrics.items():
            baseline = self.baseline_stats.get(metric_name)
            if not baseline:
                continue

            # 计算偏差
            deviation = abs(value - baseline['mean'])
            threshold = self.anomaly_threshold * baseline['std']

            if deviation > threshold:
                anomaly = {
                    'metric': metric_name,
                    'value': value,
                    'expected': baseline['mean'],
                    'deviation': deviation,
                    'threshold': threshold,
                    'severity': self._calculate_anomaly_severity(deviation, threshold),
                    'timestamp': timestamp,
                    'description': self._generate_anomaly_description(metric_name, value, baseline)
                }

                anomalies.append(anomaly)

                # 存储异常历史
                self.anomaly_history.append(anomaly)

                # 触发异常回调
                self._trigger_anomaly_callbacks(anomaly)

        return anomalies

    def _calculate_anomaly_severity(self, deviation: float, threshold: float) -> str:
        """计算异常严重程度"""
        ratio = deviation / threshold

        if ratio >= 3.0:
            return "critical"
        elif ratio >= 2.0:
            return "high"
        elif ratio >= 1.5:
            return "medium"
        else:
            return "low"

    def _generate_anomaly_description(self, metric_name: str, value: float,


                                      baseline: Dict[str, float]) -> str:
        """生成异常描述"""
        mean_val = baseline['mean']
        deviation = abs(value - mean_val)

        descriptions = {
            'cpu_usage': f"CPU使用率异常: {value:.1f}%, 超出基线 {mean_val:.1f}% 的 {deviation:.1f}%",
            'memory_usage': f"内存使用率异常: {value:.1f}%, 超出基线 {mean_val:.1f}% 的 {deviation:.1f}%",
            'disk_usage': f"磁盘使用率异常: {value:.1f}%, 超出基线 {mean_val:.1f}% 的 {deviation:.1f}%",
            'system_load_1': f"系统负载异常: {value:.2f}, 超出基线 {mean_val:.2f} 的 {deviation:.2f}",
        }

        return descriptions.get(metric_name, f"{metric_name} 指标异常: {value}, 基线: {mean_val}")

    def _trigger_anomaly_callbacks(self, anomaly: Dict[str, Any]):
        """触发异常回调"""
        for callback in self.anomaly_callbacks:
            try:
                callback(anomaly)
            except Exception as e:
                logger.error(f"异常回调执行失败: {e}")

    def _analyze_bottlenecks(self):
        """分析系统瓶颈"""
        if not self.baseline_calculated:
            return

        bottlenecks = []

        # CPU瓶颈分析
        cpu_analysis = self._analyze_cpu_bottleneck()
        if cpu_analysis:
            bottlenecks.append(cpu_analysis)

        # 内存瓶颈分析
        memory_analysis = self._analyze_memory_bottleneck()
        if memory_analysis:
            bottlenecks.append(memory_analysis)

        # 磁盘瓶颈分析
        disk_analysis = self._analyze_disk_bottleneck()
        if disk_analysis:
            bottlenecks.append(disk_analysis)

        # 网络瓶颈分析
        network_analysis = self._analyze_network_bottleneck()
        if network_analysis:
            bottlenecks.append(network_analysis)

        # 触发瓶颈回调
        for bottleneck in bottlenecks:
            self._trigger_bottleneck_callbacks(bottleneck)

    def _analyze_cpu_bottleneck(self) -> Optional[BottleneckAnalysis]:
        """分析CPU瓶颈"""
        cpu_history = list(self.performance_history['cpu_usage'])
        if len(cpu_history) < 10:
            return None

        recent_cpu = [point['value'] for point in cpu_history[-10:]]

        if np.mean(recent_cpu) > 80:
            return BottleneckAnalysis(
                component="CPU",
                severity="high",
                description="CPU使用率持续高于80%，可能存在计算密集型任务",
                recommendations=[
                    "检查CPU密集型进程",
                    "考虑增加CPU核心或优化算法",
                    "实施CPU使用率限制",
                    "启用CPU亲和性设置"
                ],
                impact_score=0.8,
                confidence=0.9
            )

        return None

    def _analyze_memory_bottleneck(self) -> Optional[BottleneckAnalysis]:
        """分析内存瓶颈"""
        memory_history = list(self.performance_history['memory_usage'])
        if len(memory_history) < 10:
            return None

        recent_memory = [point['value'] for point in memory_history[-10:]]

        if np.mean(recent_memory) > 85:
            return BottleneckAnalysis(
                component="Memory",
                severity="high",
                description="内存使用率持续高于85%，可能存在内存泄漏",
                recommendations=[
                    "检查内存使用情况和泄漏",
                    "增加系统内存容量",
                    "优化内存分配策略",
                    "启用内存压缩或交换"
                ],
                impact_score=0.9,
                confidence=0.85
            )

        return None

    def _analyze_disk_bottleneck(self) -> Optional[BottleneckAnalysis]:
        """分析磁盘瓶颈"""
        disk_history = list(self.performance_history['disk_usage'])
        if len(disk_history) < 10:
            return None

        recent_disk = [point['value'] for point in disk_history[-10:]]

        if np.mean(recent_disk) > 90:
            return BottleneckAnalysis(
                component="Disk",
                severity="medium",
                description="磁盘使用率持续高于90%，存储空间不足",
                recommendations=[
                    "清理不必要的文件和日志",
                    "增加磁盘存储容量",
                    "实施磁盘使用监控和告警",
                    "考虑数据归档和压缩策略"
                ],
                impact_score=0.6,
                confidence=0.8
            )

        return None

    def _analyze_network_bottleneck(self) -> Optional[BottleneckAnalysis]:
        """分析网络瓶颈"""
        # 网络分析需要更复杂的指标，这里简化处理
        return None

    def _trigger_bottleneck_callbacks(self, bottleneck: BottleneckAnalysis):
        """触发瓶颈回调"""
        for callback in self.bottleneck_callbacks:
            try:
                callback(bottleneck)
            except Exception as e:
                logger.error(f"瓶颈回调执行失败: {e}")

    def add_metric_callback(self, callback: Callable[[PerformanceData], None]):
        """添加指标回调"""
        self.metric_callbacks.append(callback)

    def add_anomaly_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加异常回调"""
        self.anomaly_callbacks.append(callback)

    def add_bottleneck_callback(self, callback: Callable[[BottleneckAnalysis], None]):
        """添加瓶颈回调"""
        self.bottleneck_callbacks.append(callback)

    def get_performance_report(self, start_time: Optional[datetime] = None,


                               end_time: Optional[datetime] = None) -> PerformanceReport:
        """生成性能报告"""
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()

        # 收集分析期间的数据
        report_data = self._collect_report_data(start_time, end_time)

        # 生成总结
        summary = self._generate_report_summary(report_data)

        # 识别瓶颈
        bottlenecks = self._identify_bottlenecks_in_period(report_data, start_time, end_time)

        # 分析趋势
        try:
            trends = self._analyze_performance_trends(report_data)
        except TypeError:
            # 如果方法签名不匹配，使用简单的趋势分析
            trends = self._analyze_simple_trends(report_data)

        # 生成建议
        recommendations = self._generate_performance_recommendations(bottlenecks, trends)

        return PerformanceReport(
            analysis_period=(start_time, end_time),
            summary=summary,
            bottlenecks=bottlenecks,
            trends=trends,
            recommendations=recommendations,
            generated_at=datetime.now()
        )

    def _collect_report_data(self, start_time: datetime, end_time: datetime) -> Dict[str, List]:
        """收集报告数据"""
        report_data = {}

        for metric_name, history in self.performance_history.items():
            filtered_data = [
                point for point in history
                if start_time <= point['timestamp'] <= end_time
            ]
            report_data[metric_name] = filtered_data

        return report_data

    def _generate_report_summary(self, report_data: Dict[str, List]) -> Dict[str, Any]:
        """生成报告总结"""
        summary = {
            'total_metrics': len(report_data),
            'data_points': sum(len(data) for data in report_data.values()),
            'analysis_duration': 0,  # 将在后续计算
            'peak_values': {},
            'average_values': {},
            'anomaly_count': len([a for a in self.anomaly_history
                                  if a['timestamp'] >= datetime.now() - timedelta(hours=1)])
        }

        # 计算峰值和平均值
        for metric_name, data in report_data.items():
            if data:
                values = [point['value'] for point in data]
                summary['peak_values'][metric_name] = max(values)
                summary['average_values'][metric_name] = np.mean(values)

        return summary

    def _identify_bottlenecks_in_period(self, report_data: Dict[str, List],


                                        start_time: datetime, end_time: datetime) -> List[BottleneckAnalysis]:
        """识别分析期间的瓶颈"""
        bottlenecks = []

        # 这里可以实现更复杂的瓶颈识别逻辑
        # 目前返回最近检测到的瓶颈
        recent_bottlenecks = []
        # 在实际实现中，这里应该从历史瓶颈数据中筛选

        return recent_bottlenecks

    def _analyze_performance_trends(self, report_data: Dict[str, List]) -> Dict[str, Any]:
        """分析性能趋势"""
        trends = {}

        for metric_name, data in report_data.items():
            if len(data) < 10:
                continue

            values = [point['value'] for point in data]

            # 计算趋势斜率
            if len(values) > 1:
                slope = np.polyfit(range(len(values)), values, 1)[0]
                trends[metric_name] = {
                    'slope': slope,
                    'trend': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable',
                    'volatility': np.std(values),
                    'description': self._generate_trend_description(metric_name, slope)
                }

        return trends

    def _analyze_simple_trends(self, report_data: Dict[str, List]) -> Dict[str, Any]:
        """简单的趋势分析（作为回退方案）"""
        trends = {}

        for metric_name, data in report_data.items():
            if len(data) < 2:
                trends[metric_name] = "insufficient_data"
                continue

            # 计算简单趋势
            values = [point.get('value', 0) for point in data]
            if len(values) >= 2:
                first_half = sum(values[:len(values)//2]) / len(values[:len(values)//2])
                second_half = sum(values[len(values)//2:]) / len(values[len(values)//2:])

                if second_half > first_half * 1.05:
                    trends[metric_name] = "increasing"
                elif second_half < first_half * 0.95:
                    trends[metric_name] = "decreasing"
                else:
                    trends[metric_name] = "stable"
            else:
                trends[metric_name] = "insufficient_data"

        return trends

    def _generate_trend_description(self, metric_name: str, slope: float) -> str:
        """生成趋势描述"""
        if abs(slope) < 0.1:
            return f"{metric_name} 保持稳定"
        elif slope > 0:
            return f"{metric_name} 呈上升趋势 (斜率: {slope:.3f})"
        else:
            return f"{metric_name} 呈下降趋势 (斜率: {slope:.3f})"

    def _generate_performance_recommendations(self, bottlenecks: List[BottleneckAnalysis],


                                              trends: Dict[str, Any]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []

        # 基于瓶颈的建议
        for bottleneck in bottlenecks:
            recommendations.extend(bottleneck.recommendations)

        # 基于趋势的建议
        for metric_name, trend_info in trends.items():
            if trend_info['trend'] == 'increasing' and trend_info['slope'] > 0.5:
                recommendations.append(f"关注 {metric_name} 的持续上升趋势，及时采取优化措施")

        # 通用建议
        if not recommendations:
            recommendations.extend([
                "保持当前性能监控策略",
                "定期检查系统资源使用情况",
                "关注关键性能指标的变化趋势"
            ])

        return list(set(recommendations))  # 去重

    def export_performance_data(self, filename: str, format: str = 'json'):
        """导出性能数据"""
        export_data = {
            'export_time': datetime.now().isoformat(),
            'configuration': self.config,
            'baseline_stats': self.baseline_stats,
            'performance_history': dict(self.performance_history),
            'anomaly_history': list(self.anomaly_history),
            'system_info': self._get_system_info()
        }

        if format == 'json':
            with open(filename, 'w', encoding='utf - 8') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            logger.error(f"不支持的导出格式: {format}")

        logger.info(f"性能数据已导出到: {filename}")

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import platform
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total / (1024 ** 3),  # GB
            'disk_total': psutil.disk_usage('/').total / (1024 ** 3),  # GB
            'platform': platform.platform(),
            'python_version': platform.python_version()
        }

    def get_current_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'is_monitoring': self.is_monitoring,
            'baseline_calculated': self.baseline_calculated,
            'total_metrics': len(self.performance_history),
            'total_data_points': sum(len(data) for data in self.performance_history.values()),
            'anomaly_count': len(self.anomaly_history),
            'ml_models_trained': len(self.ml_models_trained),
            'prediction_enabled': self.prediction_enabled,
            'anomaly_detection_enabled': self.anomaly_detection_enabled,
            'last_update': datetime.now().isoformat(),
            'system_metrics': self._collect_system_metrics()
        }

    # ================ 深度学习增强功能 ================

    def enable_ml_prediction(self, enabled: bool = True):
        """启用 / 禁用ML预测功能"""
        self.prediction_enabled = enabled
        logger.info(f"ML预测功能已{'启用' if enabled else '禁用'}")

    def enable_ml_anomaly_detection(self, enabled: bool = True):
        """启用 / 禁用ML异常检测功能"""
        self.anomaly_detection_enabled = enabled
        logger.info(f"ML异常检测功能已{'启用' if enabled else '禁用'}")

    def train_ml_model_for_metric(self, metric_name: str, **kwargs) -> Dict[str, Any]:
        """
        为指定指标训练ML模型

        Args:
            metric_name: 指标名称
            **kwargs: 训练参数

        Returns:
            训练结果
        """
        if not self.prediction_enabled:
            return {'status': 'error', 'message': 'ML预测功能未启用'}

        try:
            # 检查是否有足够的历史数据
            if metric_name not in self.performance_history:
                return {'status': 'error', 'message': f'没有找到指标 {metric_name} 的历史数据'}

            history_data = list(self.performance_history[metric_name])
            if len(history_data) < 100:  # 需要至少100个数据点
                return {'status': 'error', 'message': f'历史数据不足，需要至少100个数据点，当前有{len(history_data)}个'}

            # 转换为DataFrame
            df_data = []
            for data_point in history_data:
                df_data.append({
                    'timestamp': data_point.timestamp,
                    'value': getattr(data_point, metric_name.split('_')[1] if '_' in metric_name else 'usage', 0)
                })

            df = pd.DataFrame(df_data)

            # 训练LSTM模型
            result = self.dl_predictor.train_lstm_predictor(
                metric_name,
                df,
                **kwargs
            )

            if result['status'] == 'success':
                self.ml_models_trained[metric_name] = {
                    'model_type': 'lstm',
                    'trained_at': datetime.now(),
                    'training_result': result
                }
                logger.info(f"为指标 {metric_name} 训练LSTM模型成功")

            return result

        except Exception as e:
            logger.error(f"训练ML模型失败: {e}")
            return {'status': 'error', 'message': str(e)}

    def train_autoencoder_for_anomaly_detection(self, metric_name: str, **kwargs) -> Dict[str, Any]:
        """
        为指定指标训练Autoencoder异常检测模型

        Args:
            metric_name: 指标名称
            **kwargs: 训练参数

        Returns:
            训练结果
        """
        if not self.anomaly_detection_enabled:
            return {'status': 'error', 'message': 'ML异常检测功能未启用'}

        try:
            # 检查是否有足够的历史数据
            if metric_name not in self.performance_history:
                return {'status': 'error', 'message': f'没有找到指标 {metric_name} 的历史数据'}

            history_data = list(self.performance_history[metric_name])
            if len(history_data) < 50:  # 需要至少50个数据点
                return {'status': 'error', 'message': f'历史数据不足，需要至少50个数据点，当前有{len(history_data)}个'}

            # 转换为DataFrame
            df_data = []
            for data_point in history_data[-500:]:  # 使用最近500个数据点
                df_data.append({
                    'timestamp': data_point.timestamp,
                    'value': getattr(data_point, metric_name.split('_')[1] if '_' in metric_name else 'usage', 0)
                })

            df = pd.DataFrame(df_data)

            # 训练Autoencoder模型
            result = self.dl_predictor.train_autoencoder_anomaly_detector(
                metric_name,
                df,
                **kwargs
            )

            if result['status'] == 'success':
                self.ml_models_trained[f"{metric_name}_anomaly"] = {
                    'model_type': 'autoencoder',
                    'trained_at': datetime.now(),
                    'training_result': result
                }
                logger.info(f"为指标 {metric_name} 训练Autoencoder异常检测模型成功")

            return result

        except Exception as e:
            logger.error(f"训练Autoencoder模型失败: {e}")
            return {'status': 'error', 'message': str(e)}

    def predict_metric_with_ml(self, metric_name: str, steps: int = 1) -> Dict[str, Any]:
        """
        使用ML模型预测指标值

        Args:
            metric_name: 指标名称
            steps: 预测步数

        Returns:
            预测结果
        """
        if not self.prediction_enabled:
            return {'status': 'error', 'message': 'ML预测功能未启用'}

        try:
            if metric_name not in self.ml_models_trained:
                return {'status': 'error', 'message': f'指标 {metric_name} 没有训练好的ML模型'}

            # 获取最近的数据用于预测
            if metric_name not in self.performance_history:
                return {'status': 'error', 'message': f'没有找到指标 {metric_name} 的历史数据'}

            history_data = list(self.performance_history[metric_name])[-100:]  # 使用最近100个数据点

            # 转换为DataFrame
            df_data = []
            for data_point in history_data:
                df_data.append({
                    'timestamp': data_point.timestamp,
                    'value': getattr(data_point, metric_name.split('_')[1] if '_' in metric_name else 'usage', 0)
                })

            df = pd.DataFrame(df_data)

            # 使用LSTM进行预测
            result = self.dl_predictor.predict_with_lstm(
                metric_name,
                df,
                steps=steps
            )

            if result['status'] == 'success':
                logger.info(f"使用ML模型成功预测指标 {metric_name}")

            return result

        except Exception as e:
            logger.error(f"ML预测失败: {e}")
            return {'status': 'error', 'message': str(e)}

    def detect_anomalies_with_ml(self, metric_name: str) -> Dict[str, Any]:
        """
        使用ML模型检测异常

        Args:
            metric_name: 指标名称

        Returns:
            异常检测结果
        """
        if not self.anomaly_detection_enabled:
            return {'status': 'error', 'message': 'ML异常检测功能未启用'}

        try:
            anomaly_model_key = f"{metric_name}_anomaly"
            if anomaly_model_key not in self.ml_models_trained:
                return {'status': 'error', 'message': f'指标 {metric_name} 没有训练好的异常检测模型'}

            # 获取最近的数据用于异常检测
            if metric_name not in self.performance_history:
                return {'status': 'error', 'message': f'没有找到指标 {metric_name} 的历史数据'}

            history_data = list(self.performance_history[metric_name])[-200:]  # 使用最近200个数据点

            # 转换为DataFrame
            df_data = []
            for data_point in history_data:
                df_data.append({
                    'timestamp': data_point.timestamp,
                    'value': getattr(data_point, metric_name.split('_')[1] if '_' in metric_name else 'usage', 0)
                })

            df = pd.DataFrame(df_data)

            # 使用Autoencoder进行异常检测
            result = self.dl_predictor.detect_anomalies_with_autoencoder(
                metric_name,
                df
            )

            if result['status'] == 'success':
                logger.info(f"使用ML模型成功检测指标 {metric_name} 的异常")

                # 如果检测到异常，添加到异常历史记录
                if result['anomalies_detected'] > 0:
                    anomaly_record = {
                        'timestamp': datetime.now(),
                        'metric': metric_name,
                        'anomalies_detected': result['anomalies_detected'],
                        'anomaly_percentage': result['anomaly_percentage'],
                        'detection_method': 'ml_autoencoder',
                        'details': result
                    }
                    self.anomaly_history.append(anomaly_record)

                    # 调用异常回调
                    for callback in self.anomaly_callbacks:
                        try:
                            callback(anomaly_record)
                        except Exception as e:
                            logger.error(f"异常回调执行失败: {e}")

            return result

        except Exception as e:
            logger.error(f"ML异常检测失败: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_ml_model_status(self) -> Dict[str, Any]:
        """获取ML模型状态"""
        return {
            'prediction_enabled': self.prediction_enabled,
            'anomaly_detection_enabled': self.anomaly_detection_enabled,
            'trained_models': {
                name: {
                    'type': info['model_type'],
                    'trained_at': info['trained_at'].isoformat(),
                    'status': 'ready'
                }
                for name, info in self.ml_models_trained.items()
            },
            'predictor_info': self.dl_predictor.get_model_info()
        }

    def train_all_ml_models(self, min_data_points: int = 100) -> Dict[str, Any]:
        """
        为所有有足够数据的指标训练ML模型

        Args:
            min_data_points: 最少数据点要求

        Returns:
            训练结果汇总
        """
        results = {
            'total_metrics': len(self.performance_history),
            'trained_models': 0,
            'failed_models': 0,
            'details': []
        }

        for metric_name in self.performance_history.keys():
            data_count = len(self.performance_history[metric_name])

            if data_count >= min_data_points:
                logger.info(f"为指标 {metric_name} 训练LSTM模型...")

                # 训练LSTM预测模型
                lstm_result = self.train_ml_model_for_metric(
                    metric_name,
                    epochs=20,  # 减少训练轮数以加快速度
                    seq_length=24
                )

                # 训练Autoencoder异常检测模型
                ae_result = self.train_autoencoder_for_anomaly_detection(
                    metric_name,
                    epochs=15  # 减少训练轮数以加快速度
                )

                detail = {
                    'metric': metric_name,
                    'data_points': data_count,
                    'lstm_training': lstm_result,
                    'autoencoder_training': ae_result
                }

                results['details'].append(detail)

                if lstm_result['status'] == 'success':
                    results['trained_models'] += 1
                else:
                    results['failed_models'] += 1

                if ae_result['status'] == 'success':
                    results['trained_models'] += 1
                else:
                    results['failed_models'] += 1
            else:
                logger.info(f"指标 {metric_name} 数据不足({data_count})，跳过训练")

        logger.info(f"ML模型训练完成，共训练 {results['trained_models']} 个模型，失败 {results['failed_models']} 个")
        return results

    def get_ml_predictions_for_all_metrics(self, steps: int = 1) -> Dict[str, Any]:
        """
        获取所有指标的ML预测结果

        Args:
            steps: 预测步数

        Returns:
            预测结果汇总
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'errors': []
        }

        for metric_name in self.ml_models_trained.keys():
            if not metric_name.endswith('_anomaly'):  # 只对预测模型进行预测
                prediction = self.predict_metric_with_ml(metric_name, steps)
                if prediction['status'] == 'success':
                    results['predictions'][metric_name] = prediction
                else:
                    results['errors'].append({
                        'metric': metric_name,
                        'error': prediction.get('message', 'Unknown error')
                    })

        return results

    # ============================================================================
    # 增强监控功能 (新增)
    # ============================================================================

    async def start_enhanced_monitoring(self):
        """
        启动增强监控功能

        包括服务健康监控、实时优化建议、智能告警等
        """
        logger.info("启动增强监控功能...")

        if self.service_monitoring_enabled:
            asyncio.create_task(self._monitor_services_health())
            logger.info("服务健康监控已启动")

        if self.auto_optimization_enabled:
            asyncio.create_task(self._auto_optimization_worker())
            logger.info("自动优化功能已启动")

        if self.real_time_alerts_enabled:
            asyncio.create_task(self._real_time_alert_processor())
            logger.info("实时告警处理已启动")

        logger.info("增强监控功能启动完成")

    async def _monitor_services_health(self):
        """
        监控服务健康状态

        定期检查各个服务的健康状况并记录历史数据
        """
        logger.info("开始服务健康监控...")

        # 获取需要监控的服务列表
        services_to_monitor = self._get_services_to_monitor()

        while True:
            try:
                for service_name in services_to_monitor:
                    health_score = await self._check_service_health(service_name)

                    # 记录健康评分
                    self.service_health_scores[service_name] = health_score

                    # 收集性能指标
                    performance_metrics = await self._collect_service_metrics(service_name)
                    self.service_performance_history[service_name].append(performance_metrics)

                    # 检查健康异常
                    if health_score < 0.8:  # 健康评分低于80%
                        await self._handle_service_health_alert(service_name, health_score, performance_metrics)

                # 等待30秒后进行下一轮检查
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"服务健康监控异常: {e}")
                await asyncio.sleep(30)  # 出错后等待30秒重试

    def _get_services_to_monitor(self) -> List[str]:
        """
        获取需要监控的服务列表

        Returns:
            服务名称列表
        """
        # 这里可以从配置或服务发现中获取服务列表
        # 暂时返回一个默认列表
        return [
            'trading - engine',
            'market - data - service',
            'risk - management - service',
            'alert - analysis - service',
            'performance - monitor - service'
        ]

    async def _check_service_health(self, service_name: str) -> float:
        """
        检查服务健康状态

        Args:
            service_name: 服务名称

        Returns:
            健康评分 (0 - 1)
        """
        try:
            # 使用云原生优化器检查服务健康
            health_status = await self.cloud_optimizer.health_monitor.perform_health_check(service_name)

            # 计算健康评分
            response_time = health_status.get('response_time', 1.0)
            error_rate = health_status.get('error_rate', 0.0)
            cpu_usage = health_status.get('cpu_usage', 50.0)

            # 基于多个指标计算综合评分
            response_score = max(0, 1 - (response_time / 5.0))  # 响应时间评分
            error_score = max(0, 1 - error_rate * 2)  # 错误率评分
            cpu_score = max(0, 1 - (cpu_usage / 100))  # CPU使用率评分

            # 加权平均
            health_score = (response_score * 0.4 + error_score * 0.4 + cpu_score * 0.2)

            return health_score

        except Exception as e:
            logger.error(f"检查服务健康状态失败 {service_name}: {e}")
            return 0.0

    async def _collect_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """
        收集服务性能指标

        Args:
            service_name: 服务名称

        Returns:
            性能指标字典
        """
        try:
            # 使用云原生优化器收集指标
            metrics = await self.cloud_optimizer.health_monitor.collect_metrics(service_name)

            # 添加时间戳
            metrics['timestamp'] = datetime.now()

            return metrics

        except Exception as e:
            logger.error(f"收集服务指标失败 {service_name}: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }

    async def _handle_service_health_alert(self, service_name: str, health_score: float, metrics: Dict[str, Any]):
        """
        处理服务健康告警

        Args:
            service_name: 服务名称
            health_score: 健康评分
            metrics: 性能指标
        """
        alert_message = f"服务健康告警: {service_name}, 健康评分: {health_score:.2f}"

        # 确定告警级别
        if health_score < 0.5:
            severity = "critical"
        elif health_score < 0.7:
            severity = "high"
        else:
            severity = "medium"

        # 生成告警
        alert = {
            'alert_id': f"health_{service_name}_{int(time.time())}",
            'service_name': service_name,
            'alert_type': 'service_health',
            'severity': severity,
            'message': alert_message,
            'health_score': health_score,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'recommendations': await self._generate_health_recommendations(service_name, health_score, metrics)
        }

        # 记录告警
        logger.warning(alert_message)

        # 调用告警回调
        for callback in self.anomaly_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调异常: {e}")

    async def _generate_health_recommendations(self, service_name: str, health_score: float, metrics: Dict[str, Any]) -> List[str]:
        """
        生成健康建议

        Args:
            service_name: 服务名称
            health_score: 健康评分
            metrics: 性能指标

        Returns:
            建议列表
        """
        recommendations = []

        if health_score < 0.5:
            recommendations.append(f"紧急：服务 {service_name} 健康状况严重，建议立即检查和重启")
        elif health_score < 0.7:
            recommendations.append(f"警告：服务 {service_name} 健康状况不佳，建议检查配置和资源")

        # 基于具体指标生成建议
        if metrics.get('response_time', 0) > 3.0:
            recommendations.append(f"服务 {service_name} 响应时间过长，建议优化查询或增加缓存")

        if metrics.get('error_rate', 0) > 0.05:
            recommendations.append(f"服务 {service_name} 错误率偏高，建议检查错误处理和依赖服务")

        if metrics.get('cpu_usage', 0) > 80:
            recommendations.append(f"服务 {service_name} CPU使用率过高，建议增加CPU资源或优化代码")

        return recommendations

    def get_enhanced_monitoring_status(self) -> Dict[str, Any]:
        """
        获取增强监控状态

        Returns:
            监控状态信息
        """
        return {
            'service_monitoring_enabled': self.service_monitoring_enabled,
            'auto_optimization_enabled': self.auto_optimization_enabled,
            'real_time_alerts_enabled': self.real_time_alerts_enabled,
            'service_health_scores': dict(self.service_health_scores),
            'active_optimization_recommendations': len(self.optimization_recommendations)
        }

    async def get_real_time_insights(self) -> Dict[str, Any]:
        """
        获取实时洞察

        Returns:
            实时洞察信息
        """
        insights = {
            'timestamp': datetime.now(),
            'system_health': await self._analyze_system_health(),
            'performance_trends': await self._analyze_performance_trends(),
            'anomaly_summary': await self._summarize_recent_anomalies()
        }

        return insights

    async def _analyze_system_health(self) -> Dict[str, Any]:
        """分析系统整体健康状况"""
        current_metrics = self._collect_system_metrics()

        health_score = 1.0

        # 基于关键指标计算健康评分
        for metric_name, value in current_metrics.items():
            if metric_name in self.baseline_stats:
                baseline = self.baseline_stats[metric_name]
                deviation = abs(value - baseline['mean']) / \
                    baseline['std'] if baseline['std'] > 0 else 0

                if deviation > 2:  # 超过2倍标准差
                    health_score -= 0.1

        health_score = max(0.0, min(1.0, health_score))

        return {
            'overall_score': health_score,
            'status': 'healthy' if health_score > 0.8 else 'warning' if health_score > 0.6 else 'critical',
            'metrics': current_metrics
        }

    async def _summarize_recent_anomalies(self) -> Dict[str, Any]:
        """总结最近的异常情况"""
        recent_anomalies = list(self.anomaly_history)[-10:] if self.anomaly_history else []

        summary = {
            'total_recent_anomalies': len(recent_anomalies),
            'anomaly_types': {},
            'most_affected_metrics': {},
            'trend': 'stable'
        }

        if recent_anomalies:
            # 统计异常类型
            for anomaly in recent_anomalies:
                anomaly_type = anomaly.get('type', 'unknown')
                summary['anomaly_types'][anomaly_type] = summary['anomaly_types'].get(
                    anomaly_type, 0) + 1

                # 统计受影响的指标
                affected_metrics = anomaly.get('affected_metrics', [])
                for metric in affected_metrics:
                    summary['most_affected_metrics'][metric] = summary['most_affected_metrics'].get(
                        metric, 0) + 1

            # 判断趋势
            recent_count = len([a for a in recent_anomalies if (
                datetime.now() - a.get('timestamp', datetime.now())).seconds < 3600])
            if recent_count > len(recent_anomalies) * 0.7:
                summary['trend'] = 'increasing'
            elif recent_count < len(recent_anomalies) * 0.3:
                summary['trend'] = 'decreasing'

        return summary

    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """
        分析性能趋势

        Returns:
            趋势分析结果
        """
        trends = {}

        for metric_name, history in self.performance_history.items():
            if len(history) >= 10:  # 至少需要10个数据点
                values = [h.get(metric_name, 0) for h in history if isinstance(h, dict)]

                if values:
                    # 计算趋势
                    recent_avg = np.mean(values[-5:])  # 最近5个点的平均值
                    earlier_avg = np.mean(values[:5])  # 前面5个点的平均值

                    if earlier_avg > 0:
                        trend_percentage = ((recent_avg - earlier_avg) / earlier_avg) * 100
                        trend_direction = "上升" if trend_percentage > 5 else "下降" if trend_percentage < -5 else "稳定"

                        trends[metric_name] = {
                            'direction': trend_direction,
                            'change_percentage': trend_percentage,
                            'recent_avg': recent_avg,
                            'earlier_avg': earlier_avg
                        }

        return trends


# 工厂函数
def get_performance_analyzer() -> PerformanceAnalyzer:
    """
    获取性能分析器实例

    Returns:
        PerformanceAnalyzer: 性能分析器实例
    """
    return PerformanceAnalyzer()
