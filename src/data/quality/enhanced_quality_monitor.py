"""
增强版数据质量监控器
提供高级质量检查、趋势分析和异常检测功能
"""

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

import time
import threading
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque

from src.infrastructure.logging import get_infrastructure_logger

logger = get_infrastructure_logger('enhanced_quality_monitor')


@dataclass
class QualityMetrics:

    """质量指标数据类"""
    timestamp: str
    data_type: str
    completeness: float
    accuracy: float
    consistency: float
    timeliness: float
    validity: float
    uniqueness: float
    overall_score: float
    details: Dict[str, Any]


@dataclass
class QualityTrend:

    """质量趋势数据类"""
    metric_name: str
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0 - 1
    change_rate: float
    prediction: float
    confidence: float


@dataclass
class AnomalyReport:

    """异常报告数据类"""
    anomaly_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_data: str
    detection_time: str
    suggested_action: str


class EnhancedQualityMonitor:

    """
    增强版数据质量监控器

    功能：
    - 多维度质量检查
    - 质量趋势分析
    - 异常检测和告警
    - 质量预测
    - 自动质量修复建议
    """

    def __init__(self, enable_alerting: bool = True, enable_trend_analysis: bool = True):
        """
        初始化增强版质量监控器

        Args:
            enable_alerting: 是否启用告警
            enable_trend_analysis: 是否启用趋势分析
        """
        self.enable_alerting = enable_alerting
        self.enable_trend_analysis = enable_trend_analysis

        # 质量指标历史
        self._quality_history = deque(maxlen=1000)

        # 趋势分析器
        self._trend_analyzers = {
            'completeness': TrendAnalyzer(),
            'accuracy': TrendAnalyzer(),
            'consistency': TrendAnalyzer(),
            'timeliness': TrendAnalyzer(),
            'validity': TrendAnalyzer(),
            'uniqueness': TrendAnalyzer(),
            'overall_score': TrendAnalyzer()
        }

        # 异常检测器
        self._anomaly_detectors = {
            'statistical': StatisticalAnomalyDetector(),
            'temporal': TemporalAnomalyDetector(),
            'pattern': PatternAnomalyDetector()
        }

        # 质量规则引擎
        self._quality_rules = QualityRuleEngine()

        # 告警管理器
        if enable_alerting:
            self._alert_manager = QualityAlertManager()

        # 启动质量监控
        self._start_quality_monitoring()

    def check_data_quality(self, data: pd.DataFrame, data_type: str, identifier: str) -> QualityMetrics:
        """
        检查数据质量

        Args:
            data: 要检查的数据
            data_type: 数据类型
            identifier: 数据标识符

        Returns:
            质量指标
        """
        start_time = time.time()

        # 基础质量检查
        completeness = self._check_completeness(data)
        accuracy = self._check_accuracy(data, data_type)
        consistency = self._check_consistency(data, data_type)
        timeliness = self._check_timeliness(data)
        validity = self._check_validity(data, data_type)
        uniqueness = self._check_uniqueness(data)

        # 计算综合分数
        overall_score = self._calculate_overall_score({
            'completeness': completeness,
            'accuracy': accuracy,
            'consistency': consistency,
            'timeliness': timeliness,
            'validity': validity,
            'uniqueness': uniqueness
        })

        # 创建质量指标
        metrics = QualityMetrics(
            timestamp=datetime.now().isoformat(),
            data_type=data_type,
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            uniqueness=uniqueness,
            overall_score=overall_score,
            details={
                'identifier': identifier,
                'data_shape': data.shape,
                'check_time': time.time() - start_time
            }
        )

        # 记录质量历史
        self._quality_history.append(metrics)

        # 更新趋势分析
        if self.enable_trend_analysis:
            self._update_trend_analysis(metrics)

        # 异常检测
        anomalies = self._detect_anomalies(metrics)
        if anomalies and self.enable_alerting:
            self._alert_manager.send_alerts(anomalies)

        return metrics

    def _check_completeness(self, data: pd.DataFrame) -> float:
        """检查数据完整性"""
        if data.empty:
            return 0.0

        # 检查空值比例
        null_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        completeness = 1.0 - null_ratio

        # 检查必要字段是否存在
        required_fields = ['date', 'symbol']
        if 'date' in data.columns and 'symbol' in data.columns:
            completeness += 0.1

        return min(completeness, 1.0)

    def _check_accuracy(self, data: pd.DataFrame, data_type: str) -> float:
        """检查数据准确性"""
        if data.empty:
            return 0.0

        accuracy_score = 0.0

        # 价格数据准确性检查
        if 'close' in data.columns:
            # 检查价格是否为正数
            positive_prices = (data['close'] > 0).sum() / len(data)
            accuracy_score += positive_prices * 0.3

            # 检查价格变化是否合理
            if len(data) > 1:
                price_changes = data['close'].pct_change().abs()
                reasonable_changes = (price_changes < 0.5).sum() / len(price_changes)
                accuracy_score += reasonable_changes * 0.3

        # 成交量数据准确性检查
        if 'volume' in data.columns:
            # 检查成交量是否为正数
            positive_volumes = (data['volume'] >= 0).sum() / len(data)
            accuracy_score += positive_volumes * 0.2

        # 日期格式检查
        if 'date' in data.columns:
            try:
                pd.to_datetime(data['date'])
                accuracy_score += 0.2
            except BaseException:
                pass

        return min(accuracy_score, 1.0)

    def _check_consistency(self, data: pd.DataFrame, data_type: str) -> float:
        """检查数据一致性"""
        if data.empty:
            return 0.0

        consistency_score = 0.0

        # 检查数据类型一致性
        if 'close' in data.columns:
            numeric_consistency = pd.to_numeric(
                data['close'], errors='coerce').notna().sum() / len(data)
            consistency_score += numeric_consistency * 0.4

        # 检查时间序列一致性
        if 'date' in data.columns:
            try:
                dates = pd.to_datetime(data['date'])
                sorted_dates = dates.sort_values()
                time_consistency = (dates == sorted_dates).sum() / len(dates)
                consistency_score += time_consistency * 0.3
            except BaseException:
                pass

        # 检查数据范围一致性
        if 'close' in data.columns:
            price_range = data['close'].max() - data['close'].min()
            if price_range > 0:
                reasonable_range = min(price_range / data['close'].mean(), 10.0) / 10.0
                consistency_score += reasonable_range * 0.3

        return min(consistency_score, 1.0)

    def _check_timeliness(self, data: pd.DataFrame) -> float:
        """检查数据时效性"""
        if data.empty or 'date' not in data.columns:
            return 0.0

        try:
            # 检查最新数据时间
            latest_date = pd.to_datetime(data['date'].max())
            current_date = datetime.now()
            time_diff = (current_date - latest_date).days

            # 根据数据类型确定时效性要求
            if time_diff <= 1:  # 1天内
                return 1.0
            elif time_diff <= 7:  # 1周内
                return 0.8
            elif time_diff <= 30:  # 1月内
                return 0.6
            else:
                return 0.3
        except BaseException:
            return 0.5

    def _check_validity(self, data: pd.DataFrame, data_type: str) -> float:
        """检查数据有效性"""
        if data.empty:
            return 0.0

        validity_score = 0.0

        # 检查数据格式有效性
        if 'symbol' in data.columns:
            # 检查股票代码格式
            valid_symbols = data['symbol'].str.match(r'^\d{6}\.(SH|SZ)$').sum() / len(data)
            validity_score += valid_symbols * 0.4

        # 检查数值有效性
        if 'close' in data.columns:
            # 检查价格是否在合理范围内
            reasonable_prices = ((data['close'] > 0) & (data['close'] < 10000)).sum() / len(data)
            validity_score += reasonable_prices * 0.3

        # 检查日期有效性
        if 'date' in data.columns:
            try:
                valid_dates = pd.to_datetime(
                    data['date'], errors='coerce').notna().sum() / len(data)
                validity_score += valid_dates * 0.3
            except BaseException:
                pass

        return min(validity_score, 1.0)

    def _check_uniqueness(self, data: pd.DataFrame) -> float:
        """检查数据唯一性"""
        if data.empty:
            return 0.0

        # 检查重复行
        total_rows = len(data)
        unique_rows = len(data.drop_duplicates())
        uniqueness = unique_rows / total_rows

        return uniqueness

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """计算综合质量分数"""
        weights = {
            'completeness': 0.2,
            'accuracy': 0.25,
            'consistency': 0.2,
            'timeliness': 0.15,
            'validity': 0.15,
            'uniqueness': 0.05
        }

        overall_score = sum(scores[metric] * weights[metric] for metric in weights)
        return overall_score

    def _update_trend_analysis(self, metrics: QualityMetrics):
        """更新趋势分析"""
        for metric_name, analyzer in self._trend_analyzers.items():
            value = getattr(metrics, metric_name)
            analyzer.add_value(value, metrics.timestamp)

    def _detect_anomalies(self, metrics: QualityMetrics) -> List[AnomalyReport]:
        """检测异常"""
        anomalies = []

        # 统计异常检测
        for detector in self._anomaly_detectors.values():
            detector_anomalies = detector.detect(metrics)
            anomalies.extend(detector_anomalies)

        return anomalies

    def get_overall_quality_score(self) -> float:
        """获取整体质量分数"""
        if not self._quality_history:
            return 0.0

        recent_metrics = list(self._quality_history)[-10:]  # 最近10次
        avg_score = sum(m.overall_score for m in recent_metrics) / len(recent_metrics)
        return avg_score

    def get_quality_trends(self) -> Dict[str, QualityTrend]:
        """获取质量趋势"""
        trends = {}
        for metric_name, analyzer in self._trend_analyzers.items():
            trends[metric_name] = analyzer.get_trend()
        return trends

    def get_quality_report(self, data_type: str = None, time_range: str = "7d") -> Dict[str, Any]:
        """获取质量报告"""
        if not self._quality_history:
            return {"error": "No quality data available"}

        # 过滤数据
        filtered_metrics = self._filter_metrics_by_type_and_time(data_type, time_range)

        if not filtered_metrics:
            return {"error": "No data matching criteria"}

        # 计算统计信息
        scores = {
            'completeness': [m.completeness for m in filtered_metrics],
            'accuracy': [m.accuracy for m in filtered_metrics],
            'consistency': [m.consistency for m in filtered_metrics],
            'timeliness': [m.timeliness for m in filtered_metrics],
            'validity': [m.validity for m in filtered_metrics],
            'uniqueness': [m.uniqueness for m in filtered_metrics],
            'overall_score': [m.overall_score for m in filtered_metrics]
        }

        report = {
            'summary': {
                'total_checks': len(filtered_metrics),
                'avg_overall_score': np.mean(scores['overall_score']),
                'min_overall_score': np.min(scores['overall_score']),
                'max_overall_score': np.max(scores['overall_score'])
            },
            'metrics': {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                for metric, values in scores.items()
            },
            'trends': self.get_quality_trends(),
            'anomalies': self._get_recent_anomalies()
        }

        return report

    def _filter_metrics_by_type_and_time(self, data_type: str, time_range: str) -> List[QualityMetrics]:
        """根据类型和时间范围过滤指标"""
        filtered = list(self._quality_history)

        # 按类型过滤
        if data_type:
            filtered = [m for m in filtered if m.data_type == data_type]

        # 按时间范围过滤
        if time_range:
            cutoff_time = datetime.now() - self._parse_time_range(time_range)
            filtered = [m for m in filtered if datetime.fromisoformat(m.timestamp) > cutoff_time]

        return filtered

    def _parse_time_range(self, time_range: str) -> timedelta:
        """解析时间范围"""
        if time_range.endswith('d'):
            days = int(time_range[:-1])
            return timedelta(days=days)
        elif time_range.endswith('h'):
            hours = int(time_range[:-1])
            return timedelta(hours=hours)
        else:
            return timedelta(days=7)  # 默认7天

    def _get_recent_anomalies(self) -> List[Dict[str, Any]]:
        """获取最近的异常"""
        # 这里可以实现异常历史记录
        return []

    def _start_quality_monitoring(self):
        """启动质量监控"""

        def monitor_quality():

            while True:
                try:
                    # 定期质量检查
                    self._perform_periodic_quality_check()

                    # 清理历史数据
                    self._cleanup_old_data()

                    time.sleep(300)  # 每5分钟检查一次
                except Exception as e:
                    logger.error(f"质量监控错误: {e}")
                    time.sleep(600)

        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_quality, daemon=True)
        monitor_thread.start()

    def _perform_periodic_quality_check(self):
        """执行定期质量检查"""
        # 检查质量趋势
        trends = self.get_quality_trends()
        for metric_name, trend in trends.items():
            if trend.trend_direction == 'declining' and trend.trend_strength > 0.7:
                logger.warning(f"质量指标 {metric_name} 呈下降趋势，强度: {trend.trend_strength}")

    def _cleanup_old_data(self):
        """清理旧数据"""
        # 清理超过30天的质量历史
        cutoff_time = datetime.now() - timedelta(days=30)
        self._quality_history = deque(
            [m for m in self._quality_history if datetime.fromisoformat(m.timestamp) > cutoff_time],
            maxlen=1000
        )


class TrendAnalyzer:

    """趋势分析器"""

    def __init__(self, window_size: int = 10):

        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

    def add_value(self, value: float, timestamp: str):
        """添加新值"""
        self.values.append(value)
        self.timestamps.append(timestamp)

    def get_trend(self) -> QualityTrend:
        """获取趋势"""
        if len(self.values) < 3:
            return QualityTrend(
                metric_name="unknown",
                trend_direction="stable",
                trend_strength=0.0,
                change_rate=0.0,
                prediction=0.0,
                confidence=0.0
            )

        # 计算趋势
        values_array = np.array(list(self.values))
        x = np.arange(len(values_array))

        # 线性回归
        slope, intercept = np.polyfit(x, values_array, 1)

        # 计算趋势强度
        trend_strength = abs(slope) / (np.std(values_array) + 1e-8)
        trend_strength = min(trend_strength, 1.0)

        # 确定趋势方向
        if slope > 0.01:
            direction = "improving"
        elif slope < -0.01:
            direction = "declining"
        else:
            direction = "stable"

        # 计算变化率
        change_rate = (values_array[-1] - values_array[0]) / (values_array[0] + 1e-8)

        # 预测下一个值
        prediction = slope * len(values_array) + intercept
        prediction = max(0.0, min(1.0, prediction))

        # 计算置信度
        confidence = 1.0 - (np.std(values_array) / (np.mean(values_array) + 1e-8))
        confidence = max(0.0, min(1.0, confidence))

        return QualityTrend(
            metric_name="unknown",
            trend_direction=direction,
            trend_strength=trend_strength,
            change_rate=change_rate,
            prediction=prediction,
            confidence=confidence
        )


class StatisticalAnomalyDetector:

    """统计异常检测器"""

    def detect(self, metrics: QualityMetrics) -> List[AnomalyReport]:
        """检测统计异常"""
        anomalies = []

        # 检查质量分数异常
        if metrics.overall_score < 0.7:
            anomalies.append(AnomalyReport(
                anomaly_type="low_quality_score",
                severity="high",
                description=f"整体质量分数过低: {metrics.overall_score:.2f}",
                affected_data=metrics.data_type,
                detection_time=metrics.timestamp,
                suggested_action="检查数据源和数据处理流程"
            ))

        # 检查完整性异常
        if metrics.completeness < 0.8:
            anomalies.append(AnomalyReport(
                anomaly_type="low_completeness",
                severity="medium",
                description=f"数据完整性过低: {metrics.completeness:.2f}",
                affected_data=metrics.data_type,
                detection_time=metrics.timestamp,
                suggested_action="检查数据源连接和空值处理"
            ))

        return anomalies


class TemporalAnomalyDetector:

    """时间异常检测器"""

    def detect(self, metrics: QualityMetrics) -> List[AnomalyReport]:
        """检测时间异常"""
        # 这里可以实现基于时间的异常检测
        return []


class PatternAnomalyDetector:

    """模式异常检测器"""

    def detect(self, metrics: QualityMetrics) -> List[AnomalyReport]:
        """检测模式异常"""
        # 这里可以实现基于模式的异常检测
        return []


class QualityRuleEngine:

    """质量规则引擎"""

    def __init__(self):

        self.rules = self._load_default_rules()

    def _load_default_rules(self) -> Dict[str, Any]:
        """加载默认规则"""
        return {
            'completeness_threshold': 0.8,
            'accuracy_threshold': 0.9,
            'consistency_threshold': 0.85,
            'timeliness_threshold': 0.7,
            'validity_threshold': 0.9,
            'uniqueness_threshold': 0.95
        }


class QualityAlertManager:

    """质量告警管理器"""

    def __init__(self):

        self.alert_history = []

    def send_alerts(self, anomalies: List[AnomalyReport]):
        """发送告警"""
        for anomaly in anomalies:
            logger.warning(f"质量异常: {anomaly.description}")
            self.alert_history.append(anomaly)


def create_enhanced_quality_monitor(enable_alerting: bool = True, enable_trend_analysis: bool = True) -> EnhancedQualityMonitor:
    """创建增强版质量监控器"""
    return EnhancedQualityMonitor(enable_alerting, enable_trend_analysis)
