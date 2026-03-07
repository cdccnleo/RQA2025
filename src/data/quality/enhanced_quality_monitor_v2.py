#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 增强版数据质量监控器 V2

实现高级数据质量监控功能：
- 实时质量监控和告警
- 智能质量评估和评分
- 自动数据修复机制
- 跨数据源一致性检查
- 数据质量趋势分析
- 预测性质量监控
- 质量指标可视化
"""

import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import json
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from advanced_quality_monitor import (
    QualityDimension, QualityLevel, QualityMetric, QualityAlert, DataQualityReport
)

logger = logging.getLogger(__name__)


class QualityThreshold(Enum):

    """质量阈值枚举"""
    CRITICAL = 0.6    # 临界值
    WARNING = 0.8     # 警告值
    GOOD = 0.9        # 良好值
    EXCELLENT = 0.95  # 优秀值


class RepairStrategyV2(Enum):

    """修复策略枚举"""
    INTERPOLATION = "interpolation"      # 插值修复
    FORWARD_FILL = "forward_fill"        # 前向填充
    BACKWARD_FILL = "backward_fill"      # 后向填充
    MEAN_FILL = "mean_fill"              # 均值填充
    MEDIAN_FILL = "median_fill"          # 中位数填充
    DROP_NA = "drop_na"                  # 删除缺失值
    CUSTOM_RULE = "custom_rule"          # 自定义规则


@dataclass
class QualityTrend:

    """质量趋势数据类"""
    dimension: QualityDimension
    trend_direction: str  # increasing, decreasing, stable
    trend_strength: float  # 0 - 1
    prediction: float
    confidence: float
    timestamp: datetime


@dataclass
class RepairAction:

    """修复动作数据类"""
    action_id: str
    dimension: QualityDimension
    approach: RepairStrategyV2
    affected_rows: int
    success_rate: float
    timestamp: datetime
    details: Dict[str, Any]


class EnhancedQualityMonitorV2:

    """增强版数据质量监控器 V2"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化增强版质量监控器

        Args:
            config: 配置参数
        """
        self.config = config or {}

        # 质量阈值配置
        self.thresholds = {
            QualityDimension.COMPLETENESS: QualityThreshold.GOOD.value,
            QualityDimension.ACCURACY: QualityThreshold.EXCELLENT.value,
            QualityDimension.CONSISTENCY: QualityThreshold.GOOD.value,
            QualityDimension.TIMELINESS: QualityThreshold.WARNING.value,
            QualityDimension.VALIDITY: QualityThreshold.EXCELLENT.value,
            QualityDimension.RELIABILITY: QualityThreshold.GOOD.value,
            QualityDimension.UNIQUENESS: QualityThreshold.EXCELLENT.value,
            QualityDimension.INTEGRITY: QualityThreshold.EXCELLENT.value,
            QualityDimension.PRECISION: QualityThreshold.GOOD.value,
            QualityDimension.AVAILABILITY: QualityThreshold.WARNING.value
        }

        # 质量历史记录
        self.quality_history: List[QualityMetric] = []
        self.trend_analysis: List[QualityTrend] = []
        self.repair_actions: List[RepairAction] = []

        # 实时监控
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self._monitor_thread = None
        self._stop_monitoring = False

        # 异常检测模型
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()

        # 启动实时监控
        if self.monitoring_enabled:
            self._start_real_time_monitoring()

        logger.info("EnhancedQualityMonitorV2 initialized")

    async def monitor_quality_real_time(self, data: pd.DataFrame, data_source: str = "unknown") -> DataQualityReport:
        """
        实时质量监控

        Args:
            data: 待监控的数据
            data_source: 数据源标识

        Returns:
            DataQualityReport: 质量报告
        """
        start_time = time.time()

        # 1. 多维度质量检查
        metrics = {}
        for dimension in QualityDimension:
            metric = await self._check_dimension_quality(data, dimension, data_source)
            metrics[dimension] = metric
            self.quality_history.append(metric)

        # 2. 异常检测
        anomalies = await self._detect_anomalies(data, metrics)

        # 3. 质量趋势分析
        trends = await self._analyze_quality_trends(metrics)

        # 4. 自动修复建议
        repair_suggestions = await self._generate_repair_suggestions(data, metrics)

        # 5. 生成告警
        alerts = await self._generate_quality_alerts(metrics, anomalies)

        # 6. 计算综合评分
        overall_score = self._calculate_overall_score(metrics)
        overall_level = self._get_quality_level(overall_score)

        # 7. 生成报告
        report = DataQualityReport(
            report_id=f"quality_report_{int(time.time())}",
            overall_score=overall_score,
            overall_level=overall_level,
            metrics=metrics,
            alerts=alerts,
            recommendations=repair_suggestions,
            timestamp=datetime.now(),
            data_source=data_source
        )

        # 8. 记录监控统计
        monitoring_time = time.time() - start_time
        logger.info(f"Real - time quality monitoring completed in {monitoring_time:.2f}s")

        return report

    async def _check_dimension_quality(self, data: pd.DataFrame, dimension: QualityDimension,
                                       data_source: str) -> QualityMetric:
        """检查特定维度的质量"""
        if dimension == QualityDimension.COMPLETENESS:
            score = self._calculate_completeness_score(data)
        elif dimension == QualityDimension.ACCURACY:
            score = self._calculate_accuracy_score(data)
        elif dimension == QualityDimension.CONSISTENCY:
            score = self._calculate_consistency_score(data)
        elif dimension == QualityDimension.TIMELINESS:
            score = self._calculate_timeliness_score(data)
        elif dimension == QualityDimension.VALIDITY:
            score = self._calculate_validity_score(data)
        elif dimension == QualityDimension.RELIABILITY:
            score = self._calculate_reliability_score(data)
        elif dimension == QualityDimension.UNIQUENESS:
            score = self._calculate_uniqueness_score(data)
        elif dimension == QualityDimension.INTEGRITY:
            score = self._calculate_integrity_score(data)
        elif dimension == QualityDimension.PRECISION:
            score = self._calculate_precision_score(data)
        elif dimension == QualityDimension.AVAILABILITY:
            score = self._calculate_availability_score(data)
        else:
            score = 0.0

        level = self._get_quality_level(score)

        return QualityMetric(
            dimension=dimension,
            score=score,
            level=level,
            details={'data_source': data_source, 'data_size': len(data)},
            timestamp=datetime.now(),
            source=data_source
        )

    def _calculate_completeness_score(self, data: pd.DataFrame) -> float:
        """计算完整性评分"""
        if data.empty:
            return 0.0

        # 计算非空值比例
        non_null_ratio = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))

        # 简化逻辑，直接返回非空值比例
        return non_null_ratio

    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """计算准确性评分"""
        if data.empty:
            return 0.0

        # 检查数值范围合理性
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.8  # 非数值数据默认准确性较高

        accuracy_scores = []
        for col in numeric_cols:
            # 检查异常值
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 计算非异常值比例
            valid_ratio = ((data[col] >= lower_bound) & (data[col] <= upper_bound)).mean()
            accuracy_scores.append(valid_ratio)

        return np.mean(accuracy_scores) if accuracy_scores else 0.8

    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """计算一致性评分"""
        if data.empty:
            return 0.0

        # 检查字符串列的一致性
        string_cols = data.select_dtypes(include=['object']).columns
        if len(string_cols) == 0:
            return 1.0  # 没有字符串列，默认一致性高

        consistency_scores = []
        for col in string_cols:
            # 计算字符串长度的变异系数
            str_lengths = data[col].astype(str).str.len()
            if len(str_lengths) > 1:
                mean_len = str_lengths.mean()
                std_len = str_lengths.std()
                if mean_len > 0:
                    cv = std_len / mean_len
                    # 变异系数越小，一致性越好
                    col_consistency = max(0, 1 - cv)
                    consistency_scores.append(col_consistency)
                else:
                    consistency_scores.append(1.0)
            else:
                consistency_scores.append(1.0)

        # 如果没有有效的字符串列，返回默认值
        if not consistency_scores:
            return 1.0

        return np.mean(consistency_scores)

    def _calculate_timeliness_score(self, data: pd.DataFrame) -> float:
        """计算时效性评分"""
        if data.empty:
            return 0.0

        # 检查是否有时间列
        time_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        if not time_cols:
            return 0.7  # 没有时间列，默认时效性中等

        # 检查时间序列的连续性
        time_col = time_cols[0]
        try:
            time_series = pd.to_datetime(data[time_col])
            time_diff = time_series.diff().dropna()

            if len(time_diff) > 0:
                # 计算时间间隔的一致性
                mean_interval = time_diff.mean()
                std_interval = time_diff.std()
                consistency = 1 - \
                    (std_interval / mean_interval) if mean_interval.total_seconds() > 0 else 0
                return max(0, consistency)
        except BaseException:
            pass

        return 0.7

    def _calculate_validity_score(self, data: pd.DataFrame) -> float:
        """计算有效性评分"""
        if data.empty:
            return 0.0

        # 检查数据是否符合预期格式
        validity_scores = []

        for col in data.columns:
            if data[col].dtype == 'object':
                # 检查字符串长度合理性
                str_lengths = data[col].astype(str).str.len()
                if len(str_lengths) > 0:
                    # 长度在合理范围内的比例
                    reasonable_length = ((str_lengths > 0) & (str_lengths < 1000)).mean()
                    validity_scores.append(reasonable_length)
            else:
                # 数值列默认有效
                validity_scores.append(1.0)

        return np.mean(validity_scores) if validity_scores else 0.8

    def _calculate_reliability_score(self, data: pd.DataFrame) -> float:
        """计算可靠性评分"""
        if data.empty:
            return 0.0

        # 基于数据源历史可靠性评分
        # 这里可以根据数据源的历史表现来评分
        base_reliability = 0.8

        # 考虑数据质量因素
        quality_factor = self._calculate_completeness_score(data) * 0.5 + \
            self._calculate_accuracy_score(data) * 0.5

        return base_reliability * quality_factor

    def _calculate_uniqueness_score(self, data: pd.DataFrame) -> float:
        """计算唯一性评分"""
        if data.empty:
            return 0.0

        # 检查重复行比例
        duplicate_ratio = data.duplicated().mean()
        uniqueness_score = 1 - duplicate_ratio

        return uniqueness_score

    def _calculate_integrity_score(self, data: pd.DataFrame) -> float:
        """计算完整性评分"""
        if data.empty:
            return 0.0

        # 检查主键完整性
        integrity_scores = []

        # 检查是否有明显的ID列
        id_cols = [col for col in data.columns if 'id' in col.lower() or 'key' in col.lower()]
        if id_cols:
            for col in id_cols:
                # 检查ID的唯一性
                uniqueness = 1 - (data[col].duplicated().sum() / len(data))
                integrity_scores.append(uniqueness)

        # 检查外键引用完整性（简化版本）
        if len(integrity_scores) > 0:
            return np.mean(integrity_scores)
        else:
            return 0.8  # 默认完整性评分

    def _calculate_precision_score(self, data: pd.DataFrame) -> float:
        """计算精确性评分"""
        if data.empty:
            return 0.0

        # 检查数值列的精度
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.8  # 非数值数据默认精确性较高

        precision_scores = []
        for col in numeric_cols:
            # 检查数值的有效位数
            if data[col].dtype in ['float64', 'float32']:
                # 浮点数精度检查
                precision = data[col].astype(str).str.split('.').str[1].str.len().max()
                precision_score = min(1.0, precision / 10)  # 假设10位精度为满分
                precision_scores.append(precision_score)
            else:
                precision_scores.append(1.0)

        return np.mean(precision_scores) if precision_scores else 0.8

    def _calculate_availability_score(self, data: pd.DataFrame) -> float:
        """计算可用性评分"""
        if data.empty:
            return 0.0

        # 检查数据是否可用（非空、格式正确等）
        availability_scores = []

        for col in data.columns:
            # 检查列是否包含有效数据
            valid_data = data[col].notna().sum()
            total_data = len(data)
            availability_ratio = valid_data / total_data if total_data > 0 else 0
            availability_scores.append(availability_ratio)

        return np.mean(availability_scores) if availability_scores else 0.8

    def _get_quality_level(self, score: float) -> QualityLevel:
        """根据评分获取质量等级"""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.8:
            return QualityLevel.GOOD
        elif score >= 0.7:
            return QualityLevel.FAIR
        elif score >= 0.6:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE

    async def _detect_anomalies(self, data: pd.DataFrame, metrics: Dict[QualityDimension, QualityMetric]) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []

        # 基于质量指标的异常检测
        for dimension, metric in metrics.items():
            if metric.score < self.thresholds[dimension]:
                severity = 'critical' if metric.score < QualityThreshold.CRITICAL.value else 'warning'
                anomalies.append({
                    'dimension': dimension.value,
                    'score': metric.score,
                    'threshold': self.thresholds[dimension],
                    'severity': severity
                })

        # 基于数据内容的异常检测
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty and len(numeric_data) > 10:
            try:
                # 使用隔离森林检测异常
                scaled_data = self.scaler.fit_transform(numeric_data.fillna(0))
                anomaly_labels = self.anomaly_detector.fit_predict(scaled_data)
                anomaly_ratio = (anomaly_labels == -1).mean()

                if anomaly_ratio > 0.1:  # 异常比例超过10%
                    severity = 'critical' if anomaly_ratio > 0.2 else 'warning'
                    anomalies.append({
                        'dimension': 'data_anomaly',
                        'score': 1 - anomaly_ratio,
                        'threshold': 0.9,
                        'severity': severity
                    })
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")

        return anomalies

    async def _analyze_quality_trends(self, metrics: Dict[QualityDimension, QualityMetric]) -> List[QualityTrend]:
        """分析质量趋势"""
        trends = []

        # 分析每个维度的趋势
        for dimension in QualityDimension:
            if dimension in metrics:
                trend = await self._analyze_dimension_trend(dimension, metrics[dimension])
                if trend:
                    trends.append(trend)

        return trends

    async def _analyze_dimension_trend(self, dimension: QualityDimension, current_metric: QualityMetric) -> Optional[QualityTrend]:
        """分析特定维度的趋势"""
        # 获取历史数据
        historical_metrics = [m for m in self.quality_history if m.dimension == dimension]

        if len(historical_metrics) < 5:
            return None

        # 计算趋势
        scores = [m.score for m in historical_metrics[-10:]]  # 最近10次
        if len(scores) < 3:
            return None

        # 简单线性趋势分析
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]

        # 确定趋势方向
        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"

        # 计算趋势强度
        trend_strength = abs(slope) / max(scores) if max(scores) > 0 else 0

        # 简单预测
        prediction = min(1.0, max(0.0, current_metric.score + slope))
        confidence = max(0.5, 1 - trend_strength)  # 趋势越强，置信度越低

        return QualityTrend(
            dimension=dimension,
            trend_direction=direction,
            trend_strength=trend_strength,
            prediction=prediction,
            confidence=confidence,
            timestamp=datetime.now()
        )

    async def _generate_repair_suggestions(self, data: pd.DataFrame,
                                           metrics: Dict[QualityDimension, QualityMetric]) -> List[str]:
        """生成修复建议"""
        suggestions = []

        for dimension, metric in metrics.items():
            if metric.score < self.thresholds[dimension]:
                suggestion = self._get_repair_suggestion(dimension, data, metric.score)
                if suggestion:
                    suggestions.append(suggestion)

        return suggestions

    def _get_repair_suggestion(self, dimension: QualityDimension, data: pd.DataFrame, score: float) -> Optional[str]:
        """获取修复建议"""
        if dimension == QualityDimension.COMPLETENESS and score < 0.8:
            return "建议使用插值或前向填充方法修复缺失值"
        elif dimension == QualityDimension.ACCURACY and score < 0.9:
            return "建议检查并清理异常值，使用统计方法识别离群点"
        elif dimension == QualityDimension.CONSISTENCY and score < 0.8:
            return "建议统一数据格式，建立数据标准化规则"
        elif dimension == QualityDimension.TIMELINESS and score < 0.7:
            return "建议检查数据更新频率，确保数据时效性"
        elif dimension == QualityDimension.VALIDITY and score < 0.9:
            return "建议建立数据验证规则，确保数据有效性"
        elif dimension == QualityDimension.UNIQUENESS and score < 0.9:
            return "建议检查并删除重复数据，确保数据唯一性"

        return None

    async def _generate_quality_alerts(self, metrics: Dict[QualityDimension, QualityMetric],
                                       anomalies: List[Dict[str, Any]]) -> List[QualityAlert]:
        """生成质量告警"""
        alerts = []

        # 基于质量指标的告警
        for dimension, metric in metrics.items():
            if metric.score < self.thresholds[dimension]:
                severity = 'critical' if metric.score < QualityThreshold.CRITICAL.value else 'warning'
                alert = QualityAlert(
                    alert_id=f"alert_{dimension.value}_{int(time.time())}",
                    dimension=dimension,
                    severity=severity,
                    message=f"{dimension.value}质量评分过低: {metric.score:.2f}",
                    details={'score': metric.score, 'threshold': self.thresholds[dimension]},
                    timestamp=datetime.now(),
                    status='active'
                )
                alerts.append(alert)

        # 基于异常检测的告警
        for anomaly in anomalies:
            # 将异常严重性级别映射到告警级别
            severity = 'critical' if anomaly.get('severity') == 'critical' else 'warning'
            alert = QualityAlert(
                alert_id=f"alert_anomaly_{int(time.time())}",
                dimension=QualityDimension.ACCURACY,  # 异常通常影响准确性
                severity=severity,
                message=f"检测到数据异常: {anomaly['dimension']}",
                details=anomaly,
                timestamp=datetime.now(),
                status='active'
            )
            alerts.append(alert)

        return alerts

    def _calculate_overall_score(self, metrics: Dict[QualityDimension, QualityMetric]) -> float:
        """计算综合评分"""
        if not metrics:
            return 0.0

        # 加权平均，不同维度权重不同
        weights = {
            QualityDimension.COMPLETENESS: 0.15,
            QualityDimension.ACCURACY: 0.20,
            QualityDimension.CONSISTENCY: 0.15,
            QualityDimension.TIMELINESS: 0.10,
            QualityDimension.VALIDITY: 0.15,
            QualityDimension.RELIABILITY: 0.10,
            QualityDimension.UNIQUENESS: 0.05,
            QualityDimension.INTEGRITY: 0.05,
            QualityDimension.PRECISION: 0.03,
            QualityDimension.AVAILABILITY: 0.02
        }

        total_score = 0.0
        total_weight = 0.0

        for dimension, metric in metrics.items():
            weight = weights.get(dimension, 0.05)
            total_score += metric.score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _start_real_time_monitoring(self):
        """启动实时监控"""

        def monitoring_worker():

            while not self._stop_monitoring:
                try:
                    # 定期检查质量趋势
                    if len(self.quality_history) > 10:
                        self._check_quality_trends()

                    time.sleep(300)  # 每5分钟检查一次
                except Exception as e:
                    logger.error(f"Real - time monitoring error: {e}")
                    time.sleep(600)  # 出错后等待10分钟

        self._monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self._monitor_thread.start()

    def _check_quality_trends(self):
        """检查质量趋势"""
        # 分析最近的质量趋势
        recent_metrics = self.quality_history[-20:]  # 最近20次

        # 按维度分组分析
        for dimension in QualityDimension:
            dimension_metrics = [m for m in recent_metrics if m.dimension == dimension]
            if len(dimension_metrics) >= 5:
                scores = [m.score for m in dimension_metrics]
                trend = self._calculate_trend(scores)

                if trend['direction'] == 'decreasing' and trend['strength'] > 0.1:
                    logger.warning(
                        f"Quality trend decreasing for {dimension.value}: {trend['strength']:.2f}")

    def _calculate_trend(self, scores: List[float]) -> Dict[str, Any]:
        """计算趋势"""
        if len(scores) < 2:
            return {'direction': 'stable', 'strength': 0.0}

        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]

        if slope > 0.01:
            direction = 'increasing'
        elif slope < -0.01:
            direction = 'decreasing'
        else:
            direction = 'stable'

        strength = abs(slope) / max(scores) if max(scores) > 0 else 0

        return {
            'direction': direction,
            'strength': strength
        }

    def get_quality_summary(self) -> Dict[str, Any]:
        """获取质量摘要"""
        if not self.quality_history:
            return {}

        recent_metrics = self.quality_history[-50:]  # 最近50次

        summary = {
            'total_checks': len(self.quality_history),
            'recent_checks': len(recent_metrics),
            'average_scores': {},
            'trend_analysis': {},
            'repair_actions': len(self.repair_actions)
        }

        # 按维度统计平均分
        for dimension in QualityDimension:
            dimension_metrics = [m for m in recent_metrics if m.dimension == dimension]
            if dimension_metrics:
                avg_score = np.mean([m.score for m in dimension_metrics])
                summary['average_scores'][dimension.value] = avg_score

        return summary

    def export_quality_report(self, report: DataQualityReport, format: str = "json") -> str:
        """导出质量报告"""
        if format == "json":
            # 转换枚举类型为字符串
            report_dict = asdict(report)
            # 处理枚举类型
            if 'metrics' in report_dict:
                metrics_dict = {}
                for key, value in report_dict['metrics'].items():
                    if hasattr(key, 'value'):
                        key_str = key.value
                    else:
                        key_str = str(key)

                    if hasattr(value, '__dict__'):  # 检查是否是dataclass实例
                        metrics_dict[key_str] = asdict(value)
                    else:
                        metrics_dict[key_str] = value
                report_dict['metrics'] = metrics_dict

            # 处理其他枚举类型
            if 'overall_level' in report_dict and hasattr(report_dict['overall_level'], 'value'):
                report_dict['overall_level'] = report_dict['overall_level'].value

            # 处理告警列表
            if 'alerts' in report_dict:
                alerts_list = []
                for alert in report_dict['alerts']:
                    if hasattr(alert, '__dict__'):  # 检查是否是dataclass实例
                        alert_dict = asdict(alert)
                    else:
                        alert_dict = alert
                    if 'dimension' in alert_dict and hasattr(alert_dict['dimension'], 'value'):
                        alert_dict['dimension'] = alert_dict['dimension'].value
                    alerts_list.append(alert_dict)
                report_dict['alerts'] = alerts_list

            return json.dumps(report_dict, default=str, indent=2)
        elif format == "csv":
            # 转换为CSV格式
            report_data = {
                'report_id': report.report_id,
                'overall_score': report.overall_score,
                'overall_level': report.overall_level.value,
                'timestamp': report.timestamp.isoformat(),
                'data_source': report.data_source
            }

            # 添加各维度评分
            for dimension, metric in report.metrics.items():
                report_data[f"{dimension.value}_score"] = metric.score
                report_data[f"{dimension.value}_level"] = metric.level.value

            df = pd.DataFrame([report_data])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def shutdown(self):
        """关闭监控器"""
        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

        logger.info("EnhancedQualityMonitorV2 shutdown")


# 工厂函数

def create_enhanced_quality_monitor(config: Optional[Dict[str, Any]] = None) -> EnhancedQualityMonitorV2:
    """创建增强版质量监控器实例"""
    return EnhancedQualityMonitorV2(config)


# 便捷函数
async def monitor_data_quality_real_time(data: pd.DataFrame, data_source: str = "unknown",
                                         config: Optional[Dict[str, Any]] = None) -> DataQualityReport:
    """便捷的实时质量监控函数"""
    monitor = create_enhanced_quality_monitor(config)
    try:
        result = await monitor.monitor_quality_real_time(data, data_source)
        return result
    finally:
        monitor.shutdown()
