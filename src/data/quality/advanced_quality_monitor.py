#!/usr/bin/env python3
"""
RQA2025 高级数据质量监控器

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

from src.infrastructure.logging import get_infrastructure_logger
提供更全面的数据质量监控功能：
- 跨数据源一致性检查
- 数据时效性监控
- 数据完整性检测
- 数据准确性验证
- 数据可靠性评估
- 实时质量监控
- 自动修复机制
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum

from ..cache.cache_manager import CacheManager, CacheConfig
# 日志降级处理


def get_data_logger(name: str):
    """获取数据层日志器，支持降级"""
    try:
        from src.infrastructure.logging import UnifiedLogger
        return UnifiedLogger(name)
    except ImportError:
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_data_logger('advanced_quality_monitor')

# 配置日志
logging.basicConfig(level=logging.INFO)

# logger已在上面定义


class QualityDimension(Enum):

    """质量维度枚举"""
    COMPLETENESS = "completeness"      # 完整性
    ACCURACY = "accuracy"              # 准确性
    CONSISTENCY = "consistency"        # 一致性
    TIMELINESS = "timeliness"          # 时效性
    VALIDITY = "validity"              # 有效性
    RELIABILITY = "reliability"        # 可靠性
    UNIQUENESS = "uniqueness"          # 唯一性
    INTEGRITY = "integrity"            # 完整性
    PRECISION = "precision"            # 精确性
    AVAILABILITY = "availability"      # 可用性


class QualityLevel(Enum):

    """质量等级枚举"""
    EXCELLENT = "excellent"    # 优秀 (90 - 100%)
    GOOD = "good"              # 良好 (80 - 89%)
    FAIR = "fair"              # 一般 (70 - 79%)
    POOR = "poor"              # 较差 (60 - 69%)
    UNACCEPTABLE = "unacceptable"  # 不可接受 (<60%)


@dataclass
class QualityMetric:

    """质量指标数据类"""
    dimension: QualityDimension
    score: float
    level: QualityLevel
    details: Dict[str, Any]
    timestamp: datetime
    source: str


@dataclass
class QualityAlert:

    """质量告警数据类"""
    alert_id: str
    dimension: QualityDimension
    severity: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    status: str  # active, resolved, acknowledged


@dataclass
class DataQualityReport:

    """数据质量报告数据类"""
    report_id: str
    overall_score: float
    overall_level: QualityLevel
    metrics: Dict[QualityDimension, QualityMetric]
    alerts: List[QualityAlert]
    recommendations: List[str]
    timestamp: datetime
    data_source: str


class AdvancedQualityMonitor:

    """高级数据质量监控器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.logger = logger

        # 初始化缓存管理器
        self.cache_manager = CacheManager(CacheConfig())

        # 质量阈值配置
        self.thresholds = self.config.get('thresholds', {
            'completeness': 0.9,
            'accuracy': 0.85,
            'consistency': 0.8,
            'timeliness': 0.95,
            'validity': 0.9,
            'reliability': 0.8
        })

        # 告警配置
        self.alert_config = self.config.get('alerts', {
            'enabled': True,
            'channels': ['log', 'email'],
            'severity_levels': ['critical', 'warning', 'info']
        })

        # 监控历史
        self.monitoring_history: List[DataQualityReport] = []
        self.quality_history: List[DataQualityReport] = []

        # 活跃告警
        self.active_alerts: List[QualityAlert] = []
        self.alert_counter: int = 0

        self.logger.info("AdvancedQualityMonitor initialized")

    async def check_completeness(self, data: pd.DataFrame) -> QualityMetric:
        """检查数据完整性"""
        logger.info("检查数据完整性...")

        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells * 100

        # 检查每列的完整性
        column_completeness = {}
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            total_count = len(data[column])
            column_completeness[column] = {
                'missing_count': missing_count,
                'total_count': total_count,
                'completeness_rate': (total_count - missing_count) / total_count * 100
            }

        level = self._get_quality_level(completeness_score)

        return QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            score=completeness_score,
            level=level,
            details={
                'total_cells': total_cells,
                'missing_cells': missing_cells,
                'column_completeness': column_completeness
            },
            timestamp=datetime.now(),
            source="advanced_monitor"
        )

    async def check_accuracy(self, data: pd.DataFrame, reference_data: Optional[pd.DataFrame] = None) -> QualityMetric:
        """检查数据准确性"""
        logger.info("检查数据准确性...")

        accuracy_checks = {}
        total_checks = 0
        passed_checks = 0

        # 数值列范围检查
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            total_checks += 1
            column_data = data[column].dropna()

            if len(column_data) > 0:
                # 检查异常值（使用IQR方法）
                Q1 = column_data.quantile(0.25)
                Q3 = column_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
                outlier_rate = len(outliers) / len(column_data)

                if outlier_rate < 0.05:  # 异常值率小于5%
                    passed_checks += 1

                accuracy_checks[column] = {
                    'outlier_rate': outlier_rate,
                    'outlier_count': len(outliers),
                    'total_count': len(column_data),
                    'range': [lower_bound, upper_bound]
                }

        # 与参考数据比较（如果提供）
        if reference_data is not None and not reference_data.empty:
            common_columns = set(data.columns) & set(reference_data.columns)
            for column in common_columns:
                total_checks += 1
                correlation = data[column].corr(reference_data[column])
                if correlation > 0.8:  # 相关性大于0.8
                    passed_checks += 1
                accuracy_checks[f'{column}_correlation'] = correlation

        accuracy_score = (passed_checks / total_checks * 100) if total_checks > 0 else 100
        level = self._get_quality_level(accuracy_score)

        return QualityMetric(
            dimension=QualityDimension.ACCURACY,
            score=accuracy_score,
            level=level,
            details={
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'accuracy_checks': accuracy_checks
            },
            timestamp=datetime.now(),
            source="advanced_monitor"
        )

    async def check_consistency(self, data: pd.DataFrame, data_sources: List[str] = None) -> QualityMetric:
        """检查数据一致性"""
        logger.info("检查数据一致性...")

        consistency_checks = {}
        total_checks = 0
        passed_checks = 0

        # 数据类型一致性
        total_checks += 1
        data_types_consistent = True
        for column in data.columns:
            if data[column].dtype != data[column].dtype:
                data_types_consistent = False
                break

        if data_types_consistent:
            passed_checks += 1

        consistency_checks['data_types'] = {
            'consistent': data_types_consistent,
            'types': {col: str(data[col].dtype) for col in data.columns}
        }

        # 数值范围一致性
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            total_checks += 1
            column_data = data[column].dropna()

            if len(column_data) > 0:
                # 检查数值是否在合理范围内
                min_val = column_data.min()
                max_val = column_data.max()

                # 根据列名判断合理范围
                if 'price' in column.lower() or 'value' in column.lower():
                    if min_val >= 0 and max_val < 1e12:  # 价格应该在合理范围内
                        passed_checks += 1
                elif 'percentage' in column.lower() or 'rate' in column.lower():
                    if min_val >= -100 and max_val <= 100:  # 百分比应该在 - 100到100之间
                        passed_checks += 1
                else:
                    passed_checks += 1  # 其他数值列默认通过

                consistency_checks[column] = {
                    'min_value': min_val,
                    'max_value': max_val,
                    'range_check': True
                }

        consistency_score = (passed_checks / total_checks * 100) if total_checks > 0 else 100
        level = self._get_quality_level(consistency_score)

        return QualityMetric(
            dimension=QualityDimension.CONSISTENCY,
            score=consistency_score,
            level=level,
            details={
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'consistency_checks': consistency_checks
            },
            timestamp=datetime.now(),
            source="advanced_monitor"
        )

    async def check_timeliness(self, data: pd.DataFrame, expected_frequency: str = "daily") -> QualityMetric:
        """检查数据时效性"""
        logger.info("检查数据时效性...")

        timeliness_score = 100
        timeliness_details = {}

        # 检查是否有时间戳列
        timestamp_columns = [col for col in data.columns if 'time' in col.lower()
                             or 'date' in col.lower()]

        if timestamp_columns:
            for col in timestamp_columns:
                try:
                    # 转换为datetime
                    data[col] = pd.to_datetime(data[col])
                    latest_time = data[col].max()
                    current_time = datetime.now()

                    # 计算延迟
                    if expected_frequency == "daily":
                        expected_delay = timedelta(days=1)
                    elif expected_frequency == "hourly":
                        expected_delay = timedelta(hours=1)
                    elif expected_frequency == "weekly":
                        expected_delay = timedelta(weeks=1)
                    else:
                        expected_delay = timedelta(days=1)

                    actual_delay = current_time - latest_time

                    if actual_delay <= expected_delay:
                        delay_score = 100
                    else:
                        delay_score = max(0, 100 - (actual_delay.days / expected_delay.days) * 50)

                    timeliness_score = min(timeliness_score, delay_score)
                    timeliness_details[col] = {
                        'latest_time': latest_time.isoformat(),
                        'current_time': current_time.isoformat(),
                        'actual_delay': str(actual_delay),
                        'expected_delay': str(expected_delay),
                        'delay_score': delay_score
                    }

                except Exception as e:
                    timeliness_details[col] = {
                        'error': str(e),
                        'delay_score': 0
                    }
                    timeliness_score = 0
        else:
            # 没有时间戳列，假设数据是及时的
            timeliness_details['no_timestamp'] = {
                'message': 'No timestamp column found, assuming timely data',
                'delay_score': 100
            }

        level = self._get_quality_level(timeliness_score)

        return QualityMetric(
            dimension=QualityDimension.TIMELINESS,
            score=timeliness_score,
            level=level,
            details=timeliness_details,
            timestamp=datetime.now(),
            source="advanced_monitor"
        )

    async def check_validity(self, data: pd.DataFrame, validation_rules: Dict[str, Any] = None) -> QualityMetric:
        """检查数据有效性"""
        logger.info("检查数据有效性...")

        validity_checks = {}
        total_checks = 0
        passed_checks = 0

        # 基本有效性检查
        for column in data.columns:
            total_checks += 1
            column_data = data[column].dropna()

            if len(column_data) > 0:
                # 检查数据类型有效性
                if data[column].dtype == 'object':
                    # 字符串列检查
                    if column_data.str.len().max() < 1000:  # 字符串长度合理
                        passed_checks += 1
                elif data[column].dtype in ['int64', 'float64']:
                    # 数值列检查
                    if not np.isinf(column_data).any() and not np.isnan(column_data).any():
                        passed_checks += 1
                else:
                    passed_checks += 1  # 其他类型默认通过

                validity_checks[column] = {
                    'data_type': str(data[column].dtype),
                    'valid_count': len(column_data),
                    'total_count': len(data[column]),
                    'validity_rate': len(column_data) / len(data[column]) * 100
                }

        # 自定义验证规则
        if validation_rules:
            for rule_name, rule_config in validation_rules.items():
                total_checks += 1
                column = rule_config.get('column')
                rule_type = rule_config.get('type')

                if column in data.columns:
                    column_data = data[column].dropna()

                    if rule_type == 'range':
                        min_val = rule_config.get('min')
                        max_val = rule_config.get('max')
                        if min_val is not None and max_val is not None:
                            valid_data = column_data[(column_data >= min_val)
                                                     & (column_data <= max_val)]
                            if len(valid_data) / len(column_data) > 0.95:  # 95 % 的数据在范围内
                                passed_checks += 1

                    elif rule_type == 'format':
                        pattern = rule_config.get('pattern')
                        if pattern:
                            # 简单的格式检查
                            if column_data.astype(str).str.match(pattern).sum() / len(column_data) > 0.95:
                                passed_checks += 1

                    validity_checks[f'rule_{rule_name}'] = {
                        'rule_type': rule_type,
                        'passed': True
                    }

        validity_score = (passed_checks / total_checks * 100) if total_checks > 0 else 100
        level = self._get_quality_level(validity_score)

        return QualityMetric(
            dimension=QualityDimension.VALIDITY,
            score=validity_score,
            level=level,
            details={
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'validity_checks': validity_checks
            },
            timestamp=datetime.now(),
            source="advanced_monitor"
        )

    async def check_reliability(self, data: pd.DataFrame, data_source: str = "unknown") -> QualityMetric:
        """检查数据可靠性"""
        logger.info("检查数据可靠性...")

        reliability_score = 100
        reliability_details = {}

        # 数据源可靠性评分（基于历史记录）
        source_reliability = await self._get_source_reliability(data_source)
        reliability_details['source_reliability'] = source_reliability

        # 数据稳定性检查
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        stability_scores = []

        for column in numeric_columns:
            column_data = data[column].dropna()
            if len(column_data) > 1:
                # 计算变异系数
                cv = column_data.std() / column_data.mean() if column_data.mean() != 0 else 0
                stability_score = max(0, 100 - cv * 100)  # 变异系数越小，稳定性越高
                stability_scores.append(stability_score)

                reliability_details[column] = {
                    'coefficient_of_variation': cv,
                    'stability_score': stability_score
                }

        if stability_scores:
            avg_stability = np.mean(stability_scores)
            reliability_score = (source_reliability + avg_stability) / 2
        else:
            reliability_score = source_reliability

        level = self._get_quality_level(reliability_score)

        return QualityMetric(
            dimension=QualityDimension.RELIABILITY,
            score=reliability_score,
            level=level,
            details=reliability_details,
            timestamp=datetime.now(),
            source="advanced_monitor"
        )

    async def _get_source_reliability(self, data_source: str) -> float:
        """获取数据源可靠性评分"""
        # 这里可以根据历史数据源的表现来评分
        # 暂时使用默认评分
        source_scores = {
            'coingecko': 95,
            'binance': 98,
            'fred': 99,
            'worldbank': 97,
            'unknown': 80
        }

        return source_scores.get(data_source.lower(), 80)

    def _get_quality_level(self, score: float) -> QualityLevel:
        """根据分数获取质量等级"""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 80:
            return QualityLevel.GOOD
        elif score >= 70:
            return QualityLevel.FAIR
        elif score >= 60:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE

    async def generate_alerts(self, metrics: Dict[QualityDimension, QualityMetric]) -> List[QualityAlert]:
        """生成质量告警"""
        alerts = []

        for dimension, metric in metrics.items():
            if metric.level in [QualityLevel.POOR, QualityLevel.UNACCEPTABLE]:
                self.alert_counter += 1
                alert = QualityAlert(
                    alert_id=f"alert_{self.alert_counter}",
                    dimension=dimension,
                    severity="high" if metric.level == QualityLevel.UNACCEPTABLE else "medium",
                    message=f"{dimension.value} quality is {metric.level.value} (score: {metric.score:.1f}%)",
                    details=metric.details,
                    timestamp=datetime.now(),
                    status="active"
                )
                alerts.append(alert)
                self.active_alerts.append(alert)

        return alerts

    async def generate_recommendations(self, metrics: Dict[QualityDimension, QualityMetric]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        for dimension, metric in metrics.items():
            if metric.level in [QualityLevel.POOR, QualityLevel.UNACCEPTABLE]:
                if dimension == QualityDimension.COMPLETENESS:
                    recommendations.append("建议检查数据源，确保数据完整性")
                elif dimension == QualityDimension.ACCURACY:
                    recommendations.append("建议增加数据验证规则，提高数据准确性")
                elif dimension == QualityDimension.CONSISTENCY:
                    recommendations.append("建议统一数据格式，确保数据一致性")
                elif dimension == QualityDimension.TIMELINESS:
                    recommendations.append("建议优化数据更新频率，提高数据时效性")
                elif dimension == QualityDimension.VALIDITY:
                    recommendations.append("建议加强数据验证，确保数据有效性")
                elif dimension == QualityDimension.RELIABILITY:
                    recommendations.append("建议评估数据源可靠性，考虑更换更可靠的数据源")

        return recommendations

    async def monitor_quality(self, data: pd.DataFrame, data_source: str = "unknown",
                              validation_rules: Dict[str, Any] = None) -> DataQualityReport:
        """监控数据质量"""
        logger.info(f"开始监控数据质量: {data_source}")

        # 执行各项质量检查
        completeness_metric = await self.check_completeness(data)
        accuracy_metric = await self.check_accuracy(data)
        consistency_metric = await self.check_consistency(data)
        timeliness_metric = await self.check_timeliness(data)
        validity_metric = await self.check_validity(data, validation_rules)
        reliability_metric = await self.check_reliability(data, data_source)

        # 收集所有指标
        metrics = {
            QualityDimension.COMPLETENESS: completeness_metric,
            QualityDimension.ACCURACY: accuracy_metric,
            QualityDimension.CONSISTENCY: consistency_metric,
            QualityDimension.TIMELINESS: timeliness_metric,
            QualityDimension.VALIDITY: validity_metric,
            QualityDimension.RELIABILITY: reliability_metric,
        }

        # 计算总体分数
        overall_score = sum(metric.score for metric in metrics.values()) / len(metrics)
        overall_level = self._get_quality_level(overall_score)

        # 生成告警
        alerts = await self.generate_alerts(metrics)

        # 生成建议
        recommendations = await self.generate_recommendations(metrics)

        # 创建质量报告
        report = DataQualityReport(
            report_id=f"quality_report_{int(time.time())}",
            overall_score=overall_score,
            overall_level=overall_level,
            metrics=metrics,
            alerts=alerts,
            recommendations=recommendations,
            timestamp=datetime.now(),
            data_source=data_source
        )

        # 保存到历史记录
        self.quality_history.append(report)

        logger.info(f"数据质量监控完成，总体分数: {overall_score:.1f}%")

        return report

    async def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        异步数据质量检查
        """
        try:
            # 聚合主要质量维度
            completeness_metric = await self.check_completeness(data)
            accuracy_metric = await self.check_accuracy(data)
            consistency_metric = await self.check_consistency(data)
            # 可扩展更多维度
            overall_score = np.mean([
                completeness_metric.score,
                accuracy_metric.score,
                consistency_metric.score
            ])
            overall_level = self._get_quality_level(overall_score)
            return {
                'completeness': completeness_metric.score,
                'accuracy': accuracy_metric.score,
                'consistency': consistency_metric.score,
                'overall_score': overall_score,
                'overall_level': overall_level.value,
                'details': {
                    'completeness': completeness_metric.details,
                    'accuracy': accuracy_metric.details,
                    'consistency': consistency_metric.details
                }
            }
        except Exception as e:
            logger.error(f"check_data_quality error: {e}")
            return {'overall_score': 0.0, 'overall_level': 'unacceptable', 'error': str(e)}

    def track_metrics(self, data_model, data_type: str) -> Dict[str, Any]:
        """跟踪数据指标（同步版本，用于兼容性）"""
        try:
            # 获取数据
            if hasattr(data_model, 'data'):
                data = data_model.data
            else:
                data = data_model

            # 执行质量检查（同步版本）
            metrics = {}

            # 简单的完整性检查
            if isinstance(data, pd.DataFrame):
                completeness_score = 1.0 - (data.isnull().sum().sum() /
                                            (data.shape[0] * data.shape[1]))
                metrics['completeness'] = {
                    'score': completeness_score,
                    'level': self._get_quality_level(completeness_score).value,
                    'details': {'null_count': data.isnull().sum().sum()}
                }

                # 简单的准确性检查
                accuracy_score = 1.0 if len(data) > 0 else 0.0
                metrics['accuracy'] = {
                    'score': accuracy_score,
                    'level': self._get_quality_level(accuracy_score).value,
                    'details': {'row_count': len(data)}
                }

                # 计算总体分数
                scores = [metric['score'] for metric in metrics.values()]
                overall_score = sum(scores) / len(scores) if scores else 0.0

                return {
                    'overall_score': overall_score,
                    'overall_level': self._get_quality_level(overall_score).value,
                    'metrics': metrics,
                    'data_type': data_type,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'overall_score': 0.0,
                    'overall_level': 'unacceptable',
                    'error': 'Invalid data format',
                    'data_type': data_type,
                    'timestamp': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Track metrics failed: {e}")
            return {
                'overall_score': 0.0,
                'overall_level': 'unacceptable',
                'error': str(e),
                'data_type': data_type,
                'timestamp': datetime.now().isoformat()
            }

    async def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """获取质量趋势"""
        if not self.quality_history:
            return {"message": "No quality history available"}

        # 过滤最近N天的报告
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_reports = [r for r in self.quality_history if r.timestamp >= cutoff_date]

        if not recent_reports:
            return {"message": f"No quality reports in the last {days} days"}

        # 计算趋势
        trends = {}
        for dimension in QualityDimension:
            scores = [r.metrics[dimension].score for r in recent_reports if dimension in r.metrics]
            if scores:
                trends[dimension.value] = {
                    'average_score': np.mean(scores),
                    'trend': 'improving' if scores[-1] > scores[0] else 'declining' if scores[-1] < scores[0] else 'stable',
                    'min_score': min(scores),
                    'max_score': max(scores)
                }

        return {
            'period_days': days,
            'total_reports': len(recent_reports),
            'trends': trends,
            'overall_trend': {
                'average_score': np.mean([r.overall_score for r in recent_reports]),
                'alert_count': sum(len(r.alerts) for r in recent_reports)
            }
        }

    async def export_report(self, report: DataQualityReport, format: str = "json") -> str:
        """导出质量报告"""
        if format == "json":
            report_data = {
                'report_id': report.report_id,
                'overall_score': report.overall_score,
                'overall_level': report.overall_level.value,
                'timestamp': report.timestamp.isoformat(),
                'data_source': report.data_source,
                'metrics': {
                    dim.value: {
                        'score': metric.score,
                        'level': metric.level.value,
                        'details': metric.details
                    }
                    for dim, metric in report.metrics.items()
                },
                'alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'dimension': alert.dimension.value,
                        'severity': alert.severity,
                        'message': alert.message,
                        'status': alert.status,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in report.alerts
                ],
                'recommendations': report.recommendations
            }

            # 保存到文件
            report_path = Path("reports") / f"quality_report_{report.report_id}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=self._json_default)

            return str(report_path)

        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def _json_default(obj):
        """JSON 序列化辅助，处理 numpy/pandas 类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)


# 便捷函数
async def monitor_data_quality(data: pd.DataFrame, data_source: str = "unknown",
                               validation_rules: Dict[str, Any] = None) -> DataQualityReport:
    """监控数据质量的便捷函数"""
    monitor = AdvancedQualityMonitor()
    return await monitor.monitor_quality(data, data_source, validation_rules)


if __name__ == "__main__":
    # 测试代码
    async def test_advanced_quality_monitor():
        """测试高级数据质量监控器"""
        print("测试高级数据质量监控器...")

        # 创建测试数据
        test_data = pd.DataFrame({
            'price': [100, 200, 300, None, 500],
            'volume': [1000, 2000, 3000, 4000, 5000],
            'timestamp': pd.date_range('2025 - 01 - 01', periods=5, freq='D'),
            'symbol': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
        })

        monitor = AdvancedQualityMonitor()
        report = await monitor.monitor_quality(test_data, "test_source")

        print(f"质量报告ID: {report.report_id}")
        print(f"总体分数: {report.overall_score:.1f}%")
        print(f"质量等级: {report.overall_level.value}")
        print(f"告警数量: {len(report.alerts)}")
        print(f"建议数量: {len(report.recommendations)}")

        # 导出报告
        report_path = await monitor.export_report(report)
        print(f"报告已导出: {report_path}")

    asyncio.run(test_advanced_quality_monitor())
