"""
数据质量监控系统
提供全面的数据质量检查、异常检测和报告功能
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class QualityLevel(Enum):

    """质量等级枚举"""
    EXCELLENT = "excellent"  # 90 - 100
    GOOD = "good"           # 80 - 89
    FAIR = "fair"           # 70 - 79
    POOR = "poor"           # 60 - 69
    CRITICAL = "critical"   # <60


class AlertLevel(Enum):

    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityMetric:

    """质量指标"""
    name: str
    value: float
    threshold: float
    weight: float = 1.0
    status: str = "unknown"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:

    """质量报告"""
    timestamp: datetime
    data_source: str
    data_shape: Tuple[int, int]
    overall_score: float
    quality_level: QualityLevel
    metrics: Dict[str, QualityMetric]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    alert_level: Optional[AlertLevel] = None


class QualityCheckResult(dict):
    """带有属性访问能力的质量检查结果容器。"""

    __slots__ = ("quality_report",)

    def __init__(self, *args, quality_report: Optional[QualityReport] = None, **kwargs):
        super().__init__(*args, **kwargs)
        super().__setattr__("quality_report", quality_report or self.get("quality_report"))

    def __getattr__(self, item: str):
        report = super().__getattribute__("quality_report")
        if report is not None and hasattr(report, item):
            return getattr(report, item)
        if item in self:
            return self[item]
        raise AttributeError(item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "quality_report":
            super().__setattr__(key, value)
        else:
            self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


@dataclass
class AnomalyRecord:

    """异常记录"""
    timestamp: datetime
    anomaly_type: str
    severity: str
    description: str
    affected_columns: List[str]
    affected_rows: int
    value: Any
    threshold: Any


class DataQualityRule:

    """数据质量规则基类"""

    def __init__(self, name: str, weight: float = 1.0):

        self.name = name
        self.weight = weight

    def check(self, data: pd.DataFrame) -> QualityMetric:
        """检查数据质量"""
        raise NotImplementedError


class CompletenessRule(DataQualityRule):

    """完整性检查规则"""

    def __init__(self, columns: Optional[List[str]] = None, threshold: float = 0.95):

        super().__init__("completeness")
        self.columns = columns
        self.threshold = threshold

    def check(self, data: pd.DataFrame) -> QualityMetric:

        if data.empty:
            return QualityMetric(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                weight=self.weight,
                status="critical",
                details={"message": "数据为空"}
            )

        if self.columns:
            check_columns = [col for col in self.columns if col in data.columns]
        else:
            check_columns = data.columns.tolist()

        if not check_columns:
            return QualityMetric(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                weight=self.weight,
                status="critical",
                details={"message": "没有可检查的列"}
            )

        missing_ratios = []
        for col in check_columns:
            missing_ratio = data[col].isnull().sum() / len(data)
            missing_ratios.append(missing_ratio)

        completeness_score = 1.0 - np.mean(missing_ratios)

        status = "excellent" if completeness_score >= self.threshold else "poor"

        return QualityMetric(
            name=self.name,
            value=completeness_score,
            threshold=self.threshold,
            weight=self.weight,
            status=status,
            details={
                "missing_ratios": dict(zip(check_columns, missing_ratios)),
                "average_missing_ratio": np.mean(missing_ratios)
            }
        )


class ConsistencyRule(DataQualityRule):

    """一致性检查规则"""

    def __init__(self, threshold: float = 0.9):

        super().__init__("consistency")
        self.threshold = threshold

    def check(self, data: pd.DataFrame) -> QualityMetric:

        if data.empty or len(data) < 2:
            return QualityMetric(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                weight=self.weight,
                status="excellent",
                details={"message": "数据量不足，跳过一致性检查"}
            )

        # 检查数据类型一致性
        type_consistency = 1.0
        for col in data.columns:
            if data[col].dtype == 'object':
                # 对于对象类型，检查是否有混合类型
                unique_types = data[col].apply(type).nunique()
                if unique_types > 1:
                    type_consistency *= 0.8

        # 检查时间序列一致性（如果有时间列）
        time_consistency = 1.0
        time_columns = [col for col in data.columns if 'time' in col.lower()
                        or 'date' in col.lower()]
        if time_columns:
            for col in time_columns:
                try:
                    time_data = pd.to_datetime(data[col], errors='coerce')
                    if not time_data.isnull().all():
                        # 检查时间间隔是否一致
                        time_diff = time_data.diff().dropna()
                        if len(time_diff) > 1:
                            time_std = time_diff.std()
                            time_mean = time_diff.mean()
                            if time_mean.total_seconds() > 0:
                                cv = time_std.total_seconds() / time_mean.total_seconds()
                                time_consistency *= max(0.5, 1.0 - cv)
                except BaseException:
                    time_consistency *= 0.9

        consistency_score = (type_consistency + time_consistency) / 2
        status = "excellent" if consistency_score >= self.threshold else "poor"

        return QualityMetric(
            name=self.name,
            value=consistency_score,
            threshold=self.threshold,
            weight=self.weight,
            status=status,
            details={
                "type_consistency": type_consistency,
                "time_consistency": time_consistency
            }
        )


class AccuracyRule(DataQualityRule):

    """准确性检查规则"""

    def __init__(self, numeric_columns: Optional[List[str]] = None, threshold: float = 0.8):

        super().__init__("accuracy")
        self.numeric_columns = numeric_columns
        self.threshold = threshold

    def check(self, data: pd.DataFrame) -> QualityMetric:

        if data.empty:
            return QualityMetric(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                weight=self.weight,
                status="critical",
                details={"message": "数据为空"}
            )

        if self.numeric_columns:
            check_columns = [col for col in self.numeric_columns if col in data.columns]
        else:
            check_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if not check_columns:
            return QualityMetric(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                weight=self.weight,
                status="excellent",
                details={"message": "没有数值列可检查"}
            )

        accuracy_scores = []
        outlier_details = {}

        for col in check_columns:
            col_data = data[col].dropna()
            if len(col_data) < 3:
                accuracy_scores.append(1.0)
                continue

            # 使用IQR方法检测异常值
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_ratio = len(outliers) / len(col_data)

            # 异常值比例越低，准确性越高
            accuracy_score = max(0.0, 1.0 - outlier_ratio)
            accuracy_scores.append(accuracy_score)

            outlier_details[col] = {
                "outlier_count": len(outliers),
                "outlier_ratio": outlier_ratio,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }

        overall_accuracy = np.mean(accuracy_scores)
        status = "excellent" if overall_accuracy >= self.threshold else "poor"

        return QualityMetric(
            name=self.name,
            value=overall_accuracy,
            threshold=self.threshold,
            weight=self.weight,
            status=status,
            details={"outlier_details": outlier_details}
        )


class TimelinessRule(DataQualityRule):

    """时效性检查规则"""

    def __init__(self, time_column: str = "timestamp", max_delay_hours: int = 24, threshold: float = 0.9):

        super().__init__("timeliness")
        self.time_column = time_column
        self.max_delay_hours = max_delay_hours
        self.threshold = threshold

    def check(self, data: pd.DataFrame) -> QualityMetric:

        if data.empty or self.time_column not in data.columns:
            return QualityMetric(
                name=self.name,
                value=1.0,
                threshold=self.threshold,
                weight=self.weight,
                status="excellent",
                details={"message": "没有时间列或数据为空"}
            )

        try:
            time_data = pd.to_datetime(data[self.time_column], errors='coerce')
            time_data = time_data.dropna()

            if len(time_data) == 0:
                return QualityMetric(
                    name=self.name,
                    value=0.0,
                    threshold=self.threshold,
                    weight=self.weight,
                    status="critical",
                    details={"message": "时间列数据无效"}
                )

            current_time = datetime.now()
            delays = []

            for timestamp in time_data:
                if pd.notna(timestamp):
                    delay = (current_time - timestamp).total_seconds() / 3600  # 转换为小时
                    delays.append(delay)

            if not delays:
                return QualityMetric(
                    name=self.name,
                    value=0.0,
                    threshold=self.threshold,
                    weight=self.weight,
                    status="critical",
                    details={"message": "无法计算延迟"}
                )

            # 计算时效性得分：延迟越短，得分越高
            avg_delay = np.mean(delays)
            timeliness_score = max(0.0, 1.0 - (avg_delay / self.max_delay_hours))

            status = "excellent" if timeliness_score >= self.threshold else "poor"

            return QualityMetric(
                name=self.name,
                value=timeliness_score,
                threshold=self.threshold,
                weight=self.weight,
                status=status,
                details={
                    "average_delay_hours": avg_delay,
                    "max_delay_hours": max(delays),
                    "min_delay_hours": min(delays)
                }
            )

        except Exception as e:
            return QualityMetric(
                name=self.name,
                value=0.0,
                threshold=self.threshold,
                weight=self.weight,
                status="critical",
                details={"message": f"时间处理错误: {str(e)}"}
            )


class DataQualityMonitor:

    """数据质量监控器"""

    _DEFAULT_ENABLED_METRICS = ["completeness", "accuracy", "consistency", "timeliness"]

    def __init__(self, data_source: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **legacy_kwargs):
        """
        初始化数据质量监控器

        Args:
            data_source: 数据源标识
            config: 配置字典
        """
        if isinstance(data_source, dict) and config is None:
            config = data_source
            data_source = legacy_kwargs.get("data_source")

        self.config = config or {}
        self.data_source = data_source or self.config.get("data_source", "unknown")
        self.metrics_enabled: List[str] = list(self.config.get("metrics_enabled", self._DEFAULT_ENABLED_METRICS))
        if not self.metrics_enabled:
            self.metrics_enabled = self._DEFAULT_ENABLED_METRICS.copy()

        self.alert_enabled: bool = bool(self.config.get("alert_enabled", True))
        self.alert_threshold: float = float(self.config.get("alert_threshold", 0.8))
        self.auto_repair: bool = bool(self.config.get("auto_repair", False))
        self.thresholds: Dict[str, float] = dict(self.config.get("thresholds", {}))

        self.rules: List[DataQualityRule] = []
        self.anomaly_history: List[AnomalyRecord] = []
        self.report_history: List[QualityCheckResult] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.alert_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self.alert_config = self.config.get('alerts', {
            'enabled': True,
            'levels': {
                'critical': {'threshold': 0.6, 'channels': ['email', 'sms']},
                'error': {'threshold': 0.7, 'channels': ['email']},
                'warning': {'threshold': 0.8, 'channels': ['log']},
                'info': {'threshold': 0.9, 'channels': ['log']}
            }
        })
        self.lock = threading.Lock()

        # 初始化默认规则
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认规则"""
        with self.lock:
            self.rules.clear()
            if "completeness" in self.metrics_enabled:
                self.rules.append(CompletenessRule())
            if "consistency" in self.metrics_enabled:
                self.rules.append(ConsistencyRule())
            if "accuracy" in self.metrics_enabled:
                self.rules.append(AccuracyRule())
            if "timeliness" in self.metrics_enabled:
                self.rules.append(TimelinessRule())

    def add_rule(self, rule: DataQualityRule):
        """添加质量检查规则"""
        with self.lock:
            if self.metrics_enabled and rule.name not in self.metrics_enabled:
                self.metrics_enabled.append(rule.name)
            self.rules.append(rule)

    def remove_rule(self, rule_name: str):
        """移除质量检查规则"""
        with self.lock:
            self.rules = [rule for rule in self.rules if rule.name != rule_name]

    def check_quality(self, data: pd.DataFrame, data_source: Optional[str] = None) -> Dict[str, Any]:
        """
        检查数据质量

        Args:
            data: 数据框
            data_source: 数据源标识

        Returns:
            Dict[str, Any]: 质量报告字典
        """
        start_time = datetime.now()
        source = data_source or self.data_source

        repair_actions: List[Dict[str, Any]] = []
        working_data = data
        if self.auto_repair and isinstance(data, pd.DataFrame):
            working_data, repair_actions = self._auto_repair_data(data)

        # 执行所有规则检查
        metrics: Dict[str, QualityMetric] = {}
        anomalies: List[AnomalyRecord] = []

        for rule in list(self.rules):
            if self.metrics_enabled and rule.name not in self.metrics_enabled:
                continue
            try:
                metric = rule.check(working_data)
                metrics[metric.name] = metric

                # 检查是否需要记录异常
                if metric.status in ['poor', 'critical']:
                    anomaly = AnomalyRecord(
                        timestamp=start_time,
                        anomaly_type=metric.name,
                        severity=metric.status,
                        description=f"{metric.name} 质量检查失败",
                        affected_columns=[],
                        affected_rows=len(working_data) if hasattr(working_data, "__len__") else 0,
                        value=metric.value,
                        threshold=metric.threshold
                    )
                    anomalies.append(anomaly)

            except Exception as e:
                logger.error(f"规则 {rule.name} 执行失败: {e}")
                metric = QualityMetric(
                    name=rule.name,
                    value=0.0,
                    threshold=0.0,
                    weight=rule.weight,
                    status="critical",
                    details={"error": str(e)}
                )
                metrics[metric.name] = metric

        # 计算总体得分
        total_weight = sum(metric.weight for metric in metrics.values())
        if total_weight > 0:
            overall_score = sum(
                metric.value * metric.weight for metric in metrics.values()) / total_weight
        else:
            overall_score = 0.0

        # 确定质量等级
        quality_level = self._determine_quality_level(overall_score)

        # 确定告警级别
        alert_level = self._determine_alert_level(overall_score)

        # 生成建议
        recommendations = self._generate_recommendations(metrics, overall_score)

        # 创建报告
        report = QualityReport(
            timestamp=start_time,
            data_source=source,
            data_shape=working_data.shape if isinstance(working_data, pd.DataFrame) else (0, 0),
            overall_score=overall_score,
            quality_level=quality_level,
            metrics=metrics,
            anomalies=anomalies,
            recommendations=recommendations,
            alert_level=alert_level
        )

        # 触发告警
        if self.alert_config.get('enabled', True) and alert_level:
            self._trigger_alert(alert_level, report)

        metric_values = {name: float(metric.value) for name, metric in metrics.items()}
        threshold_violations = self._collect_threshold_violations(metric_values)

        self.metrics_history.append({
            "timestamp": report.timestamp,
            "metrics": metric_values,
            "overall_score": overall_score,
            "data_source": source
        })

        alert_triggered = False
        if self.alert_enabled and overall_score < self.alert_threshold:
            alert_triggered = True
            self._send_alert({
                "quality_score": overall_score,
                "data_source": source,
                "timestamp": report.timestamp,
                "metrics": metric_values
            })

        result = {
            "data_source": source,
            "overall_score": overall_score,
            "quality_level": quality_level.value,
            "alert_level": alert_level.value if alert_level else None,
            "metrics": metric_values,
            "timestamp": report.timestamp.isoformat(),
            "anomalies": [anomaly.__dict__ for anomaly in anomalies],
            "recommendations": recommendations,
            "threshold_violations": threshold_violations,
            "alert_triggered": alert_triggered,
            "quality_report": report
        }

        for name, value in metric_values.items():
            if name not in result:
                result[name] = value

        if self.auto_repair:
            result["repair_actions"] = repair_actions

        result_obj = QualityCheckResult(result, quality_report=report)

        # 记录报告历史
        with self.lock:
            self.report_history.append(result_obj)
            self.anomaly_history.extend(anomalies)

        return result_obj

    def calculate_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """计算数据质量指标。"""
        metric_values: Dict[str, float] = {}
        for rule in list(self.rules):
            if self.metrics_enabled and rule.name not in self.metrics_enabled:
                continue
            try:
                metric = rule.check(data)
                metric_values[rule.name] = float(max(0.0, min(1.0, metric.value)))
            except Exception:
                metric_values[rule.name] = 0.0
        return metric_values

    def check_thresholds(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """检查指标阈值，返回违规列表。"""
        metrics = self.calculate_metrics(data)
        return self._collect_threshold_violations(metrics)

    def record_metrics(self, data: pd.DataFrame, timestamp: Optional[datetime] = None, data_source: Optional[str] = None) -> Dict[str, Any]:
        """记录一次指标快照。"""
        metrics = self.calculate_metrics(data)
        entry = {
            "timestamp": (timestamp or datetime.now()),
            "metrics": metrics,
            "data_source": data_source or self.data_source
        }
        self.metrics_history.append(entry)
        return entry

    def get_metrics_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取指标历史。"""
        history = self.metrics_history[-limit:] if limit else self.metrics_history[:]
        result = []
        for record in history:
            record_copy = dict(record)
            ts = record_copy.get("timestamp")
            if isinstance(ts, datetime):
                record_copy["timestamp"] = ts.isoformat()
            result.append(record_copy)
        return result

    def register_alert_handler(self, handler):
        """注册告警处理器."""
        if handler not in self.alert_handlers:
            self.alert_handlers.append(handler)

    def _determine_quality_level(self, score: float) -> QualityLevel:
        """确定质量等级"""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.8:
            return QualityLevel.GOOD
        elif score >= 0.7:
            return QualityLevel.FAIR
        elif score >= 0.6:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    def _determine_alert_level(self, score: float) -> Optional[AlertLevel]:
        """确定告警级别"""
        levels = self.alert_config.get('levels', {})

        if score < levels.get('critical', {}).get('threshold', 0.6):
            return AlertLevel.CRITICAL
        elif score < levels.get('error', {}).get('threshold', 0.7):
            return AlertLevel.ERROR
        elif score < levels.get('warning', {}).get('threshold', 0.8):
            return AlertLevel.WARNING
        elif score < levels.get('info', {}).get('threshold', 0.9):
            return AlertLevel.INFO

        return None

    def _generate_recommendations(self, metrics: Dict[str, QualityMetric], overall_score: float) -> List[str]:
        """生成改进建议"""
        recommendations = []

        for metric in metrics.values():
            if metric.status in ['poor', 'critical']:
                if metric.name == 'completeness':
                    recommendations.append("数据完整性不足，建议检查数据源和采集流程")
                elif metric.name == 'consistency':
                    recommendations.append("数据一致性存在问题，建议统一数据格式和类型")
                elif metric.name == 'accuracy':
                    recommendations.append("数据准确性较低，建议检查异常值和数据清洗流程")
                elif metric.name == 'timeliness':
                    recommendations.append("数据时效性不足，建议优化数据更新频率")

        if overall_score < 0.7:
            recommendations.append("整体数据质量较低，建议进行全面数据质量评估")

        return recommendations

    def _trigger_alert(self, alert_level: AlertLevel, report: QualityReport):
        """触发告警"""
        try:
            alert_config = self.alert_config.get('levels', {}).get(alert_level.value, {})
            channels = alert_config.get('channels', [])

            message = self._create_alert_message(alert_level, report)

            for channel in channels:
                if channel == 'email':
                    self._send_email_alert(message)
                elif channel == 'sms':
                    self._send_sms_alert(message)
                elif channel == 'log':
                    logger.warning(f"数据质量告警: {message}")

        except Exception as e:
            logger.error(f"告警触发失败: {e}")

    def _create_alert_message(self, alert_level: AlertLevel, report: QualityReport) -> str:
        """创建告警消息"""
        return (
            f"【{alert_level.value.upper()}】数据质量告警\n"
            f"时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"数据源: {report.data_source}\n"
            f"数据形状: {report.data_shape}\n"
            f"总体得分: {report.overall_score:.2f}\n"
            f"质量等级: {report.quality_level.value}\n"
            f"异常数量: {len(report.anomalies)}"
        )

    def _send_email_alert(self, message: str):
        """发送邮件告警"""
        # TODO: 实现邮件发送逻辑
        logger.info(f"邮件告警: {message}")

    def _send_sms_alert(self, message: str):
        """发送短信告警"""
        # TODO: 实现短信发送逻辑
        logger.info(f"短信告警: {message}")

    def get_quality_history(self, days: int = 7) -> List[QualityCheckResult]:
        """获取质量历史"""
        cutoff_date = datetime.now() - timedelta(days=days)

        with self.lock:
            return [
                report for report in self.report_history
                if report.timestamp >= cutoff_date
            ]

    def get_anomaly_history(self, days: int = 7) -> List[AnomalyRecord]:
        """获取异常历史"""
        cutoff_date = datetime.now() - timedelta(days=days)

        with self.lock:
            return [
                anomaly for anomaly in self.anomaly_history
                if anomaly.timestamp >= cutoff_date
            ]

    def generate_summary_report(self, days: int = 7) -> Dict[str, Any]:
        """生成汇总报告"""
        history = self.get_quality_history(days)

        if not history:
            return {"message": "没有历史数据"}

        scores = [report.overall_score for report in history]

        return {
            "period_days": days,
            "total_reports": len(history),
            "average_score": np.mean(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_std": np.std(scores),
            "quality_level_distribution": self._get_quality_level_distribution(history),
            "alert_level_distribution": self._get_alert_level_distribution(history),
            "top_issues": self._get_top_issues(history)
        }

    def _get_quality_level_distribution(self, history: List[QualityReport]) -> Dict[str, int]:
        """获取质量等级分布"""
        distribution = defaultdict(int)
        for report in history:
            distribution[report.quality_level.value] += 1
        return dict(distribution)

    def _get_alert_level_distribution(self, history: List[QualityReport]) -> Dict[str, int]:
        """获取告警级别分布"""
        distribution = defaultdict(int)
        for report in history:
            if report.alert_level:
                distribution[report.alert_level.value] += 1
        return dict(distribution)

    def _get_top_issues(self, history: List[QualityReport]) -> List[Dict[str, Any]]:
        """获取主要问题"""
        issue_counts = defaultdict(int)

        for report in history:
            for metric in report.metrics.values():
                if metric.status in ['poor', 'critical']:
                    issue_counts[metric.name] += 1

        # 按出现次数排序
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {"issue": issue, "count": count}
            for issue, count in sorted_issues[:5]  # 返回前5个问题
        ]

    def export_report(self, report: QualityReport, format: str = "json") -> str:
        """导出报告"""
        if format == "json":
            return json.dumps(report.__dict__, default=str, ensure_ascii=False, indent=2)
        elif format == "csv":
            # 简化的CSV导出
            return f"timestamp,data_source,overall_score,quality_level\n{report.timestamp},{report.data_source},{report.overall_score},{report.quality_level.value}"
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    def track_metrics(self, data_model=None, data_source: str = "unknown") -> Dict[str, Any]:
        """跟踪指标（为了向后兼容）"""
        try:
            # 如果有数据模型，使用它；否则创建空的DataFrame
            if data_model is not None and hasattr(data_model, 'data'):
                data = data_model.data
            else:
                data = pd.DataFrame()

            report = self.check_quality(data, data_source)
            return {
                'score': report.get("overall_score", 0.0),
                'quality_level': report.get("quality_level", "unknown"),
                'timestamp': report.get("timestamp", datetime.now().isoformat()),
                'data_source': data_source
            }
        except Exception as e:
            logger.error(f"跟踪指标失败: {e}")
            return {
                'score': 0.0,
                'quality_level': 'critical',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'data_source': data_source
            }

    def _auto_repair_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """对数据执行简单的自动修复。"""
        repaired = data.copy(deep=True)
        actions: List[Dict[str, Any]] = []

        for column in repaired.columns:
            if not repaired[column].isna().any():
                continue

            action: Dict[str, Any] = {"column": column}
            if pd.api.types.is_numeric_dtype(repaired[column]):
                fill_value = repaired[column].mean()
                if pd.isna(fill_value):
                    fill_value = 0.0
                repaired[column] = repaired[column].fillna(fill_value)
                action.update({"method": "mean_fill", "fill_value": float(fill_value)})
            else:
                modes = repaired[column].mode(dropna=True)
                fill_value = modes.iloc[0] if not modes.empty else ""
                repaired[column] = repaired[column].fillna(fill_value)
                action.update({"method": "mode_fill", "fill_value": str(fill_value)})
            actions.append(action)

        return repaired, actions

    def _collect_threshold_violations(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        for metric_name, threshold in self.thresholds.items():
            value = metrics.get(metric_name)
            if value is not None and value < threshold:
                violations.append({
                    "metric": metric_name,
                    "value": value,
                    "threshold": threshold
                })
        return violations

    def _send_alert(self, payload: Dict[str, Any]) -> None:
        """通知所有注册的告警处理器。"""
        for handler in list(self.alert_handlers):
            try:
                handler(payload)
            except Exception as exc:  # pragma: no cover
                logger.error(f"告警处理器执行失败: {exc}")
