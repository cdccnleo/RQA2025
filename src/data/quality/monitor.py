from typing import Dict, List, Optional, TypedDict
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from .validator import ValidationResult
import json
import statistics

class QualityMetric(TypedDict):
    """数据质量指标定义"""
    completeness: float  # 数据完整率
    accuracy: float      # 数据准确率
    timeliness: float   # 数据及时性
    consistency: float  # 数据一致性
    uniqueness: float   # 数据唯一性

class DataQualityMonitor:
    """数据质量实时监控"""

    def __init__(self):
        self.alert_rules = {
            'critical': {'threshold': 0.7, 'channels': ['email', 'sms']},
            'warning': {'threshold': 0.85, 'channels': ['email']},
            'info': {'threshold': 0.95, 'channels': []}
        }
        self.history: List[Dict] = []

    def monitor(self, validation_result: ValidationResult) -> None:
        """执行实时监控"""
        # 记录历史数据
        self._record_history(validation_result)

        # 评估告警级别
        alert_level = self._evaluate_alert_level(validation_result)

        # 触发告警
        if alert_level:
            self._trigger_alert(alert_level, validation_result)

    def generate_report(self, days: int = 7) -> Dict:
        """生成质量报告"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        filtered = [
            item for item in self.history
            if datetime.fromisoformat(item['timestamp']) >= start_date
        ]

        if not filtered:
            return {}

        # 计算各项指标的平均值
        metrics = filtered[0]['metrics'].keys()
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'metrics': {
                metric: {
                    'avg': statistics.mean(item['metrics'][metric] for item in filtered),
                    'min': min(item['metrics'][metric] for item in filtered),
                    'max': max(item['metrics'][metric] for item in filtered)
                }
                for metric in metrics
            },
            'total_alerts': sum(1 for item in filtered if item['alert_level'])
        }

        return report

    def _record_history(self, result: ValidationResult) -> None:
        """记录验证结果到历史数据"""
        self.history.append({
            'timestamp': result.timestamp,
            'metrics': result.metrics,
            'errors': result.errors,
            'alert_level': self._evaluate_alert_level(result)
        })

    def _evaluate_alert_level(self, result: ValidationResult) -> Optional[str]:
        """评估告警级别"""
        if not result.is_valid:
            return 'critical'

        min_score = min(result.metrics.values())

        if min_score < self.alert_rules['critical']['threshold']:
            return 'critical'
        elif min_score < self.alert_rules['warning']['threshold']:
            return 'warning'
        elif min_score < self.alert_rules['info']['threshold']:
            return 'info'

        return None

    def _trigger_alert(self, level: str, result: ValidationResult) -> None:
        """触发告警"""
        channels = self.alert_rules[level]['channels']
        message = self._create_alert_message(level, result)

        if 'email' in channels:
            self._send_email_alert(message)

        if 'sms' in channels:
            self._send_sms_alert(message)

    def _create_alert_message(self, level: str, result: ValidationResult) -> str:
        """创建告警消息"""
        return (
            f"【{level.upper()}】数据质量告警\n"
            f"时间: {result.timestamp}\n"
            f"指标得分: {json.dumps(result.metrics, indent=2)}\n"
            f"错误信息: {', '.join(result.errors) if result.errors else '无'}"
        )

    def _send_email_alert(self, message: str) -> None:
        """发送邮件告警"""
        # 实现细节...
        pass

    def _send_sms_alert(self, message: str) -> None:
        """发送短信告警"""
        # 实现细节...
        pass
