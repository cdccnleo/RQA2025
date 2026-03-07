"""
健康报告构建器

负责构建和组装健康报告的核心结构。
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from ...models.alert_dataclasses import PerformanceMetrics
from ...core.shared_interfaces import ILogger, StandardLogger


class HealthReportBuilder:
    """健康报告构建器"""
    
    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
        self.metrics_formatter = MetricsFormatter(logger)
    
    def build_health_report(self, health_assessment: Dict[str, Any],
                           metrics: Optional[PerformanceMetrics] = None,
                           alert_stats: Optional[Dict[str, Any]] = None,
                           test_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """构建完整的健康报告"""
        try:
            report = self._initialize_health_report()
            self._populate_overall_health(report, health_assessment)
            self._populate_detailed_scores(report, health_assessment)
            self._populate_current_metrics(report, metrics)
            self._populate_statistics(report, alert_stats, test_stats)
            self._populate_issues_and_recommendations(report, health_assessment)
            self._populate_metadata(report, health_assessment)

            self.logger.log_info("成功构建健康报告")
            return report

        except Exception as e:
            self.logger.log_error(f"构建健康报告失败: {e}")
            return self._generate_error_report(str(e))

    def _initialize_health_report(self) -> Dict[str, Any]:
        """初始化健康报告基础结构"""
        return {
            'report_type': 'system_health_report',
            'generated_at': datetime.now().isoformat(),
            'report_period': 'current',
        }

    def _populate_overall_health(self, report: Dict[str, Any], health_assessment: Dict[str, Any]) -> None:
        """填充整体健康状态信息"""
        report['overall_health'] = {
            'status': health_assessment.get('overall_status', 'unknown'),
            'score': health_assessment.get('overall_score', 0.0),
            'issues_count': len(health_assessment.get('issues', [])),
            'recommendations_count': len(health_assessment.get('recommendations', []))
        }

    def _populate_detailed_scores(self, report: Dict[str, Any], health_assessment: Dict[str, Any]) -> None:
        """填充详细评分信息"""
        report['detailed_scores'] = {
            'performance': health_assessment.get('performance_score', 0.0),
            'alerts': health_assessment.get('alert_score', 0.0),
            'tests': health_assessment.get('test_score', 0.0)
        }

    def _populate_current_metrics(self, report: Dict[str, Any], metrics: Optional[PerformanceMetrics]) -> None:
        """填充当前指标信息"""
        report['current_metrics'] = self.metrics_formatter.format_metrics(metrics)

    def _populate_statistics(self, report: Dict[str, Any], alert_stats: Optional[Dict[str, Any]], 
                           test_stats: Optional[Dict[str, Any]]) -> None:
        """填充统计信息"""
        report['statistics'] = {
            'alerts': alert_stats or {},
            'tests': test_stats or {}
        }

    def _populate_issues_and_recommendations(self, report: Dict[str, Any], health_assessment: Dict[str, Any]) -> None:
        """填充问题和建议信息"""
        report['issues'] = health_assessment.get('issues', [])
        report['recommendations'] = health_assessment.get('recommendations', [])

    def _populate_metadata(self, report: Dict[str, Any], health_assessment: Dict[str, Any]) -> None:
        """填充元数据信息"""
        report['metadata'] = {
            'thresholds': health_assessment.get('thresholds', {}),
            'evaluation_timestamp': health_assessment.get('evaluation_timestamp'),
            'report_version': '1.0'
        }

    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """生成错误报告"""
        return {
            'report_type': 'error_report',
            'generated_at': datetime.now().isoformat(),
            'error': error_message,
            'status': 'error'
        }


class MetricsFormatter:
    """指标格式化器"""
    
    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
    
    def format_metrics(self, metrics: Optional[PerformanceMetrics]) -> Dict[str, Any]:
        """格式化性能指标"""
        if not metrics:
            return {
                'cpu_percent': None,
                'memory_percent': None,
                'disk_usage': None,
                'network_io': {},
                'process_count': None,
                'thread_count': None,
                'timestamp': None
            }

        return {
            'cpu_percent': metrics.cpu_percent,
            'memory_percent': metrics.memory_percent,
            'disk_usage': metrics.disk_usage,
            'network_io': dict(metrics.network_io),
            'process_count': metrics.process_count,
            'thread_count': metrics.thread_count,
            'timestamp': metrics.timestamp
        }
