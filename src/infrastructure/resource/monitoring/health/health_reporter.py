

from ..alert_dataclasses import PerformanceMetrics
from ..shared_interfaces import ILogger, StandardLogger
from .health_report_builder import HealthReportBuilder
from .trend_analyzer import TrendAnalyzer
from .component_report_manager import ComponentReportManager
from datetime import datetime
from typing import Dict, List, Optional, Any
"""
健康报告器

职责：专门负责生成和格式化健康报告
"""


class HealthReporter:
    """
    健康报告器

    职责：生成结构化的健康报告，格式化健康数据输出
    """

    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
        
        # 使用专门的管理器类
        self.report_builder = HealthReportBuilder(logger)
        self.trend_analyzer = TrendAnalyzer(logger)
        self.component_manager = ComponentReportManager(logger)

    def generate_health_report(self, health_assessment: Dict[str, Any],
                               metrics: Optional[PerformanceMetrics] = None,
                               alert_stats: Optional[Dict[str, Any]] = None,
                               test_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成完整的健康报告"""
        return self.report_builder.build_health_report(
            health_assessment, metrics, alert_stats, test_stats
        )

    # 这些方法已移到 HealthReportBuilder 类中

    def generate_trend_report(self, health_history: List[Dict[str, Any]],
                              hours: int = 24) -> Dict[str, Any]:
        """生成健康趋势报告"""
        try:
            if not health_history:
                return self._generate_empty_trend_report()

            # 使用趋势分析器
            trend_analysis = self.trend_analyzer.analyze_health_trends(health_history)
            time_range = self.trend_analyzer.get_time_range(health_history)
            average_score = self.trend_analyzer.calculate_average_score(health_history)
            key_metrics_trends = self.trend_analyzer.analyze_key_metrics_trends(health_history)

            report = {
                'report_type': 'health_trend_report',
                'generated_at': datetime.now().isoformat(),
                'analysis_period_hours': hours,
                'trend_analysis': trend_analysis,
                'historical_summary': {
                    'total_records': len(health_history),
                    'time_range': time_range,
                    'average_score': average_score
                },
                'key_metrics_trends': key_metrics_trends
            }

            return report

        except Exception as e:
            self.logger.log_error(f"生成趋势报告失败: {e}")
            return self._generate_error_report(str(e))

    def generate_component_health_report(self, component_statuses: Dict[str, Any]) -> Dict[str, Any]:
        """生成组件健康报告"""
        return self.component_manager.generate_component_health_report(component_statuses)

    # 这些方法已移到对应的专门类中：
    # - ComponentReportManager: _initialize_component_report, _populate_component_details 等
    # - TrendAnalyzer: _analyze_health_trends, _get_time_range 等

    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """生成错误报告"""
        return {
            'report_type': 'error_report',
            'generated_at': datetime.now().isoformat(),
            'error': error_message,
            'status': 'error'
        }

    def _generate_empty_trend_report(self) -> Dict[str, Any]:
        """生成空的趋势报告"""
        return {
            'report_type': 'health_trend_report',
            'generated_at': datetime.now().isoformat(),
            'analysis_period_hours': 0,
            'trend_analysis': {'trend': 'no_data', 'direction': 'unknown', 'confidence': 0.0},
            'historical_summary': {'total_records': 0, 'time_range': {'start': None, 'end': None, 'duration_hours': 0}, 'average_score': 0.0}
        }
