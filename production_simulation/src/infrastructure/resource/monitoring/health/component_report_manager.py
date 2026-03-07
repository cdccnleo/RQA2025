"""
组件报告管理器

负责生成和管理组件健康报告。
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from ..shared_interfaces import ILogger, StandardLogger


class ComponentReportManager:
    """组件报告管理器"""
    
    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
    
    def generate_component_health_report(self, component_statuses: Dict[str, Any]) -> Dict[str, Any]:
        """生成组件健康报告"""
        try:
            report = self._initialize_component_report(component_statuses)
            self._populate_component_details(report, component_statuses)
            return report

        except Exception as e:
            self.logger.log_error(f"生成组件健康报告失败: {e}")
            return self._generate_error_report(str(e))

    def _initialize_component_report(self, component_statuses: Dict[str, Any]) -> Dict[str, Any]:
        """初始化组件报告基础结构"""
        return {
            'report_type': 'component_health_report',
            'generated_at': datetime.now().isoformat(),
            'components': {},
            'summary': {
                'total_components': len(component_statuses),
                'healthy_components': 0,
                'warning_components': 0,
                'critical_components': 0,
                'unknown_components': 0
            }
        }

    def _populate_component_details(self, report: Dict[str, Any], component_statuses: Dict[str, Any]) -> None:
        """填充组件详细信息"""
        for component_name, status in component_statuses.items():
            health_status = status.get('status', 'unknown')
            
            # 更新汇总统计
            self._update_component_summary(report['summary'], health_status)
            
            # 添加组件详情
            report['components'][component_name] = self._create_component_detail(status, health_status)

    def _update_component_summary(self, summary: Dict[str, int], health_status: str) -> None:
        """更新组件汇总统计"""
        status_count_map = {
            'healthy': 'healthy_components',
            'warning': 'warning_components', 
            'critical': 'critical_components'
        }
        
        count_key = status_count_map.get(health_status, 'unknown_components')
        summary[count_key] += 1

    def _create_component_detail(self, status: Dict[str, Any], health_status: str) -> Dict[str, Any]:
        """创建组件详情信息"""
        return {
            'status': health_status,
            'last_check': status.get('last_check'),
            'response_time': status.get('response_time'),
            'details': status.get('details', {}),
            'issues': status.get('issues', [])
        }

    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """生成错误报告"""
        return {
            'report_type': 'error_report',
            'generated_at': datetime.now().isoformat(),
            'error': error_message,
            'status': 'error'
        }
