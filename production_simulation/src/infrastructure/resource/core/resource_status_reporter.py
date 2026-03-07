"""
resource_status_reporter 模块

提供 resource_status_reporter 相关功能和接口。
"""

import json

import yaml

from .resource_allocation_manager import ResourceAllocationManager
from .resource_consumer_registry import ResourceConsumerRegistry
from .resource_provider_registry import ResourceProviderRegistry
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from datetime import datetime
from typing import Dict, List, Optional, Any
"""
资源状态报告器

Phase 3: 质量提升 - 文件拆分优化

负责生成和报告资源系统的整体状态信息。
"""


class ResourceStatusReporter:
    """资源状态报告器"""

    def __init__(self, provider_registry: Optional[ResourceProviderRegistry] = None,
                 consumer_registry: Optional[ResourceConsumerRegistry] = None,
                 allocation_manager: Optional[ResourceAllocationManager] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.provider_registry = provider_registry
        self.consumer_registry = consumer_registry
        self.allocation_manager = allocation_manager
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

    def get_resource_status(self) -> Dict[str, Any]:
        """获取资源状态"""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "summary": self._get_summary_status(),
                "providers": self._get_provider_status(),
                "consumers": self._get_consumer_status(),
                "allocations": self._get_allocation_status(),
                "health": self._get_health_status()
            }

            return status

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取资源状态失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _get_summary_status(self) -> Dict[str, Any]:
        """获取汇总状态"""
        summary = {
            "providers_count": 0,
            "consumers_count": 0,
            "active_allocations": 0,
            "pending_requests": 0,
            "total_capacity": {},
            "total_usage": {}
        }

        if self.provider_registry:
            summary["providers_count"] = self.provider_registry.get_provider_count()

        if self.consumer_registry:
            summary["consumers_count"] = self.consumer_registry.get_consumer_count()

        if self.allocation_manager:
            summary["active_allocations"] = self.allocation_manager.get_allocation_count()
            summary["pending_requests"] = self.allocation_manager.get_request_count()

        # 计算总容量和使用情况
        if self.provider_registry:
            provider_status = self.provider_registry.get_all_provider_status()
            for resource_type, status in provider_status.items():
                if status.get("status") == "healthy":
                    total_capacity = status.get("total_capacity", 0)
                    if total_capacity > 0:
                        summary["total_capacity"][resource_type] = total_capacity

        return summary

    def _get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """获取提供者状态"""
        if self.provider_registry:
            return self.provider_registry.get_all_provider_status()
        return {}

    def _get_consumer_status(self) -> Dict[str, Dict[str, Any]]:
        """获取消费者状态"""
        if self.consumer_registry:
            return self.consumer_registry.get_all_consumer_info()
        return {}

    def _get_allocation_status(self) -> Dict[str, Any]:
        """获取分配状态"""
        if self.allocation_manager:
            return {
                "summary": self.allocation_manager.get_allocation_summary(),
                "active_count": self.allocation_manager.get_allocation_count(),
                "pending_count": self.allocation_manager.get_request_count()
            }
        return {"summary": {}, "active_count": 0, "pending_count": 0}

    def _get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        try:
            health_score = 1.0
            issues = []

            # 检查提供者健康
            if self.provider_registry:
                provider_status = self.provider_registry.get_all_provider_status()
                error_providers = sum(1 for status in provider_status.values()
                                      if status.get("status") == "error")
                if error_providers > 0:
                    health_score -= min(0.3, error_providers * 0.1)
                    issues.append(f"{error_providers} 个资源提供者异常")

            # 检查分配状态
            if self.allocation_manager:
                active_allocations = self.allocation_manager.get_allocation_count()
                if active_allocations > 100:  # 假设100是个合理的上限
                    health_score -= 0.1
                    issues.append(f"活跃分配过多: {active_allocations}")

            # 确定健康状态
            if health_score >= 0.9:
                status = "excellent"
            elif health_score >= 0.8:
                status = "good"
            elif health_score >= 0.7:
                status = "fair"
            elif health_score >= 0.6:
                status = "poor"
            else:
                status = "critical"

            return {
                "health_score": max(0.0, health_score),
                "health_status": status,
                "issues": issues
            }

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取健康状态失败"})
            return {
                "health_score": 0.0,
                "health_status": "unknown",
                "error": str(e)
            }

    def get_detailed_report(self) -> Dict[str, Any]:
        """获取详细报告"""
        try:
            report = self.get_resource_status()

            # 添加更多详细信息
            report["performance"] = self._get_performance_metrics()
            report["trends"] = self._get_trend_analysis()
            report["recommendations"] = self._get_recommendations()

            return report

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "生成详细报告失败"})
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "response_time_avg": 0.0,  # 可以扩展实现
            "throughput": 0.0,
            "error_rate": 0.0,
            "resource_utilization": {}
        }

    def _get_trend_analysis(self) -> Dict[str, Any]:
        """获取趋势分析"""
        return {
            "allocation_trend": "stable",
            "usage_trend": "stable",
            "error_trend": "stable",
            "predictions": {}
        }

    def _get_recommendations(self) -> List[str]:
        """获取建议"""
        recommendations = []

        try:
            status = self.get_resource_status()
            health = status.get("health", {})

            if health.get("health_score", 1.0) < 0.8:
                recommendations.append("检查系统健康问题并及时处理")

            if status.get("summary", {}).get("active_allocations", 0) > 50:
                recommendations.append("考虑优化资源分配策略")

            if status.get("summary", {}).get("pending_requests", 0) > 10:
                recommendations.append("检查资源请求积压原因")

        except Exception:
            recommendations.append("建议定期监控系统状态")

        return recommendations

    def export_report(self, format: str = "json") -> str:
        """导出报告"""
        try:
            report = self.get_detailed_report()

            if format.lower() == "json":
                return json.dumps(report, indent=2, ensure_ascii=False)
            elif format.lower() == "yaml":
                return yaml.dump(report, default_flow_style=False)
            else:
                raise ValueError(f"不支持的导出格式: {format}")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "导出报告失败"})
            return f"导出失败: {str(e)}"

    def export_status_to_json(self) -> str:
        """导出状态为JSON格式"""
        try:
            status = self.get_resource_status()
            return json.dumps(status, indent=2, ensure_ascii=False)
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "导出JSON状态失败"})
            return f"导出失败: {str(e)}"

    def export_status_to_yaml(self) -> str:
        """导出状态为YAML格式"""
        try:
            status = self.get_resource_status()
            return yaml.dump(status, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "导出YAML状态失败"})
            return f"导出失败: {str(e)}"

    def generate_detailed_report(self) -> Dict[str, Any]:
        """生成详细报告"""
        try:
            report = self.get_detailed_report()
            report["report_type"] = "detailed"
            return report
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "生成详细报告失败"})
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "report_type": "detailed"
            }

    def get_provider_status_summary(self) -> Dict[str, Any]:
        """获取提供者状态摘要"""
        try:
            provider_status = self._get_provider_status()
            return {
                "total_providers": len(provider_status),
                "healthy_providers": sum(1 for status in provider_status.values() 
                                       if status.get("status") == "healthy"),
                "error_providers": sum(1 for status in provider_status.values() 
                                     if status.get("status") == "error"),
                "providers": provider_status
            }
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取提供者状态摘要失败"})
            return {"error": str(e)}

    def get_consumer_status_summary(self) -> Dict[str, Any]:
        """获取消费者状态摘要"""
        try:
            consumer_status = self._get_consumer_status()
            return {
                "total_consumers": len(consumer_status),
                "consumers": consumer_status
            }
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取消费者状态摘要失败"})
            return {"error": str(e)}

    def generate_summary_report(self) -> Dict[str, Any]:
        """生成摘要报告"""
        try:
            status = self.get_resource_status()
            summary = status.get("summary", {})
            
            return {
                "report_type": "summary",
                "timestamp": status.get("timestamp"),
                "providers_count": summary.get("providers_count", 0),
                "consumers_count": summary.get("consumers_count", 0),
                "active_allocations": summary.get("active_allocations", 0),
                "health_status": status.get("health", {}).get("health_status", "unknown")
            }
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "生成摘要报告失败"})
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "report_type": "summary"
            }