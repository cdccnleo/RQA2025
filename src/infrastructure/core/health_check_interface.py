
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
"""
RQA2025 基础设施层统一健康检查接口

定义了基础设施层所有服务的标准健康检查接口协议
确保所有服务都提供一致的健康检查能力

作者: 架构团队
创建时间: 2025年9月29日
"""


class HealthCheckResult:
    """健康检查结果数据结构"""

    def __init__(self,
                 service_name: str,
                 healthy: bool,
                 status: str,
                 timestamp: Optional[datetime] = None,
                 version: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 issues: Optional[list] = None,
                 recommendations: Optional[list] = None):
        self.service_name = service_name
        self.healthy = healthy
        self.status = status
        self.timestamp = timestamp or datetime.now()
        self.version = version or "1.0.0"
        self.details = details or {}
        self.issues = issues or []
        self.recommendations = recommendations or []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'service': self.service_name,
            'healthy': self.healthy,
            'status': self.status,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version,
            'details': self.details,
            'issues': self.issues,
            'recommendations': self.recommendations
        }


class HealthCheckInterface(ABC):
    """基础设施层统一健康检查接口协议

    所有基础设施服务都必须实现此接口，
    确保提供一致的健康检查能力
    """

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """执行健康检查

        Returns:
            Dict[str, Any]: 健康检查结果，包含以下字段:
            - service: 服务名称
            - healthy: 是否健康 (bool)
            - status: 状态字符串 ('healthy', 'unhealthy', 'error')
            - timestamp: 检查时间 (ISO格式字符串)
            - version: 服务版本
            - details: 详细状态信息 (可选)
            - issues: 发现的问题列表 (可选)
            - recommendations: 修复建议列表 (可选)
        """

    @property
    @abstractmethod
    def service_name(self) -> str:
        """服务名称"""

    @property
    @abstractmethod
    def service_version(self) -> str:
        """服务版本"""


class InfrastructureHealthChecker:
    """基础设施层统一健康检查管理器

    管理所有基础设施服务的健康检查，
    提供统一的健康状态查询接口
    """

    def __init__(self):
        self._services: Dict[str, HealthCheckInterface] = {}

    def register_service(self, service: HealthCheckInterface) -> None:
        """注册健康检查服务"""
        self._services[service.service_name] = service

    def unregister_service(self, service_name: str) -> None:
        """注销健康检查服务"""
        if service_name in self._services:
            del self._services[service_name]

    def check_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """检查单个服务的健康状态"""
        if service_name not in self._services:
            return None
        return self._services[service_name].health_check()

    def check_all_services(self) -> Dict[str, Any]:
        """检查所有服务的健康状态"""
        results = {}
        overall_healthy = True

        for service_name, service in self._services.items():
            try:
                result = service.health_check()
                results[service_name] = result
                if not result.get('healthy', False):
                    overall_healthy = False
            except Exception as e:
                # 服务健康检查失败
                results[service_name] = {
                    'service': service_name,
                    'healthy': False,
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'message': f'健康检查执行失败: {e}'
                }
                overall_healthy = False

        return {
            'overall_healthy': overall_healthy,
            'total_services': len(self._services),
            'healthy_services': sum(1 for r in results.values() if r.get('healthy', False)),
            'unhealthy_services': sum(1 for r in results.values() if not r.get('healthy', False)),
            'services': results,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'infrastructure_status': 'healthy' if overall_healthy else 'unhealthy',
                'last_check': datetime.now().isoformat()
            }
        }

    def get_service_list(self) -> list:
        """获取已注册的服务列表"""
        return list(self._services.keys())

    def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """获取服务信息"""
        if service_name not in self._services:
            return None

        service = self._services[service_name]
        return {
            'name': service.service_name,
            'version': service.service_version,
            'has_health_check': True
        }


# 全局健康检查管理器实例
infrastructure_health_checker = InfrastructureHealthChecker()


def get_infrastructure_health_checker() -> InfrastructureHealthChecker:
    """获取全局基础设施健康检查管理器"""
    return infrastructure_health_checker


def register_infrastructure_service(service: HealthCheckInterface) -> None:
    """注册基础设施服务到全局健康检查管理器"""
    infrastructure_health_checker.register_service(service)


def check_infrastructure_health() -> Dict[str, Any]:
    """检查整个基础设施层的健康状态"""
    return infrastructure_health_checker.check_all_services()
