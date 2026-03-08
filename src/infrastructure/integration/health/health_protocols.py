"""
健康检查协议定义

定义健康层相关的Protocol接口。
"""

from typing import Dict, Any, List, Protocol

from .health_models import HealthCheckType, HealthCheckResult, AsyncHealthCheckConfig


class HealthInfrastructureManager(Protocol):
    """健康基础设施管理器协议"""
    def initialize_infrastructure(self) -> Dict[str, Any]: ...
    def get_infrastructure_services(self) -> Dict[str, Any]: ...


class HealthCheckExecutor(Protocol):
    """健康检查执行器协议"""
    async def perform_health_check(self, service_name: str, check_type: HealthCheckType) -> HealthCheckResult: ...
    async def perform_batch_health_check(self, service_names: List[str], check_type: HealthCheckType) -> Dict[str, HealthCheckResult]: ...


class HealthConfigManager(Protocol):
    """健康配置管理器协议"""
    def get_config(self) -> AsyncHealthCheckConfig: ...
    def update_config(self, config: Dict[str, Any]): ...

