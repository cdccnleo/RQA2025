"""
服务注册表

管理服务的注册、查找和注销功能。
"""

import logging
from typing import Dict, List, Optional

from .integration_models import ServiceEndpoint

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """服务注册表 - 职责：管理服务注册和查找"""

    def __init__(self):
        self._service_registry: Dict[str, ServiceEndpoint] = {}

    def register_service(self, service_name: str, endpoint: ServiceEndpoint) -> None:
        """注册服务"""
        self._service_registry[service_name] = endpoint
        logger.info(f"服务 '{service_name}' 已注册: {endpoint.endpoint_url}")

    def unregister_service(self, service_name: str) -> bool:
        """注销服务"""
        if service_name in self._service_registry:
            del self._service_registry[service_name]
            logger.info(f"服务 '{service_name}' 已注销")
            return True
        return False

    def get_service(self, service_name: str) -> Optional[ServiceEndpoint]:
        """获取服务"""
        return self._service_registry.get(service_name)

    def list_services(self) -> List[str]:
        """列出所有服务"""
        return list(self._service_registry.keys())

    def get_service_count(self) -> int:
        """获取服务数量"""
        return len(self._service_registry)

