
import threading

from .event_bus import EventBus, create_resource_event
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .unified_resource_interfaces import IResourceProvider
from typing import Dict, List, Optional, Any
"""
资源提供者注册表

Phase 3: 质量提升 - 文件拆分优化

管理资源提供者的注册、注销和查询功能。
"""


class ResourceProviderRegistry:
    """资源提供者注册表"""

    def __init__(self, event_bus: Optional[EventBus] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.event_bus = event_bus
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

        # 提供者存储
        self._providers: Dict[str, IResourceProvider] = {}
        self._lock = threading.RLock()

    def register_provider(self, provider: IResourceProvider) -> bool:
        """注册资源提供者"""
        with self._lock:
            try:
                resource_type = provider.resource_type

                if resource_type in self._providers:
                    raise ValueError(f"资源类型 '{resource_type}' 的提供者已存在")

                self._providers[resource_type] = provider

                # 注册到依赖注入容器（如果有的话）
                # 这里可以扩展依赖注入逻辑

                self.logger.log_info(f"资源提供者已注册: {resource_type}")

                # 发布事件
                if self.event_bus:
                    self.event_bus.publish(create_resource_event(
                        resource_type=resource_type,
                        resource_id="",
                        action="provider_registered",
                        source="ResourceProviderRegistry",
                        provider_type=type(provider).__name__
                    ))

                return True

            except Exception as e:
                self.error_handler.handle_error(e, {
                    "context": "注册资源提供者失败",
                    "provider_type": type(provider).__name__
                })
                return False

    def unregister_provider(self, resource_type: str) -> bool:
        """注销资源提供者"""
        with self._lock:
            try:
                if resource_type not in self._providers:
                    return False

                provider = self._providers[resource_type]
                del self._providers[resource_type]

                # 从依赖注入容器中移除（如果有的话）
                # 这里可以扩展依赖注入逻辑

                self.logger.log_info(f"资源提供者已注销: {resource_type}")

                # 发布事件
                if self.event_bus:
                    self.event_bus.publish(create_resource_event(
                        resource_type=resource_type,
                        resource_id="",
                        action="provider_unregistered",
                        source="ResourceProviderRegistry",
                        provider_type=type(provider).__name__
                    ))

                return True

            except Exception as e:
                self.error_handler.handle_error(e, {
                    "context": "注销资源提供者失败",
                    "resource_type": resource_type
                })
                return False

    def get_provider(self, resource_type: str) -> Optional[IResourceProvider]:
        """获取指定类型的资源提供者"""
        with self._lock:
            return self._providers.get(resource_type)

    def get_providers(self) -> List[IResourceProvider]:
        """获取所有提供者"""
        with self._lock:
            return list(self._providers.values())

    def get_provider_types(self) -> List[str]:
        """获取所有提供者类型"""
        with self._lock:
            return list(self._providers.keys())

    def has_provider(self, resource_type: str) -> bool:
        """检查是否有指定类型的提供者"""
        with self._lock:
            return resource_type in self._providers

    def get_provider_count(self) -> int:
        """获取提供者数量"""
        with self._lock:
            return len(self._providers)

    def get_provider_status(self, resource_type: str) -> Optional[Dict[str, Any]]:
        """获取提供者状态"""
        provider = self.get_provider(resource_type)
        if not provider:
            return None

        try:
            available_resources = provider.get_available_resources()
            return {
                "resource_type": resource_type,
                "available_count": len(available_resources),
                "total_capacity": sum(r.capacity.get("total", 0) for r in available_resources if r.capacity),
                "provider_type": type(provider).__name__,
                "status": "healthy"
            }
        except Exception as e:
            self.error_handler.handle_error(e, {
                "context": "获取提供者状态失败",
                "resource_type": resource_type
            })
            return {
                "resource_type": resource_type,
                "status": "error",
                "error": str(e)
            }

    def get_all_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有提供者状态"""
        status = {}
        for resource_type in self.get_provider_types():
            provider_status = self.get_provider_status(resource_type)
            if provider_status:
                status[resource_type] = provider_status
        return status

    def clear(self):
        """清空所有提供者"""
        with self._lock:
            provider_types = list(self._providers.keys())
            self._providers.clear()

            self.logger.log_info(f"已清空所有资源提供者，共 {len(provider_types)} 个")

            # 发布批量注销事件
            if self.event_bus:
                for resource_type in provider_types:
                    self.event_bus.publish(create_resource_event(
                        resource_type=resource_type,
                        resource_id="",
                        action="provider_bulk_unregistered",
                        source="ResourceProviderRegistry"
                    ))

    def get_provider_info(self, resource_type: str) -> Optional[Dict[str, Any]]:
        """获取提供者信息"""
        provider = self.get_provider(resource_type)
        if not provider:
            return None
        
        return {
            "resource_type": resource_type,
            "provider_type": type(provider).__name__,
            "status": "available"
        }

    def update_provider_health(self, resource_type: str, health_status: str) -> bool:
        """更新提供者健康状态"""
        with self._lock:
            if resource_type not in self._providers:
                return False
            
            # 记录健康状态更新
            self.logger.log_info(f"提供者 {resource_type} 健康状态更新为: {health_status}")
            return True