
import time as time_module
import threading
import time

from .event_bus import EventBus, create_resource_event
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .unified_resource_interfaces import (
    IResourceProvider, IResourceConsumer, ResourceAllocation,
    ResourceRequest, ResourceError, ResourceNotFoundError
)
from typing import Dict, List, Optional, Any

"""
资源分配管理器

Phase 3: 质量提升 - 文件拆分优化

管理资源的分配、释放和跟踪功能。
"""


class ResourceAllocationManager:
    """资源分配管理器"""

    def __init__(self, provider_registry=None, event_bus: Optional[EventBus] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.provider_registry = provider_registry
        self.event_bus = event_bus
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()

        # 分配和请求跟踪
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._requests: Dict[str, ResourceRequest] = {}
        self._lock = threading.RLock()

    def _get_resource_type(self, allocation) -> str:
        """从ResourceAllocation对象获取资源类型"""
        resource_type = getattr(allocation, 'resource_type', None)
        if resource_type is None:
            # 从resource_id推断资源类型，例如"cpu_res1" -> "cpu"
            resource_type = allocation.resource_id.split('_')[0] if '_' in allocation.resource_id else allocation.resource_id
        return resource_type

    def request_resource(self, consumer_id: str, resource_type: str,
                         requirements: Dict[str, Any], priority: int = 1) -> Optional[str]:
        """请求资源"""
        try:
            # 获取并验证资源提供者
            provider = self._validate_and_get_provider(resource_type)

            # 创建资源请求
            request = self._create_resource_request(
                consumer_id, resource_type, requirements, priority)

            # 存储请求
            self._store_request(request)

            # 尝试分配资源
            allocation = self._attempt_allocation(provider, request)

            if allocation:
                return self._handle_successful_allocation(allocation, consumer_id)
            else:
                return self._handle_failed_allocation(resource_type, consumer_id, requirements)

        except Exception as e:
            self.error_handler.handle_error(e, {
                "context": "请求资源失败",
                "consumer_id": consumer_id,
                "resource_type": resource_type
            })
            return None

    def _validate_and_get_provider(self, resource_type: str):
        """验证并获取资源提供者"""
        if not self.provider_registry or not self.provider_registry.has_provider(resource_type):
            raise ResourceNotFoundError(f"资源类型 '{resource_type}' 没有可用的提供者")
        return self.provider_registry.get_provider(resource_type)

    def _create_resource_request(self, consumer_id: str, resource_type: str,
                                 requirements: Dict[str, Any], priority: int) -> ResourceRequest:
        """创建资源请求对象"""
        request_id = f"req_{int(time.time() * 1000)}_{consumer_id}"
        return ResourceRequest(
            request_id=request_id,
            resource_type=resource_type,
            requester_id=consumer_id,
            requirements=requirements,
            priority=priority
        )

    def _store_request(self, request: ResourceRequest) -> None:
        """存储资源请求"""
        with self._lock:
            self._requests[request.request_id] = request

    def _attempt_allocation(self, provider, request: ResourceRequest):
        """尝试分配资源"""
        return provider.allocate_resource(request)

    def _handle_successful_allocation(self, allocation, consumer_id: str) -> str:
        """处理成功的资源分配"""
        # 存储分配
        with self._lock:
            self._allocations[allocation.allocation_id] = allocation

        # 发布资源分配事件
        if self.event_bus:
            resource_type = self._get_resource_type(allocation)
            self.event_bus.publish(create_resource_event(
                resource_type=resource_type,
                resource_id=allocation.resource_id,
                action="allocated",
                source="ResourceAllocationManager",
                allocation_id=allocation.allocation_id,
                consumer_id=consumer_id
            ))

        self.logger.log_info(f"资源已分配: {allocation.allocation_id}")
        return allocation.allocation_id

    def _handle_failed_allocation(self, resource_type: str, consumer_id: str,
                                  requirements: Dict[str, Any]) -> None:
        """处理失败的资源分配"""
        # 发布资源分配失败事件
        if self.event_bus:
            self.event_bus.publish(create_resource_event(
                resource_type=resource_type,
                resource_id="",
                action="allocation_failed",
                source="ResourceAllocationManager",
                consumer_id=consumer_id,
                requirements=requirements
            ))

        self.logger.log_warning(f"资源分配失败: {resource_type}")
        return None

    def release_resource(self, allocation_id: str) -> bool:
        """释放资源"""
        try:
            with self._lock:
                if allocation_id not in self._allocations:
                    return False

                allocation = self._allocations[allocation_id]
                del self._allocations[allocation_id]

            # 获取资源提供者
            resource_type = self._get_resource_type(allocation)
            if not self.provider_registry or not self.provider_registry.has_provider(resource_type):
                raise ResourceNotFoundError(f"资源类型 '{resource_type}' 没有可用的提供者")

            provider = self.provider_registry.get_provider(resource_type)

            # 释放资源
            success = provider.release_resource(allocation_id)

            if success:
                # 发布资源释放事件
                if self.event_bus:
                    self.event_bus.publish(create_resource_event(
                        resource_type=resource_type,
                        resource_id=allocation.resource_id,
                        action="released",
                        source="ResourceAllocationManager",
                        allocation_id=allocation_id
                    ))

                self.logger.log_info(f"资源已释放: {allocation_id}")
            else:
                self.logger.log_warning(f"资源释放失败: {allocation_id}")

            return success

        except Exception as e:
            self.error_handler.handle_error(e, {
                "context": "释放资源失败",
                "allocation_id": allocation_id
            })
            return False

    def get_allocation(self, allocation_id: str) -> Optional[ResourceAllocation]:
        """获取分配信息"""
        with self._lock:
            return self._allocations.get(allocation_id)

    def get_allocations_for_consumer(self, consumer_id: str) -> List[ResourceAllocation]:
        """获取消费者的所有分配"""
        with self._lock:
            return [alloc for alloc in self._allocations.values()
                    if alloc.request_id and alloc.request_id.endswith(f"_{consumer_id}")]

    def get_allocations_for_resource_type(self, resource_type: str) -> List[ResourceAllocation]:
        """获取指定资源类型的分配"""
        with self._lock:
            return [alloc for alloc in self._allocations.values()
                    if alloc.resource_type == resource_type]

    def get_allocation_count(self) -> int:
        """获取活跃分配数量"""
        with self._lock:
            return len(self._allocations)

    def get_request_count(self) -> int:
        """获取待处理请求数量"""
        with self._lock:
            return len(self._requests)

    def get_request(self, request_id: str) -> Optional[ResourceRequest]:
        """获取请求信息"""
        with self._lock:
            return self._requests.get(request_id)

    def get_allocation_summary(self) -> Dict[str, Any]:
        """获取分配汇总信息"""
        summary = {
            "total_allocations": 0,
            "by_resource_type": {},
            "by_consumer": {}
        }
        
        with self._lock:
            for allocation in self._allocations.values():
                # 增加总分配数
                summary["total_allocations"] += 1
                
                # 获取资源类型
                resource_type = self._get_resource_type(allocation)
                
                # 按资源类型分组
                if resource_type not in summary["by_resource_type"]:
                    summary["by_resource_type"][resource_type] = 0
                summary["by_resource_type"][resource_type] += 1
                
                # 按消费者分组（从request_id推断）
                consumer_id = allocation.request_id.replace("req", "consumer") if allocation.request_id.startswith("req") else allocation.request_id
                if consumer_id not in summary["by_consumer"]:
                    summary["by_consumer"][consumer_id] = 0
                summary["by_consumer"][consumer_id] += 1

        return summary

    def get_active_allocations(self) -> List[ResourceAllocation]:
        """获取所有活跃分配"""
        with self._lock:
            return list(self._allocations.values())

    def clear_expired_allocations(self, max_age_seconds: int = 3600) -> int:
        """清理过期的分配（基于时间戳）"""
        current_time = time_module.time()
        expired_ids = []

        with self._lock:
            for alloc_id, allocation in self._allocations.items():
                # 这里可以根据allocation的过期时间或其他逻辑判断是否过期
                # 暂时使用简单的启发式方法
                if hasattr(allocation, 'allocated_at') and allocation.allocated_at:
                    age = current_time - allocation.allocated_at.timestamp()
                    if age > max_age_seconds:
                        expired_ids.append(alloc_id)

            # 清理过期分配
            for alloc_id in expired_ids:
                del self._allocations[alloc_id]

        if expired_ids:
            self.logger.log_info(f"已清理 {len(expired_ids)} 个过期分配")

        return len(expired_ids)

    def force_release_allocation(self, allocation_id: str) -> bool:
        """强制释放分配（不调用provider.release_resource）"""
        with self._lock:
            if allocation_id in self._allocations:
                allocation = self._allocations[allocation_id]
                del self._allocations[allocation_id]

                # 发布强制释放事件
                if self.event_bus:
                    resource_type = self._get_resource_type(allocation)
                    self.event_bus.publish(create_resource_event(
                        resource_type=resource_type,
                        resource_id=allocation.resource_id,
                        action="force_released",
                        source="ResourceAllocationManager",
                        allocation_id=allocation_id
                    ))

                self.logger.log_warning(f"强制释放分配: {allocation_id}")
                return True

        return False

    def clear_all_allocations(self):
        """清空所有分配"""
        with self._lock:
            allocation_count = len(self._allocations)
            allocation_ids = list(self._allocations.keys())
            self._allocations.clear()

            self.logger.log_info(f"已清空所有分配，共 {allocation_count} 个")

            # 发布批量释放事件
            if self.event_bus:
                for alloc_id in allocation_ids:
                    self.event_bus.publish(create_resource_event(
                        resource_type="unknown",
                        resource_id="",
                        action="bulk_released",
                        source="ResourceAllocationManager",
                        allocation_id=alloc_id
                    ))
