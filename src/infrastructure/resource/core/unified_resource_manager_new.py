
from .dependency_container import DependencyContainer
from .event_bus import EventBus, create_system_event
from .resource_allocation_manager import ResourceAllocationManager
from .resource_consumer_registry import ResourceConsumerRegistry
from .resource_provider_registry import ResourceProviderRegistry
from .resource_status_reporter import ResourceStatusReporter
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .unified_resource_interfaces import IResourceManager, IResourceProvider, IResourceConsumer
from datetime import datetime
from typing import Dict, List, Optional, Any
"""
统一资源管理器 (重构后版本)

Phase 3: 质量提升 - 文件拆分优化

简化的资源管理器协调器，使用专用管理器组件。
"""


class NewUnifiedResourceManager(IResourceManager):
    """统一资源管理器 (重构后版本)

    使用专用管理器组件的协调器，提供简洁的资源管理接口。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        # 初始化基础组件
        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()
        self.config = config or {}

        # 初始化核心组件
        self.event_bus = EventBus(self.logger)
        self.container = DependencyContainer(self.logger)

        # 初始化专用管理器
        self.provider_registry = ResourceProviderRegistry(
            self.event_bus, self.logger, self.error_handler
        )
        self.consumer_registry = ResourceConsumerRegistry(
            self.logger, self.error_handler
        )
        self.allocation_manager = ResourceAllocationManager(
            self.provider_registry, self.event_bus, self.logger, self.error_handler
        )
        self.status_reporter = ResourceStatusReporter(
            self.provider_registry, self.consumer_registry,
            self.allocation_manager, self.logger, self.error_handler
        )

        # 启动状态
        self._running = False
        self._start_time = None

        self.logger.log_info("统一资源管理器已初始化 (重构后版本)")

    def start(self):
        """启动资源管理器"""
        if self._running:
            return

        try:
            self._running = True
            self._start_time = datetime.now()

            # 启动事件总线
            self.event_bus.start()

            # 发布系统启动事件
            self.event_bus.publish(create_system_event(
                severity="info",
                component="UnifiedResourceManager",
                message="统一资源管理器已启动",
                source="UnifiedResourceManager"
            ))

            self.logger.log_info("统一资源管理器已启动")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "启动资源管理器失败"})
            raise

    def stop(self):
        """停止资源管理器"""
        if not self._running:
            return

        try:
            self._running = False

            # 停止事件总线
            self.event_bus.stop()

            # 发布系统停止事件
            self.event_bus.publish(create_system_event(
                severity="info",
                component="UnifiedResourceManager",
                message="统一资源管理器已停止",
                source="UnifiedResourceManager"
            ))

            self.logger.log_info("统一资源管理器已停止")

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "停止资源管理器失败"})
            raise

    # 提供者管理接口
    def register_provider(self, provider: IResourceProvider) -> bool:
        """注册资源提供者"""
        return self.provider_registry.register_provider(provider)

    def unregister_provider(self, resource_type: str) -> bool:
        """注销资源提供者"""
        return self.provider_registry.unregister_provider(resource_type)

    def get_providers(self) -> List[IResourceProvider]:
        """获取所有提供者"""
        return self.provider_registry.get_providers()

    # 消费者管理接口
    def register_consumer(self, consumer: IResourceConsumer) -> bool:
        """注册资源消费者"""
        return self.consumer_registry.register_consumer(consumer)

    def unregister_consumer(self, consumer_id: str) -> bool:
        """注销资源消费者"""
        return self.consumer_registry.unregister_consumer(consumer_id)

    def get_consumers(self) -> List[IResourceConsumer]:
        """获取所有消费者"""
        return self.consumer_registry.get_consumers()

    # 资源分配接口
    def request_resource(self, consumer_id: str, resource_type: str,
                         requirements: Dict[str, Any], priority: int = 1) -> Optional[str]:
        """请求资源"""
        return self.allocation_manager.request_resource(
            consumer_id, resource_type, requirements, priority
        )

    def release_resource(self, allocation_id: str) -> bool:
        """释放资源"""
        return self.allocation_manager.release_resource(allocation_id)

    # 状态报告接口
    def get_resource_status(self) -> Dict[str, Any]:
        """获取资源状态"""
        return self.status_reporter.get_resource_status()

    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        return self.status_reporter.get_detailed_report()

    def optimize_resources(self) -> Dict[str, Any]:
        """优化资源分配"""
        # 这里可以实现资源优化逻辑
        return {
            "status": "completed",
            "optimizations_applied": [],
            "message": "资源优化功能待实现"
        }

    # 组件访问接口
    def get_event_bus(self) -> EventBus:
        """获取事件总线"""
        return self.event_bus

    def get_container(self) -> DependencyContainer:
        """获取依赖注入容器"""
        return self.container
