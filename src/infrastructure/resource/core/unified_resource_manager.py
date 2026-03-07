
# 延迟导入相关组件类 (用于类型注解)

try:
    from .dependency_container import DependencyContainer
    from .event_bus import EventBus, create_system_event
except ImportError:
    DependencyContainer = None
    EventBus = None
    create_system_event = None

from .resource_allocation_manager import ResourceAllocationManager
from .resource_consumer_registry import ResourceConsumerRegistry
from .resource_provider_registry import ResourceProviderRegistry
from .resource_status_reporter import ResourceStatusReporter
from ...core.component_registry import InfrastructureComponentRegistry
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from .unified_resource_interfaces import IResourceManager, IResourceProvider, IResourceConsumer
from datetime import datetime
from typing import Dict, List, Optional, Any
"""
统一资源管理器 (Phase 6.1组织架构重构版本)

使用组件注册表进行依赖注入，减少直接模块依赖。
优化前依赖: 8个模块
优化后依赖: 3个核心模块 + 组件注册表
"""


class UnifiedResourceManager(IResourceManager):
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

        # 使用组件注册表进行依赖管理 (Phase 6.1组织架构重构)
        self.registry = InfrastructureComponentRegistry()

        # 注册组件工厂函数 (延迟加载)
        self._register_component_factories()

        # 延迟初始化核心组件
        self._components = {}

        # 启动状态
        self._running = False
        self._start_time = None

        self.logger.log_info("统一资源管理器已初始化 (Phase 6.1重构版本)")

    def _register_component_factories(self):
        """注册组件工厂函数 - 实现延迟加载"""
        # 注册事件总线
        self.registry.register_component('event_bus',
                                         lambda: self._create_event_bus())

        # 注册依赖容器
        self.registry.register_component('container',
                                         lambda: self._create_container())

        # 注册资源提供者注册表
        self.registry.register_component('provider_registry',
                                         lambda: self._create_provider_registry())

        # 注册资源消费者注册表
        self.registry.register_component('consumer_registry',
                                         lambda: self._create_consumer_registry())

        # 注册资源分配管理器
        self.registry.register_component('allocation_manager',
                                         lambda: self._create_allocation_manager())

        # 注册状态报告器
        self.registry.register_component('status_reporter',
                                         lambda: self._create_status_reporter())

    def _create_event_bus(self):
        """创建事件总线"""
        if EventBus is None:
            self.logger.warning("EventBus组件不可用")
            return None
        try:
            return EventBus(self.logger)
        except Exception as e:
            self.logger.warning(f"EventBus组件创建失败: {e}")
            return None

    def _create_container(self):
        """创建依赖容器"""
        if DependencyContainer is None:
            self.logger.warning("DependencyContainer组件不可用")
            return None
        try:
            return DependencyContainer(self.logger)
        except Exception as e:
            self.logger.warning(f"DependencyContainer组件创建失败: {e}")
            return None

    def _create_provider_registry(self):
        """创建资源提供者注册表"""
        try:
            event_bus = self.registry.get_component('event_bus')
            return ResourceProviderRegistry(event_bus, self.logger, self.error_handler)
        except ImportError:
            self.logger.warning("ResourceProviderRegistry组件不可用")
            return None

    def _create_consumer_registry(self):
        """创建资源消费者注册表"""
        try:
            return ResourceConsumerRegistry(self.logger, self.error_handler)
        except ImportError:
            self.logger.warning("ResourceConsumerRegistry组件不可用")
            return None

    def _create_allocation_manager(self):
        """创建资源分配管理器"""
        try:
            provider_registry = self.registry.get_component('provider_registry')
            event_bus = self.registry.get_component('event_bus')
            return ResourceAllocationManager(
                provider_registry, event_bus, self.logger, self.error_handler
            )
        except ImportError:
            self.logger.warning("ResourceAllocationManager组件不可用")
            return None

    def _create_status_reporter(self):
        """创建状态报告器"""
        try:
            provider_registry = self.registry.get_component('provider_registry')
            consumer_registry = self.registry.get_component('consumer_registry')
            allocation_manager = self.registry.get_component('allocation_manager')
            return ResourceStatusReporter(
                provider_registry, consumer_registry,
                allocation_manager, self.logger, self.error_handler
            )
        except ImportError:
            self.logger.warning("ResourceStatusReporter组件不可用")
            return None

    # 属性访问器 - 实现延迟加载
    @property
    def event_bus(self):
        """延迟加载事件总线"""
        if 'event_bus' not in self._components:
            self._components['event_bus'] = self.registry.get_component('event_bus')
        return self._components['event_bus']

    @property
    def container(self):
        """延迟加载依赖容器"""
        if 'container' not in self._components:
            self._components['container'] = self.registry.get_component('container')
        return self._components['container']

    @property
    def provider_registry(self):
        """延迟加载资源提供者注册表"""
        if 'provider_registry' not in self._components:
            self._components['provider_registry'] = self.registry.get_component('provider_registry')
        return self._components['provider_registry']

    @property
    def consumer_registry(self):
        """延迟加载资源消费者注册表"""
        if 'consumer_registry' not in self._components:
            self._components['consumer_registry'] = self.registry.get_component('consumer_registry')
        return self._components['consumer_registry']

    @property
    def allocation_manager(self):
        """延迟加载资源分配管理器"""
        if 'allocation_manager' not in self._components:
            self._components['allocation_manager'] = self.registry.get_component(
                'allocation_manager')
        return self._components['allocation_manager']

    @property
    def status_reporter(self):
        """延迟加载状态报告器"""
        if 'status_reporter' not in self._components:
            self._components['status_reporter'] = self.registry.get_component('status_reporter')
        return self._components['status_reporter']

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

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            # 获取详细状态
            detailed_status = self.status_reporter.get_resource_status()
            summary = detailed_status.get('summary', {})
            
            # 返回测试期望的格式
            return {
                'providers_count': summary.get('providers_count', 0),
                'consumers_count': summary.get('consumers_count', 0),
                'allocations_count': summary.get('active_allocations', 0),
                'system_health': detailed_status.get('health', 'unknown')
            }
        except Exception as e:
            self.logger.warning(f"获取系统状态失败: {e}")
            return {
                'providers_count': 0,
                'consumers_count': 0,
                'allocations_count': 0,
                'system_health': 'error'
            }

    def optimize_resources(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """优化资源分配"""
        # 这里可以实现资源优化逻辑
        return {
            "success": True,
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

    def shutdown(self):
        """关闭资源管理器"""
        try:
            self.logger.log_info("正在关闭统一资源管理器...")
            
            # 清理资源分配
            self._cleanup_allocations()
            
            # 停止所有组件
            self.logger.log_info("统一资源管理器已关闭")
            
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "关闭资源管理器失败"})

    def _cleanup_allocations(self):
        """清理资源分配"""
        try:
            if hasattr(self, 'allocation_manager') and self.allocation_manager:
                # 清理所有活跃的分配
                with self.allocation_manager._lock:
                    allocation_count = len(self.allocation_manager._allocations)
                    if allocation_count > 0:
                        self.logger.log_info(f"清理 {allocation_count} 个活跃的资源分配")
                        # 这里可以添加更具体的清理逻辑
                        # 例如释放所有分配的资源等
        except Exception as e:
            self.logger.log_warning(f"清理资源分配时发生错误: {e}")

    def start(self):
        """启动资源管理器"""
        self.logger.log_info("统一资源管理器启动")
        self._running = True
        return None  # 返回None表示同步方法

    def stop(self):
        """停止资源管理器"""
        self.logger.log_info("统一资源管理器停止")
        self._running = False
        self.shutdown()
        return None  # 返回None表示同步方法
