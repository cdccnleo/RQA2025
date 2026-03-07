"""
error_handler_factory 模块

提供 error_handler_factory 相关功能和接口。
"""

import logging

# 注册处理器类

from .error_handler import ErrorHandler
from .infrastructure_error_handler import InfrastructureErrorHandler
from .specialized_error_handler import SpecializedErrorHandler
from enum import Enum
from ..core.interfaces import IErrorHandler
from typing import Dict, Any, Optional, Type, Callable, List
"""
基础设施层 - 错误处理器工厂

提供统一的处理器创建和管理接口，实现处理器工厂模式。
支持动态注册、配置管理和处理器选择。
"""

logger = logging.getLogger(__name__)


class HandlerType(Enum):
    """处理器类型"""
    GENERAL = "general"           # 通用错误处理器
    INFRASTRUCTURE = "infrastructure"  # 基础设施错误处理器
    SPECIALIZED = "specialized"   # 专用错误处理器
    BUSINESS = "business"         # 业务错误处理器 (如果需要)


class HandlerConfig:
    """处理器配置"""

    def __init__(self,
                 handler_type: HandlerType,
                 max_history: int = 1000,
                 enable_boundary_check: bool = True,
                 enable_retry: bool = True,
                 custom_config: Optional[Dict[str, Any]] = None):
        self.handler_type = handler_type
        self.max_history = max_history
        self.enable_boundary_check = enable_boundary_check
        self.enable_retry = enable_retry
        self.custom_config = custom_config or {}


class ErrorHandlerFactory:
    """
    错误处理器工厂

    统一管理所有错误处理器的创建、配置和生命周期。
    支持动态注册、配置管理和智能选择。
    """

    def __init__(self):
        self._handler_classes: Dict[HandlerType, Type[IErrorHandler]] = {}
        self._handler_configs: Dict[HandlerType, HandlerConfig] = {}
        self._handler_instances: Dict[str, IErrorHandler] = {}
        self._creation_strategies: Dict[HandlerType, Callable] = {}

        # 注册默认处理器
        self._register_default_handlers()

    def _register_default_handlers(self):
        """注册默认处理器"""
        try:
            # 动态导入处理器类
            self.register_handler_class(HandlerType.GENERAL, ErrorHandler)
            self.register_handler_class(HandlerType.INFRASTRUCTURE, InfrastructureErrorHandler)
            self.register_handler_class(HandlerType.SPECIALIZED, SpecializedErrorHandler)

            # 设置默认配置
            self.set_handler_config(HandlerType.GENERAL, HandlerConfig(HandlerType.GENERAL))
            self.set_handler_config(HandlerType.INFRASTRUCTURE,
                                    HandlerConfig(HandlerType.INFRASTRUCTURE))
            self.set_handler_config(HandlerType.SPECIALIZED, HandlerConfig(HandlerType.SPECIALIZED))

        except ImportError as e:
            logger.warning(f"无法导入默认处理器类: {e}")

    def register_handler_class(self, handler_type: HandlerType, handler_class: Type[IErrorHandler]) -> None:
        """注册处理器类"""
        if not issubclass(handler_class, IErrorHandler):
            raise ValueError(f"处理器类 {handler_class} 必须实现 IErrorHandler 接口")

        self._handler_classes[handler_type] = handler_class
        logger.info(f"已注册处理器类: {handler_type.value} -> {handler_class.__name__}")

    def set_handler_config(self, handler_type: HandlerType, config: HandlerConfig) -> None:
        """设置处理器配置"""
        self._handler_configs[handler_type] = config
        logger.info(f"已设置处理器配置: {handler_type.value}")

    def set_creation_strategy(self, handler_type: HandlerType, strategy: Callable) -> None:
        """设置处理器创建策略"""
        self._creation_strategies[handler_type] = strategy
        logger.info(f"已设置处理器创建策略: {handler_type.value}")

    def create_handler(self, handler_type: HandlerType, instance_id: Optional[str] = None) -> IErrorHandler:
        """创建处理器实例"""
        if handler_type not in self._handler_classes:
            raise ValueError(f"未注册的处理器类型: {handler_type.value}")

        handler_class = self._handler_classes[handler_type]
        config = self._handler_configs.get(handler_type, HandlerConfig(handler_type))

        # 生成实例ID
        if instance_id is None:
            instance_id = f"{handler_type.value}_{id(self)}"

        # 检查是否已有实例
        if instance_id in self._handler_instances:
            logger.warning(f"处理器实例已存在，使用现有实例: {instance_id}")
            return self._handler_instances[instance_id]

        try:
            # 使用创建策略或默认方式创建实例
            if handler_type in self._creation_strategies:
                strategy = self._creation_strategies[handler_type]
                handler_instance = strategy(config)
            else:
                # 默认创建方式
                handler_instance = self._create_handler_default(handler_class, config)

            # 缓存实例
            self._handler_instances[instance_id] = handler_instance

            logger.info(f"成功创建处理器实例: {handler_type.value} ({instance_id})")
            return handler_instance

        except Exception as e:
            logger.error(f"创建处理器实例失败: {handler_type.value}, {e}")
            raise

    def _create_handler_default(self, handler_class: Type[IErrorHandler], config: HandlerConfig) -> IErrorHandler:
        """默认处理器创建方式"""
        # 根据处理器类型使用不同的参数
        if config.handler_type == HandlerType.GENERAL:
            return handler_class(max_history=config.max_history)
        elif config.handler_type == HandlerType.INFRASTRUCTURE:
            return handler_class(max_history=config.max_history)
        elif config.handler_type == HandlerType.SPECIALIZED:
            return handler_class(max_history=config.max_history)
        else:
            # 默认构造
            return handler_class()

    def get_handler(self, instance_id: str) -> Optional[IErrorHandler]:
        """获取处理器实例"""
        return self._handler_instances.get(instance_id)

    def destroy_handler(self, instance_id: str) -> bool:
        """销毁处理器实例"""
        if instance_id in self._handler_instances:
            try:
                # 这里可以添加清理逻辑
                del self._handler_instances[instance_id]
                logger.info(f"已销毁处理器实例: {instance_id}")
                return True
            except Exception as e:
                logger.error(f"销毁处理器实例失败: {instance_id}, {e}")
                return False
        return False

    def list_registered_handlers(self) -> List[str]:
        """列出已注册的处理器类型"""
        return [ht.value for ht in self._handler_classes.keys()]

    def list_active_instances(self) -> List[str]:
        """列出活跃的处理器实例"""
        return list(self._handler_instances.keys())

    def get_handler_stats(self) -> Dict[str, Any]:
        """获取处理器统计信息"""
        stats = {
            'registered_handlers': len(self._handler_classes),
            'active_instances': len(self._handler_instances),
            'handler_types': self.list_registered_handlers(),
            'instance_ids': self.list_active_instances()
        }

        # 收集各实例的统计信息
        instance_stats = {}
        for instance_id, handler in self._handler_instances.items():
            try:
                if hasattr(handler, 'get_stats'):
                    instance_stats[instance_id] = handler.get_stats()
                else:
                    instance_stats[instance_id] = {'status': 'no_stats_method'}
            except Exception as e:
                instance_stats[instance_id] = {'error': str(e)}

        stats['instance_stats'] = instance_stats
        return stats

    def select_handler_for_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> HandlerType:
        """根据错误类型智能选择处理器"""
        error_type = type(error).__name__

        # 业务相关错误
        if any(keyword in error_type.lower() for keyword in ['trade', 'order', 'market', 'strategy']):
            return HandlerType.BUSINESS

        # 基础设施相关错误
        elif any(keyword in error_type.lower() for keyword in ['connection', 'network', 'database', 'timeout']):
            return HandlerType.INFRASTRUCTURE

        # 专用错误处理
        elif any(keyword in error_type.lower() for keyword in ['influx', 'archive', 'retry']):
            return HandlerType.SPECIALIZED

        # 默认使用通用处理器
        else:
            return HandlerType.GENERAL

    def handle_error_smart(self, error: Exception,
                           context: Optional[Dict[str, Any]] = None,
                           handler_type: Optional[HandlerType] = None) -> Dict[str, Any]:
        """智能错误处理"""
        # 如果没有指定处理器类型，智能选择
        if handler_type is None:
            handler_type = self.select_handler_for_error(error, context)

        # 创建或获取处理器实例
        instance_id = f"smart_{handler_type.value}"
        handler = self.create_handler(handler_type, instance_id)

        # 处理错误
        try:
            result = handler.handle_error(error, context)
            result['selected_handler'] = handler_type.value
            result['instance_id'] = instance_id
            return result
        except Exception as handler_error:
            logger.error(f"智能错误处理失败: {handler_error}")
            return {
                'handled': False,
                'error': str(handler_error),
                'selected_handler': handler_type.value,
                'fallback': True
            }

    def cleanup(self) -> None:
        """清理所有处理器实例"""
        instance_ids = list(self._handler_instances.keys())
        for instance_id in instance_ids:
            self.destroy_handler(instance_id)
        logger.info("已清理所有处理器实例")


# 全局工厂实例
_global_factory: Optional[ErrorHandlerFactory] = None


def get_global_factory() -> ErrorHandlerFactory:
    """获取全局工厂实例"""
    global _global_factory
    if _global_factory is None:
        _global_factory = ErrorHandlerFactory()
    return _global_factory


def create_handler(handler_type: HandlerType, instance_id: Optional[str] = None) -> IErrorHandler:
    """便捷函数：创建处理器"""
    return get_global_factory().create_handler(handler_type, instance_id)


def handle_error_smart(error: Exception,
                       context: Optional[Dict[str, Any]] = None,
                       handler_type: Optional[HandlerType] = None) -> Dict[str, Any]:
    """便捷函数：智能错误处理"""
    return get_global_factory().handle_error_smart(error, context, handler_type)
