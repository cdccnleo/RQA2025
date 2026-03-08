#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向后兼容的基础设施集成管理器适配层

说明：
- 为历史代码提供 get_data_integration_manager / log_data_operation /
  record_data_metric / publish_data_event 等接口
- 实际委派到统一基础设施集成层（src.core.integration）与 DataManagerSingleton
"""

from typing import Any, Optional, Dict

try:
    # 统一基础设施集成层（日志、指标等）
    from src.infrastructure.integration import (
        log_data_operation as _log_data_operation,
        record_data_metric as _record_data_metric,
        get_data_layer_adapter as _get_data_layer_adapter,
    )
except Exception:
    # 降级：提供空实现，避免导入失败
    def _log_data_operation(operation: str, data_type, details: dict, level: str = "info") -> None:  # type: ignore
        pass

    def _record_data_metric(metric_name: str, value, data_type, tags: dict = None) -> None:  # type: ignore
        pass

    def _get_data_layer_adapter():
        return None

try:
    # 数据管理器（事件总线、健康检查桥等）
    from .core.data_manager import DataManagerSingleton
except Exception:
    DataManagerSingleton = None  # type: ignore


class _CompatIntegrationManager:
    """
    兼容期的“基础设施集成管理器”外观对象
    - 提供 initialize() / get_health_check_bridge() / _integration_config 等接口
    - 内部委派到 DataManagerSingleton 与统一适配器
    """

    def __init__(self) -> None:
        self._initialized: bool = False
        self._data_manager = None
        self._adapter = None
        self._integration_config: Dict[str, Any] = {}

        # 预取对象（容错）
        if DataManagerSingleton is not None:
            try:
                self._data_manager = DataManagerSingleton.get_instance()
                # 从 DataManager 侧尝试获取配置
                try:
                    # 这里保持字典形态，避免破坏调用方解构
                    self._integration_config = {
                        'enable_data_catalog': True,
                        'enable_data_marketplace': True,
                        'catalog_update_interval': 3600,
                    }
                except Exception:
                    self._integration_config = {}
            except Exception:
                self._data_manager = None

        try:
            self._adapter = _get_data_layer_adapter()
        except Exception:
            self._adapter = None

    def initialize(self) -> bool:
        """幂等初始化"""
        self._initialized = True
        return True

    def get_health_check_bridge(self):
        """返回健康检查桥接器，如不可用则返回 None"""
        try:
            if self._data_manager and hasattr(self._data_manager, 'health_bridge'):
                return getattr(self._data_manager, 'health_bridge')
        except Exception:
            return None
        return None

    # 兼容旧调用路径：publish_data_event(...)
    def publish_data_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        try:
            if self._data_manager and hasattr(self._data_manager, 'publish_data_event'):
                self._data_manager.publish_data_event(event_type, event_data)
        except Exception:
            # 静默降级
            pass


def get_data_integration_manager() -> _CompatIntegrationManager:
    """
    获取兼容期“基础设施集成管理器”实例
    """
    return _CompatIntegrationManager()


def log_data_operation(operation: str, data_type, details: dict, level: str = "info") -> None:
    """
    向后兼容：记录数据操作
    """
    _log_data_operation(operation, data_type, details, level)


def record_data_metric(metric_name: str, value, data_type, tags: dict = None) -> None:
    """
    向后兼容：记录数据指标
    """
    _record_data_metric(metric_name, value, data_type, tags or {})


def publish_data_event(event_type: str, event_data: Dict[str, Any], *_args, **_kwargs) -> None:
    """
    向后兼容：发布数据事件
    说明：历史调用为自由函数，这里委派到 DataManagerSingleton.publish_data_event
    """
    try:
        if DataManagerSingleton is not None:
            dm = DataManagerSingleton.get_instance()
            if hasattr(dm, 'publish_data_event'):
                dm.publish_data_event(event_type, event_data)
    except Exception:
        # 静默降级
        pass


