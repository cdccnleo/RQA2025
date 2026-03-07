
import logging
import threading
import time
import os
import json
import asyncio
from typing import Any, Callable, Dict, List, Optional, Union
from src.infrastructure.logging.utils.logger import get_logger
"""
common_mixins 模块

提供 common_mixins 相关功能和接口。
"""

# from src.infrastructure.config.core.imports import time
# from src.infrastructure.config.core.imports import threading
# from src.infrastructure.config.core.imports import time
# from src.infrastructure.config.core.common_exception_handler import (
# from src.infrastructure.config.core.common_logger import (
# from src.infrastructure.config.core.imports import (
# ==================== 公共Mixin类 ====================

# 导入通用工具 (暂时注释以避免循环依赖)
# from src.infrastructure.config.core.common_logger import (
#     get_logger,
#     create_operation_context,
#     LogContext,
#     OperationType,
#     default_logger as common_logger
# )
# from src.infrastructure.config.core.imports import (
#     Dict, Any, Optional, List, Union, Callable, Type, TypeVar,
#     threading, time, json, os, logging, Path, datetime, dataclass, field
# )


class ConfigComponentMixin:
    """配置组件基础Mixin类

    提供通用的初始化和基础功能，避免重复代码。
    """

    def _init_threading_support(self):
        """初始化线程安全支持"""
        self._lock = threading.RLock()

    def _init_config_storage(self, config: Dict[str, Any] = None):
        """初始化配置存储"""
        self._config = config or {}

    def _init_metrics_collection(self):
        """初始化指标收集"""
        self._metrics = {}

    def _init_alert_system(self):
        """初始化告警系统"""
        self._alerts = []

    def _init_history_tracking(self):
        """初始化历史跟踪"""
        self._history = []

    def _init_data_structures(self):
        """初始化数据结构"""
        self._data = {}

    def _init_component_attributes(self,
                                   enable_threading=True,
                                   enable_config: Dict[str, Any] = True,
                                   enable_metrics=False,
                                   enable_alerts=False,
                                   enable_history=False,
                                   enable_data=False,
                                   config: Dict[str, Any] = None):
        """统一初始化组件属性

        Args:
            enable_threading: 是否启用线程安全
            enable_config: 是否启用配置存储
            enable_metrics: 是否启用指标收集
            enable_alerts: 是否启用告警系统
            enable_history: 是否启用历史跟踪
            enable_data: 是否启用通用数据结构
            config: 初始配置数据
        """
        if enable_threading:
            self._init_threading_support()

        if enable_config:
            self._init_config_storage(config)

        if enable_metrics:
            self._init_metrics_collection()

        if enable_alerts:
            self._init_alert_system()

        if enable_history:
            self._init_history_tracking()

        if enable_data:
            self._init_data_structures()


class MonitoringMixin(ConfigComponentMixin):
    """监控组件Mixin类"""

    def __init__(self, enable_metrics=True, enable_alerts=True, enable_history=True):
        """初始化监控组件"""
        super().__init__()
        self._init_component_attributes(
            enable_threading=True,
            enable_config=True,
            enable_metrics=enable_metrics,
            enable_alerts=enable_alerts,
            enable_history=enable_history
        )

    def record_metric(self, name: str, value, timestamp=None):
        """记录指标"""
        if not hasattr(self, '_metrics'):
            self._init_metrics_collection()

        if timestamp is None:
            timestamp = time.time()

        if name not in self._metrics:
            self._metrics[name] = []

        self._metrics[name].append({
            'value': value,
            'timestamp': timestamp
        })

        # 限制历史记录数量
        if len(self._metrics[name]) > 1000:
            self._metrics[name] = self._metrics[name][-500:]

    def get_latest_metric(self, name: str):
        """获取最新指标"""
        if name in self._metrics and self._metrics[name]:
            return self._metrics[name][-1]
        return None


class CRUDOperationsMixin(ConfigComponentMixin):
    """CRUD操作Mixin类"""

    def __init__(self):
        """初始化CRUD操作组件"""
        super().__init__()
        self._init_component_attributes(enable_threading=True, enable_config=True)

    def create(self, key: str, value):
        """创建记录"""
        with self._lock:
            self._config[key] = value
            self._record_operation('create', key, value)

    def read(self, key: str):
        """读取记录"""
        with self._lock:
            return self._config.get(key)

    def update(self, key: str, value):
        """更新记录"""
        with self._lock:
            if key in self._config:
                old_value = self._config[key]
                self._config[key] = value
                self._record_operation('update', key, value, old_value)
                return True
            return False

    def delete(self, key: str):
        """删除记录"""
        with self._lock:
            if key in self._config:
                value = self._config[key]
                del self._config[key]
                self._record_operation('delete', key, value)
                return True
            return False

    def _record_operation(self, operation: str, key: str, value=None, old_value=None):
        """记录操作历史"""
        if not hasattr(self, '_history'):
            self._init_history_tracking()

        record = {
            'operation': operation,
            'key': key,
            'value': value,
            'old_value': old_value,
            'timestamp': time.time()
        }

        self._history.append(record)

        # 限制历史记录数量
        if len(self._history) > 1000:
            self._history = self._history[-500:]


class ComponentLifecycleMixin(ConfigComponentMixin):
    """组件生命周期Mixin类"""

    def __init__(self):
        """初始化生命周期管理"""
        super().__init__()
        self._init_component_attributes(enable_threading=True)
        self._initialized = False
        self._started = False
        self._stopped = False

    def initialize(self):
        """初始化组件"""
        if not self._initialized:
            self._do_initialize()
            self._initialized = True

    def start(self):
        """启动组件"""
        if not self._started:
            if not self._initialized:
                self.initialize()
            self._do_start()
            self._started = True

    def stop(self):
        """停止组件"""
        if self._started and not self._stopped:
            self._do_stop()
            self._stopped = True

    def restart(self):
        """重启组件"""
        self.stop()
        self._started = False
        self._stopped = False
        self.start()

    def _do_initialize(self):
        """子类实现具体的初始化逻辑"""

    def _do_start(self):
        """子类实现具体的启动逻辑"""

    def _do_stop(self):
        """子类实现具体的停止逻辑"""

    @property
    def is_initialized(self):
        """检查是否已初始化"""
        return self._initialized

    @property
    def is_started(self):
        """检查是否已启动"""
        return self._started

    @property
    def is_stopped(self):
        """检查是否已停止"""
        return self._stopped


class BatchOperationsMixin(ConfigComponentMixin):
    """批量操作Mixin类"""

    def __init__(self):
        """初始化批量操作组件"""
        super().__init__()
        self._init_component_attributes(enable_threading=True, enable_config=True)
        self._logger = get_logger(__name__)

    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取配置"""
        result = {}
        for key in keys:
            if hasattr(self, 'get'):
                result[key] = self.get(key)
            else:
                result[key] = None
        return result

    def batch_set(self, config: Dict[str, Any]) -> bool:
        """批量设置配置"""
        try:
            for key, value in config.items():
                if hasattr(self, 'set'):
                    self.set(key, value)
            return True
        except Exception as e:
            self._logger.error(f"批量设置配置失败: {e}")
            return False

# ==================== 向后兼容性 ====================


# 为现有类提供别名，确保向后兼容
ConfigComponentMixinAlias = ConfigComponentMixin
MonitoringMixinAlias = MonitoringMixin
CRUDOperationsMixinAlias = CRUDOperationsMixin
ComponentLifecycleMixinAlias = ComponentLifecycleMixin
BatchOperationsMixinAlias = BatchOperationsMixin




