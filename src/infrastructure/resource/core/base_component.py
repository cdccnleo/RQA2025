"""
base_component 模块

提供 base_component 相关功能和接口。
"""

import logging

import threading
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
"""
基础设施层 - 统一基础组件

提供所有资源管理组件的统一基类，减少模板代码重复。

重构说明：
- 添加参数对象模式来解决长参数列表问题
- 创建ParameterConfig类封装相关参数
"""


@dataclass
class ParameterConfig:
    """
    参数配置类

    用于封装函数参数，解决长参数列表问题
    """

    # 基本参数
    operation_name: Optional[str] = None
    resource_type: Optional[str] = None
    resource_name: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3

    # 性能阈值参数
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    network_threshold: float = 100.0
    response_time_threshold: float = 10.0
    error_rate_threshold: float = 0.05

    # 监控参数
    enable_alerts: bool = True
    alert_channels: Optional[List[str]] = None
    custom_metrics: Optional[Dict[str, Any]] = None
    log_level: str = "INFO"

    # 分页和查询参数
    page: int = 1
    page_size: int = 50
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    filters: Optional[Dict[str, Any]] = None

    # 时间范围参数
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    time_range: int = 3600  # 默认1小时

    # 其他扩展参数
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: 'ParameterConfig') -> 'ParameterConfig':
        """合并参数配置"""
        if not other:
            return self

        # 创建新实例，合并参数
        merged = ParameterConfig()
        for field_name in self.__dataclass_fields__:
            value = getattr(other, field_name)
            if value is not None:
                setattr(merged, field_name, value)
            else:
                setattr(merged, field_name, getattr(self, field_name))

        return merged

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterConfig':
        """从字典创建参数配置"""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


class IBaseResourceComponent(ABC):
    """资源组件接口"""

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化组件"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def shutdown(self) -> None:
        """关闭组件"""


# 为了保持向后兼容性，添加别名
IResourceComponent = IBaseResourceComponent


class BaseResourceComponent(IBaseResourceComponent):
    """
    统一的基础资源组件

    提供所有资源管理组件的通用功能：
    - 配置管理
    - 状态管理
    - 日志记录
    - 线程安全
    - 生命周期管理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, component_name: str = "resource"):
        """
        初始化基础组件

        Args:
            config: 组件配置
            component_name: 组件名称，用于日志和状态标识
        """
        self.config = config or {}
        self.component_name = component_name
        self._initialized = False
        self._status = "stopped"
        self._start_time = None
        self._lock = threading.RLock()
        self._stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'last_operation_time': None,
            'average_response_time': 0.0
        }

        # 初始化日志
        self.logger = logging.getLogger(f"{__name__}.{component_name}")

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        初始化组件

        Args:
            config: 组件配置

        Returns:
            初始化是否成功
        """
        with self._lock:
            try:
                # 更新配置
                self.config.update(config)

                # 设置状态
                self._initialized = True
                self._status = "running"
                self._start_time = time.time()

                # 记录初始化成功
                self.logger.info(f"{self.component_name} 组件初始化成功")

                # 调用子类特定的初始化逻辑
                self._initialize_component()

                return True

            except Exception as e:
                self._status = "error"
                self._initialized = False
                self.logger.error(f"{self.component_name} 组件初始化失败: {e}")
                return False

    def _initialize_component(self):
        """子类特定的初始化逻辑，子类可以重写"""

    def get_status(self) -> Dict[str, Any]:
        """
        获取组件状态

        Returns:
            组件状态信息
        """
        with self._lock:
            uptime = time.time() - self._start_time if self._start_time else 0

            return {
                "component": self.component_name,
                "status": self._status,
                "initialized": self._initialized,
                "uptime": uptime,
                "config": self.config.copy(),  # 返回配置副本
                "stats": self._stats.copy()
            }

    def shutdown(self) -> None:
        """关闭组件"""
        with self._lock:
            try:
                # 调用子类特定的关闭逻辑
                self._shutdown_component()

                # 只有在成功关闭时才更新状态
                self._initialized = False
                self._status = "stopped"

                self.logger.info(f"{self.component_name} 组件已关闭")

            except Exception as e:
                self.logger.error(f"{self.component_name} 组件关闭失败: {e}")
                # 关闭失败时状态保持不变

    def _shutdown_component(self):
        """子类特定的关闭逻辑，子类可以重写"""

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新组件配置

        Args:
            new_config: 新配置

        Returns:
            更新是否成功
        """
        with self._lock:
            try:
                self.config.update(new_config)
                self.logger.info(f"{self.component_name} 配置已更新")
                return True
            except Exception as e:
                self.logger.error(f"{self.component_name} 配置更新失败: {e}")
                return False

    def is_healthy(self) -> bool:
        """
        检查组件健康状态

        Returns:
            是否健康
        """
        return self._initialized and self._status == "running"

    def record_operation(self, success: bool, response_time: Optional[float] = None):
        """
        记录操作统计

        Args:
            success: 操作是否成功
            response_time: 响应时间（秒）
        """
        with self._lock:
            self._stats['total_operations'] += 1
            self._stats['last_operation_time'] = time.time()

            if success:
                self._stats['successful_operations'] += 1
            else:
                self._stats['failed_operations'] += 1

            if response_time is not None:
                # 更新平均响应时间
                total_time = self._stats['average_response_time'] * \
                    (self._stats['total_operations'] - 1)
                total_time += response_time
                self._stats['average_response_time'] = total_time / self._stats['total_operations']

    def get_operation_stats(self) -> Dict[str, Any]:
        """
        获取操作统计信息

        Returns:
            统计信息
        """
        with self._lock:
            stats = self._stats.copy()
            stats['success_rate'] = (
                stats['successful_operations'] / stats['total_operations'] * 100
                if stats['total_operations'] > 0 else 0
            )
            return stats

    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self._stats = {
                'total_operations': 0,
                'successful_operations': 0,
                'failed_operations': 0,
                'last_operation_time': None,
                'average_response_time': 0.0
            }
            self.logger.info(f"{self.component_name} 统计信息已重置")

    def log_operation(self, operation: str, details: Optional[Dict[str, Any]] = None, level: str = "info"):
        """
        记录操作日志

        Args:
            operation: 操作名称
            details: 操作详情
            level: 日志级别
        """
        message = f"{self.component_name} 执行操作: {operation}"
        if details:
            message += f" - {details}"

        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
