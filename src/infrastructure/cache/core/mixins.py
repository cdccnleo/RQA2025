"""
mixins 模块

提供 mixins 相关功能和接口。
"""

import logging

import threading
import time

from ..interfaces import PerformanceMetrics
from datetime import datetime
from typing import Dict, Any, Optional, List
#!/usr/bin/env python3
"""
基础设施层 - Mixin类定义

提供各种功能的Mixin类，用于减少代码重复，提高代码复用性。
使用Mixin模式实现关注点分离。
"""

logger = logging.getLogger(__name__)


class MonitoringMixin:
    """
    监控功能Mixin类

    提供统一的监控启动、停止和指标收集功能。
    减少监控相关代码的重复。
    """

    def __init__(self, enable_monitoring: bool = True, monitor_interval: int = 30):
        """
        初始化监控Mixin

        Args:
            enable_monitoring: 是否启用监控
            monitor_interval: 监控间隔（秒）
        """
        self._enable_monitoring = enable_monitoring
        self._monitor_interval = monitor_interval
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._last_metrics: Optional[PerformanceMetrics] = None

    def start_monitoring(self) -> bool:
        """
        启动监控

        Returns:
            bool: 启动是否成功
        """
        if not self._enable_monitoring:
            logger.info("监控功能已禁用")
            return False

        if self._monitoring_active:
            logger.warning("监控已在运行中")
            return False

        try:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name=f"{self.__class__.__name__}_monitor"
            )
            self._monitoring_thread.start()
            logger.info("监控已启动")
            return True
        except Exception as e:
            logger.error(f"启动监控失败: {e}")
            self._monitoring_active = False
            return False

    def stop_monitoring(self) -> bool:
        """
        停止监控

        Returns:
            bool: 停止是否成功
        """
        if not self._monitoring_active:
            return True

        try:
            self._monitoring_active = False
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)

            logger.info("监控已停止")
            return True
        except Exception as e:
            logger.error(f"停止监控失败: {e}")
            return False

    def _monitoring_loop(self) -> None:
        """监控主循环"""
        while self._monitoring_active:
            try:
                # 收集指标
                metrics = self._collect_metrics()

                # 检查告警
                self._check_alerts(metrics)

                # 保存最新指标
                self._last_metrics = metrics

                # 等待下次检查
                time.sleep(self._monitor_interval)

            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                time.sleep(self._monitor_interval)

    def _collect_metrics(self) -> PerformanceMetrics:
        """
        收集性能指标

        子类应该重写此方法以提供具体的指标收集逻辑。

        Returns:
            PerformanceMetrics: 收集到的性能指标
        """
        # 基础指标收集
        return PerformanceMetrics.create_current(
            hit_rate=getattr(self, 'hit_rate', 0.0),
            response_time=getattr(self, 'avg_response_time', 0.0),
            throughput=getattr(self, 'requests_per_second', 0),
            memory_usage=getattr(self, 'memory_usage_mb', 0.0),
            cache_size=getattr(self, 'cache_size', 0),
            eviction_rate=getattr(self, 'eviction_rate', 0.0),
            miss_penalty=getattr(self, 'miss_penalty', 0.0)
        )

    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """
        检查告警条件

        Args:
            metrics: 当前性能指标
        """
        # 基础告警检查逻辑
        alerts = []

        if metrics.hit_rate < 0.5:  # 命中率低于50%
            alerts.append(f"低命中率: {metrics.hit_rate:.2%}")

        if metrics.response_time > 100:  # 响应时间超过100ms
            alerts.append(f"高响应时间: {metrics.response_time:.1f}ms")

        if metrics.memory_usage > 500:  # 内存使用超过500MB
            alerts.append(f"高内存使用: {metrics.memory_usage:.1f}MB")

        if alerts:
            logger.warning(f"性能告警: {', '.join(alerts)}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        获取监控状态

        Returns:
            Dict[str, Any]: 监控状态信息
        """
        return {
            'monitoring_enabled': self._enable_monitoring,
            'monitoring_active': self._monitoring_active,
            'monitor_interval': self._monitor_interval,
            'last_metrics': self._last_metrics.to_dict() if self._last_metrics else None,
            'thread_alive': self._monitoring_thread.is_alive() if self._monitoring_thread else False
        }


class CRUDOperationsMixin:
    """
    CRUD操作Mixin类

    提供统一的缓存CRUD操作实现。
    减少重复的get/set/delete逻辑。
    """

    def __init__(self, storage_backend=None):
        """
        初始化CRUD操作Mixin

        Args:
            storage_backend: 存储后端，如果为None则使用默认字典
        """
        self._storage = storage_backend or {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            Optional[Any]: 缓存值，None表示不存在
        """
        with self._lock:
            try:
                return self._storage.get(key)
            except Exception as e:
                logger.error(f"获取缓存失败 {key}: {e}")
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            bool: 设置是否成功
        """
        try:
            with self._lock:
                self._storage[key] = value
                # 这里可以添加TTL逻辑
                if ttl:
                    # 设置过期时间（简化实现）
                    pass
            return True
        except Exception as e:
            logger.error(f"设置缓存失败 {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        删除缓存值

        Args:
            key: 缓存键

        Returns:
            bool: 删除是否成功
        """
        try:
            with self._lock:
                return self._storage.pop(key, None) is not None
        except Exception as e:
            logger.error(f"删除缓存失败 {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        检查键是否存在

        Args:
            key: 缓存键

        Returns:
            bool: 是否存在
        """
        with self._lock:
            return key in self._storage

    def clear(self) -> bool:
        """
        清空所有缓存

        Returns:
            bool: 清空是否成功
        """
        try:
            with self._lock:
                self._storage.clear()
            return True
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False

    def size(self) -> int:
        """
        获取缓存大小

        Returns:
            int: 缓存项数量
        """
        with self._lock:
            return len(self._storage)

    def keys(self) -> List[str]:
        """
        获取所有键

        Returns:
            List[str]: 所有缓存键的列表
        """
        with self._lock:
            return list(self._storage.keys())

    def __setitem__(self, key: str, value: Any) -> None:
        """设置缓存项（字典风格接口）"""
        self.set(key, value)

    def __getitem__(self, key: str) -> Any:
        """获取缓存项（字典风格接口）"""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """检查键是否存在（字典风格接口）"""
        return self.exists(key)


class ComponentLifecycleMixin:
    """
    组件生命周期Mixin类

    提供统一的组件初始化、状态管理和清理功能。
    减少组件管理代码的重复。
    """

    def __init__(self, component_id: Optional[int] = None,
                 component_type: str = "component",
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化组件生命周期Mixin

        Args:
            component_id: 组件ID
            component_type: 组件类型
            config: 组件配置
        """
        self._component_id = component_id
        self._component_type = component_type
        self._config = config or {}
        self._initialized = False
        self._status = "stopped"
        self._creation_time = datetime.now()
        self._error_count = 0
        self._last_check = datetime.now()

    def _initialize_component(self):
        """组件初始化钩子方法，子类可以重写"""
        pass

    @property
    def component_id(self) -> Optional[int]:
        """组件ID"""
        return self._component_id

    @property
    def component_type(self) -> str:
        """组件类型"""
        return self._component_type

    def start_component(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        启动组件（兼容性方法，调用initialize_component）

        Args:
            config: 组件配置

        Returns:
            bool: 启动是否成功
        """
        return self.initialize_component(config)

    def initialize_component(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        初始化组件

        Args:
            config: 组件配置

        Returns:
            bool: 初始化是否成功
        """
        try:
            if config:
                self._config.update(config)

            # 调用子类初始化钩子
            self._initialize_component()

            self._initialized = True
            self._status = "healthy"
            self._last_check = datetime.now()

            logger.info(f"组件 {self.component_type} 初始化成功")
            return True
        except Exception as e:
            self._error_count += 1
            self._status = "error"
            logger.error(f"组件 {self.component_type} 初始化失败: {e}")
            return False

    def get_component_status(self) -> Dict[str, Any]:
        """
        获取组件状态

        Returns:
            Dict[str, Any]: 组件状态信息
        """
        self._last_check = datetime.now()

        return {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'status': self._status,
            'initialized': self._initialized,
            'creation_time': self._creation_time.isoformat(),
            'last_check': self._last_check.isoformat(),
            'error_count': self._error_count,
            'config': self._config.copy()
        }

    def stop_component(self) -> bool:
        """
        停止组件（兼容性方法，调用shutdown_component）

        Returns:
            bool: 停止是否成功
        """
        return self.shutdown_component()

    def shutdown_component(self) -> bool:
        """关闭组件"""
        try:
            # 调用子类关闭钩子
            self._shutdown_component()

            self._initialized = False
            self._status = "stopped"
            logger.info(f"组件 {self.component_type} 已关闭")
            return True
        except Exception as e:
            logger.error(f"组件 {self.component_type} 关闭失败: {e}")
            return False

    def _shutdown_component(self):
        """组件关闭钩子方法，子类可以重写"""
        pass

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: 组件是否健康
        """
        try:
            self._last_check = datetime.now()
            # 基础健康检查：组件已初始化且状态正常，且错误计数未超过阈值
            is_healthy = (self._initialized and
                         self._status in ["healthy", "initialized"] and
                         self._error_count <= 5)  # 错误阈值
            return is_healthy
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def reset_error_count(self) -> None:
        """重置错误计数"""
        self._error_count = 0

    def get_uptime_seconds(self) -> float:
        """
        获取运行时间（秒）

        Returns:
            float: 从创建到现在的秒数
        """
        return (datetime.now() - self._creation_time).total_seconds()


class CacheTierMixin:
    """
    缓存层级操作Mixin

    为不同缓存层级提供通用的CRUD操作模板。
    各层级需要实现具体的存储后端操作。
    """

    def __init__(self):
        self.stats = {}  # 统计信息
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.CacheTierMixin")
        self._storage = {}  # 存储后端，子类可以重写

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值 - 通用模板

        Args:
            key: 缓存键

        Returns:
            Optional[Any]: 缓存值，不存在返回None
        """
        with self.lock:
            try:
                # 检查是否存在
                if not self._key_exists(key):
                    self._update_stats('misses', 1)
                    return None

                # 检查是否过期
                if self._is_expired(key):
                    self._remove_expired(key)
                    self._update_stats('misses', 1)
                    return None

                # 获取值
                value = self._get_value(key)

                # 更新统计和访问时间
                self._update_stats('hits', 1)
                self._update_access_time(key)

                return value

            except Exception as e:
                self.logger.error(f"获取缓存值失败 {key}: {e}")
                self._update_stats('misses', 1)
                return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值 - 通用模板

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒）

        Returns:
            bool: 设置是否成功
        """
        with self.lock:
            try:
                # 检查容量限制
                if self._should_evict():
                    self._evict_oldest()

                # 设置值和元数据
                success = self._set_value(key, value, ttl)

                if success:
                    self._update_stats('sets', 1)
                    self._update_stats('size', self._get_size())

                return success

            except Exception as e:
                self.logger.error(f"设置缓存值失败 {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        """
        删除缓存值 - 通用模板

        Args:
            key: 缓存键

        Returns:
            bool: 删除是否成功
        """
        with self.lock:
            try:
                if not self._key_exists(key):
                    return False

                success = self._delete_value(key)

                if success:
                    self._update_stats('deletes', 1)
                    self._update_stats('size', self._get_size())

                return success

            except Exception as e:
                self.logger.error(f"删除缓存值失败 {key}: {e}")
                return False

    def exists(self, key: str) -> bool:
        """
        检查键是否存在 - 通用模板

        Args:
            key: 缓存键

        Returns:
            bool: 键是否存在
        """
        with self.lock:
            try:
                exists = self._key_exists(key)
                if exists and self._is_expired(key):
                    self._remove_expired(key)
                    return False
                return exists
            except Exception as e:
                self.logger.error(f"检查键存在失败 {key}: {e}")
                return False

    def clear(self) -> bool:
        """
        清空缓存 - 通用模板

        Returns:
            bool: 清空是否成功
        """
        with self.lock:
            try:
                success = self._clear_all()
                if success:
                    self._reset_stats()
                return success
            except Exception as e:
                self.logger.error(f"清空缓存失败: {e}")
                return False

    def size(self) -> int:
        """
        获取缓存大小 - 通用模板

        Returns:
            int: 缓存项数量
        """
        with self.lock:
            try:
                return self._get_size()
            except Exception as e:
                self.logger.error(f"获取缓存大小失败: {e}")
                return 0

    # 抽象方法 - 各层级需要实现
    def _key_exists(self, key: str) -> bool:
        """检查键是否存在 - 抽象方法"""
        raise NotImplementedError

    def _is_expired(self, key: str) -> bool:
        """检查键是否过期 - 抽象方法"""
        raise NotImplementedError

    def _get_value(self, key: str) -> Any:
        """获取值 - 抽象方法"""
        raise NotImplementedError

    def _set_value(self, key: str, value: Any, ttl: Optional[int]) -> bool:
        """设置值 - 抽象方法"""
        raise NotImplementedError

    def _delete_value(self, key: str) -> bool:
        """删除值 - 抽象方法"""
        raise NotImplementedError

    def _clear_all(self) -> bool:
        """清空所有 - 抽象方法"""
        raise NotImplementedError

    def _get_size(self) -> int:
        """获取大小 - 抽象方法"""
        raise NotImplementedError

    def _should_evict(self) -> bool:
        """是否需要驱逐 - 可重写"""
        # 默认实现：检查容量限制
        capacity = getattr(self.config, 'capacity', float('inf'))
        return self._get_size() >= capacity

    def _evict_oldest(self) -> None:
        """驱逐最旧项 - 可重写"""
        # 默认实现：简单的FIFO策略
        # 各层级可以实现更复杂的策略如LRU

    def _remove_expired(self, key: str) -> None:
        """移除过期项 - 可重写"""
        try:
            self._delete_value(key)
        except Exception:
            pass

    def _update_access_time(self, key: str) -> None:
        """更新访问时间 - 可重写"""

    def _update_stats(self, stat_name: str, value: Any) -> None:
        """更新统计信息"""
        if isinstance(value, (int, float)):
            if stat_name == 'size':
                self.stats['size'] = value
            else:
                self.stats[stat_name] = self.stats.get(stat_name, 0) + value
        else:
            self.stats[stat_name] = value

    def _reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {'size': 0}

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = dict(self.stats)

        # 计算派生统计信息
        total_requests = stats.get('hits', 0) + stats.get('misses', 0)
        if total_requests > 0:
            stats['total_requests'] = total_requests
            stats['hit_rate'] = stats.get('hits', 0) / total_requests
            stats['miss_rate'] = stats.get('misses', 0) / total_requests

        return stats


# 导出所有Mixin类
__all__ = [
    'MonitoringMixin',
    'CRUDOperationsMixin',
    'ComponentLifecycleMixin',
    'CacheTierMixin'
]
