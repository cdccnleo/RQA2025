"""
optimized_connection_pool 模块

提供 optimized_connection_pool 相关功能和接口。
"""

import logging

# -*- coding: utf-8 -*-
import threading
import time

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Callable, Tuple
"""
基础设施层 - 日志系统组件

optimized_connection_pool 模块

日志系统相关的文件
提供日志系统相关的功能实现。
"""

#!/usr/bin/env python3
"""
优化的连接池管理器
实现动态调整、连接泄漏检测和健康检查
"""

# 连接池常量


class ConnectionPoolConstants:
    """连接池相关常量"""

    # 默认池大小配置
    DEFAULT_MIN_SIZE = 5
    DEFAULT_MAX_SIZE = 20
    DEFAULT_INITIAL_SIZE = 10

    # 时间配置 (秒)
    DEFAULT_CONNECTION_TIMEOUT = 30.0
    DEFAULT_IDLE_TIMEOUT = 300.0  # 5分钟
    DEFAULT_MAX_LIFETIME = 3600.0  # 1小时

    # 检测和健康检查配置 (秒)
    DEFAULT_LEAK_DETECTION_THRESHOLD = 60.0
    DEFAULT_HEALTH_CHECK_INTERVAL = 30.0

    # 重试和错误处理
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0

    # 清理配置
    CLEANUP_BATCH_SIZE = 10
    MAX_CLEANUP_ITERATIONS = 50


class PoolState(Enum):
    """连接池状态"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class ConnectionInfo:
    """连接信息"""

    connection_id: str
    created_at: datetime
    last_used: datetime
    use_count: int
    is_active: bool
    error_count: int = 0
    last_error: Optional[str] = None
    connection: Optional[Any] = None


class OptimizedConnectionPool:
    """优化的连接池管理器"""

    def __init__(
        self,
        min_size: int = ConnectionPoolConstants.DEFAULT_MIN_SIZE,
        max_size: int = ConnectionPoolConstants.DEFAULT_MAX_SIZE,
        initial_size: int = ConnectionPoolConstants.DEFAULT_INITIAL_SIZE,
        connection_timeout: float = ConnectionPoolConstants.DEFAULT_CONNECTION_TIMEOUT,
        idle_timeout: float = ConnectionPoolConstants.DEFAULT_IDLE_TIMEOUT,
        max_lifetime: float = ConnectionPoolConstants.DEFAULT_MAX_LIFETIME,
        leak_detection_threshold: float = ConnectionPoolConstants.DEFAULT_LEAK_DETECTION_THRESHOLD,
        health_check_interval: float = ConnectionPoolConstants.DEFAULT_HEALTH_CHECK_INTERVAL,
        max_usage: int = None,
        leak_detection: bool = False,
    ):
        """
            初始化连接池

            Args:
        min_size: 最小连接数
        max_size: 最大连接数
        initial_size: 初始连接数
        connection_timeout: 连接超时时间
        idle_timeout: 空闲超时时间
        max_lifetime: 连接最大生命周期
        leak_detection_threshold: 泄漏检测阈值
        health_check_interval: 健康检查间隔
        max_usage: 连接最大使用次数
        leak_detection: 是否启用泄漏检测
        """
        # 初始化池参数
        self._initialize_pool_parameters(
            min_size, max_size, initial_size, connection_timeout,
            idle_timeout, max_lifetime, leak_detection_threshold,
            health_check_interval, max_usage, leak_detection
        )

        # 初始化数据结构
        self._initialize_data_structures()

        # 初始化监控和统计
        self._initialize_monitoring()

        # 启动维护线程
        self._start_maintenance_thread()

    def _initialize_pool_parameters(
        self,
        min_size: int,
        max_size: int,
        initial_size: int,
        connection_timeout: float,
        idle_timeout: float,
        max_lifetime: float,
        leak_detection_threshold: float,
        health_check_interval: float,
        max_usage: int,
        leak_detection: bool,
    ):
        """初始化连接池参数"""
        self._min_size = min_size
        self._max_size = max_size
        self._initial_size = initial_size
        self._connection_timeout = connection_timeout
        self._idle_timeout = idle_timeout
        self._max_lifetime = max_lifetime
        self._leak_detection_threshold = leak_detection_threshold
        self._health_check_interval = health_check_interval
        self._max_usage = max_usage
        self._leak_detection = leak_detection

    def _initialize_data_structures(self):
        """初始化数据结构"""
        self._connections: deque = deque()
        self._available_connections: deque = deque()
        self._in_use_connections: Dict[str, ConnectionInfo] = {}
        self._connection_factory: Optional[Callable] = None
        self._connection_validator: Optional[Callable] = None
        self._lock = threading.RLock()

    def _initialize_monitoring(self):
        """初始化监控和统计"""
        try:
            # 导入新的组件
            from .connection_health_checker import ConnectionHealthChecker
            from .connection_pool_monitor import ConnectionPoolMonitor
            from .connection_lifecycle_manager import ConnectionLifecycleManager
            
            self._health_checker = ConnectionHealthChecker(self._connection_validator)
            self._monitor = ConnectionPoolMonitor(self._leak_detection_threshold)
            self._lifecycle_manager = ConnectionLifecycleManager(
                self._connection_factory, self._idle_timeout, 
                self._max_lifetime, self._max_usage
            )
            self.COMPONENTS_AVAILABLE = True
            
        except ImportError as e:
            logger.warning(f"无法导入连接池组件，使用兼容模式: {e}")
            self._health_checker = None
            self._monitor = None
            self._lifecycle_manager = None
            self.COMPONENTS_AVAILABLE = False
        
        # 向后兼容的属性
        self._stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "failed_connections": 0,
            "connection_requests": 0,
            "connection_timeouts": 0,
            "leak_detections": 0,
        }
        self._state = PoolState.HEALTHY
        self._last_health_check = datetime.now()
        self._logger = logging.getLogger("connection_pool")

    def _start_maintenance_thread(self):
        """启动维护线程"""
        self._maintenance_thread = threading.Thread(target=self._maintenance_worker, daemon=True)
        self._maintenance_thread.start()

        # 属性访问器

    @property
    def min_size(self):
        """获取最小连接数"""
        return self._min_size

    @property
    def max_size(self):
        """获取最大连接数"""
        return self._max_size

    @property
    def idle_timeout(self):
        """获取空闲超时时间"""
        return self._idle_timeout

    @property
    def max_usage(self):
        """获取最大使用次数"""
        return self._max_usage

    @property
    def connection_factory(self):
        """获取连接工厂函数"""
        return self._connection_factory

    @property
    def connection_validator(self):
        """获取连接验证器函数"""
        return self._connection_validator

    @property
    def connections(self):
        """获取所有连接"""
        return self._connections

    @property
    def available_connections(self):
        """获取可用连接"""
        return self._available_connections

    def set_connection_factory(self, factory: Callable) -> None:
        """
            设置连接工厂函数

        Args:
        factory: 连接工厂函数
        """
        self._connection_factory = factory
        # 设置工厂后，尝试创建最小数量的连接
        self._ensure_min_connections()

    def set_connection_validator(self, validator: Callable) -> None:
        """
        设置连接验证函数

        Args:
        validator: 连接验证函数
        """
        self._connection_validator = validator

    def get_connection(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        获取连接

        Args:
        timeout: 超时时间

        Returns:
        数据库连接，如果获取失败则返回None
        """
        # 初始化连接请求
        start_time, effective_timeout = self._initialize_connection_request(timeout)

        # 尝试获取可用连接
        available_connection = self._try_get_available_connection()
        if available_connection:
            return available_connection

        # 尝试创建新连接
        new_connection = self._try_create_new_connection()
        if new_connection:
            return new_connection

        # 等待可用连接
        waited_connection = self._wait_for_available_connection(effective_timeout, start_time)
        if waited_connection:
            return waited_connection

        # 处理超时
        return self._handle_connection_timeout(effective_timeout)

    def _initialize_connection_request(self, timeout: Optional[float]) -> Tuple[float, float]:
        """初始化连接请求"""
        effective_timeout = timeout if timeout is not None else self._connection_timeout
        start_time = time.time()

        with self._lock:
            self._stats["connection_requests"] += 1

        return start_time, effective_timeout

    def _try_get_available_connection(self) -> Optional[Any]:
        """尝试获取可用连接"""
        while self._available_connections:
            connection_info = self._available_connections.popleft()

            # 检查连接是否有效
            if self._is_connection_valid(connection_info):
                return self._activate_connection(connection_info)
            else:
                # 移除无效连接
                self._remove_connection(connection_info)

        return None

    def _activate_connection(self, connection_info: ConnectionInfo) -> Any:
        """激活连接并返回连接对象"""
        connection_info.last_used = datetime.now()
        connection_info.use_count += 1
        connection_info.is_active = True

        self._in_use_connections[connection_info.connection_id] = connection_info
        self._stats["active_connections"] = len(self._in_use_connections)
        self._stats["idle_connections"] = len(self._available_connections)

        return self._get_connection_object(connection_info)

    def _try_create_new_connection(self) -> Optional[Any]:
        """尝试创建新连接"""
        if len(self._connections) < self._max_size:
            connection_info = self._create_connection()
            if connection_info:
                return self._activate_connection(connection_info)
        return None

    def _wait_for_available_connection(self, timeout: float, start_time: float) -> Optional[Any]:
        """等待可用连接"""
        while time.time() - start_time < timeout:
            # 临时释放锁，让其他线程可以归还连接
            self._lock.release()
            try:
                time.sleep(0.01)  # 减少等待时间
            finally:
                # 重新获取锁
                self._lock.acquire()

            if self._available_connections:
                connection_info = self._available_connections.popleft()
                if self._is_connection_valid(connection_info):
                    return self._activate_connection(connection_info)

        return None

    def _handle_connection_timeout(self, timeout: float) -> None:
        """处理连接超时"""
        self._stats["connection_timeouts"] += 1
        self._logger.warning(f"获取连接超时: {timeout}秒")
        return None

    def release_connection(self, connection: Any) -> None:
        """
        释放连接

        Args:
        connection: 数据库连接
        """
        with self._lock:
            connection_info = self._find_connection_info(connection)
            if connection_info:
                connection_info.is_active = False
                connection_info.last_used = datetime.now()

                # 检查连接是否仍然有效
                if self._is_connection_valid(connection_info):
                    self._available_connections.append(connection_info)
                    self._stats["idle_connections"] = len(self._available_connections)
                else:
                    self._remove_connection(connection_info)

                # 从使用中移除
                if connection_info.connection_id in self._in_use_connections:
                    del self._in_use_connections[connection_info.connection_id]
                    self._stats["active_connections"] = len(self._in_use_connections)

    def get_pool_status(self) -> Dict[str, Any]:
        """
        获取连接池状态

        Returns:
        连接池状态信息
        """
        with self._lock:
            # 使用监控器组件获取统计
            if self._monitor:
                monitor_stats = self._monitor.get_statistics(
                    list(self._connections),
                    len(self._available_connections),
                    len(self._in_use_connections)
                )
                
                return {
                    "state": self._state.value,
                    "min_size": self._min_size,
                    "max_size": self._max_size,
                    "current_size": len(self._connections),
                    **monitor_stats,
                    "last_health_check": self._last_health_check.isoformat(),
                    "utilization_rate": len(self._in_use_connections) / max(len(self._connections), 1),
                }
            
            # 回退到原有方法
            return {
                "state": self._state.value,
                "min_size": self._min_size,
                "max_size": self._max_size,
                "current_size": len(self._connections),
                "total_connections": len(self._connections),
                "active_connections": len(self._in_use_connections),
                "available_connections": len(self._available_connections),
                "available_size": len(self._available_connections),
                "in_use_size": len(self._in_use_connections),
                "stats": self._stats.copy(),
                "last_health_check": self._last_health_check.isoformat(),
                "utilization_rate": len(self._in_use_connections) / max(len(self._connections), 1),
            }

    def resize_pool(self, new_min_size: int, new_max_size: int) -> None:
        """
        调整连接池大小

        Args:
        new_min_size: 新的最小连接数
        new_max_size: 新的最大连接数
        """
        with self._lock:
            self._min_size = new_min_size
            self._max_size = new_max_size

            # 如果当前连接数超过新的最大值，关闭多余的连接
            while len(self._connections) > new_max_size:
                if self._available_connections:
                    connection_info = self._available_connections.popleft()
                    self._remove_connection(connection_info)
                else:
                    break

            self._logger.info(f"连接池大小已调整: min={new_min_size}, max={new_max_size}")

    def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查

        Returns:
        健康检查结果
        """
        with self._lock:
            # 使用健康检查器组件
            if self._health_checker:
                return self._health_checker.health_check(
                    list(self._connections),
                    self._available_connections,
                    self._in_use_connections,
                    self._max_size
                )
            
            # 回退到原有方法
            # 更新健康检查时间
            self._update_health_check_time()

            # 评估连接池健康状态
            pool_metrics = self._assess_pool_health()

            # 验证所有连接
            valid_connections = self._validate_all_connections()

            # 计算错误率
            error_rate = self._calculate_error_rate()

            # 构建健康检查结果
            return self._build_health_check_result(pool_metrics, valid_connections, error_rate)

    def _update_health_check_time(self):
        """更新健康检查时间"""
        self._last_health_check = datetime.now()

    def _assess_pool_health(self) -> Dict[str, int]:
        """评估连接池健康状态"""
        total_connections = len(self._connections)
        available_connections = len(self._available_connections)
        in_use_connections = len(self._in_use_connections)

        # 计算健康状态
        if total_connections == 0:
            self._state = PoolState.FAILED
        elif in_use_connections >= self._max_size * 0.9:
            self._state = PoolState.CRITICAL
        elif in_use_connections >= self._max_size * 0.7:
            self._state = PoolState.WARNING
        else:
            self._state = PoolState.HEALTHY

        return {
            "total_connections": total_connections,
            "available_connections": available_connections,
            "in_use_connections": in_use_connections,
        }

    def _validate_all_connections(self) -> int:
        """验证所有连接，返回有效连接数量"""
        valid_connections = 0
        for connection_info in list(self._connections):
            if self._is_connection_valid(connection_info):
                valid_connections += 1
            else:
                self._remove_connection(connection_info)
        return valid_connections

    def _calculate_error_rate(self) -> float:
        """计算错误率"""
        total_requests = self._stats.get("connection_requests", 0)
        failed_connections = self._stats.get("failed_connections", 0)
        return failed_connections / max(total_requests, 1)

    def _build_health_check_result(
        self,
        pool_metrics: Dict[str, int],
        valid_connections: int,
        error_rate: float
    ) -> Dict[str, Any]:
        """构建健康检查结果"""
        return {
            "status": self._state.value,
            "state": self._state.value,
            "total_connections": pool_metrics["total_connections"],
            "active_connections": pool_metrics["in_use_connections"],
            "valid_connections": valid_connections,
            "available_connections": pool_metrics["available_connections"],
            "in_use_connections": pool_metrics["in_use_connections"],
            "error_rate": error_rate,
            "utilization_rate": pool_metrics["in_use_connections"] / max(pool_metrics["total_connections"], 1),
            "last_check": self._last_health_check.isoformat(),
        }

    def _create_connection(self) -> Optional[ConnectionInfo]:
        """
        创建新连接

        Returns:
        连接信息，如果创建失败则返回None
        """
        if not self._connection_factory:
            # 仅在首次遇到时记录，避免日志泛滥
            if not hasattr(self, '_factory_warning_logged'):
                self._logger.debug("连接工厂函数未设置，跳过连接创建")
                self._factory_warning_logged = True
            return None

        try:
            connection = self._connection_factory()
            if connection:
                connection_id = f"conn_{len(self._connections)}_{int(time.time())}"
                connection_info = ConnectionInfo(
                    connection_id=connection_id,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    use_count=0,
                    is_active=False,
                    connection=connection,
                )

                self._connections.append(connection_info)
                self._available_connections.append(connection_info)
                self._stats["total_connections"] = len(self._connections)
                self._stats["idle_connections"] = len(self._available_connections)

                return connection_info
            else:
                self._stats["failed_connections"] += 1
                return None

        except Exception as e:
            self._logger.error(f"创建连接失败: {e}")
            self._stats["failed_connections"] += 1
            return None

    def _is_connection_valid(self, connection_info: ConnectionInfo) -> bool:
        """
        检查连接是否有效

        Args:
        connection_info: 连接信息

        Returns:
        连接是否有效
        """
        # 检查连接生命周期
        if (datetime.now() - connection_info.created_at).total_seconds() > self._max_lifetime:
            return False

        # 检查空闲超时
        if (datetime.now() - connection_info.last_used).total_seconds() > self._idle_timeout:
            return False

        # 检查错误次数
        if connection_info.error_count > 3:
            return False

        # 检查使用次数限制
        if self._max_usage is not None and connection_info.use_count >= self._max_usage:
            return False

        # 使用验证器检查连接
        if self._connection_validator:
            try:
                connection = self._get_connection_object(connection_info)
                return self._connection_validator(connection)
            except Exception:
                connection_info.error_count += 1
                return False

        return True

    def _remove_connection(self, connection_info: ConnectionInfo) -> None:
        """
        移除连接

        Args:
        connection_info: 连接信息
        """
        # 从所有集合中移除
        if connection_info in self._connections:
            self._connections.remove(connection_info)

        if connection_info in self._available_connections:
            self._available_connections.remove(connection_info)

        if connection_info.connection_id in self._in_use_connections:
            del self._in_use_connections[connection_info.connection_id]

        # 更新统计
        self._stats["total_connections"] = len(self._connections)
        self._stats["idle_connections"] = len(self._available_connections)
        self._stats["active_connections"] = len(self._in_use_connections)

    def _find_connection_info(self, connection: Any) -> Optional[ConnectionInfo]:
        """
        根据连接对象查找连接信息

        Args:
        connection: 连接对象

        Returns:
        连接信息
        """
        # 根据连接对象查找连接信息
        for connection_info in self._connections:
            if connection_info.connection == connection:
                return connection_info
        return None

    def _get_connection_object(self, connection_info: ConnectionInfo) -> Any:
        """
        获取连接对象

        Args:
        connection_info: 连接信息

        Returns:
        连接对象
        """
        # 返回存储在ConnectionInfo中的连接对象
        return connection_info.connection

    def _maintenance_worker(self) -> None:
        """维护工作线程"""
        while True:
            try:
                # 定期健康检查
                self.health_check()

                # 清理过期连接
                self._cleanup_expired_connections()

                # 检测连接泄漏
                self._detect_connection_leaks()

                # 确保最小连接数
                self._ensure_min_connections()

                time.sleep(self._health_check_interval)

            except Exception as e:
                self._logger.error(f"维护线程异常: {e}")
                time.sleep(5)

    def _cleanup_expired_connections(self) -> None:
        """清理过期连接"""
        with self._lock:
            expired_connections = []

            for connection_info in self._connections:
                if not self._is_connection_valid(connection_info):
                    expired_connections.append(connection_info)

            for connection_info in expired_connections:
                self._remove_connection(connection_info)

    def _detect_connection_leaks(self) -> None:
        """检测连接泄漏"""
        with self._lock:
            current_time = datetime.now()
            leaked_connections = []

            for connection_info in list(self._in_use_connections.values()):
                if connection_info.is_active:
                    idle_time = (current_time - connection_info.last_used).total_seconds()
                    if idle_time > self._leak_detection_threshold:
                        leaked_connections.append(connection_info)
                        self._stats["leak_detections"] += 1
                        self._logger.warning(f"检测到连接泄漏: {connection_info.connection_id}")

            # 强制释放泄漏的连接
            for connection_info in leaked_connections:
                connection_info.is_active = False
                if connection_info.connection_id in self._in_use_connections:
                    del self._in_use_connections[connection_info.connection_id]
                    self._available_connections.append(connection_info)
                    self._stats["idle_connections"] = len(self._available_connections)

    def _ensure_min_connections(self) -> None:
        """确保最小连接数"""
        # 如果没有连接工厂，不尝试创建连接
        if not self._connection_factory:
            return
            
        with self._lock:
            while len(self._connections) < self._min_size:
                connection_info = self._create_connection()
                if not connection_info:
                    break

                # 将新创建的连接添加到可用连接池
                self._available_connections.append(connection_info)
                self._stats["idle_connections"] = len(self._available_connections)

    def shutdown(self) -> bool:
        """关闭连接池"""
        try:
            with self._lock:
                # 停止维护线程
                if hasattr(self, "_maintenance_thread") and self._maintenance_thread.is_alive():
                    # 注意：这里只是设置标志，实际的守护线程会在程序结束时自动停止
                    pass

                # 清理所有连接
                for conn_info in list(self._connections):
                    try:
                        # 如果连接有close方法，调用它
                        if hasattr(conn_info.connection, "close"):
                            conn_info.connection.close()
                    except Exception as e:
                        self._logger.warning(f"关闭连接失败: {e}")

                # 清空连接池
                self._connections.clear()
                self._available_connections.clear()
                self._in_use_connections.clear()

            self._logger.info("连接池已关闭")
            return True

        except Exception as e:
            self._logger.error(f"关闭连接池失败: {e}")
            return False
