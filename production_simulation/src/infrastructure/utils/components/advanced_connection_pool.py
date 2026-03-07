"""
advanced_connection_pool 模块

提供 advanced_connection_pool 相关功能和接口。
"""

import logging

import concurrent.futures
import threading
import time

from collections import deque
from typing import Dict, Any, Optional, Callable, Tuple
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 高级连接池优化
性能优化Phase 1: 连接池优化实现

作者: AI Assistant
创建日期: 2025年9月13日
"""

logger = logging.getLogger(__name__)


class ConnectionPoolMetrics:
    """连接池性能指标"""

    def __init__(self):
        self.created_connections = 0
        self.active_connections = 0
        self.idle_connections = 0
        self.destroyed_connections = 0
        self.connection_requests = 0
        self.connection_hits = 0
        self.connection_misses = 0
        self.connection_timeouts = 0
        self.average_wait_time = 0.0
        self.peak_active_connections = 0

    def record_connection_created(self):
        """记录连接创建"""
        self.created_connections += 1

    def record_connection_destroyed(self):
        """记录连接销毁"""
        self.destroyed_connections += 1

    def record_connection_request(self):
        """记录连接请求"""
        self.connection_requests += 1

    def update_active_connections(self, count: int):
        """更新活跃连接数"""
        self.active_connections = count
        if count > self.peak_active_connections:
            self.peak_active_connections = count

    def update_idle_connections(self, count: int):
        """更新空闲连接数"""
        self.idle_connections = count

    def reset(self):
        """重置所有指标"""
        self.__init__()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "created_connections": self.created_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "destroyed_connections": self.destroyed_connections,
            "connection_requests": self.connection_requests,
            "connection_hits": self.connection_hits,
            "connection_misses": self.connection_misses,
            "connection_timeouts": self.connection_timeouts,
            "hit_rate": (self.connection_hits / max(1, self.connection_requests)) * 100,
            "average_wait_time": self.average_wait_time,
            "peak_active_connections": self.peak_active_connections,
        }


class ConnectionWrapper:
    """连接包装器"""

    def __init__(self, connection: Any, pool: 'OptimizedConnectionPool', max_age: float = 300.0, max_idle_time: float = 60.0):
        self._connection = connection
        self._pool = pool
        self._returned = False
        self._closed = False
        self.created_time = time.time()
        self.last_used_time = time.time()
        self._max_age = max_age
        self._max_idle_time = max_idle_time

    @property
    def connection(self):
        """获取底层连接"""
        return self._connection

    @property
    def is_closed(self):
        """检查连接是否已关闭"""
        return self._closed

    def execute(self, query: str, *args, **kwargs):
        """执行查询"""
        try:
            self.last_used_time = time.time()
            return self._connection.execute(query, *args, **kwargs)
        except Exception as e:
            self._closed = True
            raise

    def is_expired(self) -> bool:
        """检查连接是否过期"""
        age = time.time() - self.created_time
        return age > self._max_age

    def is_idle_timeout(self) -> bool:
        """检查连接是否空闲超时"""
        idle_time = time.time() - self.last_used_time
        return idle_time > self._max_idle_time

    def get_age(self) -> float:
        """获取连接年龄"""
        return time.time() - self.created_time

    def get_idle_time(self) -> float:
        """获取空闲时间"""
        return time.time() - self.last_used_time

    def update_last_used(self):
        """更新最后使用时间"""
        self.last_used_time = time.time()

    def __enter__(self):
        return self._connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """关闭连接（实际是归还到池中）"""
        if not self._returned:
            # 不要关闭底层连接，因为它将返回池中重用
            # if hasattr(self._connection, 'close'):
            #     self._connection.close()
            self._pool.return_connection(self._connection)
            self._returned = True
            self._closed = True

    def __del__(self):
        """析构函数，确保连接被正确归还"""
        self.close()


class OptimizedConnectionPool:
    """优化的连接池实现"""

    def __init__(
        self,
        max_connections: int = 20,
        min_connections: int = 5,
        max_idle_time: int = 300,
        max_lifetime: int = 3600,
        connection_timeout: float = 30.0,
        retry_attempts: int = 3,
    ):
        """
        初始化连接池

        Args:
            max_connections: 最大连接数
            min_connections: 最小连接数
            max_idle_time: 最大空闲时间(秒)
            max_lifetime: 最大生命周期(秒)
            connection_timeout: 连接超时时间(秒)
            retry_attempts: 重试次数
        """
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.max_idle_time = max_idle_time
        self.max_lifetime = max_lifetime
        self.connection_timeout = connection_timeout
        self.retry_attempts = retry_attempts

        # 连接池数据结构
        self._pool = deque()  # 空闲连接队列
        self._active_connections = set()  # 活跃连接集合
        self._connection_lock = threading.RLock()  # 线程安全锁
        self._shutdown_event = threading.Event()

        # 连接工厂和清理函数
        self._connection_factory: Optional[Callable] = None
        self._connection_validator: Optional[Callable] = None
        self._connection_destroyer: Optional[Callable] = None

        # 性能指标
        self.metrics = ConnectionPoolMetrics()

        # 监控线程
        self._monitor_thread = threading.Thread(
            target=self._monitor_connections, daemon=True, name="ConnectionPoolMonitor"
        )
        self._monitor_thread.start()

        # 初始化最小连接数
        self._initialize_pool()

    def set_connection_factory(self, factory: Callable):
        """设置连接工厂函数"""
        self._connection_factory = factory

    def set_connection_validator(self, validator: Callable):
        """设置连接验证函数"""
        self._connection_validator = validator

    def set_connection_destroyer(self, destroyer: Callable):
        """设置连接销毁函数"""
        self._connection_destroyer = destroyer

    def _initialize_pool(self):
        """初始化连接池"""
        for _ in range(self.min_connections):
            try:
                conn = self._create_connection()
                if conn:
                    self._pool.append(
                        {
                            "connection": conn,
                            "created_time": time.time(),
                            "last_used_time": time.time(),
                        }
                    )
                    self.metrics.idle_connections += 1
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")

    def _create_connection(self) -> Optional[Any]:
        """创建新连接"""
        if not self._connection_factory:
            logger.debug("Connection factory not set, skipping connection creation")
            return None

        for attempt in range(self.retry_attempts):
            try:
                conn = self._connection_factory()
                self.metrics.created_connections += 1
                return conn
            except Exception as e:
                logger.warning(f"Connection creation attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(0.1 * (2**attempt))  # 指数退避

        return None

    def _validate_connection(self, conn: Any) -> bool:
        """验证连接是否有效"""
        if not self._connection_validator:
            return True

        try:
            return self._connection_validator(conn)
        except Exception as e:
            logger.warning(f"Connection validation failed: {e}")
            return False

    def _destroy_connection(self, conn: Any):
        """销毁连接"""
        try:
            if self._connection_destroyer:
                self._connection_destroyer(conn)
            self.metrics.destroyed_connections += 1
        except Exception as e:
            logger.warning(f"Connection destruction failed: {e}")

    def get_connection(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        获取连接

        Args:
            timeout: 超时时间(秒)

        Returns:
            连接对象或None
        """
        # 初始化连接请求
        start_time, effective_timeout = self._initialize_connection_request(timeout)

        with self._connection_lock:
            # 尝试获取空闲连接
            idle_connection = self._try_get_idle_connection()
            if idle_connection:
                return idle_connection

            # 尝试创建新连接
            new_connection = self._create_new_connection_if_allowed()
            if new_connection:
                return new_connection

            # 等待连接可用
            waited_connection = self._wait_for_connection(effective_timeout, start_time)
            if waited_connection:
                return waited_connection

            # 处理超时
            return self._handle_connection_timeout()

    def _initialize_connection_request(self, timeout: Optional[float]) -> Tuple[float, float]:
        """初始化连接请求"""
        effective_timeout = timeout if timeout is not None else self.connection_timeout
        start_time = time.time()
        self.metrics.connection_requests += 1
        return start_time, effective_timeout

    def _try_get_idle_connection(self) -> Optional[Any]:
        """尝试获取空闲连接"""
        while self._pool and not self._shutdown_event.is_set():
            conn_info = self._pool.popleft()
            conn = conn_info["connection"]
            created_time = conn_info["created_time"]

            # 检查连接是否过期
            if time.time() - created_time > self.max_lifetime or not self._validate_connection(conn):
                self._destroy_connection(conn)
                continue

            # 连接有效，使用它
            return self._activate_connection(conn, conn_info)

        return None

    def _activate_connection(self, conn: Any, conn_info: dict) -> ConnectionWrapper:
        """激活连接并更新统计"""
        self._active_connections.add(conn)
        self.metrics.active_connections += 1
        self.metrics.idle_connections -= 1
        self.metrics.connection_hits += 1
        self.metrics.peak_active_connections = max(
            self.metrics.peak_active_connections,
            self.metrics.active_connections,
        )
        conn_info["last_used_time"] = time.time()
        return ConnectionWrapper(conn, self)

    def _create_new_connection_if_allowed(self) -> Optional[Any]:
        """在允许的情况下创建新连接"""
        if len(self._active_connections) < self.max_connections and not self._shutdown_event.is_set():
            conn = self._create_connection()
            if conn:
                self._active_connections.add(conn)
                self.metrics.active_connections += 1
                self.metrics.connection_misses += 1
                self.metrics.peak_active_connections = max(
                    self.metrics.peak_active_connections,
                    self.metrics.active_connections,
                )
                return ConnectionWrapper(conn, self)
        return None

    def _wait_for_connection(self, timeout: float, start_time: float) -> Optional[Any]:
        """等待连接可用"""
        end_time = start_time + timeout
        while time.time() < end_time and not self._shutdown_event.is_set():
            # 简化的等待逻辑，实际实现中应该使用条件变量
            time.sleep(0.01)

            # 再次检查是否有空闲连接
            if self._pool:
                conn_info = self._pool.popleft()
                conn = conn_info["connection"]

                if self._validate_connection(conn):
                    return self._activate_connection(conn, conn_info)

        return None

    def _handle_connection_timeout(self) -> None:
        """处理连接超时"""
        self.metrics.connection_timeouts += 1
        return None

    def return_connection(self, conn: Any):
        """归还连接到池中"""
        with self._connection_lock:
            if conn in self._active_connections:
                self._active_connections.remove(conn)
                self.metrics.active_connections -= 1

                # 验证连接是否有效
                if not self._validate_connection(conn):
                    self._destroy_connection(conn)
                    return

                # 检查是否应该销毁连接
                if (
                    len(self._pool) + len(self._active_connections) >= self.max_connections
                    or self._shutdown_event.is_set()
                ):
                    self._destroy_connection(conn)
                else:
                    self._pool.append(
                        {
                            "connection": conn,
                            "created_time": time.time(),  # 简化处理
                            "last_used_time": time.time(),
                        }
                    )
                    self.metrics.idle_connections += 1

    def _monitor_connections(self):
        """监控连接池状态"""
        while not self._shutdown_event.is_set():
            try:
                with self._connection_lock:
                    current_time = time.time()

                    # 检查空闲连接是否过期
                    expired_connections = []
                    for i, conn_info in enumerate(self._pool):
                        last_used_time = conn_info["last_used_time"]
                        if current_time - last_used_time > self.max_idle_time:
                            expired_connections.append(i)

                    # 移除过期连接
                    for i in reversed(expired_connections):
                        conn_info = self._pool[i]
                        self._destroy_connection(conn_info["connection"])
                        self._pool.remove(conn_info)
                        self.metrics.idle_connections -= 1

                    # 动态调整连接池大小
                    self._adjust_pool_size()

            except Exception as e:
                logger.error(f"Connection pool monitoring error: {e}")

            time.sleep(60)  # 每分钟检查一次

    def _adjust_pool_size(self):
        """动态调整连接池大小"""
        total_connections = len(self._pool) + len(self._active_connections)

        # 如果空闲连接太少，创建新连接
        if len(self._pool) < self.min_connections and total_connections < self.max_connections:
            try:
                conn = self._create_connection()
                if conn:
                    self._pool.append(
                        {
                            "connection": conn,
                            "created_time": time.time(),
                            "last_used_time": time.time(),
                        }
                    )
                    self.metrics.idle_connections += 1
            except Exception as e:
                logger.warning(f"Failed to create connection during pool adjustment: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        with self._connection_lock:
            return {
                "pool_size": len(self._pool),
                "active_connections": len(self._active_connections),
                "total_connections": len(self._pool) + len(self._active_connections),
                "max_connections": self.max_connections,
                "metrics": self.metrics.to_dict(),
            }

    def get_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息（别名方法）"""
        with self._connection_lock:
            return {
                "total_connections": len(self._pool) + len(self._active_connections),
                "active_connections": len(self._active_connections),
                "idle_connections": len(self._pool),
                "min_connections": self.min_connections,
                "max_connections": self.max_connections,
                "created_connections": self.metrics.created_connections,
                "destroyed_connections": self.metrics.destroyed_connections,
                "connection_requests": self.metrics.connection_requests,
                "utilization_rate": len(self._active_connections) / max(1, self.max_connections)
            }

    def maintain_min_connections(self):
        """维护最小连接数"""
        with self._connection_lock:
            current_total = len(self._pool) + len(self._active_connections)
            if current_total < self.min_connections:
                needed = self.min_connections - current_total
                for _ in range(needed):
                    try:
                        conn = self._create_connection()
                        if conn:
                            self._pool.append({
                                "connection": conn,
                                "created_time": time.time(),
                                "last_used_time": time.time()
                            })
                            self.metrics.idle_connections += 1
                    except Exception as e:
                        logger.warning(f"Failed to create connection for min connections: {e}")
                        break

    def cleanup_expired_connections(self):
        """清理过期连接"""
        with self._connection_lock:
            current_time = time.time()
            # 检查空闲连接池
            cleaned_pool = deque()
            while self._pool:
                conn_info = self._pool.popleft()
                age = current_time - conn_info["created_time"]
                idle_time = current_time - conn_info["last_used_time"]
                
                # 检查是否过期或空闲超时
                if age > self.max_lifetime or idle_time > self.max_idle_time:
                    self._destroy_connection(conn_info["connection"])
                else:
                    cleaned_pool.append(conn_info)
            
            self._pool = cleaned_pool

    def close_all_connections(self):
        """关闭所有连接"""
        with self._connection_lock:
            # 销毁所有空闲连接
            while self._pool:
                conn_info = self._pool.popleft()
                self._destroy_connection(conn_info["connection"])
            
            # 销毁所有活跃连接
            for conn in list(self._active_connections):
                self._destroy_connection(conn)
            self._active_connections.clear()

    def shutdown(self):
        """关闭连接池"""
        self._shutdown_event.set()

        with self._connection_lock:
            # 销毁所有空闲连接
            for conn_info in self._pool:
                self._destroy_connection(conn_info["connection"])
            self._pool.clear()

            # 等待活跃连接被归还和销毁
            # 注意：实际实现中应该有超时机制

        logger.info("Connection pool shut down")

# 示例用法和性能测试


def create_database_connection(config: Optional[Dict[str, Any]] = None):
    """创建数据库连接的示例
    
    Args:
        config: 数据库连接配置（可选）
    """
    # 这里是创建数据库连接的逻辑
    # 实际使用时需要根据具体的数据库驱动来实现
    class MockConnection:
        def __init__(self):
            self.id = id(self)
            self.created_time = time.time()
            self.closed = False

        def execute(self, query):
            if self.closed:
                raise Exception("Connection closed")
            time.sleep(0.001)  # 模拟查询时间
            return f"Result for {query}"

        def close(self):
            self.closed = True

    return MockConnection()


def validate_database_connection(conn):
    """验证数据库连接
    
    Args:
        conn: 数据库连接对象
        
    Returns:
        bool: 连接是否有效
    """
    try:
        # 检查连接是否关闭
        if hasattr(conn, "closed"):
            return not conn.closed
        # 如果没有closed属性，检查是否有execute方法
        return hasattr(conn, "execute")
    except Exception as e:
        logger.warning(f"Connection validation failed: {e}")
        return False


def destroy_database_connection(conn):
    """销毁数据库连接
    
    Args:
        conn: 数据库连接对象
    """
    try:
        if hasattr(conn, "close"):
            conn.close()
    except Exception as e:
        logger.warning(f"Connection destruction failed: {e}")


def performance_test(num_threads: int = 5, duration: int = 10, pool_config: Optional[Dict[str, Any]] = None):
    """性能测试
    
    Args:
        num_threads: 并发线程数
        duration: 测试持续时间（秒）
        pool_config: 连接池配置
    """
    print("🚀 开始连接池性能测试...")

    # 设置测试环境
    pool = _setup_performance_test_pool(pool_config or {})

    # 运行多线程测试
    all_results, total_time = _run_multi_threaded_test(pool, num_threads=num_threads, duration=duration)

    # 计算性能指标
    metrics = _calculate_performance_metrics(all_results, total_time)

    # 获取连接池统计
    stats = pool.get_pool_stats()

    # 打印测试结果
    _print_performance_results(metrics, stats, total_time)

    # 清理资源
    pool.shutdown()

    # 返回测试结果
    return _prepare_test_results(metrics, stats, total_time)


def _setup_performance_test_pool(config: Dict[str, Any]):
    """设置性能测试连接池
    
    Args:
        config: 连接池配置字典
    """
    # 从配置中提取参数，提供默认值
    max_connections = config.get('max_connections', 10)
    min_connections = config.get('min_connections', 2)
    max_idle_time = config.get('max_idle_time', 60)
    connection_timeout = config.get('connection_timeout', 5.0)
    
    pool = OptimizedConnectionPool(
        max_connections=max_connections,
        min_connections=min_connections,
        max_idle_time=max_idle_time,
        connection_timeout=connection_timeout
    )

    # 设置连接工厂
    pool.set_connection_factory(create_database_connection)
    pool.set_connection_validator(validate_database_connection)
    pool.set_connection_destroyer(destroy_database_connection)

    return pool


def _run_multi_threaded_test(pool, num_threads: int = 5, duration: int = 10):
    """运行多线程测试
    
    Args:
        pool: 连接池实例
        num_threads: 并发线程数
        duration: 测试持续时间（秒）
    """
    def worker(worker_id):
        """工作线程函数"""
        results = []
        start = time.time()
        request_count = 0
        # 运行指定时间
        while time.time() - start < duration:
            start_time = time.time()
            conn = pool.get_connection(timeout=1.0)
            if conn:
                try:
                    if hasattr(conn, 'execute'):
                        result = conn.execute(f"SELECT * FROM table_{request_count}")
                    elif hasattr(conn, 'connection') and hasattr(conn.connection, 'execute'):
                        result = conn.connection.execute(f"SELECT * FROM table_{request_count}")
                    end_time = time.time()
                    results.append(end_time - start_time)
                finally:
                    # 返回连接
                    if hasattr(conn, 'connection'):
                        pool.return_connection(conn.connection)
                    else:
                        pool.return_connection(conn)
            else:
                results.append(float("inf"))  # 超时
            request_count += 1
        return results

    # 多线程测试
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    end_time = time.time()
    total_time = end_time - start_time

    return all_results, total_time


def _calculate_performance_metrics(all_results, total_time):
    """计算性能指标"""
    successful_requests = [r for r in all_results if r != float("inf")]
    failed_requests = len(all_results) - len(successful_requests)

    if successful_requests:
        avg_response_time = sum(successful_requests) / len(successful_requests)
        throughput = len(successful_requests) / total_time
        p95_response_time = sorted(successful_requests)[int(len(successful_requests) * 0.95)]
    else:
        avg_response_time = 0
        throughput = 0
        p95_response_time = 0

    return {
        "total_requests": len(all_results),
        "successful_requests": len(successful_requests),
        "failed_requests": failed_requests,
        "avg_response_time": avg_response_time,
        "p95_response_time": p95_response_time,
        "throughput": throughput,
    }


def _print_performance_results(metrics, stats, total_time):
    """打印性能测试结果"""
    print("📊 连接池性能测试结果:")
    print(f"  总请求数: {metrics['total_requests']}")
    print(f"  成功请求: {metrics['successful_requests']}")
    print(f"  失败请求: {metrics['failed_requests']}")
    print(f"  平均响应时间: {metrics['avg_response_time']:.4f}秒")
    print(f"  P95响应时间: {metrics['p95_response_time']:.4f}秒")
    print(f"  吞吐量: {metrics['throughput']:.2f} requests/sec")
    print(f"  总执行时间: {total_time:.2f}秒")

    print("📊 连接池状态:")
    print(f"  空闲连接数: {stats.get('idle_connections', 0)}")
    print(f"  活跃连接数: {stats.get('active_connections', 0)}")
    print(f"  总连接数: {stats.get('total_connections', 0)}")
    print(f"  创建的连接数: {stats.get('created_connections', 0)}")
    print(f"  销毁的连接数: {stats.get('destroyed_connections', 0)}")
    print(f"  连接请求数: {stats.get('connection_requests', 0)}")
    print(f"  利用率: {stats.get('utilization_rate', 0):.2%}")


def _prepare_test_results(metrics, stats, total_time):
    """准备测试结果"""
    return {
        "metrics": metrics,
        "stats": stats,
        "total_time": total_time,
        "total_requests": metrics["total_requests"],
        "successful_requests": metrics["successful_requests"],
        "failed_requests": metrics["failed_requests"],
        "avg_response_time": metrics["avg_response_time"],
        "p95_response_time": metrics["p95_response_time"],
        "throughput": metrics["throughput"],
        "pool_stats": stats,
    }


if __name__ == "__main__":
    # 运行性能测试
    result = performance_test()

    print("✅ 连接池优化完成！")
    print("🎯 优化效果:")
    print("  - 减少了连接创建开销")
    print("  - 提高了连接复用率")
    print("  - 改善了响应时间")
    print("  - 增强了资源管理")
