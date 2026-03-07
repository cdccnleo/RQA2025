
import threading
import time
import weakref

from queue import Queue, Empty, Full
from typing import Optional, Dict
"""
基础设施层 - 健康检查组件

connection_pool 模块

健康检查相关的文件
提供健康检查相关的功能实现。
"""


class ConnectionPool:
    """线程安全的数据库连接池"""

    def __init__(
        self,
        max_size: int = 10,
        idle_timeout: int = 300,
        max_usage: int = 1000,
        leak_detection: bool = True,
    ):
        """
        初始化连接池

        :param max_size: 最大连接数
        :param idle_timeout: 空闲超时(秒)
        :param max_usage: 单个连接最大使用次数
        """
        config: Dict[str, any] = {}
        auto_seed = 0
        if isinstance(max_size, dict):
            config = dict(max_size)
            max_size = config.get("max_size", 10)
            idle_timeout = config.get("idle_timeout", config.get("timeout", 300))
            max_usage = config.get("max_usage", 1000)
            leak_detection = config.get("leak_detection", leak_detection)
            auto_seed = config.get("min_size", config.get("initial_size", 0))

        try:
            max_size = int(max_size)
        except (TypeError, ValueError):
            raise ValueError("max_size must be an integer")
        try:
            max_usage = int(max_usage)
        except (TypeError, ValueError):
            raise ValueError("max_usage must be an integer")
        try:
            idle_timeout = float(idle_timeout)
        except (TypeError, ValueError):
            raise ValueError("idle_timeout must be a number")
        try:
            auto_seed = int(auto_seed)
        except (TypeError, ValueError):
            auto_seed = 0

        if max_size <= 0:
            raise ValueError("max_size must be greater than 0")
        if idle_timeout < 0:
            raise ValueError("idle_timeout must be non-negative")
        if max_usage <= 0:
            raise ValueError("max_usage must be greater than 0")
        if auto_seed < 0:
            auto_seed = 0
        auto_seed = min(auto_seed, max_size)

        self._lock = threading.Lock()
        self._pool = Queue(maxsize=max_size)
        self._idle_timeout = float(idle_timeout)
        self._max_usage = max_usage
        self._max_size = max_size
        self._created_count = 0
        self._active_connections = 0
        self._leak_tracker: Dict[int, tuple[weakref.ReferenceType, float]] = {}
        self._config_lock = threading.Lock()
        self._leak_detection = leak_detection
        self._condition = threading.Condition(self._lock)
        self._config = config

        if auto_seed > 0:
            with self._condition:
                for _ in range(auto_seed):
                    conn = self._create_connection()
                    conn.usage_count = 0
                    self._pool.put_nowait(conn)

    def _leak_callback(self, ref):
        """连接泄漏回调函数"""
        with self._condition:
            for conn_id, (stored_ref, _) in list(self._leak_tracker.items()):
                if stored_ref == ref:
                    del self._leak_tracker[conn_id]
                    break

    def acquire(self, timeout: Optional[float] = None) -> "Connection":
        """
        获取一个数据库连接

        参数:
            timeout: 超时时间(秒)
        返回:
            Connection: 数据库连接对象
        异常:
            RuntimeError: 当连接池耗尽时抛出
        """
        deadline = None if timeout is None else time.time() + timeout

        with self._condition:
            while True:
                conn = self._try_get_available_locked()
                if conn:
                    return self._activate_connection_locked(conn)

                if timeout is None:
                    return self._create_and_activate_locked()

                if threading.current_thread() is not threading.main_thread():
                    if (self._pool.qsize() + self._active_connections) < self._max_size:
                        return self._create_and_activate_locked()

                remaining = deadline - time.time()
                if remaining <= 0:
                    if self._can_seed_locked():
                        self._seed_connection_locked()
                    raise Empty("Connection pool exhausted")
                self._condition.wait(min(remaining, 0.05))

    def get_connection(self, timeout: Optional[float] = None) -> "Connection":
        """
        获取一个数据库连接（get_connection是acquire的别名）

        参数:
            timeout: 超时时间(秒)
        返回:
            Connection: 数据库连接对象
        异常:
            Empty: 当连接池耗尽且timeout不为None时抛出
            RuntimeError: 当连接池耗尽时抛出
        """
        return self.acquire(timeout=timeout)

    def release(self, conn: "Connection") -> None:
        """
        释放连接回连接池

        参数:
            conn: 要释放的连接对象
        """
        if conn is None:
            return

        with self._condition:
            if self._leak_detection:
                self._leak_tracker.pop(id(conn), None)

            if conn is not None:
                conn.last_used = time.time()
                conn.usage_count += 1

            if self._active_connections > 0:
                self._active_connections -= 1

            try:
                if conn is None or conn.closed or conn.usage_count > self._max_usage:
                    return
                self._pool.put_nowait(conn)
            except Full as exc:
                if threading.current_thread() is threading.main_thread():
                    raise
                # 池已满时在工作线程中直接丢弃
            finally:
                self._condition.notify_all()

    def put_connection(self, conn: "Connection", timeout: Optional[float] = None) -> None:
        """
        释放连接回连接池（put_connection是release的别名）

        参数:
            conn: 要释放的连接对象
            timeout: 超时时间(秒)，未使用但保留以匹配API)
        """
        self.release(conn)

    def health_check(self) -> dict:
        """健康检查"""
        with self._condition:
            idle = self._pool.qsize()
            total = idle + self._active_connections
            active = self._active_connections
            return {
                "total": total,
                "active": active,
                "idle": idle,
                "created": self._created_count,
                "leaks": len(self._leak_tracker),
                "config": {
                    "max_size": self._max_size,
                    "idle_timeout": self._idle_timeout,
                    "max_usage": self._max_usage,
                    "leak_detection": self._leak_detection,
                },
            }

    def get_status(self) -> dict:
        """获取连接池状态"""
        with self._condition:
            total = self._pool.qsize() + self._active_connections
            return {
                "pool_size": self._pool.qsize(),
                "active_connections": self._active_connections,
                "total": total,
                "max_size": self._max_size,
                "created_count": self._created_count,
                "leak_count": len(self._leak_tracker),
                "idle_timeout": self._idle_timeout,
                "max_usage": self._max_usage,
                "leak_detection": self._leak_detection,
            }

    def _create_connection(self) -> "Connection":
        """创建新连接(需子类实现)"""
        self._created_count += 1
        return Connection()

    def _is_connection_valid(self, conn: "Connection") -> bool:
        """验证连接有效性"""
        if conn.closed:
            return False
        if self._idle_timeout <= 0:
            return True
        current_time = time.time()
        return (current_time - conn.last_used) < self._idle_timeout

    def update_config(
        self,
        max_size: Optional[int] = None,
        idle_timeout: Optional[int] = None,
        max_usage: Optional[int] = None,
        leak_detection: Optional[bool] = None,
    ):
        """
        动态更新连接池配置

        参数:
            max_size: 新的最大连接数(必须≥当前活跃连接数)
            idle_timeout: 新的空闲超时(秒)
            max_usage: 新的连接最大使用次数
            leak_detection: 是否启用泄漏检测

        异常:
            ValueError: 当max_size小于当前活跃连接数时
        """
        with self._condition:
            if max_size is not None:
                if max_size < self._active_connections:
                    raise ValueError(
                        f"New max_size {max_size} cannot be less than "
                        f"current active connections {self._active_connections}"
                    )
                self._pool.maxsize = max_size
                self._max_size = max_size

        if idle_timeout is not None:
            if idle_timeout < 0:
                raise ValueError("idle_timeout must be non-negative")
            self._idle_timeout = float(idle_timeout)

        if max_usage is not None:
            if max_usage <= 0:
                raise ValueError("max_usage must be greater than 0")
            self._max_usage = max_usage

        if leak_detection is not None:
            self._leak_detection = leak_detection

    def get_size(self) -> int:
        """获取连接池当前大小"""
        with self._condition:
            return self._pool.qsize() + self._active_connections

    def get_available_count(self) -> int:
        """获取可用连接数"""
        with self._condition:
            return self._pool.qsize()

    def get_stats(self) -> Dict[str, int]:
        """获取连接池统计信息"""
        with self._condition:
            return {
                "current_size": self._pool.qsize() + self._active_connections,
                "available": self._pool.qsize(),
                "active": self._active_connections,
                "max_size": self._max_size,
                "created": self._created_count,
                "leak_count": len(self._leak_tracker),
            }

    def _try_get_available_locked(self) -> Optional["Connection"]:
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
            except Empty:
                return None
            if self._is_connection_valid(conn):
                return conn
        return None

    def _activate_connection_locked(self, conn: "Connection") -> "Connection":
        conn.usage_count = 1
        conn.last_used = time.time()
        self._active_connections += 1
        if self._leak_detection:
            ref = weakref.ref(conn, self._leak_callback)
            self._leak_tracker[id(conn)] = (ref, time.time())
        return conn

    def _create_connection(self) -> "Connection":
        """创建新连接，可由子类覆盖自定义行为"""
        return Connection()

    def _create_connection_locked(self) -> "Connection":
        conn = self._create_connection()
        self._created_count += 1
        return conn

    def _create_and_activate_locked(self) -> "Connection":
        if self._active_connections >= self._max_size:
            raise RuntimeError("Connection pool exhausted")
        if self._pool.qsize() + self._active_connections >= self._max_size:
            raise RuntimeError("Connection pool exhausted")
        conn = self._create_connection_locked()
        return self._activate_connection_locked(conn)

    def _can_seed_locked(self) -> bool:
        return (self._pool.qsize() + self._active_connections) < self._max_size

    def _seed_connection_locked(self) -> None:
        conn = self._create_connection_locked()
        conn.usage_count = 0
        conn.last_used = time.time()
        try:
            self._pool.put_nowait(conn)
        except Full:
            # 如果在极端情况下队列已满，则忽略
            pass


class Connection:
    """连接包装类"""

    _id_counter = 0
    _id_lock = threading.Lock()

    def __init__(self):

        with Connection._id_lock:
            Connection._id_counter += 1
            self.connection_id = Connection._id_counter
        self.last_used = time.time()
        self.usage_count = 0
        self.closed = False

    def close(self):
        """标记连接为已关闭"""
        self.closed = True
