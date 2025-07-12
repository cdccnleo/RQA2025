import threading
import time
import weakref
from queue import Queue
from typing import Optional, Dict

class ConnectionPool:
    """线程安全的数据库连接池"""
    
    def __init__(self,
                 max_size: int = 10,
                 idle_timeout: int = 300,
                 max_usage: int = 1000,
                 leak_detection: bool = True):
        """
        初始化连接池
        
        :param max_size: 最大连接数
        :param idle_timeout: 空闲超时(秒) 
        :param max_usage: 单个连接最大使用次数
        """
        self._lock = threading.Lock()
        self._pool = Queue(maxsize=max_size)
        self._idle_timeout = idle_timeout
        self._max_usage = max_usage
        self._created_count = 0
        self._active_connections = 0
        self._leak_tracker: Dict[int, weakref.ReferenceType] = {}
        self._config_lock = threading.Lock()
        self._leak_detection = leak_detection

    def _leak_callback(self, ref):
        """连接泄漏回调函数"""
        with self._lock:
            for conn_id, (stored_ref, _) in list(self._leak_tracker.items()):
                if stored_ref == ref:
                    del self._leak_tracker[conn_id]
                    break

    def acquire(self, timeout: Optional[float] = None) -> 'Connection':
        """
        获取一个数据库连接
        
        参数:
            timeout: 超时时间(秒)
        返回:
            Connection: 数据库连接对象
        异常:
            RuntimeError: 当连接池耗尽时抛出
        """
        with self._lock:
            if not self._pool.empty():
                conn = self._pool.get_nowait()
                if self._is_connection_valid(conn):
                    self._active_connections += 1
                    if self._leak_detection:
                        ref = weakref.ref(conn, self._leak_callback)
                        self._leak_tracker[id(conn)] = (ref, time.time())
                    return conn
                
            if self._active_connections >= self._pool.maxsize:
                raise RuntimeError("Connection pool exhausted")
                
            conn = self._create_connection()
            self._active_connections += 1
            if self._leak_detection:
                ref = weakref.ref(conn, self._leak_callback)
                self._leak_tracker[id(conn)] = (ref, time.time())
            return conn

    def release(self, conn: 'Connection') -> None:
        """
        释放连接回连接池
        
        参数:
            conn: 要释放的连接对象
        """
        with self._lock:
            conn.last_used = time.time()
            conn.usage_count += 1
            if self._active_connections > 0:
                self._active_connections -= 1
            # 清理泄漏追踪
            if self._leak_detection and id(conn) in self._leak_tracker:
                del self._leak_tracker[id(conn)]
            if conn.usage_count < self._max_usage:
                self._pool.put_nowait(conn)

    def health_check(self) -> dict:
        """
        返回连接池健康状态
        
        返回:
            dict: 包含以下指标:
                - total: 连接池总容量
                - active: 活跃连接数
                - idle: 空闲连接数
                - created: 已创建连接总数
                - leaks (可选): 检测到的潜在泄漏数
                - config (可选): 当前配置参数
        """
        with self._lock:
            stats = {
                'total': self._pool.maxsize,
                'active': self._active_connections,
                'idle': self._pool.qsize(),
                'created': self._created_count
            }
            if self._leak_detection:
                stats['leaks'] = len(self._leak_tracker)
                stats['config'] = {
                    'max_size': self._pool.maxsize,
                    'idle_timeout': self._idle_timeout,
                    'max_usage': self._max_usage,
                    'leak_detection': self._leak_detection
                }
            return stats

    def _create_connection(self) -> 'Connection':
        """创建新连接(需子类实现)"""
        self._created_count += 1
        return Connection()

    def _is_connection_valid(self, conn: 'Connection') -> bool:
        """验证连接有效性"""
        current_time = time.time()
        return (
            current_time - conn.last_used < self._idle_timeout 
            and not conn.closed
        )

    def update_config(self,
                    max_size: Optional[int] = None,
                    idle_timeout: Optional[int] = None,
                    max_usage: Optional[int] = None,
                    leak_detection: Optional[bool] = None) -> None:
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
        with self._lock:
            if max_size is not None:
                if max_size < self._active_connections:
                    raise ValueError(
                        f"New max_size {max_size} cannot be less than "
                        f"current active connections {self._active_connections}"
                    )
                self._pool.maxsize = max_size
                
            if idle_timeout is not None:
                self._idle_timeout = idle_timeout
                
            if max_usage is not None:
                self._max_usage = max_usage
                
            if leak_detection is not None:
                self._leak_detection = leak_detection

class Connection:
    """连接包装类"""
    def __init__(self):
        self.last_used = time.time()
        self.usage_count = 0
        self.closed = False
        
    def close(self):
        """标记连接为已关闭"""
        self.closed = True
