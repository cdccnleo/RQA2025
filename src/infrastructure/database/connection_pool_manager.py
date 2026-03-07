"""
数据库连接池管理器

提供连接池管理、健康检查、自动重连等功能。

Author: RQA2025 Development Team
Date: 2026-02-13
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncpg
import threading

logger = logging.getLogger(__name__)


class PoolStatus(Enum):
    """连接池状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RECOVERING = "recovering"


@dataclass
class PoolConfig:
    """连接池配置"""
    host: str = "localhost"
    port: int = 5432
    database: str = "rqa2025"
    user: str = "postgres"
    password: str = ""
    
    # 连接池大小
    min_size: int = 5
    max_size: int = 20
    
    # 超时配置
    command_timeout: float = 60.0
    max_inactive_time: float = 300.0  # 5分钟
    
    # 健康检查配置
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0
    
    # 重连配置
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    reconnect_backoff: float = 2.0
    max_reconnect_delay: float = 60.0
    
    # 性能配置
    max_queries: int = 50000  # 连接最大查询次数
    max_lifetime: float = 3600.0  # 连接最大生命周期（1小时）


@dataclass
class PoolStats:
    """连接池统计"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    waiting_requests: int = 0
    total_queries: int = 0
    failed_queries: int = 0
    avg_query_time_ms: float = 0.0
    max_query_time_ms: float = 0.0
    last_health_check: Optional[datetime] = None
    status: PoolStatus = PoolStatus.UNAVAILABLE


@dataclass
class ConnectionInfo:
    """连接信息"""
    connection_id: str
    created_at: float
    last_used_at: float
    query_count: int = 0
    is_active: bool = False


class ConnectionPoolManager:
    """
    数据库连接池管理器
    
    提供以下功能：
    1. 连接池生命周期管理
    2. 自动健康检查
    3. 连接自动重连
    4. 连接池性能监控
    5. 连接池统计
    
    Attributes:
        config: 连接池配置
        pool: asyncpg连接池
        status: 连接池状态
    """
    
    def __init__(self, config: Optional[PoolConfig] = None):
        self.config = config or PoolConfig()
        self.pool: Optional[asyncpg.Pool] = None
        self.status = PoolStatus.UNAVAILABLE
        
        # 统计信息
        self._stats = PoolStats()
        self._connection_infos: Dict[str, ConnectionInfo] = {}
        
        # 锁
        self._lock = threading.RLock()
        
        # 运行状态
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._maintenance_task: Optional[asyncio.Task] = None
        
        # 回调函数
        self._status_callbacks: List[Callable[[PoolStatus], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        
        # 重连计数
        self._reconnect_attempts = 0
        
        logger.info("ConnectionPoolManager initialized")
    
    async def initialize(self) -> bool:
        """
        初始化连接池
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            await self._create_pool()
            
            self._running = True
            
            # 启动健康检查任务
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            # 启动维护任务
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
            
            logger.info("ConnectionPoolManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"ConnectionPoolManager initialization failed: {e}")
            self.status = PoolStatus.UNAVAILABLE
            return False
    
    async def shutdown(self) -> bool:
        """
        关闭连接池
        
        Returns:
            bool: 关闭是否成功
        """
        try:
            self._running = False
            
            # 停止健康检查任务
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # 停止维护任务
            if self._maintenance_task:
                self._maintenance_task.cancel()
                try:
                    await self._maintenance_task
                except asyncio.CancelledError:
                    pass
            
            # 关闭连接池
            if self.pool:
                await self.pool.close()
                self.pool = None
            
            self.status = PoolStatus.UNAVAILABLE
            
            logger.info("ConnectionPoolManager shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"ConnectionPoolManager shutdown failed: {e}")
            return False
    
    async def _create_pool(self) -> asyncpg.Pool:
        """创建连接池"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                command_timeout=self.config.command_timeout,
                init=self._init_connection,
                setup=self._setup_connection
            )
            
            self.status = PoolStatus.HEALTHY
            self._reconnect_attempts = 0
            
            logger.info(f"Connection pool created: {self.config.min_size}-{self.config.max_size} connections")
            return self.pool
            
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    async def _init_connection(self, conn: asyncpg.Connection):
        """初始化连接"""
        # 设置连接参数
        await conn.set_type_codec(
            'json',
            encoder=str,
            decoder=str,
            schema='pg_catalog'
        )
        
        # 记录连接信息
        connection_id = f"conn-{id(conn)}"
        self._connection_infos[connection_id] = ConnectionInfo(
            connection_id=connection_id,
            created_at=time.time(),
            last_used_at=time.time()
        )
    
    async def _setup_connection(self, conn: asyncpg.Connection):
        """设置连接"""
        # 可以在这里设置会话参数
        pass
    
    async def acquire(self) -> asyncpg.Connection:
        """
        获取连接
        
        Returns:
            asyncpg.Connection: 数据库连接
            
        Raises:
            Exception: 连接池不可用时抛出
        """
        if not self.pool or self.status == PoolStatus.UNAVAILABLE:
            raise Exception("Connection pool is not available")
        
        try:
            conn = await self.pool.acquire()
            
            # 更新连接信息
            connection_id = f"conn-{id(conn)}"
            if connection_id in self._connection_infos:
                self._connection_infos[connection_id].is_active = True
                self._connection_infos[connection_id].last_used_at = time.time()
            
            return conn
            
        except Exception as e:
            logger.error(f"Failed to acquire connection: {e}")
            raise
    
    async def release(self, conn: asyncpg.Connection):
        """释放连接"""
        if self.pool:
            # 更新连接信息
            connection_id = f"conn-{id(conn)}"
            if connection_id in self._connection_infos:
                self._connection_infos[connection_id].is_active = False
                self._connection_infos[connection_id].query_count += 1
            
            await self.pool.release(conn)
    
    async def execute(self, query: str, *args) -> str:
        """
        执行SQL语句
        
        Args:
            query: SQL语句
            *args: 参数
            
        Returns:
            str: 执行结果
        """
        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(query, *args)
                
                # 更新统计
                execution_time = (time.time() - start_time) * 1000
                self._update_query_stats(execution_time)
                
                return result
                
        except Exception as e:
            self._stats.failed_queries += 1
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """
        执行查询
        
        Args:
            query: SQL语句
            *args: 参数
            
        Returns:
            List[asyncpg.Record]: 查询结果
        """
        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetch(query, *args)
                
                # 更新统计
                execution_time = (time.time() - start_time) * 1000
                self._update_query_stats(execution_time)
                
                return result
                
        except Exception as e:
            self._stats.failed_queries += 1
            logger.error(f"Query fetch failed: {e}")
            raise
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """
        执行查询，返回单行
        
        Args:
            query: SQL语句
            *args: 参数
            
        Returns:
            Optional[asyncpg.Record]: 查询结果
        """
        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow(query, *args)
                
                # 更新统计
                execution_time = (time.time() - start_time) * 1000
                self._update_query_stats(execution_time)
                
                return result
                
        except Exception as e:
            self._stats.failed_queries += 1
            logger.error(f"Query fetchrow failed: {e}")
            raise
    
    async def fetchval(self, query: str, *args):
        """
        执行查询，返回单个值
        
        Args:
            query: SQL语句
            *args: 参数
            
        Returns:
            查询结果
        """
        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(query, *args)
                
                # 更新统计
                execution_time = (time.time() - start_time) * 1000
                self._update_query_stats(execution_time)
                
                return result
                
        except Exception as e:
            self._stats.failed_queries += 1
            logger.error(f"Query fetchval failed: {e}")
            raise
    
    def _update_query_stats(self, execution_time_ms: float):
        """更新查询统计"""
        with self._lock:
            self._stats.total_queries += 1
            
            # 使用指数移动平均
            alpha = 0.1
            self._stats.avg_query_time_ms = (
                alpha * execution_time_ms + (1 - alpha) * self._stats.avg_query_time_ms
            )
            self._stats.max_query_time_ms = max(self._stats.max_query_time_ms, execution_time_ms)
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_check(self):
        """执行健康检查"""
        try:
            if not self.pool:
                self.status = PoolStatus.UNAVAILABLE
                await self._try_reconnect()
                return
            
            # 执行简单查询测试连接
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            # 更新状态
            if self.status != PoolStatus.HEALTHY:
                logger.info("Connection pool is healthy again")
                self.status = PoolStatus.HEALTHY
                self._reconnect_attempts = 0
            
            self._stats.last_health_check = datetime.utcnow()
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self.status = PoolStatus.DEGRADED
            
            # 触发错误回调
            for callback in self._error_callbacks:
                try:
                    callback(e)
                except Exception as cb_error:
                    logger.error(f"Error callback failed: {cb_error}")
            
            # 尝试重连
            await self._try_reconnect()
    
    async def _try_reconnect(self):
        """尝试重连"""
        if self._reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error(f"Max reconnect attempts ({self.config.max_reconnect_attempts}) reached")
            self.status = PoolStatus.UNAVAILABLE
            return
        
        self._reconnect_attempts += 1
        self.status = PoolStatus.RECOVERING
        
        # 计算延迟（指数退避）
        delay = min(
            self.config.reconnect_delay * (self.config.reconnect_backoff ** (self._reconnect_attempts - 1)),
            self.config.max_reconnect_delay
        )
        
        logger.info(f"Attempting to reconnect in {delay:.1f}s (attempt {self._reconnect_attempts}/{self.config.max_reconnect_attempts})")
        
        await asyncio.sleep(delay)
        
        try:
            # 关闭旧连接池
            if self.pool:
                await self.pool.close()
            
            # 创建新连接池
            await self._create_pool()
            
            logger.info("Reconnection successful")
            
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
    
    async def _maintenance_loop(self):
        """维护循环"""
        while self._running:
            try:
                await self._perform_maintenance()
                await asyncio.sleep(60)  # 每分钟执行一次维护
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_maintenance(self):
        """执行维护"""
        try:
            # 清理过期连接信息
            current_time = time.time()
            expired_connections = [
                conn_id for conn_id, info in self._connection_infos.items()
                if current_time - info.last_used_at > self.config.max_lifetime
            ]
            
            for conn_id in expired_connections:
                del self._connection_infos[conn_id]
            
            if expired_connections:
                logger.debug(f"Cleaned up {len(expired_connections)} expired connection records")
            
        except Exception as e:
            logger.error(f"Maintenance failed: {e}")
    
    def get_stats(self) -> PoolStats:
        """
        获取连接池统计
        
        Returns:
            PoolStats: 统计信息
        """
        with self._lock:
            if self.pool:
                self._stats.total_connections = len(self._connection_infos)
                self._stats.active_connections = sum(
                    1 for info in self._connection_infos.values() if info.is_active
                )
                self._stats.idle_connections = self._stats.total_connections - self._stats.active_connections
            
            return self._stats
    
    def get_status(self) -> PoolStatus:
        """
        获取连接池状态
        
        Returns:
            PoolStatus: 当前状态
        """
        return self.status
    
    def register_status_callback(self, callback: Callable[[PoolStatus], None]):
        """注册状态变更回调"""
        self._status_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable[[Exception], None]):
        """注册错误回调"""
        self._error_callbacks.append(callback)
    
    def _notify_status_change(self, new_status: PoolStatus):
        """通知状态变更"""
        old_status = self.status
        self.status = new_status
        
        if old_status != new_status:
            for callback in self._status_callbacks:
                try:
                    callback(new_status)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")


# 便捷函数
async def create_pool_manager(
    host: str = "localhost",
    port: int = 5432,
    database: str = "rqa2025",
    user: str = "postgres",
    password: str = "",
    min_size: int = 5,
    max_size: int = 20
) -> ConnectionPoolManager:
    """
    创建连接池管理器
    
    Args:
        host: 主机地址
        port: 端口
        database: 数据库名
        user: 用户名
        password: 密码
        min_size: 最小连接数
        max_size: 最大连接数
        
    Returns:
        ConnectionPoolManager: 连接池管理器
    """
    config = PoolConfig(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        min_size=min_size,
        max_size=max_size
    )
    
    manager = ConnectionPoolManager(config)
    await manager.initialize()
    return manager


__all__ = [
    'ConnectionPoolManager',
    'PoolConfig',
    'PoolStats',
    'PoolStatus',
    'ConnectionInfo',
    'create_pool_manager'
]