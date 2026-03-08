#!/usr/bin/env python3
"""
异步数据库连接池
实现高性能异步数据库访问
"""

import asyncio
import asyncpg
from typing import Optional, List, Dict, Any, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """连接池配置"""
    min_size: int = 5
    max_size: int = 50
    max_queries: int = 50000
    max_inactive_time: float = 300.0
    command_timeout: float = 60.0
    setup_timeout: float = 10.0


class AsyncDatabasePool:
    """
    异步数据库连接池
    
    特性：
    1. 异步连接管理
    2. 连接复用
    3. 自动扩缩容
    4. 连接健康检查
    5. 查询超时控制
    """
    
    def __init__(
        self,
        dsn: str,
        config: Optional[PoolConfig] = None
    ):
        self.dsn = dsn
        self.config = config or PoolConfig()
        self._pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        
        # 统计信息
        self._stats = {
            'total_queries': 0,
            'failed_queries': 0,
            'connections_created': 0,
            'connections_closed': 0
        }
    
    async def initialize(self):
        """初始化连接池"""
        if self._initialized:
            return
        
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                max_queries=self.config.max_queries,
                max_inactive_time=self.config.max_inactive_time,
                command_timeout=self.config.command_timeout,
                setup_timeout=self.config.setup_timeout,
                init=self._init_connection
            )
            
            self._initialized = True
            logger.info(f"Database pool initialized: min={self.config.min_size}, max={self.config.max_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
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
        self._stats['connections_created'] += 1
    
    async def close(self):
        """关闭连接池"""
        if self._pool:
            await self._pool.close()
            self._initialized = False
            logger.info("Database pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """获取连接上下文管理器"""
        if not self._initialized:
            await self.initialize()
        
        async with self._pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                logger.error(f"Database error: {e}")
                raise
    
    async def fetch(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> List[asyncpg.Record]:
        """执行查询"""
        async with self.acquire() as conn:
            try:
                result = await conn.fetch(query, *args, timeout=timeout)
                self._stats['total_queries'] += 1
                return result
            except Exception as e:
                self._stats['failed_queries'] += 1
                logger.error(f"Query failed: {e}, Query: {query}")
                raise
    
    async def fetchrow(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[asyncpg.Record]:
        """执行查询，返回单行"""
        async with self.acquire() as conn:
            try:
                result = await conn.fetchrow(query, *args, timeout=timeout)
                self._stats['total_queries'] += 1
                return result
            except Exception as e:
                self._stats['failed_queries'] += 1
                logger.error(f"Query failed: {e}")
                raise
    
    async def fetchval(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Any:
        """执行查询，返回单个值"""
        async with self.acquire() as conn:
            try:
                result = await conn.fetchval(query, *args, timeout=timeout)
                self._stats['total_queries'] += 1
                return result
            except Exception as e:
                self._stats['failed_queries'] += 1
                logger.error(f"Query failed: {e}")
                raise
    
    async def execute(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> str:
        """执行命令"""
        async with self.acquire() as conn:
            try:
                result = await conn.execute(query, *args, timeout=timeout)
                self._stats['total_queries'] += 1
                return result
            except Exception as e:
                self._stats['failed_queries'] += 1
                logger.error(f"Execute failed: {e}")
                raise
    
    async def executemany(
        self,
        query: str,
        args: List[tuple],
        timeout: Optional[float] = None
    ) -> str:
        """批量执行"""
        async with self.acquire() as conn:
            try:
                result = await conn.executemany(query, args, timeout=timeout)
                self._stats['total_queries'] += len(args)
                return result
            except Exception as e:
                self._stats['failed_queries'] += 1
                logger.error(f"Executemany failed: {e}")
                raise
    
    async def transaction(self):
        """事务上下文管理器"""
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        if self._pool:
            stats['pool_size'] = len(self._pool._holders)
            stats['free_connections'] = len(self._pool._queue)
        return stats


class DatabaseManager:
    """数据库管理器"""
    
    _instance: Optional['DatabaseManager'] = None
    _pools: Dict[str, AsyncDatabasePool] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def add_pool(
        self,
        name: str,
        dsn: str,
        config: Optional[PoolConfig] = None
    ):
        """添加连接池"""
        pool = AsyncDatabasePool(dsn, config)
        await pool.initialize()
        self._pools[name] = pool
        logger.info(f"Added database pool: {name}")
    
    def get_pool(self, name: str = "default") -> AsyncDatabasePool:
        """获取连接池"""
        if name not in self._pools:
            raise KeyError(f"Database pool '{name}' not found")
        return self._pools[name]
    
    async def close_all(self):
        """关闭所有连接池"""
        for name, pool in self._pools.items():
            await pool.close()
        self._pools.clear()
        logger.info("All database pools closed")


# 全局数据库管理器
_db_manager: Optional[DatabaseManager] = None


async def setup_database(dsn: str, config: Optional[PoolConfig] = None):
    """设置数据库"""
    global _db_manager
    _db_manager = DatabaseManager()
    await _db_manager.add_pool("default", dsn, config)
    logger.info("Database setup complete")


def get_db() -> AsyncDatabasePool:
    """获取默认数据库连接池"""
    if _db_manager is None:
        raise RuntimeError("Database not initialized")
    return _db_manager.get_pool("default")


# 示例用法
if __name__ == "__main__":
    import os
    
    async def test_database():
        print("=== 异步数据库连接池测试 ===\n")
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        
        # 测试配置（使用环境变量或默认值）
        dsn = os.environ.get(
            'DATABASE_URL',
            'postgresql://user:password@localhost/testdb'
        )
        
        config = PoolConfig(
            min_size=2,
            max_size=10
        )
        
        try:
            # 初始化数据库
            await setup_database(dsn, config)
            db = get_db()
            
            # 测试1: 基本查询
            print("测试1: 基本查询")
            result = await db.fetch("SELECT version()")
            print(f"PostgreSQL版本: {result[0]['version']}")
            
            # 测试2: 参数化查询
            print("\n测试2: 参数化查询")
            result = await db.fetchrow(
                "SELECT $1::int + $2::int as sum",
                10, 20
            )
            print(f"10 + 20 = {result['sum']}")
            
            # 测试3: 统计信息
            print("\n测试3: 统计信息")
            stats = db.get_stats()
            print(f"总查询数: {stats['total_queries']}")
            print(f"失败查询: {stats['failed_queries']}")
            print(f"连接池大小: {stats.get('pool_size', 'N/A')}")
            
            print("\n=== 测试完成 ===")
            
        except Exception as e:
            print(f"测试失败: {e}")
        finally:
            if _db_manager:
                await _db_manager.close_all()
    
    # 运行测试
    asyncio.run(test_database())
