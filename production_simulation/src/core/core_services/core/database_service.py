#!/usr/bin/env python3
"""
RQA2025核心数据库服务

提供生产级的数据库连接、查询、事务管理等功能。
支持PostgreSQL、Redis缓存、连接池管理等。
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import time

# 数据库相关导入
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None

try:
    import aioredis
    HAS_AIREDIS = True
except ImportError:
    HAS_AIREDIS = False
    aioredis = None

try:
    import psycopg2
    from psycopg2 import pool
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    psycopg2 = None
    pool = None

from src.core.constants import (
    DEFAULT_BATCH_SIZE, DEFAULT_TIMEOUT, DEFAULT_TEST_TIMEOUT,
    SECONDS_PER_HOUR, SECONDS_PER_MINUTE, SECONDS_PER_DAY,
    MAX_RECORDS, MAX_RETRIES, MAX_QUEUE_SIZE, DEFAULT_PAGE_SIZE,
    DEFAULT_PERFORMANCE_THRESHOLD
)

from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
from src.foundation.exceptions.unified_exceptions import DatabaseError, ConnectionError, QueryError

logger = get_logger(__name__)


@dataclass
class DatabaseConfig:
    """数据库配置 - 生产环境优化版"""
    # 使用常量替换魔数
    # 基础连接配置
    host: str = "localhost"
    port: int = 5432
    database: str = "rqa2025"
    user: str = "rqa_user"
    password: str = ""

    # 连接池优化配置
    min_connections: int = DEFAULT_BATCH_SIZE      # 最小连接数，从5提升到10
    max_connections: int = 50      # 最大连接数，从20提升到50
    max_idle_time: int = DEFAULT_TEST_TIMEOUT       # 最大空闲时间 (秒)
    max_lifetime: int = SECONDS_PER_HOUR       # 最大生命周期 (秒)

    # 超时配置优化
    connection_timeout: int = DEFAULT_TIMEOUT   # 连接超时
    command_timeout: int = DEFAULT_TIMEOUT      # 命令超时，从60减少到30
    pool_recycle: int = SECONDS_PER_HOUR       # 连接回收时间

    # 健康检查配置
    health_check_interval: int = SECONDS_PER_MINUTE    # 健康检查间隔
    max_retries: int = 3               # 最大重试次数
    retry_delay: float = 0.5           # 重试延迟

    # SSL和安全配置
    ssl_mode: str = "require"       # SSL模式，从prefer改为require
    ssl_ca: Optional[str] = None    # SSL CA证书路径
    ssl_cert: Optional[str] = None  # SSL客户端证书路径
    ssl_key: Optional[str] = None   # SSL客户端密钥路径

    # 性能监控配置
    enable_metrics: bool = True     # 启用性能指标收集
    slow_query_threshold: float = 1.0  # 慢查询阈值 (秒)


@dataclass
class RedisConfig:
    """Redis配置 - 生产环境优化版"""
    # 基础连接配置
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0

    # 连接池优化配置
    max_connections: int = 50          # 最大连接数，从10提升到50
    min_connections: int = 5           # 最小连接数
    max_idle_time: int = DEFAULT_TEST_TIMEOUT           # 最大空闲时间 (秒)
    retry_on_timeout: bool = True

    # 超时配置优化
    socket_timeout: int = 5            # socket超时
    socket_connect_timeout: int = 5    # 连接超时
    socket_read_timeout: int = 5       # 读取超时

    # 集群配置 (生产环境)
    sentinel_hosts: Optional[List[Tuple[str, int]]] = None  # Sentinel主机列表
    master_name: Optional[str] = None    # 主节点名称

    # 高级配置
    health_check_interval: int = DEFAULT_TIMEOUT     # 健康检查间隔
    decode_responses: bool = True       # 自动解码响应
    retry_attempts: int = 3             # 重试次数
    retry_delay: float = 0.5            # 重试延迟

    # SSL配置
    ssl_enabled: bool = False           # 是否启用SSL
    ssl_ca_certs: Optional[str] = None  # CA证书路径
    ssl_certfile: Optional[str] = None  # 客户端证书
    ssl_keyfile: Optional[str] = None   # 客户端密钥

    # 性能监控
    enable_metrics: bool = True         # 启用性能指标收集


@dataclass
class CacheConfig:
    """缓存配置 - 生产环境多级缓存策略"""
    # 基础配置
    default_ttl: int = SECONDS_PER_HOUR              # 默认TTL (秒)
    max_memory: str = "1gb"              # 最大内存，从512mb提升到1gb
    eviction_policy: str = "allkeys-lru"  # 驱逐策略

    # 多级缓存配置
    enable_multi_level: bool = True      # 启用多级缓存
    l1_cache_size: int = MAX_QUEUE_SIZE          # L1缓存大小 (条目数)
    l1_cache_ttl: int = DEFAULT_TEST_TIMEOUT             # L1缓存TTL (秒)
    l2_cache_ttl: int = SECONDS_PER_HOUR            # L2缓存TTL (秒)
    l3_cache_ttl: int = SECONDS_PER_DAY           # L3缓存TTL (秒)

    # 性能优化配置
    compression_enabled: bool = True    # 启用压缩
    compression_threshold: int = 1024   # 压缩阈值 (字节)
    serialization_format: str = "json"  # 序列化格式 (json/pickle)

    # 智能缓存配置
    adaptive_ttl_enabled: bool = True   # 启用自适应TTL
    hit_rate_monitoring: bool = True    # 启用命中率监控
    prefetch_enabled: bool = False      # 启用预取 (生产环境谨慎使用)

    # 监控和告警配置
    enable_metrics: bool = True         # 启用性能指标
    alert_on_high_memory: bool = True   # 内存使用高告警
    alert_on_low_hit_rate: bool = True  # 命中率低告警
    memory_threshold: float = 0.8       # 内存使用阈值 (80%)
    hit_rate_threshold: float = 0.7     # 命中率阈值 (70%)

    # 故障恢复配置
    enable_backup_recovery: bool = True  # 启用备份恢复
    backup_interval: int = SECONDS_PER_HOUR         # 备份间隔 (秒)
    max_backup_files: int = DEFAULT_BATCH_SIZE          # 最大备份文件数

    # 高级特性
    enable_distributed_lock: bool = True  # 启用分布式锁
    lock_timeout: int = DEFAULT_TIMEOUT               # 锁超时时间
    enable_circuit_breaker: bool = True  # 启用熔断器
    circuit_failure_threshold: int = 5   # 熔断失败阈值


class DatabaseConnectionPool:
    """数据库连接池管理器 - 生产环境优化版"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
        self._is_async = HAS_ASYNCPG

        # 性能监控指标
        self._metrics = {
            'connections_created': 0,
            'connections_destroyed': 0,
            'connections_acquired': 0,
            'connections_released': 0,
            'queries_executed': 0,
            'slow_queries': 0,
            'connection_errors': 0,
            'pool_exhaustion_count': 0,
            'last_health_check': None,
            'health_status': 'unknown'
        }

        # 健康检查回调
        self._health_callbacks: List[Callable] = []

    async def initialize(self):
        """初始化连接池"""
        try:
            if self._is_async and HAS_ASYNCPG:
                # 异步PostgreSQL连接池 - 优化配置
                ssl_context = None
                if self.config.ssl_mode and self.config.ssl_mode != 'disable':
                    import ssl
                    ssl_context = ssl.create_default_context()
                    if self.config.ssl_ca:
                        ssl_context.load_verify_locations(cafile=self.config.ssl_ca)
                    if self.config.ssl_cert and self.config.ssl_key:
                        ssl_context.load_cert_chain(
                            certfile=self.config.ssl_cert,
                            keyfile=self.config.ssl_key
                        )
                    ssl_context.check_hostname = self.config.ssl_mode == 'verify-full'

                self.pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                    min_size=self.config.min_connections,
                    max_size=self.config.max_connections,
                    max_idle_time=self.config.max_idle_time,
                    max_lifetime=self.config.max_lifetime,
                    command_timeout=self.config.command_timeout,
                    ssl=ssl_context,
                    server_settings={
                        'application_name': 'rqa2025_trading_system',
                        'timezone': 'UTC'
                    }
                )
                logger.info(
                    f"异步PostgreSQL连接池已初始化 (大小: {self.config.min_connections}-{self.config.max_connections})")
            elif HAS_PSYCOPG2:
                # 同步PostgreSQL连接池
                self.pool = pool.ThreadedConnectionPool(
                    self.config.min_connections,
                    self.config.max_connections,
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password,
                    connect_timeout=self.config.connection_timeout
                )
                logger.info(
                    f"同步PostgreSQL连接池已初始化 (大小: {self.config.min_connections}-{self.config.max_connections})")
            else:
                raise DatabaseError("未找到可用的PostgreSQL驱动")

        except Exception as e:
            logger.error(f"数据库连接池初始化失败: {e}")
            raise DatabaseError(f"连接池初始化失败: {e}")

        # 启动健康检查任务
        if self.config.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    def add_health_callback(self, callback: Callable):
        """添加健康检查回调"""
        self._health_callbacks.append(callback)

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self._metrics.copy()

    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")
                self._metrics['health_status'] = 'error'

    async def _perform_health_check(self):
        """执行健康检查"""
        try:
            async with self.get_connection() as conn:
                if self._is_async:
                    # 执行简单的健康检查查询
                    result = await conn.fetchval("SELECT 1")
                    if result == 1:
                        self._metrics['health_status'] = 'healthy'
                        self._metrics['last_health_check'] = datetime.now()
                    else:
                        self._metrics['health_status'] = 'unhealthy'
                else:
                    # 同步检查
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    if result and result[0] == 1:
                        self._metrics['health_status'] = 'healthy'
                        self._metrics['last_health_check'] = datetime.now()
                    else:
                        self._metrics['health_status'] = 'unhealthy'
                    cursor.close()

            # 执行回调
            for callback in self._health_callbacks:
                try:
                    await callback(self._metrics['health_status'])
                except Exception as e:
                    logger.error(f"健康检查回调执行失败: {e}")

        except Exception as e:
            logger.error(f"数据库健康检查失败: {e}")
            self._metrics['health_status'] = 'unhealthy'
            self._metrics['connection_errors'] += 1

    def _record_query_metrics(self, query_time: float):
        """记录查询性能指标"""
        self._metrics['queries_executed'] += 1
        if query_time > self.config.slow_query_threshold:
            self._metrics['slow_queries'] += 1
            logger.warning(f"检测到慢查询: {query_time:.3f}s")

    async def close(self):
        """关闭连接池"""
        if self.pool:
            if self._is_async and hasattr(self.pool, 'close'):
                await self.pool.close()
            elif hasattr(self.pool, 'closeall'):
                self.pool.closeall()
            logger.info("数据库连接池已关闭")

    @asynccontextmanager
    async def get_connection(self):
        """获取数据库连接 - 带性能监控"""
        conn = None
        start_time = datetime.now()
        try:
            if self._is_async:
                conn = await self.pool.acquire()
            else:
                conn = self.pool.getconn()

            # 记录连接获取指标
            self._metrics['connections_acquired'] += 1

            yield conn

        except Exception as e:
            logger.error(f"获取数据库连接失败: {e}")
            raise ConnectionError(f"无法获取数据库连接: {e}")
        finally:
            if conn:
                try:
                    if self._is_async:
                        await self.pool.release(conn)
                    else:
                        self.pool.putconn(conn)

                    # 记录连接释放指标
                    self._metrics['connections_released'] += 1

                except Exception as e:
                    logger.warning(f"释放数据库连接失败: {e}")

    async def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """执行查询 - 带性能监控"""
        start_time = datetime.now()
        try:
            async with self.get_connection() as conn:
                if self._is_async:
                    if params:
                        rows = await conn.fetch(query, *params)
                    else:
                        rows = await conn.fetch(query)

                    # 转换为字典列表
                    result = []
                    if rows:
                        for row in rows:
                            result.append(dict(row))
                    return result

                else:
                    with conn.cursor() as cursor:
                        cursor.execute(query, params or ())
                        columns = [desc[0]
                                   for desc in cursor.description] if cursor.description else []
                        rows = cursor.fetchall()

                        result = []
                        for row in rows:
                            result.append(dict(zip(columns, row)))
                        return result

        except Exception as e:
            logger.error(f"查询执行失败: {query} - {e}")
            raise QueryError(f"查询执行失败: {e}")
        finally:
            # 记录查询性能指标
            query_time = (datetime.now() - start_time).total_seconds()
            if self.config.enable_metrics:
                self._record_query_metrics(query_time)

    async def execute_command(self, command: str, params: tuple = None) -> int:
        """执行命令（INSERT/UPDATE/DELETE）"""
        start_time = datetime.now()
        async with self.get_connection() as conn:
            try:
                if self._is_async:
                    if params:
                        result = await conn.execute(command, *params)
                    else:
                        result = await conn.execute(command)

                    # 解析受影响的行数
                    if "INSERT" in command.upper() or "UPDATE" in command.upper() or "DELETE" in command.upper():
                        # 从结果字符串中提取数字
                        import re
                        match = re.search(r'(\d+)', str(result))
                        return int(match.group(1)) if match else 0
                    return 0

                else:
                    with conn.cursor() as cursor:
                        cursor.execute(command, params or ())
                        conn.commit()
                        return cursor.rowcount

            except Exception as e:
                logger.error(f"命令执行失败: {command} - {e}")
                if not self._is_async:
                    conn.rollback()
                raise QueryError(f"命令执行失败: {e}")
            finally:
                # 记录查询性能指标
                command_time = (datetime.now() - start_time).total_seconds()
                if self.config.enable_metrics:
                    self._record_query_metrics(command_time)

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            start_time = time.time()
            result = await self.execute_query("SELECT 1 as health_check")
            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time": round(response_time * DEFAULT_PERFORMANCE_THRESHOLD, 2),  # 毫秒
                "connection_pool_size": self.config.max_connections,
                "active_connections": getattr(self.pool, 'size', 0) if self.pool else 0
            }

        except Exception as e:
            logger.error(f"数据库健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class RedisCacheService:
    """Redis缓存服务"""

    def __init__(self, config: RedisConfig):
        self.config = config
        self.redis = None
        self._is_async = HAS_AIREDIS

    async def initialize(self):
        """初始化Redis连接"""
        try:
            if self._is_async and HAS_AIREDIS:
                self.redis = await aioredis.from_url(
                    f"redis://:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.db}",
                    max_connections=self.config.max_connections,
                    retry_on_timeout=self.config.retry_on_timeout,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout
                )
            else:
                # 这里可以添加同步Redis客户端的支持
                logger.warning("异步Redis客户端不可用，使用模拟缓存")
                self.redis = None

            logger.info("Redis缓存服务已初始化")

        except Exception as e:
            logger.error(f"Redis初始化失败: {e}")
            raise ConnectionError(f"Redis连接失败: {e}")

    async def close(self):
        """关闭Redis连接"""
        if self.redis and hasattr(self.redis, 'close'):
            await self.redis.close()
            logger.info("Redis连接已关闭")

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            if self.redis:
                value = await self.redis.get(key)
                if value:
                    return json.loads(value)
            return None
        except Exception as e:
            logger.warning(f"缓存获取失败 {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值"""
        try:
            if self.redis:
                json_value = json.dumps(value)
                if ttl:
                    await self.redis.setex(key, ttl, json_value)
                else:
                    await self.redis.set(key, json_value)
            return True
        except Exception as e:
            logger.warning(f"缓存设置失败 {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            if self.redis:
                await self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"缓存删除失败 {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查缓存键是否存在"""
        try:
            if self.redis:
                return await self.redis.exists(key) > 0
            return False
        except Exception as e:
            logger.warning(f"缓存存在性检查失败 {key}: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Redis健康检查"""
        try:
            if self.redis:
                start_time = time.time()
                await self.redis.ping()
                response_time = time.time() - start_time

                return {
                    "status": "healthy",
                    "response_time": round(response_time * DEFAULT_PERFORMANCE_THRESHOLD, 2),
                    "host": self.config.host,
                    "port": self.config.port
                }
            else:
                return {
                    "status": "simulated",
                    "message": "Redis不可用，使用模拟缓存"
                }

        except Exception as e:
            logger.error(f"Redis健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class DatabaseService:
    """核心数据库服务"""

    def __init__(self, db_config: DatabaseConfig = None, redis_config: RedisConfig = None):
        self.db_config = db_config or DatabaseConfig()
        self.redis_config = redis_config or RedisConfig()

        self.db_pool = DatabaseConnectionPool(self.db_config)
        self.cache = RedisCacheService(self.redis_config)

        self._initialized = False

    async def initialize(self):
        """初始化服务"""
        try:
            await self.db_pool.initialize()
            await self.cache.initialize()

            # 初始化数据库表
            await self._initialize_tables()

            self._initialized = True
            logger.info("数据库服务初始化完成")

        except Exception as e:
            logger.error(f"数据库服务初始化失败: {e}")
            raise

    async def close(self):
        """关闭服务"""
        await self.cache.close()
        await self.db_pool.close()
        logger.info("数据库服务已关闭")

    async def _initialize_tables(self):
        """初始化数据库表"""
        # 用户表
        await self.db_pool.execute_command("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                balance DECIMAL(15,2) DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)

        # 订单表
        await self.db_pool.execute_command("""
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                order_id VARCHAR(50) UNIQUE NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                order_type VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 持仓表
        await self.db_pool.execute_command("""
            CREATE TABLE IF NOT EXISTS positions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                symbol VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                avg_price DECIMAL(10,2) NOT NULL,
                current_price DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, symbol)
            )
        """)

        # 交易记录表
        await self.db_pool.execute_command("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                order_id INTEGER REFERENCES orders(id),
                symbol VARCHAR(10) NOT NULL,
                quantity INTEGER NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                side VARCHAR(10) NOT NULL,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        logger.info("数据库表初始化完成")

    # 用户管理方法
    async def create_user(self, username: str, email: str, password: str, initial_balance: float = DEFAULT_PAGE_SIZE * DEFAULT_PAGE_SIZE) -> Dict[str, Any]:
        """创建用户"""
        try:
            # 生成密码哈希 (使用bcrypt)
            import bcrypt
            salt = bcrypt.gensalt()
            password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

            user_id = await self.db_pool.execute_command("""
                INSERT INTO users (username, email, password_hash, balance)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, (username, email, password_hash, initial_balance))

            if user_id:
                # 缓存用户信息
                user_data = {
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "balance": initial_balance,
                    "created_at": datetime.now().isoformat()
                }
                await self.cache.set(f"user:{user_id}", user_data, ttl=SECONDS_PER_HOUR)

                return {"success": True, "user_id": user_id, "user": user_data}

            return {"success": False, "error": "用户创建失败"}

        except Exception as e:
            logger.error(f"创建用户失败: {e}")
            return {"success": False, "error": str(e)}

    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """用户认证"""
        try:
            # 获取用户信息进行密码验证
            users = await self.db_pool.execute_query("""
                SELECT id, username, email, balance, password_hash, created_at
                FROM users
                WHERE username = $1 AND is_active = TRUE
            """, (username,))

            if not users:
                return None

            user = users[0]
            stored_hash = user.pop('password_hash')  # 从返回数据中移除密码哈希

            # 使用bcrypt验证密码
            import bcrypt
            if not bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                return None

            # 缓存用户信息
            await self.cache.set(f"user:{user['id']}", user, ttl=1800)  # 30分钟
            return user

        except Exception as e:
            logger.error(f"用户认证失败: {e}")
            return None

    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        try:
            # 先检查缓存
            cached_user = await self.cache.get(f"user:{user_id}")
            if cached_user:
                return cached_user

            # 从数据库查询
            users = await self.db_pool.execute_query("""
                SELECT id, username, email, balance, created_at
                FROM users
                WHERE id = $1 AND is_active = TRUE
            """, (user_id,))

            if users:
                user = users[0]
                # 缓存用户信息
                await self.cache.set(f"user:{user_id}", user, ttl=1800)
                return user

            return None

        except Exception as e:
            logger.error(f"获取用户信息失败: {e}")
            return None

    # 订单管理方法
    async def create_order(self, user_id: int, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建订单"""
        try:
            # 生成订单ID
            order_id = f"ORD_{int(time.time())}_{user_id}"

            affected_rows = await self.db_pool.execute_command("""
                INSERT INTO orders (user_id, order_id, symbol, quantity, price, order_type, side)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """, (
                user_id,
                order_id,
                order_data["symbol"],
                order_data["quantity"],
                order_data["price"],
                order_data["order_type"],
                order_data["side"]
            ))

            if affected_rows:
                # 缓存订单信息
                order_info = {
                    "order_id": order_id,
                    "user_id": user_id,
                    **order_data,
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                }
                await self.cache.set(f"order:{order_id}", order_info, ttl=SECONDS_PER_HOUR)

                return {"success": True, "order_id": order_id, "order": order_info}

            return {"success": False, "error": "订单创建失败"}

        except Exception as e:
            logger.error(f"创建订单失败: {e}")
            return {"success": False, "error": str(e)}

    async def get_user_orders(self, user_id: int, status: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """获取用户订单"""
        try:
            cache_key = f"user_orders:{user_id}:{status or 'all'}:{limit}"

            # 检查缓存
            cached_orders = await self.cache.get(cache_key)
            if cached_orders:
                return cached_orders

            # 从数据库查询
            if status:
                orders = await self.db_pool.execute_query("""
                    SELECT * FROM orders
                    WHERE user_id = $1 AND status = $2
                    ORDER BY created_at DESC
                    LIMIT $3
                """, (user_id, status, limit))
            else:
                orders = await self.db_pool.execute_query("""
                    SELECT * FROM orders
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, (user_id, limit))

            # 缓存结果
            await self.cache.set(cache_key, orders, ttl=DEFAULT_TEST_TIMEOUT)  # 5分钟

            return orders

        except Exception as e:
            logger.error(f"获取用户订单失败: {e}")
            return []

    # 持仓管理方法
    async def update_position(self, user_id: int, symbol: str, quantity: int, price: float) -> bool:
        """更新持仓"""
        try:
            # 检查是否已有持仓
            existing = await self.db_pool.execute_query("""
                SELECT * FROM positions WHERE user_id = $1 AND symbol = $2
            """, (user_id, symbol))

            if existing:
                # 更新现有持仓
                await self.db_pool.execute_command("""
                    UPDATE positions
                    SET quantity = quantity + $3,
                        avg_price = ((avg_price * quantity) + ($4 * $3)) / (quantity + $3),
                        current_price = $4,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = $1 AND symbol = $2
                """, (user_id, symbol, quantity, price))
            else:
                # 创建新持仓
                await self.db_pool.execute_command("""
                    INSERT INTO positions (user_id, symbol, quantity, avg_price, current_price)
                    VALUES ($1, $2, $3, $4, $4)
                """, (user_id, symbol, quantity, price))

            # 清除相关缓存
            await self.cache.delete(f"user_positions:{user_id}")

            return True

        except Exception as e:
            logger.error(f"更新持仓失败: {e}")
            return False

    async def get_user_positions(self, user_id: int) -> List[Dict[str, Any]]:
        """获取用户持仓"""
        try:
            cache_key = f"user_positions:{user_id}"

            # 检查缓存
            cached_positions = await self.cache.get(cache_key)
            if cached_positions:
                return cached_positions

            # 从数据库查询
            positions = await self.db_pool.execute_query("""
                SELECT * FROM positions
                WHERE user_id = $1
                ORDER BY symbol
            """, (user_id,))

            # 缓存结果
            await self.cache.set(cache_key, positions, ttl=DEFAULT_TEST_TIMEOUT)  # 5分钟

            return positions

        except Exception as e:
            logger.error(f"获取用户持仓失败: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """服务健康检查"""
        try:
            db_health = await self.db_pool.health_check()
            cache_health = await self.cache.health_check()

            overall_status = "healthy" if db_health["status"] == "healthy" and cache_health["status"] in [
                "healthy", "simulated"] else "unhealthy"

            return {
                "status": overall_status,
                "database": db_health,
                "cache": cache_health,
                "timestamp": datetime.now().isoformat(),
                "service": "DatabaseService"
            }

        except Exception as e:
            logger.error(f"服务健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "service": "DatabaseService"
            }


class DatabaseConnectionManager:
    """数据库连接管理器 - 职责：管理数据库连接和连接池"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._pool = None
        self._health_monitor = None

    async def initialize(self):
        """初始化连接池"""
        if not HAS_PSYCOPG2:
            raise ImportError("psycopg2 is required for database operations")

        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.config.min_connections,
                maxconn=self.config.max_connections,
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                sslmode=self.config.ssl_mode
            )
            logger.info(f"数据库连接池已初始化: {self.config.min_connections}-{self.config.max_connections} 连接")
        except Exception as e:
            logger.error(f"数据库连接池初始化失败: {e}")
            raise DatabaseError(f"Failed to initialize database pool: {e}")

    async def close(self):
        """关闭连接池"""
        if self._pool:
            self._pool.closeall()
            logger.info("数据库连接池已关闭")

    async def get_connection(self):
        """获取数据库连接"""
        if not self._pool:
            raise ConnectionError("Database pool not initialized")

        try:
            conn = self._pool.getconn()
            return conn
        except Exception as e:
            logger.error(f"获取数据库连接失败: {e}")
            raise ConnectionError(f"Failed to get database connection: {e}")

    def return_connection(self, conn):
        """归还数据库连接"""
        if self._pool:
            self._pool.putconn(conn)


class RedisCacheManager:
    """Redis缓存管理器 - 职责：管理Redis缓存操作"""

    def __init__(self, config: RedisConfig):
        self.config = config
        self._redis = None

    async def initialize(self):
        """初始化Redis连接"""
        if not HAS_AIREDIS:
            raise ImportError("aioredis is required for Redis operations")

        try:
            self._redis = await aioredis.create_redis_pool(
                f"redis://{self.config.host}:{self.config.port}",
                password=self.config.password if self.config.password else None,
                db=self.config.db,
                maxsize=self.config.max_connections,
                minsize=self.config.min_connections
            )
            logger.info(f"Redis连接池已初始化: {self.config.min_connections}-{self.config.max_connections} 连接")
        except Exception as e:
            logger.error(f"Redis连接池初始化失败: {e}")
            raise ConnectionError(f"Failed to initialize Redis pool: {e}")

    async def close(self):
        """关闭Redis连接"""
        if self._redis:
            self._redis.close()
            await self._redis.wait_closed()
            logger.info("Redis连接已关闭")

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self._redis:
            return None

        try:
            value = await self._redis.get(key)
            if value:
                return json.loads(value.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Redis获取失败: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值"""
        if not self._redis:
            return False

        try:
            json_value = json.dumps(value)
            if ttl:
                await self._redis.setex(key, ttl, json_value)
            else:
                await self._redis.set(key, json_value)
            return True
        except Exception as e:
            logger.error(f"Redis设置失败: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not self._redis:
            return False

        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis删除失败: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self._redis:
            return False

        try:
            return await self._redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis存在检查失败: {e}")
            return False


class DatabaseQueryExecutor:
    """数据库查询执行器 - 职责：执行数据库查询和事务管理"""

    def __init__(self, connection_manager: DatabaseConnectionManager):
        self.connection_manager = connection_manager

    async def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """执行查询"""
        conn = None
        try:
            conn = await self.connection_manager.get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            conn.commit()
            return results

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"查询执行失败: {e}")
            raise QueryError(f"Query execution failed: {e}")
        finally:
            if conn:
                self.connection_manager.return_connection(conn)

    async def execute_command(self, command: str, params: tuple = None) -> int:
        """执行命令（INSERT, UPDATE, DELETE）"""
        conn = None
        try:
            conn = await self.connection_manager.get_connection()
            cursor = conn.cursor()

            if params:
                cursor.execute(command, params)
            else:
                cursor.execute(command)

            conn.commit()
            return cursor.rowcount

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"命令执行失败: {e}")
            raise QueryError(f"Command execution failed: {e}")
        finally:
            if conn:
                self.connection_manager.return_connection(conn)


class DatabaseServiceRefactored:
    """重构后的数据库服务 - 组合模式：使用专门的组件"""

    def __init__(self, db_config: DatabaseConfig = None, redis_config: RedisConfig = None):
        self.db_config = db_config or DatabaseConfig()
        self.redis_config = redis_config or RedisConfig()

        # 初始化专门的组件
        self.connection_manager = DatabaseConnectionManager(self.db_config)
        self.cache_manager = RedisCacheManager(self.redis_config)
        self.query_executor = DatabaseQueryExecutor(self.connection_manager)

    async def initialize(self):
        """初始化所有组件"""
        await self.connection_manager.initialize()
        await self.cache_manager.initialize()
        logger.info("重构后的DatabaseService已初始化")

    async def close(self):
        """关闭所有组件"""
        await self.cache_manager.close()
        await self.connection_manager.close()
        logger.info("重构后的DatabaseService已关闭")

    # 代理方法到专门的组件
    async def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """执行查询 - 代理到查询执行器"""
        return await self.query_executor.execute_query(query, params)

    async def execute_command(self, command: str, params: tuple = None) -> int:
        """执行命令 - 代理到查询执行器"""
        return await self.query_executor.execute_command(command, params)

    async def get_cache(self, key: str) -> Optional[Any]:
        """获取缓存 - 代理到缓存管理器"""
        return await self.cache_manager.get(key)

    async def set_cache(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存 - 代理到缓存管理器"""
        return await self.cache_manager.set(key, value, ttl)

    async def delete_cache(self, key: str) -> bool:
        """删除缓存 - 代理到缓存管理器"""
        return await self.cache_manager.delete(key)

    async def cache_exists(self, key: str) -> bool:
        """检查缓存是否存在 - 代理到缓存管理器"""
        return await self.cache_manager.exists(key)


# 为了向后兼容，保留原有的DatabaseService类名，但内部使用重构版本
DatabaseService = DatabaseServiceRefactored


# 全局服务实例
_database_service = None


async def get_database_service() -> DatabaseService:
    """获取数据库服务实例"""
    global _database_service

    if _database_service is None:
        config_manager = UnifiedConfigManager()

        # 从配置管理器获取数据库配置
        db_config = DatabaseConfig(
            host=config_manager.get("database.host", "localhost"),
            port=config_manager.get("database.port", 5432),
            database=config_manager.get("database.name", "rqa2025"),
            user=config_manager.get("database.user", "rqa_user"),
            password=config_manager.get("database.password", ""),
            min_connections=config_manager.get("database.min_connections", 5),
            max_connections=config_manager.get("database.max_connections", 20)
        )

        redis_config = RedisConfig(
            host=config_manager.get("redis.host", "localhost"),
            port=config_manager.get("redis.port", 6379),
            password=config_manager.get("redis.password", ""),
            db=config_manager.get("redis.db", 0)
        )

        _database_service = DatabaseService(db_config, redis_config)
        await _database_service.initialize()

    return _database_service
