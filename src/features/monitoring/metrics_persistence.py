#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控数据持久化管理器 (增强版集成)

集成了高性能、可扩展的监控数据持久化解决方案，包括：
1. 高性能数据存储和检索
2. 数据压缩和归档
3. 实时数据流处理
4. 智能数据生命周期管理
5. 多级缓存机制
6. PostgreSQL优先存储策略

存储优先级:
- 主存储: PostgreSQL 数据库
- 降级存储: SQLite / 文件系统
"""

import json
import sqlite3
import os
import time
import logging
import asyncio
import threading
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from .features_monitor import MetricType

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"


class StorageBackend(Enum):
    """存储后端类型"""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    JSON = "json"
    PARQUET = "parquet"


class DataLifecyclePolicy(Enum):
    """数据生命周期策略"""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class MetricRecord:
    """优化的指标记录"""
    component_name: str
    metric_name: str
    metric_value: float
    metric_type: str
    timestamp: float
    labels: Dict[str, str]
    created_at: str
    ttl: Optional[float] = None
    priority: int = 1


@dataclass
class ArchiveConfig:
    """归档配置"""
    hot_data_days: int = 7
    warm_data_days: int = 30
    cold_data_days: int = 365
    compression_ratio: float = 0.8
    batch_size: int = 1000


class EnhancedMetricsPersistenceManager:
    """
    增强的监控数据持久化管理器
    
    实现PostgreSQL优先存储，数据库连接失败时自动降级到SQLite。
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化增强的持久化管理器"""
        self.config = config or {}
        self.storage_path = Path(self.config.get('path', './monitoring_data_enhanced'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.compression_type = CompressionType(self.config.get('compression', 'lz4'))
        self.archive_config = ArchiveConfig(**self.config.get('archive', {}))
        
        self.hot_cache = {}
        self.warm_cache = deque(maxlen=10000)
        
        self.write_queue = queue.Queue(maxsize=50000)
        self.batch_size = self.config.get('batch_size', 500)
        self.batch_timeout = self.config.get('batch_timeout', 2.0)
        
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        
        self.stream_processors = []
        
        self.cache_lock = threading.RLock()
        self.write_lock = threading.Lock()
        
        self._pg_config = None
        self._pg_available = False
        self.primary_backend = StorageBackend.SQLITE
        
        if self.config.get('enable_postgresql', True):
            self._pg_config = self._get_postgresql_config()
            self._pg_available = self._test_postgresql_connection()
            if self._pg_available:
                self.primary_backend = StorageBackend.POSTGRESQL
                self._ensure_postgresql_tables()
                logger.info("✅ MetricsPersistenceManager: PostgreSQL 存储已启用")
            else:
                logger.warning("⚠️ MetricsPersistenceManager: PostgreSQL 不可用，使用SQLite存储")
        
        self._init_storage()
        self._start_background_tasks()
        
        logger.info(f"增强的监控数据持久化管理器初始化完成，主存储: {self.primary_backend.value}")

    def _get_postgresql_config(self) -> Optional[Dict[str, str]]:
        """获取PostgreSQL配置（使用统一配置模块）"""
        try:
            from src.infrastructure.persistence.database_config import get_db_config
            config = get_db_config()
            return config.to_dict()
        except ImportError:
            logger.debug("统一数据库配置模块导入失败，尝试从环境变量获取")
            return {
                "host": os.getenv("POSTGRES_HOST", "postgres"),
                "port": os.getenv("POSTGRES_PORT", "5432"),
                "database": os.getenv("POSTGRES_DB", "rqa2025_prod"),
                "user": os.getenv("POSTGRES_USER", "rqa2025_admin"),
                "password": os.getenv("POSTGRES_PASSWORD", "SecurePass123!")
            }
        except Exception as e:
            logger.error(f"获取PostgreSQL配置失败: {e}")
            return None

    def _get_db_connection(self):
        """获取PostgreSQL数据库连接（带重试机制）"""
        if not self._pg_config:
            return None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=self._pg_config["host"],
                    port=self._pg_config["port"],
                    database=self._pg_config["database"],
                    user=self._pg_config["user"],
                    password=self._pg_config["password"],
                    connect_timeout=5
                )
                return conn
            except Exception as e:
                logger.debug(f"PostgreSQL连接失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_BASE * (2 ** attempt))
        
        return None

    def _test_postgresql_connection(self) -> bool:
        """测试PostgreSQL连接是否可用"""
        conn = self._get_db_connection()
        if conn:
            conn.close()
            return True
        return False

    def _ensure_postgresql_tables(self):
        """确保PostgreSQL表存在"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return
            
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS monitoring_metrics (
                        id SERIAL PRIMARY KEY,
                        component_name VARCHAR(255) NOT NULL,
                        metric_name VARCHAR(255) NOT NULL,
                        metric_value DOUBLE PRECISION NOT NULL,
                        metric_type VARCHAR(50) NOT NULL,
                        timestamp DOUBLE PRECISION NOT NULL,
                        labels JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ttl DOUBLE PRECISION,
                        priority INTEGER DEFAULT 1,
                        data_tier VARCHAR(10) DEFAULT 'hot'
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_component_time 
                    ON monitoring_metrics(component_name, metric_name, timestamp DESC)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp_tier 
                    ON monitoring_metrics(timestamp, data_tier)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_type_time 
                    ON monitoring_metrics(metric_type, timestamp DESC)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_priority 
                    ON monitoring_metrics(priority DESC, timestamp DESC)
                """)
            
            conn.commit()
            logger.debug("PostgreSQL监控指标表已确保存在")
            
        except Exception as e:
            logger.error(f"创建PostgreSQL监控指标表失败: {e}")
        finally:
            if conn:
                conn.close()

    def _init_storage(self):
        """初始化存储后端"""
        if self.primary_backend == StorageBackend.SQLITE:
            self._init_enhanced_sqlite()
        elif self.primary_backend == StorageBackend.PARQUET:
            self._init_parquet_storage()

    def _init_enhanced_sqlite(self):
        """初始化增强的SQLite存储"""
        self.db_path = self.storage_path / "metrics_enhanced.db"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    labels TEXT,
                    created_at TEXT NOT NULL,
                    ttl REAL,
                    priority INTEGER DEFAULT 1,
                    data_tier TEXT DEFAULT 'hot'
                )
            """)
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_component_metric_time ON metrics(component_name, metric_name, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_timestamp_tier ON metrics(timestamp, data_tier)",
                "CREATE INDEX IF NOT EXISTS idx_metric_type_time ON metrics(metric_type, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_priority_time ON metrics(priority DESC, timestamp DESC)"
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(index_sql)
                except sqlite3.OperationalError:
                    pass
            
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")

    def _init_parquet_storage(self):
        """初始化Parquet存储"""
        self.parquet_path = self.storage_path / "parquet"
        self.parquet_path.mkdir(exist_ok=True)
        
        for tier in ['hot', 'warm', 'cold']:
            (self.parquet_path / tier).mkdir(exist_ok=True)

    def _start_background_tasks(self):
        """启动后台任务"""
        self._stop_background = False
        
        self.write_thread = threading.Thread(target=self._batch_writer_loop, daemon=True)
        self.write_thread.start()
        
        self.archive_thread = threading.Thread(target=self._archiver_loop, daemon=True)
        self.archive_thread.start()
        
        self.cache_cleanup_thread = threading.Thread(target=self._cache_cleanup_loop, daemon=True)
        self.cache_cleanup_thread.start()

    async def store_metric_async(self,
                                 component_name: str,
                                 metric_name: str,
                                 metric_value: float,
                                 metric_type: str,
                                 labels: Optional[Dict[str, str]] = None,
                                 priority: int = 1,
                                 ttl: Optional[float] = None) -> bool:
        """异步存储指标数据"""
        try:
            record = MetricRecord(
                component_name=component_name,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_type=metric_type,
                timestamp=time.time(),
                labels=labels or {},
                created_at=datetime.now().isoformat(),
                priority=priority,
                ttl=ttl
            )
            
            self.write_queue.put_nowait(record)
            
            cache_key = f"{component_name}:{metric_name}"
            with self.cache_lock:
                self.hot_cache[cache_key] = record
            
            await self._trigger_stream_processors(record)
            
            return True
            
        except Exception as e:
            logger.error(f"异步存储指标失败: {e}")
            return False

    def store_metric_sync(self,
                          component_name: str,
                          metric_name: str,
                          metric_value: float,
                          metric_type: str,
                          labels: Optional[Dict[str, str]] = None,
                          priority: int = 1,
                          ttl: Optional[float] = None) -> bool:
        """同步存储指标数据"""
        try:
            record = MetricRecord(
                component_name=component_name,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_type=metric_type,
                timestamp=time.time(),
                labels=labels or {},
                created_at=datetime.now().isoformat(),
                priority=priority,
                ttl=ttl
            )
            
            self._write_records_batch([record])
            
            cache_key = f"{component_name}:{metric_name}"
            with self.cache_lock:
                self.hot_cache[cache_key] = record
            
            return True
            
        except Exception as e:
            logger.error(f"同步存储指标失败: {e}")
            return False

    def _batch_writer_loop(self):
        """批量写入循环"""
        batch = []
        last_write_time = time.time()
        
        while not self._stop_background:
            try:
                try:
                    record = self.write_queue.get(timeout=0.1)
                    batch.append(record)
                except queue.Empty:
                    pass
                
                current_time = time.time()
                should_write = (
                    len(batch) >= self.batch_size or
                    (batch and current_time - last_write_time >= self.batch_timeout)
                )
                
                if should_write:
                    self._write_records_batch(batch)
                    batch.clear()
                    last_write_time = current_time
                
            except Exception as e:
                logger.error(f"批量写入循环异常: {e}")
                time.sleep(1)

    def _write_records_batch(self, records: List[MetricRecord]):
        """批量写入记录"""
        if not records:
            return
        
        try:
            with self.write_lock:
                if self.primary_backend == StorageBackend.POSTGRESQL:
                    if not self._write_postgresql_batch(records):
                        logger.warning("PostgreSQL写入失败，降级到SQLite")
                        self._write_sqlite_batch(records)
                elif self.primary_backend == StorageBackend.SQLITE:
                    self._write_sqlite_batch(records)
                elif self.primary_backend == StorageBackend.PARQUET:
                    self._write_parquet_batch(records)
                    
        except Exception as e:
            logger.error(f"批量写入记录失败: {e}")

    def _write_postgresql_batch(self, records: List[MetricRecord]) -> bool:
        """批量写入PostgreSQL"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                for record in records:
                    cur.execute("""
                        INSERT INTO monitoring_metrics (
                            component_name, metric_name, metric_value, metric_type,
                            timestamp, labels, created_at, ttl, priority, data_tier
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        record.component_name,
                        record.metric_name,
                        record.metric_value,
                        record.metric_type,
                        record.timestamp,
                        json.dumps(record.labels),
                        record.created_at,
                        record.ttl,
                        record.priority,
                        'hot'
                    ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"写入PostgreSQL失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def _write_sqlite_batch(self, records: List[MetricRecord]):
        """批量写入SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT INTO metrics (
                    component_name, metric_name, metric_value, metric_type,
                    timestamp, labels, created_at, ttl, priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    record.component_name, record.metric_name, record.metric_value,
                    record.metric_type, record.timestamp, json.dumps(record.labels),
                    record.created_at, record.ttl, record.priority
                ) for record in records
            ])

    def _write_parquet_batch(self, records: List[MetricRecord]):
        """批量写入Parquet"""
        try:
            data = [asdict(record) for record in records]
            df = pd.DataFrame(data)
            df['labels'] = df['labels'].apply(json.dumps)
            
            current_time = time.time()
            file_path = self.parquet_path / f"metrics_{int(current_time)}.parquet"
            df.to_parquet(file_path, compression='snappy')
            
        except Exception as e:
            logger.warning(f"Parquet写入失败，回退到SQLite: {e}")
            self._write_sqlite_batch(records)

    async def _trigger_stream_processors(self, record: MetricRecord):
        """触发流处理器"""
        for processor in self.stream_processors:
            try:
                if asyncio.iscoroutinefunction(processor):
                    await processor(record)
                else:
                    processor(record)
            except Exception as e:
                logger.error(f"流处理器执行失败: {e}")

    def _archiver_loop(self):
        """数据归档循环"""
        while not self._stop_background:
            try:
                self._perform_data_archival()
                time.sleep(3600)
            except Exception as e:
                logger.error(f"数据归档异常: {e}")
                time.sleep(300)

    def _perform_data_archival(self):
        """执行数据归档"""
        current_time = time.time()
        
        hot_threshold = current_time - (self.archive_config.hot_data_days * 24 * 3600)
        warm_threshold = current_time - (self.archive_config.warm_data_days * 24 * 3600)
        cold_threshold = current_time - (self.archive_config.cold_data_days * 24 * 3600)
        
        if self.primary_backend == StorageBackend.POSTGRESQL:
            self._archive_postgresql_data(hot_threshold, warm_threshold, cold_threshold)
        elif self.primary_backend == StorageBackend.SQLITE:
            self._archive_sqlite_data(hot_threshold, warm_threshold, cold_threshold)

    def _archive_postgresql_data(self, hot_threshold: float, warm_threshold: float, cold_threshold: float):
        """归档PostgreSQL数据"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return
            
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM monitoring_metrics WHERE timestamp < %s AND data_tier = %s",
                    (cold_threshold, 'cold')
                )
                deleted_count = cur.rowcount
                if deleted_count > 0:
                    logger.info(f"删除了 {deleted_count} 条过期冷数据")
                
                cur.execute("""
                    UPDATE monitoring_metrics 
                    SET data_tier = %s 
                    WHERE timestamp < %s AND data_tier = %s
                """, ('cold', warm_threshold, 'warm'))
                
                cur.execute("""
                    UPDATE monitoring_metrics 
                    SET data_tier = %s 
                    WHERE timestamp < %s AND data_tier = %s
                """, ('warm', hot_threshold, 'hot'))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"归档PostgreSQL数据失败: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _archive_sqlite_data(self, hot_threshold: float, warm_threshold: float, cold_threshold: float):
        """归档SQLite数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM metrics WHERE timestamp < ? AND data_tier = 'cold'",
                (cold_threshold,)
            )
            if cursor.rowcount > 0:
                logger.info(f"删除了 {cursor.rowcount} 条过期冷数据")
            
            conn.execute("""
                UPDATE metrics 
                SET data_tier = 'cold' 
                WHERE timestamp < ? AND data_tier = 'warm'
            """, (warm_threshold,))
            
            conn.execute("""
                UPDATE metrics 
                SET data_tier = 'warm' 
                WHERE timestamp < ? AND data_tier = 'hot'
            """, (hot_threshold,))

    def _cache_cleanup_loop(self):
        """缓存清理循环"""
        while not self._stop_background:
            try:
                self._cleanup_cache()
                time.sleep(300)
            except Exception as e:
                logger.error(f"缓存清理异常: {e}")
                time.sleep(60)

    def _cleanup_cache(self):
        """清理缓存"""
        current_time = time.time()
        cache_ttl = 600
        
        with self.cache_lock:
            expired_keys = [
                key for key, record in self.hot_cache.items()
                if current_time - record.timestamp > cache_ttl
            ]
            
            for key in expired_keys:
                del self.hot_cache[key]
            
            if expired_keys:
                logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")

    def stop(self):
        """停止持久化管理器"""
        logger.info("正在停止增强的监控数据持久化管理器...")
        
        self._stop_background = True
        
        if hasattr(self, 'write_thread') and self.write_thread.is_alive():
            self.write_thread.join(timeout=5.0)
        
        if hasattr(self, 'archive_thread') and self.archive_thread.is_alive():
            self.archive_thread.join(timeout=5.0)
        
        if hasattr(self, 'cache_cleanup_thread') and self.cache_cleanup_thread.is_alive():
            self.cache_cleanup_thread.join(timeout=5.0)
        
        try:
            remaining_records = []
            while not self.write_queue.empty():
                remaining_records.append(self.write_queue.get_nowait())
            
            if remaining_records:
                self._write_records_batch(remaining_records)
                logger.info(f"刷新了 {len(remaining_records)} 条剩余记录")
        except Exception as e:
            logger.error(f"刷新剩余记录失败: {e}")
        
        self.executor.shutdown(wait=True)
        
        logger.info("增强的监控数据持久化管理器已停止")

    async def query_metrics_async(self,
                                   component_name: Optional[str] = None,
                                   metric_name: Optional[str] = None,
                                   start_time: Optional[float] = None,
                                   end_time: Optional[float] = None,
                                   metric_type: Optional[str] = None,
                                   limit: Optional[int] = None) -> pd.DataFrame:
        """异步查询指标数据"""
        try:
            if self.primary_backend == StorageBackend.POSTGRESQL:
                return self._query_postgresql_metrics(
                    component_name, metric_name, start_time, end_time, metric_type, limit
                )
            else:
                return self._query_sqlite_metrics(
                    component_name, metric_name, start_time, end_time, metric_type, limit
                )
                
        except Exception as e:
            logger.error(f"异步查询指标失败: {e}")
            return pd.DataFrame()

    def _query_postgresql_metrics(self,
                                   component_name: Optional[str] = None,
                                   metric_name: Optional[str] = None,
                                   start_time: Optional[float] = None,
                                   end_time: Optional[float] = None,
                                   metric_type: Optional[str] = None,
                                   limit: Optional[int] = None) -> pd.DataFrame:
        """从PostgreSQL查询指标数据"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return self._query_sqlite_metrics(
                    component_name, metric_name, start_time, end_time, metric_type, limit
                )
            
            conditions = []
            params = []
            
            if component_name:
                conditions.append("component_name = %s")
                params.append(component_name)
            
            if metric_name:
                conditions.append("metric_name = %s")
                params.append(metric_name)
            
            if start_time:
                conditions.append("timestamp >= %s")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= %s")
                params.append(end_time)
            
            if metric_type:
                conditions.append("metric_type = %s")
                params.append(metric_type)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            query = f"""
                SELECT component_name, metric_name, metric_value, metric_type,
                       timestamp, labels, created_at
                FROM monitoring_metrics
                WHERE {where_clause}
                ORDER BY timestamp DESC
                {limit_clause}
            """
            
            import psycopg2.extras
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            
            if rows:
                df = pd.DataFrame(rows)
                df['labels'] = df['labels'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
                return df
            
            return pd.DataFrame(columns=[
                'component_name', 'metric_name', 'metric_value', 'metric_type',
                'timestamp', 'labels', 'created_at'
            ])
            
        except Exception as e:
            logger.error(f"从PostgreSQL查询指标失败: {e}")
            return self._query_sqlite_metrics(
                component_name, metric_name, start_time, end_time, metric_type, limit
            )
        finally:
            if conn:
                conn.close()

    def _query_sqlite_metrics(self,
                               component_name: Optional[str] = None,
                               metric_name: Optional[str] = None,
                               start_time: Optional[float] = None,
                               end_time: Optional[float] = None,
                               metric_type: Optional[str] = None,
                               limit: Optional[int] = None) -> pd.DataFrame:
        """从SQLite查询指标数据"""
        conditions = []
        params = []
        
        if component_name:
            conditions.append("component_name = ?")
            params.append(component_name)
        
        if metric_name:
            conditions.append("metric_name = ?")
            params.append(metric_name)
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        
        if metric_type:
            conditions.append("metric_type = ?")
            params.append(metric_type)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        with sqlite3.connect(self.db_path) as conn:
            query = f"""
                SELECT component_name, metric_name, metric_value, metric_type,
                       timestamp, labels, created_at
                FROM metrics
                WHERE {where_clause}
                ORDER BY timestamp DESC
                {limit_clause}
            """
            df = pd.read_sql_query(query, conn, params=params)
        
        return df if not df.empty else pd.DataFrame(columns=[
            'component_name', 'metric_name', 'metric_value', 'metric_type',
            'timestamp', 'labels', 'created_at'
        ])

    def get_metrics_count(self, component_name: Optional[str] = None,
                          metric_name: Optional[str] = None) -> int:
        """获取指标数量"""
        try:
            if self.primary_backend == StorageBackend.POSTGRESQL:
                return self._get_postgresql_metrics_count(component_name, metric_name)
            else:
                return self._get_sqlite_metrics_count(component_name, metric_name)
                
        except Exception as e:
            logger.error(f"获取指标数量失败: {e}")
            return 0

    def _get_postgresql_metrics_count(self, component_name: Optional[str] = None,
                                       metric_name: Optional[str] = None) -> int:
        """从PostgreSQL获取指标数量"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return self._get_sqlite_metrics_count(component_name, metric_name)
            
            conditions = []
            params = []
            
            if component_name:
                conditions.append("component_name = %s")
                params.append(component_name)
            
            if metric_name:
                conditions.append("metric_name = %s")
                params.append(metric_name)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM monitoring_metrics WHERE {where_clause}", params)
                result = cur.fetchone()
                return result[0] if result else 0
                
        except Exception as e:
            logger.error(f"从PostgreSQL获取指标数量失败: {e}")
            return self._get_sqlite_metrics_count(component_name, metric_name)
        finally:
            if conn:
                conn.close()

    def _get_sqlite_metrics_count(self, component_name: Optional[str] = None,
                                   metric_name: Optional[str] = None) -> int:
        """从SQLite获取指标数量"""
        conditions = []
        params = []
        
        if component_name:
            conditions.append("component_name = ?")
            params.append(component_name)
        
        if metric_name:
            conditions.append("metric_name = ?")
            params.append(metric_name)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with sqlite3.connect(self.db_path) as conn:
            query = f"SELECT COUNT(*) as count FROM metrics WHERE {where_clause}"
            result = conn.execute(query, params).fetchone()
            return result[0] if result else 0

    def get_latest_metrics(self, component_name: str,
                           metric_name: Optional[str] = None) -> Optional[MetricRecord]:
        """获取最新指标"""
        try:
            if self.primary_backend == StorageBackend.POSTGRESQL:
                return self._get_latest_postgresql_metrics(component_name, metric_name)
            else:
                return self._get_latest_sqlite_metrics(component_name, metric_name)
                
        except Exception as e:
            logger.error(f"获取最新指标失败: {e}")
            return None

    def _get_latest_postgresql_metrics(self, component_name: str,
                                        metric_name: Optional[str] = None) -> Optional[MetricRecord]:
        """从PostgreSQL获取最新指标"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return self._get_latest_sqlite_metrics(component_name, metric_name)
            
            conditions = ["component_name = %s"]
            params = [component_name]
            
            if metric_name:
                conditions.append("metric_name = %s")
                params.append(metric_name)
            
            where_clause = " AND ".join(conditions)
            
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT component_name, metric_name, metric_value, metric_type,
                           timestamp, labels, created_at, ttl, priority
                    FROM monitoring_metrics
                    WHERE {where_clause}
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, params)
                
                result = cur.fetchone()
                
                if result:
                    return MetricRecord(
                        component_name=result[0],
                        metric_name=result[1],
                        metric_value=result[2],
                        metric_type=result[3],
                        timestamp=result[4],
                        labels=result[5] if isinstance(result[5], dict) else json.loads(result[5] or '{}'),
                        created_at=str(result[6]),
                        ttl=result[7],
                        priority=result[8] or 1
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"从PostgreSQL获取最新指标失败: {e}")
            return self._get_latest_sqlite_metrics(component_name, metric_name)
        finally:
            if conn:
                conn.close()

    def _get_latest_sqlite_metrics(self, component_name: str,
                                    metric_name: Optional[str] = None) -> Optional[MetricRecord]:
        """从SQLite获取最新指标"""
        conditions = ["component_name = ?"]
        params = [component_name]
        
        if metric_name:
            conditions.append("metric_name = ?")
            params.append(metric_name)
        
        where_clause = " AND ".join(conditions)
        
        with sqlite3.connect(self.db_path) as conn:
            query = f"""
                SELECT component_name, metric_name, metric_value, metric_type,
                       timestamp, labels, created_at, ttl, priority
                FROM metrics
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT 1
            """
            result = conn.execute(query, params).fetchone()
            
            if result:
                return MetricRecord(
                    component_name=result[0],
                    metric_name=result[1],
                    metric_value=result[2],
                    metric_type=result[3],
                    timestamp=result[4],
                    labels=json.loads(result[5]) if result[5] else {},
                    created_at=result[6],
                    ttl=result[7],
                    priority=result[8]
                )
        
        return None

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = {
            "primary_backend": self.primary_backend.value,
            "postgresql_available": self._pg_available,
            "cache_size": len(self.hot_cache),
            "queue_size": self.write_queue.qsize(),
            "metrics_count": self.get_metrics_count()
        }
        
        if self._pg_available:
            conn = None
            try:
                conn = self._get_db_connection()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT data_tier, COUNT(*) 
                            FROM monitoring_metrics 
                            GROUP BY data_tier
                        """)
                        tier_counts = dict(cur.fetchall())
                        stats["postgresql_tiers"] = tier_counts
            except Exception:
                pass
            finally:
                if conn:
                    conn.close()
        
        return stats

    def sync_to_postgresql(self) -> int:
        """将SQLite中的数据同步到PostgreSQL"""
        if not self._pg_available:
            logger.warning("PostgreSQL不可用，无法同步")
            return 0
        
        synced = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT component_name, metric_name, metric_value, metric_type,
                           timestamp, labels, created_at, ttl, priority
                    FROM metrics
                """)
                
                records = []
                for row in cursor.fetchall():
                    records.append(MetricRecord(
                        component_name=row[0],
                        metric_name=row[1],
                        metric_value=row[2],
                        metric_type=row[3],
                        timestamp=row[4],
                        labels=json.loads(row[5]) if row[5] else {},
                        created_at=row[6],
                        ttl=row[7],
                        priority=row[8] or 1
                    ))
                
                if records:
                    if self._write_postgresql_batch(records):
                        synced = len(records)
                        conn.execute("DELETE FROM metrics")
                        logger.info(f"已同步 {synced} 条指标到PostgreSQL")
                        
        except Exception as e:
            logger.error(f"同步指标到PostgreSQL失败: {e}")
        
        return synced


def get_enhanced_persistence_manager(config: Optional[Dict] = None) -> EnhancedMetricsPersistenceManager:
    """获取增强的持久化管理器实例"""
    return EnhancedMetricsPersistenceManager(config)


class MetricsPersistenceManager:
    """原有接口的兼容性适配器"""

    def __init__(self, storage_config: Optional[Dict] = None):
        """初始化（兼容原有接口）"""
        self._enhanced_manager = EnhancedMetricsPersistenceManager(storage_config)

    def store_metric(self, component_name: str, metric_name: str,
                     metric_value: float, metric_type: MetricType,
                     labels: Optional[Dict[str, str]] = None) -> None:
        """存储指标数据（兼容原有接口）"""
        self._enhanced_manager.store_metric_sync(
            component_name=component_name,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_type=metric_type.value if hasattr(metric_type, 'value') else str(metric_type),
            labels=labels
        )

    def query_metrics(self, component_name: Optional[str] = None,
                      metric_name: Optional[str] = None,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      metric_type: Optional[MetricType] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """查询指标数据（兼容原有接口）"""
        return asyncio.run(self._enhanced_manager.query_metrics_async(
            component_name=component_name,
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            metric_type=metric_type.value if metric_type and hasattr(
                metric_type, 'value') else None,
            limit=limit
        ))

    def stop(self) -> None:
        """停止管理器（兼容原有接口）"""
        self._enhanced_manager.stop()


def get_persistence_manager(config: Optional[Dict] = None) -> MetricsPersistenceManager:
    """获取持久化管理器实例（兼容原有接口）"""
    return MetricsPersistenceManager(config)
