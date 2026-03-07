"""
监控数据持久化管理器 (增强版集成)

集成了高性能、可扩展的监控数据持久化解决方案，包括：
1. 高性能数据存储和检索
2. 数据压缩和归档
3. 实时数据流处理
4. 智能数据生命周期管理
5. 多级缓存机制

原始功能保持兼容，新增增强功能。
"""

import json
import sqlite3
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import deque
import asyncio

# 保持原有导入兼容性
from .features_monitor import MetricType

logger = logging.getLogger(__name__)

# 从增强版本导入核心组件


class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"


class StorageBackend(Enum):
    """存储后端类型"""
    SQLITE = "sqlite"
    JSON = "json"
    PARQUET = "parquet"


class DataLifecyclePolicy(Enum):
    """数据生命周期策略"""
    HOT = "hot"          # 热数据 - 快速访问
    WARM = "warm"        # 温数据 - 中等访问速度
    COLD = "cold"        # 冷数据 - 慢速访问，高压缩


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

    提供高性能、可扩展的监控数据持久化解决方案
    """

    def __init__(self, config: Optional[Dict] = None):
        """初始化增强的持久化管理器"""
        self.config = config or {}
        self.storage_path = Path(self.config.get('path', './monitoring_data_enhanced'))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 存储后端配置
        self.primary_backend = StorageBackend(self.config.get('primary_backend', 'sqlite'))
        self.compression_type = CompressionType(self.config.get('compression', 'lz4'))

        # 归档配置
        self.archive_config = ArchiveConfig(**self.config.get('archive', {}))

        # 多级缓存
        self.hot_cache = {}
        self.warm_cache = deque(maxlen=10000)

        # 批量写入队列
        self.write_queue = queue.Queue(maxsize=50000)
        self.batch_size = self.config.get('batch_size', 500)
        self.batch_timeout = self.config.get('batch_timeout', 2.0)

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))

        # 数据流处理
        self.stream_processors = []

        # 锁
        self.cache_lock = threading.RLock()
        self.write_lock = threading.Lock()

        # 初始化存储
        self._init_storage()

        # 启动后台任务
        self._start_background_tasks()

        logger.info("增强的监控数据持久化管理器初始化完成")

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
            # 创建主表
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

            # 创建高效索引
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

            # 启用WAL模式提高并发性能
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

        # 批量写入线程
        self.write_thread = threading.Thread(target=self._batch_writer_loop, daemon=True)
        self.write_thread.start()

        # 数据归档线程
        self.archive_thread = threading.Thread(target=self._archiver_loop, daemon=True)
        self.archive_thread.start()

        # 缓存清理线程
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

            # 添加到写入队列
            self.write_queue.put_nowait(record)

            # 更新热缓存
            cache_key = f"{component_name}:{metric_name}"
            with self.cache_lock:
                self.hot_cache[cache_key] = record

            # 触发流处理器
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

            # 直接写入存储
            self._write_records_batch([record])

            # 更新缓存
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
                # 尝试从队列获取记录
                try:
                    record = self.write_queue.get(timeout=0.1)
                    batch.append(record)
                except queue.Empty:
                    pass

                # 检查是否需要写入
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
                if self.primary_backend == StorageBackend.SQLITE:
                    self._write_sqlite_batch(records)
                elif self.primary_backend == StorageBackend.PARQUET:
                    self._write_parquet_batch(records)

        except Exception as e:
            logger.error(f"批量写入记录失败: {e}")

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
            # 转换为DataFrame
            data = [asdict(record) for record in records]
            df = pd.DataFrame(data)

            # 序列化labels字段
            df['labels'] = df['labels'].apply(json.dumps)

            # 写入文件
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
                time.sleep(3600)  # 每小时执行一次归档
            except Exception as e:
                logger.error(f"数据归档异常: {e}")
                time.sleep(300)  # 出错后等待5分钟

    def _perform_data_archival(self):
        """执行数据归档"""
        current_time = time.time()

        # 计算时间阈值
        hot_threshold = current_time - (self.archive_config.hot_data_days * 24 * 3600)
        warm_threshold = current_time - (self.archive_config.warm_data_days * 24 * 3600)
        cold_threshold = current_time - (self.archive_config.cold_data_days * 24 * 3600)

        if self.primary_backend == StorageBackend.SQLITE:
            self._archive_sqlite_data(hot_threshold, warm_threshold, cold_threshold)

    def _archive_sqlite_data(self, hot_threshold: float, warm_threshold: float, cold_threshold: float):
        """归档SQLite数据"""
        with sqlite3.connect(self.db_path) as conn:
            # 删除过期的冷数据
            cursor = conn.execute(
                "DELETE FROM metrics WHERE timestamp < ? AND data_tier = 'cold'",
                (cold_threshold,)
            )
            if cursor.rowcount > 0:
                logger.info(f"删除了 {cursor.rowcount} 条过期冷数据")

            # 将温数据移动到冷存储
            conn.execute("""
                UPDATE metrics 
                SET data_tier = 'cold' 
                WHERE timestamp < ? AND data_tier = 'warm'
            """, (warm_threshold,))

            # 将热数据移动到温存储
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
                time.sleep(300)  # 每5分钟清理一次
            except Exception as e:
                logger.error(f"缓存清理异常: {e}")
                time.sleep(60)

    def _cleanup_cache(self):
        """清理缓存"""
        current_time = time.time()
        cache_ttl = 600  # 10分钟缓存TTL

        with self.cache_lock:
            # 清理热缓存
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

        # 设置停止标志
        self._stop_background = True

        # 等待线程结束
        if hasattr(self, 'write_thread') and self.write_thread.is_alive():
            self.write_thread.join(timeout=5.0)

        if hasattr(self, 'archive_thread') and self.archive_thread.is_alive():
            self.archive_thread.join(timeout=5.0)

        if hasattr(self, 'cache_cleanup_thread') and self.cache_cleanup_thread.is_alive():
            self.cache_cleanup_thread.join(timeout=5.0)

        # 刷新剩余的记录
        try:
            remaining_records = []
            while not self.write_queue.empty():
                remaining_records.append(self.write_queue.get_nowait())

            if remaining_records:
                self._write_records_batch(remaining_records)
                logger.info(f"刷新了 {len(remaining_records)} 条剩余记录")
        except Exception as e:
            logger.error(f"刷新剩余记录失败: {e}")

        # 关闭线程池
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
            # 构建查询条件
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

            # 执行查询
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

        except Exception as e:
            logger.error(f"异步查询指标失败: {e}")
            return pd.DataFrame()

    def get_metrics_count(self, component_name: Optional[str] = None,
                          metric_name: Optional[str] = None) -> int:
        """获取指标数量"""
        try:
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

        except Exception as e:
            logger.error(f"获取指标数量失败: {e}")
            return 0

    def get_latest_metrics(self, component_name: str,
                           metric_name: Optional[str] = None) -> Optional[MetricRecord]:
        """获取最新指标"""
        try:
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

        except Exception as e:
            logger.error(f"获取最新指标失败: {e}")
            return None


def get_enhanced_persistence_manager(config: Optional[Dict] = None) -> EnhancedMetricsPersistenceManager:
    """获取增强的持久化管理器实例"""
    return EnhancedMetricsPersistenceManager(config)

# 兼容性适配器 - 保持原有接口不变


class MetricsPersistenceManager:
    """原有接口的兼容性适配器"""

    def __init__(self, storage_config: Optional[Dict] = None):
        """初始化（兼容原有接口）"""
        # 使用增强的管理器
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


# 保持原有函数接口
def get_persistence_manager(config: Optional[Dict] = None) -> MetricsPersistenceManager:
    """获取持久化管理器实例（兼容原有接口）"""
    return MetricsPersistenceManager(config)

# 新增：获取增强版本的直接访问


def get_enhanced_persistence_manager(config: Optional[Dict] = None) -> EnhancedMetricsPersistenceManager:
    """获取增强的持久化管理器实例（新功能）"""
    return EnhancedMetricsPersistenceManager(config)
