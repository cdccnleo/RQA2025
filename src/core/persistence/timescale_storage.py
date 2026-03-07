#!/usr/bin/env python3
"""
TimescaleDB时序数据存储优化
提供历史数据的分区存储、压缩和查询优化
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncpg
import json

logger = logging.getLogger(__name__)


@dataclass
class TimescaleConfig:
    """TimescaleDB配置"""
    host: str = "localhost"
    port: int = 5432
    database: str = "rqa2025"
    user: str = "rqa2025_admin"
    password: str = ""
    min_connections: int = 5
    max_connections: int = 20
    command_timeout: float = 60.0
    hypertable_chunk_size: str = "1 year"  # 超表分块大小
    compression_policy: str = "3 months"  # 压缩策略


@dataclass
class StorageStats:
    """存储统计信息"""
    table_name: str
    total_records: int = 0
    compressed_records: int = 0
    uncompressed_records: int = 0
    total_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 0.0
    oldest_record: Optional[datetime] = None
    newest_record: Optional[datetime] = None


class TimescaleStorage:
    """TimescaleDB时序数据存储"""

    def __init__(self, config: Dict[str, Any]):
        self.config = TimescaleConfig(**config)
        self.logger = logging.getLogger(__name__)

        # 连接池
        self.pool: Optional[asyncpg.Pool] = None

        # 表定义
        self.table_definitions = {
            "historical_stock_data": """
                CREATE TABLE IF NOT EXISTS historical_stock_data (
                    symbol VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    open_price DECIMAL(10,2),
                    high_price DECIMAL(10,2),
                    low_price DECIMAL(10,2),
                    close_price DECIMAL(10,2),
                    volume BIGINT,
                    amount DECIMAL(15,2),
                    adj_close DECIMAL(10,2),
                    data_source VARCHAR(50),
                    quality_score DECIMAL(3,2),
                    batch_id VARCHAR(100),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (symbol, date)
                );
            """,
            "historical_index_data": """
                CREATE TABLE IF NOT EXISTS historical_index_data (
                    symbol VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    open_value DECIMAL(12,2),
                    high_value DECIMAL(12,2),
                    low_value DECIMAL(12,2),
                    close_value DECIMAL(12,2),
                    volume BIGINT,
                    amount DECIMAL(15,2),
                    pe_ratio DECIMAL(8,2),
                    pb_ratio DECIMAL(8,2),
                    data_source VARCHAR(50),
                    quality_score DECIMAL(3,2),
                    batch_id VARCHAR(100),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (symbol, date)
                );
            """
        }

    async def initialize(self):
        """初始化TimescaleDB连接和表结构"""
        try:
            # 创建连接池
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout
            )

            self.logger.info("TimescaleDB连接池创建成功")

            # 创建表结构
            await self._create_tables()

            # 转换为超表
            await self._create_hypertables()

            # 设置压缩策略
            await self._setup_compression_policies()

            # 创建索引
            await self._create_indexes()

            self.logger.info("TimescaleDB初始化完成")

        except Exception as e:
            self.logger.error(f"TimescaleDB初始化失败: {e}")
            raise

    async def _create_tables(self):
        """创建基础表"""
        async with self.pool.acquire() as conn:
            for table_name, create_sql in self.table_definitions.items():
                try:
                    await conn.execute(create_sql)
                    self.logger.info(f"创建表成功: {table_name}")
                except Exception as e:
                    self.logger.error(f"创建表失败 {table_name}: {e}")
                    raise

    async def _create_hypertables(self):
        """将普通表转换为超表"""
        async with self.pool.acquire() as conn:
            for table_name in self.table_definitions.keys():
                try:
                    # 检查是否已经是超表
                    result = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT 1 FROM timescaledb_information.hypertables
                            WHERE hypertable_name = $1
                        )
                    """, table_name)

                    if not result:
                        # 转换为超表，按日期分块
                        await conn.execute(f"""
                            SELECT create_hypertable(
                                '{table_name}',
                                'date',
                                chunk_time_interval => INTERVAL '{self.config.hypertable_chunk_size}'
                            )
                        """)
                        self.logger.info(f"创建超表成功: {table_name}")
                    else:
                        self.logger.info(f"超表已存在: {table_name}")

                except Exception as e:
                    self.logger.error(f"创建超表失败 {table_name}: {e}")
                    raise

    async def _setup_compression_policies(self):
        """设置压缩策略"""
        async with self.pool.acquire() as conn:
            for table_name in self.table_definitions.keys():
                try:
                    # 检查是否已有压缩策略
                    result = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT 1 FROM timescaledb_information.compression_settings
                            WHERE hypertable_name = $1
                        )
                    """, table_name)

                    if not result:
                        # 设置压缩策略：3个月前的旧数据自动压缩
                        await conn.execute(f"""
                            ALTER TABLE {table_name} SET (
                                timescaledb.compress,
                                timescaledb.compress_segmentby = 'symbol',
                                timescaledb.compress_orderby = 'date DESC'
                            )
                        """)

                        # 添加压缩策略
                        await conn.execute(f"""
                            SELECT add_compression_policy(
                                '{table_name}',
                                INTERVAL '{self.config.compression_policy}'
                            )
                        """)

                        self.logger.info(f"设置压缩策略成功: {table_name}")
                    else:
                        self.logger.info(f"压缩策略已存在: {table_name}")

                except Exception as e:
                    self.logger.error(f"设置压缩策略失败 {table_name}: {e}")

    async def _create_indexes(self):
        """创建优化索引"""
        async with self.pool.acquire() as conn:
            index_definitions = [
                # 按标的和日期的复合索引（主键已包含）
                # 时间范围查询索引
                "CREATE INDEX IF NOT EXISTS idx_historical_stock_date_symbol ON historical_stock_data (date DESC, symbol)",
                "CREATE INDEX IF NOT EXISTS idx_historical_index_date_symbol ON historical_index_data (date DESC, symbol)",

                # 数据源索引
                "CREATE INDEX IF NOT EXISTS idx_historical_stock_source ON historical_stock_data (data_source)",
                "CREATE INDEX IF NOT EXISTS idx_historical_index_source ON historical_index_data (data_source)",

                # 质量分数索引
                "CREATE INDEX IF NOT EXISTS idx_historical_stock_quality ON historical_stock_data (quality_score)",
                "CREATE INDEX IF NOT EXISTS idx_historical_index_quality ON historical_index_data (quality_score)",

                # 批量ID索引（用于追踪）
                "CREATE INDEX IF NOT EXISTS idx_historical_stock_batch ON historical_stock_data (batch_id)",
                "CREATE INDEX IF NOT EXISTS idx_historical_index_batch ON historical_index_data (batch_id)",

                # 复合索引用于常见查询模式
                "CREATE INDEX IF NOT EXISTS idx_historical_stock_symbol_date_quality ON historical_stock_data (symbol, date DESC, quality_score)",
                "CREATE INDEX IF NOT EXISTS idx_historical_index_symbol_date_quality ON historical_index_data (symbol, date DESC, quality_score)",
            ]

            for index_sql in index_definitions:
                try:
                    await conn.execute(index_sql)
                except Exception as e:
                    self.logger.warning(f"创建索引失败: {index_sql[:50]}...: {e}")

            self.logger.info("索引创建完成")

    async def store_historical_data(self, symbol: str, data: List[Dict[str, Any]], data_type: str = "stock"):
        """
        存储历史数据

        Args:
            symbol: 标的代码
            data: 数据记录列表
            data_type: 数据类型 (stock/index/fund等)
        """
        if not data:
            return

        table_name = f"historical_{data_type}_data"
        if table_name not in self.table_definitions:
            raise ValueError(f"不支持的数据类型: {data_type}")

        async with self.pool.acquire() as conn:
            try:
                # 批量插入数据，使用ON CONFLICT DO UPDATE进行更新
                await self._batch_insert_data(conn, table_name, symbol, data)

                self.logger.info(f"存储历史数据成功: {symbol}, {len(data)}条记录, 类型: {data_type}")

            except Exception as e:
                self.logger.error(f"存储历史数据失败 {symbol}: {e}")
                raise

    async def _batch_insert_data(self, conn: asyncpg.Connection, table_name: str,
                                symbol: str, data: List[Dict[str, Any]]):
        """批量插入数据"""
        if not data:
            return

        # 确定字段映射
        field_mapping = self._get_field_mapping(table_name)

        # 准备数据
        records = []
        for record in data:
            processed_record = self._process_record(record, field_mapping)
            records.append(processed_record)

        # 批量插入
        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            # 构建插入语句
            columns = list(field_mapping.keys())
            placeholders = [f"${j+1}" for j in range(len(columns))]

            insert_sql = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT (symbol, date) DO UPDATE SET
                    {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['symbol', 'date']])},
                    updated_at = NOW()
            """

            # 批量执行
            batch_values = [
                tuple(record[col] for col in columns)
                for record in batch
            ]

            await conn.executemany(insert_sql, batch_values)

    def _get_field_mapping(self, table_name: str) -> Dict[str, str]:
        """获取字段映射"""
        if table_name == "historical_stock_data":
            return {
                "symbol": "symbol",
                "date": "date",
                "open_price": "open",
                "high_price": "high",
                "low_price": "low",
                "close_price": "close",
                "volume": "volume",
                "amount": "amount",
                "adj_close": "adj_close",
                "data_source": "source",
                "quality_score": "quality_score",
                "batch_id": "batch_id"
            }
        elif table_name == "historical_index_data":
            return {
                "symbol": "symbol",
                "date": "date",
                "open_value": "open",
                "high_value": "high",
                "low_value": "low",
                "close_value": "close",
                "volume": "volume",
                "amount": "amount",
                "pe_ratio": "pe_ratio",
                "pb_ratio": "pb_ratio",
                "data_source": "source",
                "quality_score": "quality_score",
                "batch_id": "batch_id"
            }
        else:
            raise ValueError(f"未知的表名: {table_name}")

    def _process_record(self, record: Dict[str, Any], field_mapping: Dict[str, str]) -> Dict[str, Any]:
        """处理单条记录"""
        processed = {}

        for db_field, data_field in field_mapping.items():
            value = record.get(data_field)

            # 类型转换
            if db_field in ['open_price', 'high_price', 'low_price', 'close_price', 'adj_close']:
                processed[db_field] = float(value) if value is not None else None
            elif db_field in ['volume']:
                processed[db_field] = int(float(value)) if value is not None else None
            elif db_field in ['amount']:
                processed[db_field] = float(value) if value is not None else None
            elif db_field == 'quality_score':
                processed[db_field] = float(value) if value is not None else 0.0
            elif db_field == 'date':
                if isinstance(value, str):
                    processed[db_field] = datetime.strptime(value, '%Y-%m-%d').date()
                elif isinstance(value, datetime):
                    processed[db_field] = value.date()
                else:
                    processed[db_field] = value
            else:
                processed[db_field] = value

        return processed

    async def get_data_stats(self, symbol: str, start_year: int, end_year: int,
                           data_type: str = "stock") -> Dict[str, Any]:
        """
        获取数据统计信息

        Args:
            symbol: 标的代码
            start_year: 开始年份
            end_year: 结束年份
            data_type: 数据类型

        Returns:
            统计信息字典
        """
        table_name = f"historical_{data_type}_data"
        start_date = datetime(start_year, 1, 1).date()
        end_date = datetime(end_year, 12, 31).date()

        async with self.pool.acquire() as conn:
            try:
                # 基本统计
                stats_result = await conn.fetchrow(f"""
                    SELECT
                        COUNT(*) as total_records,
                        MIN(date) as oldest_date,
                        MAX(date) as newest_date,
                        AVG(quality_score) as avg_quality,
                        COUNT(CASE WHEN quality_score >= 0.8 THEN 1 END) as high_quality_records
                    FROM {table_name}
                    WHERE symbol = $1 AND date BETWEEN $2 AND $3
                """, symbol, start_date, end_date)

                if not stats_result:
                    return {
                        "total_records": 0,
                        "oldest_date": None,
                        "newest_date": None,
                        "avg_quality": 0.0,
                        "years_with_data": 0,
                        "completeness_ratio": 0.0,
                        "is_complete": False
                    }

                # 计算年度覆盖率
                years_covered = set()
                if stats_result['oldest_date'] and stats_result['newest_date']:
                    current_year = stats_result['oldest_date'].year
                    while current_year <= stats_result['newest_date'].year:
                        years_covered.add(current_year)
                        current_year += 1

                total_years = end_year - start_year + 1
                years_with_data = len(years_covered)

                # 估算完整性（简化计算）
                expected_records = total_years * 252  # 每年约252个交易日
                actual_records = stats_result['total_records'] or 0
                completeness_ratio = min(actual_records / expected_records, 1.0) if expected_records > 0 else 0.0

                return {
                    "total_records": actual_records,
                    "oldest_date": stats_result['oldest_date'],
                    "newest_date": stats_result['newest_date'],
                    "avg_quality": float(stats_result['avg_quality'] or 0),
                    "high_quality_records": stats_result['high_quality_records'] or 0,
                    "years_with_data": years_with_data,
                    "total_years": total_years,
                    "completeness_ratio": completeness_ratio,
                    "is_complete": completeness_ratio >= 0.85  # 85%作为完整阈值
                }

            except Exception as e:
                self.logger.error(f"获取数据统计失败 {symbol}: {e}")
                return {
                    "error": str(e),
                    "total_records": 0,
                    "is_complete": False
                }

    async def query_historical_data(self, symbol: str, start_date: datetime, end_date: datetime,
                                  data_type: str = "stock", quality_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        查询历史数据

        Args:
            symbol: 标的代码
            start_date: 开始日期
            end_date: 结束日期
            data_type: 数据类型
            quality_threshold: 质量阈值

        Returns:
            数据记录列表
        """
        table_name = f"historical_{data_type}_data"

        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(f"""
                    SELECT * FROM {table_name}
                    WHERE symbol = $1
                      AND date BETWEEN $2 AND $3
                      AND quality_score >= $4
                    ORDER BY date ASC
                """, symbol, start_date.date(), end_date.date(), quality_threshold)

                # 转换为字典格式
                result = []
                for row in rows:
                    record = dict(row)
                    # 将日期转换为字符串格式
                    if 'date' in record:
                        record['date'] = record['date'].isoformat()
                    result.append(record)

                self.logger.info(f"查询历史数据成功: {symbol}, {len(result)}条记录")
                return result

            except Exception as e:
                self.logger.error(f"查询历史数据失败 {symbol}: {e}")
                return []

    async def get_storage_stats(self, table_name: str = None) -> List[StorageStats]:
        """
        获取存储统计信息

        Args:
            table_name: 表名，如果为None则返回所有表

        Returns:
            存储统计列表
        """
        tables_to_check = [table_name] if table_name else list(self.table_definitions.keys())

        stats_list = []

        async with self.pool.acquire() as conn:
            for table in tables_to_check:
                try:
                    # 获取表大小信息
                    size_result = await conn.fetchrow("""
                        SELECT
                            pg_size_pretty(pg_total_relation_size($1)) as total_size,
                            pg_total_relation_size($1) as total_size_bytes,
                            pg_size_pretty(pg_table_size($1)) as table_size,
                            pg_table_size($1) as table_size_bytes
                        FROM information_schema.tables
                        WHERE table_name = $1
                    """, table)

                    # 获取记录数和压缩信息
                    data_result = await conn.fetchrow(f"""
                        SELECT
                            COUNT(*) as total_records,
                            COUNT(CASE WHEN _timescaledb_internal.compressed_chunk_stats IS NOT NULL THEN 1 END) as compressed_chunks,
                            MIN(date) as oldest_record,
                            MAX(date) as newest_record
                        FROM {table}
                    """)

                    # 计算压缩统计（简化版）
                    compressed_records = 0
                    uncompressed_records = data_result['total_records'] or 0

                    # 估算压缩比（实际应该从TimescaleDB元数据获取）
                    compression_ratio = 0.3 if data_result['compressed_chunks'] > 0 else 0.0

                    stats = StorageStats(
                        table_name=table,
                        total_records=data_result['total_records'] or 0,
                        compressed_records=compressed_records,
                        uncompressed_records=uncompressed_records,
                        total_size_bytes=size_result['total_size_bytes'] if size_result else 0,
                        compressed_size_bytes=int((size_result['total_size_bytes'] if size_result else 0) * compression_ratio),
                        compression_ratio=compression_ratio,
                        oldest_record=data_result['oldest_record'],
                        newest_record=data_result['newest_record']
                    )

                    stats_list.append(stats)

                except Exception as e:
                    self.logger.error(f"获取存储统计失败 {table}: {e}")

        return stats_list

    async def optimize_table(self, table_name: str):
        """优化表性能"""
        async with self.pool.acquire() as conn:
            try:
                # 执行VACUUM ANALYZE
                await conn.execute(f"VACUUM ANALYZE {table_name}")

                # 重新整理分块（如果需要）
                await conn.execute("""
                    SELECT reorder_chunk($1, if_order => true)
                    FROM show_chunks($1, older_than => INTERVAL '1 month')
                """, table_name)

                self.logger.info(f"表优化完成: {table_name}")

            except Exception as e:
                self.logger.error(f"表优化失败 {table_name}: {e}")

    async def cleanup_old_data(self, table_name: str, older_than_days: int = 3650):
        """清理旧数据（谨慎使用）"""
        async with self.pool.acquire() as conn:
            try:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)

                result = await conn.execute(f"""
                    DELETE FROM {table_name}
                    WHERE date < $1
                """, cutoff_date.date())

                deleted_count = int(result.split()[-1])  # 获取删除的行数
                self.logger.info(f"清理旧数据完成: {table_name}, 删除 {deleted_count} 条记录")

                return deleted_count

            except Exception as e:
                self.logger.error(f"清理旧数据失败 {table_name}: {e}")
                return 0

    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.close()
            self.logger.info("TimescaleDB连接池已关闭")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()