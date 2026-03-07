"""
PostgreSQL批量插入优化模块
实现高效的批量数据插入和事务管理
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from decimal import Decimal
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from .postgresql_persistence import get_db_connection, return_db_connection, ensure_table_exists

logger = logging.getLogger(__name__)

# 批量插入配置
BATCH_CONFIG = {
    'max_batch_size': 1000,  # 单批次最大记录数
    'max_workers': 4,        # 最大并发工作线程数
    'timeout_seconds': 300,  # 批量操作超时时间
    'retry_attempts': 3,     # 重试次数
    'retry_delay': 1.0,      # 重试延迟（秒）
}


class PostgreSQLBatchInserter:
    """PostgreSQL批量插入器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**BATCH_CONFIG, **(config or {})}
        self._connection_pool = []

    @contextmanager
    def _get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = get_db_connection()
        if not conn:
            raise Exception("无法获取数据库连接")

        try:
            # 设置连接参数
            conn.autocommit = False  # 使用事务
            yield conn
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            return_db_connection(conn)

    def insert_batch_stock_data(self, source_id: str, data: List[Dict[str, Any]],
                               metadata: Dict[str, Any], source_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        批量插入股票数据

        Args:
            source_id: 数据源ID
            data: 数据列表
            metadata: 元数据
            source_config: 数据源配置

        Returns:
            批量插入结果
        """
        start_time = time.time()

        try:
            # 确保表存在
            if not ensure_table_exists():
                raise Exception("无法确保数据库表存在")

            # 数据预处理和验证
            processed_data = self._preprocess_stock_data(data, source_id)
            if not processed_data:
                return {
                    "success": False,
                    "error": "没有有效的股票数据需要插入",
                    "processing_time": time.time() - start_time
                }

            # 分批处理
            total_inserted = 0
            total_updated = 0
            total_failed = 0

            batches = self._split_into_batches(processed_data, self.config['max_batch_size'])

            with self._get_connection() as conn:
                for batch in batches:
                    batch_result = self._insert_single_batch_stock(
                        conn, source_id, batch, metadata, source_config
                    )

                    total_inserted += batch_result.get('inserted', 0)
                    total_updated += batch_result.get('updated', 0)
                    total_failed += batch_result.get('failed', 0)

                # 提交所有批次
                conn.commit()

            processing_time = time.time() - start_time

            # 计算数据质量评分
            quality_score = self._calculate_data_quality_score(processed_data)

            # 记录性能统计
            self._record_performance_stats(source_id, {
                'total_records': len(processed_data),
                'inserted_records': total_inserted,
                'updated_records': total_updated,
                'failed_records': total_failed,
                'processing_time': processing_time,
                'data_quality_score': quality_score
            })

            logger.info(
                f"批量插入股票数据完成: {source_id}, "
                f"总记录: {len(processed_data)}, 插入: {total_inserted}, "
                f"更新: {total_updated}, 失败: {total_failed}, "
                f"质量评分: {quality_score:.1f}%, 耗时: {processing_time:.2f}秒"
            )

            return {
                "success": True,
                "storage_type": "postgresql_batch",
                "total_records": len(processed_data),
                "inserted_count": total_inserted,
                "updated_count": total_updated,
                "failed_count": total_failed,
                "processing_time": processing_time,
                "data_quality_score": quality_score,
                "batches_processed": len(batches),
                "message": f"批量插入完成，处理{len(processed_data)}条记录，"
                          f"插入{total_inserted}条，更新{total_updated}条，"
                          f"数据质量评分{quality_score:.1f}%"
            }

        except Exception as e:
            error_msg = f"批量插入股票数据异常: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return {
                "success": False,
                "error": error_msg,
                "processing_time": time.time() - start_time
            }

    def _preprocess_stock_data(self, data: List[Dict[str, Any]], source_id: str) -> List[Dict[str, Any]]:
        """预处理股票数据"""
        processed_data = []

        for record in data:
            try:
                # 数据验证和转换
                processed_record = self._validate_and_convert_stock_record(record, source_id)
                if processed_record:
                    processed_data.append(processed_record)
                else:
                    logger.debug(f"跳过无效记录: {record.get('symbol', 'unknown')}")

            except Exception as e:
                logger.warning(f"预处理记录失败: {record.get('symbol', 'unknown')}, 错误: {e}")
                continue

        return processed_data

    def _validate_and_convert_stock_record(self, record: Dict[str, Any], source_id: str) -> Optional[Dict[str, Any]]:
        """验证和转换股票数据记录"""
        try:
            # 必需字段检查
            symbol = record.get('symbol', '').strip()
            date_str = record.get('date', '').strip()

            if not symbol or not date_str:
                return None

            # 日期转换
            try:
                if isinstance(date_str, str):
                    if "T" in date_str:
                        date_obj = datetime.fromisoformat(date_str.split("T")[0]).date()
                    else:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                else:
                    date_obj = date_str

                # 验证日期合理性
                if date_obj > datetime.now().date() or date_obj < datetime(1990, 1, 1).date():
                    return None

            except Exception:
                return None

            # 价格数据转换和验证
            def safe_decimal(value, field_name: str) -> Optional[Decimal]:
                if value is None or value == '':
                    return None
                try:
                    decimal_value = Decimal(str(value))
                    if decimal_value <= 0:
                        logger.warning(f"无效的{field_name}值: {decimal_value} for {symbol}")
                        return None
                    return decimal_value
                except Exception:
                    return None

            open_price = safe_decimal(record.get('open'), 'open_price')
            high_price = safe_decimal(record.get('high'), 'high_price')
            low_price = safe_decimal(record.get('low'), 'low_price')
            close_price = safe_decimal(record.get('close'), 'close_price')

            # 价格逻辑验证
            if open_price and high_price and low_price and close_price:
                if not (low_price <= open_price <= high_price and
                        low_price <= close_price <= high_price):
                    logger.warning(f"价格逻辑错误 for {symbol}: open={open_price}, high={high_price}, low={low_price}, close={close_price}")
                    return None

            # 成交量和成交额
            def safe_int(value) -> Optional[int]:
                if value is None or value == '':
                    return None
                try:
                    int_value = int(float(value))
                    return int_value if int_value >= 0 else None
                except Exception:
                    return None

            volume = safe_int(record.get('volume'))
            amount = safe_decimal(record.get('amount'), 'amount')

            # 其他字段
            pct_change = safe_decimal(record.get('pct_change'), 'pct_change')
            change = safe_decimal(record.get('change'), 'change')
            turnover_rate = safe_decimal(record.get('turnover_rate'), 'turnover_rate')
            amplitude = safe_decimal(record.get('amplitude'), 'amplitude')

            data_type = record.get('data_type', 'daily')

            return {
                'source_id': source_id,
                'symbol': symbol,
                'date': date_obj,
                'data_type': data_type,
                'open_price': open_price,
                'high_price': high_price,
                'low_price': low_price,
                'close_price': close_price,
                'volume': volume,
                'amount': amount,
                'pct_change': pct_change,
                'change': change,
                'turnover_rate': turnover_rate,
                'amplitude': amplitude,
                'data_source': record.get('data_source', 'akshare'),
                'collected_at': datetime.fromtimestamp(record.get('collected_at', time.time())),
            }

        except Exception as e:
            logger.warning(f"验证记录失败: {record.get('symbol', 'unknown')}, 错误: {e}")
            return None

    def _split_into_batches(self, data: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """将数据分割成批次"""
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def _insert_single_batch_stock(self, conn, source_id: str, batch: List[Dict[str, Any]],
                                  metadata: Dict[str, Any], source_config: Dict[str, Any]) -> Dict[str, int]:
        """插入单个批次的股票数据"""
        try:
            cursor = conn.cursor()

            # 准备批量插入数据
            insert_query = """
                INSERT INTO akshare_stock_data (
                    source_id, symbol, date, data_type, open_price, high_price, low_price,
                    close_price, volume, amount, pct_change, change,
                    turnover_rate, amplitude, data_source, collected_at, persistence_timestamp
                ) VALUES %s
                ON CONFLICT (source_id, symbol, date, data_type)
                DO UPDATE SET
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume = EXCLUDED.volume,
                    amount = EXCLUDED.amount,
                    pct_change = EXCLUDED.pct_change,
                    change = EXCLUDED.change,
                    turnover_rate = EXCLUDED.turnover_rate,
                    amplitude = EXCLUDED.amplitude,
                    persistence_timestamp = CURRENT_TIMESTAMP
            """

            # 构建批量插入数据
            values = []
            persistence_timestamp = datetime.now()

            for record in batch:
                values.append((
                    record['source_id'],
                    record['symbol'],
                    record['date'],
                    record['data_type'],
                    record['open_price'],
                    record['high_price'],
                    record['low_price'],
                    record['close_price'],
                    record['volume'],
                    record['amount'],
                    record['pct_change'],
                    record['change'],
                    record['turnover_rate'],
                    record['amplitude'],
                    record['data_source'],
                    record['collected_at'],
                    persistence_timestamp
                ))

            # 执行批量插入
            psycopg2.extras.execute_values(cursor, insert_query, values)

            # 统计结果
            inserted_count = len(batch)  # 假设所有记录都被处理（插入或更新）

            cursor.close()

            return {
                'inserted': inserted_count,
                'updated': 0,  # 批量操作无法精确区分插入和更新
                'failed': 0
            }

        except Exception as e:
            logger.error(f"批量插入股票数据失败: {e}")
            return {
                'inserted': 0,
                'updated': 0,
                'failed': len(batch)
            }

    def _calculate_data_quality_score(self, data: List[Dict[str, Any]]) -> float:
        """计算数据质量评分"""
        if not data:
            return 0.0

        total_records = len(data)
        valid_records = 0

        for record in data:
            # 检查关键字段是否完整
            has_prices = all(record.get(field) is not None for field in
                           ['open_price', 'high_price', 'low_price', 'close_price'])
            has_volume = record.get('volume') is not None

            if has_prices and has_volume:
                valid_records += 1

        return (valid_records / total_records) * 100 if total_records > 0 else 0.0

    def _record_performance_stats(self, source_id: str, stats: Dict[str, Any]):
        """记录性能统计"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                insert_query = """
                    INSERT INTO data_collection_performance (
                        source_id, collection_date, total_records, inserted_records,
                        updated_records, failed_records, processing_time_seconds,
                        data_quality_score
                    ) VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_id, collection_date)
                    DO UPDATE SET
                        total_records = EXCLUDED.total_records,
                        inserted_records = EXCLUDED.inserted_records,
                        updated_records = EXCLUDED.updated_records,
                        failed_records = EXCLUDED.failed_records,
                        processing_time_seconds = EXCLUDED.processing_time_seconds,
                        data_quality_score = EXCLUDED.data_quality_score,
                        created_at = CURRENT_TIMESTAMP
                """

                cursor.execute(insert_query, (
                    source_id,
                    stats['total_records'],
                    stats['inserted_records'],
                    stats['updated_records'],
                    stats['failed_records'],
                    round(stats['processing_time'], 3),
                    round(stats['data_quality_score'], 2)
                ))

                conn.commit()
                cursor.close()

        except Exception as e:
            logger.warning(f"记录性能统计失败: {e}")


# 全局批量插入器实例
_batch_inserter = None


def get_batch_inserter() -> PostgreSQLBatchInserter:
    """获取批量插入器实例（单例模式）"""
    global _batch_inserter
    if _batch_inserter is None:
        _batch_inserter = PostgreSQLBatchInserter()
    return _batch_inserter


def persist_akshare_data_batch(
    source_id: str,
    data: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    source_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    批量持久化AKShare股票数据

    Args:
        source_id: 数据源ID
        data: 采集的数据列表
        metadata: 元数据
        source_config: 数据源配置

    Returns:
        批量持久化结果
    """
    batch_inserter = get_batch_inserter()
    return batch_inserter.insert_batch_stock_data(source_id, data, metadata, source_config)