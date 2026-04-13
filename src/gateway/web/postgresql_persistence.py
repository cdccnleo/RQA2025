"""
PostgreSQL数据持久化模块
支持AKShare股票数据存储到PostgreSQL + TimescaleDB
优化容器化环境支持
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal
import os
import traceback

# 延迟导入pandas,避免在模块级别导入失败
try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

# 全局数据库连接池(延迟初始化)
_db_pool = None
_db_config = None
_db_initialized = False

# 容器化环境检测
def is_containerized_env():
    """
    检测是否在容器化环境中运行

    Returns:
        bool: 是否在容器化环境中
    """
    # 检查常见的容器环境标志
    return any([
        os.path.exists('/.dockerenv'),
        os.environ.get('KUBERNETES_SERVICE_HOST'),
        os.environ.get('DOCKER_CONTAINER'),
        os.environ.get('CONTAINER_ENV')
    ])

def get_db_connection():
    """
    获取数据库连接

    Returns:
        数据库连接对象
    """
    global _db_pool, _db_config, _db_initialized

    # 尝试从环境变量获取数据库配置
    # 容器化环境中使用标准环境变量名称
    db_host = os.getenv('RQA_DB_HOST',
                      os.getenv('DB_HOST',
                                os.getenv('POSTGRES_HOST',
                                          'postgres' if is_containerized_env() else 'localhost')))
    db_port = os.getenv('RQA_DB_PORT',
                      os.getenv('DB_PORT',
                                os.getenv('POSTGRES_PORT', '5432')))
    db_name = os.getenv('RQA_DB_NAME',
                      os.getenv('DB_NAME',
                                os.getenv('POSTGRES_DB', 'rqa2025_prod')))
    db_user = os.getenv('RQA_DB_USER',
                      os.getenv('DB_USER',
                                os.getenv('POSTGRES_USER', 'rqa2025_admin')))
    db_password = os.getenv('RQA_DB_PASSWORD',
                          os.getenv('DB_PASSWORD',
                                    os.getenv('POSTGRES_PASSWORD', 'SecurePass123!')))

    # 连接池配置
    # 修复: 减小连接池大小以避免耗尽PostgreSQL max_connections(100)
    # 默认最小连接数:2,最大连接数:20
    pool_min_size = int(os.getenv('DB_POOL_MIN_SIZE', '2'))
    pool_max_size = int(os.getenv('DB_POOL_MAX_SIZE', '20'))
    pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))

    # 尝试导入psycopg2
    try:
        import psycopg2
        from psycopg2 import pool
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

        # 初始化数据库连接池
        if _db_pool is None:
            try:
                _db_config = {
                    'host': db_host,
                    'port': db_port,
                    'database': db_name,
                    'user': db_user,
                    'password': db_password,
                    'connect_timeout': pool_timeout,
                    'application_name': 'RQA2025_Quant_Strategy'
                }

                # 容器化环境中增加重试机制
                max_retries = 5
                retry_interval = 3

                for attempt in range(max_retries):
                    try:
                        # 修复: 使用ThreadedConnectionPool替代SimpleConnectionPool
                        # 原因: SimpleConnectionPool不支持多线程安全共享，会导致连接泄漏
                        _db_pool = psycopg2.pool.ThreadedConnectionPool(
                            pool_min_size, pool_max_size, **_db_config
                        )
                        logger.info(f"数据库连接池初始化成功: {db_host}:{db_port}/{db_name}")
                        logger.info(f"连接池配置: 最小={pool_min_size}, 最大={pool_max_size}, 超时={pool_timeout}秒")
                        logger.info(f"运行环境: {'容器化' if is_containerized_env() else '本地'}")
                        _db_initialized = True
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"数据库连接池初始化失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                            logger.info(f"{retry_interval}秒后重试...")
                            time.sleep(retry_interval)
                        else:
                            logger.error(f"数据库连接池初始化失败 (最大重试次数已达): {e}")
                            return None
            except Exception as e:
                logger.error(f"数据库连接池初始化失败: {e}")
                logger.debug(traceback.format_exc())
                return None

        # 从连接池获取连接
        if _db_pool:
            try:
                conn = _db_pool.getconn()
                # 设置自动提交模式
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                logger.debug("成功从连接池获取数据库连接")
                return conn
            except Exception as e:
                logger.warning(f"从连接池获取连接失败: {e}")
                # 如果从连接池获取连接失败,尝试重新初始化连接池
                try:
                    _db_pool = None
                    # 修复: 重启时也用ThreadedConnectionPool
                    _db_pool = psycopg2.pool.ThreadedConnectionPool(
                        pool_min_size, pool_max_size, **_db_config
                    )
                    conn = _db_pool.getconn()
                    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                    logger.debug("重新初始化连接池后成功获取连接")
                    return conn
                except Exception as reinit_e:
                    logger.error(f"重新初始化连接池失败: {reinit_e}")
                    logger.debug(traceback.format_exc())
                    return None

    except ImportError:
        logger.debug("psycopg2未安装,无法连接PostgreSQL")
    except Exception as e:
        logger.error(f"获取数据库连接失败: {e}")
        logger.debug(traceback.format_exc())

    return None

def return_db_connection(conn):
    """
    归还数据库连接

    Args:
        conn: 数据库连接对象
    """
    global _db_pool

    # 尝试将连接归还到连接池
    if conn and _db_pool:
        try:
            # 检查连接是否仍然有效
            if not conn.closed:
                _db_pool.putconn(conn)
                logger.debug("数据库连接已归还到连接池")
            else:
                logger.warning("连接已关闭,无法归还到连接池")
        except Exception as e:
            logger.warning(f"归还数据库连接失败: {e}")
    elif conn:
        try:
            if not conn.closed:
                conn.close()
                logger.debug("数据库连接已直接关闭")
        except Exception as e:
            logger.warning(f"关闭数据库连接失败: {e}")

def close_db_pool():
    """
    关闭数据库连接池
    """
    global _db_pool

    if _db_pool:
        try:
            _db_pool.closeall()
            logger.info("数据库连接池已关闭")
            _db_pool = None
        except Exception as e:
            logger.error(f"关闭数据库连接池失败: {e}")

def ensure_backtest_table_indexes() -> bool:
    """
    确保backtest_results表存在适当的索引

    Returns:
        bool: 是否成功创建索引
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return False

        cursor = conn.cursor()

        # 检查backtest_results表是否存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'backtest_results'
            );
        """)

        if not cursor.fetchone()[0]:
            logger.warning("backtest_results表不存在,跳过索引创建")
            cursor.close()
            return False

        # 创建索引
        indexes = [
            # 为strategy_id创建索引,用于按策略ID分组和查询
            "CREATE INDEX IF NOT EXISTS idx_backtest_strategy_id ON backtest_results(strategy_id);",
            # 为created_at创建索引,用于按时间排序
            "CREATE INDEX IF NOT EXISTS idx_backtest_created_at ON backtest_results(created_at);",
            # 为status创建索引,用于过滤状态
            "CREATE INDEX IF NOT EXISTS idx_backtest_status ON backtest_results(status);",
            # 为strategy_id和created_at创建复合索引,用于按策略ID分组并按时间排序
            "CREATE INDEX IF NOT EXISTS idx_backtest_strategy_created ON backtest_results(strategy_id, created_at);"
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logger.debug(f"执行索引创建语句: {index_sql}")
            except Exception as e:
                logger.warning(f"创建索引失败: {e}")

        conn.commit()
        cursor.close()
        logger.info("backtest_results表索引创建/验证完成")
        return True

    except Exception as e:
        logger.error(f"确保backtest_results表索引失败: {e}")
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn:
            return_db_connection(conn)

# 注册退出处理
try:
    import atexit
    atexit.register(close_db_pool)
except Exception:
    pass

# 导入数据库分区模块
try:
    from .database_partitioning import ensure_partitioning
    # 在模块初始化时实施分区策略
    if ensure_partitioning():
        logger.info("数据库分区策略实施成功")
    else:
        logger.info("数据库分区策略实施跳过或失败")
except Exception as e:
    logger.warning(f"实施数据库分区策略失败: {e}")

def get_stocks_by_industry(industry: str) -> List[Dict[str, Any]]:
    """
    根据行业获取股票列表

    Args:
        industry: 行业名称

    Returns:
        股票列表
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return []

        cursor = conn.cursor()

        cursor.execute("""
            SELECT symbol, name, ipo_date, industry, market,
                   total_share, float_share, pe, pb, roe
            FROM stock_basic_info
            WHERE industry = %s
            ORDER BY symbol
        """, (industry,))

        rows = cursor.fetchall()
        cursor.close()

        stocks = []
        for row in rows:
            stocks.append({
                "symbol": row[0],
                "name": row[1],
                "ipo_date": row[2].isoformat() if row[2] else None,
                "industry": row[3],
                "market": row[4],
                "total_share": row[5],
                "float_share": row[6],
                "pe": row[7],
                "pb": row[8],
                "roe": row[9]
            })

        return stocks

    except Exception as e:
        logger.error(f"根据行业获取股票列表失败: {e}")
        return []
    finally:
        if conn:
            return_db_connection(conn)

def ensure_table_exists() -> bool:
    """确保 akshare_stock_data 表存在,不存在则创建"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return False
        cursor = conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'akshare_stock_data'
            );
        """)
        if cursor.fetchone()[0]:
            cursor.close()
            return True
        logger.info("创建 akshare_stock_data 表...")
        cursor.execute("""
            CREATE TABLE akshare_stock_data (
                id BIGSERIAL PRIMARY KEY,
                source_id VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                data_type VARCHAR(20) NOT NULL DEFAULT 'daily',
                open_price DECIMAL(15, 6),
                high_price DECIMAL(15, 6),
                low_price DECIMAL(15, 6),
                close_price DECIMAL(15, 6),
                volume BIGINT,
                amount DECIMAL(20, 2),
                pct_change DECIMAL(10, 4),
                change DECIMAL(15, 6),
                turnover_rate DECIMAL(10, 4),
                amplitude DECIMAL(10, 4),
                data_source VARCHAR(50) DEFAULT 'akshare',
                collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_akshare_record UNIQUE(source_id, symbol, date, data_type)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_akshare_stock_symbol ON akshare_stock_data(symbol);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_akshare_stock_date ON akshare_stock_data(date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_akshare_stock_source ON akshare_stock_data(source_id);")
        conn.commit()
        cursor.close()
        logger.info("akshare_stock_data 表创建成功")
        return True
    except Exception as e:
        logger.error(f"确保表存在失败: {e}")
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn:
            return_db_connection(conn)


def _norm_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """标准化单条记录字段名(支持 AKShare 中英文列名)"""
    out = dict(record)
    cn_to_en = {
        "开盘价": "open", "收盘价": "close", "最高价": "high", "最低价": "low",
        "成交量": "volume", "成交额": "amount", "涨跌幅": "pct_change", "涨跌额": "change",
        "换手率": "turnover_rate", "振幅": "amplitude", "日期": "date", "代码": "symbol",
    }
    for cn, en in cn_to_en.items():
        if cn in out and (en not in out or out.get(en) is None):
            out[en] = out[cn]
    return out


def persist_akshare_data_to_postgresql(
    source_id: str,
    data: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    source_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    将 AKShare 股票数据持久化到 PostgreSQL。

    Args:
        source_id: 数据源 ID
        data: 采集的数据列表
        metadata: 元数据
        source_config: 数据源配置

    Returns:
        持久化结果字典,包含 success, inserted_count, error 等
    """
    start_time = time.time()
    conn = None
    try:
        if not ensure_table_exists():
            return {"success": False, "error": "无法确保数据库表存在", "processing_time": time.time() - start_time}
        conn = get_db_connection()
        if not conn:
            return {"success": False, "error": "无法获取数据库连接", "processing_time": time.time() - start_time}
        cursor = conn.cursor()
        inserted_count = 0
        updated_count = 0
        skipped_count = 0
        error_count = 0
        data_type_default = "daily"
        collected_at = datetime.fromtimestamp(metadata.get("collection_timestamp", time.time()))
        persistence_ts = datetime.now()
        insert_sql = """
            INSERT INTO akshare_stock_data (
                source_id, symbol, date, data_type, open_price, high_price, low_price,
                close_price, volume, amount, pct_change, change,
                turnover_rate, amplitude, data_source, collected_at, persistence_timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                persistence_timestamp = EXCLUDED.persistence_timestamp
        """
        for record in data or []:
            rec = _norm_record(record)
            date_str = rec.get("date") or rec.get("日期")
            if not date_str:
                skipped_count += 1
                continue
            try:
                if isinstance(date_str, str):
                    if "T" in date_str:
                        date_obj = datetime.fromisoformat(date_str.split("T")[0]).date()
                    else:
                        date_obj = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
                else:
                    date_obj = getattr(date_str, "date", lambda: date_str)() if hasattr(date_str, "date") else date_str
            except Exception:
                skipped_count += 1
                continue
            symbol = str(rec.get("symbol") or rec.get("代码") or "").strip()
            if not symbol:
                skipped_count += 1
                continue
            data_type = str(rec.get("data_type") or data_type_default)
            vol = rec.get("volume") or rec.get("成交量")
            volume = int(vol) if vol is not None and vol != "" else None
            try:
                cursor.execute(insert_sql, (
                    source_id,
                    symbol,
                    date_obj,
                    data_type,
                    rec.get("open"),
                    rec.get("high"),
                    rec.get("low"),
                    rec.get("close"),
                    volume,
                    rec.get("amount"),
                    rec.get("pct_change"),
                    rec.get("change"),
                    rec.get("turnover_rate"),
                    rec.get("amplitude"),
                    rec.get("data_source", "akshare"),
                    collected_at,
                    persistence_ts,
                ))
                inserted_count += 1
            except Exception as e:
                error_count += 1
                logger.debug(f"插入记录失败 {symbol} {date_obj}: {e}")
        conn.commit()
        cursor.close()
        elapsed = time.time() - start_time
        logger.info(
            f"PostgreSQL 持久化完成: {source_id}, 插入: {inserted_count}, 更新: {updated_count}, 跳过: {skipped_count}, 错误: {error_count}, 耗时: {elapsed:.2f}s"
        )
        return {
            "success": True,
            "storage_type": "postgresql",
            "inserted_count": inserted_count,
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "processing_time": elapsed,
        }
    except Exception as e:
        logger.error(f"PostgreSQL 持久化异常: {e}", exc_info=True)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time,
        }
    finally:
        if conn:
            return_db_connection(conn)


def query_latest_stock_data_from_postgresql(source_id: str, limit: int = 10, data_type: str = None) -> List[Dict[str, Any]]:
    """
    从PostgreSQL查询最新的股票数据样本

    Args:
        source_id: 数据源ID (API使用的source_id)
        limit: 返回记录数限制
        data_type: 数据类型过滤

    Returns:
        股票数据列表
    """
    # API source_id -> (table_name, stored_source_id)
    SOURCE_ID_MAP = {
        # A股
        'akshare_stock_a': ('akshare_stock_data', 'akshare_stock_a'),
        'baostock_stock_a': ('akshare_stock_data', 'baostock_stock_a'),
        # 港股
        'akshare_stock_hk': ('akshare_hk_stock_data', 'akshare_hk_historical'),
        # 指数
        'akshare_index': ('akshare_index_data', 'akshare_index_historical'),
        # 债券 (国债收益率曲线)
        'akshare_bond': ('akshare_bond_yield', 'akshare_bond_yield'),
        # 外汇 (BOC历史牌价)
        'akshare_forex': ('akshare_forex_history', 'akshare_forex_history'),
        # 宏观经济
        'akshare_macro': ('akshare_macro_data', 'akshare_macro_china'),
        'akshare_macro_china': ('akshare_macro_data', 'akshare_macro_china'),
        'akshare_macro_usa': ('akshare_macro_data', 'akshare_macro_usa'),
        # 新闻
        'akshare_news_js': ('akshare_news_data', 'akshare_news_js'),
        'akshare_news_eastmoney': ('akshare_news_data', 'akshare_news_eastmoney'),
        'akshare_news_all': ('akshare_news_data', None),
        # 大宗商品
        'akshare_commodity_gold': ('akshare_commodity_gold', 'akshare_gold'),
        'akshare_commodity_energy': ('akshare_commodity_energy', 'akshare_energy'),
        'akshare_commodity_crude': ('akshare_commodity_crude', 'akshare_crude'),
        'akshare_commodity_natural_gas': ('akshare_commodity_natural_gas', 'akshare_natural_gas'),
        'akshare_commodity': ('akshare_commodity_gold', 'akshare_gold'),  # 默认黄金  # None = all news
        # 新增数据源
        'akshare_vix': ('akshare_vix_index', 'akshare_vix'),
        'akshare_margin': ('akshare_margin_summary', 'akshare_margin'),
        'akshare_commodity_futures': ('akshare_commodity_futures', 'akshare_futures'),
        'akshare_hsgt': ('akshare_hsgt_hist', 'akshare_hsgt'),
        'akshare_bond_futures': ('akshare_bond_futures', 'akshare_bond_futures'),
    }

    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.warning("无法获取数据库连接,无法查询股票数据")
            return []

        cursor = conn.cursor()

        # 获取表名和存储的source_id
        table_info = SOURCE_ID_MAP.get(source_id)
        if not table_info:
            logger.info(f"数据源 {source_id} 未映射到任何表,返回空数据")
            cursor.close()
            return []

        table_name, stored_source_id = table_info

        # 检查表是否存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            );
        """, (table_name,))

        if not cursor.fetchone()[0]:
            logger.info(f"表 {table_name} 不存在,返回空数据")
            cursor.close()
            return []

        # 根据不同的表使用不同的查询
        if table_name == 'akshare_stock_data':
            query = """
                SELECT 
                    symbol, date, open_price, high_price, low_price, close_price,
                    volume, amount, pct_change, change, turnover_rate, amplitude
                FROM akshare_stock_data 
                WHERE source_id = %s
            """
            cols = ['symbol', 'date', 'open_price', 'high_price', 'low_price', 'close_price',
                    'volume', 'amount', 'pct_change', 'change', 'turnover_rate', 'amplitude']
            # 添加数据类型过滤
            if data_type:
                query += f" AND data_type = '{data_type}'"
            query += " ORDER BY date DESC LIMIT %s"
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                record = {}
                for i, col in enumerate(cols):
                    val = row[i]
                    if hasattr(val, 'isoformat'):
                        val = val.isoformat()
                    record[col] = val
                result.append(record)
            return result
        
        elif table_name == 'akshare_hk_stock_data':
            query = """
                SELECT
                    symbol, name, date, open_price, high_price, low_price, close_price,
                    change_ratio, volume, amount, turnover_rate, pe_ratio, pb_ratio, market_cap
                FROM akshare_hk_stock_data
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'symbol': row[0], 'name': row[1], 'date': row[2].isoformat() if row[2] else None,
                    'open_price': row[3], 'high_price': row[4], 'low_price': row[5], 'close_price': row[6],
                    'change_ratio': row[7], 'volume': row[8], 'amount': row[9],
                    'turnover_rate': row[10], 'pe_ratio': row[11], 'pb_ratio': row[12], 'market_cap': row[13]
                })
            return result

        elif table_name == 'akshare_index_data':
            query = """
                SELECT
                    index_code, index_name, date, open_price, high_price, low_price, close_price,
                    change_ratio, volume, amount
                FROM akshare_index_data
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'index_code': row[0], 'index_name': row[1], 'date': row[2].isoformat() if row[2] else None,
                    'open_price': row[3], 'high_price': row[4], 'low_price': row[5], 'close_price': row[6],
                    'change_ratio': row[7], 'volume': row[8], 'amount': row[9]
                })
            return result

        elif table_name == 'akshare_forex_history':
            query = """
                SELECT
                    currency_pair, base_currency, quote_currency, date, price, change_ratio
                FROM akshare_forex_history
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'currency_pair': row[0], 'base_currency': row[1], 'quote_currency': row[2],
                    'date': row[3].isoformat() if row[3] else None,
                    'price': float(row[4]) if row[4] is not None else None,
                    'change_ratio': float(row[5]) if row[5] is not None else None,
                })
            return result

        elif table_name == 'akshare_bond_yield':
            query = """
                SELECT
                    curve_name, curve_type, date,
                    yield_3m, yield_6m, yield_1y,
                    yield_3y, yield_5y, yield_7y, yield_10y, yield_30y
                FROM akshare_bond_yield
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'curve_name': row[0], 'curve_type': row[1],
                    'date': row[2].isoformat() if row[2] else None,
                    'yield_3m': float(row[3]) if row[3] is not None else None,
                    'yield_6m': float(row[4]) if row[4] is not None else None,
                    'yield_1y': float(row[5]) if row[5] is not None else None,
                    'yield_3y': float(row[6]) if row[6] is not None else None,
                    'yield_5y': float(row[7]) if row[7] is not None else None,
                    'yield_7y': float(row[8]) if row[8] is not None else None,
                    'yield_10y': float(row[9]) if row[9] is not None else None,
                    'yield_30y': float(row[10]) if row[10] is not None else None,
                })
            return result

        elif table_name == 'akshare_macro_data':
            query = """
                SELECT
                    indicator_type, indicator_name, period, date,
                    value, forecast_value, previous_value, country
                FROM akshare_macro_data
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'indicator_type': row[0], 'indicator_name': row[1], 'period': row[2],
                    'date': row[3].isoformat() if row[3] else None,
                    'value': float(row[4]) if row[4] is not None else None,
                    'forecast_value': float(row[5]) if row[5] is not None else None,
                    'previous_value': float(row[6]) if row[6] is not None else None,
                    'country': row[7]
                })
            return result

        elif table_name == 'akshare_news_data':
            if stored_source_id:
                query = """
                    SELECT title, content, publish_date, url, keywords, news_source, category
                    FROM akshare_news_data
                    WHERE source_id = %s
                    ORDER BY publish_date DESC LIMIT %s
                """
                cursor.execute(query, (stored_source_id, limit))
            else:
                # akshare_news_all: 返回所有新闻
                query = """
                    SELECT title, content, publish_date, url, keywords, news_source, category
                    FROM akshare_news_data
                    ORDER BY publish_date DESC LIMIT %s
                """
                cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'title': row[0], 'content': row[1],
                    'publish_date': row[2].isoformat() if row[2] else None,
                    'url': row[3], 'keywords': row[4],
                    'news_source': row[5], 'category': row[6]
                })
            return result

        elif table_name == 'akshare_commodity_gold':
            query = """
                SELECT date, morning_price, evening_price, price_diff, commodity_type
                FROM akshare_commodity_gold
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'morning_price': float(row[1]) if row[1] is not None else None,
                    'evening_price': float(row[2]) if row[2] is not None else None,
                    'price_diff': float(row[3]) if row[3] is not None else None,
                    'commodity_type': row[4],
                })
            return result

        elif table_name == 'akshare_commodity_energy':
            query = """
                SELECT date, commodity_type, price, change_amount
                FROM akshare_commodity_energy
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'commodity_type': row[1],
                    'price': float(row[2]) if row[2] is not None else None,
                    'change_amount': float(row[3]) if row[3] is not None else None,
                })
            return result

        elif table_name == 'akshare_commodity_crude':
            query = """
                SELECT date, indicator_type, value, change_value, forecast_value, previous_value, unit, region
                FROM akshare_commodity_crude
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'indicator_type': row[1],
                    'value': float(row[2]) if row[2] is not None else None,
                    'change_value': float(row[3]) if row[3] is not None else None,
                    'forecast_value': float(row[4]) if row[4] is not None else None,
                    'previous_value': float(row[5]) if row[5] is not None else None,
                    'unit': row[6],
                    'region': row[7],
                })
            return result

        elif table_name == 'akshare_commodity_natural_gas':
            query = """
                SELECT date, indicator_type, value, change_value, unit, region
                FROM akshare_commodity_natural_gas
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'indicator_type': row[1],
                    'value': float(row[2]) if row[2] is not None else None,
                    'change_value': float(row[3]) if row[3] is not None else None,
                    'unit': row[4],
                    'region': row[5],
                })
            return result

        elif table_name == 'akshare_vix_index':
            query = """
                SELECT date, open, high, low, close
                FROM akshare_vix_index
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'open': float(row[1]) if row[1] is not None else None,
                    'high': float(row[2]) if row[2] is not None else None,
                    'low': float(row[3]) if row[3] is not None else None,
                    'close': float(row[4]) if row[4] is not None else None,
                })
            return result

        elif table_name == 'akshare_margin_summary':
            query = """
                SELECT date, exchange, rz_balance, rz_buy_amt, rq_shares, rq_balance, total_margin
                FROM akshare_margin_summary
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'exchange': row[1],
                    'rz_balance': float(row[2]) if row[2] is not None else None,
                    'rz_buy_amt': float(row[3]) if row[3] is not None else None,
                    'rq_shares': float(row[4]) if row[4] is not None else None,
                    'rq_balance': float(row[5]) if row[5] is not None else None,
                    'total_margin': float(row[6]) if row[6] is not None else None,
                })
            return result

        elif table_name == 'akshare_margin_detail':
            query = """
                SELECT date, exchange, stock_code, stock_name, rz_buy_amt, rz_balance, rq_shares
                FROM akshare_margin_detail
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'exchange': row[1],
                    'stock_code': row[2],
                    'stock_name': row[3],
                    'rz_buy_amt': float(row[4]) if row[4] is not None else None,
                    'rz_balance': float(row[5]) if row[5] is not None else None,
                    'rq_shares': float(row[6]) if row[6] is not None else None,
                })
            return result

        elif table_name == 'akshare_commodity_futures':
            query = """
                SELECT date, commodity_type, symbol, open_price, high_price, low_price, close_price, settle_price, volume, hold
                FROM akshare_commodity_futures
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'commodity_type': row[1],
                    'symbol': row[2],
                    'open': float(row[3]) if row[3] is not None else None,
                    'high': float(row[4]) if row[4] is not None else None,
                    'low': float(row[5]) if row[5] is not None else None,
                    'close': float(row[6]) if row[6] is not None else None,
                    'settle': float(row[7]) if row[7] is not None else None,
                    'volume': float(row[8]) if row[8] is not None else None,
                    'hold': float(row[9]) if row[9] is not None else None,
                })
            return result

        elif table_name == 'akshare_hsgt_hist':
            query = """
                SELECT date, net_buy_amt, buy_amt, sell_amt, cum_net_buy, inflow_amt, balance, hs300, hs300_pct, top_stock_name, top_stock_pct
                FROM akshare_hsgt_hist
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'net_buy_amt': float(row[1]) if row[1] is not None else None,
                    'buy_amt': float(row[2]) if row[2] is not None else None,
                    'sell_amt': float(row[3]) if row[3] is not None else None,
                    'cum_net_buy': float(row[4]) if row[4] is not None else None,
                    'inflow_amt': float(row[5]) if row[5] is not None else None,
                    'balance': float(row[6]) if row[6] is not None else None,
                    'hs300': float(row[7]) if row[7] is not None else None,
                    'hs300_pct': float(row[8]) if row[8] is not None else None,
                    'top_stock_name': row[9],
                    'top_stock_pct': float(row[10]) if row[10] is not None else None,
                })
            return result

        elif table_name == 'akshare_bond_futures':
            query = """
                SELECT date, bond_type, symbol, open_price, high_price, low_price, close_price, settle_price, volume, hold
                FROM akshare_bond_futures
                WHERE source_id = %s
                ORDER BY date DESC LIMIT %s
            """
            cursor.execute(query, (stored_source_id, limit))
            rows = cursor.fetchall()
            cursor.close()
            result = []
            for row in rows:
                result.append({
                    'date': row[0].isoformat() if row[0] else None,
                    'bond_type': row[1],
                    'symbol': row[2],
                    'open': float(row[3]) if row[3] is not None else None,
                    'high': float(row[4]) if row[4] is not None else None,
                    'low': float(row[5]) if row[5] is not None else None,
                    'close': float(row[6]) if row[6] is not None else None,
                    'settle': float(row[7]) if row[7] is not None else None,
                    'volume': float(row[8]) if row[8] is not None else None,
                    'hold': float(row[9]) if row[9] is not None else None,
                })
            return result

        else:
            # 默认查询 (akshare_stock_data)
            query = """
                SELECT
                    symbol, date, open_price, high_price, low_price, close_price,
                    volume, amount, pct_change, change, turnover_rate, amplitude
                FROM akshare_stock_data
                WHERE source_id = %s
            """
            cols = ['symbol', 'date', 'open_price', 'high_price', 'low_price', 'close_price',
                    'volume', 'amount', 'pct_change', 'change', 'turnover_rate', 'amplitude']

        # 添加数据类型过滤和排序限制
        if data_type:
            query += f" AND data_type = '{data_type}'"
        query += " ORDER BY date DESC LIMIT %s"

        # 执行查询
        cursor.execute(query, (stored_source_id, limit))
        rows = cursor.fetchall()
        cursor.close()

        # 处理查询结果
        result = []
        for row in rows:
            record = {}
            for i, col in enumerate(cols):
                val = row[i]
                if hasattr(val, 'isoformat'):
                    val = val.isoformat()
                record[col] = val
            result.append(record)

        return result

    except Exception as e:
        logger.error(f"查询最新股票数据失败: {e}")
        return []
    finally:
        if conn:
            return_db_connection(conn)


def query_stock_data_from_postgresql(
    source_id: str,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime
) -> Dict[str, Any]:
    """
    从PostgreSQL查询股票数据

    Args:
        source_id: 数据源ID
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        字典,key为股票代码,value为该股票的DataFrame
    """
    if pd is None:
        logger.error("pandas未安装,无法查询股票数据")
        return {}

    conn = None
    result = {}

    try:
        conn = get_db_connection()
        if not conn:
            logger.warning("无法获取数据库连接,无法查询股票数据")
            return {}

        cursor = conn.cursor()

        for symbol in symbols:
            try:
                # 查询该股票在指定日期范围内的数据
                cursor.execute("""
                    SELECT
                        date,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        volume,
                        amount,
                        pct_change,
                        change,
                        turnover_rate,
                        amplitude
                    FROM akshare_stock_data
                    WHERE source_id = %s
                      AND symbol = %s
                      AND date >= %s
                      AND date <= %s
                    ORDER BY date ASC
                """, (source_id, symbol, start_date.date(), end_date.date()))

                rows = cursor.fetchall()

                if rows:
                    # 转换为DataFrame
                    df = pd.DataFrame(rows, columns=[
                        'date', 'open_price', 'high_price', 'low_price',
                        'close_price', 'volume', 'amount', 'pct_change',
                        'change', 'turnover_rate', 'amplitude'
                    ])

                    # 设置日期为索引
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)

                    # 重命名列为标准OHLCV格式
                    df.rename(columns={
                        'open_price': 'open',
                        'high_price': 'high',
                        'low_price': 'low',
                        'close_price': 'close',
                        'volume': 'volume'
                    }, inplace=True)

                    # 添加timestamp列(用于兼容性)
                    df['timestamp'] = df.index

                    result[symbol] = df
                    logger.debug(f"从数据库加载股票 {symbol} 数据: {len(df)} 条记录")
                else:
                    logger.debug(f"股票 {symbol} 在指定日期范围内无数据")
                    result[symbol] = pd.DataFrame()

            except Exception as e:
                logger.warning(f"查询股票 {symbol} 数据失败: {e}")
                result[symbol] = pd.DataFrame()

        cursor.close()
        return result

    except Exception as e:
        logger.error(f"查询股票数据失败: {e}")
        return {}
    finally:
        if conn:
            return_db_connection(conn)
