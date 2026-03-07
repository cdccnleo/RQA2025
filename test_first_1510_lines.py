"""
PostgreSQL数据持久化模块
支持AKShare股票数据存储到PostgreSQL + TimescaleDB
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal
import os

# 延迟导入pandas，避免在模块级别导入失败
try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

# 全局数据库连接池（延迟初始化）
_db_pool = None
_db_config = None


def get_db_config() -> Dict[str, Any]:
    """获取数据库配置"""
    global _db_config
    
    if _db_config is None:
        # 优先从DATABASE_URL解析
        database_url = os.getenv("DATABASE_URL", "")
        # #region agent log
        import json as json_module
        import time
        try:
            debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run4","hypothesisId":"D","location":"postgresql_persistence.py:get_db_config:31","message":"检查DATABASE_URL环境变量","data":{"database_url_exists":bool(database_url),"database_url_length":len(database_url) if database_url else 0},"timestamp":int(time.time()*1000)})+'\n')
        except: pass
        # #endregion
        if database_url:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(database_url)
                _db_config = {
                    "host": parsed.hostname or "localhost",
                    "port": parsed.port or 5432,
                    "database": parsed.path.lstrip('/') or "rqa2025",
                    "user": parsed.username or "rqa2025",
                    "password": parsed.password or "rqa2025pass",
                }
                logger.info(f"从DATABASE_URL解析数据库配置: {_db_config['host']}:{_db_config['port']}/{_db_config['database']}")
                # #region agent log
                try:
                    debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json_module.dumps({"sessionId":"debug-session","runId":"run4","hypothesisId":"D","location":"postgresql_persistence.py:get_db_config:43","message":"DATABASE_URL解析成功","data":{"host":_db_config["host"],"port":_db_config["port"],"database":_db_config["database"],"user":_db_config["user"],"has_password":bool(_db_config["password"])},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
            except Exception as e:
                logger.warning(f"解析DATABASE_URL失败: {e}")
                # #region agent log
                try:
                    debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json_module.dumps({"sessionId":"debug-session","runId":"run4","hypothesisId":"D","location":"postgresql_persistence.py:get_db_config:45","message":"DATABASE_URL解析失败","data":{"error":str(e)},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                _db_config = None
        
        # 如果DATABASE_URL解析失败，尝试从环境变量获取
        if _db_config is None:
            try:
                from src.infrastructure.utils.components.environment import get_database_config
                config = get_database_config()
                _db_config = {
                    "host": config.get("host", os.getenv("DB_HOST", "localhost")),  # Windows环境默认使用localhost
                    "port": config.get("port", int(os.getenv("DB_PORT", "5432"))),
                    "database": config.get("name", os.getenv("DB_NAME", "rqa2025")),
                    "user": config.get("user", os.getenv("DB_USER", "rqa2025")),
                    "password": config.get("password", os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "rqa2025pass"))),
                }
            except Exception as e:
                logger.warning(f"无法从环境获取数据库配置，使用默认配置: {e}")
                _db_config = {
                    "host": os.getenv("DB_HOST", "localhost"),  # Windows环境默认使用localhost
                    "port": int(os.getenv("DB_PORT", "5432")),
                    "database": os.getenv("DB_NAME", "rqa2025"),
                    "user": os.getenv("DB_USER", "rqa2025"),
                    "password": os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "rqa2025pass")),
                }
        
        # 验证密码是否为空（Windows环境下GSSAPI失败时，如果密码为空会导致"no password supplied"错误）
        if not _db_config.get("password"):
            logger.warning("PostgreSQL密码未配置，连接可能失败。请设置DB_PASSWORD或POSTGRES_PASSWORD环境变量")
    
    return _db_config


def get_db_connection():
    """获取数据库连接"""
    global _db_pool
    
    # #region agent log
    import json as json_module
    try:
        debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
        with open(debug_log_path, 'a', encoding='utf-8') as f:
            f.write(json_module.dumps({"sessionId":"debug-session","runId":"run3","hypothesisId":"C","location":"postgresql_persistence.py:get_db_connection:77","message":"get_db_connection被调用","data":{},"timestamp":int(time.time()*1000)})+'\n')
    except: pass
    # #endregion
    
    try:
        import psycopg2
        from psycopg2 import pool
        
        if _db_pool is None:
            config = get_db_config()
            
            # #region agent log
            try:
                debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(json_module.dumps({"sessionId":"debug-session","runId":"run3","hypothesisId":"C","location":"postgresql_persistence.py:get_db_connection:90","message":"数据库配置获取","data":{"host":config.get("host") if config else None,"port":config.get("port") if config else None,"database":config.get("database") if config else None,"user":config.get("user") if config else None,"has_password":bool(config.get("password")) if config else False},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            
            # 构建连接参数字典，添加Windows环境下的兼容性参数
            connection_params = {
                "host": config["host"],
                "port": config["port"],
                "database": config["database"],
                "user": config["user"],
                "password": config["password"],
                "connect_timeout": 10
            }
            
            # Windows环境下禁用GSSAPI认证，强制使用密码认证
            # 这可以解决 "could not initiate GSSAPI security context" 错误
            import platform
            if platform.system() == "Windows":
                # 在Windows上，禁用GSSAPI，使用密码认证
                # 通过设置sslmode和gssencmode参数来避免GSSAPI问题
                connection_params["sslmode"] = "prefer"  # 优先使用SSL，但不强制
                # 注意：psycopg2不支持直接设置gssencmode，但可以通过连接字符串设置
                # 如果仍有问题，可以尝试使用连接字符串而不是参数字典
            
            # 创建连接池
            try:
                # #region agent log
                try:
                    debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json_module.dumps({"sessionId":"debug-session","runId":"run4","hypothesisId":"D","location":"postgresql_persistence.py:get_db_connection:125","message":"尝试创建连接池","data":{"host":config["host"],"port":config["port"],"database":config["database"],"user":config["user"]},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                _db_pool = psycopg2.pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=10,
                    **connection_params
                )
                logger.info(f"PostgreSQL连接池创建成功: {config['host']}:{config['port']}/{config['database']}")
                # #region agent log
                try:
                    debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json_module.dumps({"sessionId":"debug-session","runId":"run4","hypothesisId":"D","location":"postgresql_persistence.py:get_db_connection:131","message":"连接池创建成功","data":{},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
            except psycopg2.OperationalError as e:
                # #region agent log
                try:
                    debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json_module.dumps({"sessionId":"debug-session","runId":"run4","hypothesisId":"D","location":"postgresql_persistence.py:get_db_connection:132","message":"连接池创建失败（OperationalError）","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                # 如果使用参数字典失败（可能因为GSSAPI或sslmode参数），尝试使用连接字符串
                if "sslmode" in connection_params or "GSSAPI" in str(e) or "gssapi" in str(e).lower():
                    logger.debug(f"使用参数字典连接失败，尝试使用连接字符串（禁用GSSAPI）: {e}")
                    # 构建连接字符串，明确禁用GSSAPI
                    # 注意：gssencmode参数需要psycopg2 2.9+，如果版本不支持，可以尝试其他方法
                    conn_string = (
                        f"host={config['host']} "
                        f"port={config['port']} "
                        f"dbname={config['database']} "
                        f"user={config['user']} "
                        f"password={config['password']} "
                        f"connect_timeout=10 "
                        f"sslmode=prefer"
                    )
                    # 尝试添加gssencmode参数（如果psycopg2版本支持）
                    try:
                        # 检查psycopg2版本
                        import psycopg2
                        psycopg2_version = tuple(map(int, psycopg2.__version__.split('.')[:2]))
                        if psycopg2_version >= (2, 9):
                            conn_string += " gssencmode=disable"  # 禁用GSSAPI加密
                    except:
                        pass  # 如果无法获取版本，继续使用基本连接字符串
                    
                    # 使用连接字符串创建连接池
                    try:
                        # psycopg2.pool.SimpleConnectionPool 不支持 dsn 参数，需要使用连接参数字典
                        # 从连接字符串解析参数，构建参数字典
                        conn_params_dict = {}
                        for part in conn_string.split():
                            if '=' in part:
                                key, value = part.split('=', 1)
                                # 转换参数名（psycopg2使用不同的参数名）
                                if key == 'dbname':
                                    conn_params_dict['database'] = value
                                elif key == 'host':
                                    conn_params_dict['host'] = value
                                elif key == 'port':
                                    conn_params_dict['port'] = int(value)
                                elif key == 'user':
                                    conn_params_dict['user'] = value
                                elif key == 'password':
                                    conn_params_dict['password'] = value
                                elif key == 'connect_timeout':
                                    conn_params_dict['connect_timeout'] = int(value)
                                # sslmode等参数在连接字符串中设置，但不直接传递给SimpleConnectionPool
                        
                        _db_pool = psycopg2.pool.SimpleConnectionPool(
                            minconn=1,
                            maxconn=10,
                            **conn_params_dict
                        )
                        logger.info(f"PostgreSQL连接池创建成功（使用连接字符串，已禁用GSSAPI）: {config['host']}:{config['port']}/{config['database']}")
                    except psycopg2.OperationalError as e2:
                        # 如果仍然失败，记录详细错误但不抛出异常（允许回退到文件系统）
                        logger.warning(f"PostgreSQL连接失败（将使用文件系统存储）: {e2}")
                        logger.debug(f"连接字符串: {conn_string.replace(config.get('password', ''), '***')}")
                        # #region agent log
                        try:
                            debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                            with open(debug_log_path, 'a', encoding='utf-8') as f:
                                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run4","hypothesisId":"D","location":"postgresql_persistence.py:get_db_connection:186","message":"连接字符串方式也失败","data":{"error":str(e2),"error_type":type(e2).__name__},"timestamp":int(time.time()*1000)})+'\n')
                        except: pass
                        # #endregion
                        _db_pool = None
                else:
                    # 其他类型的错误，记录但不创建连接池
                    logger.warning(f"PostgreSQL连接失败（将使用文件系统存储）: {e}")
                    # #region agent log
                    try:
                        debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                        with open(debug_log_path, 'a', encoding='utf-8') as f:
                            f.write(json_module.dumps({"sessionId":"debug-session","runId":"run4","hypothesisId":"D","location":"postgresql_persistence.py:get_db_connection:192","message":"其他类型错误","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(time.time()*1000)})+'\n')
                    except: pass
                    # #endregion
                    _db_pool = None
        
        if _db_pool is None:
            # #region agent log
            try:
                debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(json_module.dumps({"sessionId":"debug-session","runId":"run4","hypothesisId":"D","location":"postgresql_persistence.py:get_db_connection:196","message":"连接池为None，返回None","data":{},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            return None
        
        return _db_pool.getconn()
        
    except ImportError:
        logger.error("psycopg2未安装，无法使用PostgreSQL持久化")
        return None
    except Exception as e:
        logger.error(f"获取数据库连接失败: {e}")
        return None


def return_db_connection(conn):
    """归还数据库连接到连接池"""
    global _db_pool
    
    if _db_pool and conn:
        try:
            _db_pool.putconn(conn)
        except Exception as e:
            logger.error(f"归还数据库连接失败: {e}")


def ensure_table_exists():
    """确保表存在，如果不存在则创建"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'akshare_stock_data'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.info("创建akshare_stock_data表...")

            # 创建表
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

            # 创建索引
            cursor.execute("CREATE INDEX idx_akshare_symbol_date ON akshare_stock_data(symbol, date DESC);")
            cursor.execute("CREATE INDEX idx_akshare_source_collected ON akshare_stock_data(source_id, collected_at DESC);")
            cursor.execute("CREATE INDEX idx_akshare_date_range ON akshare_stock_data(date DESC);")

            conn.commit()
            logger.info("akshare_stock_data表创建成功")
        else:
            logger.debug("akshare_stock_data表已存在，检查表结构是否需要更新...")

            # 检查并添加缺失的字段（表结构迁移）
            try:
                # 检查data_type字段是否存在
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns
                        WHERE table_schema = 'public'
                        AND table_name = 'akshare_stock_data'
                        AND column_name = 'data_type'
                    );
                """)
                data_type_exists = cursor.fetchone()[0]

                if not data_type_exists:
                    logger.info("添加缺失的data_type字段...")
                    cursor.execute("""
                        ALTER TABLE akshare_stock_data
                        ADD COLUMN data_type VARCHAR(20) NOT NULL DEFAULT 'daily';
                    """)
                    logger.info("data_type字段添加成功")

                # 检查唯一约束是否包含data_type
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                        WHERE tc.table_schema = 'public'
                        AND tc.table_name = 'akshare_stock_data'
                        AND tc.constraint_type = 'UNIQUE'
                        AND kcu.column_name = 'data_type'
                    );
                """)
                constraint_has_data_type = cursor.fetchone()[0]

                if not constraint_has_data_type:
                    logger.info("更新唯一约束以包含data_type字段...")

                    # 先删除旧的唯一约束（如果存在）
                    try:
                        cursor.execute("""
                            ALTER TABLE akshare_stock_data
                            DROP CONSTRAINT IF EXISTS unique_akshare_record;
                        """)
                    except Exception as e:
                        logger.debug(f"删除旧约束可能失败（正常）: {e}")

                    # 添加新的唯一约束
                    cursor.execute("""
                        ALTER TABLE akshare_stock_data
                        ADD CONSTRAINT unique_akshare_record UNIQUE(source_id, symbol, date, data_type);
                    """)
                    logger.info("唯一约束更新成功")

                conn.commit()
                logger.info("表结构迁移完成")

            except Exception as migration_error:
                logger.error(f"表结构迁移失败: {migration_error}")
                # 不抛出异常，继续运行，以免阻塞数据采集
        
        # 创建股票基本信息表
        ensure_stock_basic_info_table(cursor)
        
        cursor.close()
        return True
        
    except Exception as e:
        logger.error(f"确保表存在失败: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            return_db_connection(conn)


def ensure_stock_basic_info_table(cursor):
    """确保股票基本信息表存在，如果不存在则创建"""
    try:
        # 检查表是否存在
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'stock_basic_info'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.info("创建stock_basic_info表...")

            # 创建表
            cursor.execute("""
                CREATE TABLE stock_basic_info (
                    id BIGSERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL UNIQUE,
                    name VARCHAR(100) NOT NULL,
                    ipo_date DATE,
                    industry VARCHAR(100),
                    market VARCHAR(20),
                    total_share BIGINT,
                    float_share BIGINT,
                    pe DECIMAL(10, 4),
                    pb DECIMAL(10, 4),
                    roe DECIMAL(10, 4),
                    update_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                );
            """)

            # 创建索引
            cursor.execute("CREATE INDEX idx_stock_basic_industry ON stock_basic_info(industry);")
            cursor.execute("CREATE INDEX idx_stock_basic_ipo_date ON stock_basic_info(ipo_date);")
            cursor.execute("CREATE INDEX idx_stock_basic_market ON stock_basic_info(market);")

            logger.info("stock_basic_info表创建成功")
        else:
            logger.debug("stock_basic_info表已存在，跳过创建")
            
    except Exception as e:
        logger.error(f"确保股票基本信息表存在失败: {e}")
        # 不抛出异常，继续运行，以免阻塞数据采集


def query_existing_dates(
    source_id: str,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    data_type: str = "daily"
) -> Dict[str, set]:
    """
    查询数据库中已存在的数据日期

    Args:
        source_id: 数据源ID
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        data_type: 数据类型，默认'daily'

    Returns:
        字典，key为股票代码，value为该股票已存在的日期集合
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.warning("无法获取数据库连接，无法查询已存在日期")
            return {}
        
        cursor = conn.cursor()
        
        # 查询每个股票已存在的日期
        existing_dates = {}
        
        for symbol in symbols:
            try:
                cursor.execute("""
                    SELECT DISTINCT date
                    FROM akshare_stock_data
                    WHERE source_id = %s
                      AND symbol = %s
                      AND data_type = %s
                      AND date >= %s
                      AND date <= %s
                    ORDER BY date
                """, (source_id, symbol, data_type, start_date.date(), end_date.date()))
                
                dates = {row[0] for row in cursor.fetchall()}
                existing_dates[symbol] = dates
                
                if dates:
                    logger.debug(f"股票 {symbol} 已存在 {len(dates)} 个日期的数据")
                    
            except Exception as e:
                logger.warning(f"查询股票 {symbol} 已存在日期失败: {e}")
                existing_dates[symbol] = set()
        
        cursor.close()
        return existing_dates
        
    except Exception as e:
        logger.error(f"查询已存在日期失败: {e}")
        return {}
    finally:
        if conn:
            return_db_connection(conn)


def calculate_missing_date_ranges(
    start_date: datetime,
    end_date: datetime,
    existing_dates: set
) -> List[tuple]:
    """
    计算缺失的日期范围
    
    Args:
        start_date: 请求的开始日期
        end_date: 请求的结束日期
        existing_dates: 已存在的日期集合
        
    Returns:
        缺失日期范围的列表，每个元素为 (start_date, end_date) 元组
    """
    from datetime import timedelta
    
    if not existing_dates:
        # 如果没有任何已存在的数据，返回整个范围
        return [(start_date, end_date)]
    
    # 将日期转换为date对象以便比较
    start = start_date.date() if isinstance(start_date, datetime) else start_date
    end = end_date.date() if isinstance(end_date, datetime) else end_date
    
    missing_ranges = []
    current_start = start
    
    # 按日期排序已存在的日期
    sorted_dates = sorted(existing_dates)
    
    for existing_date in sorted_dates:
        # 如果当前开始日期小于已存在日期，说明有缺失
        if current_start < existing_date:
            # 缺失范围：current_start 到 existing_date - 1天
            missing_end = existing_date - timedelta(days=1)
            if missing_end >= current_start:
                missing_ranges.append((
                    datetime.combine(current_start, datetime.min.time()),
                    datetime.combine(missing_end, datetime.min.time())
                ))
        
        # 更新当前开始日期为已存在日期的下一天
        current_start = existing_date + timedelta(days=1)
    
    # 检查最后一段是否有缺失
    if current_start <= end:
        missing_ranges.append((
            datetime.combine(current_start, datetime.min.time()),
            datetime.combine(end, datetime.min.time())
        ))
    
    return missing_ranges


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
        字典，key为股票代码，value为该股票的DataFrame
    """
    if pd is None:
        logger.error("pandas未安装，无法查询股票数据")
        return {}
    
    conn = None
    result = {}
    
    try:
        conn = get_db_connection()
        if not conn:
            logger.warning("无法获取数据库连接，无法查询股票数据")
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
                    
                    # 添加timestamp列（用于兼容性）
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


def persist_akshare_data_to_postgresql(
    source_id: str,
    data: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    source_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    将AKShare数据持久化到PostgreSQL
    
    Args:
        source_id: 数据源ID
        data: 采集的数据列表
        metadata: 元数据
        source_config: 数据源配置
        
    Returns:
        持久化结果字典
    """
    start_time = time.time()
    conn = None
    
    try:
        # 确保表存在
        if not ensure_table_exists():
            raise Exception("无法确保数据库表存在")
        
        # 获取数据库连接
        conn = get_db_connection()
        if not conn:
            raise Exception("无法获取数据库连接")
        
        cursor = conn.cursor()
        
        # 准备批量插入数据
        inserted_count = 0
        updated_count = 0
        skipped_count = 0
        error_count = 0

        # 查询是否存在记录的辅助函数
        def record_exists(cursor, source_id, symbol, date_obj, data_type):
            cursor.execute("""
                SELECT 1 FROM akshare_stock_data
                WHERE source_id = %s AND symbol = %s AND date = %s AND data_type = %s
                LIMIT 1
            """, (source_id, symbol, date_obj, data_type))
            return cursor.fetchone() is not None

        insert_query = """
            INSERT INTO akshare_stock_data (
                source_id, symbol, date, data_type, open_price, high_price, low_price,
                close_price, volume, amount, pct_change, change,
                turnover_rate, amplitude, data_source, collected_at, persistence_timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
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
        
        collected_at = datetime.fromtimestamp(metadata.get("collection_timestamp", time.time()))
        persistence_timestamp = datetime.now()

        # 数据质量预检和详细诊断
        if data:
            logger.info(f"数据质量预检: 总记录数 {len(data)}")

            # 检查前几条记录的详细数据质量
            quality_check_count = min(5, len(data))
            quality_issues = []
            sample_prices = []

            logger.info(f"🔍 开始数据质量详细检查，检查记录数: {quality_check_count}")

            for i in range(quality_check_count):
                record = data[i]
                issues = []

                # 记录样例数据用于诊断
                if i == 0:
                    logger.info(f"样例记录{i+1}字段: {list(record.keys())}")
                    logger.info(f"样例记录{i+1}数据类型: {[(k, type(v).__name__) for k, v in record.items()]}")

                # 检查关键字段
                if not record.get('date'):
                    issues.append("date缺失")
                else:
                    logger.debug(f"记录{i+1} date: {record['date']} (类型: {type(record['date'])})")

                if not record.get('symbol'):
                    issues.append("symbol缺失")

                # 检查价格字段
                price_fields = ['open', 'close', 'high', 'low']
                for field in price_fields:
                    field_value = record.get(field)
                    if field_value is None or field_value == '':
                        issues.append(f"{field}缺失")
                        logger.warning(f"记录{i+1} {field}为空: {field_value} (类型: {type(field_value)})")
                    else:
                        sample_prices.append(f"{field}={field_value}")
                        logger.debug(f"记录{i+1} {field}: {field_value} (类型: {type(field_value)})")

                        # 检查是否为有效数值
                        try:
                            float_val = float(field_value)
                            if not (float_val >= 0):  # 价格不能为负数
                                issues.append(f"{field}价格异常: {float_val}")
                        except (ValueError, TypeError) as e:
                            issues.append(f"{field}格式错误: {field_value} ({e})")

                if not record.get('volume'):
                    issues.append("volume缺失")

                if issues:
                    quality_issues.append(f"记录{i+1}: {', '.join(issues)}")

            # 报告价格字段状态
            if sample_prices:
                logger.info(f"价格字段样例: {', '.join(sample_prices[:4])}")  # 只显示前4个价格样例
            else:
                logger.error("所有检查的记录都没有价格数据！")

            if quality_issues:
                logger.warning(f"数据质量问题检测: {len(quality_issues)}/{quality_check_count} 条记录存在问题")
                for issue in quality_issues[:3]:  # 只显示前3个问题
                    logger.warning(f"  {issue}")
            else:
                logger.info(f"数据质量良好: 检查的 {quality_check_count} 条记录均完整")

            # 检查中文字段
            chinese_fields_found = []
            for record in data[:3]:
                chinese_fields = [k for k in record.keys() if any('\u4e00' <= c <= '\u9fff' for c in k)]
                if chinese_fields:
                    chinese_fields_found.extend(chinese_fields)

            if chinese_fields_found:
                logger.warning(f"发现未映射的中文字段: {list(set(chinese_fields_found))}")
            else:
                logger.debug("字段名映射完整，无中文字段残留")

        for record in data:
            try:
                # 解析日期
                date_str = record.get("date", "")
                if not date_str:
                    logger.warning(f"记录缺少日期字段，跳过: {record.get('symbol', 'unknown')} - 字段列表: {list(record.keys())}")
                    skipped_count += 1
                    continue

                # 转换日期格式
                try:
                    if isinstance(date_str, str):
                        # 支持多种日期格式
                        if "T" in date_str:
                            date_obj = datetime.fromisoformat(date_str.split("T")[0]).date()
                        else:
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                    else:
                        date_obj = date_str
                except Exception as e:
                    logger.warning(f"日期解析失败: {date_str} (类型: {type(date_str)}), 错误: {e}, 记录: {record.get('symbol', 'unknown')}")
                    skipped_count += 1
                    continue

                symbol = record.get("symbol", "")
                data_type = record.get("data_type", "daily")

                # 检查记录是否已存在，用于准确统计
                exists_before = record_exists(cursor, source_id, symbol, date_obj, data_type)

                # 准备插入值
                values = (
                    source_id,
                    symbol,
                    date_obj,
                    data_type,
                    record.get("open"),
                    record.get("high"),
                    record.get("low"),
                    record.get("close"),
                    int(record.get("volume", 0)) if record.get("volume") else None,
                    record.get("amount"),
                    record.get("pct_change"),
                    record.get("change"),
                    record.get("turnover_rate"),
                    record.get("amplitude"),
                    record.get("data_source", "akshare"),
                    collected_at,
                    persistence_timestamp
                )

                cursor.execute(insert_query, values)

                # 根据是否存在来统计插入或更新
                exists_after = record_exists(cursor, source_id, symbol, date_obj, data_type)
                if exists_before:
                    updated_count += 1
                    logger.debug(f"更新记录: {symbol} {date_str} (data_type: {data_type})")
                elif exists_after:
                    inserted_count += 1
                    logger.debug(f"插入记录: {symbol} {date_str} (data_type: {data_type})")
                else:
                    error_count += 1  # 插入失败
                    logger.warning(f"插入记录后验证失败: {symbol} {date_str}, 检查价格字段: open={record.get('open')}, close={record.get('close')}")

                # 数据质量监控：检查关键字段
                quality_issues = []
                if not record.get('open'):
                    quality_issues.append("open为空")
                if not record.get('close'):
                    quality_issues.append("close为空")
                if not record.get('high'):
                    quality_issues.append("high为空")
                if not record.get('low'):
                    quality_issues.append("low为空")
                if not record.get('volume'):
                    quality_issues.append("volume为空")

                if quality_issues:
                    logger.warning(f"数据质量问题: {symbol} {date_str} ({data_type}) - {', '.join(quality_issues)}")
                else:
                    logger.debug(f"数据质量正常: {symbol} {date_str} ({data_type}) - open:{record.get('open')}, close:{record.get('close')}")

            except Exception as e:
                error_count += 1
                logger.warning(f"处理记录失败: {record.get('symbol', 'unknown')} {record.get('date', 'unknown')}, 错误: {e}")
                continue
        
        # 提交事务
        conn.commit()
        cursor.close()
        
        processing_time = time.time() - start_time

        # 计算数据质量评分
        quality_score = 0.0
        if len(data) > 0:
            # 统计有价格数据的记录数
            valid_price_records = 0
            for record in data:
                if (record.get('open') and record.get('close') and
                    record.get('high') and record.get('low')):
                    try:
                        # 检查价格是否为有效数值
                        float(record['open'])
                        float(record['close'])
                        float(record['high'])
                        float(record['low'])
                        valid_price_records += 1
                    except (ValueError, TypeError):
                        pass
            quality_score = (valid_price_records / len(data)) * 100

        logger.info(
            f"PostgreSQL持久化完成: {source_id}, "
            f"插入: {inserted_count}, 更新: {updated_count}, 跳过: {skipped_count}, 错误: {error_count}, "
            f"数据质量评分: {quality_score:.1f}%, "
            f"耗时: {processing_time:.2f}秒"
        )

        return {
            "success": True,
            "storage_type": "postgresql",
            "inserted_count": inserted_count,
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "processing_time": processing_time,
            "data_quality_score": quality_score,
            "message": f"数据已成功持久化到PostgreSQL，插入{inserted_count}条记录，更新{updated_count}条记录，数据质量评分{quality_score:.1f}%"
        }
        
    except Exception as e:
        error_msg = f"PostgreSQL持久化异常: {str(e)}"
        logger.error(error_msg)
        
        if conn:
            conn.rollback()
        
        return {
            "success": False,
            "error": error_msg,
            "processing_time": time.time() - start_time
        }
    finally:
        if conn:
            return_db_connection(conn)


def persist_akshare_index_data_to_postgresql(
    source_id: str,
    data: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    source_config: Dict[str, Any]
) -> Dict[str, Any]:
    """将AKShare指数数据持久化到PostgreSQL"""
    start_time = time.time()
    conn = None
    
    try:
        # 确保表存在
        conn = get_db_connection()
        if not conn:
            raise Exception("无法获取数据库连接")
        
        cursor = conn.cursor()
        
        # 读取并执行SQL schema文件
        # 尝试多个可能的路径
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "sql", "akshare_index_data_schema.sql"),
            os.path.join("scripts", "sql", "akshare_index_data_schema.sql"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts", "sql", "akshare_index_data_schema.sql")
        ]
        script_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                script_path = abs_path
                break
        
        if script_path and os.path.exists(script_path):
            with open(script_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            try:
                cursor.execute(sql_script)
                conn.commit()
            except Exception as e:
                logger.debug(f"表可能已存在: {e}")
        else:
            # 如果SQL文件不存在，直接创建表（简化版）
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS akshare_index_data (
                        id BIGSERIAL PRIMARY KEY,
                        source_id VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        date DATE NOT NULL,
                        open_price DECIMAL(15, 6),
                        high_price DECIMAL(15, 6),
                        low_price DECIMAL(15, 6),
                        close_price DECIMAL(15, 6),
                        volume BIGINT,
                        amount DECIMAL(20, 2),
                        pct_change DECIMAL(10, 4),
                        change DECIMAL(15, 6),
                        data_source VARCHAR(50) DEFAULT 'akshare',
                        collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(source_id, symbol, date)
                    );
                """)
                conn.commit()
            except Exception as e:
                logger.debug(f"表创建可能已存在: {e}")
        
        # 批量插入数据
        inserted_count = 0
        skipped_count = 0
        
        insert_query = """
            INSERT INTO akshare_index_data (
                source_id, symbol, date, open_price, high_price, low_price,
                close_price, volume, amount, pct_change, change,
                data_source, collected_at, persistence_timestamp
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (source_id, symbol, date) 
            DO UPDATE SET
                open_price = EXCLUDED.open_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                close_price = EXCLUDED.close_price,
                volume = EXCLUDED.volume,
                amount = EXCLUDED.amount,
                pct_change = EXCLUDED.pct_change,
                change = EXCLUDED.change,
                persistence_timestamp = EXCLUDED.persistence_timestamp
        """
        
        collected_at = datetime.fromtimestamp(metadata.get("collection_timestamp", time.time()))
        persistence_timestamp = datetime.now()
        
        for record in data:
            try:
                date_value = record.get("date")
                if isinstance(date_value, str):
                    date_value = datetime.strptime(date_value[:10], "%Y-%m-%d").date()
                elif isinstance(date_value, datetime):
                    date_value = date_value.date()
                
                cursor.execute(insert_query, (
                    source_id,
                    record.get("symbol", ""),
                    date_value,
                    record.get("open"),
                    record.get("high"),
                    record.get("low"),
                    record.get("close"),
                    int(record.get("volume", 0)),
                    record.get("amount"),
                    record.get("pct_change"),
                    record.get("change"),
                    "akshare",
                    collected_at,
                    persistence_timestamp
                ))
                inserted_count += 1
            except Exception as e:
                logger.warning(f"插入指数数据失败: {e}")
                skipped_count += 1
        
        conn.commit()
        cursor.close()
        
        return {
            "success": True,
            "storage_type": "postgresql",
            "inserted_count": inserted_count,
            "skipped_count": skipped_count,
            "total_count": len(data),
            "processing_time": time.time() - start_time
        }
    except Exception as e:
        logger.error(f"指数数据持久化失败: {e}")
        if conn:
            conn.rollback()
        return {
            "success": False,
            "error": str(e),
            "storage_type": "postgresql",
            "processing_time": time.time() - start_time
        }
    finally:
        if conn:
            return_db_connection(conn)


def persist_akshare_fundamental_data_to_postgresql(
    source_id: str,
    data: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    source_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    将AKShare基本面数据持久化到PostgreSQL
    
    Args:
        source_id: 数据源ID
        data: 采集的基本面数据列表
        metadata: 元数据
        source_config: 数据源配置
        
    Returns:
        持久化结果字典
    """
    start_time = time.time()
    conn = None
    
    try:
        # 获取数据库连接
        conn = get_db_connection()
        if not conn:
            raise Exception("无法获取数据库连接")
        
        cursor = conn.cursor()
        
        # 确保基本面数据表存在
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "sql", "akshare_fundamental_data_schema.sql"),
            os.path.join("scripts", "sql", "akshare_fundamental_data_schema.sql"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts", "sql", "akshare_fundamental_data_schema.sql")
        ]
        script_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                script_path = abs_path
                break
        
        if script_path and os.path.exists(script_path):
            with open(script_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            try:
                cursor.execute(sql_script)
                conn.commit()
                logger.info("基本面数据表结构初始化成功")
            except Exception as e:
                logger.debug(f"基本面数据表可能已存在: {e}")
        else:
            # 如果SQL文件不存在，直接创建表（简化版）
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS akshare_fundamental_data (
                        id BIGSERIAL PRIMARY KEY,
                        source_id VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        report_date DATE NOT NULL,
                        company_name VARCHAR(100),
                        industry VARCHAR(50),
                        pe DECIMAL(10, 4),
                        pb DECIMAL(10, 4),
                        market_cap DECIMAL(20, 2),
                        revenue DECIMAL(20, 2),
                        net_profit DECIMAL(20, 2),
                        roe DECIMAL(10, 4),
                        data_source VARCHAR(50) DEFAULT 'akshare',
                        collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT unique_fundamental_record UNIQUE(source_id, symbol, report_date)
                    );
                    
                    -- 创建索引
                    CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_symbol_date ON akshare_fundamental_data(symbol, report_date DESC);
                    CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_industry ON akshare_fundamental_data(industry);
                    CREATE INDEX IF NOT EXISTS idx_akshare_fundamental_source_collected ON akshare_fundamental_data(source_id, collected_at DESC);
                """)
                conn.commit()
                logger.debug("基本面数据表结构检查完成")
            except Exception as e:
                logger.debug(f"基本面数据表可能已存在: {e}")
        
        # 数据处理和插入逻辑
        inserted_count = 0
        
        for item in data:
            try:
                # 构建插入SQL
                insert_sql = """
                    INSERT INTO akshare_fundamental_data (
                        source_id, symbol, report_date, company_name, industry, 
                        pe, pb, market_cap, revenue, net_profit, roe
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_id, symbol, report_date) DO NOTHING
                """
                
                # 准备参数
                params = (
                    source_id,
                    item.get('symbol', ''),
                    item.get('report_date', ''),
                    item.get('company_name', ''),
                    item.get('industry', ''),
                    item.get('pe', None),
                    item.get('pb', None),
                    item.get('market_cap', None),
                    item.get('revenue', None),
                    item.get('net_profit', None),
                    item.get('roe', None)
                )
                
                cursor.execute(insert_sql, params)
                inserted_count += 1
                
            except Exception as e:
                logger.debug(f"插入基本面数据时出错: {e}")
                continue
        
        if inserted_count > 0:
            conn.commit()
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "inserted_count": inserted_count,
            "total_count": len(data),
            "processing_time": processing_time,
            "message": f"基本面数据持久化完成，插入 {inserted_count} 条记录"
        }
        
    except Exception as e:
        logger.error(f"持久化基本面数据到PostgreSQL失败: {e}")
        if conn:
            try:
                conn.rollback()
            except:
                pass
        return {
            "success": False,
            "error": str(e),
            "inserted_count": 0,
            "total_count": len(data),
            "message": "基本面数据持久化失败"
        }
    finally:
        if 'cursor' in locals():
            try:
                cursor.close()
            except:
                pass
        if conn:
            try:
                conn.close()
            except:
                pass


def save_stock_basic_info(stock_info: pd.DataFrame) -> Dict[str, Any]:
    """
    保存股票基本信息到数据库
    
    Args:
        stock_info: 股票基本信息DataFrame
        
    Returns:
        保存结果字典
    """
    start_time = time.time()
    conn = None
    
    try:
        # 确保表存在
        if not ensure_table_exists():
            return {
                "success": False,
                "error": "无法确保数据库表存在"
            }
        
        # 获取数据库连接
        conn = get_db_connection()
        if not conn:
            return {
                "success": False,
                "error": "无法获取数据库连接"
            }
        
        cursor = conn.cursor()
        
        # 批量插入或更新
        inserted_count = 0
        updated_count = 0
        skipped_count = 0
        error_count = 0
        
        for _, row in stock_info.iterrows():
            try:
                symbol = row.get('symbol')
                if not symbol:
                    skipped_count += 1
                    continue
                
                # 检查是否已存在
                cursor.execute("""
                    SELECT 1 FROM stock_basic_info
                    WHERE symbol = %s
                    LIMIT 1
                """, (symbol,))
                
                exists = cursor.fetchone() is not None
                
                if exists:
                    # 更新现有记录
                    cursor.execute("""
                        UPDATE stock_basic_info
                        SET name = %s, ipo_date = %s, industry = %s,
                            market = %s, total_share = %s, float_share = %s,
                            pe = %s, pb = %s, roe = %s,
                            update_time = CURRENT_TIMESTAMP
                        WHERE symbol = %s
                    """, (
                        row.get('name', ''),
                        row.get('ipo_date'),
                        row.get('industry', ''),
                        row.get('market', ''),
                        row.get('total_share'),
                        row.get('float_share'),
                        row.get('pe'),
                        row.get('pb'),
                        row.get('roe'),
                        symbol
                    ))
                    updated_count += 1
                else:
                    # 插入新记录
                    cursor.execute("""
                        INSERT INTO stock_basic_info (
                            symbol, name, ipo_date, industry, market,
                            total_share, float_share, pe, pb, roe
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        symbol,
                        row.get('name', ''),
                        row.get('ipo_date'),
                        row.get('industry', ''),
                        row.get('market', ''),
                        row.get('total_share'),
                        row.get('float_share'),
                        row.get('pe'),
                        row.get('pb'),
                        row.get('roe')
                    ))
                    inserted_count += 1
                    
            except Exception as e:
                logger.warning(f"处理股票 {row.get('symbol', 'unknown')} 基本信息失败: {e}")
                error_count += 1
        
        # 提交事务
        conn.commit()
        cursor.close()
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"股票基本信息持久化完成: "
            f"插入: {inserted_count}, 更新: {updated_count}, 跳过: {skipped_count}, 错误: {error_count}, "
            f"耗时: {processing_time:.2f}秒"
        )
        
        return {
            "success": True,
            "storage_type": "postgresql",
            "inserted_count": inserted_count,
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "processing_time": processing_time,
            "message": f"成功保存 {inserted_count + updated_count} 条股票基本信息"
        }
        
    except Exception as e:
        error_msg = f"股票基本信息持久化异常: {str(e)}"
        logger.error(error_msg)
        
        if conn:
            conn.rollback()
        
        return {
            "success": False,
            "error": error_msg,
            "processing_time": time.time() - start_time
        }
    finally:
        if conn:
            return_db_connection(conn)


def get_stock_basic_info(symbol: str) -> Dict[str, Any]:
    """
    根据股票代码获取基本信息
    
    Args:
        symbol: 股票代码
        
    Returns:
        股票基本信息字典
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return {}
        
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT symbol, name, ipo_date, industry, market,
                   total_share, float_share, pe, pb, roe,
                   update_time
            FROM stock_basic_info
            WHERE symbol = %s
        "", (symbol,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if row:
            return {
                "symbol": row[0],
                "name": row[1],
                "ipo_date": row[2].isoformat() if row[2] else None,
                "industry": row[3],
                "market": row[4],
                "total_share": row[5],
                "float_share": row[6],
                "pe": row[7],
                "pb": row[8],
                "roe": row[9],
                "update_time": row[10].isoformat() if row[10] else None
            }
        else:
            return {}
    except Exception as e:
        logger.error(f"获取股票基本信息失败: {e}")
        return {}
    finally:
        if conn:
            return_db_connection(conn)


def get_stocks_by_industry(industry: str) -> List[Dict[str, Any]]:
    """
    根据行业获取股票列表
    
    Args:
        industry: 行业名称
        
    Returns:
