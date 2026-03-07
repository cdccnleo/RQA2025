#!/usr/bin/env python3
"""
初始化AKShare数据库表结构
支持PostgreSQL + TimescaleDB
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
from psycopg2 import sql
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_config():
    """获取数据库配置"""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "rqa2025"),
        "user": os.getenv("DB_USER", "rqa_user"),
        "password": os.getenv("DB_PASSWORD", ""),
    }


def check_timescaledb_extension(conn):
    """检查TimescaleDB扩展是否可用"""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb');")
        exists = cursor.fetchone()[0]
        cursor.close()
        return exists
    except Exception as e:
        logger.warning(f"检查TimescaleDB扩展失败: {e}")
        return False


def create_timescaledb_extension(conn):
    """创建TimescaleDB扩展"""
    try:
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        conn.commit()
        cursor.close()
        logger.info("TimescaleDB扩展创建成功")
        return True
    except Exception as e:
        logger.warning(f"创建TimescaleDB扩展失败（可能未安装）: {e}")
        conn.rollback()
        return False


def init_akshare_tables():
    """初始化AKShare数据库表"""
    config = get_db_config()
    
    try:
        # 连接数据库
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
            connect_timeout=10
        )
        
        logger.info(f"数据库连接成功: {config['host']}:{config['port']}/{config['database']}")
        
        cursor = conn.cursor()
        
        # 读取SQL文件
        sql_file = project_root / "scripts" / "sql" / "akshare_stock_data_schema.sql"
        if sql_file.exists():
            logger.info(f"读取SQL文件: {sql_file}")
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 执行SQL
            cursor.execute(sql_content)
            conn.commit()
            logger.info("SQL脚本执行成功")
        else:
            logger.warning(f"SQL文件不存在: {sql_file}")
            # 手动创建表
            logger.info("手动创建表结构...")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS akshare_stock_data (
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
                    turnover_rate DECIMAL(10, 4),
                    amplitude DECIMAL(10, 4),
                    data_source VARCHAR(50) DEFAULT 'akshare',
                    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unique_akshare_record UNIQUE(source_id, symbol, date)
                );
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_akshare_symbol_date ON akshare_stock_data(symbol, date DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_akshare_source_collected ON akshare_stock_data(source_id, collected_at DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_akshare_date_range ON akshare_stock_data(date DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_akshare_symbol_source ON akshare_stock_data(symbol, source_id);")
            
            conn.commit()
            logger.info("表结构创建成功")
        
        # 检查TimescaleDB
        if check_timescaledb_extension(conn):
            logger.info("TimescaleDB扩展已安装")
            
            # 尝试创建超表
            try:
                cursor.execute("""
                    SELECT create_hypertable(
                        'akshare_stock_data',
                        'date',
                        chunk_time_interval => INTERVAL '1 month',
                        if_not_exists => TRUE
                    );
                """)
                conn.commit()
                logger.info("TimescaleDB超表创建成功")
            except Exception as e:
                logger.warning(f"创建TimescaleDB超表失败（可能已存在）: {e}")
                conn.rollback()
        else:
            logger.info("TimescaleDB扩展未安装，使用标准PostgreSQL表")
        
        cursor.close()
        conn.close()
        
        logger.info("✅ 数据库初始化完成")
        return True
        
    except psycopg2.OperationalError as e:
        logger.error(f"数据库连接失败: {e}")
        logger.error("请检查:")
        logger.error("  1. PostgreSQL服务是否运行")
        logger.error("  2. 数据库配置是否正确（环境变量或配置文件）")
        logger.error("  3. 数据库用户权限是否足够")
        return False
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("AKShare数据库初始化脚本")
    logger.info("=" * 60)
    
    success = init_akshare_tables()
    
    if success:
        logger.info("\n✅ 初始化成功！")
        sys.exit(0)
    else:
        logger.error("\n❌ 初始化失败！")
        sys.exit(1)

