#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据表分区管理

实现按月的表分区策略，优化查询性能。
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Tuple

import psycopg2

logger = logging.getLogger(__name__)


def get_db_connection():
    """获取数据库连接"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
        user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
        password=os.getenv('POSTGRES_PASSWORD', '')
    )


def create_partition_table():
    """
    创建分区表
    
    将 akshare_stock_data 转换为分区表
    """
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    
    try:
        # 检查是否已经是分区表
        cur.execute("""
            SELECT partstrat FROM pg_partitioned_table pt
            JOIN pg_class c ON pt.partrelid = c.oid
            WHERE c.relname = 'akshare_stock_data'
        """)
        
        if cur.fetchone():
            logger.info("表已经是分区表，跳过创建")
            return True
        
        # 创建新的分区表
        logger.info("开始创建分区表...")
        
        # 重命名现有表
        cur.execute("""
            ALTER TABLE IF EXISTS akshare_stock_data 
            RENAME TO akshare_stock_data_old
        """)
        
        # 创建分区主表
        cur.execute("""
            CREATE TABLE akshare_stock_data (
                id BIGSERIAL,
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
                PRIMARY KEY (id, date)
            ) PARTITION BY RANGE (date)
        """)
        
        # 创建默认分区
        cur.execute("""
            CREATE TABLE akshare_stock_data_default 
            PARTITION OF akshare_stock_data DEFAULT
        """)
        
        logger.info("✅ 分区表创建成功")
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建分区表失败: {e}")
        return False
    finally:
        cur.close()
        conn.close()


def create_monthly_partition(year: int, month: int) -> bool:
    """
    创建指定月份的分区
    
    Args:
        year: 年份
        month: 月份
    
    Returns:
        是否创建成功
    """
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    
    try:
        # 计算分区范围
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        
        start_date = datetime(year, month, 1)
        end_date = next_month
        
        partition_name = f"akshare_stock_data_{year}_{month:02d}"
        
        # 检查分区是否已存在
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM pg_class WHERE relname = '{partition_name}'
            )
        """)
        
        if cur.fetchone()[0]:
            logger.info(f"分区 {partition_name} 已存在，跳过")
            return True
        
        # 创建分区
        cur.execute(f"""
            CREATE TABLE {partition_name}
            PARTITION OF akshare_stock_data
            FOR VALUES FROM ('{start_date.strftime('%Y-%m-%d')}') 
            TO ('{end_date.strftime('%Y-%m-%d')}')
        """)
        
        logger.info(f"✅ 创建分区: {partition_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建分区失败: {e}")
        return False
    finally:
        cur.close()
        conn.close()


def create_partitions_for_range(start_date: datetime, end_date: datetime) -> int:
    """
    为日期范围创建分区
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        创建的分区数量
    """
    count = 0
    current = datetime(start_date.year, start_date.month, 1)
    
    while current <= end_date:
        if create_monthly_partition(current.year, current.month):
            count += 1
        
        # 移动到下个月
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    return count


def get_partition_info() -> List[Tuple[str, str, str]]:
    """
    获取分区信息
    
    Returns:
        分区信息列表 [(分区名, 开始日期, 结束日期)]
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT 
                c.relname AS partition_name,
                pg_get_expr(c.relpartbound, c.oid) AS partition_bound
            FROM pg_class c
            JOIN pg_inherits i ON c.oid = i.inhrelid
            JOIN pg_class p ON i.inhparent = p.oid
            WHERE p.relname = 'akshare_stock_data'
            ORDER BY c.relname
        """)
        
        partitions = []
        for row in cur.fetchall():
            partitions.append((row[0], row[1] or 'default', ''))
        
        return partitions
        
    except Exception as e:
        logger.error(f"获取分区信息失败: {e}")
        return []
    finally:
        cur.close()
        conn.close()


def migrate_data_from_old_table(batch_size: int = 10000) -> int:
    """
    从旧表迁移数据到分区表
    
    Args:
        batch_size: 批量迁移大小
    
    Returns:
        迁移的记录数
    """
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    
    try:
        # 检查旧表是否存在
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM pg_class WHERE relname = 'akshare_stock_data_old'
            )
        """)
        
        if not cur.fetchone()[0]:
            logger.info("旧表不存在，无需迁移")
            return 0
        
        # 获取总记录数
        cur.execute("SELECT COUNT(*) FROM akshare_stock_data_old")
        total = cur.fetchone()[0]
        
        if total == 0:
            logger.info("旧表无数据，无需迁移")
            return 0
        
        logger.info(f"开始迁移 {total} 条记录...")
        
        # 批量迁移
        migrated = 0
        offset = 0
        
        while offset < total:
            cur.execute(f"""
                INSERT INTO akshare_stock_data
                SELECT * FROM akshare_stock_data_old
                ORDER BY id
                LIMIT {batch_size} OFFSET {offset}
                ON CONFLICT DO NOTHING
            """)
            
            migrated += cur.rowcount
            offset += batch_size
            
            logger.info(f"已迁移 {migrated}/{total} 条记录...")
        
        logger.info(f"✅ 数据迁移完成: {migrated} 条记录")
        return migrated
        
    except Exception as e:
        logger.error(f"❌ 数据迁移失败: {e}")
        return 0
    finally:
        cur.close()
        conn.close()


def create_indexes_on_partitions():
    """在分区表上创建索引"""
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    
    indexes = [
        ("idx_akshare_stock_symbol", "symbol"),
        ("idx_akshare_stock_source", "source_id"),
        ("idx_akshare_stock_source_date", "source_id, date"),
        ("idx_akshare_stock_data_type", "data_type"),
    ]
    
    try:
        for idx_name, idx_cols in indexes:
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {idx_name}
                ON akshare_stock_data ({idx_cols})
            """)
            logger.info(f"✅ 创建索引: {idx_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建索引失败: {e}")
        return False
    finally:
        cur.close()
        conn.close()


def setup_partition_maintenance_function():
    """创建分区自动维护函数"""
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    
    try:
        # 创建自动创建下月分区的函数
        cur.execute("""
            CREATE OR REPLACE FUNCTION auto_create_next_month_partition()
            RETURNS void AS $$
            DECLARE
                next_month DATE;
                next_next_month DATE;
                partition_name TEXT;
            BEGIN
                next_month := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month');
                next_next_month := DATE_TRUNC('month', next_month + INTERVAL '1 month');
                partition_name := 'akshare_stock_data_' || TO_CHAR(next_month, 'YYYY_MM');
                
                EXECUTE format(
                    'CREATE TABLE IF NOT EXISTS %I PARTITION OF akshare_stock_data 
                     FOR VALUES FROM (%L) TO (%L)',
                    partition_name, next_month, next_next_month
                );
            END;
            $$ LANGUAGE plpgsql
        """)
        
        logger.info("✅ 创建分区维护函数")
        return True
        
    except Exception as e:
        logger.error(f"❌ 创建分区维护函数失败: {e}")
        return False
    finally:
        cur.close()
        conn.close()


def main():
    """主函数：执行完整的分区设置"""
    print("=" * 60)
    print("股票数据表分区优化")
    print("=" * 60)
    
    # 1. 创建分区表
    print("\n1. 创建分区表...")
    if not create_partition_table():
        print("❌ 分区表创建失败")
        return
    
    # 2. 创建当前和未来月份的分区
    print("\n2. 创建月度分区...")
    today = datetime.now()
    # 创建过去3个月和未来3个月的分区
    start = datetime(today.year, today.month, 1) - timedelta(days=90)
    end = datetime(today.year, today.month, 1) + timedelta(days=180)
    
    count = create_partitions_for_range(start, end)
    print(f"✅ 创建了 {count} 个分区")
    
    # 3. 创建索引
    print("\n3. 创建索引...")
    create_indexes_on_partitions()
    
    # 4. 迁移数据
    print("\n4. 迁移数据...")
    migrated = migrate_data_from_old_table()
    print(f"✅ 迁移了 {migrated} 条记录")
    
    # 5. 设置维护函数
    print("\n5. 设置分区维护函数...")
    setup_partition_maintenance_function()
    
    # 6. 显示分区信息
    print("\n6. 分区信息:")
    partitions = get_partition_info()
    for name, bound, _ in partitions:
        print(f"  - {name}: {bound}")
    
    print("\n" + "=" * 60)
    print("✅ 分区优化完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
