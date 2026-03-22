#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""创建股票数据语义化视图"""

import os
import psycopg2

def create_views():
    """创建股票数据视图"""
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
        user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
        password=os.getenv('POSTGRES_PASSWORD', '')
    )
    cur = conn.cursor()
    
    views_sql = """
    -- 日线行情视图
    CREATE OR REPLACE VIEW v_stock_daily_price AS
    SELECT 
        source_id, symbol, date,
        open_price AS open, high_price AS high, low_price AS low, close_price AS close,
        volume, amount, pct_change, change, turnover_rate, amplitude, collected_at
    FROM akshare_stock_data 
    WHERE data_type = 'daily';
    
    -- 最新价格视图
    CREATE OR REPLACE VIEW v_stock_latest_price AS
    SELECT DISTINCT ON (source_id, symbol)
        source_id, symbol, date, close_price, volume, amount, pct_change, turnover_rate, collected_at
    FROM akshare_stock_data
    ORDER BY source_id, symbol, date DESC;
    
    -- 数据源统计视图
    CREATE OR REPLACE VIEW v_stock_data_source_stats AS
    SELECT 
        source_id, data_type,
        COUNT(*) AS total_records,
        COUNT(DISTINCT symbol) AS symbol_count,
        MIN(date) AS earliest_date,
        MAX(date) AS latest_date,
        MAX(collected_at) AS last_collected
    FROM akshare_stock_data
    GROUP BY source_id, data_type;
    """
    
    try:
        cur.execute(views_sql)
        conn.commit()
        print('✅ 视图创建成功')
        
        # 验证视图
        cur.execute("SELECT table_name FROM information_schema.views WHERE table_schema = 'public' AND table_name LIKE 'v_stock%'")
        views = cur.fetchall()
        print(f'已创建的视图: {[v[0] for v in views]}')
    except Exception as e:
        print(f'❌ 创建视图失败: {e}')
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    create_views()
