#!/usr/bin/env python3
"""
外汇历史数据采集脚本
从AkShare(currency_boc_safe)采集中国银行外汇牌价并持久化到PostgreSQL
"""
import os, sys, time, logging
from datetime import datetime
sys.path.insert(0, '/app')
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_db():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=5432, user='rqa2025_admin',
        password=os.getenv('POSTGRES_PASSWORD', 'RQA2025_Postgres_Secure_2026'),
        database=os.getenv('POSTGRES_DB', 'rqa2025_prod')
    )


def create_forex_table():
    """创建外汇历史数据表"""
    conn = get_db()
    cur = conn.cursor()
    
    # 外汇历史数据表 (BOC中国银行外汇牌价)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS akshare_forex_history (
            id BIGSERIAL PRIMARY KEY,
            source_id VARCHAR(50) NOT NULL DEFAULT 'akshare_forex_history',
            currency_pair VARCHAR(20) NOT NULL,  -- 'USD/CNY', 'EUR/CNY'
            base_currency VARCHAR(10) NOT NULL,
            quote_currency VARCHAR(10) NOT NULL DEFAULT 'CNY',
            date DATE NOT NULL,
            price NUMERIC(16, 6) NOT NULL,  -- per 100 units of foreign currency
            change_ratio NUMERIC(10, 4),
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(currency_pair, date)
        )
    """)
    
    for idx in [
        'CREATE INDEX IF NOT EXISTS idx_forex_pair_date ON akshare_forex_history(currency_pair, date DESC)',
        'CREATE INDEX IF NOT EXISTS idx_forex_date ON akshare_forex_history(date DESC)',
    ]:
        cur.execute(idx)
    
    conn.commit()
    cur.close()
    conn.close()
    logger.info("akshare_forex_history table created OK")


def validate_price(val) -> float:
    """验证价格有效性"""
    if val is None:
        return None
    try:
        v = float(val)
        if v <= 0 or v > 100000:
            return None
        return v
    except:
        return None


def collect_forex_history():
    """从BOC采集外汇历史数据（只采集2020年至今）"""
    import akshare as ak
    import pandas as pd
    
    logger.info("Fetching currency_boc_safe data...")
    df = ak.currency_boc_safe()
    logger.info(f"Total rows: {len(df)}, date range: {df['日期'].min()} to {df['日期'].max()}")
    
    # 只采集2020年至今的数据（减少数据量）
    df = df[df['日期'] >= pd.Timestamp('2020-01-01').date()]
    logger.info(f"After filter (2020+): {len(df)} rows")
    
    # 主要货币对
    currencies = {
        'USD/CNY': ('美元', 'USD'),
        'EUR/CNY': ('欧元', 'EUR'),
        'JPY/CNY': ('日元', 'JPY'),
        'HKD/CNY': ('港元', 'HKD'),
        'GBP/CNY': ('英镑', 'GBP'),
        'AUD/CNY': ('澳元', 'AUD'),
    }
    
    records = []
    for _, row in df.iterrows():
        date_val = row.get('日期')
        if date_val is None:
            continue
        
        if hasattr(date_val, 'strftime'):
            date_str = date_val.strftime('%Y-%m-%d')
        else:
            date_str = str(date_val)[:10]
        
        for pair, (col_name, code) in currencies.items():
            if col_name in df.columns:
                price = validate_price(row.get(col_name))
                if price is not None:
                    records.append({
                        'pair': pair,
                        'base': code,
                        'date': date_str,
                        'price': price,
                    })
    
    logger.info(f"Total records to insert: {len(records)}")
    return records


def persist_forex(records: list) -> int:
    if not records:
        return 0
    
    conn = get_db()
    cur = conn.cursor()
    n = 0
    for r in records:
        try:
            cur.execute("""
                INSERT INTO akshare_forex_history 
                (source_id, currency_pair, base_currency, quote_currency, date, price)
                VALUES ('akshare_forex_history', %s, %s, 'CNY', %s, %s)
                ON CONFLICT (currency_pair, date) DO UPDATE SET price = EXCLUDED.price
            """, (r['pair'], r['base'], r['date'], r['price']))
            n += 1
        except Exception as e:
            logger.error(f"Insert failed: {e}")
    conn.commit()
    cur.close()
    conn.close()
    return n


def main():
    create_forex_table()
    records = collect_forex_history()
    
    if records:
        # 分批插入避免内存问题
        batch_size = 1000
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            n = persist_forex(batch)
            total += n
            logger.info(f"Batch {i//batch_size + 1}: {len(batch)} inserted")
        
        logger.info(f"=== Total: {total} records inserted ===")
    
    # 验证
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM akshare_forex_history")
    logger.info(f"Total in DB: {cur.fetchone()[0]}")
    cur.execute("SELECT currency_pair, COUNT(*) FROM akshare_forex_history GROUP BY currency_pair ORDER BY currency_pair")
    for r in cur.fetchall():
        logger.info(f"  {r[0]}: {r[1]}")
    cur.execute("SELECT date FROM akshare_forex_history GROUP BY date ORDER BY date DESC LIMIT 3")
    logger.info(f"Latest dates: {[r[0] for r in cur.fetchall()]}")
    cur.close()
    conn.close()


if __name__ == '__main__':
    main()
