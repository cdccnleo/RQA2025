#!/usr/bin/env python3
"""
宏观经济数据采集脚本
从AkShare采集宏观经济数据并持久化到PostgreSQL
"""
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, '/app')

import akshare as ak
import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'RQA2025_Postgres_Secure_2026'),
        database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
        connect_timeout=10
    )

def parse_date(date_val) -> Optional[str]:
    """将各种日期格式转换为YYYY-MM-DD"""
    if date_val is None:
        return None
    if hasattr(date_val, 'isoformat'):
        return date_val.isoformat()
    if isinstance(date_val, str):
        # 处理"2026年03月份" -> "2026-03"
        import re
        m = re.match(r'(\d{4})年(\d{2})月?', date_val)
        if m:
            return f"{m.group(1)}-{m.group(2)}-01"
        return date_val[:10] if len(date_val) >= 10 else date_val
    return str(date_val)

def collect_macro_china_gdp() -> List[Dict]:
    """采集中国GDP年度数据"""
    try:
        df = ak.macro_china_gdp_yearly()
        records = []
        for _, row in df.iterrows():
            records.append({
                'source_id': 'akshare_macro_china',
                'country': 'china',
                'indicator_type': 'gdp',
                'indicator_name': str(row.iloc[0]) if len(row) > 0 else '中国GDP年率',
                'period': 'yearly',
                'date': parse_date(row.iloc[1] if len(row) > 1 else None),
                'value': float(row.iloc[2]) if len(row) > 2 and row.iloc[2] is not None else None,
                'forecast_value': float(row.iloc[3]) if len(row) > 3 and row.iloc[3] is not None else None,
                'previous_value': float(row.iloc[4]) if len(row) > 4 and row.iloc[4] is not None else None,
            })
        logger.info(f"collect_macro_china_gdp: {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"collect_macro_china_gdp failed: {e}")
        return []

def collect_macro_china_ppi() -> List[Dict]:
    """采集中国PPI数据"""
    try:
        df = ak.macro_china_ppi()
        records = []
        for _, row in df.iterrows():
            records.append({
                'source_id': 'akshare_macro_china',
                'country': 'china',
                'indicator_type': 'ppi',
                'indicator_name': '中国PPI',
                'period': 'monthly',
                'date': parse_date(row.iloc[0] if len(row) > 0 else None),
                'value': float(row.iloc[1]) if len(row) > 1 and row.iloc[1] is not None else None,
                'forecast_value': None,
                'previous_value': float(row.iloc[2]) if len(row) > 2 and row.iloc[2] is not None else None,
            })
        logger.info(f"collect_macro_china_ppi: {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"collect_macro_china_ppi failed: {e}")
        return []

def collect_macro_china_cpi() -> List[Dict]:
    """采集中国CPI数据"""
    try:
        df = ak.macro_china_cpi()
        records = []
        for _, row in df.iterrows():
            records.append({
                'source_id': 'akshare_macro_china',
                'country': 'china',
                'indicator_type': 'cpi',
                'indicator_name': '中国CPI',
                'period': 'monthly',
                'date': parse_date(row.iloc[0] if len(row) > 0 else None),
                'value': float(row.iloc[1]) if len(row) > 1 and row.iloc[1] is not None else None,
                'forecast_value': None,
                'previous_value': float(row.iloc[3]) if len(row) > 3 and row.iloc[3] is not None else None,
            })
        logger.info(f"collect_macro_china_cpi: {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"collect_macro_china_cpi failed: {e}")
        return []

def collect_macro_china_money_supply() -> List[Dict]:
    """采集中国货币供应量数据"""
    try:
        df = ak.macro_china_money_supply()
        records = []
        for _, row in df.iterrows():
            records.append({
                'source_id': 'akshare_macro_china',
                'country': 'china',
                'indicator_type': 'money_supply',
                'indicator_name': '中国货币供应量M2',
                'period': 'monthly',
                'date': parse_date(row.iloc[0] if len(row) > 0 else None),
                'value': float(row.iloc[1]) if len(row) > 1 and row.iloc[1] is not None else None,
                'forecast_value': None,
                'previous_value': float(row.iloc[2]) if len(row) > 2 and row.iloc[2] is not None else None,
            })
        logger.info(f"collect_macro_china_money_supply: {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"collect_macro_china_money_supply failed: {e}")
        return []

def collect_macro_usa_gdp() -> List[Dict]:
    """采集美国GDP数据"""
    try:
        df = ak.macro_usa_gdp_monthly()
        records = []
        for _, row in df.iterrows():
            records.append({
                'source_id': 'akshare_macro_usa',
                'country': 'usa',
                'indicator_type': 'gdp',
                'indicator_name': str(row.iloc[0]) if len(row) > 0 else '美国GDP',
                'period': 'monthly',
                'date': parse_date(row.iloc[1] if len(row) > 1 else None),
                'value': float(row.iloc[2]) if len(row) > 2 and row.iloc[2] is not None else None,
                'forecast_value': float(row.iloc[3]) if len(row) > 3 and row.iloc[3] is not None else None,
                'previous_value': float(row.iloc[4]) if len(row) > 4 and row.iloc[4] is not None else None,
            })
        logger.info(f"collect_macro_usa_gdp: {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"collect_macro_usa_gdp failed: {e}")
        return []

def collect_macro_usa_cpi() -> List[Dict]:
    """采集美国CPI数据"""
    try:
        df = ak.macro_usa_cpi_monthly()
        records = []
        for _, row in df.iterrows():
            records.append({
                'source_id': 'akshare_macro_usa',
                'country': 'usa',
                'indicator_type': 'cpi',
                'indicator_name': str(row.iloc[0]) if len(row) > 0 else '美国CPI月率',
                'period': 'monthly',
                'date': parse_date(row.iloc[1] if len(row) > 1 else None),
                'value': float(row.iloc[2]) if len(row) > 2 and row.iloc[2] is not None else None,
                'forecast_value': float(row.iloc[3]) if len(row) > 3 and row.iloc[3] is not None else None,
                'previous_value': float(row.iloc[4]) if len(row) > 4 and row.iloc[4] is not None else None,
            })
        logger.info(f"collect_macro_usa_cpi: {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"collect_macro_usa_cpi failed: {e}")
        return []

def persist_macro_data(records: List[Dict]) -> int:
    """持久化宏观经济数据到PostgreSQL"""
    if not records:
        return 0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    inserted = 0
    for rec in records:
        try:
            cursor.execute("""
                INSERT INTO akshare_macro_data 
                (source_id, country, indicator_type, indicator_name, period, date, value, forecast_value, previous_value)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_id, country, indicator_type, date) 
                DO UPDATE SET
                    value = EXCLUDED.value,
                    forecast_value = EXCLUDED.forecast_value,
                    previous_value = EXCLUDED.previous_value,
                    indicator_name = EXCLUDED.indicator_name
            """, (
                rec['source_id'], rec['country'], rec['indicator_type'],
                rec.get('indicator_name'), rec.get('period'),
                rec['date'], rec.get('value'), rec.get('forecast_value'), rec.get('previous_value')
            ))
            inserted += 1
        except Exception as e:
            logger.error(f"Insert failed for {rec.get('date')}: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    return inserted

def main():
    logger.info("=== Starting Macro Economic Data Collection ===")
    total_inserted = 0
    
    # 采集中国宏观经济数据
    china_collectors = [
        ('GDP', collect_macro_china_gdp),
        ('PPI', collect_macro_china_ppi),
        ('CPI', collect_macro_china_cpi),
        ('MoneySupply', collect_macro_china_money_supply),
    ]
    
    for name, collector in china_collectors:
        records = collector()
        if records:
            inserted = persist_macro_data(records)
            total_inserted += inserted
            logger.info(f"  {name}: collected {len(records)}, inserted {inserted}")
        time.sleep(1)  # 避免请求过快
    
    # 采集美国宏观经济数据
    usa_collectors = [
        ('USAGDP', collect_macro_usa_gdp),
        ('USACPI', collect_macro_usa_cpi),
    ]
    
    for name, collector in usa_collectors:
        records = collector()
        if records:
            inserted = persist_macro_data(records)
            total_inserted += inserted
            logger.info(f"  {name}: collected {len(records)}, inserted {inserted}")
        time.sleep(1)
    
    logger.info(f"=== Macro Data Collection Complete: {total_inserted} total records ===")
    
    # 验证
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM akshare_macro_data;")
    count = cursor.fetchone()[0]
    cursor.execute("SELECT country, indicator_type, COUNT(*) FROM akshare_macro_data GROUP BY country, indicator_type ORDER BY country, indicator_type;")
    stats = cursor.fetchall()
    cursor.close()
    conn.close()
    
    logger.info(f"Total records in DB: {count}")
    for stat in stats:
        logger.info(f"  {stat[0]} - {stat[1]}: {stat[2]}")

if __name__ == '__main__':
    main()
