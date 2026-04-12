#!/usr/bin/env python3
"""
中国国债收益率曲线数据采集脚本
从AkShare采集国债收益率曲线并持久化到PostgreSQL
支持2020年至今的历史数据
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

sys.path.insert(0, '/app')

import akshare as ak
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
        password=os.getenv('POSTGRES_PASSWORD', 'RQA2025_Postgres_Secure_2026'),
        database=os.getenv('POSTGRES_DB', 'rqa2025_prod')
    )


def validate_yield(value: Any) -> Optional[float]:
    """验证收益率数据有效性"""
    if value is None:
        return None
    try:
        v = float(value)
        # 合理的收益率范围: -5% to 20%
        if -5 <= v <= 20:
            return v
        return None
    except (TypeError, ValueError):
        return None


def get_curve_type(curve_name: str) -> str:
    """根据曲线名称判断类型"""
    if '国债' in curve_name:
        return 'treasury'
    elif '商业银行' in curve_name:
        return 'commercial_bank'
    elif '中短期' in curve_name or '票据' in curve_name:
        return 'corporate_aaa'
    return 'other'


def collect_bond_yield_range(start_date: str, end_date: str) -> List[Dict]:
    """采集指定日期范围的国债收益率数据"""
    try:
        df = ak.bond_china_yield(start_date=start_date, end_date=end_date)
        records = []
        
        for _, row in df.iterrows():
            date_val = row.get('日期')
            if date_val is None:
                continue
            
            # 解析日期
            if hasattr(date_val, 'isoformat'):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)[:10]
            
            curve_name = str(row.get('曲线名称', ''))
            curve_type = get_curve_type(curve_name)
            
            # 验证至少有一个有效收益率
            yields = {
                'yield_3m': validate_yield(row.get('3月')),
                'yield_6m': validate_yield(row.get('6月')),
                'yield_1y': validate_yield(row.get('1年')),
                'yield_3y': validate_yield(row.get('3年')),
                'yield_5y': validate_yield(row.get('5年')),
                'yield_7y': validate_yield(row.get('7年')),
                'yield_10y': validate_yield(row.get('10年')),
                'yield_30y': validate_yield(row.get('30年')),
            }
            
            # 至少需要一个有效收益率
            if any(v is not None for v in yields.values()):
                records.append({
                    'source_id': 'akshare_bond_yield',
                    'curve_name': curve_name,
                    'curve_type': curve_type,
                    'date': date_str,
                    **yields
                })
        
        return records
        
    except Exception as e:
        logger.error(f"采集失败 {start_date}-{end_date}: {e}")
        return []


def persist_bond_yield(records: List[Dict]) -> int:
    """持久化到PostgreSQL"""
    if not records:
        return 0
    
    conn = get_db_connection()
    cur = conn.cursor()
    inserted = 0
    
    for rec in records:
        try:
            cur.execute("""
                INSERT INTO akshare_bond_yield 
                (source_id, curve_name, curve_type, date, yield_3m, yield_6m, yield_1y, 
                 yield_3y, yield_5y, yield_7y, yield_10y, yield_30y)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_id, curve_name, date)
                DO UPDATE SET
                    yield_3m = EXCLUDED.yield_3m, yield_6m = EXCLUDED.yield_6m,
                    yield_1y = EXCLUDED.yield_1y, yield_3y = EXCLUDED.yield_3y,
                    yield_5y = EXCLUDED.yield_5y, yield_7y = EXCLUDED.yield_7y,
                    yield_10y = EXCLUDED.yield_10y, yield_30y = EXCLUDED.yield_30y
            """, (
                rec['source_id'], rec['curve_name'], rec['curve_type'], rec['date'],
                rec['yield_3m'], rec['yield_6m'], rec['yield_1y'],
                rec['yield_3y'], rec['yield_5y'], rec['yield_7y'], rec['yield_10y'], rec['yield_30y']
            ))
            inserted += 1
        except Exception as e:
            logger.error(f"Insert failed: {e}")
    
    conn.commit()
    cur.close()
    conn.close()
    return inserted


def main():
    logger.info("=== Starting Bond Yield Curve Collection ===")
    
    # 分批采集: 每次3个月跨度，避免数据量过大
    # 从2020-01到2026-04，每3个月一批
    start_year = 2020
    end_year = 2026
    end_month = 4
    
    total_records = 0
    current_date = datetime(start_year, 1, 1)
    end_dt = datetime(end_year, end_month, 1)
    
    batch_count = 0
    while current_date < end_dt:
        batch_start = current_date.strftime('%Y%m%d')
        # 3个月后
        next_month = current_date + timedelta(days=95)
        if next_month > end_dt:
            next_month = end_dt + timedelta(days=1)
        batch_end = next_month.strftime('%Y%m%d')
        
        batch_count += 1
        logger.info(f"Batch {batch_count}: {batch_start} to {batch_end}")
        
        records = collect_bond_yield_range(batch_start, batch_end)
        if records:
            inserted = persist_bond_yield(records)
            total_records += inserted
            logger.info(f"  -> {len(records)} records collected, {inserted} inserted")
        else:
            logger.warning(f"  -> No data retrieved")
        
        current_date = next_month
        time.sleep(1)  # 避免请求过快
    
    logger.info(f"=== Collection Complete: {total_records} total records ===")
    
    # 验证数据
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM akshare_bond_yield")
    total = cur.fetchone()[0]
    cur.execute("SELECT curve_name, COUNT(*) FROM akshare_bond_yield GROUP BY curve_name ORDER BY COUNT(*) DESC")
    stats = cur.fetchall()
    cur.execute("SELECT date, COUNT(*) FROM akshare_bond_yield GROUP BY date ORDER BY date DESC LIMIT 5")
    recent = cur.fetchall()
    cur.close()
    conn.close()
    
    logger.info(f"Total in DB: {total}")
    for s in stats:
        logger.info(f"  {s[0]}: {s[1]}")
    logger.info(f"Recent dates: {[r[0] for r in recent]}")


if __name__ == '__main__':
    main()
