#!/usr/bin/env python3
"""采集2022-08至今的国债收益率数据"""
import os, sys, time, logging
from datetime import datetime, timedelta
sys.path.insert(0, '/app')
import akshare as ak
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

def validate(v):
    if v is None: return None
    try:
        f = float(v)
        return f if -5 <= f <= 20 else None
    except: return None

def curve_type(name):
    if '国债' in name: return 'treasury'
    if '商业银行' in name: return 'commercial_bank'
    return 'corporate_aaa'

def persist(records):
    if not records: return 0
    conn = get_db()
    cur = conn.cursor()
    n = 0
    for r in records:
        cur.execute("""
            INSERT INTO akshare_bond_yield 
            (source_id, curve_name, curve_type, date, yield_3m, yield_6m, yield_1y,
             yield_3y, yield_5y, yield_7y, yield_10y, yield_30y)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (source_id, curve_name, date) DO UPDATE SET
                yield_3m=EXCLUDED.yield_3m, yield_6m=EXCLUDED.yield_6m, yield_1y=EXCLUDED.yield_1y,
                yield_3y=EXCLUDED.yield_3y, yield_5y=EXCLUDED.yield_5y, yield_7y=EXCLUDED.yield_7y,
                yield_10y=EXCLUDED.yield_10y, yield_30y=EXCLUDED.yield_30y
        """, (r['s'], r['n'], r['t'], r['d'], r['y3m'], r['y6m'], r['y1y'],
              r['y3y'], r['y5y'], r['y7y'], r['y10y'], r['y30y']))
        n += 1
    conn.commit()
    cur.close(); conn.close()
    return n

# 分批: 2022-08 到 2026-04
batches = [
    ('20220801', '20230201'),
    ('20230201', '20230801'),
    ('20230801', '20240201'),
    ('20240201', '20240801'),
    ('20240801', '20250201'),
    ('20250201', '20260201'),
    ('20260201', '20260412'),
]

total = 0
for i, (s, e) in enumerate(batches):
    logger.info(f"Batch {i+1}: {s} -> {e}")
    try:
        df = ak.bond_china_yield(start_date=s, end_date=e)
        recs = []
        for _, row in df.iterrows():
            dv = row.get('日期')
            if dv is None: continue
            date_str = dv.strftime('%Y-%m-%d') if hasattr(dv, 'strftime') else str(dv)[:10]
            recs.append({
                's': 'akshare_bond_yield', 'n': str(row.get('曲线名称', '')),
                't': curve_type(str(row.get('曲线名称', ''))), 'd': date_str,
                'y3m': validate(row.get('3月')), 'y6m': validate(row.get('6月')),
                'y1y': validate(row.get('1年')), 'y3y': validate(row.get('3年')),
                'y5y': validate(row.get('5年')), 'y7y': validate(row.get('7年')),
                'y10y': validate(row.get('10年')), 'y30y': validate(row.get('30年')),
            })
        if recs:
            n = persist(recs)
            total += n
            logger.info(f"  -> {len(recs)} records, {n} inserted")
        else:
            logger.warning(f"  -> No data")
    except Exception as ex:
        logger.error(f"  -> Failed: {ex}")
    time.sleep(1)

logger.info(f"=== Done: {total} new records ===")

conn = get_db(); cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM akshare_bond_yield")
logger.info(f"Total in DB: {cur.fetchone()[0]}")
cur.execute("SELECT date, COUNT(*) FROM akshare_bond_yield GROUP BY date ORDER BY date DESC LIMIT 3")
logger.info(f"Latest dates: {[r[0] for r in cur.fetchall()]}")
cur.close(); conn.close()
