import psycopg2, os
conn = psycopg2.connect(host='postgres', port=5432, user='rqa2025_admin', password=os.getenv('POSTGRES_PASSWORD','RQA2025_Postgres_Secure_2026'), database='rqa2025_prod')
cur = conn.cursor()

# 检查债券数据
cur.execute('SELECT source_id, bond_code, bond_name, date, open_price, high_price, close_price, yield_to_maturity, change_ratio FROM akshare_bond_data LIMIT 5')
print('=== BOND DATA (sample) ===')
for row in cur.fetchall():
    print(row)

# 检查外汇数据
cur.execute('SELECT source_id, currency_pair, base_currency, quote_currency, date, open_price, close_price, change_ratio FROM akshare_forex_data LIMIT 5')
print('\n=== FOREX DATA (sample) ===')
for row in cur.fetchall():
    print(row)

# 统计
cur.execute("SELECT source_id, count(*) FROM akshare_bond_data GROUP BY source_id")
print('\n=== BOND counts:', cur.fetchall())
cur.execute("SELECT source_id, count(*) FROM akshare_forex_data GROUP BY source_id")
print('=== FOREX counts:', cur.fetchall())

cur.close()
conn.close()
