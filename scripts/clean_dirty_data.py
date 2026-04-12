import psycopg2
import os

conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST', 'postgres'),
    port=5432,
    user='rqa2025_admin',
    password=os.getenv('POSTGRES_PASSWORD', 'RQA2025_Postgres_Secure_2026'),
    database='rqa2025_prod'
)
cur = conn.cursor()

# Clean bond dirty data (data before 2020 or zero prices)
cur.execute("DELETE FROM akshare_bond_data WHERE date < '2020-01-01' OR close_price = 0 OR close_price IS NULL")
deleted_bond = cur.rowcount

# Clean forex dirty data
cur.execute("DELETE FROM akshare_forex_data WHERE date < '2020-01-01' OR close_price = 0 OR close_price IS NULL")
deleted_forex = cur.rowcount

conn.commit()
print(f"Deleted bond records: {deleted_bond}")
print(f"Deleted forex records: {deleted_forex}")

# Verify remaining
cur.execute("SELECT COUNT(*) FROM akshare_bond_data")
print(f"Remaining bond records: {cur.fetchone()[0]}")
cur.execute("SELECT COUNT(*) FROM akshare_forex_data")
print(f"Remaining forex records: {cur.fetchone()[0]}")

# Show sample of remaining
cur.execute("SELECT date, close_price, yield_to_maturity FROM akshare_bond_data ORDER BY date DESC LIMIT 5")
print("\nBond sample:")
for row in cur.fetchall():
    print(row)

cur.close()
conn.close()
