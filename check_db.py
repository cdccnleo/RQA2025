import sqlite3

conn = sqlite3.connect('data/coverage_monitor.db')
cursor = conn.cursor()

# 检查表
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print('Tables:', tables)

# 检查数据
cursor.execute('SELECT COUNT(*) FROM coverage_history')
count = cursor.fetchone()[0]
print('Coverage history records:', count)

if count > 0:
    cursor.execute('SELECT * FROM coverage_history LIMIT 5')
    rows = cursor.fetchall()
    print('Sample data:', rows)

conn.close()
