#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复数据库中的错误数据"""

import sys
sys.path.insert(0, '.')
from src.gateway.web.postgresql_persistence import get_db_connection

conn = get_db_connection()
cursor = conn.cursor()

# 删除2026年的错误数据
cursor.execute("DELETE FROM akshare_stock_data WHERE date >= '2026-01-01'")
deleted = cursor.rowcount
conn.commit()
print(f'删除了 {deleted} 条错误数据')

cursor.close()
conn.close()
