#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证数据库表结构"""

import os
import psycopg2

def main():
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
        user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
        password=os.getenv('POSTGRES_PASSWORD', '')
    )
    cur = conn.cursor()
    
    # 查询所有表
    cur.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name
    """)
    tables = cur.fetchall()
    
    print('=' * 60)
    print('数据库中的表:')
    print('=' * 60)
    for t in tables:
        print(f'  - {t[0]}')
    print(f'\n总计: {len(tables)} 个表')
    
    # 检查关键表
    key_tables = [
        'ml_models', 'ml_training_history', 'inference_cache', 'feature_cache',
        'strategy_configs', 'backtest_results', 'trading_orders', 'trading_positions',
        'risk_checks', 'risk_alerts', 'risk_metrics'
    ]
    
    existing = {t[0] for t in tables}
    missing = [t for t in key_tables if t not in existing]
    
    if missing:
        print(f'\n⚠️ 缺少的关键表: {missing}')
    else:
        print('\n✅ 所有关键表都已创建')
    
    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
