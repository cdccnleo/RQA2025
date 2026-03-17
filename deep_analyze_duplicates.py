#!/usr/bin/env python3
"""
深入分析重复特征提取任务
"""
import os
os.environ.setdefault('ENVIRONMENT', 'production')

print('=== 深入分析重复特征提取任务 ===')

try:
    from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection
    
    conn = get_db_connection()
    if not conn:
        print('❌ 无法连接到PostgreSQL')
        exit(1)
    
    cursor = conn.cursor()
    
    # 1. 总体统计
    print('\n📊 总体统计:')
    cursor.execute("SELECT COUNT(*) FROM feature_engineering_tasks")
    total = cursor.fetchone()[0]
    print(f'   总任务数: {total}')
    
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM feature_engineering_tasks WHERE symbol IS NOT NULL")
    unique_symbols = cursor.fetchone()[0]
    print(f'   不同股票数: {unique_symbols}')
    print(f'   平均每只股票任务数: {total / unique_symbols:.1f}')
    
    # 2. 详细分析重复情况（按 symbol + data_source + start_date + end_date）
    print('\n📊 按 symbol + data_source + start_date + end_date 分组统计:')
    cursor.execute("""
        SELECT symbol, data_source, start_date, end_date, COUNT(*) as count,
               ARRAY_AGG(task_id ORDER BY created_at) as task_ids,
               ARRAY_AGG(created_at ORDER BY created_at) as created_times
        FROM feature_engineering_tasks
        WHERE symbol IS NOT NULL AND data_source IS NOT NULL
        GROUP BY symbol, data_source, start_date, end_date
        HAVING COUNT(*) > 1
        ORDER BY count DESC, symbol
        LIMIT 10
    """)
    
    for row in cursor.fetchall():
        symbol = row[0]
        data_source = row[1]
        start_date = row[2]
        end_date = row[3]
        count = row[4]
        task_ids = row[5]
        created_times = row[6]
        
        print(f'\n   {symbol} | {data_source} | {start_date} ~ {end_date}')
        print(f'   任务数: {count}')
        
        # 计算任务创建间隔
        if len(created_times) > 1:
            intervals = []
            for i in range(1, len(created_times)):
                delta = (created_times[i] - created_times[i-1]).total_seconds()
                intervals.append(delta)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                print(f'   平均创建间隔: {avg_interval:.1f} 秒')
                print(f'   创建时间:')
                for i, t in enumerate(created_times[:5]):
                    print(f'      {t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}')
                if len(created_times) > 5:
                    print(f'      ... 还有 {len(created_times) - 5} 个')
    
    # 3. 检查任务创建时间分布
    print('\n📊 任务创建时间分布（按小时）:')
    cursor.execute("""
        SELECT 
            DATE_TRUNC('hour', created_at) as hour,
            COUNT(*) as count,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM feature_engineering_tasks
        WHERE created_at > NOW() - INTERVAL '24 hours'
        GROUP BY DATE_TRUNC('hour', created_at)
        ORDER BY hour DESC
        LIMIT 10
    """)
    
    for row in cursor.fetchall():
        hour = row[0]
        count = row[1]
        unique_symbols = row[2]
        print(f'   {hour.strftime("%Y-%m-%d %H:%M")}: {count} 个任务, {unique_symbols} 只股票')
    
    # 4. 检查特定股票的详细任务信息
    print('\n📊 特定股票（002837）的任务详情:')
    cursor.execute("""
        SELECT task_id, status, created_at, start_date, end_date, config
        FROM feature_engineering_tasks
        WHERE symbol = '002837'
        ORDER BY created_at
    """)
    
    rows = cursor.fetchall()
    print(f'   总任务数: {len(rows)}')
    
    # 按日期范围分组
    from collections import defaultdict
    date_range_groups = defaultdict(list)
    
    for row in rows:
        task_id = row[0]
        status = row[1]
        created_at = row[2]
        start_date = row[3]
        end_date = row[4]
        config = row[5]
        
        key = f"{start_date}~{end_date}"
        date_range_groups[key].append({
            'task_id': task_id,
            'status': status,
            'created_at': created_at
        })
    
    print('\n   按日期范围分组:')
    for date_range, tasks in sorted(date_range_groups.items()):
        print(f'   {date_range}: {len(tasks)} 个任务')
        for task in tasks[:3]:
            print(f'      - {task["created_at"].strftime("%H:%M:%S.%f")[:-3]} | {task["status"]} | {task["task_id"][:30]}...')
        if len(tasks) > 3:
            print(f'      ... 还有 {len(tasks) - 3} 个')
    
    cursor.close()
    return_db_connection(conn)
    
    print('\n=== 分析完成 ===')
    
except Exception as e:
    print(f'❌ 分析失败: {e}')
    import traceback
    traceback.print_exc()
