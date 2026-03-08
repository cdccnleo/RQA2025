#!/usr/bin/env python3
"""验证Baostock数据采集"""
import requests
import time

print('=== 验证Baostock数据采集 ===')

# 等待应用启动
time.sleep(15)

# 1. 检查自动采集状态
print('\n1. 自动采集服务状态')
print('-' * 60)
try:
    response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
    data = response.json()
    
    if data.get('success'):
        auto_data = data.get('data', {})
        print(f"自动采集运行中: {auto_data.get('running', False)}")
        print(f"总检查次数: {auto_data.get('total_checks', 0)}")
        print(f"已提交任务: {auto_data.get('tasks_submitted', 0)}")
    else:
        print(f"获取失败: {data.get('detail', '未知错误')}")
except Exception as e:
    print(f"获取失败: {e}")

# 2. 等待任务执行
print('\n2. 等待任务执行（60秒）...')
print('-' * 60)
time.sleep(60)

# 3. 检查日志
print('\n3. 检查最近日志（数据采集相关）')
print('-' * 60)
print("请运行以下命令查看日志:")
print("docker-compose -f docker-compose.prod.yml logs --tail=100 app | grep -E 'Baostock|baostock|采集|collect'")

# 4. 检查数据库
print('\n4. 检查PostgreSQL数据库')
print('-' * 60)
print("请运行以下命令检查数据:")
print("docker-compose -f docker-compose.prod.yml exec postgres psql -U rqa2025 -d rqa2025 -c 'SELECT COUNT(*) FROM akshare_stock_data WHERE data_source = \"baostock\";'")

print('\n' + '=' * 60)
print('验证完成')
print('=' * 60)
