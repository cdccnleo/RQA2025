#!/usr/bin/env python3
"""等待并检查baostock数据采集"""
import requests
import time

print('=== 等待baostock数据采集 ===')

# 等待自动采集服务检查
check_interval = 10  # 每10秒检查一次
total_wait = 120     # 总共等待120秒

for i in range(0, total_wait, check_interval):
    print(f'\n检查 #{i//check_interval + 1} (已等待 {i} 秒)')
    print('-' * 60)
    
    try:
        # 检查自动采集状态
        response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
        data = response.json()
        
        if data.get('success'):
            auto_data = data.get('data', {})
            print(f"自动采集运行中: {auto_data.get('running', False)}")
            print(f"总检查次数: {auto_data.get('total_checks', 0)}")
            print(f"已提交任务: {auto_data.get('tasks_submitted', 0)}")
            print(f"已检查数据源: {auto_data.get('sources_checked', 0)}")
            
            # 如果任务已提交，检查任务状态
            if auto_data.get('tasks_submitted', 0) > 0:
                print("\n✅ 任务已提交！检查任务执行状态...")
                
                # 检查调度器任务
                response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
                data = response.json()
                
                unified_info = data.get('unified_scheduler', {})
                print(f"\n调度器任务统计:")
                print(f"  总任务数: {unified_info.get('total_tasks', 0)}")
                print(f"  待处理任务: {unified_info.get('pending_tasks', 0)}")
                print(f"  运行中任务: {unified_info.get('running_tasks', 0)}")
                print(f"  已完成任务: {unified_info.get('completed_tasks', 0)}")
                
                # 检查baostock数据源状态
                response = requests.get('http://localhost:8000/api/v1/data/sources', timeout=10)
                sources = response.json()
                
                for source in sources.get('data', []):
                    if 'baostock' in source.get('id', '').lower():
                        print(f"\nbaostock数据源状态:")
                        print(f"  启用: {source.get('is_active', False)}")
                        print(f"  状态: {source.get('status', '未知')}")
                        print(f"  最后采集时间: {source.get('last_collection_time') or '从未采集'}")
                        
                        if source.get('last_collection_time'):
                            print("\n🎉 baostock数据采集成功！")
                            print('=' * 60)
                            exit(0)
                        break
        
    except Exception as e:
        print(f"检查失败: {e}")
    
    if i < total_wait - check_interval:
        print(f"\n等待 {check_interval} 秒后再次检查...")
        time.sleep(check_interval)

print('\n' + '=' * 60)
print('等待完成，但未检测到采集完成')
print('=' * 60)
