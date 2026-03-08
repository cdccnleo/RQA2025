#!/usr/bin/env python3
"""检查统一调度器状态和数据采集情况"""
import requests
import json

def check_scheduler_status():
    print('=' * 70)
    print('🔍 统一调度器启动状态与数据采集检查')
    print('=' * 70)
    
    # 1. 检查调度器仪表盘
    print('\n📊 1. 统一调度器状态')
    print('-' * 70)
    try:
        response = requests.get('http://localhost:8000/api/v1/data/scheduler/dashboard', timeout=10)
        data = response.json()
        
        scheduler = data.get('scheduler', {})
        unified = data.get('unified_scheduler', {})
        
        # 调度器状态
        running = scheduler.get('running', False)
        print(f"  调度器运行状态: {'✅ 运行中' if running else '❌ 未运行'}")
        print(f"  运行时间: {scheduler.get('uptime', '未知')}")
        print(f"  调度器类型: {scheduler.get('scheduler_type', '未知')}")
        
        # 数据源状态
        active_sources = scheduler.get('active_sources', 0)
        total_sources = scheduler.get('total_sources', 0)
        print(f"\n  活跃数据源: {active_sources}")
        print(f"  总数据源: {total_sources}")
        print(f"  并发限制: {scheduler.get('concurrent_limit', 0)}")
        print(f"  活跃任务: {scheduler.get('active_tasks', 0)}")
        
        # 统一调度器详情
        is_running = unified.get('is_running', False)
        print(f"\n  统一调度器核心: {'✅ 运行中' if is_running else '❌ 未运行'}")
        print(f"  总任务数: {unified.get('total_tasks', 0)}")
        print(f"  待处理任务: {unified.get('pending_tasks', 0)}")
        print(f"  运行中任务: {unified.get('running_tasks', 0)}")
        print(f"  已完成任务: {unified.get('completed_tasks', 0)}")
        print(f"  失败任务: {unified.get('failed_tasks', 0)}")
        print(f"  数据采集器数量: {unified.get('data_collectors_count', 0)}")
        
        if unified.get('queue_sizes'):
            print(f"\n  队列大小:")
            for queue, size in unified.get('queue_sizes', {}).items():
                print(f"    {queue}: {size}")
        
    except Exception as e:
        print(f"  ❌ 获取失败: {e}")
        return False
    
    # 2. 检查自动采集状态
    print('\n🔄 2. 自动采集状态')
    print('-' * 70)
    try:
        response = requests.get('http://localhost:8000/api/v1/data/scheduler/auto-collection/status', timeout=10)
        data = response.json()
        
        if data.get('success'):
            auto_data = data.get('data', {})
            running = auto_data.get('running', False)
            print(f"  自动采集运行: {'✅ 运行中' if running else '⏸️ 已停止'}")
            print(f"  检查间隔: {auto_data.get('check_interval', 0)}秒")
            print(f"  总检查次数: {auto_data.get('total_checks', 0)}")
            print(f"  已提交任务: {auto_data.get('tasks_submitted', 0)}")
            print(f"  已检查数据源: {auto_data.get('sources_checked', 0)}")
            print(f"  上次检查: {auto_data.get('last_check_time') or '无'}")
            print(f"  下次检查: {auto_data.get('next_check_time') or '无'}")
            print(f"  待处理任务: {auto_data.get('pending_tasks_count', 0)}")
        else:
            print(f"  ⚠️ {data.get('detail', '未知错误')}")
    except Exception as e:
        print(f"  ❌ 获取失败: {e}")
    
    # 3. 检查数据源配置
    print('\n📡 3. 数据源配置')
    print('-' * 70)
    try:
        response = requests.get('http://localhost:8000/api/v1/data/sources', timeout=10)
        sources = response.json()
        
        total = len(sources)
        active = [s for s in sources if s.get('is_active', False)]
        inactive = [s for s in sources if not s.get('is_active', False)]
        
        print(f"  总数据源: {total}")
        print(f"  活跃数据源: {len(active)} {'✅' if len(active) > 0 else '⚠️'}")
        print(f"  非活跃数据源: {len(inactive)}")
        
        if active:
            print(f"\n  活跃数据源详情:")
            for source in active[:5]:
                name = source.get('name', 'N/A')
                source_type = source.get('source_type', 'N/A')
                last_time = source.get('last_collection_time') or '从未采集'
                print(f"    - {name} ({source_type})")
                print(f"      最后采集: {last_time}")
            
            if len(active) > 5:
                print(f"    ... 还有 {len(active) - 5} 个活跃数据源")
        else:
            print("\n  ⚠️ 没有活跃数据源！")
            print("     建议: 检查数据源配置或启用数据源")
        
    except Exception as e:
        print(f"  ❌ 获取失败: {e}")
    
    # 4. 检查最近的数据采集任务
    print('\n📈 4. 最近数据采集任务')
    print('-' * 70)
    try:
        # 尝试获取任务列表
        response = requests.get('http://localhost:8000/api/v1/data/scheduler/tasks', timeout=10)
        if response.status_code == 200:
            tasks = response.json()
            if tasks:
                print(f"  总任务数: {len(tasks)}")
                for task in tasks[:5]:
                    print(f"    - {task.get('id', 'N/A')}: {task.get('status', 'N/A')}")
            else:
                print("  暂无任务")
        else:
            print(f"  获取任务列表失败: HTTP {response.status_code}")
    except Exception as e:
        print(f"  无法获取任务列表: {e}")
    
    # 5. 总结
    print('\n' + '=' * 70)
    print('📋 检查总结')
    print('=' * 70)
    
    scheduler_running = scheduler.get('running', False) if 'scheduler' in locals() else False
    auto_collection_running = auto_data.get('running', False) if 'auto_data' in locals() else False
    active_count = len(active) if 'active' in locals() else 0
    
    print(f"\n  ✅ 统一调度器: {'运行中' if scheduler_running else '未运行'}")
    print(f"  {'✅' if auto_collection_running else '⏸️'} 自动采集: {'运行中' if auto_collection_running else '已停止'}")
    print(f"  {'✅' if active_count > 0 else '⚠️'} 活跃数据源: {active_count}个")
    
    if scheduler_running and active_count > 0:
        if not auto_collection_running:
            print("\n  💡 建议: 自动采集已停止，可以启动自动采集来开始数据采集")
            print("     启动命令: POST /api/v1/data/scheduler/auto-collection/start")
        else:
            print("\n  ✅ 系统状态正常，正在按配置进行数据采集")
    elif not scheduler_running:
        print("\n  ❌ 警告: 统一调度器未运行，无法执行数据采集任务")
    elif active_count == 0:
        print("\n  ⚠️ 警告: 没有活跃数据源，请配置并启用数据源")
    
    print('\n' + '=' * 70)

if __name__ == '__main__':
    check_scheduler_status()
