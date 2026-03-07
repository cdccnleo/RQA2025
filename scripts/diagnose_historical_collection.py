#!/usr/bin/env python3
"""
历史数据采集系统诊断脚本

检查调度器状态、工作进程状态和任务队列状态
"""

import requests
import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_scheduler_status():
    """检查调度器状态"""
    print("🔍 检查调度器状态...")
    try:
        response = requests.get("http://localhost/api/v1/monitoring/historical-collection/scheduler/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 调度器状态: {data.get('status', '未知')}")
            print(f"   📊 队列大小: {data.get('queue_size', 0)}")
            print(f"   👷 工作进程总数: {data.get('workers', {}).get('total', 0)}")
            print(f"   ⚡ 活跃工作进程: {data.get('workers', {}).get('active', 0)}")
            print(f"   📋 待处理任务: {data.get('tasks', {}).get('pending', 0)}")
            print(f"   🔄 运行中任务: {data.get('tasks', {}).get('running', 0)}")
            return data
        else:
            print(f"❌ 调度器状态检查失败: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 调度器状态检查异常: {e}")
        return None

def check_monitoring_status():
    """检查监控状态"""
    print("\n🔍 检查监控状态...")
    try:
        response = requests.get("http://localhost/api/v1/monitoring/historical-collection/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ 监控状态获取成功")
            print(f"   📊 活跃任务数: {len(data.get('active_tasks', []))}")
            print(f"   👷 工作进程数: {len(data.get('workers', []))}")
            return data
        else:
            print(f"❌ 监控状态检查失败: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 监控状态检查异常: {e}")
        return None

def check_workers_list():
    """检查工作进程列表"""
    print("\n🔍 检查工作进程列表...")
    try:
        response = requests.get("http://localhost/api/v1/monitoring/historical-collection/scheduler/workers", timeout=10)
        if response.status_code == 200:
            data = response.json()
            workers = data.get('workers', [])
            print(f"✅ 获取到 {len(workers)} 个工作进程")
            for worker in workers:
                print(f"   👷 {worker.get('worker_id', '未知')}: {worker.get('status', '未知')} (任务: {len(worker.get('active_tasks', []))})")
            return workers
        else:
            print(f"❌ 工作进程列表检查失败: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 工作进程列表检查异常: {e}")
        return []

def check_task_queue():
    """检查任务队列"""
    print("\n🔍 检查任务队列...")
    try:
        response = requests.get("http://localhost/api/v1/monitoring/historical-collection/scheduler/tasks/queue", timeout=10)
        if response.status_code == 200:
            data = response.json()
            queue_tasks = data.get('queue', [])
            print(f"✅ 队列中有 {len(queue_tasks)} 个任务")
            for task in queue_tasks[:5]:  # 只显示前5个
                print(f"   📋 {task.get('task_id', '未知')}: {task.get('symbol', '未知')} ({task.get('priority', 'normal')})")
            if len(queue_tasks) > 5:
                print(f"   ... 还有 {len(queue_tasks) - 5} 个任务")
            return queue_tasks
        else:
            print(f"❌ 任务队列检查失败: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ 任务队列检查异常: {e}")
        return []

def provide_solutions(scheduler_data, monitoring_data, workers, queue_tasks):
    """提供解决方案"""
    print("\n🔧 问题分析与解决方案:")

    issues = []
    solutions = []

    # 检查调度器状态
    if scheduler_data and scheduler_data.get('status') != 'running':
        issues.append("调度器未启动")
        solutions.append("请点击 '▶️ 启动调度器' 按钮启动调度器")

    # 检查工作进程
    if not workers:
        issues.append("没有注册的工作进程")
        solutions.append("请点击 '⚙️ 注册演示工作进程' 按钮注册工作进程")

    # 检查队列任务
    if queue_tasks and not workers:
        issues.append("有任务在队列中但没有工作进程执行")
        solutions.append("需要先注册工作进程，然后任务才能被分配执行")

    # 检查活跃任务
    active_tasks = monitoring_data.get('active_tasks', []) if monitoring_data else []
    if not active_tasks and queue_tasks and workers:
        issues.append("任务队列中有任务但没有活跃任务")
        solutions.append("可能需要等待调度器分配任务，或检查工作进程心跳状态")

    if not issues:
        print("✅ 系统状态正常")
        print("💡 如果仍有问题，请检查:")
        print("   - 工作进程是否能正常连接到系统")
        print("   - 防火墙设置是否阻止了工作进程通信")
        print("   - 系统资源是否充足")
    else:
        print("❌ 发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        print("\n✅ 建议解决方案:")
        for i, solution in enumerate(solutions, 1):
            print(f"   {i}. {solution}")

def main():
    """主函数"""
    print("🚀 RQA2025 历史数据采集系统诊断")
    print("=" * 60)

    # 检查服务是否运行
    try:
        response = requests.get("http://localhost/health", timeout=5)
        if response.status_code != 200:
            print("❌ RQA2025服务未运行，请先启动系统")
            return 1
    except:
        print("❌ 无法连接到RQA2025服务，请确保系统已启动")
        return 1

    # 执行各项检查
    scheduler_data = check_scheduler_status()
    monitoring_data = check_monitoring_status()
    workers = check_workers_list()
    queue_tasks = check_task_queue()

    # 提供解决方案
    provide_solutions(scheduler_data, monitoring_data, workers, queue_tasks)

    print("\n" + "=" * 60)
    print("📋 诊断完成")
    print("🔗 访问监控页面: http://localhost/data-collection-monitor.html")

    return 0

if __name__ == "__main__":
    exit_code = main()
    print(f"\n程序退出码: {exit_code}")
    sys.exit(exit_code)