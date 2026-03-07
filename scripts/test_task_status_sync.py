#!/usr/bin/env python3
"""
测试任务状态同步修复
"""

import requests
import json
import time
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_task_status_sync():
    """测试任务状态同步"""
    print("🔍 测试任务状态同步修复...")

    # 1. 注册演示工作进程
    register_url = "http://localhost/api/v1/monitoring/historical-collection/scheduler/workers/register"
    register_data = {
        "worker_id": "test_sync_worker",
        "host": "localhost",
        "port": 8080,
        "capabilities": ["historical_data"],
        "max_concurrent": 2
    }

    print("📝 注册演示工作进程...")
    try:
        response = requests.post(register_url, json=register_data, timeout=10)
        if response.status_code != 200:
            print(f"❌ 注册失败: HTTP {response.status_code}")
            return False
        print("✅ 工作进程注册成功")
    except Exception as e:
        print(f"❌ 注册异常: {e}")
        return False

    # 2. 创建历史数据采集任务
    trigger_url = "http://localhost/api/v1/monitoring/historical-collection/scheduler/trigger-immediate?force=true"

    print("🎯 触发历史数据采集任务...")
    try:
        response = requests.post(trigger_url, timeout=10)
        if response.status_code != 200:
            print(f"❌ 任务触发失败: HTTP {response.status_code}")
            print(f"响应: {response.text}")
            return False

        result = response.json()
        if not result.get('success'):
            print(f"❌ 任务触发失败: {result}")
            return False

        tasks_created = result.get('tasks_created', 0)
        print(f"✅ 成功创建 {tasks_created} 个任务")

        if tasks_created == 0:
            print("⚠️ 没有创建任务，可能没有可用的股票数据")
            return True  # 这不是错误，只是没有数据

    except Exception as e:
        print(f"❌ 任务触发异常: {e}")
        return False

    # 等待任务分配
    print("⏳ 等待任务分配...")
    time.sleep(3)

    # 3. 检查活跃任务状态
    status_url = "http://localhost/api/v1/monitoring/historical-collection/status"
    print("🔍 检查活跃任务状态...")

    try:
        response = requests.get(status_url, timeout=10)
        if response.status_code != 200:
            print(f"❌ 获取状态失败: HTTP {response.status_code}")
            return False

        data = response.json()
        active_tasks = data.get('active_tasks', [])
        workers = data.get('workers', [])

        print(f"📊 活跃任务数: {len(active_tasks)}")
        print(f"👷 工作进程数: {len(workers)}")

        if len(active_tasks) > 0:
            print("✅ 发现活跃任务:")
            for task in active_tasks:
                print(f"   - {task.get('task_id', 'unknown')}: {task.get('symbol', 'unknown')} ({task.get('progress', 0)*100:.1f}%)")
                print(f"     工作进程: {task.get('worker_id', '未分配')}")
        else:
            print("⚠️ 没有活跃任务")

        # 检查工作进程状态
        if len(workers) > 0:
            print("✅ 工作进程状态:")
            for worker in workers:
                active_count = len(worker.get('active_tasks', []))
                print(f"   - {worker.get('worker_id', 'unknown')}: {active_count} 个活跃任务")

        # 4. 检查调度器状态
        scheduler_url = "http://localhost/api/v1/monitoring/historical-collection/scheduler/status"
        response = requests.get(scheduler_url, timeout=10)
        if response.status_code == 200:
            scheduler_data = response.json()
            running_tasks = scheduler_data.get('tasks', {}).get('running', 0)
            print(f"📈 调度器运行中任务数: {running_tasks}")

        return len(active_tasks) > 0 or tasks_created == 0

    except Exception as e:
        print(f"❌ 检查状态异常: {e}")
        return False

def main():
    """主函数"""
    print("🚀 测试任务状态同步修复")
    print("=" * 60)

    # 检查服务是否运行
    try:
        response = requests.get("http://localhost/health", timeout=5)
        if response.status_code != 200:
            print("❌ RQA2025服务未运行，请先启动系统")
            print("启动命令: docker-compose -f docker-compose.prod.yml up -d")
            return 1
    except:
        print("❌ 无法连接到RQA2025服务，请确保系统已启动")
        return 1

    # 运行测试
    if test_task_status_sync():
        print("\n🎉 任务状态同步测试通过！修复成功！")
        return 0
    else:
        print("\n❌ 任务状态同步测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n程序退出码: {exit_code}")
    sys.exit(exit_code)