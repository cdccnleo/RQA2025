#!/usr/bin/env python3
"""
健康检查问题诊断脚本
用于诊断数据采集调度器导致的健康检查失败问题
"""

import asyncio
import time
import requests
import psutil
import threading
from datetime import datetime

def monitor_system_resources():
    """监控系统资源使用情况"""
    print("🔍 开始系统资源监控...")

    try:
        while True:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            print(f"[{datetime.now().strftime('%H:%M:%S')}] CPU: {cpu_percent:.1f}%, 内存: {memory_percent:.1f}%")

            if cpu_percent > 60 or memory_percent > 70:
                print(f"⚠️  系统负载过高! CPU: {cpu_percent:.1f}%, 内存: {memory_percent:.1f}%")

            time.sleep(10)

    except KeyboardInterrupt:
        print("监控已停止")

def test_health_check_endpoint():
    """测试健康检查端点响应"""
    print("🏥 开始健康检查测试...")

    health_url = "http://localhost:8000/health"

    try:
        start_time = time.time()
        response = requests.get(health_url, timeout=15)
        response_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')

            print(f"✅ 健康检查成功 - 状态: {status}, 响应时间: {response_time:.2f}s")

            # 检查是否有警告
            warnings = data.get('warnings', [])
            if warnings:
                print(f"⚠️  健康检查警告: {warnings}")

            # 检查数据采集状态
            collection = data.get('data_collection', {})
            active_tasks = collection.get('active_tasks', 0)
            if active_tasks > 0:
                print(f"📊 数据采集活动: {active_tasks} 个活跃任务")

            return True

        elif response.status_code == 503:
            print("❌ 健康检查失败 - 服务不可用")
            error_data = response.json()
            print(f"错误详情: {error_data}")
            return False

        else:
            print(f"❌ 健康检查失败 - HTTP {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        print("❌ 健康检查超时 - 响应超过15秒")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 健康检查失败 - 连接错误")
        return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def check_data_collection_scheduler():
    """检查数据采集调度器状态"""
    print("📅 检查数据采集调度器状态...")

    try:
        # 这里可以添加对调度器状态的检查
        # 由于无法直接访问内部状态，我们通过健康检查端点获取信息

        health_url = "http://localhost:8000/health"
        response = requests.get(health_url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            collection = data.get('data_collection', {})

            scheduler_running = collection.get('scheduler_running', False)
            active_tasks = collection.get('active_tasks', 0)
            pending_tasks = collection.get('pending_tasks', 0)
            status = collection.get('status', 'unknown')

            print(f"调度器运行状态: {'✅ 运行中' if scheduler_running else '❌ 已停止'}")
            print(f"活跃任务数量: {active_tasks}")
            print(f"待处理任务数量: {pending_tasks}")
            print(f"数据采集状态: {status}")

            if active_tasks > 2:
                print("⚠️  活跃任务过多，可能影响系统性能")
            if pending_tasks > 5:
                print("⚠️  待处理任务过多，可能存在队列积压")

    except Exception as e:
        print(f"无法检查调度器状态: {e}")

def main():
    """主函数"""
    print("🔧 RQA2025 健康检查问题诊断工具")
    print("=" * 50)

    # 启动系统资源监控线程
    monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
    monitor_thread.start()

    print("\n1. 初始健康检查测试...")
    initial_healthy = test_health_check_endpoint()

    print("\n2. 数据采集调度器状态检查...")
    check_data_collection_scheduler()

    print("\n3. 连续健康检查监控...")
    print("按 Ctrl+C 停止监控")
    print("-" * 30)

    healthy_count = 0
    total_count = 0

    try:
        while True:
            total_count += 1
            if test_health_check_endpoint():
                healthy_count += 1

            # 每10次检查输出一次统计信息
            if total_count % 10 == 0:
                success_rate = (healthy_count / total_count) * 100
                print(".1f")

            time.sleep(30)  # 每30秒检查一次

    except KeyboardInterrupt:
        print("\n\n监控已停止")
        print("=" * 50)
        print("📊 最终统计:"        success_rate = (healthy_count / total_count) * 100 if total_count > 0 else 0
        print(f"总检查次数: {total_count}")
        print(".1f")

        if success_rate > 95:
            print("✅ 系统健康状况良好")
        elif success_rate > 80:
            print("⚠️  系统存在轻微问题")
        else:
            print("❌ 系统存在严重问题，建议检查配置")

        print("\n🔧 建议解决方案:")
        if success_rate < 95:
            print("1. 检查数据采集调度器的并发任务数量")
            print("2. 调整健康检查的超时时间")
            print("3. 监控系统资源使用情况")
            print("4. 考虑降低数据采集频率")

if __name__ == "__main__":
    main()