#!/usr/bin/env python3
"""
测试演示工作进程心跳功能
"""

import requests
import json
import time
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_worker_heartbeat():
    """测试工作进程心跳"""
    print("🔍 测试工作进程心跳功能...")

    # 先注册一个演示工作进程
    worker_id = f"test_demo_worker_{int(time.time())}"

    # 注册工作进程
    register_url = "http://localhost/api/v1/monitoring/historical-collection/workers/register"
    register_data = {
        "worker_id": worker_id,
        "max_concurrent": 2
    }

    print(f"📝 注册演示工作进程: {worker_id}")

    try:
        response = requests.post(register_url, json=register_data, timeout=10)

        if response.status_code != 200:
            print(f"❌ 注册失败: HTTP {response.status_code}")
            print(f"错误详情: {response.text}")
            return False

        result = response.json()
        if not result.get('success'):
            print(f"❌ 注册失败: {result}")
            return False

        print("✅ 工作进程注册成功")
    except Exception as e:
        print(f"❌ 注册异常: {e}")
        return False

    # 发送心跳
    heartbeat_url = f"http://localhost/api/v1/monitoring/historical-collection/workers/{worker_id}/heartbeat"

    print("💓 发送心跳信号...")

    try:
        response = requests.post(heartbeat_url, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("✅ 心跳发送成功")
            print(f"响应: {result}")
        else:
            print(f"❌ 心跳发送失败: HTTP {response.status_code}")
            print(f"错误详情: {response.text}")
            return False

    except Exception as e:
        print(f"❌ 心跳异常: {e}")
        return False

    # 检查工作进程状态
    print("🔍 检查工作进程状态...")

    try:
        response = requests.get("http://localhost/api/v1/monitoring/historical-collection/workers", timeout=10)

        if response.status_code == 200:
            result = response.json()
            workers = result.get('workers', [])

            # 查找我们的测试工作进程
            test_worker = None
            for worker in workers:
                if worker.get('worker_id') == worker_id:
                    test_worker = worker
                    break

            if test_worker:
                last_heartbeat = test_worker.get('last_heartbeat', 0)
                heartbeat_time = time.strftime('%H:%M:%S', time.localtime(last_heartbeat))
                print("✅ 工作进程状态正常")
                print(f"   最后心跳时间: {heartbeat_time}")
                print(f"   活跃任务: {test_worker.get('active_tasks', [])}")
            else:
                print(f"❌ 工作进程 {worker_id} 未找到")
                return False
        else:
            print(f"❌ 获取工作进程列表失败: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 检查状态异常: {e}")
        return False

    # 等待一段时间，让心跳超时检查运行
    print("⏳ 等待心跳超时检查...")
    time.sleep(65)  # 等待超过60秒的超时时间

    # 再次检查工作进程状态，应该仍然存在（因为我们发送了心跳）
    print("🔍 再次检查工作进程状态（应该仍然活跃）...")

    try:
        response = requests.get("http://localhost/api/v1/monitoring/historical-collection/workers", timeout=10)

        if response.status_code == 200:
            result = response.json()
            workers = result.get('workers', [])

            test_worker = None
            for worker in workers:
                if worker.get('worker_id') == worker_id:
                    test_worker = worker
                    break

            if test_worker:
                print("✅ 工作进程仍然活跃，心跳机制正常")
                return True
            else:
                print(f"❌ 工作进程 {worker_id} 已被注销，心跳机制失败")
                return False
        else:
            print(f"❌ 获取工作进程列表失败: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ 检查状态异常: {e}")
        return False

def main():
    """主函数"""
    print("🚀 测试演示工作进程心跳功能")
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
    if test_worker_heartbeat():
        print("\n🎉 演示工作进程心跳测试通过！")
        return 0
    else:
        print("\n❌ 演示工作进程心跳测试失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n程序退出码: {exit_code}")
    sys.exit(exit_code)