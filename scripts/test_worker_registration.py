#!/usr/bin/env python3
"""
测试工作进程注册API
"""

import requests
import json
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_worker_registration():
    """测试工作进程注册"""
    print("🔍 测试工作进程注册API...")

    url = "http://localhost/api/v1/monitoring/historical-collection/scheduler/workers/register"

    # 测试数据
    data = {
        "worker_id": "test_worker_123",
        "host": "localhost",
        "port": 8080,
        "capabilities": ["historical_data"],
        "max_concurrent": 2
    }

    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2)}")

    try:
        response = requests.post(url, json=data, timeout=10)

        print(f"响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")

        if response.status_code == 422:
            # 422错误通常包含详细的验证错误信息
            try:
                error_data = response.json()
                print(f"错误详情: {json.dumps(error_data, indent=2)}")
            except:
                print(f"错误内容: {response.text}")

        elif response.status_code == 200:
            result = response.json()
            print(f"成功响应: {json.dumps(result, indent=2)}")

        else:
            print(f"其他错误: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")

def test_simple_registration():
    """测试简化版本的注册"""
    print("\n🔍 测试简化版本注册API...")

    # 尝试使用另一个路由
    url = "http://localhost/api/v1/monitoring/historical-collection/workers/register"

    data = {
        "worker_id": "simple_worker_123",
        "max_concurrent": 2
    }

    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(data, indent=2)}")

    try:
        response = requests.post(url, json=data, timeout=10)

        print(f"响应状态码: {response.status_code}")

        if response.status_code == 422:
            try:
                error_data = response.json()
                print(f"错误详情: {json.dumps(error_data, indent=2)}")
            except:
                print(f"错误内容: {response.text}")
        elif response.status_code == 200:
            result = response.json()
            print(f"成功响应: {json.dumps(result, indent=2)}")
        else:
            print(f"其他错误: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")

def main():
    """主函数"""
    print("🚀 测试工作进程注册API")
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

    # 测试两种注册方式
    test_worker_registration()
    test_simple_registration()

    return 0

if __name__ == "__main__":
    exit_code = main()
    print(f"\n程序退出码: {exit_code}")
    sys.exit(exit_code)