#!/usr/bin/env python3
"""
Docker应用调试脚本

检查Docker容器中应用的状态和路由配置
"""

import requests
import json
import time
import subprocess
import sys
from pathlib import Path

def check_docker_containers():
    """检查Docker容器状态"""
    print("🔍 检查Docker容器状态...")

    try:
        result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'],
                              capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("Docker容器状态:")
            print(result.stdout)
        else:
            print(f"Docker命令失败: {result.stderr}")
    except Exception as e:
        print(f"无法检查Docker状态: {e}")

def test_container_connectivity():
    """测试容器连接性"""
    print("\n🔍 测试容器连接性...")

    # 测试nginx
    try:
        response = requests.get('http://localhost/health', timeout=5)
        print(f"nginx健康检查: {response.status_code}")
    except Exception as e:
        print(f"nginx连接失败: {e}")

    # 测试应用直接连接
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        print(f"应用直接连接: {response.status_code}")
    except Exception as e:
        print(f"应用直接连接失败: {e}")

def test_api_endpoints():
    """测试API端点"""
    print("\n🔍 测试API端点...")

    endpoints = [
        ('/api/v1/monitoring/data-collection/health', '数据采集健康检查'),
        ('/api/v1/monitoring/historical-collection/status', '历史数据采集状态'),
        ('/api/v1/monitoring/historical-collection/scheduler/status', '调度器状态'),
    ]

    for endpoint, description in endpoints:
        try:
            response = requests.get(f'http://localhost{endpoint}', timeout=10)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"{status} {description}: {response.status_code}")
        except Exception as e:
            print(f"❌ {description}: 连接失败 - {e}")

def check_application_logs():
    """检查应用日志"""
    print("\n🔍 检查应用日志...")

    try:
        # 获取应用容器的日志
        result = subprocess.run(['docker', 'logs', '--tail', '50', 'rqa2025-app'],
                              capture_output=True, text=True, timeout=15)

        if result.returncode == 0:
            print("应用容器最近日志:")
            # 只显示包含历史数据采集的关键日志
            lines = result.stdout.split('\n')
            relevant_lines = [line for line in lines if 'historical' in line.lower() or 'scheduler' in line.lower() or '404' in line]

            if relevant_lines:
                for line in relevant_lines[-10:]:  # 显示最后10行相关日志
                    print(f"  {line}")
            else:
                print("  未找到相关日志")
        else:
            print(f"获取应用日志失败: {result.stderr}")

    except Exception as e:
        print(f"检查应用日志失败: {e}")

def check_nginx_logs():
    """检查nginx日志"""
    print("\n🔍 检查nginx日志...")

    try:
        # 获取nginx容器的日志
        result = subprocess.run(['docker', 'logs', '--tail', '20', 'rqa2025-nginx'],
                              capture_output=True, text=True, timeout=15)

        if result.returncode == 0:
            print("nginx容器最近日志:")
            lines = result.stdout.split('\n')
            # 显示最近的错误或相关日志
            for line in lines[-10:]:
                if line.strip():
                    print(f"  {line}")
        else:
            print(f"获取nginx日志失败: {result.stderr}")

    except Exception as e:
        print(f"检查nginx日志失败: {e}")

def diagnose_issue():
    """诊断问题"""
    print("\n🔍 问题诊断...")

    # 检查应用是否在容器中正常运行
    try:
        response = requests.get('http://localhost:8000/docs', timeout=5)
        if response.status_code == 200:
            print("✅ 应用在容器中正常运行")
        else:
            print(f"⚠️ 应用响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 应用在容器中无法访问: {e}")
        return

    # 检查路由注册
    print("\n检查路由注册情况...")
    test_api_endpoints()

    print("\n可能的解决方案:")
    print("1. 检查Docker容器中的应用启动日志")
    print("2. 确认历史数据采集模块正确导入")
    print("3. 检查应用配置和环境变量")
    print("4. 重启应用容器")

def main():
    """主函数"""
    print("🚀 RQA2025 Docker应用调试")
    print("=" * 50)

    check_docker_containers()
    test_container_connectivity()
    test_api_endpoints()
    check_application_logs()
    check_nginx_logs()
    diagnose_issue()

    print("\n" + "=" * 50)
    print("调试完成。如问题持续存在，请检查:")
    print("- Docker容器日志: docker logs rqa2025-app")
    print("- 应用启动脚本: scripts/start_api_server.py")
    print("- 路由配置: src/gateway/web/api.py")

if __name__ == "__main__":
    main()