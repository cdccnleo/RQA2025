#!/usr/bin/env python3
"""
RQA2025 健康检查脚本
"""

import requests
import time
import sys


def check_service_health():
    """检查服务健康状态"""
    services = {
        "API服务": "http://localhost:8000/health",
        "数据库": "http://localhost:5432",
        "Redis": "http://localhost:6379",
        "监控": "http://localhost:9090"
    }

    all_healthy = True

    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name}: 健康")
            else:
                print(f"❌ {service_name}: 异常 (状态码: {response.status_code})")
                all_healthy = False
        except Exception as e:
            print(f"❌ {service_name}: 无法连接 ({e})")
            all_healthy = False

    return all_healthy


if __name__ == "__main__":
    print("🏥 开始健康检查...")

    # 等待服务启动
    time.sleep(10)

    if check_service_health():
        print("🎉 所有服务健康！")
        sys.exit(0)
    else:
        print("❌ 部分服务异常！")
        sys.exit(1)
