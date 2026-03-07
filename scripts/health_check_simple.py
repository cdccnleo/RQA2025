#!/usr/bin/env python3
"""
RQA2025 简化健康检查脚本
"""

import requests
from datetime import datetime


def check_service_health():
    """检查服务健康状态"""
    print("🔍 开始健康检查...")

    # 检查主应用服务
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ 主应用服务正常")
        else:
            print(f"⚠️ 主应用服务异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 主应用服务不可用: {e}")

    # 检查数据库连接
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="rqa2025",
            user="user",
            password="pass"
        )
        conn.close()
        print("✅ 数据库连接正常")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")

    # 检查Redis连接
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis连接正常")
    except Exception as e:
        print(f"❌ Redis连接失败: {e}")


def main():
    """主函数"""
    print("🚀 RQA2025 健康检查")
    print(f"⏰ 检查时间: {datetime.now()}")
    print("-" * 50)

    check_service_health()

    print("-" * 50)
    print("✅ 健康检查完成")


if __name__ == "__main__":
    main()
