#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成测试环境设置脚本
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def start_docker_services():
    """启动Docker服务"""
    print("🐳 启动Docker服务...")

    try:
        # 检查docker是否可用
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Docker不可用，跳过Docker服务启动")
            return False

        # 启动测试服务
        compose_file = 'docker-compose.test.yml'
        if os.path.exists(compose_file):
            result = subprocess.run([
                'docker-compose', '-f', compose_file, 'up', '-d'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Docker服务启动成功")
                # 等待服务就绪
                time.sleep(10)
                return True
            else:
                print(f"❌ Docker服务启动失败: {result.stderr}")
                return False
        else:
            print(f"⚠️  Docker Compose文件不存在: {compose_file}")
            return False

    except Exception as e:
        print(f"❌ Docker服务启动异常: {e}")
        return False


def setup_test_data():
    """设置测试数据"""
    print("📊 设置测试数据...")

    test_data_dir = Path('test_data')
    test_data_dir.mkdir(exist_ok=True)

    # 创建测试配置文件
    config_content = """
# 测试配置文件
[database]
host = localhost
port = 5432
name = testdb
user = postgres
password = testpassword

[redis]
host = localhost
port = 6379
db = 1

[api]
host = localhost
port = 8080
debug = true

[logging]
level = INFO
file = test.log
"""

    config_file = test_data_dir / 'test_config.ini'
    with open(config_file, 'w') as f:
        f.write(config_content)

    # 创建测试数据文件
    import json

    test_market_data = {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
        'base_prices': {
            'AAPL': 150.50,
            'GOOGL': 2800.00,
            'MSFT': 300.25,
            'AMZN': 3300.75,
            'TSLA': 250.80
        },
        'volatility': 0.02,
        'data_points': 1000
    }

    market_data_file = test_data_dir / 'market_data.json'
    with open(market_data_file, 'w') as f:
        json.dump(test_market_data, f, indent=2)

    print("✅ 测试数据设置完成")


def setup_test_services():
    """设置测试服务"""
    print("🔧 设置测试服务...")

    # 设置环境变量
    env_vars = {
        'TESTING': 'true',
        'DATABASE_URL': 'postgresql://postgres:testpassword@localhost:5432/testdb',
        'REDIS_URL': 'redis://localhost:6379/1',
        'API_HOST': 'localhost',
        'API_PORT': '8080',
        'LOG_LEVEL': 'INFO',
        'SECRET_KEY': 'test_secret_key_for_integration_tests',
        'JWT_SECRET': 'test_jwt_secret',
        'CACHE_TYPE': 'redis',
        'METRICS_ENABLED': 'true'
    }

    # 写入环境变量文件
    env_file = Path('.env.test')
    with open(env_file, 'w') as f:
        for key, value in env_vars.items():
            f.write(f'{key}={value}\n')

    print("✅ 测试服务配置完成")


def wait_for_services():
    """等待服务就绪"""
    print("⏳ 等待服务就绪...")

    services = [
        ('PostgreSQL', 'localhost', 5432, check_postgres),
        ('Redis', 'localhost', 6379, check_redis),
        ('API', 'localhost', 8080, check_http_service)
    ]

    max_attempts = 30
    attempt = 0

    while attempt < max_attempts:
        all_ready = True

        for service_name, host, port, check_func in services:
            if not check_func(host, port):
                all_ready = False
                print(f"⏳ 等待 {service_name} ({host}:{port})...")
                break

        if all_ready:
            print("✅ 所有服务已就绪")
            return True

        attempt += 1
        time.sleep(2)

    print("❌ 服务启动超时")
    return False


def check_postgres(host, port):
    """检查PostgreSQL连接"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=host,
            port=port,
            user='postgres',
            password='testpassword',
            database='testdb',
            connect_timeout=5
        )
        conn.close()
        return True
    except:
        return False


def check_redis(host, port):
    """检查Redis连接"""
    try:
        import redis
        client = redis.Redis(host=host, port=port, db=1, socket_timeout=5)
        return client.ping()
    except:
        return False


def check_http_service(host, port):
    """检查HTTP服务"""
    try:
        import requests
        response = requests.get(f'http://{host}:{port}/health', timeout=5)
        return response.status_code == 200
    except:
        return False


def setup_integration_test_env():
    """设置集成测试环境"""
    print("🚀 开始设置集成测试环境...")

    # 启动Docker服务
    docker_started = start_docker_services()

    # 设置测试数据
    setup_test_data()

    # 设置测试服务
    setup_test_services()

    # 等待服务就绪
    if docker_started:
        services_ready = wait_for_services()
        if not services_ready:
            print("⚠️  服务未能完全就绪，但继续执行测试")
    else:
        print("ℹ️  Docker服务未启动，使用模拟服务")

    # 创建集成测试配置文件
    integration_config = {
        'test_mode': 'integration',
        'database_url': 'postgresql://postgres:testpassword@localhost:5432/testdb',
        'redis_url': 'redis://localhost:6379/1',
        'api_base_url': 'http://localhost:8080',
        'test_data_dir': 'test_data',
        'log_file': 'integration_test.log',
        'timeout': 30,
        'retries': 3
    }

    import json
    config_file = Path('integration_test_config.json')
    with open(config_file, 'w') as f:
        json.dump(integration_config, f, indent=2)

    print("✅ 集成测试环境设置完成")
    print("📋 环境配置:")
    print(f"   数据库: {integration_config['database_url']}")
    print(f"   Redis: {integration_config['redis_url']}")
    print(f"   API: {integration_config['api_base_url']}")
    print(f"   配置: {config_file}")


def main():
    """主函数"""
    try:
        setup_integration_test_env()
        print("🎉 集成测试环境准备就绪！")
    except Exception as e:
        print(f"❌ 集成测试环境设置失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
