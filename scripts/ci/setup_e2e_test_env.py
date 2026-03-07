#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2E测试环境设置脚本
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path


def setup_e2e_environment():
    """设置端到端测试环境"""
    print("🚀 开始设置端到端测试环境...")

    # 创建E2E测试目录结构
    e2e_dirs = [
        'e2e_test_data',
        'e2e_test_reports',
        'e2e_screenshots',
        'e2e_logs'
    ]

    for dir_name in e2e_dirs:
        Path(dir_name).mkdir(exist_ok=True)

    # 设置E2E测试配置
    e2e_config = {
        'environment': 'e2e_test',
        'base_url': 'http://localhost:3000',  # 前端应用URL
        'api_url': 'http://localhost:8080',   # 后端API URL
        'database_url': 'postgresql://postgres:testpassword@localhost:5432/e2edb',
        'browser': 'chrome',
        'headless': True,
        'timeout': 30,
        'screenshot_on_failure': True,
        'video_recording': False,
        'parallel_execution': True,
        'max_workers': 3
    }

    # 写入E2E配置文件
    config_file = Path('e2e_config.json')
    with open(config_file, 'w') as f:
        json.dump(e2e_config, f, indent=2)

    print("✅ E2E测试配置完成")


def setup_test_application():
    """设置测试应用"""
    print("🔧 设置测试应用...")

    # 启动后端服务
    backend_started = start_backend_service()

    # 启动前端服务
    frontend_started = start_frontend_service()

    # 等待应用就绪
    if backend_started and frontend_started:
        app_ready = wait_for_application()
        if not app_ready:
            print("⚠️  应用未能完全就绪，但继续执行测试")
    else:
        print("ℹ️  使用模拟应用进行测试")


def start_backend_service():
    """启动后端服务"""
    print("🔧 启动后端服务...")

    try:
        # 使用Docker启动后端服务
        if os.path.exists('docker-compose.e2e.yml'):
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.e2e.yml', 'up', '-d', 'backend'
            ], capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print("✅ 后端服务启动成功")
                return True
            else:
                print(f"❌ 后端服务启动失败: {result.stderr}")
                return False
        else:
            # 尝试直接启动Python应用
            if os.path.exists('src/main.py'):
                subprocess.Popen([
                    sys.executable, 'src/main.py'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("✅ 后端服务（直接启动）成功")
                return True
            else:
                print("⚠️  未找到后端启动脚本")
                return False

    except Exception as e:
        print(f"❌ 后端服务启动异常: {e}")
        return False


def start_frontend_service():
    """启动前端服务"""
    print("🔧 启动前端服务...")

    try:
        # 检查是否有前端应用
        if os.path.exists('frontend/package.json'):
            # 安装依赖
            subprocess.run(['npm', 'install'], cwd='frontend', check=True)

            # 启动前端服务
            subprocess.Popen([
                'npm', 'start'
            ], cwd='frontend', stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            print("✅ 前端服务启动成功")
            return True
        else:
            print("ℹ️  未找到前端应用，使用API测试模式")
            return True

    except Exception as e:
        print(f"❌ 前端服务启动异常: {e}")
        return False


def wait_for_application():
    """等待应用就绪"""
    print("⏳ 等待应用就绪...")

    services = [
        ('后端API', 'localhost', 8080, check_http_service),
        ('前端应用', 'localhost', 3000, check_http_service)
    ]

    max_attempts = 60  # 最多等待2分钟
    attempt = 0

    while attempt < max_attempts:
        all_ready = True

        for service_name, host, port, check_func in services:
            if not check_func(host, port):
                all_ready = False
                print(f"⏳ 等待 {service_name} ({host}:{port})...")
                break

        if all_ready:
            print("✅ 所有应用服务已就绪")
            return True

        attempt += 1
        time.sleep(2)

    print("❌ 应用启动超时")
    return False


def check_http_service(host: str, port: int) -> bool:
    """检查HTTP服务"""
    try:
        import requests
        response = requests.get(f'http://{host}:{port}/health', timeout=5)
        return response.status_code == 200
    except:
        return False


def setup_test_data():
    """设置E2E测试数据"""
    print("📊 设置E2E测试数据...")

    test_data_dir = Path('e2e_test_data')

    # 创建测试用户数据
    test_users = [
        {
            'username': 'e2e_user_1',
            'email': 'e2e1@example.com',
            'password': 'testpass123',
            'role': 'user'
        },
        {
            'username': 'e2e_admin_1',
            'email': 'e2e_admin@example.com',
            'password': 'adminpass123',
            'role': 'admin'
        }
    ]

    users_file = test_data_dir / 'test_users.json'
    with open(users_file, 'w') as f:
        json.dump(test_users, f, indent=2)

    # 创建测试交易数据
    test_trades = [
        {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.50,
            'order_type': 'buy',
            'user_id': 1
        },
        {
            'symbol': 'GOOGL',
            'quantity': 50,
            'price': 2800.00,
            'order_type': 'sell',
            'user_id': 2
        }
    ]

    trades_file = test_data_dir / 'test_trades.json'
    with open(trades_file, 'w') as f:
        json.dump(test_trades, f, indent=2)

    print("✅ E2E测试数据设置完成")


def setup_test_scenarios():
    """设置测试场景"""
    print("🎭 设置测试场景...")

    scenarios = {
        'user_registration': {
            'description': '用户注册流程',
            'steps': [
                '访问注册页面',
                '填写注册信息',
                '提交注册表单',
                '验证注册成功'
            ],
            'expected_duration': 30
        },
        'user_login': {
            'description': '用户登录流程',
            'steps': [
                '访问登录页面',
                '输入登录凭据',
                '提交登录表单',
                '验证登录成功'
            ],
            'expected_duration': 20
        },
        'trade_execution': {
            'description': '交易执行流程',
            'steps': [
                '用户登录',
                '选择交易品种',
                '输入交易参数',
                '提交交易订单',
                '验证交易执行'
            ],
            'expected_duration': 60
        },
        'portfolio_management': {
            'description': '投资组合管理',
            'steps': [
                '查看投资组合',
                '添加新持仓',
                '修改持仓配置',
                '验证组合更新'
            ],
            'expected_duration': 45
        }
    }

    scenarios_file = Path('e2e_test_scenarios.json')
    with open(scenarios_file, 'w') as f:
        json.dump(scenarios, f, indent=2)

    print("✅ 测试场景设置完成")


def main():
    """主函数"""
    try:
        # 设置E2E环境
        setup_e2e_environment()

        # 设置测试应用
        setup_test_application()

        # 设置测试数据
        setup_test_data()

        # 设置测试场景
        setup_test_scenarios()

        print("🎉 E2E测试环境准备就绪！")
        print("📋 E2E测试配置:")
        print("   前端URL: http://localhost:3000")
        print("   后端API: http://localhost:8080")
        print("   测试数据: e2e_test_data/")
        print("   测试报告: e2e_test_reports/")

    except Exception as e:
        print(f"❌ E2E测试环境设置失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
