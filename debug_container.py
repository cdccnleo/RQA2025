#!/usr/bin/env python3
"""
容器启动调试脚本
用于诊断容器启动问题
"""

import os
import sys
import time
import subprocess

def check_files():
    """检查关键文件是否存在"""
    print("🔍 检查关键文件...")

    files_to_check = [
        '/app/main.py',
        '/app/simple_api.py',
        '/app/scripts/start_api_server.py',
        '/app/src/gateway/web/api.py',
        '/app/src/gateway/web/app_factory.py'
    ]

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path} 存在")
            # 显示文件大小
            size = os.path.getsize(file_path)
            print(f"     大小: {size} bytes")
        else:
            print(f"  ❌ {file_path} 不存在")

    print()

def check_environment():
    """检查环境变量"""
    print("🔍 检查环境变量...")

    env_vars = ['PYTHONPATH', 'RQA_ENV', 'PATH']
    for var in env_vars:
        value = os.getenv(var, 'NOT_SET')
        print(f"  {var}: {value}")

    print()

def check_python():
    """检查Python环境"""
    print("🔍 检查Python环境...")

    try:
        result = subprocess.run([sys.executable, '--version'],
                              capture_output=True, text=True, timeout=5)
        print(f"  Python版本: {result.stdout.strip()}")
    except Exception as e:
        print(f"  Python检查失败: {e}")

    print(f"  Python可执行文件: {sys.executable}")
    print(f"  当前工作目录: {os.getcwd()}")
    print()

def test_imports():
    """测试关键导入"""
    print("🔍 测试关键导入...")

    # 添加路径
    sys.path.insert(0, '/app')
    sys.path.insert(0, '/app/src')

    try:
        print("  导入 main...")
        import main
        print("  ✅ main 导入成功")
    except Exception as e:
        print(f"  ❌ main 导入失败: {e}")

    try:
        print("  导入 app_factory...")
        from src.gateway.web.app_factory import create_app
        print("  ✅ app_factory 导入成功")
    except Exception as e:
        print(f"  ❌ app_factory 导入失败: {e}")

    print()

def main():
    """主函数"""
    print("🐳 容器启动调试脚本")
    print("=" * 50)
    print(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    check_files()
    check_environment()
    check_python()
    test_imports()

    print("🏁 调试完成")
    print("\n建议解决方案:")
    print("1. 确保 Dockerfile CMD 使用正确的启动脚本")
    print("2. 检查 Kubernetes deployment 的 command 配置")
    print("3. 验证文件路径和权限")

if __name__ == "__main__":
    main()