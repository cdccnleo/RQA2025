#!/usr/bin/env python3
"""
容器内测试脚本 - 验证部署环境
"""

import sys
import os
import subprocess

def check_environment():
    """检查容器环境"""
    print("🔍 检查容器环境...")

    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[:2]}")

    # 检查关键文件
    files_to_check = [
        '/app/main.py',
        '/app/src/main.py',
        '/app/scripts/start_api_server.py',
        '/app/debug_container.py'
    ]

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            print(f"❌ {file_path} 不存在")

def test_imports():
    """测试导入"""
    print("\n🔍 测试导入...")

    # 设置路径
    sys.path.insert(0, '/app')
    sys.path.insert(0, '/app/src')

    try:
        from fastapi import FastAPI
        print("✅ FastAPI导入成功")
    except Exception as e:
        print(f"❌ FastAPI导入失败: {e}")
        return False

    try:
        import uvicorn
        print("✅ Uvicorn导入成功")
    except Exception as e:
        print(f"❌ Uvicorn导入失败: {e}")
        return False

    return True

def test_core_services():
    """测试核心服务导入"""
    print("\n🔍 测试核心服务...")

    try:
        from src.core.core_services.api import APIService
        print("✅ APIService导入成功")
        return True
    except Exception as e:
        print(f"❌ APIService导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🐳 RQA2025 容器部署测试")
    print("=" * 50)

    check_environment()

    if not test_imports():
        print("\n❌ 基础导入失败，退出")
        return False

    if not test_core_services():
        print("\n❌ 核心服务导入失败，退出")
        return False

    print("\n✅ 容器环境检查通过")
    print("准备启动主服务...")

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 启动主服务...")
        # 这里可以启动主服务
        sys.exit(0)
    else:
        print("\n💥 容器测试失败")
        sys.exit(1)