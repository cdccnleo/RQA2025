#!/usr/bin/env python3
"""
基础测试 - 只测试导入，不进行初始化
"""

import sys
import os

def test_basic_imports():
    """测试基本导入"""
    print("测试基本导入...")

    try:
        print("1. 测试fastapi...")
        from fastapi import FastAPI
        print("   ✅ fastapi OK")

        print("2. 测试uvicorn...")
        import uvicorn
        print("   ✅ uvicorn OK")

        return True
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False

def test_src_imports():
    """测试src模块导入"""
    print("\n测试src模块导入...")

    try:
        print("3. 添加src路径...")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        print("   ✅ 路径添加成功")

        print("4. 测试APIService导入...")
        from src.core.core_services.api import APIService
        print("   ✅ APIService导入成功")

        return True
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🐳 基础导入测试")
    print("=" * 30)

    success = True
    success &= test_basic_imports()
    success &= test_src_imports()

    print("\n" + "=" * 30)
    if success:
        print("✅ 基础导入测试通过")
    else:
        print("❌ 基础导入测试失败")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)