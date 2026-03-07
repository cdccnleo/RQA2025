#!/usr/bin/env python3
"""
测试启动脚本 - 诊断容器启动问题
"""

import sys
import os
from pathlib import Path

def test_imports():
    """测试关键导入"""
    print("🔍 测试关键导入...")

    # 添加路径
    sys.path.insert(0, '/app')
    sys.path.insert(0, '/app/src')

    try:
        print("  测试: fastapi...")
        from fastapi import FastAPI
        print("  ✅ fastapi 导入成功")
    except Exception as e:
        print(f"  ❌ fastapi 导入失败: {e}")
        return False

    try:
        print("  测试: uvicorn...")
        import uvicorn
        print("  ✅ uvicorn 导入成功")
    except Exception as e:
        print(f"  ❌ uvicorn 导入失败: {e}")
        return False

    try:
        print("  测试: main模块...")
        import main
        print("  ✅ main 导入成功")
    except Exception as e:
        print(f"  ❌ main 导入失败: {e}")
        return False

    try:
        print("  测试: UnifiedConfigManager...")
        from src.infrastructure.config import UnifiedConfigManager
        print("  ✅ UnifiedConfigManager 导入成功")
    except Exception as e:
        print(f"  ❌ UnifiedConfigManager 导入失败: {e}")
        return False

    try:
        print("  测试: UnifiedLogger...")
        from src.infrastructure.logging import UnifiedLogger
        print("  ✅ UnifiedLogger 导入成功")
    except Exception as e:
        print(f"  ❌ UnifiedLogger 导入失败: {e}")
        return False

    return True

def test_config():
    """测试配置系统"""
    print("\n🔍 测试配置系统...")

    try:
        from src.infrastructure.config import UnifiedConfigManager
        config_manager = UnifiedConfigManager()
        print("  ✅ UnifiedConfigManager 初始化成功")
        return True
    except Exception as e:
        print(f"  ❌ UnifiedConfigManager 初始化失败: {e}")
        return False

def test_app_creation():
    """测试应用创建"""
    print("\n🔍 测试应用创建...")

    try:
        import main
        print("  ✅ 应用模块导入成功")

        # 检查是否有app对象
        if hasattr(main, 'app'):
            print("  ✅ app对象存在")
        else:
            print("  ❌ app对象不存在")
            return False

        return True
    except Exception as e:
        print(f"  ❌ 应用创建失败: {e}")
        return False

def main():
    """主函数"""
    print("🐳 容器启动诊断脚本")
    print("=" * 50)
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[:3]}...")
    print()

    success = True

    success &= test_imports()
    success &= test_config()
    success &= test_app_creation()

    print("\n" + "=" * 50)
    if success:
        print("✅ 所有测试通过，可以尝试启动应用")
    else:
        print("❌ 发现问题，请检查上述错误信息")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())