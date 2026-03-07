#!/usr/bin/env python3
"""
最小化测试

只测试基本导入，不运行完整应用
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """测试基本导入"""
    print("🔍 测试基本导入...")

    try:
        print("导入 FastAPI...")
        from fastapi import FastAPI
        print("✅ FastAPI 导入成功")

        print("导入 contextlib...")
        from contextlib import asynccontextmanager
        print("✅ contextlib 导入成功")

        print("导入 asyncio...")
        import asyncio
        print("✅ asyncio 导入成功")

        return True

    except Exception as e:
        print(f"❌ 基本导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lifespan_function():
    """测试 lifespan 函数定义"""
    print("\n🔍 测试 lifespan 函数定义...")

    try:
        # 定义简单的 lifespan 函数
        @asynccontextmanager
        async def test_lifespan(app):
            print("✅ 测试 lifespan 函数执行")
            yield

        print("✅ lifespan 函数定义成功")
        return True

    except Exception as e:
        print(f"❌ lifespan 函数定义失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fastapi_creation():
    """测试 FastAPI 创建"""
    print("\n🔍 测试 FastAPI 创建...")

    try:
        @asynccontextmanager
        async def test_lifespan(app):
            yield

        app = FastAPI(
            title="Test App",
            lifespan=test_lifespan
        )

        print("✅ FastAPI 应用创建成功")
        print(f"路由数: {len(app.routes)}")

        return True

    except Exception as e:
        print(f"❌ FastAPI 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("最小化导入测试")
    print("=" * 60)

    # 测试基本导入
    basic_test = test_basic_imports()

    # 测试 lifespan 函数
    lifespan_test = test_lifespan_function()

    # 测试 FastAPI 创建
    fastapi_test = test_fastapi_creation()

    print("\n" + "=" * 60)
    if basic_test and lifespan_test and fastapi_test:
        print("✅ 最小化测试通过 - 基本功能正常")
    else:
        print("❌ 最小化测试失败 - 存在基础问题")

    print("=" * 60)

if __name__ == "__main__":
    main()