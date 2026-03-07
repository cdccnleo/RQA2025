#!/usr/bin/env python3
"""
快速测试应用创建

验证修复后的应用创建是否正常
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_app_creation():
    """测试基本的 FastAPI 应用创建"""
    print("🔍 测试基本应用创建...")

    try:
        from fastapi import FastAPI
        from contextlib import asynccontextmanager
        import asyncio

        # 创建简单的 lifespan 函数
        @asynccontextmanager
        async def test_lifespan(app):
            print("✅ lifespan 函数执行")
            yield

        # 创建应用
        app = FastAPI(
            title="Test App",
            lifespan=test_lifespan
        )

        print(f"✅ 应用创建成功，路由数: {len(app.routes)}")
        print("✅ lifespan 参数已正确传递")

        return True

    except Exception as e:
        print(f"❌ 基本应用创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_actual_app_creation():
    """测试实际应用创建"""
    print("\n🔍 测试实际应用创建...")

    try:
        # 只测试 app_factory，不运行完整应用
        from src.gateway.web.app_factory import create_app
        print("✅ 成功导入 create_app")

        app = create_app()
        print(f"✅ 应用创建成功，路由数: {len(app.routes)}")

        return True

    except Exception as e:
        print(f"❌ 实际应用创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("FastAPI 应用创建修复验证")
    print("=" * 60)

    # 测试基本应用创建
    basic_test = test_basic_app_creation()

    # 测试实际应用创建
    actual_test = test_actual_app_creation()

    print("\n" + "=" * 60)
    if basic_test and actual_test:
        print("✅ 修复验证完成 - 应用创建正常")
        print("建议: 重启应用测试完整启动流程")
    else:
        print("❌ 修复验证失败 - 应用创建仍有问题")

    print("=" * 60)

if __name__ == "__main__":
    main()