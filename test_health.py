#!/usr/bin/env python3
"""
测试健康检查端点
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_health_import():
    """测试健康检查端点导入"""
    print("测试健康检查端点导入...")
    try:
        from src.gateway.web.api import app
        print("✅ 应用导入成功")

        # 检查路由
        routes = [r for r in app.routes if hasattr(r, 'path')]
        health_routes = [r for r in routes if r.path == '/health']
        test_routes = [r for r in routes if r.path == '/test']

        print(f"总路由数: {len(routes)}")
        print(f"健康检查路由: {len(health_routes)}")
        print(f"测试路由: {len(test_routes)}")

        for route in health_routes + test_routes:
            print(f"  - {route.methods} {route.path}")

        return True

    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_health_function():
    """测试健康检查函数"""
    print("\n测试健康检查函数...")
    try:
        from src.gateway.web.api import health_check
        result = health_check()
        print(f"✅ 健康检查函数执行成功: {result}")
        return True
    except Exception as e:
        print(f"❌ 健康检查函数执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("健康检查端点测试")
    print("=" * 40)

    success = True
    success &= test_health_import()
    success &= test_health_function()

    if success:
        print("\n✅ 所有测试通过")
    else:
        print("\n❌ 测试失败")