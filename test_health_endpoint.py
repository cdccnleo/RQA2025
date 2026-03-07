#!/usr/bin/env python3
"""
测试健康检查端点修复
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_health_endpoint_registration():
    """测试健康检查端点是否正确注册"""
    print("测试健康检查端点注册...")

    try:
        from src.gateway.web.api import app

        print(f"FastAPI应用创建成功，路由总数: {len(app.routes)}")

        # 查找健康检查路由
        health_routes = []
        for route in app.routes:
            if hasattr(route, 'path') and route.path == '/health':
                health_routes.append(route)

        print(f"找到 {len(health_routes)} 个健康检查路由")

        if health_routes:
            for i, route in enumerate(health_routes, 1):
                methods = getattr(route, 'methods', ['UNKNOWN'])
                print(f"  路由 {i}: {methods} {route.path}")
            return True
        else:
            print("❌ 未找到健康检查路由")
            return False

    except Exception as e:
        print(f"❌ 健康检查端点测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_health_function():
    """测试健康检查函数"""
    print("\n测试健康检查函数调用...")

    try:
        from src.gateway.web.api import health_check

        # 调用健康检查函数
        result = health_check()

        required_keys = ['status', 'service', 'environment', 'timestamp']
        missing_keys = [key for key in required_keys if key not in result]

        if missing_keys:
            print(f"❌ 健康检查响应缺少必要字段: {missing_keys}")
            return False

        print("✅ 健康检查函数返回正确格式")
        print(f"   状态: {result.get('status')}")
        print(f"   服务: {result.get('service')}")
        print(f"   环境: {result.get('environment')}")
        print(f"   时间戳: {result.get('timestamp')}")

        return True

    except Exception as e:
        print(f"❌ 健康检查函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_route_import():
    """测试路由导入"""
    print("\n测试健康检查路由导入...")

    try:
        from src.gateway.web.health_routes import router

        print(f"✅ 健康检查路由器导入成功，包含 {len(router.routes)} 个路由")

        # 检查路由
        for route in router.routes:
            if hasattr(route, 'path'):
                print(f"   - {route.path}")

        return True

    except Exception as e:
        print(f"❌ 健康检查路由导入失败: {e}")
        print("   这将使用降级的健康检查端点")
        return False  # 这不是错误，只是降级

def main():
    print("健康检查端点修复测试")
    print("=" * 50)

    tests = [
        ("路由导入测试", test_route_import),
        ("端点注册测试", test_health_endpoint_registration),
        ("函数调用测试", test_health_function)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        result = test_func()
        results.append((name, result))

    print("\n" + "=" * 50)
    print("测试结果总结:")

    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False

    if all_passed:
        print("\n🎉 所有测试通过！健康检查端点修复成功。")
        print("   现在API服务应该可以正常访问 http://localhost:8000/health")
    else:
        print("\n⚠️ 部分测试失败，可能需要进一步检查。")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)