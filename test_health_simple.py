#!/usr/bin/env python3
"""
简单测试健康检查路由
"""

def test_health_routes():
    """测试健康检查路由文件"""
    print("测试健康检查路由文件...")

    try:
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root / 'src'))

        from src.gateway.web.health_routes import router

        print("✅ 健康检查路由器导入成功")
        print(f"路由数: {len(router.routes)}")

        # 检查路由
        for route in router.routes:
            if hasattr(route, 'path'):
                print(f"  - {route.methods} {route.path}")

        return True

    except Exception as e:
        print(f"❌ 健康检查路由测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_health_routes()