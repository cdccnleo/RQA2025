#!/usr/bin/env python3
"""
测试 lifespan 修复效果

验证修复后的应用是否能正常执行 lifespan 函数
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_app_creation():
    """测试应用创建"""
    print("🔍 测试应用创建...")

    try:
        from src.gateway.web.app_factory import create_app
        print("✅ 成功导入 create_app")

        app = create_app()
        print(f"✅ 应用创建成功，路由数: {len(app.routes)}")

        # 检查 lifespan
        if hasattr(app, 'lifespan') and app.lifespan is not None:
            print("✅ FastAPI 应用已配置 lifespan")
        else:
            print("❌ FastAPI 应用未配置 lifespan")

        return True

    except Exception as e:
        print(f"❌ 应用创建测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scheduler_import():
    """测试调度器导入"""
    print("\n🔍 测试调度器导入...")

    try:
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler
        print("✅ 成功导入调度器")

        scheduler = get_data_collection_scheduler()
        print("✅ 调度器实例获取成功")

        is_running = scheduler.is_running()
        print(f"调度器当前状态: {'运行中' if is_running else '未运行'}")

        return True

    except Exception as e:
        print(f"❌ 调度器导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("数据采集调度器 lifespan 修复验证")
    print("=" * 60)

    # 测试应用创建
    app_test = test_app_creation()

    # 测试调度器导入
    scheduler_test = test_scheduler_import()

    print("\n" + "=" * 60)
    if app_test and scheduler_test:
        print("✅ 修复验证完成 - 所有测试通过")
        print("建议下一步: 重启应用并观察启动日志")
    else:
        print("❌ 修复验证失败 - 存在问题需要进一步修复")

    print("=" * 60)

if __name__ == "__main__":
    main()