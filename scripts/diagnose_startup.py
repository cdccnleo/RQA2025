#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025系统启动诊断脚本

诊断系统启动问题，逐步排查组件初始化失败的原因。
"""

import sys
import asyncio
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def diagnose_config_manager():
    """诊断配置管理器"""
    print("🔍 诊断配置管理器...")

    try:
        from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
        config_manager = UnifiedConfigManager()
        print("✅ UnifiedConfigManager创建成功")

        # 测试基本方法
        test_value = config_manager.get("test_key", "default_value")
        print(f"✅ get方法工作正常: {test_value}")

        # 测试数据库配置
        db_host = config_manager.get("database.host", "localhost")
        print(f"✅ 数据库配置获取正常: host={db_host}")

        return True
    except Exception as e:
        print(f"❌ 配置管理器诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def diagnose_database_service():
    """诊断数据库服务"""
    print("\n🔍 诊断数据库服务...")

    try:
        from src.core.database_service import get_database_service
        print("✅ 数据库服务模块导入成功")

        # 测试服务创建
        db_service = await get_database_service()
        print("✅ 数据库服务创建成功")

        return True
    except Exception as e:
        print(f"❌ 数据库服务诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def diagnose_business_service():
    """诊断业务服务"""
    print("\n🔍 诊断业务服务...")

    try:
        from src.core.business_service import get_business_service
        print("✅ 业务服务模块导入成功")

        # 测试服务创建
        business_service = await get_business_service()
        print("✅ 业务服务创建成功")

        return True
    except Exception as e:
        print(f"❌ 业务服务诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def diagnose_fastapi_app():
    """诊断FastAPI应用"""
    print("\n🔍 诊断FastAPI应用...")

    try:
        from src.app import RQAApplication
        print("✅ RQAApplication类导入成功")

        # 创建应用实例
        app_instance = RQAApplication()
        print("✅ RQAApplication实例创建成功")

        # 测试应用创建
        await app_instance.initialize()
        print("✅ 应用初始化成功")

        # 检查app属性
        if hasattr(app_instance, 'app') and app_instance.app is not None:
            print("✅ FastAPI应用创建成功")
            return True
        else:
            print("❌ FastAPI应用创建失败: app属性不存在或为空")
            return False

    except Exception as e:
        print(f"❌ FastAPI应用诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def diagnose_global_app():
    """诊断全局app变量"""
    print("\n🔍 诊断全局app变量...")

    try:
        import src.app as app_module
        if hasattr(app_module, 'app'):
            print(f"✅ 全局app变量存在: {type(app_module.app)}")
            return True
        else:
            print("❌ 全局app变量不存在")
            return False
    except Exception as e:
        print(f"❌ 全局app变量诊断失败: {e}")
        return False


async def run_full_diagnosis():
    """运行完整诊断"""
    print("="*80)
    print("🚀 RQA2025系统启动诊断")
    print("="*80)

    results = []

    # 1. 配置管理器诊断
    results.append(await diagnose_config_manager())

    # 2. 数据库服务诊断
    results.append(await diagnose_database_service())

    # 3. 业务服务诊断
    results.append(await diagnose_business_service())

    # 4. FastAPI应用诊断
    results.append(await diagnose_fastapi_app())

    # 5. 全局app变量诊断
    results.append(await diagnose_global_app())

    print("\n" + "="*80)
    print("📊 诊断结果汇总")
    print("="*80)

    component_names = [
        "配置管理器",
        "数据库服务",
        "业务服务",
        "FastAPI应用",
        "全局app变量"
    ]

    success_count = 0
    for i, (name, success) in enumerate(zip(component_names, results)):
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{i+1}. {name}: {status}")
        if success:
            success_count += 1

    print(f"\n总体结果: {success_count}/{len(results)} 个组件诊断通过")

    if success_count == len(results):
        print("🎉 所有组件诊断通过！系统应该可以正常启动。")
        return True
    else:
        print("⚠️ 部分组件诊断失败，需要修复后再尝试启动。")
        return False


async def main():
    """主函数"""
    try:
        success = await run_full_diagnosis()

        if success:
            print("\n💡 建议: 现在可以尝试启动系统")
            print("   python -m uvicorn src.app:app --host 0.0.0.0 --port 8000")
        else:
            print("\n💡 建议: 请修复上述失败的组件后再尝试启动")

    except KeyboardInterrupt:
        print("\n用户中断诊断")
    except Exception as e:
        print(f"\n诊断过程异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
