#!/usr/bin/env python3
"""
测试修复效果的简单脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_fpga_fix():
    """测试FPGA修复"""
    print("🔧 测试FPGA修复...")
    try:
        from src.features.acceleration.fpga.fpga_manager import FPGAManager
        manager = FPGAManager()

        # 测试get_accelerator方法
        result = manager.get_accelerator()
        if result and 'device_id' in result:
            print("✅ FPGA get_accelerator方法工作正常")
            return True
        else:
            print("❌ FPGA get_accelerator方法返回无效结果")
            return False
    except Exception as e:
        print(f"❌ FPGA测试失败: {e}")
        return False

def test_infrastructure_singleton():
    """测试基础设施单例"""
    print("🏗️ 测试基础设施单例...")
    try:
        from src.infrastructure.init_infrastructure import Infrastructure

        # 创建多个实例
        infra1 = Infrastructure()
        infra2 = Infrastructure()

        # 检查是否是同一个实例
        if infra1 is infra2:
            print("✅ 基础设施单例模式工作正常")
            return True
        else:
            print("❌ 基础设施单例模式失败")
            return False
    except Exception as e:
        print(f"❌ 基础设施测试失败: {e}")
        return False

def test_health_check_endpoint():
    """测试健康检查端点结构"""
    print("❤️ 测试健康检查端点...")
    try:
        # 这里我们只是检查代码结构，不实际调用端点
        # 因为端点需要在FastAPI应用运行时才能调用
        from src.gateway.web.api import app

        # 检查是否有health路由
        health_routes = [route for route in app.routes if hasattr(route, 'path') and route.path == '/health']
        if health_routes:
            print("✅ 健康检查端点已注册")
            return True
        else:
            print("❌ 健康检查端点未找到")
            return False
    except Exception as e:
        print(f"❌ 健康检查端点测试失败: {e}")
        return False

def test_event_bus():
    """测试事件总线"""
    print("🔄 测试事件总线...")
    try:
        from src.core.event_bus import get_event_bus
        from src.core.event_bus.types import EventType

        bus = get_event_bus()
        print(f"✅ 全局事件总线获取成功，ID: {id(bus)}")

        # 测试订阅功能
        def test_handler(event):
            pass

        bus.subscribe(EventType.APPLICATION_STARTUP_COMPLETE, test_handler)
        count = bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)

        if count > 0:
            print("✅ 事件订阅功能正常")
            return True
        else:
            print("❌ 事件订阅功能异常")
            return False
    except Exception as e:
        print(f"❌ 事件总线测试失败: {e}")
        return False

def main():
    print("🚀 修复效果测试")
    print("=" * 40)

    tests = [
        ("FPGA功能修复", test_fpga_fix),
        ("基础设施单例", test_infrastructure_singleton),
        ("健康检查端点", test_health_check_endpoint),
        ("事件总线功能", test_event_bus)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        result = test_func()
        results.append((name, result))

    print("\n" + "=" * 40)
    print("测试结果总结:")
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False

    print(f"\n总体结果: {'✅ 所有测试通过' if all_passed else '❌ 部分测试失败'}")

    if all_passed:
        print("\n🎉 所有修复已生效！可以重新构建容器镜像进行部署。")
    else:
        print("\n⚠️ 部分修复可能需要进一步检查。")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)