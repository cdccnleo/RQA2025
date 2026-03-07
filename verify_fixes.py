#!/usr/bin/env python3
"""
修复验证脚本

验证已修复的问题：
1. FPGA风险检查错误
2. 基础设施初始化冗余
3. 主启动流程
"""

import sys
import os
import time
import asyncio
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

        # 测试get_accelerator方法是否存在
        if hasattr(manager, 'get_accelerator'):
            print("✅ get_accelerator方法存在")

            # 测试方法调用
            result = manager.get_accelerator()
            if result and isinstance(result, dict):
                print("✅ get_accelerator方法返回正确格式")
                print(f"   加速器状态: {result.get('status', 'unknown')}")
                print(f"   可用性: {result.get('available', False)}")
                return True
            else:
                print("❌ get_accelerator方法返回格式不正确")
                return False
        else:
            print("❌ get_accelerator方法不存在")
            return False

    except Exception as e:
        print(f"❌ FPGA测试失败: {e}")
        return False

def test_infrastructure_singleton():
    """测试基础设施单例模式"""
    print("\n🏗️ 测试基础设施单例模式...")

    try:
        from src.infrastructure.init_infrastructure import Infrastructure

        # 创建多个实例
        infra1 = Infrastructure()
        infra2 = Infrastructure()
        infra3 = Infrastructure()

        # 检查是否是同一个实例
        if infra1 is infra2 is infra3:
            print("✅ 单例模式工作正常，所有实例都是同一个对象")
            print(f"   实例ID: {id(infra1)}")

            # 检查初始化标志
            if Infrastructure._initialized:
                print("✅ 初始化标志正确设置")
                return True
            else:
                print("❌ 初始化标志未设置")
                return False
        else:
            print("❌ 单例模式失败，不同实例被创建")
            print(f"   实例1 ID: {id(infra1)}")
            print(f"   实例2 ID: {id(infra2)}")
            print(f"   实例3 ID: {id(infra3)}")
            return False

    except Exception as e:
        print(f"❌ 基础设施单例测试失败: {e}")
        return False

async def test_event_bus_startup():
    """测试事件总线启动流程"""
    print("\n🔄 测试事件总线启动流程...")

    try:
        from src.core.event_bus import get_event_bus
        from src.core.event_bus.types import EventType
        from src.core.orchestration.business_process.app_startup_listener import get_app_startup_listener

        # 获取全局事件总线
        event_bus = get_event_bus()
        print(f"全局事件总线实例ID: {id(event_bus)}")

        # 获取监听器
        listener = get_app_startup_listener()
        print(f"监听器实例ID: {id(listener)}")
        print(f"监听器已注册: {listener._registered}")

        # 检查监听器的事件总线是否一致
        if listener.event_bus is event_bus:
            print("✅ 监听器使用全局事件总线")
        else:
            print("❌ 监听器使用不同的事件总线")
            print(f"   监听器事件总线ID: {id(listener.event_bus) if listener.event_bus else 'None'}")
            return False

        # 检查订阅者数量
        subscriber_count = event_bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)
        print(f"APPLICATION_STARTUP_COMPLETE订阅者数量: {subscriber_count}")

        if subscriber_count > 0:
            print("✅ 事件订阅正常")

            # 模拟事件发布
            print("发布测试事件...")
            event_bus.publish(
                EventType.APPLICATION_STARTUP_COMPLETE,
                {
                    "service_name": "test_verification",
                    "timestamp": time.time(),
                    "source": "verify_script"
                },
                source="verify_script"
            )

            # 等待异步处理
            await asyncio.sleep(1)

            # 检查调度器状态
            from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler
            scheduler = get_data_collection_scheduler()

            if scheduler.is_running():
                print("✅ 主启动流程工作正常，调度器已启动")
                return True
            else:
                print("❌ 事件处理失败，调度器未启动")
                return False
        else:
            print("❌ 事件订阅失败")
            return False

    except Exception as e:
        print(f"❌ 事件总线启动测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_controller():
    """测试风险控制器不再出现FPGA错误"""
    print("\n⚠️ 测试风险控制器FPGA调用...")

    try:
        # 模拟风险控制器调用FPGA
        from src.features.acceleration.fpga.fpga_manager import FPGAManager

        manager = FPGAManager()

        # 初始化FPGA（如果需要）
        if hasattr(manager, 'initialize'):
            manager.initialize()

        # 调用get_accelerator（模拟风险控制器调用）
        accelerator = manager.get_accelerator()

        if accelerator:
            print("✅ FPGA get_accelerator调用成功")
            print(f"   设备状态: {accelerator.get('status', 'unknown')}")
            print(f"   性能指标: {accelerator.get('performance_metrics', {})}")
            return True
        else:
            print("❌ FPGA get_accelerator调用返回None")
            return False

    except AttributeError as e:
        if 'get_accelerator' in str(e):
            print("❌ FPGA get_accelerator方法仍然缺失")
            return False
        else:
            print(f"❌ 其他AttributeError: {e}")
            return False
    except Exception as e:
        print(f"❌ 风险控制器测试失败: {e}")
        return False

async def main():
    """主验证函数"""
    print("🚀 修复验证工具")
    print("=" * 50)

    results = []

    # 1. 测试FPGA修复
    fpga_result = test_fpga_fix()
    results.append(("FPGA风险检查修复", fpga_result))

    # 2. 测试基础设施单例
    infra_result = test_infrastructure_singleton()
    results.append(("基础设施单例模式", infra_result))

    # 3. 测试事件总线启动流程
    event_result = await test_event_bus_startup()
    results.append(("事件总线启动流程", event_result))

    # 4. 测试风险控制器
    risk_result = test_risk_controller()
    results.append(("风险控制器FPGA调用", risk_result))

    # 输出结果摘要
    print("\n" + "=" * 50)
    print("📊 修复验证结果摘要")
    print("=" * 50)

    all_passed = True
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False

    print(f"\n🎯 总体结果: {'✅ 所有修复验证通过' if all_passed else '❌ 部分修复需要检查'}")

    # 提供修复建议
    if not all_passed:
        print("\n🔧 修复建议:")
        failed_tests = [name for name, result in results if not result]
        for failed_test in failed_tests:
            if "FPGA" in failed_test:
                print("- 检查FPGA管理器get_accelerator方法的实现")
            elif "基础设施" in failed_test:
                print("- 检查基础设施单例模式的实现")
            elif "事件总线" in failed_test:
                print("- 检查事件总线和监听器的配置")
            elif "风险控制器" in failed_test:
                print("- 检查风险控制器FPGA集成的实现")

    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)