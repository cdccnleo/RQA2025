#!/usr/bin/env python3
"""
启动流程修复验证脚本

验证修复后的启动流程是否正常工作
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

async def verify_event_bus_consistency():
    """验证事件总线实例一致性"""
    print("🔍 验证事件总线实例一致性...")

    try:
        from src.core.event_bus import get_event_bus
        from src.core.orchestration.business_process.app_startup_listener import get_app_startup_listener, register_app_startup_listener
        from src.core.event_bus.types import EventType

        # 获取全局事件总线
        global_bus = get_event_bus()
        print(f"✅ 全局事件总线实例ID: {id(global_bus)}")

        # 注册监听器
        register_app_startup_listener(global_bus)
        print("✅ 监听器注册完成")

        # 获取监听器并检查
        listener = get_app_startup_listener()
        print(f"监听器实例: {listener}")
        print(f"监听器已注册: {listener._registered}")

        if listener.event_bus:
            print(f"监听器事件总线实例ID: {id(listener.event_bus)}")

            # 检查是否是同一个实例
            if listener.event_bus is global_bus:
                print("✅ 监听器使用全局事件总线实例")
            else:
                print("❌ 监听器使用不同的事件总线实例")
                return False

            # 检查订阅者
            subscriber_count = global_bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)
            print(f"订阅者数量: {subscriber_count}")

            if subscriber_count > 0:
                print("✅ 事件订阅成功")
            else:
                print("❌ 事件订阅失败")
                return False
        else:
            print("❌ 监听器没有事件总线实例")
            return False

        # 模拟事件发布
        print("\n🎯 模拟事件发布...")
        global_bus.publish(
            EventType.APPLICATION_STARTUP_COMPLETE,
            {
                "service_name": "test_verification",
                "timestamp": time.time(),
                "source": "verify_script"
            },
            source="verify_script"
        )
        print("✅ 测试事件已发布")

        # 等待异步处理
        await asyncio.sleep(1)

        # 检查调度器状态
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler
        scheduler = get_data_collection_scheduler()

        if scheduler.is_running():
            print("✅ 调度器已启动（事件驱动成功）")
            return True
        else:
            print("❌ 调度器未启动（事件处理失败）")
            return False

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_api_module_registration():
    """验证API模块注册逻辑"""
    print("\n🏗️  验证API模块注册逻辑...")

    try:
        # 模拟api.py中的注册过程
        from src.core.orchestration.business_process.app_startup_listener import register_app_startup_listener
        from src.core.event_bus import get_event_bus
        from src.core.event_bus.types import EventType

        # 使用全局事件总线
        event_bus = get_event_bus()
        print(f"模块级别事件总线实例ID: {id(event_bus)}")

        # 注册监听器
        register_app_startup_listener(event_bus)

        # 验证注册结果
        subscriber_count = event_bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)
        print(f"注册后订阅者数量: {subscriber_count}")

        if subscriber_count > 0:
            print("✅ 模块级别注册成功")
            return True
        else:
            print("❌ 模块级别注册失败")
            return False

    except Exception as e:
        print(f"❌ API模块注册验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_lifespan_logic():
    """验证lifespan逻辑"""
    print("\n⚙️  验证lifespan逻辑...")

    try:
        from src.core.event_bus import get_event_bus
        from src.core.event_bus.types import EventType
        from src.core.orchestration.business_process.app_startup_listener import get_app_startup_listener

        # 使用全局事件总线
        event_bus = get_event_bus()
        print(f"lifespan事件总线实例ID: {id(event_bus)}")

        # 验证监听器注册状态
        listener = get_app_startup_listener()
        subscriber_count = event_bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)

        print(f"监听器注册状态: {listener._registered}")
        print(f"订阅者数量: {subscriber_count}")

        # 如果订阅者数量为0，尝试重新注册
        if subscriber_count == 0:
            print("检测到订阅者数量为0，重新注册...")
            listener.register(event_bus)
            subscriber_count = event_bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)
            print(f"重新注册后订阅者数量: {subscriber_count}")

        # 发布事件
        print("发布APPLICATION_STARTUP_COMPLETE事件...")
        event_bus.publish(
            EventType.APPLICATION_STARTUP_COMPLETE,
            {
                "service_name": "lifespan_test",
                "timestamp": time.time(),
                "source": "verify_lifespan"
            },
            source="verify_lifespan"
        )
        print("✅ lifespan事件发布完成")

        # 等待异步处理
        await asyncio.sleep(1)

        # 检查调度器状态
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler
        scheduler = get_data_collection_scheduler()

        if scheduler.is_running():
            print("✅ lifespan逻辑验证成功，调度器已启动")
            return True
        else:
            print("❌ lifespan逻辑验证失败，调度器未启动")
            return False

    except Exception as e:
        print(f"❌ lifespan逻辑验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主验证函数"""
    print("🚀 启动流程修复验证工具")
    print("=" * 50)

    results = []

    # 1. 验证事件总线实例一致性
    result1 = await verify_event_bus_consistency()
    results.append(("事件总线实例一致性", result1))

    # 2. 验证API模块注册逻辑
    result2 = await verify_api_module_registration()
    results.append(("API模块注册逻辑", result2))

    # 3. 验证lifespan逻辑
    result3 = await verify_lifespan_logic()
    results.append(("lifespan逻辑", result3))

    # 输出结果摘要
    print("\n" + "=" * 50)
    print("📊 验证结果摘要:")
    all_passed = True
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False

    if all_passed:
        print("\n🎉 所有验证通过！启动流程修复成功。")
    else:
        print("\n⚠️  部分验证失败，需要进一步检查。")

    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)