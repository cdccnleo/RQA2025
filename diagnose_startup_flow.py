#!/usr/bin/env python3
"""
启动流程诊断脚本

用于诊断为什么主启动流程没有成功触发调度器启动
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def diagnose_event_bus_setup():
    """诊断事件总线设置"""
    print("🔍 诊断事件总线设置...")

    try:
        from src.core.event_bus.core import EventBus
        from src.core.event_bus.types import EventType
        print("✅ EventBus导入成功")
    except Exception as e:
        print(f"❌ EventBus导入失败: {e}")
        return False

    # 检查模块级别注册
    print("\n📋 检查模块级别注册...")
    try:
        from src.core.orchestration.business_process.app_startup_listener import get_app_startup_listener, register_app_startup_listener

        # 获取监听器实例
        listener = get_app_startup_listener()
        print(f"监听器实例: {listener}")
        print(f"监听器已注册: {listener._registered}")
        print(f"监听器事件总线: {listener.event_bus}")

        if listener.event_bus:
            subscriber_count = listener.event_bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)
            print(f"APPLICATION_STARTUP_COMPLETE订阅者数量: {subscriber_count}")

            if subscriber_count == 0:
                print("⚠️  警告: 没有订阅者，这可能是主启动流程失败的原因")
            else:
                print("✅ 订阅者数量正常")
        else:
            print("❌ 监听器没有事件总线实例")

    except Exception as e:
        print(f"❌ 检查监听器失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def simulate_event_publication():
    """模拟事件发布"""
    print("\n🎯 模拟事件发布...")

    try:
        from src.core.orchestration.business_process.app_startup_listener import get_app_startup_listener
        from src.core.event_bus.types import EventType

        listener = get_app_startup_listener()

        if not listener.event_bus:
            print("❌ 监听器没有事件总线实例，无法模拟")
            return False

        event_bus = listener.event_bus
        subscriber_count = event_bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)
        print(f"发布前订阅者数量: {subscriber_count}")

        # 发布测试事件
        print("发布测试事件...")
        event_bus.publish(
            EventType.APPLICATION_STARTUP_COMPLETE,
            {
                "service_name": "test_simulation",
                "timestamp": time.time(),
                "source": "diagnose_script"
            },
            source="diagnose_script"
        )
        print("✅ 测试事件已发布")

        # 等待一秒让异步处理完成
        import asyncio
        async def wait_and_check():
            await asyncio.sleep(1)
            print(f"调度器启动状态: {listener._scheduler_started}")

        asyncio.run(wait_and_check())

        return True

    except Exception as e:
        print(f"❌ 模拟事件发布失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_api_module_loading():
    """检查API模块加载过程"""
    print("\n🏗️  检查API模块加载...")

    try:
        # 模拟api.py模块级别的注册过程
        print("模拟模块级别注册过程...")

        from src.core.event_bus.core import EventBus
        from src.core.orchestration.business_process.app_startup_listener import register_app_startup_listener

        # 创建事件总线实例（模拟api.py中的过程）
        event_bus = EventBus()
        if not hasattr(event_bus, '_initialized') or not event_bus._initialized:
            event_bus.initialize()
        print(f"✅ 创建事件总线实例: {id(event_bus)}")

        # 注册监听器
        register_app_startup_listener(event_bus)
        print("✅ 监听器注册完成")

        # 检查注册结果
        from src.core.orchestration.business_process.app_startup_listener import get_app_startup_listener
        from src.core.event_bus.types import EventType

        listener = get_app_startup_listener()
        print(f"监听器注册状态: {listener._registered}")

        if listener.event_bus:
            subscriber_count = listener.event_bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)
            print(f"订阅者数量: {subscriber_count}")

            if listener.event_bus is event_bus:
                print("✅ 监听器使用的是同一个事件总线实例")
            else:
                print(f"⚠️  警告: 监听器使用不同的事件总线实例 (监听器: {id(listener.event_bus)}, 创建的: {id(event_bus)})")
        else:
            print("❌ 监听器没有事件总线实例")

        return True

    except Exception as e:
        print(f"❌ API模块加载检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主诊断函数"""
    print("🚀 启动流程诊断工具")
    print("=" * 50)

    # 1. 检查事件总线设置
    if not diagnose_event_bus_setup():
        print("❌ 事件总线设置诊断失败")
        return

    # 2. 检查API模块加载
    if not check_api_module_loading():
        print("❌ API模块加载检查失败")
        return

    # 3. 模拟事件发布
    if not simulate_event_publication():
        print("❌ 事件发布模拟失败")
        return

    print("\n" + "=" * 50)
    print("✅ 诊断完成")

if __name__ == "__main__":
    main()