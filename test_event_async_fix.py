#!/usr/bin/env python3
"""
测试事件总线异步处理器修复
"""

import sys
import asyncio
sys.path.append('.')

async def test_async_handler():
    """测试异步处理器执行"""
    print("🧪 测试异步处理器执行修复")

    # 创建事件总线
    from src.core.event_bus.core import EventBus
    from src.core.event_bus.types import EventType

    event_bus = EventBus()
    event_bus.initialize()

    # 标记异步处理器是否被调用
    handler_called = False

    async def test_async_handler_func(event):
        nonlocal handler_called
        handler_called = True
        print("✅ 异步处理器被成功调用！")

    # 订阅异步事件
    event_bus.subscribe_async(EventType.APPLICATION_STARTUP_COMPLETE, test_async_handler_func)
    print("✅ 异步处理器已订阅")

    # 发布事件
    event_id = event_bus.publish(EventType.APPLICATION_STARTUP_COMPLETE, {"test": "data"})
    print(f"✅ 事件已发布，ID: {event_id}")

    # 等待事件处理
    await asyncio.sleep(2)

    # 检查处理器是否被调用
    if handler_called:
        print("✅ 测试通过：异步处理器成功执行")
        return True
    else:
        print("❌ 测试失败：异步处理器未被调用")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_async_handler())
    print(f"\n最终结果: {'通过' if result else '失败'}")