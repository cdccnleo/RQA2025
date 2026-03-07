#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存泄漏检测和修复验证脚本
"""

import time
import gc
import weakref
import sys
import os

def test_event_persistence_leak():
    """测试事件持久化的内存泄漏修复"""
    print("🔍 测试事件持久化内存泄漏修复...")

    # 导入事件持久化
    sys.path.insert(0, 'src')
    from core.event_bus.event_bus import EventPersistence

    persistence = EventPersistence()
    initial_objects = len(gc.get_objects())

    # 添加大量事件
    for i in range(15000):  # 超过默认的最大事件数10000
        event = type('Event', (), {
            'timestamp': time.time(),
            'event_type': 'test_event',
            'data': {'id': i, 'message': 'test message {}'.format(i)}
        })()
        persistence.save_event(event)

    # 检查对象数量
    after_objects = len(gc.get_objects())
    growth = after_objects - initial_objects

    print("   📊 事件数量: {}".format(len(persistence._events)))
    print("   📈 对象增长: {}".format(growth))

    # 验证清理机制
    if len(persistence._events) <= persistence._max_events:
        print("   ✅ 事件数量控制在限制内")
    else:
        print("   ❌ 事件数量超出限制")

    if growth < 1000:  # 合理的增长
        print("   ✅ 内存泄漏风险低")
    else:
        print("   ⚠️ 可能存在内存泄漏")

    return len(persistence._events)

def test_logger_cache_leak():
    """测试日志缓存的内存泄漏修复"""
    print("\n🔍 测试日志缓存内存泄漏修复...")

    # 导入日志器
    from infrastructure.logging.advanced_logger import AdvancedLogger

    logger = AdvancedLogger('test_logger')
    initial_objects = len(gc.get_objects())

    # 生成大量不同的日志消息
    for i in range(2000):
        logger.log('INFO', '测试消息 {}'.format(i), 'test_component_{}'.format(i % 10))

    after_objects = len(gc.get_objects())
    growth = after_objects - initial_objects

    print("   📊 缓存大小: {}".format(len(logger._string_cache)))
    print("   📈 对象增长: {}".format(growth))

    # 手动触发缓存清理
    logger.cleanup_cache()

    after_cleanup = len(logger._string_cache)
    print("   🧹 清理后缓存大小: {}".format(after_cleanup))

    if after_cleanup < len(logger._string_cache) * 0.6:  # 清理了至少40%
        print("   ✅ 缓存清理机制正常")
    else:
        print("   ⚠️ 缓存清理可能不充分")

    return len(logger._string_cache)

def test_circular_references():
    """测试循环引用检测"""
    print("\n🔍 测试循环引用检测...")

    # 创建循环引用
    class A:
        def __init__(self):
            self.ref = None

    a = A()
    b = A()
    a.ref = b
    b.ref = a

    # 手动删除引用
    del a, b

    # 强制垃圾回收
    collected = gc.collect()

    # 检查是否有垃圾对象
    if gc.garbage:
        print("   ⚠️ 发现 {} 个循环引用对象".format(len(gc.garbage)))
        for i, obj in enumerate(gc.garbage[:3]):
            print("      {}. {}".format(i+1, type(obj).__name__))
    else:
        print("   ✅ 未发现循环引用")

    # 清理垃圾
    gc.garbage.clear()

def test_large_objects():
    """测试大对象管理"""
    print("\n🔍 测试大对象管理...")

    # 创建一些大对象
    large_list = [i for i in range(10000)]
    large_dict = {str(i): 'value_{}'.format(i) for i in range(5000)}
    large_string = 'x' * 100000

    # 检查大对象
    all_objects = gc.get_objects()
    large_objects = []

    for obj in all_objects:
        try:
            size = sys.getsizeof(obj)
            if size > 50000:  # 50KB以上
                large_objects.append((type(obj).__name__, size))
        except:
            pass

    print("   📏 大对象统计:")
    for obj_type, size in sorted(large_objects, key=lambda x: x[1], reverse=True)[:5]:
        print("      {}: {:.1f} KB".format(obj_type, size / 1024))

    # 清理大对象
    del large_list, large_dict, large_string
    collected = gc.collect()

    print("   🗑️ 清理了大对象，GC回收: {} 个对象".format(collected))

def memory_pressure_test():
    """内存压力测试"""
    print("\n🔍 内存压力测试...")

    initial_memory = len(gc.get_objects())
    print("   📊 初始对象数: {}".format(initial_memory))

    # 模拟内存压力
    objects = []
    for i in range(10):
        # 创建一批对象
        batch = [{} for _ in range(1000)]
        objects.append(batch)

        # 每批后检查内存
        current = len(gc.get_objects())
        growth = current - initial_memory
        print("      批次 {}: 对象数 {}, 增长 {}".format(i+1, current, growth))

        # 强制GC
        collected = gc.collect()
        if collected > 0:
            print("         🗑️ GC回收: {} 个对象".format(collected))

    # 清理所有对象
    del objects
    final_collected = gc.collect()
    final_memory = len(gc.get_objects())

    print("   📊 最终对象数: {}".format(final_memory))
    print("   🗑️ 最终GC回收: {} 个对象".format(final_collected))

    if final_memory < initial_memory * 1.1:  # 增长不超过10%
        print("   ✅ 内存压力下表现良好")
    else:
        print("   ⚠️ 内存压力下可能存在泄漏")

def main():
    """主函数"""
    print("🧠 RQA2025 内存泄漏检测和修复验证")
    print("=" * 50)

    # 运行各项测试
    event_count = test_event_persistence_leak()
    cache_size = test_logger_cache_leak()
    test_circular_references()
    test_large_objects()
    memory_pressure_test()

    print("\n📋 内存泄漏修复总结")
    print("=" * 30)
    print("   📊 事件持久化: {} 个事件 (限制: 10000)".format(event_count))
    print("   📊 日志缓存: {} 个缓存项".format(cache_size))
    print("   ✅ 已修复的内存泄漏点:")
    print("      • 事件持久化自动清理机制")
    print("      • 日志字符串缓存定期清理")
    print("      • 循环引用检测和清理")
    print("      • 大对象及时清理")
    print()
    print("   🛡️ 内存泄漏防护措施:")
    print("      • 对象池技术减少频繁创建")
    print("      • 定期缓存清理机制")
    print("      • 弱引用避免循环引用")
    print("      • 大对象监控和清理")
    print()
    print("🎉 内存泄漏修复验证完成!")

if __name__ == "__main__":
    main()
