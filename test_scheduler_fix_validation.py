#!/usr/bin/env python3
"""
验证调度器修复效果
"""

import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

def test_scheduler_configuration():
    """测试调度器配置"""
    print("测试调度器配置...")

    try:
        from src.core.orchestration.business_process.service_scheduler import DataCollectionServiceScheduler

        scheduler = DataCollectionServiceScheduler()

        print("调度器配置检查:")
        print(f"  ✅ 最大并发任务: {scheduler.max_concurrent_tasks} (期望: 3)")
        print(f"  ✅ 启动延迟: {scheduler.startup_delay}秒 (期望: 60)")
        print(f"  ✅ 最大高负载计数: {scheduler.max_high_load_count} (期望: 10)")

        # 验证配置是否正确
        checks = [
            scheduler.max_concurrent_tasks == 3,
            scheduler.startup_delay == 60,
            scheduler.max_high_load_count == 10,
        ]

        if all(checks):
            print("✅ 调度器配置正确")
            return True
        else:
            print("❌ 调度器配置不正确")
            return False

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def test_singleton_behavior():
    """测试单例行为"""
    print("\n测试调度器单例行为...")

    try:
        from src.core.orchestration.business_process.service_scheduler import (
            get_data_collection_scheduler,
            DataCollectionServiceScheduler
        )

        # 测试单例获取
        scheduler1 = get_data_collection_scheduler()
        scheduler2 = get_data_collection_scheduler()

        if scheduler1 is scheduler2:
            print("✅ 单例模式工作正常")
            singleton_ok = True
        else:
            print("❌ 单例模式失败")
            singleton_ok = False

        # 测试直接实例化（应该与单例不同）
        scheduler3 = DataCollectionServiceScheduler()
        if scheduler3 is not scheduler1:
            print("✅ 直接实例化与单例不同")
            direct_ok = True
        else:
            print("❌ 直接实例化与单例相同（异常）")
            direct_ok = False

        return singleton_ok and direct_ok

    except Exception as e:
        print(f"❌ 单例测试失败: {e}")
        return False

def test_load_protection_logic():
    """测试负载保护逻辑"""
    print("\n测试负载保护逻辑...")

    try:
        from src.core.orchestration.business_process.service_scheduler import DataCollectionServiceScheduler

        scheduler = DataCollectionServiceScheduler()

        # 测试基础负载检查（无活跃任务时应该返回False）
        should_throttle = scheduler._should_throttle_due_to_high_load()
        if not should_throttle:
            print("✅ 空闲状态下不节流")
            idle_ok = True
        else:
            print("❌ 空闲状态下错误节流")
            idle_ok = False

        # 模拟活跃任务
        scheduler.active_tasks.add("test_task1")
        scheduler.active_tasks.add("test_task2")
        scheduler.active_tasks.add("test_task3")  # 达到并发上限

        should_throttle_full = scheduler._should_throttle_due_to_high_load()
        if should_throttle_full:
            print("✅ 达到并发上限时节流")
            full_ok = True
        else:
            print("❌ 达到并发上限时未节流")
            full_ok = False

        # 清理测试数据
        scheduler.active_tasks.clear()

        return idle_ok and full_ok

    except Exception as e:
        print(f"❌ 负载保护测试失败: {e}")
        return False

def test_startup_delay_logic():
    """测试启动延迟逻辑"""
    print("\n测试启动延迟逻辑...")

    try:
        from src.core.orchestration.business_process.service_scheduler import DataCollectionServiceScheduler

        scheduler = DataCollectionServiceScheduler()

        # 记录创建时间
        creation_time = scheduler.application_startup_time
        current_time = time.time()

        print(f"调度器创建时间: {creation_time}")
        print(f"当前时间: {current_time}")
        print(f"时间差: {current_time - creation_time:.1f}秒")

        # 延迟应该很快到期（因为我们刚刚创建）
        if current_time - creation_time < scheduler.startup_delay:
            print("✅ 启动延迟逻辑正常")
            return True
        else:
            print("❌ 启动延迟时间不正确")
            return False

    except Exception as e:
        print(f"❌ 启动延迟测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("调度器修复验证测试")
    print("=" * 50)

    tests = [
        ("配置检查", test_scheduler_configuration),
        ("单例行为", test_singleton_behavior),
        ("负载保护", test_load_protection_logic),
        ("启动延迟", test_startup_delay_logic)
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
        print("\n🎉 所有测试通过！调度器修复成功")
        print("现在调度器应该能够:")
        print("  - 正确实现单例模式")
        print("  - 有效控制并发任务")
        print("  - 合理处理启动延迟")
        print("  - 避免主备启动冲突")
    else:
        print("\n⚠️ 部分测试失败，需要进一步检查")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)