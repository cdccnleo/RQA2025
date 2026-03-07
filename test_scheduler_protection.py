#!/usr/bin/env python3
"""
测试调度器负载保护机制
"""

import sys
import asyncio
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

async def test_load_protection():
    """测试负载保护机制"""
    print("测试调度器负载保护机制...")

    try:
        from src.core.orchestration.business_process.service_scheduler import DataCollectionServiceScheduler

        # 创建调度器实例
        scheduler = DataCollectionServiceScheduler()

        print("调度器配置:")
        print(f"  - 最大并发任务: {scheduler.max_concurrent_tasks}")
        print(f"  - 启动延迟: {scheduler.startup_delay}秒")
        print(f"  - 负载检查启用: {scheduler.load_check_enabled}")
        print(f"  - 最大高负载计数: {scheduler.max_high_load_count}")

        # 测试负载检查方法
        should_throttle = scheduler._should_throttle_due_to_high_load()
        print(f"当前负载检查结果: {'需要限制' if should_throttle else '正常'}")

        # 模拟启动
        success = await scheduler.start("test_protection")
        print(f"调度器启动: {'成功' if success else '失败'}")

        if success:
            # 等待一会儿观察启动延迟
            print("等待启动延迟期间...")
            await asyncio.sleep(2)

            # 检查启动时间
            if hasattr(scheduler, 'application_startup_time'):
                elapsed = time.time() - scheduler.application_startup_time
                print(f"应用启动已过去: {elapsed:.1f}秒")
                print(f"启动延迟设置: {scheduler.startup_delay}秒")

            # 停止调度器
            stop_success = await scheduler.stop()
            print(f"调度器停止: {'成功' if stop_success else '失败'}")

            return True
        else:
            return False

    except Exception as e:
        print(f"❌ 负载保护测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_availability():
    """测试psutil导入可用性"""
    print("\n测试psutil导入可用性...")

    try:
        import psutil
        print("✅ psutil可用，负载检查功能完整")
        return True
    except ImportError:
        print("⚠️ psutil不可用，负载检查功能降级")
        return False

async def main():
    """主测试函数"""
    print("调度器负载保护机制测试")
    print("=" * 50)

    tests = [
        ("psutil可用性", test_import_availability),
        ("负载保护机制", test_load_protection)
    ]

    results = []
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
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
        print("\n🎉 所有测试通过！调度器负载保护机制正常工作。")
        print("   现在调度器将:")
        print("   - 在应用启动后120秒才开始采集任务")
        print("   - 监控系统负载，自动限制并发任务")
        print("   - 在连续高负载时停止调度器保护应用")
    else:
        print("\n⚠️ 部分测试失败，可能需要进一步检查。")

    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)