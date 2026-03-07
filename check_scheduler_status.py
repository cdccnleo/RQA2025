#!/usr/bin/env python3
"""
检查数据采集调度器状态
"""

import sys
import asyncio
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

async def check_scheduler_status():
    """检查调度器状态"""
    print("检查数据采集调度器状态...")

    try:
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler

        # 获取调度器实例
        scheduler = get_data_collection_scheduler()

        print("调度器基本信息:")
        print(f"  - 实例ID: {id(scheduler)}")
        print(f"  - 运行状态: {scheduler.is_running()}")
        print(f"  - 启动路径: {scheduler._startup_path}")
        print(f"  - 启动时间: {scheduler._startup_time}")

        # 检查并发控制状态
        print(f"\n并发控制状态:")
        print(f"  - 最大并发任务: {scheduler.max_concurrent_tasks}")
        print(f"  - 当前活跃任务数: {len(scheduler.active_tasks)}")
        print(f"  - 待处理队列长度: {len(scheduler.pending_sources)}")
        print(f"  - 活跃任务: {list(scheduler.active_tasks) if scheduler.active_tasks else '无'}")

        # 检查负载保护状态
        print(f"\n负载保护状态:")
        print(f"  - 负载检查启用: {scheduler.load_check_enabled}")
        print(f"  - 高负载计数: {scheduler.high_load_count}")
        print(f"  - 最大高负载计数: {scheduler.max_high_load_count}")

        # 检查启动延迟状态
        print(f"\n启动延迟状态:")
        print(f"  - 启动延迟设置: {scheduler.startup_delay}秒")
        print(f"  - 应用启动时间: {scheduler.application_startup_time}")
        if scheduler.application_startup_time:
            elapsed = time.time() - scheduler.application_startup_time
            print(f"  - 应用启动已过去: {elapsed:.1f}秒")
            remaining = max(0, scheduler.startup_delay - elapsed)
            print(f"  - 启动延迟剩余: {remaining:.1f}秒")

        # 获取详细状态
        try:
            status = scheduler.get_status()
            print(f"\n调度器详细状态:")
            for key, value in status.items():
                print(f"  - {key}: {value}")
        except Exception as e:
            print(f"获取详细状态失败: {e}")

        # 检查数据源状态
        if scheduler.data_source_manager:
            try:
                sources = scheduler.data_source_manager.get_data_sources()
                enabled_sources = [s for s in sources if s.get('enabled', False)]
                print(f"\n数据源状态:")
                print(f"  - 总数据源数: {len(sources)}")
                print(f"  - 启用数据源数: {len(enabled_sources)}")
                if enabled_sources:
                    print("  - 启用的数据源:")
                    for source in enabled_sources[:3]:  # 只显示前3个
                        print(f"    * {source.get('id', 'unknown')}: {source.get('name', 'unknown')}")
                    if len(enabled_sources) > 3:
                        print(f"    ... 还有 {len(enabled_sources) - 3} 个数据源")
            except Exception as e:
                print(f"获取数据源状态失败: {e}")
        else:
            print("数据源管理器未初始化")

        return scheduler.is_running()

    except Exception as e:
        print(f"检查调度器状态失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def check_for_multiple_schedulers():
    """检查是否存在多个调度器实例"""
    print("\n检查是否存在多个调度器实例...")

    try:
        # 创建多个调度器实例并比较
        from src.core.orchestration.business_process.service_scheduler import (
            get_data_collection_scheduler,
            DataCollectionServiceScheduler
        )

        # 获取单例实例
        scheduler1 = get_data_collection_scheduler()
        scheduler2 = get_data_collection_scheduler()

        print(f"单例实例比较:")
        print(f"  - 实例1 ID: {id(scheduler1)}")
        print(f"  - 实例2 ID: {id(scheduler2)}")
        print(f"  - 是否相同实例: {scheduler1 is scheduler2}")

        # 创建新实例进行比较
        scheduler3 = DataCollectionServiceScheduler()
        print(f"  - 新实例 ID: {id(scheduler3)}")
        print(f"  - 新实例是否与单例相同: {scheduler3 is scheduler1}")

        if scheduler1 is scheduler2:
            print("✅ 单例模式工作正常")
            return True
        else:
            print("❌ 单例模式失败，存在多个实例")
            return False

    except Exception as e:
        print(f"检查多实例失败: {e}")
        return False

async def diagnose_scheduler_failure():
    """诊断调度器失败原因"""
    print("\n诊断调度器可能失败的原因...")

    issues = []

    try:
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler
        scheduler = get_data_collection_scheduler()

        # 检查基本状态
        if not scheduler.is_running():
            issues.append("调度器未运行")

        # 检查启动延迟
        if scheduler.application_startup_time:
            elapsed = time.time() - scheduler.application_startup_time
            if elapsed < scheduler.startup_delay:
                remaining = scheduler.startup_delay - elapsed
                issues.append(f"仍在启动延迟中，还需等待 {remaining:.1f} 秒")

        # 检查负载状态
        if scheduler._should_throttle_due_to_high_load():
            issues.append("系统负载过高")

        # 检查活跃任务
        if len(scheduler.active_tasks) >= scheduler.max_concurrent_tasks:
            issues.append(f"活跃任务数达到上限 ({len(scheduler.active_tasks)}/{scheduler.max_concurrent_tasks})")

        # 检查高负载计数
        if scheduler.high_load_count >= scheduler.max_high_load_count:
            issues.append(f"连续高负载次数过多 ({scheduler.high_load_count}/{scheduler.max_high_load_count})")

        # 检查数据源
        if scheduler.data_source_manager:
            sources = scheduler.data_source_manager.get_data_sources()
            enabled_sources = [s for s in sources if s.get('enabled', False)]
            if not enabled_sources:
                issues.append("没有启用的数据源")

        if not issues:
            issues.append("未发现明显问题，调度器应该正常运行")

    except Exception as e:
        issues.append(f"诊断过程中出错: {e}")

    print("发现的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    return issues

async def main():
    """主检查函数"""
    print("数据采集调度器状态检查")
    print("=" * 50)

    # 1. 检查调度器状态
    is_running = await check_scheduler_status()

    # 2. 检查单例模式
    singleton_ok = await check_for_multiple_schedulers()

    # 3. 诊断失败原因
    issues = await diagnose_scheduler_failure()

    print("\n" + "=" * 50)
    print("总结报告")
    print("=" * 50)

    print(f"调度器运行状态: {'✅ 运行中' if is_running else '❌ 未运行'}")
    print(f"单例模式状态: {'✅ 正常' if singleton_ok else '❌ 异常'}")
    print(f"发现问题数量: {len(issues)}")

    # 总体评估
    overall_ok = is_running and singleton_ok
    print(f"总体状态: {'✅ 正常' if overall_ok else '❌ 需要修复'}")

    if not overall_ok:
        print("\n建议修复措施:")
        if not is_running:
            print("- 检查调度器启动日志，确认启动失败原因")
            print("- 验证应用启动延迟是否已过")
            print("- 检查系统负载是否过高")
        if not singleton_ok:
            print("- 修复单例模式实现，确保只有一个调度器实例")

    return overall_ok

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)