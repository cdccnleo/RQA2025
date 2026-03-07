#!/usr/bin/env python3
"""
诊断调度器问题
"""

def diagnose_without_async():
    """同步诊断，不使用async"""
    print("诊断数据采集调度器问题...")

    try:
        import sys
        from pathlib import Path

        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root / 'src'))

        # 检查调度器单例
        print("1. 检查调度器单例状态...")
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler

        scheduler1 = get_data_collection_scheduler()
        scheduler2 = get_data_collection_scheduler()

        print(f"   调度器实例1: {id(scheduler1)}")
        print(f"   调度器实例2: {id(scheduler2)}")
        print(f"   单例模式: {'✅ 正常' if scheduler1 is scheduler2 else '❌ 异常'}")

        # 检查基本状态
        print("\n2. 检查调度器基本状态...")
        print(f"   运行状态: {scheduler1.is_running()}")
        print(f"   启动路径: {scheduler1._startup_path}")
        print(f"   启动时间: {scheduler1._startup_time}")

        # 检查并发控制
        print("\n3. 检查并发控制状态...")
        print(f"   最大并发任务: {scheduler1.max_concurrent_tasks}")
        print(f"   当前活跃任务: {len(scheduler1.active_tasks)}")
        print(f"   待处理队列: {len(scheduler1.pending_sources)}")

        # 检查负载保护
        print("\n4. 检查负载保护状态...")
        print(f"   高负载计数: {scheduler1.high_load_count}")
        print(f"   最大高负载计数: {scheduler1.max_high_load_count}")

        # 检查启动延迟
        print("\n5. 检查启动延迟状态...")
        if scheduler1.application_startup_time:
            import time
            elapsed = time.time() - scheduler1.application_startup_time
            remaining = max(0, scheduler1.startup_delay - elapsed)
            print(f"   应用启动已过去: {elapsed:.1f}秒")
            print(f"   启动延迟设置: {scheduler1.startup_delay}秒")
            print(f"   延迟剩余时间: {remaining:.1f}秒")
        else:
            print("   应用启动时间未设置")

        # 检查数据源管理器
        print("\n6. 检查数据源管理器...")
        if scheduler1.data_source_manager:
            try:
                sources = scheduler1.data_source_manager.get_data_sources()
                enabled_sources = [s for s in sources if s.get('enabled', False)]
                print(f"   总数据源数: {len(sources)}")
                print(f"   启用数据源数: {len(enabled_sources)}")
            except Exception as e:
                print(f"   获取数据源失败: {e}")
        else:
            print("   数据源管理器未初始化")

        # 检查事件监听器
        print("\n7. 检查事件监听器状态...")
        try:
            from src.core.orchestration.business_process.app_startup_listener import get_app_startup_listener
            from src.core.event_bus import get_event_bus
            from src.core.event_bus.types import EventType

            listener = get_app_startup_listener()
            event_bus = get_event_bus()

            print(f"   监听器已注册: {listener._registered}")
            print(f"   事件总线实例ID: {id(event_bus)}")
            print(f"   监听器事件总线ID: {id(listener.event_bus) if listener.event_bus else 'None'}")
            print(f"   事件订阅者数量: {event_bus.get_subscriber_count(EventType.APPLICATION_STARTUP_COMPLETE)}")
        except Exception as e:
            print(f"   检查事件监听器失败: {e}")

        # 分析问题
        print("\n8. 问题分析...")
        issues = []

        if not scheduler1.is_running():
            issues.append("调度器未运行")

        if scheduler1.application_startup_time:
            import time
            elapsed = time.time() - scheduler1.application_startup_time
            if elapsed < scheduler1.startup_delay:
                issues.append("仍在启动延迟中")

        if scheduler1.high_load_count >= scheduler1.max_high_load_count:
            issues.append("连续高负载，调度器已停止")

        if not scheduler1.data_source_manager:
            issues.append("数据源管理器未初始化")

        if issues:
            print("   发现的问题:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("   未发现明显问题")

        return len(issues) == 0

    except Exception as e:
        print(f"诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = diagnose_without_async()
    print(f"\n诊断结果: {'✅ 正常' if success else '❌ 发现问题'}")