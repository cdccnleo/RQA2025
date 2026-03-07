#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署验证脚本
验证修复后的基础设施组件在生产环境中的表现
"""

import time
import sys
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_factory_patterns():
    """测试工厂模式组件"""
    print("🔧 测试工厂模式组件...")

    try:
        from src.infrastructure.core.factories.config_factory import ConfigManagerFactory
        from src.infrastructure.core.factories.monitor_factory import MonitorFactory
        from src.infrastructure.core.factories.cache_factory import CacheFactory

        # 测试配置管理器工厂
        config_factory = ConfigManagerFactory()
        config_manager = config_factory.create_manager("unified")
        print("  ✅ 配置管理器工厂: 成功创建统一配置管理器")

        # 测试监控系统工厂
        monitor_factory = MonitorFactory()
        monitor = monitor_factory.create_monitor("unified")
        print("  ✅ 监控系统工厂: 成功创建统一监控器")

        # 测试缓存系统工厂
        cache_factory = CacheFactory()
        cache_manager = cache_factory.create_manager("smart")
        print("  ✅ 缓存系统工厂: 成功创建智能缓存管理器")

        return True

    except Exception as e:
        print(f"  ❌ 工厂模式组件测试失败: {e}")
        traceback.print_exc()
        return False


def test_unified_infrastructure():
    """测试统一基础设施管理器"""
    print("🏗️ 测试统一基础设施管理器...")

    try:
        from src.infrastructure.unified_infrastructure import get_infrastructure_manager

        # 获取基础设施管理器
        manager = get_infrastructure_manager()
        print("  ✅ 基础设施管理器: 成功获取单例实例")

        # 测试组件获取
        config = manager.get_config_manager()
        monitor = manager.get_monitor()
        cache = manager.get_cache()

        print("  ✅ 组件获取: 成功获取所有核心组件")

        # 测试服务注册
        test_service = {"name": "test", "type": "test"}
        manager.register_service("test_service", test_service)

        # 测试服务获取
        retrieved_service = manager.get_service("test_service")
        if retrieved_service == test_service:
            print("  ✅ 服务注册: 成功注册和获取服务")
        else:
            print("  ❌ 服务注册: 服务获取失败")
            return False

        return True

    except Exception as e:
        print(f"  ❌ 统一基础设施管理器测试失败: {e}")
        traceback.print_exc()
        return False


def test_task_scheduler():
    """测试任务调度器"""
    print("📋 测试任务调度器...")

    try:
        from src.infrastructure.scheduler.task_scheduler import TaskScheduler, Task, TaskPriority, TaskStatus

        # 创建任务调度器
        scheduler = TaskScheduler(max_workers=2, queue_size=100)
        print("  ✅ 任务调度器: 成功创建实例")

        # 测试任务提交
        def simple_task():
            time.sleep(0.1)
            return "task completed"

        task = Task(id="test_task", name="test_task", func=simple_task, priority=TaskPriority.HIGH)
        task_id = scheduler.submit_task(task)
        print(f"  ✅ 任务提交: 成功提交任务 {task_id}")

        # 启动调度器
        scheduler.start()
        print("  ✅ 调度器启动: 成功启动任务调度器")

        # 等待任务完成
        time.sleep(0.5)

        # 检查任务状态
        if task.status == TaskStatus.COMPLETED:
            print("  ✅ 任务执行: 任务成功完成")
        else:
            print(f"  ❌ 任务执行: 任务状态异常 {task.status}")
            return False

        # 停止调度器
        scheduler.stop()
        print("  ✅ 调度器停止: 成功停止任务调度器")

        return True

    except Exception as e:
        print(f"  ❌ 任务调度器测试失败: {e}")
        traceback.print_exc()
        return False


def test_infrastructure_core():
    """测试基础设施核心功能"""
    print("⚙️ 测试基础设施核心功能...")

    try:
        from src.infrastructure.core.config.unified_config_manager import UnifiedConfigManager
        from src.infrastructure.core.monitoring.base_monitor import BaseMonitor
        from src.infrastructure.core.cache.smart_cache_strategy import SmartCacheManager

        # 测试配置管理器
        config_manager = UnifiedConfigManager()
        config_manager.set("test_key", "test_value")
        value = config_manager.get("test_key")
        if value == "test_value":
            print("  ✅ 配置管理器: 成功设置和获取配置")
        else:
            print("  ❌ 配置管理器: 配置操作失败")
            return False

        # 测试监控器
        monitor = BaseMonitor()
        monitor.record_metric("test_metric", 100.0)
        print("  ✅ 监控器: 成功记录指标")

        # 测试缓存策略
        cache_strategy = SmartCacheManager()
        cache_strategy.set_cache("test_cache", "test_data", expire=60)
        data = cache_strategy.get_cache("test_cache")
        if data == "test_data":
            print("  ✅ 缓存策略: 成功设置和获取缓存")
        else:
            print("  ❌ 缓存策略: 缓存操作失败")
            return False

        return True

    except Exception as e:
        print(f"  ❌ 基础设施核心功能测试失败: {e}")
        traceback.print_exc()
        return False


def test_performance():
    """测试性能指标"""
    print("🚀 测试性能指标...")

    try:
        # 测试组件创建性能
        start_time = time.time()
        from src.infrastructure.core.factories.config_factory import ConfigManagerFactory
        config_factory = ConfigManagerFactory()
        config_manager = config_factory.create_manager("unified")
        creation_time = (time.time() - start_time) * 1000

        if creation_time < 50:  # 50ms
            print(f"  ✅ 组件创建性能: {creation_time:.2f}ms (优秀)")
        elif creation_time < 100:  # 100ms
            print(f"  ✅ 组件创建性能: {creation_time:.2f}ms (良好)")
        else:
            print(f"  ⚠️ 组件创建性能: {creation_time:.2f}ms (需要优化)")

        # 测试任务调度性能
        from src.infrastructure.scheduler.task_scheduler import TaskScheduler, Task, TaskPriority

        scheduler = TaskScheduler(max_workers=1, queue_size=200)

        start_time = time.time()
        for i in range(100):
            task = Task(id=f"perf_{i}", name=f"perf_{i}",
                        func=lambda: None, priority=TaskPriority.NORMAL)
            scheduler.submit_task(task)
        submission_time = (time.time() - start_time) * 1000

        if submission_time < 100:  # 100ms for 100 tasks
            print(f"  ✅ 任务提交性能: {submission_time:.2f}ms for 100 tasks (优秀)")
        elif submission_time < 200:
            print(f"  ✅ 任务提交性能: {submission_time:.2f}ms for 100 tasks (良好)")
        else:
            print(f"  ⚠️ 任务提交性能: {submission_time:.2f}ms for 100 tasks (需要优化)")

        scheduler.stop()

        return True

    except Exception as e:
        print(f"  ❌ 性能测试失败: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """测试错误处理"""
    print("🛡️ 测试错误处理...")

    try:
        from src.infrastructure.core.factories.config_factory import ConfigManagerFactory

        # 测试无效配置管理器类型
        try:
            ConfigManagerFactory.create_manager("invalid_type")
            print("  ❌ 错误处理: 应该抛出异常")
            return False
        except Exception:
            print("  ✅ 错误处理: 正确抛出异常")

        # 测试任务调度器错误处理
        from src.infrastructure.scheduler.task_scheduler import TaskScheduler, Task, TaskPriority, TaskStatus

        scheduler = TaskScheduler(max_workers=1, queue_size=1)

        # 提交一个会失败的任务
        def failing_task():
            raise Exception("Intentional error for testing")

        task = Task(id="error_task", name="error_task",
                    func=failing_task, priority=TaskPriority.NORMAL)
        scheduler.submit_task(task)

        scheduler.start()
        time.sleep(0.5)

        if task.status == TaskStatus.FAILED:
            print("  ✅ 错误处理: 任务失败时正确设置状态")
        else:
            print(f"  ❌ 错误处理: 任务失败状态异常 {task.status}")
            return False

        scheduler.stop()

        return True

    except Exception as e:
        print(f"  ❌ 错误处理测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🚀 RQA2025 基础设施组件部署验证")
    print("=" * 50)

    test_results = []

    # 运行所有测试
    tests = [
        ("工厂模式组件", test_factory_patterns),
        ("统一基础设施管理器", test_unified_infrastructure),
        ("任务调度器", test_task_scheduler),
        ("基础设施核心功能", test_infrastructure_core),
        ("性能指标", test_performance),
        ("错误处理", test_error_handling),
    ]

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ 测试执行异常: {e}")
            test_results.append((test_name, False))

    # 输出测试结果摘要
    print("\n" + "=" * 50)
    print("📊 测试结果摘要")
    print("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n总体结果: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有测试通过！系统已准备好部署到生产环境。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查失败原因后再进行部署。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
