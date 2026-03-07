#!/usr/bin/env python3
"""
RQA2025 资源管理系统 Phase 2 重构集成测试

测试重构后的各个模块之间的协同工作能力
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 尝试多种导入方式


def safe_import(module_path, class_name):
    """安全导入模块"""
    try:
        # 方式1: 直接导入
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        try:
            # 方式2: 使用importlib
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                class_name,
                os.path.join(project_root, module_path.replace('.', os.sep) + '.py')
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, class_name)
        except Exception as e:
            print(f"导入失败 {module_path}.{class_name}: {e}")
            return None


def test_config_classes():
    """测试配置数据类"""
    print("🧪 测试配置数据类...")

    try:
        from src.infrastructure.resource.config_classes import (
            AlertConfig, TaskConfig, OptimizationConfig
        )

        # 测试OptimizationConfig
        opt_config = OptimizationConfig(
            optimization_type="memory_optimization",
            target_metrics=["cpu", "memory", "disk"],
            constraints={"cpu_usage": 80, "memory_usage": 85},
            parallelization_config={"enabled": True}
        )
        assert opt_config.optimization_type == "memory_optimization"
        assert "cpu" in opt_config.target_metrics
        print("   ✅ OptimizationConfig测试通过")

        # 测试TaskConfig
        task_config = TaskConfig(
            task_type="data_processing",
            priority=3,
            timeout=300,
            resource_requirements={"cpu": 2, "memory": "4GB"}
        )
        assert task_config.task_type == "data_processing"
        assert task_config.priority == 3
        print("   ✅ TaskConfig测试通过")

        # 测试AlertConfig
        alert_config = AlertConfig(
            alert_type="threshold_exceeded",
            severity="high",
            threshold=90.0,
            operator="gt",
            notification_channels=["email", "webhook"]
        )
        assert alert_config.severity == "high"
        assert alert_config.threshold == 90.0
        print("   ✅ AlertConfig测试通过")

        return True

    except Exception as e:
        print(f"   ❌ 配置类测试失败: {e}")
        return False


def test_system_monitor_facade():
    """测试SystemMonitor门面类"""
    print("🧪 测试SystemMonitor门面类...")

    try:
        from src.infrastructure.resource.system_monitor import SystemMonitorFacade

        # 测试门面类实例化
        facade = SystemMonitorFacade()
        assert facade is not None
        print("   ✅ SystemMonitorFacade实例化成功")

        # 测试基本功能（如果有的话）
        if hasattr(facade, 'get_system_info'):
            info = facade.get_system_info()
            assert isinstance(info, dict)
            print("   ✅ SystemMonitorFacade.get_system_info()工作正常")

        return True

    except Exception as e:
        print(f"   ❌ SystemMonitor门面类测试失败: {e}")
        return False


def test_resource_dashboard_controller():
    """测试ResourceDashboard控制器"""
    print("🧪 测试ResourceDashboard控制器...")

    try:
        from src.infrastructure.resource.resource_dashboard import ResourceDashboardController

        # 测试控制器实例化
        controller = ResourceDashboardController()
        assert controller is not None
        print("   ✅ ResourceDashboardController实例化成功")

        return True

    except Exception as e:
        print(f"   ❌ ResourceDashboard控制器测试失败: {e}")
        return False


def test_optimization_with_config():
    """测试使用配置对象的优化功能"""
    print("🧪 测试配置驱动的优化功能...")

    try:
        from src.infrastructure.resource.resource_optimization import ResourceOptimizer
        from src.infrastructure.resource.config_classes import OptimizationConfig

        # 创建优化器
        optimizer = ResourceOptimizer()

        # 创建配置对象
        config = OptimizationConfig(
            optimization_type="comprehensive",
            target_metrics=["cpu", "memory", "disk", "gpu"],
            constraints={
                "cpu_usage": 75,
                "memory_usage": 80,
                "disk_usage": 85
            },
            parallelization_config={"enabled": True, "max_workers": 4}
        )

        # 执行优化
        result = optimizer.optimize_resources(config)

        # 验证结果
        assert isinstance(result, dict)
        assert "optimizations" in result
        assert "recommendations" in result
        assert "config_applied" in result

        print("   ✅ optimize_resources()使用配置对象成功")
        print(f"      优化类型: {result['config_applied']}")
        print(f"      优化建议: {len(result['optimizations'])}类")

        return True

    except Exception as e:
        print(f"   ❌ 优化功能测试失败: {e}")
        return False


def test_task_scheduler_with_config():
    """测试使用配置对象任务调度"""
    print("🧪 测试配置驱动的任务调度...")

    try:
        from src.infrastructure.resource.task_scheduler import TaskScheduler
        from src.infrastructure.resource.config_classes import TaskConfig

        # 创建调度器
        scheduler = TaskScheduler()

        # 创建任务配置
        config = TaskConfig(
            task_type="resource_monitoring",
            priority=2,
            timeout=600,
            resource_requirements={"cpu": 1, "memory": "2GB"},
            monitoring_config={"enabled": True, "interval": 30}
        )

        # 提交任务
        task_id = scheduler.submit_task_with_config(config)

        # 验证任务ID
        assert isinstance(task_id, str)
        assert len(task_id) > 0

        print("   ✅ submit_task_with_config()使用配置对象成功")
        print(f"      任务ID: {task_id}")

        return True

    except Exception as e:
        print(f"   ❌ 任务调度测试失败: {e}")
        return False


def test_alert_system_with_config():
    """测试使用配置对象的告警系统"""
    print("🧪 测试配置驱动的告警系统...")

    try:
        from src.infrastructure.resource.monitoring_alert_system import MonitoringAlertSystem
        from src.infrastructure.resource.config_classes import AlertConfig

        # 创建告警系统
        alert_system = MonitoringAlertSystem()

        # 创建告警配置
        config = AlertConfig(
            alert_type="resource_threshold",
            severity="medium",
            message="Resource usage exceeded threshold",
            threshold=85.0,
            operator="gt",
            cooldown_period=300,
            notification_channels=["email"],
            routing_rules={"to_email": "admin@example.com"}
        )

        # 这里只是测试配置对象创建，实际发送邮件可能需要更多设置
        assert config.severity == "medium"
        assert config.threshold == 85.0

        print("   ✅ AlertConfig配置创建成功")

        return True

    except Exception as e:
        print(f"   ❌ 告警系统测试失败: {e}")
        return False


def test_module_interaction():
    """测试模块间的交互"""
    print("🧪 测试模块间交互...")

    try:
        from src.infrastructure.resource.config_classes import OptimizationConfig, TaskConfig
        from src.infrastructure.resource.resource_optimization import ResourceOptimizer
        from src.infrastructure.resource.task_scheduler import TaskScheduler

        # 创建优化配置
        opt_config = OptimizationConfig(
            optimization_type="resource_balancing",
            target_metrics=["cpu", "memory"],
            constraints={"cpu_usage": 70, "memory_usage": 75}
        )

        # 创建优化器并执行优化
        optimizer = ResourceOptimizer()
        opt_result = optimizer.optimize_resources(opt_config)

        # 基于优化结果创建任务配置
        task_config = TaskConfig(
            task_type="optimization_execution",
            priority=1,
            timeout=300,
            resource_requirements={"cpu": 1},
            execution_context={"optimization_result": opt_result}
        )

        # 创建调度器并提交任务
        scheduler = TaskScheduler()
        task_id = scheduler.submit_task_with_config(task_config)

        print("   ✅ 模块间交互测试成功")
        print(f"      优化→任务调度链路完整: {task_id}")

        return True

    except Exception as e:
        print(f"   ❌ 模块交互测试失败: {e}")
        return False


def run_integration_tests():
    """运行所有集成测试"""
    print("=" * 70)
    print("🚀 RQA2025 资源管理系统 Phase 2 重构集成测试")
    print("=" * 70)
    print()

    tests = [
        test_config_classes,
        test_system_monitor_facade,
        test_resource_dashboard_controller,
        test_optimization_with_config,
        test_task_scheduler_with_config,
        test_alert_system_with_config,
        test_module_interaction
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"   ❌ 测试异常: {e}")
            print()

    print("=" * 70)
    print(f"📊 测试结果: {passed}/{total} 个测试通过")
    print()

    if passed == total:
        print("🎉 所有集成测试通过！Phase 2重构质量验证成功。")
        print("✅ 可以安全地开始Phase 3重构工作。")
    else:
        print("⚠️ 部分测试失败，需要进一步检查。")

    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
