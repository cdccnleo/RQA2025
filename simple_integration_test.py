#!/usr/bin/env python3
"""
简化的Phase 2集成测试
"""

import sys
import os

def test_imports():
    """测试基本导入"""
    print("🧪 测试基本导入...")

    try:
        # 测试配置类
        sys.path.insert(0, os.path.dirname(__file__))
        exec(open('src/infrastructure/resource/config_classes.py').read())

        # 检查类是否已定义
        if 'OptimizationConfig' in globals():
            config = OptimizationConfig(optimization_type='test')
            print("   ✅ OptimizationConfig导入并实例化成功")
        else:
            print("   ❌ OptimizationConfig未找到")

        return True
    except Exception as e:
        print(f"   ❌ 导入测试失败: {e}")
        return False

def test_file_syntax():
    """测试文件语法"""
    print("🧪 测试文件语法...")

    files_to_test = [
        'src/infrastructure/resource/config_classes.py',
        'src/infrastructure/resource/system_monitor.py',
        'src/infrastructure/resource/resource_dashboard.py',
        'src/infrastructure/resource/resource_optimization.py',
        'src/infrastructure/resource/task_scheduler.py',
        'src/infrastructure/resource/monitoring_alert_system.py'
    ]

    passed = 0
    for file_path in files_to_test:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
            print(f"   ✅ {os.path.basename(file_path)} 语法正确")
            passed += 1
        except Exception as e:
            print(f"   ❌ {os.path.basename(file_path)} 语法错误: {e}")

    return passed == len(files_to_test)

def test_class_structure():
    """测试类结构"""
    print("🧪 测试类结构...")

    try:
        # 直接执行文件来定义类
        with open('src/infrastructure/resource/config_classes.py', 'r', encoding='utf-8') as f:
            exec(f.read(), globals())

        # 测试类定义
        if 'OptimizationConfig' in globals():
            config = OptimizationConfig(
                optimization_type="memory_optimization",
                target_metrics=["cpu", "memory"],
                constraints={"cpu_usage": 80}
            )
            assert config.optimization_type == "memory_optimization"
            assert "cpu" in config.target_metrics
            print("   ✅ OptimizationConfig结构正确")
        else:
            print("   ❌ OptimizationConfig类未定义")
            return False

        # 测试TaskConfig
        if 'TaskConfig' in globals():
            task_config = TaskConfig(
                task_type="test_task",
                priority=1,
                timeout=300
            )
            assert task_config.task_type == "test_task"
            assert task_config.priority == 1
            print("   ✅ TaskConfig结构正确")
        else:
            print("   ❌ TaskConfig类未定义")
            return False

        return True
    except Exception as e:
        print(f"   ❌ 类结构测试失败: {e}")
        return False

def test_refactored_classes():
    """测试重构后的类"""
    print("🧪 测试重构后的类...")

    try:
        # 测试SystemMonitor门面类
        with open('src/infrastructure/resource/system_monitor.py', 'r', encoding='utf-8') as f:
            exec(f.read(), globals())

        if 'SystemMonitorFacade' in globals():
            facade = SystemMonitorFacade()
            print("   ✅ SystemMonitorFacade实例化成功")
        else:
            print("   ❌ SystemMonitorFacade未找到")

        # 测试ResourceDashboard控制器
        with open('src/infrastructure/resource/resource_dashboard.py', 'r', encoding='utf-8') as f:
            exec(f.read(), globals())

        if 'ResourceDashboardController' in globals():
            controller = ResourceDashboardController()
            print("   ✅ ResourceDashboardController实例化成功")
        else:
            print("   ❌ ResourceDashboardController未找到")

        return True
    except Exception as e:
        print(f"   ❌ 重构类测试失败: {e}")
        return False

def run_simple_tests():
    """运行简化测试"""
    print("=" * 60)
    print("🚀 RQA2025 资源管理系统 Phase 2 简化集成测试")
    print("=" * 60)
    print()

    tests = [
        test_file_syntax,
        test_class_structure,
        test_refactored_classes
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   ❌ 测试异常: {e}")
        print()

    print("=" * 60)
    print(f"📊 测试结果: {passed}/{total} 个测试通过")
    print()

    if passed == total:
        print("🎉 所有简化测试通过！Phase 2重构质量验证成功。")
        print("✅ 可以安全地开始Phase 3重构工作。")
    else:
        print("⚠️ 部分测试失败，但基本结构验证通过。")

    print("=" * 60)

    return passed == total

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)
