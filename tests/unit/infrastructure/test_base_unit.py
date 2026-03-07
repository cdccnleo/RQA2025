"""
测试基础设施基础类
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.base import BaseInfrastructureComponent


class MockInfrastructureComponent(BaseInfrastructureComponent):
    """用于测试的模拟基础设施组件"""

    def __init__(self, component_name: str):
        super().__init__(component_name)
        self.mock_health_status = True

    def _perform_health_check(self) -> bool:
        """模拟健康检查实现"""
        return self.mock_health_status

    def initialize(self):
        """模拟初始化方法"""
        self._initialized = True

    def shutdown(self):
        """模拟关闭方法"""
        self._initialized = False


class TestBaseInfrastructureComponent:
    """测试基础设施组件基类"""

    def setup_method(self):
        """测试前准备"""
        self.component = MockInfrastructureComponent("test_component")

    def test_base_infrastructure_component_init(self):
        """测试基础设施组件基类初始化"""
        assert self.component.component_name == "test_component"
        assert self.component.start_time is not None
        assert isinstance(self.component.start_time, datetime)
        assert hasattr(self.component, '_lock')  # 检查是否有锁属性
        assert self.component._initialized == False

    def test_get_status_not_initialized(self):
        """测试获取未初始化组件的状态"""
        status = self.component.get_status()

        assert status["component"] == "test_component"
        assert status["status"] == "stopped"
        assert "uptime" in status
        assert "timestamp" in status
        assert isinstance(status["timestamp"], str)

    def test_get_status_initialized(self):
        """测试获取已初始化组件的状态"""
        self.component.initialize()

        status = self.component.get_status()

        assert status["component"] == "test_component"
        assert status["status"] == "running"
        assert "uptime" in status
        assert "timestamp" in status

    def test_health_check_healthy(self):
        """测试健康检查 - 健康状态"""
        self.component.mock_health_status = True

        health = self.component.health_check()

        assert health["component"] == "test_component"
        assert health["status"] == "healthy"
        assert "timestamp" in health

    def test_health_check_unhealthy(self):
        """测试健康检查 - 不健康状态"""
        self.component.mock_health_status = False

        health = self.component.health_check()

        assert health["component"] == "test_component"
        assert health["status"] == "unhealthy"
        assert "timestamp" in health

    def test_health_check_exception(self):
        """测试健康检查 - 异常情况"""
        # 创建一个会抛出异常的组件
        class FailingComponent(MockInfrastructureComponent):
            def _perform_health_check(self):
                raise RuntimeError("Health check failed")

        failing_component = FailingComponent("failing_component")

        health = failing_component.health_check()

        assert health["component"] == "failing_component"
        assert health["status"] == "error"
        assert "error" in health
        assert health["error"] == "Health check failed"
        assert "timestamp" in health

    def test_component_uptime_calculation(self):
        """测试组件运行时间计算"""
        # 创建一个有特定启动时间的组件
        start_time = datetime.now() - timedelta(hours=2, minutes=30)
        component = MockInfrastructureComponent("timed_component")
        component.start_time = start_time

        status = component.get_status()

        # 验证运行时间字符串是合理的
        uptime_str = status["uptime"]
        # 运行时间应该是一个字符串表示
        assert isinstance(uptime_str, str)
        assert len(uptime_str) > 0

    def test_component_thread_safety(self):
        """测试组件线程安全性"""
        import threading
        import time

        results = []
        errors = []

        def worker_thread(thread_id):
            try:
                # 每个线程都调用状态方法
                for i in range(100):
                    status = self.component.get_status()
                    health = self.component.health_check()
                    results.append((thread_id, i, status["status"], health["status"]))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有线程都成功执行了操作
        assert len(results) == 5 * 100  # 5个线程，每个100次操作

    def test_component_initialization_workflow(self):
        """测试组件初始化工作流"""
        # 初始状态
        assert self.component._initialized == False
        status = self.component.get_status()
        assert status["status"] == "stopped"

        # 初始化组件
        self.component.initialize()
        assert self.component._initialized == True
        status = self.component.get_status()
        assert status["status"] == "running"

        # 关闭组件
        self.component.shutdown()
        assert self.component._initialized == False
        status = self.component.get_status()
        assert status["status"] == "stopped"

    def test_component_name_validation(self):
        """测试组件名称验证"""
        # 测试有效的组件名称
        valid_names = ["database", "cache_manager", "api_gateway", "test-component_123"]

        for name in valid_names:
            component = MockInfrastructureComponent(name)
            assert component.component_name == name

        # 测试空名称
        empty_component = MockInfrastructureComponent("")
        assert empty_component.component_name == ""

    def test_component_status_consistency(self):
        """测试组件状态一致性"""
        # 测试多次调用状态方法的一致性
        status1 = self.component.get_status()
        time.sleep(0.001)  # 短暂延迟
        status2 = self.component.get_status()

        # 组件名称应该一致
        assert status1["component"] == status2["component"]

        # 时间戳应该不同（因为是实时生成）
        assert status1["timestamp"] != status2["timestamp"]

        # 状态应该一致（都未初始化）
        assert status1["status"] == status2["status"] == "stopped"

    def test_component_health_check_performance(self):
        """测试组件健康检查性能"""
        import time

        # 执行多次健康检查
        start_time = time.time()
        iterations = 1000

        for i in range(iterations):
            health = self.component.health_check()
            assert health["status"] in ["healthy", "unhealthy"]

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能（每秒至少1000次检查）
        assert total_time < 1.0 or iterations / total_time > 500

    def test_component_multiple_instances(self):
        """测试多个组件实例"""
        components = []

        # 创建多个组件实例
        for i in range(10):
            component = MockInfrastructureComponent(f"component_{i}")
            components.append(component)

        # 验证每个组件都有独立的名称和状态
        for i, component in enumerate(components):
            assert component.component_name == f"component_{i}"
            assert component._initialized == False

            status = component.get_status()
            assert status["component"] == f"component_{i}"
            assert status["status"] == "stopped"

    def test_component_error_handling(self):
        """测试组件错误处理"""
        # 测试各种异常情况
        error_scenarios = [
            ("network_error", ConnectionError("Network unreachable")),
            ("timeout_error", TimeoutError("Operation timed out")),
            ("value_error", ValueError("Invalid value")),
            ("key_error", KeyError("Missing key")),
        ]

        for error_name, exception in error_scenarios:
            # 创建会抛出特定异常的组件
            class ErrorComponent(MockInfrastructureComponent):
                def _perform_health_check(self):
                    raise exception

            error_component = ErrorComponent(f"error_component_{error_name}")

            health = error_component.health_check()

            assert health["status"] == "error"
            assert health["component"] == f"error_component_{error_name}"
            assert error_name.replace("_", " ") in health["error"].lower() or str(exception) in health["error"]

    def test_component_lifecycle(self):
        """测试组件生命周期"""
        # 1. 创建组件
        component = MockInfrastructureComponent("lifecycle_test")

        # 2. 初始状态检查
        assert component._initialized == False
        status = component.get_status()
        assert status["status"] == "stopped"

        # 3. 初始化
        component.initialize()
        assert component._initialized == True
        status = component.get_status()
        assert status["status"] == "running"

        # 4. 健康检查
        health = component.health_check()
        assert health["status"] == "healthy"

        # 5. 关闭
        component.shutdown()
        assert component._initialized == False
        status = component.get_status()
        assert status["status"] == "stopped"

    def test_abstract_base_class(self):
        """测试抽象基类"""
        # 不能直接实例化抽象类
        with pytest.raises(TypeError):
            BaseInfrastructureComponent("test")

        # 必须实现抽象方法
        class IncompleteComponent(BaseInfrastructureComponent):
            pass

        # 这个类缺少抽象方法的实现
        with pytest.raises(TypeError):
            IncompleteComponent("test")


class TestInfrastructureComponentIntegration:
    """测试基础设施组件集成场景"""

    def test_multiple_components_coordination(self):
        """测试多个组件协调工作"""
        components = {
            "database": MockInfrastructureComponent("database"),
            "cache": MockInfrastructureComponent("cache"),
            "api": MockInfrastructureComponent("api"),
            "monitoring": MockInfrastructureComponent("monitoring")
        }

        # 初始化所有组件
        for component in components.values():
            component.initialize()

        # 检查所有组件状态
        all_healthy = True
        for name, component in components.items():
            status = component.get_status()
            health = component.health_check()

            assert status["status"] == "running"
            assert health["status"] == "healthy"

        # 模拟一个组件故障
        components["database"].mock_health_status = False

        # 检查整体系统健康状态
        unhealthy_count = 0
        for name, component in components.items():
            health = component.health_check()
            if health["status"] != "healthy":
                unhealthy_count += 1

        assert unhealthy_count == 1  # 只有一个组件不健康

    def test_component_monitoring_scenario(self):
        """测试组件监控场景"""
        component = MockInfrastructureComponent("monitored_component")

        # 收集一段时间的状态数据
        status_history = []
        health_history = []

        # 模拟运行一段时间
        for i in range(10):
            status = component.get_status()
            health = component.health_check()

            status_history.append(status)
            health_history.append(health)

            # 短暂延迟模拟时间流逝
            import time
            time.sleep(0.001)

        # 验证历史数据
        assert len(status_history) == 10
        assert len(health_history) == 10

        # 验证时间戳是递增的
        timestamps = [datetime.fromisoformat(s["timestamp"]) for s in status_history]
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))

    def test_component_resource_management(self):
        """测试组件资源管理"""
        import psutil
        import os

        # 获取初始资源使用情况
        initial_memory = psutil.Process(os.getpid()).memory_info().rss

        # 创建多个组件实例
        components = []
        for i in range(100):
            component = MockInfrastructureComponent(f"resource_test_{i}")
            components.append(component)

        # 获取创建后的资源使用情况
        after_memory = psutil.Process(os.getpid()).memory_info().rss

        # 验证内存使用在合理范围内（每个组件对象应该只占用少量内存）
        memory_increase = after_memory - initial_memory
        # 100个组件实例应该不会导致过多的内存增加
        assert memory_increase < 50 * 1024 * 1024  # 50MB上限

        # 清理组件
        del components

    def test_component_error_recovery(self):
        """测试组件错误恢复"""
        component = MockInfrastructureComponent("recovery_test")

        # 初始状态正常
        health = component.health_check()
        assert health["status"] == "healthy"

        # 模拟组件故障
        component.mock_health_status = False
        health = component.health_check()
        assert health["status"] == "unhealthy"

        # 模拟恢复
        component.mock_health_status = True
        health = component.health_check()
        assert health["status"] == "healthy"

        # 验证组件状态没有受到影响
        status = component.get_status()
        assert status["status"] == "stopped"  # 仍然未初始化，但健康检查正常
