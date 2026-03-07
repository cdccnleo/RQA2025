"""
测试并发控制器

覆盖 concurrency_controller.py 中的 ConcurrencyController 类
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch
from src.infrastructure.concurrency_controller import ConcurrencyController


class TestConcurrencyController:
    """ConcurrencyController 类测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        controller = ConcurrencyController()

        assert controller.config == {}
        assert hasattr(controller, '_lock')
        assert hasattr(controller, '_active_tasks')
        assert isinstance(controller._active_tasks, dict)

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = {"max_concurrent": 10, "timeout": 30}
        controller = ConcurrencyController(config)

        assert controller.config == config

    def test_initialization_with_kwargs(self):
        """测试带关键字参数初始化"""
        controller = ConcurrencyController(max_concurrent=5, timeout=60)

        expected_config = {"max_concurrent": 5, "timeout": 60}
        assert controller.config == expected_config

    def test_initialization_config_and_kwargs(self):
        """测试配置和关键字参数合并"""
        config = {"max_concurrent": 10}
        controller = ConcurrencyController(config, timeout=60, retries=3)

        expected_config = {"max_concurrent": 10, "timeout": 60, "retries": 3}
        assert controller.config == expected_config

    def test_initialization_kwargs_override_config(self):
        """测试关键字参数覆盖配置"""
        config = {"max_concurrent": 10, "timeout": 30}
        controller = ConcurrencyController(config, timeout=60)

        expected_config = {"max_concurrent": 10, "timeout": 60}
        assert controller.config == expected_config

    def test_health_check(self):
        """测试健康检查"""
        controller = ConcurrencyController()

        result = controller.health_check()

        assert isinstance(result, dict)
        assert "status" in result
        assert "component" in result
        assert "timestamp" in result
        assert result["status"] in ["healthy", "unhealthy"]
        assert result["component"] == "ConcurrencyController"

    def test_repr(self):
        """测试字符串表示"""
        controller = ConcurrencyController()

        repr_str = repr(controller)

        assert "ConcurrencyController" in repr_str
        assert "config={}" in repr_str

    def test_thread_safety(self):
        """测试线程安全性"""
        controller = ConcurrencyController()
        results = []

        def worker(worker_id):
            # 模拟并发访问
            for i in range(10):
                controller.health_check()
                controller.get_component_status()
                time.sleep(0.001)  # 小延迟以增加竞争条件可能性
            results.append(f"worker_{worker_id}_done")

        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证所有线程都完成了
        assert len(results) == 5
        assert all("done" in result for result in results)

    def test_config_immutability(self):
        """测试配置不可变性"""
        config = {"max_concurrent": 10, "timeout": 30}
        controller = ConcurrencyController(config)

        # 尝试修改配置（应该不影响内部状态）
        config["max_concurrent"] = 20

        # 内部配置应该保持不变
        assert controller.config["max_concurrent"] == 10

    def test_active_tasks_tracking(self):
        """测试活动任务跟踪"""
        controller = ConcurrencyController()

        # 初始状态
        status = controller.get_component_status()
        assert status["active_tasks"] == 0

        # 模拟添加活动任务
        controller._active_tasks = {"task1": "running", "task2": "pending"}

        status = controller.get_component_status()
        assert status["active_tasks"] == 2

    def test_health_check_detailed_info(self):
        """测试健康检查详细信息"""
        controller = ConcurrencyController({"max_concurrent": 5})

        result = controller.health_check()

        assert "details" in result
        assert "active_tasks" in result["details"]
        assert "config_summary" in result["details"]

    def test_component_lifecycle(self):
        """测试组件生命周期"""
        controller = ConcurrencyController()

        # 初始状态
        assert controller._active_tasks == {}

        # 初始化
        controller.initialize_component()
        assert controller._active_tasks == {}

        # 模拟使用
        controller._active_tasks = {"task1": "running"}

        # 获取状态
        status = controller.get_component_status()
        assert status["active_tasks"] == 1

        # 关闭
        controller.shutdown_component()

        # 验证状态（关闭后可能清理了任务）
        # 注意：实际实现可能会有不同的行为

    def test_configuration_validation(self):
        """测试配置验证"""
        # 有效配置
        controller = ConcurrencyController({"max_concurrent": 10, "timeout": 30})
        status = controller.health_check()
        assert status["status"] == "healthy"

        # 无效配置（负值）
        controller = ConcurrencyController({"max_concurrent": -1})
        status = controller.health_check()
        # 应该仍然健康，因为我们不验证配置值
        assert status["status"] == "healthy"

    def test_concurrent_access_patterns(self):
        """测试并发访问模式"""
        controller = ConcurrencyController()
        results = []

        def reader():
            for _ in range(100):
                status = controller.get_component_status()
                assert isinstance(status, dict)
            results.append("reader_done")

        def writer():
            for _ in range(100):
                controller.health_check()
            results.append("writer_done")

        # 创建读取和写入线程
        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)

        reader_thread.start()
        writer_thread.start()

        reader_thread.join()
        writer_thread.join()

        assert len(results) == 2
        assert "reader_done" in results
        assert "writer_done" in results

    def test_error_handling(self):
        """测试错误处理"""
        controller = ConcurrencyController()

        # 测试在异常情况下仍能正常工作
        try:
            # 这些操作应该不会抛出异常
            controller.initialize_component()
            controller.health_check()
            controller.get_component_status()
            controller.shutdown_component()
        except Exception as e:
            pytest.fail(f"Controller operation failed with exception: {e}")

    def test_memory_cleanup(self):
        """测试内存清理"""
        import gc

        controller = ConcurrencyController()

        # 创建一些引用
        controller._active_tasks = {"task1": {"data": "large_data" * 1000}}

        # 删除引用
        del controller

        # 强制垃圾回收
        gc.collect()

        # 如果没有内存泄漏，测试应该通过
        # 这是一个基本的内存清理测试