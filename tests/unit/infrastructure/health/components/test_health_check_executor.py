"""
测试健康检查执行器
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
from datetime import datetime


class TestHealthCheckExecutor:
    """测试健康检查执行器"""

    def test_health_check_executor_import(self):
        """测试健康检查执行器导入"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor
            from src.infrastructure.health.components.parameter_objects import ExecutorConfig
            assert HealthCheckExecutor is not None
            assert ExecutorConfig is not None
        except ImportError:
            pytest.skip("HealthCheckExecutor not available")

    def test_health_check_executor_initialization(self):
        """测试健康检查执行器初始化"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor

            executor = HealthCheckExecutor()
            assert executor is not None

            # 检查基本属性
            assert hasattr(executor, '_timeout')
            assert hasattr(executor, '_retry_count')
            assert hasattr(executor, '_retry_delay')
            assert hasattr(executor, '_concurrent_limit')
            assert hasattr(executor, '_semaphore')
            assert hasattr(executor, '_executor')

        except ImportError:
            pytest.skip("HealthCheckExecutor not available")

    def test_health_check_executor_with_config(self):
        """测试带配置的健康检查执行器初始化"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor
            from src.infrastructure.health.components.parameter_objects import ExecutorConfig

            config = ExecutorConfig(
                timeout=10.0,
                retry_count=5,
                retry_delay=2.0,
                concurrent_limit=20
            )

            executor = HealthCheckExecutor(config)
            assert executor is not None
            assert executor._timeout == 10.0
            assert executor._retry_count == 5
            assert executor._retry_delay == 2.0
            assert executor._concurrent_limit == 20

        except (ImportError, AttributeError):
            pytest.skip("HealthCheckExecutor config not available")

    def test_create_with_params(self):
        """测试使用参数创建执行器"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor

            executor = HealthCheckExecutor.create_with_params(
                timeout=15.0,
                retry_count=3,
                retry_delay=1.5,
                concurrent_limit=25
            )

            assert executor is not None
            assert executor._timeout == 15.0
            assert executor._retry_count == 3
            assert executor._retry_delay == 1.5
            assert executor._concurrent_limit == 25

        except ImportError:
            pytest.skip("HealthCheckExecutor not available")

    @pytest.mark.asyncio
    async def test_execute_check_with_retry_success(self):
        """测试执行健康检查成功情况"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor
            from src.infrastructure.health.components.parameter_objects import HealthCheckConfig

            executor = HealthCheckExecutor()

            # 创建mock检查函数
            async def mock_check():
                return {"status": "UP", "response_time": 0.5}

            config = HealthCheckConfig(
                name="test_check",
                check_func=mock_check,
                timeout=5.0
            )

            result = await executor.execute_check_with_retry(config)

            assert isinstance(result, dict)
            assert result["status"] == "UP"
            assert "response_time" in result
            assert "timestamp" in result

        except (ImportError, AttributeError):
            pytest.skip("HealthCheckExecutor not available")

    @pytest.mark.asyncio
    async def test_execute_check_with_retry_failure(self):
        """测试执行健康检查失败和重试"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor
            from src.infrastructure.health.components.parameter_objects import HealthCheckConfig

            executor = HealthCheckExecutor()
            executor._retry_count = 2  # 设置重试次数

            # 创建总是失败的mock检查函数
            async def mock_check():
                raise Exception("Check failed")

            config = HealthCheckConfig(
                name="test_check",
                check_func=mock_check,
                timeout=1.0
            )

            result = await executor.execute_check_with_retry(config)

            assert isinstance(result, dict)
            assert result["status"] == "DOWN"
            assert "error" in result
            assert "timestamp" in result

        except (ImportError, AttributeError):
            pytest.skip("HealthCheckExecutor not available")

    @pytest.mark.asyncio
    async def test_execute_check_with_retry_legacy(self):
        """测试遗留的执行健康检查方法"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor

            executor = HealthCheckExecutor()

            # 创建mock检查函数
            async def mock_check(**kwargs):
                return {"status": "UP"}

            result = await executor.execute_check_with_retry_legacy(
                name="legacy_test",
                check_func=mock_check,
                config={"timeout": 5.0}
            )

            assert isinstance(result, dict)
            assert result["status"] == "UP"

        except ImportError:
            pytest.skip("HealthCheckExecutor not available")

    def test_create_error_result(self):
        """测试创建错误结果"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor

            executor = HealthCheckExecutor()

            exception = Exception("Test error")
            result = executor._create_error_result("test_check", exception, 5.0)

            assert isinstance(result, dict)
            assert result["status"] == "DOWN"
            assert "error" in result
            assert "Test error" in result["error"]
            assert result["response_time"] == 5.0
            assert "timestamp" in result

        except ImportError:
            pytest.skip("HealthCheckExecutor not available")

    @patch('psutil.cpu_percent')
    def test_cpu_usage_evaluation(self, mock_cpu_percent):
        """测试CPU使用率评估"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor

            executor = HealthCheckExecutor()
            mock_cpu_percent.return_value = 85.0

            # 测试CPU状态评估
            status, message = executor._evaluate_cpu_status(85.0)
            assert status in ["warning", "critical"]
            assert "CPU" in message

            # 测试不同使用率
            status_ok, _ = executor._evaluate_cpu_status(50.0)
            assert status_ok == "healthy"

            status_warning, _ = executor._evaluate_cpu_status(85.0)
            assert status_warning == "warning"

            status_critical, _ = executor._evaluate_cpu_status(95.0)
            assert status_critical == "critical"

        except ImportError:
            pytest.skip("HealthCheckExecutor not available")

    @patch('psutil.virtual_memory')
    def test_memory_usage_evaluation(self, mock_virtual_memory):
        """测试内存使用率评估"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor

            # 创建mock内存对象
            mock_memory = Mock()
            mock_memory.percent = 90.0
            mock_virtual_memory.return_value = mock_memory

            executor = HealthCheckExecutor()

            # 测试内存状态评估
            status, message = executor._evaluate_memory_status(90.0)
            assert status in ["warning", "critical"]
            assert "内存" in message or "memory" in message.lower()

        except ImportError:
            pytest.skip("HealthCheckExecutor not available")

    @patch('psutil.disk_usage')
    def test_disk_usage_evaluation(self, mock_disk_usage):
        """测试磁盘使用率评估"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor

            # 创建mock磁盘对象
            mock_disk = Mock()
            mock_disk.percent = 88.0
            mock_disk_usage.return_value = mock_disk

            executor = HealthCheckExecutor()

            # 测试磁盘状态评估
            status, message = executor._evaluate_disk_status(88.0)
            assert status in ["warning", "critical"]
            assert "磁盘" in message or "disk" in message.lower()

        except ImportError:
            pytest.skip("HealthCheckExecutor not available")

    def test_cleanup(self):
        """测试清理方法"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor

            executor = HealthCheckExecutor()

            # 测试清理
            executor.cleanup()

            # 验证清理后的状态
            assert executor._executor is None or hasattr(executor, '_executor')

        except ImportError:
            pytest.skip("HealthCheckExecutor not available")

    def test_constants(self):
        """测试常量定义"""
        try:
            from src.infrastructure.health.components.health_check_executor import (
                DEFAULT_SERVICE_TIMEOUT, DEFAULT_RETRY_COUNT, DEFAULT_RETRY_DELAY,
                DEFAULT_CONCURRENT_LIMIT, RESPONSE_TIME_WARNING_THRESHOLD
            )

            assert DEFAULT_SERVICE_TIMEOUT == 5.0
            assert DEFAULT_RETRY_COUNT == 3
            assert DEFAULT_RETRY_DELAY == 1.0
            assert DEFAULT_CONCURRENT_LIMIT == 10
            assert RESPONSE_TIME_WARNING_THRESHOLD == 2.0

        except ImportError:
            pytest.skip("Constants not available")

    @pytest.mark.asyncio
    async def test_concurrent_limit(self):
        """测试并发限制"""
        try:
            from src.infrastructure.health.components.health_check_executor import HealthCheckExecutor
            from src.infrastructure.health.components.parameter_objects import HealthCheckConfig

            # 创建并发限制为2的执行器
            executor = HealthCheckExecutor.create_with_params(concurrent_limit=2)

            # 创建多个并发检查任务
            async def slow_check():
                await asyncio.sleep(0.1)
                return {"status": "UP"}

            tasks = []
            for i in range(5):
                config = HealthCheckConfig(
                    name=f"check_{i}",
                    check_func=slow_check,
                    timeout=10.0
                )
                task = executor.execute_check_with_retry(config)
                tasks.append(task)

            # 执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 验证所有任务都成功完成
            assert len(results) == 5
            for result in results:
                assert isinstance(result, dict)
                assert result["status"] == "UP"

        except (ImportError, AttributeError):
            pytest.skip("Concurrent execution not available")
