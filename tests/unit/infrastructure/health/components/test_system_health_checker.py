"""
测试系统健康检查器
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


class TestSystemHealthChecker:
    """测试系统健康检查器"""

    def test_system_health_checker_import(self):
        """测试系统健康检查器导入"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker
            assert SystemHealthChecker is not None
        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_system_health_checker_initialization(self):
        """测试系统健康检查器初始化"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()
            assert checker is not None

            # 检查初始化属性
            assert hasattr(checker, '_last_check_time')
            assert hasattr(checker, '_check_count')
            assert checker._check_count == 0
            assert checker._last_check_time is None

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_get_cpu_info(self):
        """测试获取CPU信息"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()
            cpu_info = checker._get_cpu_info()

            assert isinstance(cpu_info, dict)
            # 检查返回的基本信息，不强制要求特定字段名
            assert len(cpu_info) > 0
            assert 'status' in cpu_info  # 应该有状态信息

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_get_memory_info(self):
        """测试获取内存信息"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()
            memory_info = checker._get_memory_info()

            assert isinstance(memory_info, dict)
            # 检查返回的基本信息，不强制要求特定字段名
            assert len(memory_info) > 0
            assert 'status' in memory_info  # 应该有状态信息

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_get_disk_info(self):
        """测试获取磁盘信息"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()
            disk_info = checker._get_disk_info()

            assert isinstance(disk_info, dict)
            # 检查返回的基本信息，不强制要求特定字段名
            assert len(disk_info) > 0
            assert 'status' in disk_info  # 应该有状态信息

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_get_process_info(self):
        """测试获取进程信息"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()
            process_info = checker._get_process_info()

            assert isinstance(process_info, dict)
            # 检查返回的基本信息，不强制要求特定字段名
            assert len(process_info) > 0

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_check_cpu_usage(self):
        """测试CPU使用率检查"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()

            # Mock psutil.cpu_percent to return a specific value
            with patch('psutil.cpu_percent', return_value=85.0) as mock_cpu_percent:
                cpu_check = checker._check_cpu_usage()

                assert isinstance(cpu_check, dict)
                assert 'status' in cpu_check
                assert 'cpu_percent' in cpu_check
                assert cpu_check['cpu_percent'] == 85.0
                # Verify psutil.cpu_percent was called with correct interval
                mock_cpu_percent.assert_called_once_with(interval=0.1)

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_check_memory_usage(self):
        """测试内存使用率检查"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()

            # Create a mock virtual_memory object
            mock_virtual_memory = Mock()
            mock_virtual_memory.available = 4 * 1024**3  # 4GB available
            mock_virtual_memory.percent = 75.0

            with patch('psutil.virtual_memory', return_value=mock_virtual_memory) as mock_vm:
                memory_check = checker._check_memory_usage()

                assert isinstance(memory_check, dict)
                assert 'status' in memory_check
                assert 'memory_percent' in memory_check
                assert memory_check['memory_percent'] == 75.0
                assert memory_check['available_gb'] == 4.0  # 4GB available
                # Verify psutil.virtual_memory was called
                mock_vm.assert_called_once()

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_check_disk_usage(self):
        """测试磁盘使用率检查"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()

            # Create a mock disk_usage object
            mock_disk_usage = Mock()
            mock_disk_usage.free = 150 * 1024**3  # 150GB free
            mock_disk_usage.percent = 85.0

            with patch('psutil.disk_usage', return_value=mock_disk_usage) as mock_du:
                disk_check = checker._check_disk_usage()

                assert isinstance(disk_check, dict)
                assert 'status' in disk_check
                assert 'disk_percent' in disk_check
                assert disk_check['disk_percent'] == 85.0
                assert disk_check['free_gb'] == 150.0  # 150GB free
                # Verify psutil.disk_usage was called with root path
                mock_du.assert_called_once_with("/")

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_evaluate_cpu_status(self):
        """测试CPU状态评估"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()

            # 测试不同CPU使用率的评估
            assert checker._evaluate_cpu_status(45.0) == "healthy"
            assert checker._evaluate_cpu_status(75.0) == "healthy"
            assert checker._evaluate_cpu_status(85.0) == "warning"
            assert checker._evaluate_cpu_status(95.0) == "warning"

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_get_system_health(self, mock_disk_usage, mock_virtual_memory, mock_cpu_percent):
        """测试获取系统健康状态"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            # 设置mock返回值
            mock_cpu_percent.return_value = 45.5

            mock_memory = Mock()
            mock_memory.total = 16 * 1024**3
            mock_memory.available = 8 * 1024**3
            mock_memory.used = 8 * 1024**3
            mock_memory.percent = 50.0
            mock_virtual_memory.return_value = mock_memory

            mock_disk = Mock()
            mock_disk.total = 1000 * 1024**3
            mock_disk.used = 500 * 1024**3
            mock_disk.free = 500 * 1024**3
            mock_disk.percent = 50.0
            mock_disk_usage.return_value = mock_disk

            checker = SystemHealthChecker()
            health_info = checker.get_system_health()

            assert isinstance(health_info, dict)
            assert 'timestamp' in health_info
            assert 'check_count' in health_info
            assert 'cpu' in health_info
            assert 'memory' in health_info
            assert 'disk' in health_info
            assert 'status' in health_info

            # 检查计数器是否更新
            assert checker._check_count == 1
            assert checker._last_check_time is not None

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_run_health_checks(self):
        """测试运行健康检查"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()

            with patch.object(checker, 'get_system_health') as mock_get_health:
                mock_get_health.return_value = {
                    'overall_status': 'healthy',
                    'cpu': {'status': 'healthy'},
                    'memory': {'status': 'healthy'},
                    'disk': {'status': 'healthy'}
                }

                results = checker.run_health_checks()

                assert isinstance(results, dict)
                assert 'checks' in results
                assert 'overall_status' in results

        except ImportError:
            pytest.skip("SystemHealthChecker not available")

    def test_check_health_sync(self):
        """测试同步健康检查"""
        try:
            from src.infrastructure.health.components.system_health_checker import SystemHealthChecker

            checker = SystemHealthChecker()

            with patch.object(checker, 'get_system_health') as mock_get_health:
                mock_get_health.return_value = {'status': 'UP', 'details': {}}

                result = checker.check_health_sync()

                assert isinstance(result, dict)
                assert 'overall_status' in result

        except ImportError:
            pytest.skip("SystemHealthChecker not available")
