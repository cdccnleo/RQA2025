import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from src.infrastructure.resource.gpu_manager import GPUManager as GPUMonitor

# Fixtures
@pytest.fixture
def mock_torch():
    """模拟PyTorch模块"""
    mock = MagicMock()

    # 模拟GPU设备属性
    device_prop = MagicMock()
    device_prop.name = "Tesla V100"
    device_prop.total_memory = 16 * 1024**3  # 16GB

    # 模拟CUDA函数
    mock.cuda = MagicMock()
    mock.cuda.device_count.return_value = 1
    mock.cuda.get_device_properties.return_value = device_prop
    mock.cuda.memory_allocated.return_value = 8 * 1024**3  # 8GB已用
    mock.cuda.memory_reserved.return_value = 10 * 1024**3  # 10GB保留

    return mock

@pytest.fixture
def mock_subprocess():
    """模拟subprocess模块"""
    with patch('subprocess.run') as mock:
        yield mock

# 测试用例
class TestGPUMonitor:
    def test_gpu_count(self, mock_torch):
        """测试GPU数量检测"""
        with patch('torch', mock_torch):
            monitor = GPUMonitor()
            assert monitor.get_gpu_count() == 1

    def test_no_gpu(self):
        """测试无GPU环境"""
        mock_torch = MagicMock()
        mock_torch.cuda.device_count.return_value = 0

        with patch('torch', mock_torch):
            monitor = GPUMonitor()
            assert monitor.get_gpu_count() == 0
            assert monitor.get_gpu_stats() is None

    def test_get_gpu_stats(self, mock_torch, mock_subprocess):
        """测试获取GPU统计信息"""
        # 配置subprocess模拟返回值
        mock_subprocess.return_value.stdout = "75\n"  # 模拟利用率
        mock_subprocess.return_value.returncode = 0

        with patch('torch', mock_torch):
            monitor = GPUMonitor()
            stats = monitor.get_gpu_stats()

            assert stats is not None
            assert len(stats) == 1
            assert stats[0]['name'] == "Tesla V100"
            assert stats[0]['memory']['total'] == 16 * 1024**3
            assert stats[0]['memory']['allocated'] == 8 * 1024**3
            assert stats[0]['utilization'] == 75.0
            assert isinstance(stats[0]['timestamp'], str)

    def test_gpu_temperature(self, mock_torch, mock_subprocess):
        """测试获取GPU温度"""
        # 配置subprocess模拟返回值
        mock_subprocess.return_value.stdout = "65\n"  # 模拟温度
        mock_subprocess.return_value.returncode = 0

        with patch('torch', mock_torch):
            monitor = GPUMonitor()
            stats = monitor.get_gpu_stats()

            assert stats is not None
            assert stats[0]['temperature'] == 65.0

    def test_gpu_utilization_failure(self, mock_torch, mock_subprocess):
        """测试获取GPU利用率失败"""
        # 配置subprocess模拟失败
        mock_subprocess.side_effect = Exception("Command failed")

        with patch('torch', mock_torch):
            monitor = GPUMonitor()
            stats = monitor.get_gpu_stats()

            assert stats is not None
            assert stats[0]['utilization'] == 0.0

    def test_multi_gpu(self, mock_torch):
        """测试多GPU环境"""
        # 配置模拟2个GPU
        mock_torch.cuda.device_count.return_value = 2

        # 模拟第二个GPU的属性
        device_prop2 = MagicMock()
        device_prop2.name = "Tesla T4"
        device_prop2.total_memory = 8 * 1024**3  # 8GB

        mock_torch.cuda.get_device_properties.side_effect = [
            mock_torch.cuda.get_device_properties.return_value,
            device_prop2
        ]

        with patch('torch', mock_torch), \
             patch('subprocess.run') as mock_subprocess:
            # 配置subprocess模拟返回值
            mock_subprocess.return_value.stdout = "50\n"
            mock_subprocess.return_value.returncode = 0

            monitor = GPUMonitor()
            stats = monitor.get_gpu_stats()

            assert stats is not None
            assert len(stats) == 2
            assert stats[0]['name'] == "Tesla V100"
            assert stats[1]['name'] == "Tesla T4"
            assert stats[1]['memory']['total'] == 8 * 1024**3

    def test_no_pytorch(self):
        """测试无PyTorch环境"""
        with patch('importlib.import_module', side_effect=ImportError):
            monitor = GPUMonitor()
            assert monitor.get_gpu_count() == 0
            assert monitor.get_gpu_stats() is None
