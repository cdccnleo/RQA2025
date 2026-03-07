"""
测试GPU管理器

验证GPUManager类的核心功能，包括GPU检测、分配、监控等
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import sys
import time
import threading
import subprocess
import os
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.infrastructure.resource.core.gpu_manager import GPUManager


class TestGPUManager:
    """测试GPUManager类"""

    def test_gpu_manager_initialization(self):
        """测试GPU管理器初始化"""
        manager = GPUManager()

        assert len(manager.gpus) == 0
        assert len(manager.allocated_gpus) == 0
        assert manager.monitoring_active is False
        assert manager.monitor_thread is None

    def test_detect_gpus(self):
        """测试GPU检测"""
        manager = GPUManager()

        # Mock GPUtil at module level
        mock_gpu1 = MagicMock()
        mock_gpu1.id = 0
        mock_gpu1.name = 'NVIDIA GeForce RTX 3080'
        mock_gpu1.memoryTotal = 10240
        mock_gpu1.memoryFree = 8192
        mock_gpu1.memoryUsed = 2048
        mock_gpu1.temperature = 65
        mock_gpu1.uuid = 'GPU-12345678'
        
        mock_gpu2 = MagicMock()
        mock_gpu2.id = 1
        mock_gpu2.name = 'NVIDIA GeForce RTX 3070'
        mock_gpu2.memoryTotal = 8192
        mock_gpu2.memoryFree = 6144
        mock_gpu2.memoryUsed = 2048
        mock_gpu2.temperature = 58
        mock_gpu2.uuid = 'GPU-87654321'

        with patch('src.infrastructure.resource.core.gpu_manager.GPUtil') as mock_gputil:
            mock_gputil.getGPUs.return_value = [mock_gpu1, mock_gpu2]

            detected = manager.detect_gpus()

            assert len(detected) == 2
            assert detected[0]['id'] == 0
            assert detected[0]['name'] == 'NVIDIA GeForce RTX 3080'
            assert detected[0]['memory_total'] == 10240
            assert detected[1]['id'] == 1
            assert detected[1]['name'] == 'NVIDIA GeForce RTX 3070'

    def test_detect_gpus_no_gpu(self):
        """测试无GPU情况下的检测"""
        manager = GPUManager()

        with patch('src.infrastructure.resource.core.gpu_manager.GPUtil') as mock_gputil:
            mock_gputil.getGPUs.return_value = []

            detected = manager.detect_gpus()

            assert len(detected) == 0

    def test_detect_gpus_error(self):
        """测试GPU检测错误情况"""
        manager = GPUManager()

        with patch('src.infrastructure.resource.core.gpu_manager.GPUtil') as mock_gputil:
            mock_gputil.getGPUs.side_effect = Exception("GPU detection failed")

            detected = manager.detect_gpus()

            assert len(detected) == 0

    def test_allocate_gpu(self):
        """测试分配GPU"""
        manager = GPUManager()

        # 手动添加GPU信息
        manager.gpus = {
            0: {
                'id': 0,
                'name': 'NVIDIA GeForce RTX 3080',
                'memory_total': 10240,
                'memory_free': 8192,
                'temperature': 65
            }
        }

        result = manager.allocate_gpu(0, memory_required=2048)

        assert result is True
        assert 0 in manager.allocated_gpus
        assert manager.allocated_gpus[0]['memory_required'] == 2048

    def test_allocate_gpu_insufficient_memory(self):
        """测试分配GPU内存不足"""
        manager = GPUManager()

        manager.gpus = {
            0: {
                'id': 0,
                'memory_total': 4096,  # 4GB
                'memory_free': 1024,  # 1GB
                'temperature': 65
            }
        }

        result = manager.allocate_gpu(0, memory_required=2048)  # 需要2GB

        assert result is False
        assert 0 not in manager.allocated_gpus

    def test_allocate_gpu_not_found(self):
        """测试分配不存在的GPU"""
        manager = GPUManager()

        result = manager.allocate_gpu(999, memory_required=1024)

        assert result is False

    def test_release_gpu(self):
        """测试释放GPU"""
        manager = GPUManager()

        # 先分配GPU
        manager.gpus = {
            0: {
                'id': 0,
                'name': 'NVIDIA GeForce RTX 3080',
                'memory_total': 10240,
                'memory_free': 8192,
                'temperature': 65
            }
        }
        manager.allocate_gpu(0, memory_required=2048)

        # 释放GPU
        result = manager.release_gpu(0)

        assert result is True
        assert 0 not in manager.allocated_gpus

    def test_release_gpu_not_allocated(self):
        """测试释放未分配的GPU"""
        manager = GPUManager()

        result = manager.release_gpu(0)

        assert result is False

    def test_get_gpu_status(self):
        """测试获取GPU状态"""
        manager = GPUManager()

        manager.gpus = {
            0: {
                'id': 0,
                'name': 'NVIDIA GeForce RTX 3080',
                'memory_total': 10240,
                'memory_free': 8192,
                'temperature': 65
            },
            1: {
                'id': 1,
                'name': 'NVIDIA GeForce RTX 3070',
                'memory_total': 8192,
                'memory_free': 6144,
                'temperature': 58
            }
        }

        # 分配一个GPU
        manager.allocate_gpu(0, memory_required=2048)

        status = manager.get_gpu_status()

        assert len(status) == 2
        assert status[0]['allocated'] is True
        assert status[0]['memory_required'] == 2048
        assert status[1]['allocated'] is False

    def test_get_available_gpus(self):
        """测试获取可用GPU"""
        manager = GPUManager()

        manager.gpus = {
            0: {'id': 0, 'memory_free': 8192, 'temperature': 65},
            1: {'id': 1, 'memory_free': 6144, 'temperature': 58},
            2: {'id': 2, 'memory_free': 1024, 'temperature': 75}  # 温度过高
        }

        # 分配GPU 0
        manager.allocate_gpu(0, memory_required=2048)

        available = manager.get_available_gpus(memory_required=2048, max_temperature=70)

        assert len(available) == 1
        assert available[0] == 1  # GPU 1满足条件，GPU 2温度过高

    def test_start_monitoring(self):
        """测试启动GPU监控"""
        manager = GPUManager()

        with patch.object(manager, '_monitor_loop', side_effect=KeyboardInterrupt):
            manager.start_monitoring()

        assert manager.monitoring_active is True
        assert manager.monitor_thread is not None

        # 停止监控
        manager.stop_monitoring()
        assert manager.monitoring_active is False

    def test_stop_monitoring_without_start(self):
        """测试在未启动时停止监控"""
        manager = GPUManager()

        manager.stop_monitoring()
        assert manager.monitoring_active is False

    def test_get_gpu_utilization_report(self):
        """测试获取GPU利用率报告"""
        manager = GPUManager()

        manager.gpus = {
            0: {
                'id': 0,
                'name': 'NVIDIA GeForce RTX 3080',
                'memory_total': 10240,
                'memory_used': 2048,
                'temperature': 65,
                'utilization': 75.5
            }
        }

        report = manager.get_gpu_utilization_report()

        assert 'summary' in report
        assert 'details' in report
        assert 'recommendations' in report
        assert len(report['details']) == 1
        assert report['details'][0]['id'] == 0

    def test_monitor_gpu_health(self):
        """测试监控GPU健康状态"""
        manager = GPUManager()

        manager.gpus = {
            0: {'id': 0, 'temperature': 85, 'memory_free': 1024},  # 温度过高
            1: {'id': 1, 'temperature': 65, 'memory_free': 8192}   # 正常
        }

        issues = manager.monitor_gpu_health()

        assert len(issues) == 1
        assert 'GPU 0' in issues[0]
        assert '温度过高' in issues[0]

    def test_optimize_gpu_usage(self):
        """测试优化GPU使用"""
        manager = GPUManager()

        manager.gpus = {
            0: {'id': 0, 'utilization': 95.0, 'temperature': 80},  # 高负载
            1: {'id': 1, 'utilization': 30.0, 'temperature': 60}   # 低负载
        }

        # 分配GPU 0
        manager.allocate_gpu(0, memory_required=1024)

        recommendations = manager.optimize_gpu_usage()

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('负载均衡' in rec or '重新分配' in rec for rec in recommendations)

    def test_get_gpu_memory_info(self):
        """测试获取GPU内存信息"""
        manager = GPUManager()

        manager.gpus = {
            0: {
                'id': 0,
                'memory_total': 10240,
                'memory_used': 2048,
                'memory_free': 8192
            }
        }

        memory_info = manager.get_gpu_memory_info(0)

        assert memory_info['total'] == 10240
        assert memory_info['used'] == 2048
        assert memory_info['free'] == 8192
        assert memory_info['usage_percent'] == 20.0

    def test_get_gpu_memory_info_not_found(self):
        """测试获取不存在GPU的内存信息"""
        manager = GPUManager()

        memory_info = manager.get_gpu_memory_info(999)

        assert memory_info is None

    def test_get_gpu_temperature(self):
        """测试获取GPU温度"""
        manager = GPUManager()

        manager.gpus = {
            0: {'id': 0, 'temperature': 72}
        }

        temp = manager.get_gpu_temperature(0)

        assert temp == 72

    def test_get_gpu_temperature_not_found(self):
        """测试获取不存在GPU的温度"""
        manager = GPUManager()

        temp = manager.get_gpu_temperature(999)

        assert temp is None

    def test_concurrent_gpu_operations(self):
        """测试并发GPU操作"""
        manager = GPUManager()

        # 添加GPU
        manager.gpus = {
            0: {'id': 0, 'memory_free': 8192, 'temperature': 65},
            1: {'id': 1, 'memory_free': 6144, 'temperature': 58}
        }

        results = []
        errors = []

        def concurrent_operation(operation_id):
            try:
                gpu_id = operation_id % 2
                if operation_id < 5:
                    result = manager.allocate_gpu(gpu_id, memory_required=1024)
                    results.append(('allocate', gpu_id, result))
                else:
                    result = manager.release_gpu(gpu_id)
                    results.append(('release', gpu_id, result))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_operation, args=(i,))
            threads.append(thread)

        for thread in threads:
            thread.start()

        # 使用超时机制等待线程完成，避免无限等待
        for thread in threads:
            thread.join(timeout=2.0)  # 最多等待2秒
            if thread.is_alive():
                errors.append(TimeoutError(f"Thread {thread.ident} did not complete within timeout"))

        # 验证没有出现异常
        assert len(errors) == 0
        assert len(results) == 10

    def test_gpu_manager_cleanup(self):
        """测试GPU管理器清理"""
        manager = GPUManager()

        # 添加GPU并分配
        manager.gpus = {
            0: {'id': 0, 'memory_free': 8192, 'temperature': 65}
        }
        manager.allocate_gpu(0, memory_required=1024)

        # 清理
        manager.cleanup()

        assert len(manager.gpus) == 0
        assert len(manager.allocated_gpus) == 0
        assert manager.monitoring_active is False

    @patch('subprocess.run')
    def test_detect_gpu_with_nvidia_smi_failure(self, mock_run):
        """测试GPU检测 - nvidia-smi失败的情况"""
        manager = GPUManager()
        
        # 模拟nvidia-smi调用失败
        mock_run.side_effect = subprocess.TimeoutExpired('nvidia-smi', 5)
        
        # 重新检测GPU
        result = manager._detect_gpu()
        
        # 由于nvidia-smi失败，应该返回False
        assert result is False

    @patch('subprocess.run')
    @patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0,1'})
    def test_detect_gpu_with_cuda_environment(self, mock_run):
        """测试GPU检测 - 通过CUDA环境变量检测"""
        manager = GPUManager()
        
        # 模拟nvidia-smi调用失败
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        
        # 重新检测GPU
        result = manager._detect_gpu()
        
        # 由于有CUDA环境变量，应该返回True
        assert result is True

    @patch('subprocess.run')
    @patch.dict('os.environ', {'HIP_VISIBLE_DEVICES': '0'})
    def test_detect_gpu_with_rocm_environment(self, mock_run):
        """测试GPU检测 - 通过ROCm环境变量检测"""
        manager = GPUManager()
        
        # 模拟nvidia-smi调用失败
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        
        # 重新检测GPU
        result = manager._detect_gpu()
        
        # 由于有ROCm环境变量，应该返回True
        assert result is True

    @patch('subprocess.run')
    def test_get_gpu_info_success(self, mock_run):
        """测试获取GPU信息成功"""
        manager = GPUManager()
        
        # 模拟nvidia-smi成功返回
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA RTX 3080,10240,2048,8192,45.5,65\n"
        mock_run.return_value = mock_result
        
        # 设置has_gpu为True以触发_get_gpu_info
        manager.has_gpu = True
        
        result = manager._get_gpu_info()
        
        assert len(result) == 1
        assert result[0]['name'] == 'NVIDIA RTX 3080'
        assert result[0]['memory_total_mb'] == 10240
        assert result[0]['temperature_celsius'] == 65

    @patch('subprocess.run')
    def test_get_gpu_info_failure(self, mock_run):
        """测试获取GPU信息失败"""
        manager = GPUManager()
        
        # 模拟nvidia-smi调用失败
        mock_run.side_effect = subprocess.TimeoutExpired('nvidia-smi', 5)
        
        # 设置has_gpu为True
        manager.has_gpu = True
        
        result = manager._get_gpu_info()
        
        # 失败时应该返回模拟数据
        assert len(result) == 1
        assert result[0]['name'] == 'NVIDIA RTX 3090'

    def test_get_gpu_usage_no_gpu(self):
        """测试获取GPU使用情况 - 无GPU"""
        manager = GPUManager()
        manager.has_gpu = False
        
        result = manager.get_gpu_usage()
        
        assert result['available'] is False
        assert len(result['gpus']) == 0
        assert result['summary']['total_gpus'] == 0


    def test_get_temperature_status(self):
        """测试温度状态判断"""
        manager = GPUManager()
        
        # 测试不同温度范围
        assert manager._get_temperature_status(30) == 'normal'
        assert manager._get_temperature_status(55) == 'warm'
        assert manager._get_temperature_status(75) == 'hot'
        assert manager._get_temperature_status(90) == 'critical'

    def test_get_gpu_health_status_no_gpu(self):
        """测试获取GPU健康状态 - 无GPU"""
        manager = GPUManager()
        
        with patch.object(manager, 'get_gpu_usage') as mock_usage:
            with patch.object(manager, 'get_gpu_temperature') as mock_temp:
                mock_usage.return_value = {'available': False}
                mock_temp.return_value = {'available': False}
                
                result = manager.get_gpu_health_status()
                
                assert result['overall_health'] == 'no_gpu'
                assert 'No GPU detected' in result['issues']

    def test_get_gpu_health_status_with_issues(self):
        """测试获取GPU健康状态 - 有问题"""
        manager = GPUManager()
        
        with patch.object(manager, 'get_gpu_usage') as mock_usage:
            with patch.object(manager, 'get_gpu_temperature') as mock_temp:
                mock_usage.return_value = {
                    'available': True,
                    'gpus': [{'id': 0, 'utilization_percent': 98}]
                }
                mock_temp.return_value = {
                    'available': True,
                    'temperature_info': [{'gpu_id': 0, 'temperature_celsius': 90, 'status': 'critical'}]
                }
                
                result = manager.get_gpu_health_status()
                
                assert len(result['issues']) > 0
                assert len(result['recommendations']) > 0

    def test_allocate_gpu_memory_no_gpu(self):
        """测试分配GPU内存 - 无GPU"""
        manager = GPUManager()
        manager.has_gpu = False
        
        result = manager.allocate_gpu_memory(1024)
        
        assert result is False

    def test_allocate_gpu_memory_with_gpu(self):
        """测试分配GPU内存 - 有GPU"""
        manager = GPUManager()
        manager.has_gpu = True
        
        result = manager.allocate_gpu_memory(1024)
        
        # 目前返回模拟结果
        assert result is True

    def test_free_gpu_memory_no_gpu(self):
        """测试释放GPU内存 - 无GPU"""
        manager = GPUManager()
        manager.has_gpu = False
        
        result = manager.free_gpu_memory()
        
        assert result is False

    def test_free_gpu_memory_with_gpu(self):
        """测试释放GPU内存 - 有GPU"""
        manager = GPUManager()
        manager.has_gpu = True
        
        result = manager.free_gpu_memory()
        
        # 目前返回模拟结果
        assert result is True

    @patch('threading.Thread')
    def test_monitor_loop_exception_handling(self, mock_thread):
        """测试监控循环异常处理"""
        manager = GPUManager()
        
        # 创建真实的线程实例用于测试
        real_thread = threading.Thread(target=manager._monitor_loop, daemon=True)
        
        # 模拟监控循环中的异常
        original_monitor_loop = manager._monitor_loop
        
        def failing_monitor_loop():
            manager.monitoring_active = True
            time.sleep(0.1)  # 短暂睡眠
            raise Exception("监控异常")
        
        manager._monitor_loop = failing_monitor_loop
        
        # 启动监控
        manager.monitoring_active = True
        thread = threading.Thread(target=manager._monitor_loop, daemon=True)
        thread.start()
        
        # 等待线程完成
        thread.join(timeout=1.0)
        
        # 恢复原始方法
        manager._monitor_loop = original_monitor_loop

    def test_get_gpu_memory_info_with_gpu(self):
        """测试获取指定GPU内存信息 - 有GPU"""
        manager = GPUManager()
        
        manager.gpus = {
            0: {
                'id': 0,
                'memory_total': 10240,
                'memory_used': 2048,
                'memory_free': 8192
            }
        }
        
        result = manager.get_gpu_memory_info(0)
        
        assert result is not None
        assert result['total'] == 10240
        assert result['used'] == 2048
        assert result['free'] == 8192
        assert result['usage_percent'] == 20.0

    def test_get_gpu_memory_info_zero_total(self):
        """测试获取GPU内存信息 - 总内存为0"""
        manager = GPUManager()
        
        manager.gpus = {
            0: {
                'id': 0,
                'memory_total': 0,
                'memory_used': 0,
                'memory_free': 0
            }
        }
        
        result = manager.get_gpu_memory_info(0)
        
        assert result is not None
        assert result['usage_percent'] == 0.0
