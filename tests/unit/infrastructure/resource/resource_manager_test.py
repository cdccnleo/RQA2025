import unittest
from unittest.mock import patch, MagicMock
from src.infrastructure.resource import ResourceManager, ResourceTicket

class TestResourceManager(unittest.TestCase):
    """资源管理模块单元测试"""

    def setUp(self):
        self.manager = ResourceManager()
        # 模拟GPU设备
        self.mock_gpu = MagicMock()
        self.mock_gpu.memory_used.return_value = 0
        self.manager.gpu_devices = [self.mock_gpu]

    def test_resource_allocation(self):
        """测试基础资源分配"""
        # 申请CPU和内存资源
        ticket = self.manager.request_resources({
            'cpu': {'cores': 4},
            'mem': '8GB'
        })

        self.assertIsInstance(ticket, ResourceTicket)
        self.assertEqual(ticket.cpu_cores, 4)
        self.assertEqual(ticket.memory, '8GB')

        # 释放资源
        self.manager.release_resources(ticket)
        self.assertIsNone(ticket._valid)

    def test_gpu_allocation(self):
        """测试GPU资源分配"""
        # 申请GPU资源
        ticket = self.manager.request_resources({
            'gpu': {'count': 1, 'mem': '4GB'}
        })

        self.assertEqual(ticket.gpu_count, 1)
        self.assertEqual(ticket.gpu_memory, '4GB')
        self.mock_gpu.allocate.assert_called_once()

        # 释放GPU资源
        self.manager.release_resources(ticket)
        self.mock_gpu.release.assert_called_once()

    def test_memory_quota_enforcement(self):
        """测试内存配额强制执行"""
        # 设置总内存配额
        self.manager.total_memory = '16GB'

        # 申请过多内存应失败
        with self.assertRaises(ValueError):
            self.manager.request_resources({
                'mem': '32GB'
            })

        # 申请合理内存应成功
        ticket = self.manager.request_resources({
            'mem': '8GB'
        })
        self.manager.release_resources(ticket)

    @patch('psutil.virtual_memory')
    def test_system_memory_check(self, mock_vmem):
        """测试系统内存检查"""
        # 模拟系统内存不足
        mock_vmem.return_value.available = 2 * 1024**3  # 2GB可用

        with self.assertRaises(ResourceWarning):
            self.manager.request_resources({
                'mem': '4GB'
            })

    def test_load_balancing(self):
        """测试负载均衡策略"""
        # 添加第二个GPU设备
        mock_gpu2 = MagicMock()
        mock_gpu2.memory_used.return_value = 4 * 1024**3  # 已使用4GB
        self.manager.gpu_devices.append(mock_gpu2)

        # 申请GPU资源应选择负载较低的设备
        ticket = self.manager.request_resources({
            'gpu': {'count': 1, 'mem': '2GB'}
        })

        # 验证选择了第一个GPU（使用率为0）
        self.mock_gpu.allocate.assert_called_once()
        mock_gpu2.allocate.assert_not_called()

        self.manager.release_resources(ticket)

    def test_resource_leak_detection(self):
        """测试资源泄漏检测"""
        # 申请资源但不释放
        ticket = self.manager.request_resources({
            'cpu': {'cores': 2},
            'mem': '4GB'
        })

        # 验证资源泄漏检测
        with self.assertWarns(ResourceWarning):
            del ticket

    def test_concurrent_requests(self):
        """测试并发资源申请"""
        from concurrent.futures import ThreadPoolExecutor

        # 模拟多个并发请求
        def request_resources():
            ticket = self.manager.request_resources({
                'cpu': {'cores': 1},
                'mem': '1GB'
            })
            self.manager.release_resources(ticket)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(request_resources) for _ in range(10)]
            for future in futures:
                future.result()  # 验证无异常

if __name__ == '__main__':
    unittest.main()
