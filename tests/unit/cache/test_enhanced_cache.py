import unittest
import time
import pickle
from src.data.cache.cache_manager import CacheManager

class TestEnhancedCache(unittest.TestCase):
    """增强缓存功能测试"""

    def setUp(self):
        self.cache = CacheManager()
        self.test_config = {
            '行情数据': {
                'strategy': 'aggressive',
                'ttl': 1,  # 1秒用于测试过期
                'compression': 'lz4',
                'storage': ['memory', 'disk']
            },
            '龙虎榜数据': {
                'strategy': 'incremental',
                'ttl': 2,
                'version_control': True
            }
        }
        self.cache.configure(self.test_config)

    def test_config_loading(self):
        """测试配置加载"""
        self.assertEqual(self.cache._get_config('行情数据').strategy, 'aggressive')
        self.assertEqual(self.cache._get_config('龙虎榜数据').version_control, True)
        self.assertEqual(self.cache._get_config('未知类型').strategy, 'default')

    def test_compression(self):
        """测试压缩存储"""
        test_data = {'stock': '600000', 'price': 42.0}
        self.cache.set('test_comp', test_data, '行情数据')

        # 验证内存中的是压缩数据
        raw_data = self.cache.memory_cache.get('test_comp')
        self.assertNotEqual(raw_data, test_data)

        # 验证取回的是原始数据
        retrieved = self.cache.get('test_comp', '行情数据')
        self.assertEqual(retrieved, test_data)

    def test_version_control(self):
        """测试版本控制"""
        # 第一版本
        self.cache.set('test_ver', 'v1', '龙虎榜数据')
        time.sleep(0.1)

        # 第二版本
        self.cache.set('test_ver', 'v2', '龙虎榜数据')

        # 应获取最新版本
        self.assertEqual(self.cache.get('test_ver', '龙虎榜数据'), 'v2')

        # 验证版本记录
        versions = self.cache.disk_cache.get('test_ver_versions')
        self.assertEqual(len(versions), 2)

    def test_storage_strategy(self):
        """测试存储策略"""
        test_data = {'key': 'value'}

        # 测试内存+磁盘存储
        self.cache.set('test_strat', test_data, '行情数据')
        self.assertIsNotNone(self.cache.memory_cache.get('test_strat'))
        self.assertIsNotNone(self.cache.disk_cache.get('test_strat'))

        # 测试仅内存存储
        self.cache.set('test_mem', test_data, '未知类型')
        self.assertIsNotNone(self.cache.memory_cache.get('test_mem'))
        self.assertIsNone(self.cache.disk_cache.get('test_mem'))

    def test_ttl_expiration(self):
        """测试TTL过期"""
        self.cache.set('test_ttl', 'data', '行情数据')
        self.assertIsNotNone(self.cache.get('test_ttl', '行情数据'))

        time.sleep(1.1)  # 超过TTL
        self.assertIsNone(self.cache.get('test_ttl', '行情数据'))

if __name__ == '__main__':
    unittest.main()
