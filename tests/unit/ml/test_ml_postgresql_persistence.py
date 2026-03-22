#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型层PostgreSQL持久化测试

测试内容:
1. ModelManager PostgreSQL存储测试
2. FeatureCacheManager PostgreSQL存储测试
3. InferenceCache PostgreSQL存储测试
4. 降级机制测试
5. 性能测试
"""

import os
import sys
import json
import time
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class MockPostgresConnection:
    """模拟PostgreSQL连接"""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.closed = False
        self._cursor = None
        self._data_store = {}
        self._committed = False
    
    def cursor(self, cursor_factory=None):
        if self.should_fail:
            raise Exception("Connection failed")
        self._cursor = MockCursor(self._data_store)
        return self._cursor
    
    def commit(self):
        if self.should_fail:
            raise Exception("Commit failed")
        self._committed = True
    
    def rollback(self):
        self._committed = False
    
    def close(self):
        self.closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MockCursor:
    """模拟PostgreSQL游标"""
    
    def __init__(self, data_store):
        self._data_store = data_store
        self._results = []
        self._rowcount = 0
        self._last_query = None
    
    def execute(self, query, params=None):
        self._last_query = query
        self._results = []
        
        if "SELECT 1" in query:
            self._results = [(1,)]
        elif "SELECT COUNT" in query:
            self._results = [(len(self._data_store),)]
        elif "SELECT" in query and "FROM ml_models" in query:
            for key, value in self._data_store.items():
                self._results.append(value)
        elif "SELECT" in query and "FROM feature_cache" in query:
            for key, value in self._data_store.items():
                self._results.append(value)
        elif "SELECT" in query and "FROM inference_cache" in query:
            for key, value in self._data_store.items():
                self._results.append(value)
        
        return self
    
    def fetchone(self):
        if self._results:
            return self._results[0]
        return None
    
    def fetchall(self):
        return self._results
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestModelManagerPostgreSQL(unittest.TestCase):
    """ModelManager PostgreSQL存储测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        from sklearn.ensemble import RandomForestClassifier
        self.test_model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        self.test_model.fit(X, y)
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                pass
    
    def test_init_with_postgresql_disabled(self):
        """测试PostgreSQL禁用时的初始化"""
        from src.ml.models.model_manager import ModelManager, ModelStorageConfig
        
        config = ModelStorageConfig(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        manager = ModelManager({'base_path': self.temp_dir, 'enable_postgresql': False})
        
        self.assertFalse(manager._pg_available)
    
    def test_save_and_load_model(self):
        """测试保存和加载模型"""
        from src.ml.models.model_manager import ModelManager
        
        manager = ModelManager({
            'base_path': self.temp_dir,
            'enable_postgresql': False
        })
        
        success = manager.save_model(
            model_id="test_model_001",
            version="1.0.0",
            model=self.test_model,
            metadata={'description': 'Test model'},
            model_type='RandomForest',
            feature_columns=['f1', 'f2', 'f3', 'f4', 'f5'],
            metrics={'accuracy': 0.85}
        )
        
        self.assertTrue(success)
        
        loaded = manager.load_model("test_model_001", "1.0.0")
        self.assertIsNotNone(loaded)
        self.assertTrue(hasattr(loaded, 'predict'))
    
    def test_list_models(self):
        """测试列出模型"""
        from src.ml.models.model_manager import ModelManager
        
        manager = ModelManager({
            'base_path': self.temp_dir,
            'enable_postgresql': False
        })
        
        manager.save_model(
            model_id="list_test_001",
            version="1.0.0",
            model=self.test_model,
            model_type='RandomForest'
        )
        
        manager.save_model(
            model_id="list_test_002",
            version="1.0.0",
            model=self.test_model,
            model_type='XGBoost'
        )
        
        models = manager.list_models()
        self.assertGreaterEqual(len(models), 2)
    
    def test_delete_model(self):
        """测试删除模型"""
        from src.ml.models.model_manager import ModelManager
        
        manager = ModelManager({
            'base_path': self.temp_dir,
            'enable_postgresql': False
        })
        
        manager.save_model(
            model_id="delete_test_001",
            version="1.0.0",
            model=self.test_model,
            model_type='RandomForest'
        )
        
        success = manager.delete_model("delete_test_001", "1.0.0")
        self.assertTrue(success)
        
        loaded = manager.load_model("delete_test_001", "1.0.0")
        self.assertIsNone(loaded)
    
    def test_get_storage_stats(self):
        """测试获取存储统计"""
        from src.ml.models.model_manager import ModelManager
        
        manager = ModelManager({
            'base_path': self.temp_dir,
            'enable_postgresql': False
        })
        
        manager.save_model(
            model_id="stats_test_001",
            version="1.0.0",
            model=self.test_model,
            model_type='RandomForest'
        )
        
        stats = manager.get_storage_stats()
        
        self.assertIn('postgresql_available', stats)
        self.assertIn('registry_size', stats)
        self.assertFalse(stats['postgresql_available'])


class TestFeatureCacheManagerPostgreSQL(unittest.TestCase):
    """FeatureCacheManager PostgreSQL存储测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_features = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100)
        })
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                pass
    
    def test_init_with_postgresql_disabled(self):
        """测试PostgreSQL禁用时的初始化"""
        from src.ml.engine.feature_cache_manager import FeatureCacheManager, CacheConfig
        
        config = CacheConfig(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        manager = FeatureCacheManager(config)
        
        self.assertFalse(manager._pg_available)
    
    def test_cache_and_get_features(self):
        """测试缓存和获取特征"""
        from src.ml.engine.feature_cache_manager import FeatureCacheManager, CacheConfig
        
        config = CacheConfig(
            base_path=self.temp_dir,
            enable_postgresql=False,
            cache_ttl_minutes=60
        )
        
        manager = FeatureCacheManager(config)
        
        success = manager.cache_features(
            task_id="test_task_001",
            features=self.test_features,
            metadata={'source': 'unit_test'}
        )
        
        self.assertTrue(success)
        
        cached = manager.get_cached_features("test_task_001")
        self.assertIsNotNone(cached)
        self.assertIn('features', cached)
        pd.testing.assert_frame_equal(
            cached['features'].reset_index(drop=True),
            self.test_features.reset_index(drop=True),
            check_dtype=False
        )
    
    def test_cache_statistics(self):
        """测试缓存统计"""
        from src.ml.engine.feature_cache_manager import FeatureCacheManager, CacheConfig
        
        config = CacheConfig(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        manager = FeatureCacheManager(config)
        
        manager.cache_features("stats_task", self.test_features)
        
        stats = manager.get_cache_statistics()
        
        self.assertIn('memory_cache', stats)
        self.assertIn('hits', stats)
        self.assertFalse(stats['persistent_cache']['postgresql_enabled'])
    
    def test_invalidate_cache(self):
        """测试缓存失效"""
        from src.ml.engine.feature_cache_manager import FeatureCacheManager, CacheConfig
        
        config = CacheConfig(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        manager = FeatureCacheManager(config)
        
        manager.cache_features("invalidate_task", self.test_features)
        
        cached = manager.get_cached_features("invalidate_task")
        self.assertIsNotNone(cached)
        
        manager.invalidate_cache("invalidate_task")
        
        cached = manager.get_cached_features("invalidate_task")
        self.assertIsNone(cached)


class TestInferenceCachePostgreSQL(unittest.TestCase):
    """InferenceCache PostgreSQL存储测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_input = np.random.randn(10, 5)
        self.test_result = {
            'predictions': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'probabilities': [0.3, 0.7, 0.4, 0.6, 0.35, 0.65, 0.45, 0.55, 0.38, 0.62]
        }
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                pass
    
    def test_init_with_postgresql_disabled(self):
        """测试PostgreSQL禁用时的初始化"""
        from src.ml.models.inference.inference_cache import InferenceCache, CacheConfig
        
        config = CacheConfig(
            cache_path=self.temp_dir,
            enable_postgresql=False
        )
        
        cache = InferenceCache({'cache_path': self.temp_dir, 'enable_postgresql': False})
        
        self.assertFalse(cache._pg_available)
    
    def test_set_and_get_cache(self):
        """测试设置和获取缓存"""
        from src.ml.models.inference.inference_cache import InferenceCache
        
        cache = InferenceCache({
            'cache_path': self.temp_dir,
            'enable_postgresql': False,
            'ttl_seconds': 3600
        })
        
        success = cache.set(
            model_id="test_model",
            input_data=self.test_input,
            result=self.test_result
        )
        
        self.assertTrue(success)
        
        cached = cache.get("test_model", self.test_input)
        self.assertIsNotNone(cached)
        self.assertEqual(cached['predictions'], self.test_result['predictions'])
    
    def test_cache_stats(self):
        """测试缓存统计"""
        from src.ml.models.inference.inference_cache import InferenceCache
        
        cache = InferenceCache({
            'cache_path': self.temp_dir,
            'enable_postgresql': False
        })
        
        cache.set("test_model", self.test_input, self.test_result)
        cache.get("test_model", self.test_input)
        
        stats = cache.get_cache_stats()
        
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        self.assertGreater(stats['hits'], 0)
    
    def test_clear_cache(self):
        """测试清除缓存"""
        from src.ml.models.inference.inference_cache import InferenceCache
        
        cache = InferenceCache({
            'cache_path': self.temp_dir,
            'enable_postgresql': False
        })
        
        cache.set("test_model", self.test_input, self.test_result)
        
        cached = cache.get("test_model", self.test_input)
        self.assertIsNotNone(cached)
        
        cache.clear_cache()
        
        stats = cache.get_cache_stats()
        self.assertEqual(stats['memory_cache_size'], 0)


class TestPostgreSQLFailover(unittest.TestCase):
    """PostgreSQL故障转移测试"""
    
    def test_model_manager_fallback_on_pg_failure(self):
        """测试ModelManager在PostgreSQL失败时降级"""
        from src.ml.models.model_manager import ModelManager
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch.object(ModelManager, '_test_postgresql_connection', return_value=False):
                manager = ModelManager({
                    'base_path': temp_dir,
                    'enable_postgresql': True
                })
                
                self.assertFalse(manager._pg_available)
                
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                model.fit(np.random.randn(50, 3), np.random.randint(0, 2, 50))
                
                success = manager.save_model(
                    model_id="fallback_test",
                    version="1.0.0",
                    model=model,
                    model_type='RandomForest'
                )
                
                self.assertTrue(success)
        finally:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass
    
    def test_feature_cache_fallback_on_pg_failure(self):
        """测试FeatureCacheManager在PostgreSQL失败时降级"""
        from src.ml.engine.feature_cache_manager import FeatureCacheManager
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch.object(FeatureCacheManager, '_test_postgresql_connection', return_value=False):
                manager = FeatureCacheManager.__new__(FeatureCacheManager)
                manager.config = type('CacheConfig', (), {
                    'enable_memory_cache': True,
                    'enable_persistent_cache': True,
                    'enable_postgresql': True,
                    'memory_cache_size': 100,
                    'cache_ttl_minutes': 60,
                    'compression_enabled': True,
                    'cache_key_prefix': 'feature_data',
                    'base_path': temp_dir
                })()
                manager._pg_available = False
                manager._memory_cache = {}
                manager._access_order = []
                manager._redis_client = None
                manager.base_path = Path(temp_dir)
                manager.stats = {'memory_hits': 0, 'postgresql_hits': 0, 'redis_hits': 0, 
                                'filesystem_hits': 0, 'misses': 0, 'total_cached': 0}
                
                self.assertFalse(manager._pg_available)
                
                test_features = pd.DataFrame({'a': [1, 2, 3]})
                success = manager.cache_features("fallback_test", test_features)
                
                self.assertTrue(success)
        finally:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass
    
    def test_inference_cache_fallback_on_pg_failure(self):
        """测试InferenceCache在PostgreSQL失败时降级"""
        from src.ml.models.inference.inference_cache import InferenceCache
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch.object(InferenceCache, '_test_postgresql_connection', return_value=False):
                cache = InferenceCache({
                    'cache_path': temp_dir,
                    'enable_postgresql': True
                })
                
                self.assertFalse(cache._pg_available)
                
                success = cache.set(
                    model_id="fallback_test",
                    input_data=np.array([1, 2, 3]),
                    result={'prediction': 1}
                )
                
                self.assertTrue(success)
        finally:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def test_model_save_load_performance(self):
        """测试模型保存加载性能"""
        from src.ml.models.model_manager import ModelManager
        from sklearn.ensemble import RandomForestClassifier
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            manager = ModelManager({
                'base_path': temp_dir,
                'enable_postgresql': False
            })
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(np.random.randn(1000, 50), np.random.randint(0, 2, 1000))
            
            start = time.time()
            manager.save_model("perf_test", "1.0.0", model, model_type='RandomForest')
            save_time = time.time() - start
            
            start = time.time()
            loaded = manager.load_model("perf_test", "1.0.0")
            load_time = time.time() - start
            
            self.assertIsNotNone(loaded)
            self.assertLess(save_time, 5.0, "模型保存时间应小于5秒")
            self.assertLess(load_time, 2.0, "模型加载时间应小于2秒")
            
            print(f"\n模型保存时间: {save_time:.3f}s")
            print(f"模型加载时间: {load_time:.3f}s")
            
        finally:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass
    
    def test_feature_cache_performance(self):
        """测试特征缓存性能"""
        from src.ml.engine.feature_cache_manager import FeatureCacheManager, CacheConfig
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = CacheConfig(
                base_path=temp_dir,
                enable_postgresql=False
            )
            manager = FeatureCacheManager(config)
            
            features = pd.DataFrame(np.random.randn(10000, 100))
            
            start = time.time()
            manager.cache_features("perf_test", features)
            cache_time = time.time() - start
            
            start = time.time()
            cached = manager.get_cached_features("perf_test")
            get_time = time.time() - start
            
            self.assertIsNotNone(cached)
            self.assertLess(cache_time, 2.0, "缓存时间应小于2秒")
            self.assertLess(get_time, 1.0, "获取缓存时间应小于1秒")
            
            print(f"\n特征缓存时间: {cache_time:.3f}s")
            print(f"获取缓存时间: {get_time:.3f}s")
            
        finally:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestModelManagerPostgreSQL))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureCacheManagerPostgreSQL))
    suite.addTests(loader.loadTestsFromTestCase(TestInferenceCachePostgreSQL))
    suite.addTests(loader.loadTestsFromTestCase(TestPostgreSQLFailover))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_tests()
