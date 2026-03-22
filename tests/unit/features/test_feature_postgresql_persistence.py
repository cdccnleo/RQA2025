#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程模块PostgreSQL持久化测试

测试内容:
1. FeatureSaver PostgreSQL存储测试
2. FeatureStore PostgreSQL存储测试
3. MetricsPersistenceManager PostgreSQL存储测试
4. 降级机制测试
5. 数据同步测试
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
        elif "SELECT" in query and "FROM feature_store" in query:
            for key, value in self._data_store.items():
                self._results.append(value)
        elif "SELECT" in query and "FROM feature_cache" in query:
            for key, value in self._data_store.items():
                self._results.append(value)
        elif "SELECT" in query and "FROM monitoring_metrics" in query:
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


class TestFeatureSaverPostgreSQL(unittest.TestCase):
    """FeatureSaver PostgreSQL存储测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = pd.DataFrame({
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
        from src.features.core.feature_saver import FeatureSaver
        
        saver = FeatureSaver(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        self.assertFalse(saver._pg_available)
        self.assertEqual(saver.primary_backend if hasattr(saver, 'primary_backend') else None, None)
    
    def test_save_to_filesystem_when_pg_disabled(self):
        """测试PostgreSQL禁用时保存到文件系统"""
        from src.features.core.feature_saver import FeatureSaver
        
        saver = FeatureSaver(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        success = saver.save_features(
            self.test_data,
            "test_feature",
            format="pickle",
            metadata={"source": "unit_test"}
        )
        
        self.assertTrue(success)
        
        loaded = saver.load_features(saver.last_save_info['feature_id'])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, self.test_data.shape)
    
    def test_save_and_load_roundtrip(self):
        """测试保存和加载的完整流程"""
        from src.features.core.feature_saver import FeatureSaver
        
        saver = FeatureSaver(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        feature_name = "roundtrip_test"
        success = saver.save_features(
            self.test_data,
            feature_name,
            format="pickle",
            metadata={"test": True}
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(saver.last_save_info)
        
        feature_id = saver.last_save_info['feature_id']
        loaded = saver.load_features(feature_id, format="pickle")
        
        self.assertIsNotNone(loaded)
        pd.testing.assert_frame_equal(
            loaded.reset_index(drop=True),
            self.test_data.reset_index(drop=True),
            check_dtype=False
        )
    
    def test_list_features(self):
        """测试列出特征"""
        from src.features.core.feature_saver import FeatureSaver
        
        saver = FeatureSaver(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        saver.save_features(self.test_data, "list_test_1", format="pickle")
        saver.save_features(self.test_data, "list_test_2", format="pickle")
        
        features = saver.list_features()
        self.assertGreaterEqual(len(features), 2)
    
    def test_delete_features(self):
        """测试删除特征"""
        from src.features.core.feature_saver import FeatureSaver
        
        saver = FeatureSaver(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        success = saver.save_features(self.test_data, "delete_test", format="pickle")
        self.assertTrue(success)
        
        feature_id = saver.last_save_info['feature_id']
        
        success = saver.delete_features(feature_id)
        self.assertTrue(success)
        
        loaded = saver.load_features(feature_id)
        self.assertIsNone(loaded)
    
    def test_get_storage_stats(self):
        """测试获取存储统计"""
        from src.features.core.feature_saver import FeatureSaver
        
        saver = FeatureSaver(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        saver.save_features(self.test_data, "stats_test", format="pickle")
        
        stats = saver.get_storage_stats()
        
        self.assertIn('total_features', stats)
        self.assertIn('postgresql_available', stats)
        self.assertFalse(stats['postgresql_available'])


class TestFeatureStorePostgreSQL(unittest.TestCase):
    """FeatureStore PostgreSQL存储测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = pd.DataFrame({
            'price': np.random.randn(100) * 100 + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'returns': np.random.randn(100) * 0.02
        })
    
    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                pass
    
    def _create_mock_config(self):
        """创建模拟配置"""
        from src.features.core.config import FeatureRegistrationConfig, FeatureType
        
        config = FeatureRegistrationConfig(
            name="test_feature",
            feature_type=FeatureType.TECHNICAL,
            params={'window': 20},
            dependencies=['close_price']
        )
        return config
    
    def test_init_with_postgresql_disabled(self):
        """测试PostgreSQL禁用时的初始化"""
        from src.features.core.feature_store import FeatureStore, StoreConfig
        
        config = StoreConfig(
            base_path=self.temp_dir,
            enable_postgresql=False
        )
        
        store = FeatureStore(config)
        
        self.assertFalse(store._pg_available)
    
    def test_store_and_load_feature(self):
        """测试存储和加载特征"""
        from src.features.core.feature_store import FeatureStore, StoreConfig
        
        config = StoreConfig(
            base_path=self.temp_dir,
            enable_postgresql=False,
            ttl_hours=0
        )
        
        store = FeatureStore(config)
        feature_config = self._create_mock_config()
        
        success = store.store_feature(
            "sma_20",
            self.test_data,
            feature_config,
            description="Test SMA feature"
        )
        
        self.assertTrue(success)
        
        result = store.load_feature("sma_20", {'window': 20})
        self.assertIsNotNone(result)
        
        data, metadata = result
        self.assertEqual(data.shape, self.test_data.shape)
        self.assertEqual(metadata.feature_name, "sma_20")
    
    def test_list_features(self):
        """测试列出特征"""
        from src.features.core.feature_store import FeatureStore, StoreConfig
        
        config = StoreConfig(
            base_path=self.temp_dir,
            enable_postgresql=False,
            ttl_hours=0
        )
        
        store = FeatureStore(config)
        feature_config = self._create_mock_config()
        
        store.store_feature("list_test_1", self.test_data, feature_config)
        store.store_feature("list_test_2", self.test_data, feature_config)
        
        features = store.list_features()
        self.assertGreaterEqual(len(features), 2)
    
    def test_delete_feature(self):
        """测试删除特征"""
        from src.features.core.feature_store import FeatureStore, StoreConfig
        
        config = StoreConfig(
            base_path=self.temp_dir,
            enable_postgresql=False,
            ttl_hours=0
        )
        
        store = FeatureStore(config)
        feature_config = self._create_mock_config()
        
        store.store_feature("delete_test", self.test_data, feature_config)
        
        result = store.load_feature("delete_test", {'window': 20})
        self.assertIsNotNone(result)
        
        data, metadata = result
        success = store.delete_feature(metadata.feature_id)
        self.assertTrue(success)
        
        result = store.load_feature("delete_test", {'window': 20})
        self.assertIsNone(result)
    
    def test_get_store_stats(self):
        """测试获取存储统计"""
        from src.features.core.feature_store import FeatureStore, StoreConfig
        
        config = StoreConfig(
            base_path=self.temp_dir,
            enable_postgresql=False,
            ttl_hours=0
        )
        
        store = FeatureStore(config)
        feature_config = self._create_mock_config()
        
        store.store_feature("stats_test", self.test_data, feature_config)
        
        stats = store.get_store_stats()
        
        self.assertIn('total_stored', stats)
        self.assertIn('postgresql_available', stats)
        self.assertFalse(stats['postgresql_available'])


class TestMetricsPersistenceManagerPostgreSQL(unittest.TestCase):
    """MetricsPersistenceManager PostgreSQL存储测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self._manager = None
    
    def tearDown(self):
        """清理测试环境"""
        if self._manager:
            try:
                self._manager.stop()
            except Exception:
                pass
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                pass
    
    def test_init_with_postgresql_disabled(self):
        """测试PostgreSQL禁用时的初始化"""
        from src.features.monitoring.metrics_persistence import EnhancedMetricsPersistenceManager
        
        self._manager = EnhancedMetricsPersistenceManager({
            'path': self.temp_dir,
            'enable_postgresql': False
        })
        
        self.assertFalse(self._manager._pg_available)
        self.assertEqual(self._manager.primary_backend.value, 'sqlite')
    
    def test_store_and_query_metrics(self):
        """测试存储和查询指标"""
        from src.features.monitoring.metrics_persistence import EnhancedMetricsPersistenceManager
        
        self._manager = EnhancedMetricsPersistenceManager({
            'path': self.temp_dir,
            'enable_postgresql': False
        })
        
        success = self._manager.store_metric_sync(
            component_name="test_component",
            metric_name="test_metric",
            metric_value=123.45,
            metric_type="gauge",
            labels={"env": "test"}
        )
        
        self.assertTrue(success)
        
        import asyncio
        df = asyncio.run(self._manager.query_metrics_async(
            component_name="test_component"
        ))
        
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
    
    def test_get_metrics_count(self):
        """测试获取指标数量"""
        from src.features.monitoring.metrics_persistence import EnhancedMetricsPersistenceManager
        
        self._manager = EnhancedMetricsPersistenceManager({
            'path': self.temp_dir,
            'enable_postgresql': False
        })
        
        for i in range(5):
            self._manager.store_metric_sync(
                component_name="count_test",
                metric_name=f"metric_{i}",
                metric_value=i,
                metric_type="counter"
            )
        
        count = self._manager.get_metrics_count(component_name="count_test")
        self.assertEqual(count, 5)
    
    def test_get_latest_metrics(self):
        """测试获取最新指标"""
        from src.features.monitoring.metrics_persistence import EnhancedMetricsPersistenceManager
        
        self._manager = EnhancedMetricsPersistenceManager({
            'path': self.temp_dir,
            'enable_postgresql': False
        })
        
        self._manager.store_metric_sync(
            component_name="latest_test",
            metric_name="value",
            metric_value=100.0,
            metric_type="gauge"
        )
        
        time.sleep(0.1)
        
        self._manager.store_metric_sync(
            component_name="latest_test",
            metric_name="value",
            metric_value=200.0,
            metric_type="gauge"
        )
        
        latest = self._manager.get_latest_metrics("latest_test", "value")
        
        self.assertIsNotNone(latest)
        self.assertEqual(latest.metric_value, 200.0)
    
    def test_get_storage_stats(self):
        """测试获取存储统计"""
        from src.features.monitoring.metrics_persistence import EnhancedMetricsPersistenceManager
        
        self._manager = EnhancedMetricsPersistenceManager({
            'path': self.temp_dir,
            'enable_postgresql': False
        })
        
        self._manager.store_metric_sync(
            component_name="stats_test",
            metric_name="test",
            metric_value=1.0,
            metric_type="counter"
        )
        
        stats = self._manager.get_storage_stats()
        
        self.assertIn('primary_backend', stats)
        self.assertIn('postgresql_available', stats)
        self.assertFalse(stats['postgresql_available'])
        self.assertEqual(stats['primary_backend'], 'sqlite')


class TestPostgreSQLFailover(unittest.TestCase):
    """PostgreSQL故障转移测试"""
    
    def test_feature_saver_fallback_on_pg_failure(self):
        """测试FeatureSaver在PostgreSQL失败时降级"""
        from src.features.core.feature_saver import FeatureSaver
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch.object(FeatureSaver, '_test_postgresql_connection', return_value=False):
                saver = FeatureSaver(
                    base_path=temp_dir,
                    enable_postgresql=True
                )
                
                self.assertFalse(saver._pg_available)
                
                test_data = pd.DataFrame({'a': [1, 2, 3]})
                success = saver.save_features(test_data, "fallback_test")
                
                self.assertTrue(success)
                self.assertEqual(saver.last_save_info['storage_type'], 'filesystem')
        finally:
            shutil.rmtree(temp_dir)
    
    def test_feature_store_fallback_on_pg_failure(self):
        """测试FeatureStore在PostgreSQL失败时降级"""
        from src.features.core.feature_store import FeatureStore, StoreConfig
        from src.features.core.config import FeatureRegistrationConfig, FeatureType
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch.object(FeatureStore, '_test_postgresql_connection', return_value=False):
                config = StoreConfig(
                    base_path=temp_dir,
                    enable_postgresql=True,
                    ttl_hours=0
                )
                
                store = FeatureStore(config)
                
                self.assertFalse(store._pg_available)
                
                test_data = pd.DataFrame({'a': [1, 2, 3]})
                feature_config = FeatureRegistrationConfig(
                    name="test",
                    feature_type=FeatureType.TECHNICAL,
                    params={},
                    dependencies=[]
                )
                
                success = store.store_feature("fallback_test", test_data, feature_config)
                self.assertTrue(success)
                
                stats = store.get_store_stats()
                self.assertGreater(stats['filesystem_stores'], 0)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_metrics_persistence_fallback_on_pg_failure(self):
        """测试MetricsPersistenceManager在PostgreSQL失败时降级"""
        from src.features.monitoring.metrics_persistence import EnhancedMetricsPersistenceManager
        
        temp_dir = tempfile.mkdtemp()
        manager = None
        
        try:
            with patch.object(EnhancedMetricsPersistenceManager, '_test_postgresql_connection', return_value=False):
                manager = EnhancedMetricsPersistenceManager({
                    'path': temp_dir,
                    'enable_postgresql': True
                })
                
                self.assertFalse(manager._pg_available)
                self.assertEqual(manager.primary_backend.value, 'sqlite')
                
                success = manager.store_metric_sync(
                    component_name="fallback_test",
                    metric_name="test",
                    metric_value=1.0,
                    metric_type="counter"
                )
                
                self.assertTrue(success)
        finally:
            if manager:
                manager.stop()
            time.sleep(0.5)
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass


class TestCompatibilityLayer(unittest.TestCase):
    """兼容性层测试"""
    
    def test_metrics_persistence_manager_compatibility(self):
        """测试MetricsPersistenceManager兼容性"""
        from src.features.monitoring.metrics_persistence import (
            MetricsPersistenceManager,
            MetricType
        )
        
        temp_dir = tempfile.mkdtemp()
        manager = None
        
        try:
            manager = MetricsPersistenceManager({
                'path': temp_dir,
                'enable_postgresql': False
            })
            
            manager.store_metric(
                component_name="compat_test",
                metric_name="test_metric",
                metric_value=42.0,
                metric_type=MetricType.GAUGE
            )
            
            df = manager.query_metrics(component_name="compat_test")
            
            self.assertFalse(df.empty)
        finally:
            if manager:
                manager.stop()
            time.sleep(0.5)
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                pass


def run_tests():
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureSaverPostgreSQL))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureStorePostgreSQL))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsPersistenceManagerPostgreSQL))
    suite.addTests(loader.loadTestsFromTestCase(TestPostgreSQLFailover))
    suite.addTests(loader.loadTestsFromTestCase(TestCompatibilityLayer))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_tests()
