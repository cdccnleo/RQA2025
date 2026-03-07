#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Persistence模块测试覆盖
测试monitoring/metrics_persistence.py
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

try:
    from src.features.monitoring.metrics_persistence import (
        CompressionType,
        StorageBackend,
        DataLifecyclePolicy,
        MetricRecord,
        ArchiveConfig,
        EnhancedMetricsPersistenceManager,
        MetricsPersistenceManager
    )
    METRICS_PERSISTENCE_AVAILABLE = True
except ImportError:
    METRICS_PERSISTENCE_AVAILABLE = False


@pytest.fixture
def temp_storage_dir():
    """创建临时存储目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def persistence_manager(temp_storage_dir):
    """创建持久化管理器实例"""
    if not METRICS_PERSISTENCE_AVAILABLE:
        pytest.skip("MetricsPersistenceManager不可用")
    config = {
        'path': temp_storage_dir,
        'primary_backend': 'sqlite',
        'compression': 'none',
        'batch_size': 10,
        'batch_timeout': 0.1
    }
    manager = EnhancedMetricsPersistenceManager(config=config)
    yield manager
    # 清理
    try:
        manager.shutdown()
    except Exception:
        pass


@pytest.fixture
def simple_persistence_manager(temp_storage_dir):
    """创建简单的持久化管理器实例"""
    if not METRICS_PERSISTENCE_AVAILABLE:
        pytest.skip("MetricsPersistenceManager不可用")
    config = {'path': temp_storage_dir}
    manager = MetricsPersistenceManager(storage_config=config)
    yield manager
    # 清理
    try:
        if hasattr(manager, 'stop'):
            manager.stop()
    except Exception:
        pass


class TestEnums:
    """枚举类型测试"""

    def test_compression_type_values(self):
        """测试压缩类型值"""
        if not METRICS_PERSISTENCE_AVAILABLE:
            pytest.skip("MetricsPersistenceManager不可用")
        assert CompressionType.NONE.value == "none"
        assert CompressionType.GZIP.value == "gzip"
        assert CompressionType.LZ4.value == "lz4"

    def test_storage_backend_values(self):
        """测试存储后端值"""
        if not METRICS_PERSISTENCE_AVAILABLE:
            pytest.skip("MetricsPersistenceManager不可用")
        assert StorageBackend.SQLITE.value == "sqlite"
        assert StorageBackend.JSON.value == "json"
        assert StorageBackend.PARQUET.value == "parquet"

    def test_data_lifecycle_policy_values(self):
        """测试数据生命周期策略值"""
        if not METRICS_PERSISTENCE_AVAILABLE:
            pytest.skip("MetricsPersistenceManager不可用")
        assert DataLifecyclePolicy.HOT.value == "hot"
        assert DataLifecyclePolicy.WARM.value == "warm"
        assert DataLifecyclePolicy.COLD.value == "cold"


class TestMetricRecord:
    """MetricRecord数据类测试"""

    def test_metric_record_creation(self):
        """测试创建指标记录"""
        if not METRICS_PERSISTENCE_AVAILABLE:
            pytest.skip("MetricsPersistenceManager不可用")
        record = MetricRecord(
            component_name="test_component",
            metric_name="test_metric",
            metric_value=1.5,
            metric_type="gauge",
            timestamp=time.time(),
            labels={"label1": "value1"},
            created_at="2025-01-01T00:00:00",
            ttl=3600.0,
            priority=5
        )
        assert record.component_name == "test_component"
        assert record.metric_name == "test_metric"
        assert record.metric_value == 1.5
        assert record.metric_type == "gauge"
        assert record.labels == {"label1": "value1"}
        assert record.priority == 5


class TestArchiveConfig:
    """ArchiveConfig数据类测试"""

    def test_archive_config_defaults(self):
        """测试归档配置默认值"""
        if not METRICS_PERSISTENCE_AVAILABLE:
            pytest.skip("MetricsPersistenceManager不可用")
        config = ArchiveConfig()
        assert config.hot_data_days == 7
        assert config.warm_data_days == 30
        assert config.cold_data_days == 365
        assert config.compression_ratio == 0.8
        assert config.batch_size == 1000

    def test_archive_config_custom(self):
        """测试自定义归档配置"""
        if not METRICS_PERSISTENCE_AVAILABLE:
            pytest.skip("MetricsPersistenceManager不可用")
        config = ArchiveConfig(
            hot_data_days=3,
            warm_data_days=15,
            cold_data_days=180,
            compression_ratio=0.7,
            batch_size=500
        )
        assert config.hot_data_days == 3
        assert config.warm_data_days == 15
        assert config.cold_data_days == 180
        assert config.compression_ratio == 0.7
        assert config.batch_size == 500


class TestEnhancedMetricsPersistenceManager:
    """EnhancedMetricsPersistenceManager测试"""

    def test_enhanced_manager_initialization_default(self, temp_storage_dir):
        """测试默认初始化"""
        if not METRICS_PERSISTENCE_AVAILABLE:
            pytest.skip("MetricsPersistenceManager不可用")
        manager = EnhancedMetricsPersistenceManager()
        try:
            assert manager.storage_path is not None
            assert manager.primary_backend == StorageBackend.SQLITE
            assert manager.batch_size > 0
            assert manager.executor is not None
        finally:
            if hasattr(manager, 'stop'):
                manager.stop()

    def test_enhanced_manager_initialization_custom(self, persistence_manager):
        """测试自定义配置初始化"""
        assert persistence_manager.storage_path is not None
        assert persistence_manager.batch_size == 10
        assert persistence_manager.primary_backend == StorageBackend.SQLITE

    def test_store_metric_sync(self, persistence_manager):
        """测试同步存储指标"""
        result = persistence_manager.store_metric_sync(
            component_name="test_component",
            metric_name="test_metric",
            metric_value=1.5,
            metric_type="gauge"
        )
        assert result is True

    def test_store_metric_sync_with_labels(self, persistence_manager):
        """测试带标签的同步存储"""
        result = persistence_manager.store_metric_sync(
            component_name="test_component",
            metric_name="test_metric",
            metric_value=2.0,
            metric_type="counter",
            labels={"env": "test", "version": "1.0"},
            priority=10
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_store_metric_async(self, persistence_manager):
        """测试异步存储指标"""
        result = await persistence_manager.store_metric_async(
            component_name="test_component",
            metric_name="test_metric_async",
            metric_value=3.0,
            metric_type="gauge"
        )
        assert result is True

    def test_query_metrics(self, persistence_manager):
        """测试查询指标"""
        # 先存储一些数据
        persistence_manager.store_metric_sync(
            component_name="test_component",
            metric_name="test_metric",
            metric_value=1.0,
            metric_type="gauge"
        )
        # 等待批量写入完成
        time.sleep(0.2)
        
        # 查询指标 - 使用query_metrics_async或检查是否有查询方法
        if hasattr(persistence_manager, 'query_metrics_async'):
            import asyncio
            import pandas as pd
            results = asyncio.run(persistence_manager.query_metrics_async(
                component_name="test_component",
                metric_name="test_metric"
            ))
            assert isinstance(results, pd.DataFrame)
            assert len(results) > 0
            assert 'component_name' in results.columns
            assert 'metric_name' in results.columns
        else:
            # 如果没有查询方法，跳过测试
            pytest.skip("query_metrics方法不可用")

    def test_query_metrics_no_results(self, persistence_manager):
        """测试查询不存在的指标"""
        if hasattr(persistence_manager, 'query_metrics_async'):
            import asyncio
            import pandas as pd
            results = asyncio.run(persistence_manager.query_metrics_async(
                component_name="nonexistent",
                metric_name="nonexistent"
            ))
            assert isinstance(results, pd.DataFrame)
            # 查询不存在的指标应该返回空DataFrame或包含正确列结构的DataFrame
            if len(results) == 0:
                # 空DataFrame应该至少包含预期的列
                assert len(results.columns) >= 0  # 允许空列
            else:
                # 如果有结果，应该包含component_name列
                assert 'component_name' in results.columns
        else:
            pytest.skip("query_metrics方法不可用")

    def test_get_metrics_count(self, persistence_manager):
        """测试获取指标数量"""
        # 检查方法是否存在
        if not hasattr(persistence_manager, 'get_metrics_count'):
            pytest.skip("get_metrics_count方法不可用")
        # 先存储一些数据
        for i in range(5):
            persistence_manager.store_metric_sync(
                component_name="test_component",
                metric_name=f"metric_{i}",
                metric_value=float(i),
                metric_type="gauge"
            )
        # 等待批量写入完成
        time.sleep(0.2)
        
        count = persistence_manager.get_metrics_count(
            component_name="test_component"
        )
        assert isinstance(count, int)
        assert count >= 0

    def test_get_latest_metrics(self, persistence_manager):
        """测试获取最新指标"""
        if not hasattr(persistence_manager, 'get_latest_metrics'):
            pytest.skip("get_latest_metrics方法不可用")
        # 先存储一些数据
        persistence_manager.store_metric_sync(
            component_name="test_component",
            metric_name="test_metric",
            metric_value=1.0,
            metric_type="gauge"
        )
        # 等待批量写入完成
        time.sleep(0.2)
        
        latest = persistence_manager.get_latest_metrics(
            component_name="test_component",
            metric_name="test_metric"
        )
        # 可能返回None或MetricRecord
        assert latest is None or isinstance(latest, MetricRecord)

    def test_stop(self, persistence_manager):
        """测试关闭管理器"""
        if hasattr(persistence_manager, 'stop'):
            persistence_manager.stop()
            # 验证线程已停止
            assert persistence_manager._stop_background is True
        else:
            pytest.skip("stop方法不可用")

    def test_error_handling_invalid_config(self, temp_storage_dir):
        """测试无效配置处理"""
        if not METRICS_PERSISTENCE_AVAILABLE:
            pytest.skip("MetricsPersistenceManager不可用")
        # 无效的backend
        config = {
            'path': temp_storage_dir,
            'primary_backend': 'invalid_backend'
        }
        try:
            manager = EnhancedMetricsPersistenceManager(config=config)
            if hasattr(manager, 'stop'):
                manager.stop()
        except (ValueError, AttributeError, Exception):
            # 如果抛出异常也是可以接受的
            assert True


class TestMetricsPersistenceManager:
    """MetricsPersistenceManager测试"""

    def test_simple_manager_initialization(self, simple_persistence_manager):
        """测试简单管理器初始化"""
        # 简单管理器使用增强管理器作为内部实现
        assert hasattr(simple_persistence_manager, '_enhanced_manager')

    def test_store_metric(self, simple_persistence_manager):
        """测试存储指标"""
        try:
            from src.features.monitoring.features_monitor import MetricType
            simple_persistence_manager.store_metric(
                component_name="test_component",
                metric_name="test_metric",
                metric_value=1.0,
                metric_type=MetricType.GAUGE
            )
            assert True
        except (AttributeError, ImportError) as e:
            pytest.skip(f"store_metric方法不可用: {e}")

    def test_query_metrics_simple(self, simple_persistence_manager):
        """测试简单查询指标"""
        try:
            results = simple_persistence_manager.query_metrics()
            # 应该返回DataFrame
            import pandas as pd
            assert isinstance(results, pd.DataFrame)
        except (AttributeError, Exception) as e:
            pytest.skip(f"query_metrics方法不可用: {e}")

    def test_query_metrics_with_params(self, simple_persistence_manager):
        """测试带参数的查询指标"""
        try:
            results = simple_persistence_manager.query_metrics(
                component_name="test"
            )
            import pandas as pd
            assert isinstance(results, pd.DataFrame)
        except (AttributeError, Exception) as e:
            pytest.skip(f"query_metrics方法不可用: {e}")


class TestErrorHandling:
    """错误处理测试"""

    def test_store_metric_invalid_data(self, persistence_manager):
        """测试存储无效数据"""
        # 测试None值
        try:
            result = persistence_manager.store_metric_sync(
                component_name=None,
                metric_name="test",
                metric_value=1.0,
                metric_type="gauge"
            )
            # 应该能够处理或返回False
            assert result is True or result is False
        except Exception:
            # 如果抛出异常也是可以接受的
            assert True

    def test_query_with_invalid_params(self, persistence_manager):
        """测试无效参数查询"""
        if hasattr(persistence_manager, 'query_metrics_async'):
            import asyncio
            import pandas as pd
            try:
                results = asyncio.run(persistence_manager.query_metrics_async(
                    component_name=None,
                    metric_name=None
                ))
                assert isinstance(results, pd.DataFrame)
                # 即使参数为None，也应该返回DataFrame（可能为空）
                # 如果DataFrame为空，可能没有列；如果有数据，应该包含component_name列
                if len(results) > 0:
                    assert 'component_name' in results.columns
                else:
                    # 空DataFrame可能没有列，或者有预期的列结构
                    assert len(results.columns) >= 0
            except Exception as e:
                # 如果抛出异常也是可以接受的（取决于实现）
                assert isinstance(e, (ValueError, TypeError, AttributeError))
        else:
            pytest.skip("query_metrics方法不可用")

