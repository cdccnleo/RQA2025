#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理层持久化模块单元测试

测试 PostgreSQL 优先存储策略的实现。
"""

import os
import sys
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.data.persistence.data_persistence import (
    DataPersistence,
    CachePersistence,
    LineagePersistence,
    QualityPersistence,
    CompliancePersistence,
    CacheEntryMetadata,
    LineageRecord,
    QualityMetricRecord,
    ComplianceCheckRecord,
    get_data_persistence,
    get_cache_persistence,
    get_lineage_persistence,
    get_quality_persistence
)


class TestCacheEntryMetadata:
    """缓存条目元数据测试"""
    
    def test_create_metadata(self):
        """测试创建元数据"""
        metadata = CacheEntryMetadata(
            cache_key="test_key",
            cache_type="general",
            data_size_bytes=1024,
            created_at=datetime.now(),
            tags=["test", "unit"]
        )
        
        assert metadata.cache_key == "test_key"
        assert metadata.cache_type == "general"
        assert metadata.data_size_bytes == 1024
        assert metadata.access_count == 0
        assert "test" in metadata.tags
    
    def test_metadata_with_expiration(self):
        """测试带过期时间的元数据"""
        now = datetime.now()
        expires = now + timedelta(hours=1)
        
        metadata = CacheEntryMetadata(
            cache_key="test_key",
            cache_type="general",
            data_size_bytes=512,
            created_at=now,
            expires_at=expires
        )
        
        assert metadata.expires_at is not None
        assert metadata.expires_at > metadata.created_at


class TestLineageRecord:
    """数据血缘记录测试"""
    
    def test_create_lineage_record(self):
        """测试创建血缘记录"""
        record = LineageRecord(
            data_type="stock_data",
            source_info={"source": "tushare", "table": "daily"},
            transform_info={"steps": ["clean", "normalize"]},
            dependencies=["raw_stock_data"]
        )
        
        assert record.data_type == "stock_data"
        assert record.source_info["source"] == "tushare"
        assert "clean" in record.transform_info["steps"]
        assert "raw_stock_data" in record.dependencies


class TestQualityMetricRecord:
    """数据质量指标记录测试"""
    
    def test_create_quality_record(self):
        """测试创建质量指标记录"""
        record = QualityMetricRecord(
            data_type="market_data",
            completeness=0.98,
            accuracy=0.95,
            timeliness=0.99,
            consistency=0.97,
            uniqueness=0.96,
            issues=["missing_values: 2%"]
        )
        
        assert record.data_type == "market_data"
        assert record.completeness == 0.98
        assert len(record.issues) == 1


class TestDataPersistence:
    """数据持久化测试"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """创建临时存储目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def persistence(self, temp_storage_dir):
        """创建持久化实例"""
        with patch('src.infrastructure.persistence.database_config.get_db_config') as mock_config:
            mock_config.return_value = None
            return DataPersistence(storage_dir=temp_storage_dir)
    
    def test_save_and_load_cache_entry(self, persistence):
        """测试保存和加载缓存条目"""
        key = "test_cache_key"
        data = {"symbol": "000001.SZ", "price": 10.5, "volume": 1000000}
        
        # 保存
        result = persistence.save_cache_entry(
            cache_key=key,
            data=data,
            cache_type="market",
            ttl_seconds=3600,
            tags=["stock", "daily"]
        )
        assert result is True
        
        # 加载
        loaded = persistence.load_cache_entry(key)
        assert loaded is not None
        loaded_data, metadata = loaded
        assert loaded_data["symbol"] == "000001.SZ"
        assert metadata.cache_type == "market"
        assert "stock" in metadata.tags
    
    def test_delete_cache_entry(self, persistence):
        """测试删除缓存条目"""
        key = "test_delete_key"
        data = {"test": "data"}
        
        # 保存
        persistence.save_cache_entry(key, data)
        
        # 确认存在
        assert persistence.load_cache_entry(key) is not None
        
        # 删除
        result = persistence.delete_cache_entry(key)
        assert result is True
        
        # 确认已删除
        assert persistence.load_cache_entry(key) is None
    
    def test_list_cache_entries(self, persistence):
        """测试列出缓存条目"""
        # 保存多个条目
        for i in range(3):
            persistence.save_cache_entry(
                cache_key=f"key_{i}",
                data={"index": i},
                cache_type="test"
            )
        
        # 列出所有
        entries = persistence.list_cache_entries()
        assert len(entries) >= 3
        
        # 按类型过滤
        entries = persistence.list_cache_entries(cache_type="test")
        assert len(entries) >= 3
    
    def test_cleanup_expired(self, persistence):
        """测试清理过期缓存"""
        # 保存一个立即过期的条目
        persistence.save_cache_entry(
            cache_key="expired_key",
            data={"test": "expired"},
            ttl_seconds=-1  # 立即过期
        )
        
        # 保存一个未过期的条目
        persistence.save_cache_entry(
            cache_key="valid_key",
            data={"test": "valid"},
            ttl_seconds=3600
        )
        
        # 清理过期
        count = persistence.cleanup_expired()
        assert count >= 1
        
        # 验证过期条目已删除
        assert persistence.load_cache_entry("expired_key") is None
        assert persistence.load_cache_entry("valid_key") is not None
    
    def test_get_stats(self, persistence):
        """测试获取统计信息"""
        stats = persistence.get_stats()
        
        assert "total_saves" in stats
        assert "total_loads" in stats
        assert "postgresql_available" in stats
        assert "storage_dir" in stats


class TestCachePersistence:
    """缓存持久化测试"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """创建临时存储目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def cache_persistence(self, temp_storage_dir):
        """创建缓存持久化实例"""
        with patch('src.infrastructure.persistence.database_config.get_db_config') as mock_config:
            mock_config.return_value = None
            return CachePersistence(storage_dir=temp_storage_dir)
    
    def test_get_or_set(self, cache_persistence):
        """测试 get_or_set 方法"""
        key = "test_get_or_set"
        call_count = 0
        
        def loader_func():
            nonlocal call_count
            call_count += 1
            return {"loaded": True}
        
        # 第一次调用，应该执行 loader
        result1 = cache_persistence.get_or_set(key, loader_func)
        assert result1["loaded"] is True
        assert call_count == 1
        
        # 第二次调用，应该从缓存获取
        result2 = cache_persistence.get_or_set(key, loader_func)
        assert result2["loaded"] is True
        assert call_count == 1  # loader 不应再次调用


class TestLineagePersistence:
    """数据血缘持久化测试"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """创建临时存储目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def lineage_persistence(self, temp_storage_dir):
        """创建血缘持久化实例"""
        with patch('src.infrastructure.persistence.database_config.get_db_config') as mock_config:
            mock_config.return_value = None
            return LineagePersistence(storage_dir=temp_storage_dir)
    
    def test_save_and_get_lineage(self, lineage_persistence):
        """测试保存和获取血缘记录"""
        # 保存
        lineage_persistence.save_lineage(
            data_type="stock_price",
            source_info={"source": "tushare", "api": "daily"},
            transform_info={"steps": ["clean", "validate"]},
            dependencies=["raw_data"]
        )
        
        # 获取
        records = lineage_persistence.get_lineage("stock_price")
        assert len(records) >= 1
        assert records[0].data_type == "stock_price"
        assert records[0].source_info["source"] == "tushare"


class TestQualityPersistence:
    """数据质量持久化测试"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """创建临时存储目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def quality_persistence(self, temp_storage_dir):
        """创建质量持久化实例"""
        with patch('src.infrastructure.persistence.database_config.get_db_config') as mock_config:
            mock_config.return_value = None
            return QualityPersistence(storage_dir=temp_storage_dir)
    
    def test_save_and_get_quality_metric(self, quality_persistence):
        """测试保存和获取质量指标"""
        # 保存
        quality_persistence.save_quality_metric(
            data_type="market_data",
            completeness=0.98,
            accuracy=0.95,
            timeliness=0.99,
            consistency=0.97,
            uniqueness=0.96,
            issues=["minor_issues"]
        )
        
        # 获取历史
        records = quality_persistence.get_quality_history("market_data", days=1)
        assert len(records) >= 1
        assert records[0].completeness == 0.98
        assert records[0].accuracy == 0.95


class TestCompliancePersistence:
    """合规检查持久化测试"""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """创建临时存储目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def compliance_persistence(self, temp_storage_dir):
        """创建合规持久化实例"""
        with patch('src.infrastructure.persistence.database_config.get_db_config') as mock_config:
            mock_config.return_value = None
            return CompliancePersistence(storage_dir=temp_storage_dir)
    
    def test_save_and_get_compliance_check(self, compliance_persistence):
        """测试保存和获取合规检查记录"""
        # 保存
        compliance_persistence.save_compliance_check(
            policy_id="data_privacy_policy",
            data_type="user_data",
            is_compliant=True,
            issues=[],
            check_duration_ms=150.5
        )
        
        # 获取历史
        records = compliance_persistence.get_compliance_history(
            policy_id="data_privacy_policy",
            days=1
        )
        assert len(records) >= 1
        assert records[0].policy_id == "data_privacy_policy"
        assert records[0].is_compliant is True


class TestSingletonFunctions:
    """单例函数测试"""
    
    def test_get_data_persistence_singleton(self):
        """测试数据持久化单例"""
        with patch('src.infrastructure.persistence.database_config.get_db_config') as mock_config:
            mock_config.return_value = None
            
            instance1 = get_data_persistence()
            instance2 = get_data_persistence()
            
            assert instance1 is instance2
    
    def test_get_cache_persistence_singleton(self):
        """测试缓存持久化单例"""
        with patch('src.infrastructure.persistence.database_config.get_db_config') as mock_config:
            mock_config.return_value = None
            
            instance1 = get_cache_persistence()
            instance2 = get_cache_persistence()
            
            assert instance1 is instance2


class TestPostgreSQLIntegration:
    """PostgreSQL 集成测试（需要数据库连接）"""
    
    @pytest.fixture
    def pg_persistence(self):
        """创建连接 PostgreSQL 的持久化实例"""
        # 仅在环境变量设置时运行
        if not os.getenv('POSTGRES_PASSWORD'):
            pytest.skip("PostgreSQL 环境变量未设置")
        
        return DataPersistence()
    
    @pytest.mark.skipif(
        not os.getenv('POSTGRES_PASSWORD'),
        reason="PostgreSQL 环境变量未设置"
    )
    def test_postgresql_save_and_load(self, pg_persistence):
        """测试 PostgreSQL 保存和加载"""
        key = "pg_test_key"
        data = {"test": "postgresql", "timestamp": datetime.now().isoformat()}
        
        # 保存
        result = pg_persistence.save_cache_entry(
            cache_key=key,
            data=data,
            cache_type="test"
        )
        
        if pg_persistence._pg_available:
            assert result is True
            
            # 加载
            loaded = pg_persistence.load_cache_entry(key)
            assert loaded is not None
            assert loaded[0]["test"] == "postgresql"
            
            # 清理
            pg_persistence.delete_cache_entry(key)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
