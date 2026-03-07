"""
data_ecosystem_manager.py 边界测试补充 - 第3批
覆盖未覆盖的监控、清理和统计方法
"""
import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
import threading

from src.data.ecosystem.data_ecosystem_manager import (
    DataEcosystemManager,
    EcosystemConfig,
    DataAsset,
    DataLineage,
    DataContract,
    DataMarketItem
)
from src.data.interfaces.standard_interfaces import DataSourceType


@pytest.fixture
def mock_integration_manager(monkeypatch):
    """Mock 基础设施集成管理器"""
    fake_mgr = Mock()
    fake_mgr._initialized = False
    fake_mgr._integration_config = {}
    fake_mgr.initialize = Mock()
    fake_mgr.get_health_check_bridge = Mock(return_value=None)
    
    def get_manager():
        return fake_mgr
    
    monkeypatch.setattr(
        'src.data.ecosystem.data_ecosystem_manager.get_data_integration_manager',
        get_manager
    )
    monkeypatch.setattr(
        'src.data.ecosystem.data_ecosystem_manager.log_data_operation',
        Mock()
    )
    monkeypatch.setattr(
        'src.data.ecosystem.data_ecosystem_manager.record_data_metric',
        Mock()
    )
    monkeypatch.setattr(
        'src.data.ecosystem.data_ecosystem_manager.publish_data_event',
        Mock()
    )
    
    return fake_mgr


def test_get_marketplace_items_with_filters(mock_integration_manager):
    """测试获取市场商品（带过滤条件，覆盖 670-737 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        
        # 创建市场商品
        item1 = DataMarketItem(
            item_id="item1",
            asset_id="asset1",
            title="Test Item 1",
            description="Description 1",
            price=100.0,
            currency="CNY",
            category="financial_data",
            tags=["tag1", "tag2"],
            quality_score=0.9,
            rating=4.5,
            reviews_count=10,
            published=True
        )
        item2 = DataMarketItem(
            item_id="item2",
            asset_id="asset2",
            title="Test Item 2",
            description="Description 2",
            price=200.0,
            currency="CNY",
            category="market_data",
            tags=["tag1"],
            quality_score=0.8,
            rating=3.5,
            reviews_count=5,
            published=True
        )
        manager.marketplace_items["item1"] = item1
        manager.marketplace_items["item2"] = item2
        
        # 测试按类别过滤
        items = manager.get_marketplace_items(category="financial_data")
        assert len(items) == 1
        assert items[0]["item_id"] == "item1"
        
        # 测试按价格范围过滤
        items = manager.get_marketplace_items(min_price=150.0, max_price=250.0)
        assert len(items) == 1
        assert items[0]["item_id"] == "item2"
        
        # 测试按评分过滤
        items = manager.get_marketplace_items(min_rating=4.0)
        assert len(items) == 1
        assert items[0]["item_id"] == "item1"
        
        # 测试按标签过滤
        items = manager.get_marketplace_items(tags=["tag2"])
        assert len(items) == 1
        assert items[0]["item_id"] == "item1"
        
        # 测试限制数量
        items = manager.get_marketplace_items(limit=1)
        assert len(items) == 1
        
        manager._stop_monitoring = True


def test_get_marketplace_items_exception(mock_integration_manager):
    """测试获取市场商品（异常处理，覆盖 734-737 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        
        # 模拟异常
        with patch.object(manager, 'marketplace_items', side_effect=Exception("Error")):
            items = manager.get_marketplace_items()
            assert items == []
        
        manager._stop_monitoring = True


def test_check_contracts_status_expired(mock_integration_manager):
    """测试检查契约状态（过期契约，覆盖 759-781 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        
        # 创建过期契约
        contract = DataContract(
            contract_id="contract1",
            provider_asset_id="asset1",
            consumer_id="user1",
            expires_at=datetime.now() - timedelta(days=1),
            status="active"
        )
        manager.data_contracts["contract1"] = contract
        
        # 执行检查
        manager._check_contracts_status()
        
        # 验证契约状态已更新为过期
        assert manager.data_contracts["contract1"].status == "expired"
        
        manager._stop_monitoring = True


def test_check_contracts_status_exception(mock_integration_manager):
    """测试检查契约状态（异常处理，覆盖 779-781 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        
        # 模拟异常
        with patch('src.data.ecosystem.data_ecosystem_manager.datetime') as mock_dt:
            mock_dt.now.side_effect = Exception("Error")
            # 应该不抛出异常，只记录错误
            manager._check_contracts_status()
        
        manager._stop_monitoring = True


def test_update_data_quality_scores_decay(mock_integration_manager):
    """测试更新数据质量分数（质量衰减，覆盖 783-804 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        
        # 创建超过30天未更新的资产
        old_date = datetime.now() - timedelta(days=60)
        asset = DataAsset(
            asset_id="asset1",
            name="Test Asset",
            description="Test",
            data_type=DataSourceType.STOCK,
            owner="user1",
            quality_score=0.9,
            last_updated=old_date
        )
        manager.data_assets["asset1"] = asset
        
        original_score = asset.quality_score
        
        # 执行更新
        manager._update_data_quality_scores()
        
        # 验证质量分数已衰减
        assert asset.quality_score < original_score
        
        manager._stop_monitoring = True


def test_update_data_quality_scores_exception(mock_integration_manager):
    """测试更新数据质量分数（异常处理，覆盖 802-804 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        
        # 模拟异常
        with patch('src.data.ecosystem.data_ecosystem_manager.datetime') as mock_dt:
            mock_dt.now.side_effect = Exception("Error")
            # 应该不抛出异常，只记录错误
            manager._update_data_quality_scores()
        
        manager._stop_monitoring = True


def test_cleanup_expired_data(mock_integration_manager):
    """测试清理过期数据（覆盖 806-828 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        manager.config.lineage_retention_days = 30
        
        # 创建过期的血缘数据
        old_date = datetime.now() - timedelta(days=40)
        lineage = DataLineage(
            lineage_id="lineage1",
            source_asset_id="asset1",
            target_asset_id="asset2",
            transformation_type="transform",
            created_at=old_date
        )
        manager.data_lineage["asset2"] = [lineage]
        
        # 创建未过期的血缘数据
        recent_date = datetime.now() - timedelta(days=10)
        recent_lineage = DataLineage(
            lineage_id="lineage2",
            source_asset_id="asset2",
            target_asset_id="asset3",
            transformation_type="transform",
            created_at=recent_date
        )
        manager.data_lineage["asset3"] = [recent_lineage]
        
        # 执行清理
        manager._cleanup_expired_data()
        
        # 验证过期数据已清理
        assert len(manager.data_lineage["asset2"]) == 0
        # 验证未过期数据保留
        assert len(manager.data_lineage["asset3"]) == 1
        
        manager._stop_monitoring = True


def test_cleanup_expired_data_exception(mock_integration_manager):
    """测试清理过期数据（异常处理，覆盖 826-828 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        
        # 模拟异常
        with patch('src.data.ecosystem.data_ecosystem_manager.datetime') as mock_dt:
            mock_dt.now.side_effect = Exception("Error")
            # 应该不抛出异常，只记录错误
            manager._cleanup_expired_data()
        
        manager._stop_monitoring = True


def test_monitoring_worker_loop(mock_integration_manager, monkeypatch):
    """测试监控工作线程（循环执行，覆盖 739-757 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        manager._stop_monitoring = False
        
        # Mock time.sleep 以控制循环
        sleep_calls = []
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
            manager._stop_monitoring = True  # 第一次 sleep 后停止
        
        monkeypatch.setattr('time.sleep', mock_sleep)
        
        # 执行监控工作线程
        manager._monitoring_worker()
        
        # 验证 sleep 被调用
        assert len(sleep_calls) > 0
        
        manager._stop_monitoring = True


def test_monitoring_worker_exception(mock_integration_manager, monkeypatch):
    """测试监控工作线程（异常处理，覆盖 754-757 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        manager._stop_monitoring = False
        
        # Mock _check_contracts_status 抛出异常
        def failing_check():
            raise Exception("Check error")
        
        manager._check_contracts_status = failing_check
        
        # Mock time.sleep 以控制循环
        sleep_calls = []
        def mock_sleep(seconds):
            sleep_calls.append(seconds)
            manager._stop_monitoring = True  # 第一次 sleep 后停止
        
        monkeypatch.setattr('time.sleep', mock_sleep)
        
        # 执行监控工作线程，应该捕获异常并继续
        manager._monitoring_worker()
        
        # 验证 sleep 被调用（即使有异常）
        assert len(sleep_calls) > 0
        
        manager._stop_monitoring = True


def test_get_ecosystem_stats_comprehensive(mock_integration_manager):
    """测试获取生态系统统计信息（全面统计，覆盖 869-923 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        
        # 添加资产
        asset1 = DataAsset(
            asset_id="asset1",
            name="Asset 1",
            description="Test",
            data_type=DataSourceType.STOCK,
            owner="user1",
            quality_score=0.9
        )
        asset2 = DataAsset(
            asset_id="asset2",
            name="Asset 2",
            description="Test",
            data_type=DataSourceType.INDEX,
            owner="user2",
            quality_score=0.8
        )
        manager.data_assets["asset1"] = asset1
        manager.data_assets["asset2"] = asset2
        
        # 添加血缘
        lineage = DataLineage(
            lineage_id="lineage1",
            source_asset_id="asset1",
            target_asset_id="asset2",
            transformation_type="transform"
        )
        manager.data_lineage["asset2"] = [lineage]
        
        # 添加契约
        contract = DataContract(
            contract_id="contract1",
            provider_asset_id="asset1",
            consumer_id="user1",
            status="active"
        )
        manager.data_contracts["contract1"] = contract
        
        # 添加市场商品
        item = DataMarketItem(
            item_id="item1",
            asset_id="asset1",
            title="Item 1",
            description="Test",
            price=100.0,
            category="financial_data",
            published=True
        )
        manager.marketplace_items["item1"] = item
        
        # 获取统计信息
        stats = manager.get_ecosystem_stats()
        
        # 验证统计信息
        assert stats["assets"]["total"] == 2
        assert stats["assets"]["by_type"]["stock"] == 1
        assert stats["assets"]["by_type"]["index"] == 1
        assert stats["lineages"]["total"] == 1
        assert stats["contracts"]["total"] == 1
        assert stats["marketplace"]["total_items"] == 1
        
        manager._stop_monitoring = True


def test_get_ecosystem_stats_exception(mock_integration_manager):
    """测试获取生态系统统计信息（异常处理，覆盖 917-923 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()

        # Mock内部操作使其抛出异常
        with patch('src.data.ecosystem.data_ecosystem_manager.defaultdict') as mock_defaultdict:
            mock_defaultdict.side_effect = Exception("Test error")
            stats = manager.get_ecosystem_stats()
            assert "error" in stats
            assert "Test error" in stats["error"]

        manager._stop_monitoring = True


def test_shutdown_with_monitoring_thread(mock_integration_manager):
    """测试关闭生态系统管理器（带监控线程，覆盖 925-939 行）"""
    with patch('threading.Thread') as mock_thread:
        manager = DataEcosystemManager()
        
        # 创建模拟的监控线程
        mock_monitoring_thread = Mock()
        mock_monitoring_thread.is_alive.return_value = True
        mock_monitoring_thread.join = Mock()
        manager.monitoring_thread = mock_monitoring_thread
        
        # 执行关闭
        manager.shutdown()
        
        # 验证停止标志已设置
        assert manager._stop_monitoring is True
        # 验证 join 被调用
        mock_monitoring_thread.join.assert_called_once_with(timeout=5.0)
        
        manager._stop_monitoring = True


def test_shutdown_exception(mock_integration_manager):
    """测试关闭生态系统管理器（异常处理，覆盖 937-939 行）"""
    with patch('threading.Thread'):
        manager = DataEcosystemManager()
        
        # 模拟异常
        with patch.object(manager, 'get_ecosystem_stats', side_effect=Exception("Error")):
            # 应该不抛出异常，只记录错误
            manager.shutdown()
        
        manager._stop_monitoring = True




