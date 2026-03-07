#!/usr/bin/env python3
"""
调度器持久化模块测试

测试调度器持久化功能，验证采集时间的保存和加载
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.orchestration.business_process.scheduler_persistence import SchedulerPersistence
from src.infrastructure.config.core.unified_manager_enhanced import UnifiedConfigManager


class TestSchedulerPersistence:
    """调度器持久化测试类"""
    
    def test_load_empty_collection_times(self):
        """测试加载空的采集时间"""
        config_manager = UnifiedConfigManager()
        persistence = SchedulerPersistence(config_manager)
        
        result = persistence.load_last_collection_times()
        
        assert isinstance(result, dict)
        assert len(result) == 0
    
    def test_save_and_load_collection_times(self):
        """测试保存和加载采集时间"""
        config_manager = UnifiedConfigManager()
        persistence = SchedulerPersistence(config_manager)
        
        # 准备测试数据
        test_times = {
            "akshare_stock_a": 1706035200.0,
            "akshare_stock_hk": 1706035260.0
        }
        
        # 保存
        success = persistence.save_last_collection_times(test_times)
        assert success is True
        
        # 加载
        loaded_times = persistence.load_last_collection_times()
        assert len(loaded_times) == 2
        assert loaded_times["akshare_stock_a"] == 1706035200.0
        assert loaded_times["akshare_stock_hk"] == 1706035260.0
    
    def test_get_metadata(self):
        """测试获取元数据"""
        config_manager = UnifiedConfigManager()
        persistence = SchedulerPersistence(config_manager)
        
        test_times = {"test_source": time.time()}
        persistence.save_last_collection_times(test_times)
        
        metadata = persistence.get_metadata()
        assert isinstance(metadata, dict)
        assert "last_saved" in metadata
        assert "version" in metadata
        assert metadata["version"] == "1.0"
    
    def test_clear_all(self):
        """测试清除所有数据"""
        config_manager = UnifiedConfigManager()
        persistence = SchedulerPersistence(config_manager)
        
        # 先保存一些数据
        test_times = {"test_source": time.time()}
        persistence.save_last_collection_times(test_times)
        
        # 清除
        success = persistence.clear_all()
        assert success is True
        
        # 验证已清除
        loaded_times = persistence.load_last_collection_times()
        assert len(loaded_times) == 0
    
    def test_load_invalid_timestamp(self):
        """测试加载无效时间戳"""
        config_manager = UnifiedConfigManager()
        persistence = SchedulerPersistence(config_manager)
        
        # 设置无效数据
        config_manager.set("data_collection_scheduler.last_collection_times", {
            "valid_source": 1706035200.0,
            "invalid_source": "not_a_number"
        })
        
        # 加载应该跳过无效数据
        loaded_times = persistence.load_last_collection_times()
        assert "valid_source" in loaded_times
        assert "invalid_source" not in loaded_times
