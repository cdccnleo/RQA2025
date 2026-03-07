# -*- coding: utf-8 -*-
"""
特征管理器覆盖率测试 - Phase 2
针对FeatureManager类的未覆盖方法进行补充测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from src.features.core.feature_manager import FeatureManager
from src.features.core.config_integration import ConfigScope


class TestFeatureManagerCoverage:
    """测试FeatureManager的未覆盖方法"""

    @pytest.fixture
    def sample_data(self):
        """生成示例数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        X = pd.DataFrame({
            'open': np.random.randn(100) * 10 + 100,
            'high': np.random.randn(100) * 10 + 105,
            'low': np.random.randn(100) * 10 + 95,
            'close': np.random.randn(100) * 10 + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        y = pd.Series(np.random.randint(0, 2, 100))  # 分类任务
        
        return X, y

    @pytest.fixture
    def manager(self):
        """创建FeatureManager实例"""
        with patch('src.features.core.feature_manager.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.get_config.return_value = {}
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            manager = FeatureManager()
            return manager

    def test_on_config_change_max_workers(self, manager):
        """测试配置变更处理 - max_workers"""
        old_value = manager.config.max_workers
        new_value = old_value + 1
        
        manager._on_config_change(ConfigScope.PROCESSING, "max_workers", old_value, new_value)
        
        assert manager.config.max_workers == new_value

    def test_on_config_change_batch_size(self, manager):
        """测试配置变更处理 - batch_size"""
        # 检查配置是否有batch_size属性
        if hasattr(manager.config, 'batch_size'):
            old_value = manager.config.batch_size
            new_value = old_value + 10
            
            manager._on_config_change(ConfigScope.PROCESSING, "batch_size", old_value, new_value)
            
            assert manager.config.batch_size == new_value
        else:
            pytest.skip("config没有batch_size属性")

    def test_on_config_change_timeout(self, manager):
        """测试配置变更处理 - timeout"""
        # 检查配置是否有timeout属性
        if hasattr(manager.config, 'timeout'):
            old_value = manager.config.timeout
            new_value = old_value + 10
            
            manager._on_config_change(ConfigScope.PROCESSING, "timeout", old_value, new_value)
            
            assert manager.config.timeout == new_value
        else:
            pytest.skip("config没有timeout属性")

    def test_on_config_change_feature_selection_method(self, manager):
        """测试配置变更处理 - feature_selection_method"""
        old_value = manager.config.feature_selection_method
        new_value = "mutual_info"
        
        manager._on_config_change(ConfigScope.PROCESSING, "feature_selection_method", old_value, new_value)
        
        assert manager.config.feature_selection_method == new_value

    def test_on_config_change_max_features(self, manager):
        """测试配置变更处理 - max_features"""
        old_value = manager.config.max_features
        new_value = old_value + 5
        
        manager._on_config_change(ConfigScope.PROCESSING, "max_features", old_value, new_value)
        
        assert manager.config.max_features == new_value

    def test_on_config_change_min_feature_importance(self, manager):
        """测试配置变更处理 - min_feature_importance"""
        # 检查配置是否有min_feature_importance属性
        if hasattr(manager.config, 'min_feature_importance'):
            old_value = manager.config.min_feature_importance
            new_value = old_value + 0.1
            
            manager._on_config_change(ConfigScope.PROCESSING, "min_feature_importance", old_value, new_value)
            
            assert manager.config.min_feature_importance == new_value
        else:
            pytest.skip("config没有min_feature_importance属性")

    def test_on_config_change_standardization_method(self, manager):
        """测试配置变更处理 - standardization_method"""
        old_value = manager.config.standardization_method
        new_value = "robust"
        
        manager._on_config_change(ConfigScope.PROCESSING, "standardization_method", old_value, new_value)
        
        assert manager.config.standardization_method == new_value

    def test_on_config_change_robust_scaling(self, manager):
        """测试配置变更处理 - robust_scaling"""
        # 检查配置是否有robust_scaling属性
        if hasattr(manager.config, 'robust_scaling'):
            old_value = manager.config.robust_scaling
            new_value = not old_value
            
            manager._on_config_change(ConfigScope.PROCESSING, "robust_scaling", old_value, new_value)
            
            assert manager.config.robust_scaling == new_value
        else:
            pytest.skip("config没有robust_scaling属性")

    def test_on_config_change_unknown_key(self, manager):
        """测试配置变更处理 - 未知键"""
        # 应该不抛出异常
        manager._on_config_change(ConfigScope.PROCESSING, "unknown_key", "old", "new")
        
        # 验证没有异常抛出

    def test_on_config_change_different_scope(self, manager):
        """测试配置变更处理 - 不同作用域"""
        # 应该不处理非PROCESSING作用域的配置变更
        # 获取初始配置值作为参考
        initial_max_workers = manager.config.max_workers
        
        # 尝试使用不同的作用域（如果存在）
        try:
            # 尝试使用其他作用域
            other_scope = ConfigScope.SYSTEM if hasattr(ConfigScope, 'SYSTEM') else ConfigScope.PROCESSING
            if other_scope != ConfigScope.PROCESSING:
                manager._on_config_change(other_scope, "max_workers", initial_max_workers, initial_max_workers + 1)
                # 验证配置没有改变（因为作用域不是PROCESSING）
                assert manager.config.max_workers == initial_max_workers
            else:
                # 如果没有其他作用域，测试未知键的情况
                manager._on_config_change(ConfigScope.PROCESSING, "unknown_key", "old", "new")
                # 验证配置没有改变
                assert manager.config.max_workers == initial_max_workers
        except AttributeError:
            # 如果没有SYSTEM作用域，测试未知键的情况
            manager._on_config_change(ConfigScope.PROCESSING, "unknown_key", "old", "new")
            # 验证配置没有改变
            assert manager.config.max_workers == initial_max_workers

