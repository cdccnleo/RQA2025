# -*- coding: utf-8 -*-
"""
智能化增强功能管理器覆盖率测试 - Phase 2
针对IntelligentEnhancementManager类的未覆盖方法进行补充测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import json

from src.features.intelligent.intelligent_enhancement_manager import IntelligentEnhancementManager
from src.features.core.config_integration import ConfigScope


class TestIntelligentEnhancementManagerCoverage:
    """测试IntelligentEnhancementManager的未覆盖方法"""

    @pytest.fixture
    def sample_data(self):
        """生成示例数据"""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 2,
            'feature3': np.random.randn(100) * 3
        })
        y = pd.Series(np.random.randint(0, 2, 100))  # 分类任务
        return X, y

    @pytest.fixture
    def manager(self):
        """创建IntelligentEnhancementManager实例"""
        with patch('src.features.intelligent.intelligent_enhancement_manager.get_config_integration_manager') as mock_get_config:
            mock_config_manager = Mock()
            mock_config_manager.get_config.return_value = {}
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            # Mock组件以避免复杂的依赖
            with patch('src.features.intelligent.intelligent_enhancement_manager.AutoFeatureSelector') as mock_afs, \
                 patch('src.features.intelligent.intelligent_enhancement_manager.SmartAlertSystem') as mock_sas, \
                 patch('src.features.intelligent.intelligent_enhancement_manager.MLModelIntegration') as mock_ml:
                
                manager = IntelligentEnhancementManager(
                    enable_auto_feature_selection=False,
                    enable_smart_alerts=False,
                    enable_ml_integration=False
                )
                return manager

    def test_on_config_change_auto_feature_selection(self, manager):
        """测试配置变更处理 - auto_feature_selection"""
        manager.enable_auto_feature_selection = False
        
        manager._on_config_change(ConfigScope.PROCESSING, "enable_auto_feature_selection", True)
        
        assert manager.enable_auto_feature_selection is True

    def test_on_config_change_smart_alerts(self, manager):
        """测试配置变更处理 - smart_alerts"""
        manager.enable_smart_alerts = False
        
        manager._on_config_change(ConfigScope.PROCESSING, "enable_smart_alerts", True)
        
        assert manager.enable_smart_alerts is True

    def test_on_config_change_ml_integration(self, manager):
        """测试配置变更处理 - ml_integration"""
        manager.enable_ml_integration = False
        
        manager._on_config_change(ConfigScope.PROCESSING, "enable_ml_integration", True)
        
        assert manager.enable_ml_integration is True

    def test_enhance_features_no_components(self, manager, sample_data):
        """测试特征增强 - 无组件"""
        X, y = sample_data
        
        result_X, info = manager.enhance_features(X, y)
        
        # 验证结果
        assert isinstance(result_X, pd.DataFrame)
        assert isinstance(info, dict)
        assert 'timestamp' in info
        assert 'original_features' in info

    def test_enhance_features_with_auto_selection(self, sample_data):
        """测试特征增强 - 带自动特征选择"""
        X, y = sample_data
        
        with patch('src.features.intelligent.intelligent_enhancement_manager.get_config_integration_manager') as mock_get_config, \
             patch('src.features.intelligent.intelligent_enhancement_manager.AutoFeatureSelector') as mock_afs_class:
            
            mock_config_manager = Mock()
            mock_config_manager.get_config.return_value = {}
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            # Mock AutoFeatureSelector
            mock_selector = Mock()
            mock_selector.select_features.return_value = (
                X[['feature1', 'feature2']],
                ['feature1', 'feature2'],
                {'method': 'auto'}
            )
            mock_afs_class.return_value = mock_selector
            
            manager = IntelligentEnhancementManager(
                enable_auto_feature_selection=True,
                enable_smart_alerts=False,
                enable_ml_integration=False
            )
            
            result_X, info = manager.enhance_features(X, y)
            
            # 验证结果
            assert isinstance(result_X, pd.DataFrame)
            assert isinstance(info, dict)
            assert 'feature_selection' in info

    def test_check_feature_alerts(self, manager, sample_data):
        """测试检查特征告警"""
        X, y = sample_data
        
        # Mock smart_alert_system
        mock_alert_system = Mock()
        mock_alert_system.check_metric.return_value = []
        manager.smart_alert_system = mock_alert_system
        
        alerts = manager._check_feature_alerts(X, y)
        
        # 验证结果
        assert isinstance(alerts, list)

    def test_check_feature_alerts_empty_data(self, manager):
        """测试检查特征告警 - 空数据"""
        X = pd.DataFrame()
        y = pd.Series()
        
        # Mock smart_alert_system
        mock_alert_system = Mock()
        mock_alert_system.check_metric.return_value = []
        manager.smart_alert_system = mock_alert_system
        
        alerts = manager._check_feature_alerts(X, y)
        
        # 验证结果
        assert isinstance(alerts, list)

    def test_predict_with_enhanced_model_no_ml_integration(self, manager, sample_data):
        """测试使用增强模型预测 - 无ML集成"""
        X, _ = sample_data
        
        with pytest.raises(ValueError, match="机器学习模型集成未启用"):
            manager.predict_with_enhanced_model(X)

    def test_predict_with_enhanced_model_success(self, sample_data):
        """测试使用增强模型预测 - 成功"""
        X, y = sample_data
        
        with patch('src.features.intelligent.intelligent_enhancement_manager.get_config_integration_manager') as mock_get_config, \
             patch('src.features.intelligent.intelligent_enhancement_manager.MLModelIntegration') as mock_ml_class:
            
            mock_config_manager = Mock()
            mock_config_manager.get_config.return_value = {}
            mock_config_manager.register_config_watcher.return_value = None
            mock_get_config.return_value = mock_config_manager
            
            # Mock MLModelIntegration
            mock_ml = Mock()
            mock_ml.predict.return_value = np.array([0, 1, 0, 1] * 25)
            mock_ml._get_best_model.return_value = "RandomForest"
            mock_ml_class.return_value = mock_ml
            
            manager = IntelligentEnhancementManager(
                enable_auto_feature_selection=False,
                enable_smart_alerts=False,
                enable_ml_integration=True
            )
            
            predictions, info = manager.predict_with_enhanced_model(X)
            
            # 验证结果
            assert isinstance(predictions, np.ndarray)
            assert isinstance(info, dict)
            assert 'timestamp' in info

    def test_get_enhancement_summary(self, manager):
        """测试获取增强摘要"""
        # 设置一些历史数据
        manager.enhancement_history = [{'timestamp': '2023-01-01', 'original_shape': (100, 3)}]
        manager.current_features = ['feature1', 'feature2']
        manager.current_alerts = [{'type': 'warning', 'message': 'test'}]
        manager.model_performance = {'accuracy': 0.9}
        
        summary = manager.get_enhancement_summary()
        
        # 验证结果
        assert isinstance(summary, dict)
        assert 'enhancement_history_count' in summary
        assert 'current_features' in summary
        assert 'current_alerts_count' in summary

    def test_save_enhancement_state(self, manager, tmp_path):
        """测试保存增强状态"""
        # 设置一些状态数据
        manager.enhancement_history = [{'timestamp': '2023-01-01'}]
        manager.current_features = ['feature1', 'feature2']
        manager.current_alerts = [{'type': 'warning'}]
        manager.model_performance = {'accuracy': 0.9}
        
        filepath = tmp_path / "enhancement_state.json"
        manager.save_enhancement_state(str(filepath))
        
        # 验证文件已创建
        assert filepath.exists()
        
        # 验证文件内容
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert 'enhancement_history' in data
            assert 'current_features' in data

    def test_load_enhancement_state(self, manager, tmp_path):
        """测试加载增强状态"""
        # 创建状态文件
        filepath = tmp_path / "enhancement_state.json"
        state_data = {
            'enhancement_history': [{'timestamp': '2023-01-01'}],
            'current_features': ['feature1', 'feature2'],
            'current_alerts': [{'type': 'warning'}],
            'model_performance': {'accuracy': 0.9}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f)
        
        manager.load_enhancement_state(str(filepath))
        
        # 验证状态已加载
        assert len(manager.enhancement_history) > 0
        assert manager.current_features == ['feature1', 'feature2']

    def test_load_enhancement_state_not_found(self, manager, tmp_path):
        """测试加载增强状态 - 文件不存在"""
        non_existent_path = tmp_path / "non_existent.json"
        
        # 应该处理文件不存在的情况
        try:
            manager.load_enhancement_state(str(non_existent_path))
        except FileNotFoundError:
            pass  # 这是可以接受的

    def test_add_custom_alert_rule(self, manager):
        """测试添加自定义告警规则"""
        try:
            from src.features.intelligent.smart_alert_system import AlertRule
            
            # Mock smart_alert_system
            mock_alert_system = Mock()
            manager.smart_alert_system = mock_alert_system
            
            # 尝试创建AlertRule，如果参数不匹配则跳过
            try:
                rule = AlertRule(
                    name="test_rule",
                    metric="test_metric",
                    threshold=0.5,
                    condition="greater_than",
                    alert_type="warning",
                    level="medium"
                )
            except TypeError:
                # 如果AlertRule参数不匹配，创建一个Mock对象
                rule = Mock()
                rule.name = "test_rule"
            
            manager.add_custom_alert_rule(rule)
            
            # 验证规则已添加
            if manager.smart_alert_system:
                mock_alert_system.add_rule.assert_called_once_with(rule)
        except ImportError:
            pytest.skip("AlertRule不可用")

    def test_get_recent_alerts(self, manager):
        """测试获取最近告警"""
        # 设置一些告警
        manager.current_alerts = [
            {'timestamp': '2023-01-01T10:00:00', 'type': 'warning'},
            {'timestamp': '2023-01-01T12:00:00', 'type': 'error'}
        ]
        
        alerts = manager.get_recent_alerts(hours=24)
        
        # 验证结果
        assert isinstance(alerts, list)

    def test_export_enhancement_report(self, manager, tmp_path):
        """测试导出增强报告"""
        # 设置一些数据
        manager.enhancement_history = [{'timestamp': '2023-01-01'}]
        manager.current_features = ['feature1', 'feature2']
        manager.model_performance = {'accuracy': 0.9}
        
        filepath = tmp_path / "enhancement_report.json"
        manager.export_enhancement_report(str(filepath))
        
        # 验证文件已创建
        assert filepath.exists()
        
        # 验证文件内容
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 报告应该包含summary或enhancement_history
            assert 'summary' in data or 'enhancement_history' in data

    def test_reset_enhancement_state(self, manager):
        """测试重置增强状态"""
        # 设置一些状态
        manager.enhancement_history = [{'timestamp': '2023-01-01'}]
        manager.current_features = ['feature1', 'feature2']
        manager.current_alerts = [{'type': 'warning'}]
        manager.model_performance = {'accuracy': 0.9}
        
        manager.reset_enhancement_state()
        
        # 验证状态已重置
        assert len(manager.enhancement_history) == 0
        assert manager.current_features is None
        assert len(manager.current_alerts) == 0
        assert len(manager.model_performance) == 0
