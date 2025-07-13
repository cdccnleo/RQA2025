import pytest
import pandas as pd
from unittest.mock import patch
from .conftest import sample_features
from src.features.feature_manager import FeatureManager

class TestFeatureIntegration:
    """特征层全流程集成测试"""

    def test_full_feature_pipeline(self):
        """测试从原始数据到标准化特征的全流程"""
        with patch('src.features.manager.FeatureEngineer') as mock_eng:
            with patch('src.features.manager.FeatureSelector') as mock_sel:
                with patch('src.features.manager.FeatureStandardizer') as mock_std:
                    # 配置各环节mock
                    mock_eng.return_value.generate.return_value = sample_features()
                    mock_sel.return_value.select.return_value = sample_features()[['close', 'sentiment']]
                    mock_std.return_value.transform.return_value = sample_features()[['close', 'sentiment']] * 0.1

                    # 执行全流程
                    manager = FeatureManager()
                    result = manager.process(sample_features())

                    # 验证结果
                    assert set(result.columns) == {'close', 'sentiment'}
                    assert result['close'].mean() < 1  # 标准化后数值范围验证

    def test_metadata_through_pipeline(self):
        """验证元数据在整个流程中的传递"""
        manager = FeatureManager()
        initial_meta = manager.metadata.to_dict()

        with patch.object(manager, '_engineer'):
            with patch.object(manager, '_selector'):
                with patch.object(manager, '_standardizer'):
                    manager.process(sample_features())

                    # 验证元数据更新
                    assert manager.metadata.version > initial_meta['version']
                    assert 'process' in manager.metadata.operations
