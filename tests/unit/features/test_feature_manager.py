import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from src.features.feature_manager import FeatureManager
from src.features.processors import SentimentAnalyzer
from tests.features.conftest import sample_features, broken_features

class TestFeatureManager:
    """FeatureManager流程控制测试"""

    @patch('src.features.manager.FeatureEngineer')
    @patch('src.features.manager.FeatureSelector')
    @patch('src.features.manager.FeatureStandardizer')
    def test_full_pipeline(self, mock_std, mock_sel, mock_eng):
        """测试完整特征处理流程"""
        # 配置mock返回值
        mock_eng.return_value.generate.return_value = sample_features()
        mock_sel.return_value.select.return_value = sample_features()[['close', 'sentiment']]
        mock_std.return_value.transform.return_value = sample_features()

        # 执行流程
        manager = FeatureManager()
        result = manager.process(sample_features())

        # 验证流程步骤
        mock_eng.return_value.generate.assert_called_once()
        mock_sel.return_value.select.assert_called_once()
        mock_std.return_value.transform.assert_called_once()
        assert not result.empty

    @pytest.mark.parametrize("broken_part", [
        "engineer", "selector", "standardizer"
    ])
    def test_pipeline_failure(self, broken_part):
        """测试各环节异常处理"""
        manager = FeatureManager()

        with patch.object(manager, f'_{broken_part}',
                         side_effect=Exception("Mock error")):
            with pytest.raises(Exception, match="Mock error"):
                manager.process(sample_features())

    def test_nan_handling(self, broken_features):
        """测试空值数据处理"""
        manager = FeatureManager()
        with patch.object(manager, '_engineer') as mock_eng:
            mock_eng.generate.return_value = broken_features

            # 应触发空值警告但继续执行
            with pytest.warns(UserWarning, match="NaN values detected"):
                result = manager.process(broken_features)
                assert result.isna().sum().sum() == 0

    def test_metadata_versioning(self):
        """测试元数据版本更新"""
        manager = FeatureManager()
        initial_version = manager.metadata.version

        manager.process(sample_features())
        assert manager.metadata.version > initial_version
        assert "process" in manager.metadata.operations

    @patch.object(SentimentAnalyzer, 'analyze')
    def test_sentiment_analysis_models(self, mock_analyze):
        """测试多模型情感分析集成"""
        # 配置多模型返回结果
        mock_analyze.return_value = {
            'model1': 0.8,
            'model2': -0.2,
            'model3': 0.5
        }

        manager = FeatureManager()
        result = manager.process(sample_features())

        # 验证情感分析结果合并
        assert 'sentiment_model1' in result.columns
        assert 'sentiment_model2' in result.columns
        assert 'sentiment_model3' in result.columns
        assert mock_analyze.call_count == len(sample_features())

    @pytest.mark.parametrize("input_data,expected_cols", [
        (sample_features(), ['ma5', 'ma10', 'rsi14']),
        (sample_features()[['close']], ['ma5', 'ma10']),
        (sample_features()[['close', 'volume']], ['ma5', 'ma10', 'obv'])
    ])
    def test_technical_indicators(self, input_data, expected_cols):
        """参数化测试技术指标生成"""
        manager = FeatureManager()
        result = manager.process(input_data)
        
        for col in expected_cols:
            assert col in result.columns
            assert not result[col].isna().any()

    def test_feature_selection_strategies(self):
        """测试特征选择策略"""
        manager = FeatureManager()
        
        # 测试不同策略
        strategies = ['variance', 'correlation', 'mutual_info']
        for strategy in strategies:
            with patch.object(manager, '_selector') as mock_sel:
                mock_sel.strategy = strategy
                result = manager.process(sample_features())
                
                # 验证策略应用
                mock_sel.select.assert_called_once()
                assert len(result.columns) <= len(sample_features().columns)

    def test_edge_cases(self):
        """测试边界条件"""
        manager = FeatureManager()
        
        # 空DataFrame输入
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            manager.process(pd.DataFrame())
            
        # 单行数据
        single_row = sample_features().iloc[:1]
        result = manager.process(single_row)
        assert len(result) == 1
        
        # 全NaN列
        nan_data = sample_features().copy()
        nan_data['nan_col'] = np.nan
        with pytest.warns(UserWarning, match="Dropped all-NA column"):
            result = manager.process(nan_data)
            assert 'nan_col' not in result.columns
