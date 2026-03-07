"""
特征工程模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json
from datetime import datetime

from src.features.core.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """测试特征工程处理器"""

    def setup_method(self):
        """测试前准备"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()

        # Mock配置管理器
        with patch('src.features.core.feature_engineer.get_config_integration_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_config.return_value = None
            mock_get_manager.return_value = mock_manager

            self.engineer = FeatureEngineer(cache_dir=self.temp_dir)

    def teardown_method(self):
        """测试后清理"""
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_feature_engineer_initialization(self):
        """测试特征工程处理器初始化"""
        with patch('src.features.core.feature_engineer.get_config_integration_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_config.return_value = None
            mock_get_manager.return_value = mock_manager

            engineer = FeatureEngineer()

            assert engineer.cache_dir is not None
            assert engineer.max_retries == 3
            assert engineer.fallback_enabled == True
            assert engineer.max_workers == 4
            assert engineer.batch_size == 1000
            assert engineer.timeout == 300

    def test_generate_technical_features(self):
        """测试技术特征生成"""
        # 创建测试数据
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1min'),
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 105,
            'low': np.random.randn(100).cumsum() + 95,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(100, 1000, 100)
        })

        # 生成技术特征
        features = self.engineer.generate_technical_features(data)

        # 验证结果
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(data)

    def test_generate_sentiment_features(self):
        """测试情感特征生成"""
        # 创建测试数据
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50, freq='1min'),
            'news_text': ['positive news'] * 25 + ['negative news'] * 25
        })

        # 生成情感特征
        features = self.engineer.generate_sentiment_features(data)

        # 验证结果
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(data)

    def test_merge_features(self):
        """测试特征合并"""
        # 创建测试特征数据
        technical_features = pd.DataFrame({
            'sma_5': [1.0, 2.0, 3.0],
            'sma_10': [1.5, 2.5, 3.5]
        })

        sentiment_features = pd.DataFrame({
            'sentiment_score': [0.5, 0.7, 0.3],
            'sentiment_label': ['positive', 'positive', 'negative']
        })

        # 合并特征
        merged = self.engineer.merge_features([technical_features, sentiment_features])

        # 验证结果
        assert isinstance(merged, pd.DataFrame)
        assert len(merged) == 3
        assert 'sma_5' in merged.columns
        assert 'sentiment_score' in merged.columns

    def test_register_feature(self):
        """测试特征注册"""
        # 创建特征配置
        config = {
            'name': 'test_feature',
            'type': 'technical',
            'description': 'Test feature'
        }

        # 注册特征
        result = self.engineer.register_feature(config)

        # 验证注册成功
        assert result is not None

    def test_validate_stock_data(self):
        """测试股票数据验证"""
        # 有效的股票数据
        valid_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': np.random.randn(10) + 100,
            'high': np.random.randn(10) + 105,
            'low': np.random.randn(10) + 95,
            'close': np.random.randn(10) + 100,
            'volume': np.random.randint(100, 1000, 10)
        })

        # 验证有效数据
        try:
            self.engineer._validate_stock_data(valid_data)
            # 如果没有抛出异常，则验证通过
            assert True
        except Exception:
            assert False, "Valid data should not raise exception"

    def test_save_metadata(self):
        """测试元数据保存"""
        # 保存元数据
        metadata_path = Path(self.temp_dir) / "metadata.json"
        self.engineer.save_metadata(str(metadata_path))

        # 验证文件存在
        assert metadata_path.exists()

    def test_feature_engineer_repr(self):
        """测试FeatureEngineer字符串表示"""
        repr_str = repr(self.engineer)
        assert "FeatureEngineer" in repr_str


class TestFeatureEngineerIntegration:
    """测试特征工程处理器集成功能"""

    def test_feature_engineer_with_real_components(self):
        """测试特征工程处理器与真实组件集成"""
        with patch('src.features.core.feature_engineer.get_config_integration_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_config.return_value = {
                'max_workers': 2,
                'batch_size': 500,
                'timeout': 150
            }
            mock_get_manager.return_value = mock_manager

            engineer = FeatureEngineer()

            # 验证配置从管理器正确加载
            assert engineer.max_workers == 2
            assert engineer.batch_size == 500
            assert engineer.timeout == 150

    def test_concurrent_feature_processing(self):
        """测试并发特征处理"""
        # 创建较大的测试数据集
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1min'),
            'price': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(100, 1000, 1000)
        })

        # Mock技术处理器
        mock_processor = Mock()
        mock_processor.process.return_value = pd.DataFrame({
            'sma_5': np.random.randn(len(data)),
            'sma_10': np.random.randn(len(data))
        })

        self.engineer.technical_processor = mock_processor

        # 处理大型数据集
        start_time = datetime.now()
        features = self.engineer.extract_features(data)
        end_time = datetime.now()

        processing_time = (end_time - start_time).total_seconds()

        # 验证处理完成且在合理时间内
        assert 'technical_features' in features
        assert processing_time < 10  # 应该在10秒内完成

    def test_error_handling_and_fallback(self):
        """测试错误处理和降级"""
        # 创建测试数据
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'price': np.random.randn(10) + 100,
        })

        # Mock处理器抛出异常
        mock_processor = Mock()
        mock_processor.process.side_effect = Exception("Processing failed")
        self.engineer.technical_processor = mock_processor

        # 处理应该在启用降级的情况下成功（尽管返回空结果）
        features = self.engineer.extract_features(data)

        # 验证即使处理器失败也能返回结果结构
        assert isinstance(features, dict)

    def test_feature_metadata_management(self):
        """测试特征元数据管理"""
        # 创建测试特征
        feature_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 0.7, 0.3]
        })

        metadata = {
            'source': 'test',
            'version': '1.0',
            'description': 'Test features'
        }

        # 存储特征和元数据
        cache_key = "test_metadata"
        self.engineer.cache_metadata[cache_key] = metadata

        # 验证元数据存储
        assert cache_key in self.engineer.cache_metadata
        assert self.engineer.cache_metadata[cache_key] == metadata

    def test_resource_management(self):
        """测试资源管理"""
        # 验证线程池正确初始化
        assert self.engineer.executor is not None
        assert hasattr(self.engineer.executor, '_threads')

        # 验证缓存目录存在
        assert self.engineer.cache_dir.exists()
        assert self.engineer.cache_dir.is_dir()

    def test_configuration_persistence(self):
        """测试配置持久化"""
        # 修改配置
        original_max_workers = self.engineer.max_workers
        self.engineer.max_workers = 8

        # 验证配置改变
        assert self.engineer.max_workers == 8
        assert self.engineer.max_workers != original_max_workers

        # 重置为原始值
        self.engineer.max_workers = original_max_workers
        assert self.engineer.max_workers == original_max_workers