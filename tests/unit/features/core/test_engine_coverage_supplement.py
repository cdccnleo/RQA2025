#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/engine.py补充测试覆盖
针对未覆盖的代码分支编写测试
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from src.features.core.engine import FeatureEngine
from src.features.core.config import FeatureConfig
from src.features.core.feature_config import FeatureType


class TestFeatureEngineCoverageSupplement:
    """FeatureEngine补充测试"""

    @pytest.fixture
    def engine(self):
        """创建引擎实例"""
        return FeatureEngine()

    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        return pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

    @pytest.fixture
    def config(self):
        """创建配置"""
        config = FeatureConfig()
        config.enable_feature_selection = True
        config.enable_standardization = True
        config.enable_feature_saving = True
        config.feature_types = [FeatureType.TECHNICAL, FeatureType.SENTIMENT]
        return config

    def test_engineer_property_lazy_init(self, engine):
        """测试engineer属性的延迟初始化"""
        # 第一次访问应该初始化
        engineer1 = engine.engineer
        assert engineer1 is not None
        
        # 第二次访问应该返回同一个实例
        engineer2 = engine.engineer
        assert engineer1 is engineer2

    def test_selector_property_lazy_init(self, engine):
        """测试selector属性的延迟初始化"""
        selector1 = engine.selector
        assert selector1 is not None
        
        selector2 = engine.selector
        assert selector1 is selector2

    def test_standardizer_property_lazy_init(self, engine):
        """测试standardizer属性的延迟初始化"""
        standardizer1 = engine.standardizer
        assert standardizer1 is not None
        
        standardizer2 = engine.standardizer
        assert standardizer1 is standardizer2

    def test_saver_property_lazy_init(self, engine):
        """测试saver属性的延迟初始化"""
        # 确保stats已初始化
        assert hasattr(engine, 'stats')
        assert 'processed_features' in engine.stats
        
        saver1 = engine.saver
        assert saver1 is not None
        
        saver2 = engine.saver
        assert saver1 is saver2

    def test_register_default_processors_exception(self, engine, monkeypatch):
        """测试_register_default_processors异常处理"""
        # Mock导入失败
        def mock_import_error(*args, **kwargs):
            raise ImportError("模块导入失败")
        
        monkeypatch.setattr('src.features.core.engine.TechnicalProcessor', None, raising=False)
        
        # 重新初始化引擎，应该捕获异常
        engine._register_default_processors()
        # 应该记录警告但不抛出异常

    def test_process_features_validation_failed(self, engine):
        """测试process_features数据验证失败"""
        invalid_data = pd.DataFrame()  # 空数据
        
        # validate_data返回False时会抛出ValueError
        with pytest.raises(ValueError, match="输入数据验证失败"):
            engine.process_features(invalid_data)

    def test_process_features_with_feature_selection(self, engine, sample_data, config):
        """测试process_features启用特征选择"""
        # Mock所有组件
        mock_selector = Mock()
        mock_selector.select_features = Mock(return_value=sample_data)
        
        mock_standardizer = Mock()
        mock_standardizer.standardize_features = Mock(return_value=sample_data)
        
        mock_saver = Mock()
        mock_saver.save_features = Mock()
        
        # 直接设置私有属性，避免property的延迟初始化
        engine._selector = mock_selector
        engine._standardizer = mock_standardizer
        engine._saver = mock_saver
        
        with patch.object(engine, '_engineer_features', return_value=sample_data), \
             patch.object(engine, '_process_features', return_value=sample_data):
            
            result = engine.process_features(sample_data, config)
            assert isinstance(result, pd.DataFrame)
            mock_selector.select_features.assert_called_once()

    def test_process_features_without_feature_selection(self, engine, sample_data, config):
        """测试process_features禁用特征选择"""
        config.enable_feature_selection = False
        
        # Mock其他组件
        engine._engineer_features = Mock(return_value=sample_data)
        engine._process_features = Mock(return_value=sample_data)
        engine.standardizer.standardize_features = Mock(return_value=sample_data)
        engine.saver.save_features = Mock()
        
        result = engine.process_features(sample_data, config)
        assert isinstance(result, pd.DataFrame)

    def test_process_features_without_standardization(self, engine, sample_data, config):
        """测试process_features禁用标准化"""
        config.enable_standardization = False
        
        # Mock其他组件
        engine._engineer_features = Mock(return_value=sample_data)
        engine._process_features = Mock(return_value=sample_data)
        engine.selector.select_features = Mock(return_value=sample_data)
        engine.saver.save_features = Mock()
        
        result = engine.process_features(sample_data, config)
        assert isinstance(result, pd.DataFrame)

    def test_process_features_without_feature_saving(self, engine, sample_data, config):
        """测试process_features禁用特征保存"""
        config.enable_feature_saving = False
        
        # Mock其他组件
        engine._engineer_features = Mock(return_value=sample_data)
        engine._process_features = Mock(return_value=sample_data)
        engine.selector.select_features = Mock(return_value=sample_data)
        engine.standardizer.standardize_features = Mock(return_value=sample_data)
        
        result = engine.process_features(sample_data, config)
        assert isinstance(result, pd.DataFrame)

    def test_process_features_exception(self, engine, sample_data):
        """测试process_features异常处理"""
        # Mock _engineer_features抛出异常
        engine._engineer_features = Mock(side_effect=Exception("特征工程失败"))
        
        with pytest.raises(Exception):
            engine.process_features(sample_data)
        
        # 验证错误统计增加
        assert engine.stats['errors'] > 0

    def test_engineer_features_with_technical_processor(self, engine, sample_data, config):
        """测试_engineer_features使用技术指标处理器"""
        # Mock技术指标处理器，需要继承BaseFeatureProcessor
        from src.features.processors.base_processor import BaseFeatureProcessor, ProcessorConfig
        
        class MockProcessor(BaseFeatureProcessor):
            def __init__(self):
                super().__init__(ProcessorConfig(
                    processor_type="mock",
                    feature_params={}
                ))
            
            def _compute_feature(self, *args, **kwargs):
                return pd.DataFrame({'sma': [100, 101, 102]})
            
            def _get_available_features(self):
                return ['sma']
            
            def _get_feature_metadata(self, feature_name):
                return {}
            
            def process(self, request):
                return pd.DataFrame({'sma': [100, 101, 102]})
        
        mock_processor = MockProcessor()
        # 直接设置到processors字典，避免register_processor的验证
        engine.processors["technical"] = mock_processor
        
        result = engine._engineer_features(sample_data, config)
        assert isinstance(result, pd.DataFrame)

    def test_engineer_features_without_technical_processor(self, engine, sample_data, config):
        """测试_engineer_features没有技术指标处理器"""
        # 移除技术指标处理器
        if "technical" in engine.processors:
            del engine.processors["technical"]
        
        config.feature_types = [FeatureType.SENTIMENT]
        result = engine._engineer_features(sample_data, config)
        assert isinstance(result, pd.DataFrame)

    def test_engineer_features_with_sentiment_processor(self, engine, sample_data, config):
        """测试_engineer_features使用情感分析处理器"""
        # 获取现有的情感分析处理器或创建一个Mock对象
        sentiment_processor = engine.get_processor("sentiment")
        if sentiment_processor:
            # 使用现有的处理器，Mock其process方法
            original_process = getattr(sentiment_processor, 'process', None)
            mock_result = pd.DataFrame({'sentiment': [0.5, 0.6, 0.7]})
            
            def mock_process(request):
                return mock_result
            
            sentiment_processor.process = mock_process
            try:
                result = engine._engineer_features(sample_data, config)
                assert isinstance(result, pd.DataFrame)
            finally:
                # 恢复原始方法
                if original_process:
                    sentiment_processor.process = original_process
        else:
            # 如果没有处理器，跳过测试
            pytest.skip("情感分析处理器不可用")

    def test_engineer_features_exception(self, engine, sample_data, config):
        """测试_engineer_features异常处理"""
        # 使用现有的技术指标处理器，Mock其process方法抛出异常
        technical_processor = engine.get_processor("technical")
        if technical_processor:
            original_process = getattr(technical_processor, 'process', None)
            
            def mock_process_fail(request):
                raise Exception("处理失败")
            
            technical_processor.process = mock_process_fail
            try:
                result = engine._engineer_features(sample_data, config)
                # 异常时应该返回原始数据
                assert isinstance(result, pd.DataFrame)
            finally:
                # 恢复原始方法
                if original_process:
                    technical_processor.process = original_process
        else:
            pytest.skip("技术指标处理器不可用")

    def test_process_features_with_general_processor(self, engine, sample_data, config):
        """测试_process_features使用通用处理器"""
        # 使用现有的通用处理器，Mock其process方法
        general_processor = engine.get_processor("general")
        if general_processor:
            original_process = getattr(general_processor, 'process', None)
            mock_result = pd.DataFrame({'processed': [1, 2, 3]})
            
            def mock_process(request):
                return mock_result
            
            general_processor.process = mock_process
            try:
                result = engine._process_features(sample_data, config)
                assert isinstance(result, pd.DataFrame)
            finally:
                # 恢复原始方法
                if original_process:
                    general_processor.process = original_process
        else:
            pytest.skip("通用处理器不可用")

    def test_process_features_without_general_processor(self, engine, sample_data, config):
        """测试_process_features没有通用处理器"""
        # 移除通用处理器
        if "general" in engine.processors:
            del engine.processors["general"]
        
        result = engine._process_features(sample_data, config)
        # 没有处理器时应该返回原始特征
        assert isinstance(result, pd.DataFrame)

    def test_process_features_exception_in_process(self, engine, sample_data, config):
        """测试_process_features异常处理"""
        # 使用现有的通用处理器，Mock其process方法抛出异常
        general_processor = engine.get_processor("general")
        if general_processor:
            original_process = getattr(general_processor, 'process', None)
            
            def mock_process_fail(request):
                raise Exception("处理失败")
            
            general_processor.process = mock_process_fail
            try:
                result = engine._process_features(sample_data, config)
                # 异常时应该返回原始特征
                assert isinstance(result, pd.DataFrame)
            finally:
                # 恢复原始方法
                if original_process:
                    general_processor.process = original_process
        else:
            pytest.skip("通用处理器不可用")

    def test_process_with_processor_not_found(self, engine, sample_data):
        """测试process_with_processor处理器不存在"""
        with pytest.raises(ValueError, match="未找到处理器"):
            engine.process_with_processor(sample_data, "nonexistent_processor")

    def test_process_with_processor_success(self, engine, sample_data):
        """测试process_with_processor成功"""
        # 使用现有的通用处理器，Mock其process方法
        general_processor = engine.get_processor("general")
        if general_processor:
            original_process = getattr(general_processor, 'process', None)
            mock_result = pd.DataFrame({'result': [1, 2, 3]})
            
            def mock_process(request):
                return mock_result
            
            general_processor.process = mock_process
            try:
                result = engine.process_with_processor(sample_data, "general")
                assert isinstance(result, pd.DataFrame)
            finally:
                # 恢复原始方法
                if original_process:
                    general_processor.process = original_process
        else:
            pytest.skip("通用处理器不可用")

    def test_process_with_processor_exception(self, engine, sample_data):
        """测试process_with_processor异常处理"""
        # 使用现有的通用处理器，Mock其process方法抛出异常
        general_processor = engine.get_processor("general")
        if general_processor:
            original_process = getattr(general_processor, 'process', None)
            
            def mock_process_fail(request):
                raise Exception("处理失败")
            
            general_processor.process = mock_process_fail
            try:
                with pytest.raises(Exception, match="处理失败"):
                    engine.process_with_processor(sample_data, "general")
            finally:
                # 恢复原始方法
                if original_process:
                    general_processor.process = original_process
        else:
            pytest.skip("通用处理器不可用")

    def test_get_stats(self, engine):
        """测试get_stats"""
        # 设置一些统计
        engine.stats['processed_features'] = 10
        engine.stats['processing_time'] = 1.5
        engine.stats['errors'] = 2
        
        stats = engine.get_stats()
        assert stats['processed_features'] == 10
        assert stats['processing_time'] == 1.5
        assert stats['errors'] == 2
        # 应该是副本，不是引用
        stats['processed_features'] = 20
        assert engine.stats['processed_features'] == 10

    def test_reset_stats(self, engine):
        """测试reset_stats"""
        # 设置一些统计
        engine.stats['processed_features'] = 10
        engine.stats['errors'] = 2
        
        engine.reset_stats()
        
        assert engine.stats['processed_features'] == 0
        assert engine.stats['processing_time'] == 0.0
        assert engine.stats['errors'] == 0

    def test_validate_data_empty(self, engine):
        """测试validate_data空数据"""
        empty_data = pd.DataFrame()
        result = engine.validate_data(empty_data)
        assert result is False

    def test_validate_data_missing_columns(self, engine):
        """测试validate_data缺失必要列"""
        incomplete_data = pd.DataFrame({
            'close': [100, 101],
            'high': [105, 106]
            # 缺少low和volume
        })
        result = engine.validate_data(incomplete_data)
        assert result is False

    def test_validate_data_non_numeric(self, engine):
        """测试validate_data非数值类型"""
        non_numeric_data = pd.DataFrame({
            'close': ['a', 'b'],
            'high': [105, 106],
            'low': [95, 96],
            'volume': [1000, 1100]
        })
        result = engine.validate_data(non_numeric_data)
        assert result is False

    def test_get_supported_features(self, engine):
        """测试get_supported_features"""
        # 使用现有的处理器，检查是否有list_features或_get_available_features方法
        features = engine.get_supported_features()
        assert isinstance(features, list)
        
        # 验证至少返回一个列表（即使为空也是可以接受的）
        # 如果有处理器实现了list_features或_get_available_features方法，应该有特征返回

    def test_get_supported_features_no_list_features(self, engine):
        """测试get_supported_features处理器没有list_features方法"""
        # 使用现有的处理器，它们可能没有list_features方法
        # get_supported_features应该能够处理没有list_features方法的处理器
        features = engine.get_supported_features()
        assert isinstance(features, list)

    def test_get_engine_info(self, engine):
        """测试get_engine_info"""
        # 设置一些状态
        engine.stats['processed_features'] = 10
        
        info = engine.get_engine_info()
        assert 'version' in info
        assert 'processors' in info
        assert 'supported_features' in info
        assert 'stats' in info
        assert 'config' in info
        # 验证至少有一个处理器（默认注册的）
        assert len(info['processors']) > 0

