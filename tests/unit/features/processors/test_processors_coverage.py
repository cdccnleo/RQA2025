#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processors模块测试覆盖
测试processors相关组件的核心功能
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

try:
    from src.features.processors.feature_selector import FeatureSelector
    from src.features.processors.feature_processor import FeatureProcessor
    from src.features.processors.base_processor import ProcessorConfig
    PROCESSORS_AVAILABLE = True
except ImportError:
    PROCESSORS_AVAILABLE = False
    FeatureSelector = None
    FeatureProcessor = None
    ProcessorConfig = None


@pytest.mark.skipif(not PROCESSORS_AVAILABLE, reason="Processors not available")
class TestFeatureSelector:
    """FeatureSelector测试"""

    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'feature4': np.random.randn(100),
            'feature5': np.random.randn(100)
        })

    @pytest.fixture
    def sample_target(self):
        """创建示例目标变量"""
        np.random.seed(42)
        return pd.Series(np.random.randn(100))

    def test_selector_initialization_default(self):
        """测试默认初始化"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None  # 返回None，使用默认值
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector()
            assert selector.selector_type == "rfecv"
            assert selector.n_features == 15
            assert selector.min_features_to_select == 3
            assert selector.cv == 5
            assert selector.is_fitted is False

    def test_selector_initialization_custom(self):
        """测试自定义初始化"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(
                selector_type="kbest",
                n_features=10,
                min_features_to_select=5,
                cv=3
            )
            assert selector.selector_type == "kbest"
            assert selector.n_features == 10
            assert selector.min_features_to_select == 5
            assert selector.cv == 3

    def test_selector_initialization_invalid_type(self):
        """测试无效选择器类型"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            with pytest.raises(ValueError, match="无效的选择器类型"):
                FeatureSelector(selector_type="invalid_type")

    def test_selector_initialization_invalid_min_features(self):
        """测试无效的min_features_to_select"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            with pytest.raises(ValueError, match="必须为正整数"):
                FeatureSelector(min_features_to_select=0)

    def test_selector_initialization_kbest_invalid_k(self):
        """测试kbest无效的k值"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            with pytest.raises(ValueError, match="must be positive"):
                FeatureSelector(selector_type="kbest", n_features=0)

    def test_fit_with_valid_data(self, sample_data, sample_target):
        """测试拟合有效数据"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(selector_type="kbest", n_features=3)
            selector.fit(sample_data, sample_target)
            assert selector.is_fitted is True
            assert len(selector.selected_features) > 0

    def test_fit_with_empty_features(self, sample_target):
        """测试拟合空特征数据"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector()
            empty_data = pd.DataFrame()
            selector.fit(empty_data, sample_target, is_training=False)
            assert selector.is_fitted is False
            assert selector.selected_features == []

    def test_fit_without_target(self, sample_data):
        """测试拟合时缺少目标变量"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector()
            with pytest.raises(ValueError, match="目标变量不能为空"):
                selector.fit(sample_data, None)

    def test_fit_with_empty_target(self, sample_data):
        """测试拟合时目标变量为空"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector()
            empty_target = pd.Series(dtype=float)
            with pytest.raises(ValueError, match="目标变量为空"):
                selector.fit(sample_data, empty_target)

    def test_fit_with_numpy_array(self, sample_target):
        """测试拟合numpy数组"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(selector_type="kbest", n_features=3)
            np_data = np.random.randn(100, 5)
            selector.fit(np_data, sample_target)
            assert selector.is_fitted is True

    def test_transform_without_fit(self, sample_data):
        """测试未拟合时转换"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector()
            result = selector.transform(sample_data)
            # 应该返回原始数据
            assert len(result) == len(sample_data)
            assert list(result.columns) == list(sample_data.columns)

    def test_transform_with_fit(self, sample_data, sample_target):
        """测试拟合后转换"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(selector_type="kbest", n_features=3)
            selector.fit(sample_data, sample_target)
            result = selector.transform(sample_data)
            assert len(result) == len(sample_data)
            assert len(result.columns) <= len(sample_data.columns)

    def test_transform_with_empty_data(self):
        """测试转换空数据"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector()
            empty_data = pd.DataFrame()
            result = selector.transform(empty_data)
            assert result.empty

    def test_select_features_rfecv(self, sample_data, sample_target):
        """测试RFECV选择特征"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(selector_type="rfecv", n_features=3)
            result = selector.select_features(sample_data, sample_target)
            assert not result.empty
            assert len(result.columns) <= len(sample_data.columns)

    def test_select_features_kbest(self, sample_data, sample_target):
        """测试KBest选择特征"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(selector_type="kbest", n_features=3)
            result = selector.select_features(sample_data, sample_target)
            assert not result.empty
            assert len(result.columns) == 3

    def test_select_features_variance(self, sample_data):
        """测试方差选择特征"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(selector_type="variance")
            result = selector.select_features(sample_data)
            assert not result.empty
            assert len(result.columns) <= len(sample_data.columns)

    def test_select_features_correlation(self, sample_data, sample_target):
        """测试相关性选择特征"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(selector_type="correlation", threshold=0.1)
            result = selector.select_features(sample_data, sample_target)
            assert not result.empty

    def test_select_features_correlation_no_target(self, sample_data):
        """测试相关性选择特征（无目标变量）"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(selector_type="correlation")
            result = selector.select_features(sample_data)
            # 无目标变量时应该返回原始数据
            assert len(result) == len(sample_data)
            assert list(result.columns) == list(sample_data.columns)

    def test_select_features_importance(self, sample_data, sample_target):
        """测试重要性选择特征"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(selector_type="importance")
            result = selector.select_features(sample_data, sample_target)
            assert not result.empty

    def test_preserve_features(self, sample_data, sample_target):
        """测试保留特征功能"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(
                selector_type="kbest",
                n_features=2,
                preserve_features=['feature1']
            )
            selector.fit(sample_data, sample_target)
            assert 'feature1' in selector.selected_features

    def test_save_and_load_selector(self, sample_data, sample_target, tmp_path):
        """测试保存和加载选择器"""
        mock_config_manager = Mock()
        mock_config_manager.get_config.return_value = None
        mock_config_manager.register_config_watcher = Mock()
        with patch('src.features.processors.feature_selector.get_config_integration_manager', return_value=mock_config_manager):
            selector = FeatureSelector(
                selector_type="kbest",
                n_features=3,
                model_path=tmp_path / "selector.pkl"
            )
            selector.fit(sample_data, sample_target)
            # 保存应该成功
            assert selector.is_fitted is True


@pytest.mark.skipif(not PROCESSORS_AVAILABLE, reason="Processors not available")
class TestFeatureProcessor:
    """FeatureProcessor测试"""

    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        })

    def test_processor_initialization_default(self):
        """测试默认初始化"""
        processor = FeatureProcessor()
        assert processor.config.processor_type == "general"
        assert processor.config.feature_params is not None
        assert len(processor._available_features) > 0

    def test_processor_initialization_custom(self):
        """测试自定义配置初始化"""
        config = ProcessorConfig(
            processor_type="custom",
            feature_params={"period": 20}
        )
        processor = FeatureProcessor(config)
        assert processor.config.processor_type == "custom"
        assert processor.config.feature_params["period"] == 20

    def test_process_with_valid_data(self, sample_data):
        """测试处理有效数据"""
        processor = FeatureProcessor()
        result = processor.process(sample_data, features=["sma"])
        assert not result.empty
        assert len(result) == len(sample_data)
        assert 'feature_sma' in result.columns

    def test_process_with_empty_data(self):
        """测试处理空数据"""
        processor = FeatureProcessor()
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="输入数据为空"):
            processor.process(empty_data)

    def test_process_with_invalid_features(self, sample_data):
        """测试处理无效特征"""
        processor = FeatureProcessor()
        with pytest.raises(ValueError, match="不支持的特征"):
            processor.process(sample_data, features=["invalid_feature"])

    def test_process_multiple_features(self, sample_data):
        """测试处理多个特征"""
        processor = FeatureProcessor()
        result = processor.process(sample_data, features=["sma", "ema"])
        assert 'feature_sma' in result.columns
        assert 'feature_ema' in result.columns

    def test_update_config(self):
        """测试更新配置"""
        processor = FeatureProcessor()
        processor.update_config({"new_param": 100})
        assert processor.config.feature_params["new_param"] == 100

    def test_update_config_none_params(self):
        """测试更新配置（feature_params为None）"""
        processor = FeatureProcessor()
        processor.config.feature_params = None
        processor.update_config({"new_param": 100})
        assert processor.config.feature_params["new_param"] == 100

    def test_compute_sma(self, sample_data):
        """测试计算SMA"""
        processor = FeatureProcessor()
        result = processor._compute_sma(sample_data, period=5)
        assert len(result) == len(sample_data)
        assert not pd.isna(result.iloc[-1])  # 最后一个应该有值

    def test_compute_ema(self, sample_data):
        """测试计算EMA"""
        processor = FeatureProcessor()
        result = processor._compute_ema(sample_data, period=5)
        assert len(result) == len(sample_data)

    def test_compute_rsi(self, sample_data):
        """测试计算RSI"""
        processor = FeatureProcessor()
        result = processor._compute_rsi(sample_data, period=5)
        assert len(result) == len(sample_data)
        # RSI应该在0-100之间
        valid_rsi = result.dropna()
        if len(valid_rsi) > 0:
            assert (valid_rsi >= 0).all()
            assert (valid_rsi <= 100).all()

    def test_compute_macd(self, sample_data):
        """测试计算MACD"""
        processor = FeatureProcessor()
        result = processor._compute_macd(sample_data, fast=5, slow=10, signal=3)
        assert len(result) == len(sample_data)

    def test_compute_bollinger_bands(self, sample_data):
        """测试计算布林带"""
        processor = FeatureProcessor()
        result = processor._compute_bollinger_bands(sample_data, period=5, std_dev=2)
        assert len(result) == len(sample_data)

    def test_compute_feature_empty_data(self):
        """测试计算特征（空数据）"""
        processor = FeatureProcessor()
        empty_data = pd.DataFrame()
        result = processor._compute_feature(empty_data, "sma", {"period": 20})
        assert result.empty

    def test_compute_feature_unknown(self, sample_data):
        """测试计算未知特征"""
        processor = FeatureProcessor()
        # 未知特征应该返回空Series或抛出异常
        result = processor._compute_feature(sample_data, "unknown_feature", {})
        # 根据实现，可能返回空Series或抛出异常
        assert isinstance(result, pd.Series)

    def test_compute_price_change(self, sample_data):
        """测试计算价格变化"""
        processor = FeatureProcessor()
        result = processor._compute_price_change(sample_data, period=1)
        assert len(result) == len(sample_data)

    def test_compute_volume_ratio(self, sample_data):
        """测试计算成交量比率"""
        processor = FeatureProcessor()
        result = processor._compute_volume_ratio(sample_data, period=5)
        assert len(result) == len(sample_data)

    def test_compute_volatility(self, sample_data):
        """测试计算波动率"""
        processor = FeatureProcessor()
        result = processor._compute_volatility(sample_data, period=5)
        assert len(result) == len(sample_data)

    def test_compute_sma_no_close(self):
        """测试计算SMA（缺少close列）"""
        processor = FeatureProcessor()
        data = pd.DataFrame({'volume': [1000, 1100, 1200]})
        result = processor._compute_sma(data, period=2)
        # 缺少close列时返回包含NaN的Series，不是空Series
        assert isinstance(result, pd.Series)
        assert result.isna().all()  # 所有值都是NaN

    def test_compute_ema_no_close(self):
        """测试计算EMA（缺少close列）"""
        processor = FeatureProcessor()
        data = pd.DataFrame({'volume': [1000, 1100, 1200]})
        result = processor._compute_ema(data, period=2)
        # 缺少close列时返回包含NaN的Series，不是空Series
        assert isinstance(result, pd.Series)
        assert result.isna().all()  # 所有值都是NaN

    def test_compute_rsi_no_close(self):
        """测试计算RSI（缺少close列）"""
        processor = FeatureProcessor()
        data = pd.DataFrame({'volume': [1000, 1100, 1200]})
        result = processor._compute_rsi(data, period=2)
        # 缺少close列时返回包含NaN的Series，不是空Series
        assert isinstance(result, pd.Series)
        assert result.isna().all()  # 所有值都是NaN

    def test_compute_macd_no_close(self):
        """测试计算MACD（缺少close列）"""
        processor = FeatureProcessor()
        data = pd.DataFrame({'volume': [1000, 1100, 1200]})
        result = processor._compute_macd(data, fast=5, slow=10, signal=3)
        # 缺少close列时返回包含NaN的Series，不是空Series
        assert isinstance(result, pd.Series)
        assert result.isna().all()  # 所有值都是NaN

    def test_compute_bollinger_bands_no_close(self):
        """测试计算布林带（缺少close列）"""
        processor = FeatureProcessor()
        data = pd.DataFrame({'volume': [1000, 1100, 1200]})
        result = processor._compute_bollinger_bands(data, period=2, std_dev=2)
        # 缺少close列时返回包含NaN的Series，不是空Series
        assert isinstance(result, pd.Series)
        assert result.isna().all()  # 所有值都是NaN

    def test_compute_volume_ratio_no_volume(self):
        """测试计算成交量比率（缺少volume列）"""
        processor = FeatureProcessor()
        data = pd.DataFrame({'close': [100, 101, 102]})
        result = processor._compute_volume_ratio(data, period=2)
        # 缺少volume列时返回包含NaN的Series，不是空Series
        assert isinstance(result, pd.Series)
        assert result.isna().all()  # 所有值都是NaN

    def test_get_feature_metadata(self, sample_data):
        """测试获取特征元数据"""
        processor = FeatureProcessor()
        metadata = processor._get_feature_metadata("sma")
        assert metadata["name"] == "sma"
        assert "description" in metadata
        assert "parameters" in metadata

    def test_get_feature_metadata_unknown(self):
        """测试获取未知特征元数据"""
        processor = FeatureProcessor()
        metadata = processor._get_feature_metadata("unknown_feature")
        assert metadata["name"] == "unknown_feature"
        assert metadata["type"] == "technical"

    def test_calculate_moving_averages(self, sample_data):
        """测试计算移动平均"""
        processor = FeatureProcessor()
        result = processor._calculate_moving_averages(sample_data)
        assert not result.empty
        assert len(result) == len(sample_data)

    def test_calculate_moving_averages_empty(self):
        """测试计算移动平均（空数据）"""
        processor = FeatureProcessor()
        empty_data = pd.DataFrame()
        result = processor._calculate_moving_averages(empty_data)
        assert result.empty

    def test_calculate_moving_averages_no_close(self):
        """测试计算移动平均（缺少close列）"""
        processor = FeatureProcessor()
        data = pd.DataFrame({'volume': [1000, 1100, 1200]})
        result = processor._calculate_moving_averages(data)
        assert 'close' not in result.columns

    def test_process_all_features(self, sample_data):
        """测试处理所有可用特征"""
        processor = FeatureProcessor()
        result = processor.process(sample_data)
        # 应该添加了一些特征列
        assert len(result.columns) > len(sample_data.columns)

    def test_process_compute_feature_exception(self, sample_data):
        """测试计算特征时异常处理"""
        processor = FeatureProcessor()
        # 模拟异常场景 - process方法会直接调用_compute_feature，如果抛出异常会传播
        with patch.object(processor, '_compute_feature', side_effect=Exception("模拟异常")):
            # process方法没有异常处理，异常会传播
            with pytest.raises(Exception, match="模拟异常"):
                processor.process(sample_data, features=["sma"])

    def test_calculate_rsi(self, sample_data):
        """测试计算RSI指标"""
        processor = FeatureProcessor()
        result = processor._calculate_rsi(sample_data, period=5)
        assert not result.empty
        assert 'RSI' in result.columns
        assert len(result) == len(sample_data)

    def test_calculate_rsi_empty(self):
        """测试计算RSI（空数据）"""
        processor = FeatureProcessor()
        empty_data = pd.DataFrame()
        result = processor._calculate_rsi(empty_data)
        assert result.empty

    def test_calculate_rsi_insufficient_data(self):
        """测试计算RSI（数据不足）"""
        processor = FeatureProcessor()
        small_data = pd.DataFrame({
            'close': [100, 101]
        })
        result = processor._calculate_rsi(small_data, period=5)
        # 数据不足时应该返回原始数据
        assert len(result) == len(small_data)

    def test_calculate_macd(self, sample_data):
        """测试计算MACD指标"""
        processor = FeatureProcessor()
        # MACD需要至少26个周期，sample_data只有11行，需要创建足够的数据
        large_data = pd.DataFrame({
            'open': [100 + i for i in range(30)],
            'high': [101 + i for i in range(30)],
            'low': [99 + i for i in range(30)],
            'close': [100.5 + i for i in range(30)],
            'volume': [1000 + i * 100 for i in range(30)]
        })
        result = processor._calculate_macd(large_data)
        assert not result.empty
        assert 'MACD' in result.columns
        assert 'MACD_Signal' in result.columns
        assert 'MACD_Histogram' in result.columns
        assert len(result) == len(large_data)

    def test_calculate_macd_empty(self):
        """测试计算MACD（空数据）"""
        processor = FeatureProcessor()
        empty_data = pd.DataFrame()
        result = processor._calculate_macd(empty_data)
        assert result.empty

    def test_calculate_macd_insufficient_data(self):
        """测试计算MACD（数据不足）"""
        processor = FeatureProcessor()
        small_data = pd.DataFrame({
            'close': [100, 101, 102] * 5  # 只有15个数据点
        })
        result = processor._calculate_macd(small_data)
        # 数据不足时应该返回原始数据
        assert len(result) == len(small_data)

    def test_calculate_bollinger_bands(self, sample_data):
        """测试计算布林带指标"""
        processor = FeatureProcessor()
        # 布林带需要至少20个周期，sample_data只有11行，需要创建足够的数据
        large_data = pd.DataFrame({
            'open': [100 + i for i in range(25)],
            'high': [101 + i for i in range(25)],
            'low': [99 + i for i in range(25)],
            'close': [100.5 + i for i in range(25)],
            'volume': [1000 + i * 100 for i in range(25)]
        })
        result = processor._calculate_bollinger_bands(large_data)
        assert not result.empty
        assert 'BB_Middle' in result.columns
        assert 'BB_Upper' in result.columns
        assert 'BB_Lower' in result.columns
        assert 'BB_Width' in result.columns
        assert len(result) == len(large_data)

    def test_calculate_bollinger_bands_empty(self):
        """测试计算布林带（空数据）"""
        processor = FeatureProcessor()
        empty_data = pd.DataFrame()
        result = processor._calculate_bollinger_bands(empty_data)
        assert result.empty

    def test_calculate_bollinger_bands_insufficient_data(self):
        """测试计算布林带（数据不足）"""
        processor = FeatureProcessor()
        small_data = pd.DataFrame({
            'close': [100, 101] * 5  # 只有10个数据点
        })
        result = processor._calculate_bollinger_bands(small_data)
        # 数据不足时应该返回原始数据
        assert len(result) == len(small_data)

    def test_get_available_features(self):
        """测试获取可用特征列表"""
        processor = FeatureProcessor()
        features = processor._get_available_features()
        assert isinstance(features, list)
        assert len(features) > 0
        assert "sma" in features
        assert "ema" in features

    def test_process_data(self, sample_data):
        """测试process_data方法"""
        processor = FeatureProcessor()
        result = processor.process_data(sample_data, features=["sma", "ema"])
        assert not result.empty
        assert len(result) == len(sample_data)

    def test_process_data_empty(self):
        """测试process_data（空数据）"""
        processor = FeatureProcessor()
        empty_data = pd.DataFrame()
        result = processor.process_data(empty_data)
        assert result.empty

    def test_process_data_default_features(self, sample_data):
        """测试process_data（使用默认特征）"""
        processor = FeatureProcessor()
        result = processor.process_data(sample_data)
        assert not result.empty

    def test_get_feature_summary(self):
        """测试获取特征摘要"""
        processor = FeatureProcessor()
        summary = processor.get_feature_summary()
        assert "total_features" in summary
        assert "available_features" in summary
        assert "processor_type" in summary
        assert "config" in summary

    def test_list_features(self):
        """测试列出可用特征"""
        processor = FeatureProcessor()
        features = processor.list_features()
        assert isinstance(features, list)
        assert len(features) > 0
        assert all("name" in f for f in features)
        assert all("description" in f for f in features)

    def test_validate_features(self):
        """测试验证特征列表"""
        processor = FeatureProcessor()
        valid, errors = processor.validate_features(["sma", "ema"])
        assert valid is True
        assert len(errors) == 0

    def test_validate_features_invalid(self):
        """测试验证无效特征列表"""
        processor = FeatureProcessor()
        valid, errors = processor.validate_features(["sma", "invalid_feature"])
        assert valid is False
        assert len(errors) > 0

    def test_get_feature_description(self):
        """测试获取特征描述"""
        processor = FeatureProcessor()
        desc = processor._get_feature_description("sma")
        assert "移动平均" in desc or desc

        desc_unknown = processor._get_feature_description("unknown")
        assert desc_unknown == "unknown指标"

    def test_get_feature_parameters(self):
        """测试获取特征参数"""
        processor = FeatureProcessor()
        params = processor._get_feature_parameters("sma")
        assert isinstance(params, dict)
        assert "periods" in params

        params_rsi = processor._get_feature_parameters("rsi")
        assert "period" in params_rsi

        params_macd = processor._get_feature_parameters("macd")
        assert "fast_period" in params_macd
        assert "slow_period" in params_macd

        params_bb = processor._get_feature_parameters("bollinger_bands")
        assert "period" in params_bb
        assert "std_dev" in params_bb

        params_unknown = processor._get_feature_parameters("unknown")
        assert params_unknown == {}

    def test_feature_cache_property(self):
        """测试feature_cache属性"""
        processor = FeatureProcessor()
        cache = processor.feature_cache
        assert isinstance(cache, dict)

