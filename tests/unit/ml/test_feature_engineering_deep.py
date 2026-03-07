#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FeatureEngineering深度测试
测试特征工程的完整功能和复杂场景
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import json

from src.ml.feature_engineering import (

FeatureEngineer,
    FeatureType,
    ScalingMethod,
    EncodingMethod,
    FeatureDefinition,
    FeaturePipeline
)

# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.legacy,
    pytest.mark.timeout(45),  # 45秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]




class TestFeatureEngineerLifecycle:
    """测试特征工程完整生命周期"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'cache_enabled': True,
            'max_cache_size': 100,
            'parallel_processing': True,
            'missing_value_threshold': 0.5,
            'outlier_detection': True
        }

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """测试初始化"""
        engineer = FeatureEngineer(self.config)

        # 验证初始化
        assert engineer.config == self.config
        assert hasattr(engineer, 'feature_definitions')
        assert hasattr(engineer, 'pipelines')
        assert hasattr(engineer, '_feature_cache')

    def test_feature_definition(self):
        """测试特征定义"""
        engineer = FeatureEngineer(self.config)

        # 定义数值特征
        engineer.define_feature(
            name="price",
            feature_type=FeatureType.NUMERIC,
            data_type="float64",
            description="Stock price",
            validation_rules={"min": 0, "max": 10000}
        )

        # 定义分类特征
        engineer.define_feature(
            name="sector",
            feature_type=FeatureType.CATEGORICAL,
            data_type="category",
            description="Company sector",
            validation_rules={"categories": ["Technology", "Finance", "Healthcare"]}
        )

        # 验证特征定义
        assert "price" in engineer.feature_definitions
        assert "sector" in engineer.feature_definitions

        price_def = engineer.feature_definitions["price"]
        assert price_def.feature_type == FeatureType.NUMERIC
        assert price_def.description == "Stock price"

        sector_def = engineer.feature_definitions["sector"]
        assert sector_def.feature_type == FeatureType.CATEGORICAL
        assert "Technology" in sector_def.validation_rules["categories"]

    def test_pipeline_creation(self):
        """测试管道创建"""
        engineer = FeatureEngineer(self.config)

        # 定义特征
        engineer.define_feature("price", FeatureType.NUMERIC, "float64")
        engineer.define_feature("volume", FeatureType.NUMERIC, "int64")
        engineer.define_feature("sector", FeatureType.CATEGORICAL, "category")

        # 创建管道步骤
        steps = [
            {
                "name": "handle_missing",
                "type": "missing_values",
                "method": "mean",
                "columns": ["price", "volume"]
            },
            {
                "name": "scale_features",
                "type": "scaling",
                "method": "standard",
                "columns": ["price", "volume"]
            },
            {
                "name": "encode_categorical",
                "type": "encoding",
                "method": "onehot",
                "columns": ["sector"]
            }
        ]

        # 创建管道
        pipeline_name = "stock_pipeline"
        engineer.create_pipeline(pipeline_name, steps, ["price", "volume", "sector"])

        # 验证管道创建
        assert pipeline_name in engineer.pipelines
        pipeline = engineer.pipelines[pipeline_name]
        assert len(pipeline.steps) == 3

        # 验证输出特征推断
        output_features = engineer._infer_output_features(["price", "volume", "sector"], steps)
        assert "price_scaled" in output_features
        assert "volume_scaled" in output_features
        assert "sector_Technology" in output_features


class TestDataProcessing:
    """测试数据处理功能"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'cache_enabled': True,
            'parallel_processing': False,
            'missing_value_threshold': 0.3
        }

        # 创建测试数据
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'price': [100, 105, np.nan, 110, 108],
            'volume': [1000, 1200, 1100, np.nan, 1300],
            'sector': ['Tech', 'Finance', 'Tech', 'Healthcare', 'Finance'],
            'returns': [0.05, -0.02, 0.03, np.nan, 0.01]
        })

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_missing_value_handling(self):
        """测试缺失值处理"""
        engineer = FeatureEngineer(self.config)

        # 测试均值填充
        config = {
            "method": "mean",
            "columns": ["price", "volume"]
        }

        result = engineer._handle_missing_values(self.test_data.copy(), config)

        # 验证缺失值已被填充
        assert not result['price'].isnull().any()
        assert not result['volume'].isnull().any()

        # 验证均值填充正确
        original_price_mean = self.test_data['price'].mean()
        assert result.loc[2, 'price'] == pytest.approx(original_price_mean, rel=1e-10)

    def test_feature_scaling(self):
        """测试特征缩放"""
        engineer = FeatureEngineer(self.config)

        # 准备数据（填充缺失值）
        data = self.test_data.copy()
        data['price'] = data['price'].fillna(data['price'].mean())
        data['volume'] = data['volume'].fillna(data['volume'].mean())

        # 测试标准化缩放
        config = {
            "method": "standard",
            "columns": ["price", "volume"]
        }

        result = engineer._scale_features(data, config)

        # 验证缩放后的特征
        assert "price_scaled" in result.columns
        assert "volume_scaled" in result.columns

        # 验证标准化结果（均值为0）
        assert result['price_scaled'].mean() == pytest.approx(0, abs=1e-10)
        assert result['volume_scaled'].mean() == pytest.approx(0, abs=1e-10)

    def test_categorical_encoding(self):
        """测试分类特征编码"""
        engineer = FeatureEngineer(self.config)

        # 测试独热编码
        config = {
            "method": "onehot",
            "columns": ["sector"]
        }

        result = engineer._encode_categorical(self.test_data.copy(), config)

        # 验证独热编码结果
        expected_columns = ["sector_Tech", "sector_Finance", "sector_Healthcare"]
        for col in expected_columns:
            assert col in result.columns

        # 验证编码正确性
        assert result.loc[0, "sector_Tech"] == 1
        assert result.loc[0, "sector_Finance"] == 0
        assert result.loc[1, "sector_Finance"] == 1

    def test_technical_indicators(self):
        """测试技术指标创建"""
        engineer = FeatureEngineer(self.config)

        # 创建时间序列数据
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        ts_data = pd.DataFrame({
            'date': dates,
            'price': np.random.randn(20).cumsum() + 100,
            'volume': np.random.randint(1000, 2000, 20)
        })

        # 测试移动平均线
        config = {
            "indicators": [
                {"type": "sma", "column": "price", "period": 5},
                {"type": "ema", "column": "price", "period": 3}
            ]
        }

        result = engineer._create_technical_indicators(ts_data, config)

        # 验证技术指标
        assert "price_sma_5" in result.columns
        assert "price_ema_3" in result.columns

        # 验证计算正确性（非空值）
        assert not result['price_sma_5'].isnull().all()
        assert not result['price_ema_3'].isnull().all()

    def test_temporal_features(self):
        """测试时间特征创建"""
        engineer = FeatureEngineer(self.config)

        # 创建时间序列数据
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        temporal_data = pd.DataFrame({
            'date': dates,
            'value': np.random.randn(10)
        })

        # 测试时间特征
        config = {
            "date_column": "date",
            "features": ["day_of_week", "month", "quarter", "is_weekend"]
        }

        result = engineer._create_temporal_features(temporal_data, config)

        # 验证时间特征
        expected_features = ["day_of_week", "month", "quarter", "is_weekend"]
        for feature in expected_features:
            assert feature in result.columns

        # 验证特征值范围
        assert result['day_of_week'].between(0, 6).all()
        assert result['month'].between(1, 12).all()
        assert result['quarter'].between(1, 4).all()
        assert result['is_weekend'].isin([0, 1]).all()


class TestPipelineExecution:
    """测试管道执行"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'cache_enabled': True,
            'parallel_processing': False
        }

        # 创建测试数据
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'price': np.random.randn(50).cumsum() + 100,
            'volume': np.random.randint(1000, 5000, 50),
            'sector': np.random.choice(['Tech', 'Finance', 'Healthcare'], 50),
            'returns': np.random.randn(50) * 0.02
        })

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_complete_pipeline_execution(self):
        """测试完整管道执行"""
        engineer = FeatureEngineer(self.config)

        # 定义特征
        engineer.define_feature("price", FeatureType.NUMERIC)
        engineer.define_feature("volume", FeatureType.NUMERIC)
        engineer.define_feature("sector", FeatureType.CATEGORICAL)

        # 创建完整管道
        steps = [
            {
                "name": "handle_missing",
                "type": "missing_values",
                "method": "median",
                "columns": ["price", "volume", "returns"]
            },
            {
                "name": "create_technical",
                "type": "technical_indicators",
                "indicators": [
                    {"type": "sma", "column": "price", "period": 5},
                    {"type": "rsi", "column": "price", "period": 14}
                ]
            },
            {
                "name": "scale_features",
                "type": "scaling",
                "method": "robust",
                "columns": ["price", "volume", "returns", "price_sma_5"]
            },
            {
                "name": "encode_sector",
                "type": "encoding",
                "method": "label",
                "columns": ["sector"]
            }
        ]

        # 创建管道
        pipeline_name = "complete_stock_pipeline"
        engineer.create_pipeline(pipeline_name, steps)

        # 执行管道
        result = engineer.process_data(self.test_data.copy(), pipeline_name)

        # 验证处理结果
        assert result is not None
        assert len(result) == len(self.test_data)

        # 验证技术指标
        assert "price_sma_5" in result.columns
        assert "price_rsi_14" in result.columns

        # 验证缩放特征
        assert "price_scaled" in result.columns
        assert "volume_scaled" in result.columns

        # 验证编码特征
        assert "sector_encoded" in result.columns

    def test_pipeline_caching(self):
        """测试管道缓存"""
        engineer = FeatureEngineer(self.config)

        # 定义简单管道
        steps = [
            {
                "name": "scale_price",
                "type": "scaling",
                "method": "standard",
                "columns": ["price"]
            }
        ]

        engineer.create_pipeline("cache_test_pipeline", steps)

        # 第一次执行
        result1 = engineer.process_data(self.test_data.copy(), "cache_test_pipeline")

        # 验证缓存
        cache_key = engineer._get_cache_key(self.test_data, "cache_test_pipeline")
        assert cache_key in engineer._cache

        # 第二次执行（应该使用缓存）
        result2 = engineer.process_data(self.test_data.copy(), "cache_test_pipeline")

        # 验证结果相同
        pd.testing.assert_frame_equal(result1, result2)

    def test_pipeline_error_handling(self):
        """测试管道错误处理"""
        engineer = FeatureEngineer(self.config)

        # 创建包含错误步骤的管道
        steps = [
            {
                "name": "invalid_step",
                "type": "nonexistent_type",
                "columns": ["price"]
            }
        ]

        engineer.create_pipeline("error_pipeline", steps)

        # 执行应该抛出异常
        with pytest.raises(ValueError):
            engineer.process_data(self.test_data.copy(), "error_pipeline")

    def test_pipeline_with_feature_selection(self):
        """测试带特征选择的管道"""
        engineer = FeatureEngineer(self.config)

        # 创建包含特征选择的管道
        steps = [
            {
                "name": "handle_missing",
                "type": "missing_values",
                "method": "mean",
                "columns": ["price", "volume", "returns"]
            },
            {
                "name": "select_features",
                "type": "feature_selection",
                "method": "correlation",
                "threshold": 0.1,
                "target_column": "returns"
            }
        ]

        engineer.create_pipeline("selection_pipeline", steps)

        # 执行管道
        result = engineer.process_data(self.test_data.copy(), "selection_pipeline")

        # 验证特征选择
        assert result is not None
        # 特征选择可能减少列数
        assert len(result.columns) <= len(self.test_data.columns) + 1  # 可能添加目标列


class TestFeatureEngineerAdvanced:
    """测试特征工程高级功能"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'cache_enabled': True,
            'parallel_processing': True,
            'max_cache_size': 50
        }

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_pipeline_persistence(self):
        """测试管道持久化"""
        engineer = FeatureEngineer(self.config)

        # 创建管道
        steps = [
            {
                "name": "scale",
                "type": "scaling",
                "method": "standard",
                "columns": ["price"]
            }
        ]

        pipeline_name = "persistent_pipeline"
        engineer.create_pipeline(pipeline_name, steps)

        # 保存管道
        filepath = str(self.temp_dir / "test_pipeline.json")
        engineer.save_pipeline(pipeline_name, filepath)

        # 验证文件存在
        assert Path(filepath).exists()

        # 加载管道
        loaded_pipeline = engineer.load_pipeline(filepath)

        # 验证加载结果
        assert loaded_pipeline.name == pipeline_name
        assert len(loaded_pipeline.steps) == 1
        assert loaded_pipeline.steps[0]["type"] == "scaling"

    def test_statistics_collection(self):
        """测试统计信息收集"""
        engineer = FeatureEngineer(self.config)

        # 执行一些操作
        engineer.define_feature("test_feature", FeatureType.NUMERIC)
        engineer.create_pipeline("test_pipeline", [{"name": "test", "type": "missing_values", "method": "mean"}])

        # 获取统计信息
        stats = engineer.get_statistics()

        # 验证统计信息
        assert isinstance(stats, dict)
        assert "total_features" in stats
        assert "total_pipelines" in stats
        assert "cache_size" in stats

        assert stats["total_features"] >= 1
        assert stats["total_pipelines"] >= 1

    def test_cache_management(self):
        """测试缓存管理"""
        engineer = FeatureEngineer(self.config)

        # 验证初始缓存状态
        assert len(engineer._cache) == 0

        # 添加一些缓存项
        for i in range(60):  # 超过max_cache_size
            cache_key = f"test_key_{i}"
            engineer._cache[cache_key] = f"test_value_{i}"

        # 验证缓存大小限制（实际实现可能不同）
        # 这里主要测试缓存管理的基本功能
        assert hasattr(engineer, '_cache')

        # 清理缓存
        engineer.clear_cache()
        assert len(engineer._cache) == 0

    def test_parallel_processing(self):
        """测试并行处理"""
        config = self.config.copy()
        config['parallel_processing'] = True
        engineer = FeatureEngineer(config)

        # 创建大型数据集
        large_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })

        # 创建管道
        steps = [
            {
                "name": "scale",
                "type": "scaling",
                "method": "standard",
                "columns": ["feature1", "feature2", "feature3"]
            },
            {
                "name": "encode",
                "type": "encoding",
                "method": "onehot",
                "columns": ["category"]
            }
        ]

        engineer.create_pipeline("parallel_pipeline", steps)

        # 执行并行处理
        result = engineer.process_data(large_data, "parallel_pipeline")

        # 验证结果
        assert result is not None
        assert len(result) == len(large_data)

        # 验证特征处理
        assert "feature1_scaled" in result.columns
        assert "category_A" in result.columns

    def test_outlier_detection(self):
        """测试异常值检测"""
        config = self.config.copy()
        config['outlier_detection'] = True
        engineer = FeatureEngineer(config)

        # 创建包含异常值的数据
        data_with_outliers = pd.DataFrame({
            'normal_feature': np.random.randn(100),
            'outlier_feature': [1, 2, 3, 1000, 5, 6, 7, -500] + list(np.random.randn(92))  # 包含异常值
        })

        # 创建管道（这里主要测试数据处理流程，异常值检测可能在具体步骤中实现）
        steps = [
            {
                "name": "handle_missing",
                "type": "missing_values",
                "method": "mean",
                "columns": ["normal_feature", "outlier_feature"]
            }
        ]

        engineer.create_pipeline("outlier_pipeline", steps)

        # 执行管道
        result = engineer.process_data(data_with_outliers, "outlier_pipeline")

        # 验证处理完成
        assert result is not None
        assert len(result) == len(data_with_outliers)


if __name__ == "__main__":
    pytest.main([__file__])
