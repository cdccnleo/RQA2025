#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场数据获取优化 v2.0 综合测试脚本

测试范围：
- 国际市场数据源适配器
- 另类数据适配器框架
- Level2行情适配器
- 数据压缩引擎
- 智能预处理流水线
- 智能缓存预热器
- 多因子策略
- 统计套利策略
- 策略组合优化器
- 自动化特征工程
- XGBoost/LightGBM模型训练器

作者: AI Assistant
创建日期: 2026-02-21
"""

import asyncio
import unittest
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np


class TestInternationalDataAdapters(unittest.TestCase):
    """测试国际数据源适配器"""
    
    def setUp(self):
        """测试准备"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """测试清理"""
        self.loop.close()
    
    def test_base_adapter_import(self):
        """测试基类导入"""
        try:
            from src.data.adapters.international import (
                InternationalDataAdapter,
                MarketType,
                DataFrequency
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")
    
    def test_yahoo_finance_adapter_import(self):
        """测试Yahoo Finance适配器导入"""
        try:
            from src.data.adapters.international import (
                YahooFinanceAdapter,
                get_yahoo_finance_adapter
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")
    
    def test_alpha_vantage_adapter_import(self):
        """测试Alpha Vantage适配器导入"""
        try:
            from src.data.adapters.international import (
                AlphaVantageAdapter,
                get_alpha_vantage_adapter
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")
    
    def test_market_type_enum(self):
        """测试市场类型枚举"""
        from src.data.adapters.international import MarketType
        
        self.assertEqual(MarketType.US_STOCK.value, "us_stock")
        self.assertEqual(MarketType.HK_STOCK.value, "hk_stock")
        self.assertEqual(MarketType.CRYPTO.value, "crypto")


class TestAlternativeDataAdapters(unittest.TestCase):
    """测试另类数据适配器"""
    
    def test_base_adapter_import(self):
        """测试基类导入"""
        try:
            from src.data.adapters.alternative import (
                AlternativeDataAdapter,
                AlternativeDataType,
                DataFusionEngine
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")
    
    def test_alternative_data_type_enum(self):
        """测试另类数据类型枚举"""
        from src.data.adapters.alternative import AlternativeDataType
        
        self.assertEqual(AlternativeDataType.SOCIAL_MEDIA_SENTIMENT.value, "social_media_sentiment")
        self.assertEqual(AlternativeDataType.NEWS_SENTIMENT.value, "news_sentiment")


class TestLevel2MarketData(unittest.TestCase):
    """测试Level2行情数据"""
    
    def test_adapter_import(self):
        """测试适配器导入"""
        try:
            from src.data.adapters.professional import (
                Level2MarketDataAdapter,
                Level2DataType,
                OrderBook,
                TickTrade
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")
    
    def test_level2_data_type_enum(self):
        """测试Level2数据类型枚举"""
        from src.data.adapters.professional import Level2DataType
        
        self.assertEqual(Level2DataType.ORDER_BOOK.value, "order_book")
        self.assertEqual(Level2DataType.TICK_TRADE.value, "tick_trade")


class TestCompressionEngine(unittest.TestCase):
    """测试数据压缩引擎"""
    
    def test_engine_import(self):
        """测试引擎导入"""
        try:
            from src.data.compression import (
                AdvancedCompressionEngine,
                CompressionAlgorithm,
                StorageFormat
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")
    
    def test_compression_algorithms(self):
        """测试压缩算法枚举"""
        from src.data.compression import CompressionAlgorithm
        
        algorithms = [
            CompressionAlgorithm.LZ4,
            CompressionAlgorithm.SNAPPY,
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.GZIP
        ]
        self.assertEqual(len(algorithms), 4)
    
    def test_storage_formats(self):
        """测试存储格式枚举"""
        from src.data.compression import StorageFormat
        
        formats = [
            StorageFormat.PARQUET,
            StorageFormat.FEATHER,
            StorageFormat.HDF5
        ]
        self.assertEqual(len(formats), 3)


class TestPreprocessingPipeline(unittest.TestCase):
    """测试智能预处理流水线"""
    
    def test_pipeline_import(self):
        """测试流水线导入"""
        try:
            from src.data.processing import (
                IntelligentPreprocessingPipeline,
                OutlierMethod,
                ImputationMethod,
                ScalingMethod
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")
    
    def test_outlier_methods(self):
        """测试异常值检测方法"""
        from src.data.processing import OutlierMethod
        
        methods = [
            OutlierMethod.ZSCORE,
            OutlierMethod.IQR,
            OutlierMethod.MAD
        ]
        self.assertEqual(len(methods), 3)
    
    def test_imputation_methods(self):
        """测试缺失值填充方法"""
        from src.data.processing import ImputationMethod
        
        methods = [
            ImputationMethod.MEAN,
            ImputationMethod.MEDIAN,
            ImputationMethod.ITERATIVE,
            ImputationMethod.KNN
        ]
        self.assertEqual(len(methods), 4)


class TestIntelligentCacheWarmer(unittest.TestCase):
    """测试智能缓存预热器"""
    
    def test_warmer_import(self):
        """测试预热器导入"""
        try:
            from src.data.cache import (
                IntelligentCacheWarmer,
                WarmupStrategy
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")
    
    def test_warmup_strategies(self):
        """测试预热策略"""
        from src.data.cache import WarmupStrategy
        
        strategies = [
            WarmupStrategy.TIME_BASED,
            WarmupStrategy.FREQUENCY_BASED,
            WarmupStrategy.PREDICTION_BASED,
            WarmupStrategy.HYBRID
        ]
        self.assertEqual(len(strategies), 4)


class TestMultiFactorStrategy(unittest.TestCase):
    """测试多因子策略"""
    
    def test_strategy_import(self):
        """测试策略导入"""
        try:
            from src.trading.strategy.advanced import (
                MultiFactorStrategy,
                Factor,
                FactorType,
                FactorDirection
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")
    
    def test_factor_types(self):
        """测试因子类型"""
        from src.trading.strategy.advanced import FactorType
        
        types = [
            FactorType.VALUE,
            FactorType.GROWTH,
            FactorType.MOMENTUM,
            FactorType.QUALITY,
            FactorType.VOLATILITY
        ]
        self.assertEqual(len(types), 5)


class TestStatisticalArbitrage(unittest.TestCase):
    """测试统计套利策略"""
    
    def test_strategy_import(self):
        """测试策略导入"""
        try:
            from src.trading.strategy.advanced import (
                StatisticalArbitrageStrategy,
                ArbitrageType,
                PairSignal
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")


class TestPortfolioOptimizer(unittest.TestCase):
    """测试策略组合优化器"""
    
    def test_optimizer_import(self):
        """测试优化器导入"""
        try:
            from src.trading.portfolio import (
                StrategyPortfolioOptimizer,
                OptimizationMethod,
                PortfolioAllocation
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")
    
    def test_optimization_methods(self):
        """测试优化方法"""
        from src.trading.portfolio import OptimizationMethod
        
        methods = [
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.MEAN_VARIANCE,
            OptimizationMethod.EQUAL_WEIGHT,
            OptimizationMethod.MINIMUM_VARIANCE,
            OptimizationMethod.MAXIMUM_SHARPE
        ]
        self.assertEqual(len(methods), 5)


class TestFeatureEngineering(unittest.TestCase):
    """测试自动化特征工程"""
    
    def test_engineer_import(self):
        """测试特征工程器导入"""
        try:
            from src.ml.feature_engineering import (
                AutomatedFeatureEngineer,
                FeatureSet
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")


class TestXGBoostLightGBM(unittest.TestCase):
    """测试XGBoost/LightGBM模型训练器"""
    
    def test_trainer_import(self):
        """测试训练器导入"""
        try:
            from src.ml.models import (
                XGBoostLightGBMTrainer,
                ModelMetrics,
                TrainedModel
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"导入失败: {e}")


def run_async_test(coro):
    """运行异步测试"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def create_test_summary():
    """创建测试摘要"""
    print("\n" + "="*70)
    print("市场数据获取优化 v2.0 测试摘要")
    print("="*70)
    
    test_modules = [
        ("国际数据源适配器", TestInternationalDataAdapters),
        ("另类数据适配器", TestAlternativeDataAdapters),
        ("Level2行情数据", TestLevel2MarketData),
        ("数据压缩引擎", TestCompressionEngine),
        ("智能预处理流水线", TestPreprocessingPipeline),
        ("智能缓存预热器", TestIntelligentCacheWarmer),
        ("多因子策略", TestMultiFactorStrategy),
        ("统计套利策略", TestStatisticalArbitrage),
        ("策略组合优化器", TestPortfolioOptimizer),
        ("自动化特征工程", TestFeatureEngineering),
        ("XGBoost/LightGBM", TestXGBoostLightGBM),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for name, test_class in test_modules:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        count = suite.countTestCases()
        total_tests += count
        print(f"  {name}: {count} 个测试")
    
    print(f"\n总计: {total_tests} 个测试")
    print("="*70)
    
    return total_tests


if __name__ == '__main__':
    # 创建测试摘要
    total = create_test_summary()
    
    # 运行所有测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestInternationalDataAdapters,
        TestAlternativeDataAdapters,
        TestLevel2MarketData,
        TestCompressionEngine,
        TestPreprocessingPipeline,
        TestIntelligentCacheWarmer,
        TestMultiFactorStrategy,
        TestStatisticalArbitrage,
        TestPortfolioOptimizer,
        TestFeatureEngineering,
        TestXGBoostLightGBM,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    print(f"运行测试: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过!")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败")
        sys.exit(1)
