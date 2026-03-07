#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复特征层导入问题脚本
"""

import os


def fix_import_issues():
    """修复特征层导入问题"""

    # 修复测试文件中的导入路径
    test_files_to_fix = [
        # 特征配置测试
        ('tests/unit/features/test_feature_config.py',
         'from src.features.config import FeatureConfig, FeatureConfigManager, HighFreqConfig, OrderBookConfig, FeatureType',
         'from src.features.types.enums import FeatureType'),

        # 特征引擎测试
        ('tests/unit/features/test_feature_engine.py',
         'from src.features.feature_engine import FeatureEngine',
         'from src.features.feature_engineer import FeatureEngineer'),

        # 特征管理器测试
        ('tests/unit/features/test_feature_manager.py',
         'from src.features.feature_manager import FeatureManager',
         'from src.features.processors.general_processor import FeatureProcessor'),

        ('tests/unit/features/test_feature_manager_offline.py',
         'from src.features.feature_manager import FeatureManager',
         'from src.features.processors.general_processor import FeatureProcessor'),

        # 特征选择器测试
        ('tests/unit/features/test_feature_selector.py',
         'from src.features.feature_selector import FeatureSelector',
         'from src.features.processors.feature_selector import FeatureSelector'),

        # 高频优化器测试
        ('tests/unit/features/test_high_freq_optimizer.py',
         'from src.features.high_freq_optimizer import HighFreqOptimizer, HighFreqConfig',
         'from src.features.high_freq_optimizer import HighFreqOptimizer'),

        # 集成测试
        ('tests/unit/features/test_integration.py',
         'from src.features.feature_manager import FeatureManager',
         'from src.features.processors.general_processor import FeatureProcessor'),

        # 信号生成器测试
        ('tests/unit/features/test_signal_generator.py',
         'from src.features.signal_generator import SignalGenerator, SignalConfig, ChinaSignalGenerator',
         'from src.features.signal_generator import SignalGenerator'),

        # 技术处理器测试
        ('tests/unit/features/test_technical_processor.py',
         'from src.features.technical.technical_processor import TechnicalProcessor',
         'from src.features.processors.technical.technical_processor import TechnicalProcessor'),

        # 情感分析器测试
        ('tests/unit/features/sentiment/test_sentiment_analyzer.py',
         'from src.features.feature_manager import FeatureManager',
         'from src.features.processors.general_processor import FeatureProcessor'),

        ('tests/unit/features/sentiment/test_sentiment_analyzer_full.py',
         'from src.features.sentiment.sentiment_analyzer import SentimentAnalyzer, SentimentConfig',
         'from src.features.sentiment.sentiment_analyzer import SentimentAnalyzer'),

        # 订单簿测试
        ('tests/unit/features/test_orderbook_config.py',
         'from src.features.config import OrderBookConfig',
         'from src.features.types.enums import FeatureType'),

        # 处理器测试
        ('tests/unit/features/processors/test_technical_processor.py',
         'from src.features.technical.technical_processor import TechnicalProcessor',
         'from src.features.processors.technical.technical_processor import TechnicalProcessor'),

        # 技术测试
        ('tests/unit/features/technical/test_processor.py',
         'from src.features.technical.processor import TechnicalProcessor',
         'from src.features.processors.technical.technical_processor import TechnicalProcessor'),

        # 优化器测试
        ('tests/unit/features/optimizer/test_optimizers.py',
         'from src.features.high_freq_optimizer import HighFreqOptimizer',
         'from src.features.high_freq_optimizer import HighFreqOptimizer'),

        # 订单簿分析器测试
        ('tests/unit/features/orderbook/test_analyzer.py',
         'from src.features.orderbook.analyzer import OrderbookAnalyzer',
         'from src.features.orderbook.order_book_analyzer import OrderBookAnalyzer'),

        ('tests/unit/features/orderbook/test_order_book_analyzer.py',
         'from src.features.orderbook.analyzer import OrderbookAnalyzer',
         'from src.features.orderbook.order_book_analyzer import OrderBookAnalyzer'),

        ('tests/unit/features/test_order_book_analyzer.py',
         'from src.features.orderbook.order_book_analyzer import OrderBookAnalyzer, OrderBookConfig',
         'from src.features.orderbook.order_book_analyzer import OrderBookAnalyzer'),

        ('tests/unit/features/test_level2_analyzer.py',
         'from src.features.orderbook.level2_analyzer import Level2Analyzer',
         'from src.features.orderbook.order_book_analyzer import OrderBookAnalyzer'),

        ('tests/unit/features/test_orderbook_metrics.py',
         'from src.features.orderbook.metrics import (',
         'from src.features.orderbook.order_book_analyzer import OrderBookAnalyzer'),
    ]

    for file_path, old_import, new_import in test_files_to_fix:
        if os.path.exists(file_path):
            print(f"修复 {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换导入语句
            content = content.replace(old_import, new_import)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

    # 修复特征配置文件中的导入问题
    feature_config_files = [
        'src/features/high_freq_optimizer.py',
        'src/features/orderbook/order_book_analyzer.py',
        'src/features/technical/technical_processor.py',
        'src/features/signal_generator.py'
    ]

    for file_path in feature_config_files:
        if os.path.exists(file_path):
            print(f"修复 {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换feature_config导入
            content = content.replace(
                'from .feature_config import FeatureConfig',
                '# from .feature_config import FeatureConfig  # 暂未实现'
            )
            content = content.replace(
                'from ..feature_config import FeatureConfig',
                '# from ..feature_config import FeatureConfig  # 暂未实现'
            )

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

    # 修复信号生成器中的导入
    signal_generator_file = 'src/features/signal_generator.py'
    if os.path.exists(signal_generator_file):
        print(f"修复 {signal_generator_file}")
        with open(signal_generator_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换feature_engine导入
        content = content.replace(
            'from ..features.feature_engine import FeatureEngine',
            'from .feature_engineer import FeatureEngineer as FeatureEngine'
        )

        with open(signal_generator_file, 'w', encoding='utf-8') as f:
            f.write(content)

    print("特征层导入问题修复完成")


if __name__ == "__main__":
    fix_import_issues()
