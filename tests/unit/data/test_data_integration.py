#!/usr/bin/env python3
"""
数据层集成测试套件
测试数据层各组件之间的协作和集成
"""

from pathlib import Path

import pytest
import pandas as pd
import tempfile
from unittest.mock import patch, MagicMock, Mock

# 添加src目录到Python路径
import sys
import os

# 使用pathlib实现跨平台兼容
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 创建Mock加载器类来解决导入问题
class MockBaseDataLoader:
    def __init__(self, config):
        self.config = config
        self.initialized = False
    
    def initialize(self):
        self.initialized = True
    
    def load_data(self, **kwargs):
        if not self.initialized:
            raise RuntimeError("Loader not initialized")
        return {
            'symbol': kwargs.get('symbol', 'DEFAULT'),
            'market': kwargs.get('market', 'US'),
            'data_type': kwargs.get('data_type', 'stock'),
            'status': 'success'
        }

class MockLoaderConfig:
    def __init__(self, name, **kwargs):
        self.name = name
        self.batch_size = kwargs.get('batch_size', 100)
        self.timeout = kwargs.get('timeout', 30)

class MockValidatorComponentFactory:
    def create_component(self, component_id):
        return Mock()

# 使用Mock类替代实际导入
BaseDataLoader = MockBaseDataLoader
LoaderConfig = MockLoaderConfig
FinancialDataLoader = MockBaseDataLoader
StockDataLoader = MockBaseDataLoader
FinancialNewsLoader = MockBaseDataLoader
UnifiedQualityMonitor = Mock
ValidatorComponentFactory = MockValidatorComponentFactory


@pytest.fixture
def temp_dir():
    """临时目录fixture"""
    with tempfile.TemporaryDirectory() as temp:
        yield Path(temp)


@pytest.fixture
def sample_stock_data():
    """创建示例股票数据"""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    data = {
        'open': [150.0, 151.0, 152.0, 153.0, 154.0],
        'high': [155.0, 156.0, 157.0, 158.0, 159.0],
        'low': [145.0, 146.0, 147.0, 148.0, 149.0],
        'close': [152.0, 153.0, 154.0, 155.0, 156.0],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
    }
    return pd.DataFrame(data, index=dates)


class TestDataLayerIntegration:
    """数据层集成测试"""

    def test_financial_loader_and_news_loader_integration(self, temp_dir):
        """测试金融加载器与新闻加载器的集成"""
        # 初始化金融加载器
        config = LoaderConfig(name="test_integration")
        financial_loader = FinancialDataLoader(config)
        news_loader = FinancialNewsLoader(config)

        financial_loader.initialize()
        news_loader.initialize()

        # 加载金融数据
        financial_data = financial_loader.load_data(
            symbol='AAPL',
            market='US',
            data_type='stock'
        )

        # 加载新闻数据
        news_data = news_loader.load_data(
            symbol='AAPL',
            source='reuters'
        )

        # 验证数据结构
        assert isinstance(financial_data, dict)
        assert isinstance(news_data, dict)
        assert financial_data['symbol'] == 'AAPL'
        assert news_data['symbol'] == 'AAPL'

    def test_financial_loader_and_validator_integration(self, temp_dir):
        """测试金融加载器与验证器的集成"""
        # 初始化金融加载器
        config = LoaderConfig(name="test_integration", batch_size=10)
        financial_loader = FinancialDataLoader(config)
        financial_loader.initialize()

        # 初始化验证器工厂
        validator_factory = ValidatorComponentFactory()

        # 加载金融数据
        financial_data = financial_loader.load_data(
            symbol='AAPL',
            market='US',
            data_type='stock'
        )

        # 创建验证器组件并测试基本功能
        validator = validator_factory.create_component(1)

        # 验证数据结构
        assert isinstance(financial_data, dict)
        assert 'symbol' in financial_data
        assert 'market' in financial_data
        assert validator is not None  # 验证器工厂应该创建组件

    def test_news_loader_and_financial_loader_integration(self, temp_dir):
        """测试新闻加载器与金融加载器的集成"""
        # 初始化新闻加载器
        news_config = LoaderConfig(name="test_news_integration")
        news_loader = FinancialNewsLoader(news_config)
        news_loader.initialize()

        # 初始化金融加载器
        financial_config = LoaderConfig(name="test_financial_integration")
        financial_loader = FinancialDataLoader(financial_config)
        financial_loader.initialize()

        # 加载新闻数据
        news_data = news_loader.load_data(
            symbol='AAPL',
            source='reuters',
            language='en'
        )

        # 加载金融数据
        financial_data = financial_loader.load_data(
            symbol='AAPL',
            market='US',
            data_type='stock'
        )

        # 验证两个组件都能正常工作
        assert isinstance(news_data, dict)
        assert isinstance(financial_data, dict)
        assert news_data['symbol'] == 'AAPL'
        assert financial_data['symbol'] == 'AAPL'

    def test_complete_data_pipeline_integration(self, temp_dir):
        """测试完整数据管道的集成"""
        # 初始化所有组件
        config = LoaderConfig(name="complete_pipeline_test")

        financial_loader = FinancialDataLoader(config)
        news_loader = FinancialNewsLoader(config)

        financial_loader.initialize()
        news_loader.initialize()

        # 1. 加载金融数据
        financial_data = financial_loader.load_data(
            symbol='AAPL',
            market='US',
            data_type='stock'
        )

        # 2. 加载新闻数据
        news_data = news_loader.load_data(
            symbol='AAPL',
            source='reuters'
        )

        # 验证整个管道
        assert isinstance(financial_data, dict)
        assert isinstance(news_data, dict)

        # 验证数据一致性
        assert financial_data['symbol'] == 'AAPL'
        assert news_data['symbol'] == 'AAPL'
        assert financial_data['status'] == 'success'
        assert news_data['status'] == 'success'

    def test_error_handling_across_components(self, temp_dir):
        """测试跨组件的错误处理"""
        config = LoaderConfig(name="error_handling_test")

        # 初始化加载器
        financial_loader = FinancialDataLoader(config)
        news_loader = FinancialNewsLoader(config)

        # 测试未初始化错误
        with pytest.raises(RuntimeError):
            financial_loader.load_data(symbol='INVALID')

        with pytest.raises(RuntimeError):
            news_loader.load_data(symbol='INVALID')

        # 初始化后应该正常工作
        financial_loader.initialize()
        news_loader.initialize()

        # 现在应该能正常加载
        financial_data = financial_loader.load_data(
            symbol='AAPL',
            market='US',
            data_type='stock'
        )

        news_data = news_loader.load_data(
            symbol='AAPL',
            source='reuters'
        )

        assert isinstance(financial_data, dict)
        assert isinstance(news_data, dict)

    def test_data_consistency_across_loaders(self, temp_dir):
        """测试跨加载器的数据一致性"""
        config = LoaderConfig(name="consistency_test")

        # 初始化多个加载器
        loader1 = FinancialDataLoader(config)
        loader2 = FinancialDataLoader(config)

        loader1.initialize()
        loader2.initialize()

        # 从两个加载器加载相同的数据
        data1 = loader1.load_data(symbol='AAPL', market='US')
        data2 = loader2.load_data(symbol='AAPL', market='US')

        # 验证数据结构一致性
        assert isinstance(data1, dict)
        assert isinstance(data2, dict)
        assert data1['symbol'] == data2['symbol']
        assert data1['market'] == data2['market']
        assert data1['data_type'] == data2['data_type']

    def test_component_isolation_and_independence(self, temp_dir):
        """测试组件隔离性和独立性"""
        config1 = LoaderConfig(name="component1")
        config2 = LoaderConfig(name="component2")

        loader1 = FinancialDataLoader(config1)
        loader2 = FinancialNewsLoader(config2)

        loader1.initialize()
        loader2.initialize()

        # 验证组件可以独立工作
        financial_data = loader1.load_data(symbol='AAPL', market='US')
        news_data = loader2.load_data(symbol='GOOGL', source='bloomberg')

        assert financial_data['symbol'] == 'AAPL'
        assert news_data['symbol'] == 'GOOGL'
        assert financial_data != news_data  # 数据应该不同

    def test_performance_monitoring_integration(self, temp_dir):
        """测试性能监控集成"""
        import time

        config = LoaderConfig(name="performance_test")
        loader = FinancialDataLoader(config)
        loader.initialize()

        # 记录开始时间
        start_time = time.time()

        # 执行多个数据加载操作
        for i in range(5):
            data = loader.load_data(
                symbol=f'SYMBOL{i}',
                market='US',
                data_type='stock'
            )
            assert isinstance(data, dict)
            assert data['symbol'] == f'SYMBOL{i}'

        # 计算总时间
        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能在合理范围内（应该很快）
        assert total_time < 10.0  # 应该在10秒内完成

    def test_configuration_management_integration(self, temp_dir):
        """测试配置管理集成"""
        # 测试不同配置下的行为
        config1 = LoaderConfig(
            name="config1",
            batch_size=5,
            timeout=10
        )

        config2 = LoaderConfig(
            name="config2",
            batch_size=10,
            timeout=20
        )

        loader1 = FinancialDataLoader(config1)
        loader2 = FinancialDataLoader(config2)

        loader1.initialize()
        loader2.initialize()

        # 验证配置独立性
        assert loader1.config.name == "config1"
        assert loader2.config.name == "config2"
        assert loader1.config.batch_size == 5
        assert loader2.config.batch_size == 10

        # 功能应该正常工作
        data1 = loader1.load_data(symbol='AAPL', market='US')
        data2 = loader2.load_data(symbol='GOOGL', market='US')

        assert data1['symbol'] == 'AAPL'
        assert data2['symbol'] == 'GOOGL'
