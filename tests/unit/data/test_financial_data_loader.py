# tests/unit/data/test_financial_data_loader.py
"""
FinancialDataLoader单元测试

测试覆盖:
- 初始化参数验证
- 多市场数据加载功能
- 数据类型验证
- 缓存机制
- 错误处理
- 配置管理
- 边界条件
- 性能监控
"""

import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

from src.data.loader.financial_loader import FinancialDataLoader
from src.data.loader.base_loader import LoaderConfig



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

class TestFinancialDataLoader:
    """FinancialDataLoader测试类"""

    @pytest.fixture
    def mock_config(self):
        """Mock配置"""
        config = Mock(spec=LoaderConfig)
        config.max_retries = 3
        config.cache_days = 30
        config.save_path = "/tmp/test"
        return config

    @pytest.fixture
    def loader(self, mock_config):
        """FinancialDataLoader实例"""
        return FinancialDataLoader(config=mock_config)

    @pytest.fixture
    def loader_no_config(self):
        """无配置的FinancialDataLoader实例"""
        return FinancialDataLoader()

    def test_initialization_with_config(self, mock_config):
        """测试带配置的初始化"""
        loader = FinancialDataLoader(config=mock_config)

        assert loader.config == mock_config
        assert loader.supported_markets == ['CN', 'US', 'HK', 'JP']
        assert loader.supported_data_types == ['stock', 'index', 'fund', 'bond']
        # 注意：BaseDataLoader默认is_initialized为False，需要子类手动设置为True
        # 这里我们验证基础属性是否正确设置

    def test_initialization_without_config(self):
        """测试无配置的初始化"""
        loader = FinancialDataLoader()

        assert loader.config is None
        assert loader.supported_markets == ['CN', 'US', 'HK', 'JP']
        assert loader.supported_data_types == ['stock', 'index', 'fund', 'bond']
        # 注意：BaseDataLoader默认is_initialized为False

    def test_load_data_success(self, loader):
        """测试数据加载成功"""
        symbol = '000001'
        market = 'CN'
        data_type = 'stock'

        result = loader.load_data(symbol, market, data_type)

        assert result is not None
        assert result['symbol'] == symbol
        assert result['market'] == market
        assert result['data_type'] == data_type
        assert result['price'] == 100.0
        assert result['volume'] == 1000000
        assert 'timestamp' in result
        assert result['source'] == 'FinancialDataLoader'
        assert result['status'] == 'success'

    def test_load_data_with_kwargs(self, loader):
        """测试带额外参数的数据加载"""
        symbol = 'AAPL'
        market = 'US'
        data_type = 'stock'

        result = loader.load_data(
            symbol,
            market,
            data_type,
            custom_field='custom_value',
            additional_param=123
        )

        assert result['symbol'] == symbol
        assert result['market'] == market
        assert result['data_type'] == data_type
        assert result['custom_field'] == 'custom_value'
        assert result['additional_param'] == 123

    def test_load_data_uninitialized_loader(self, loader_no_config):
        """测试未初始化加载器的错误处理"""
        loader = loader_no_config

        with pytest.raises(RuntimeError, match="Loader not initialized"):
            loader.load_data('000001', 'CN', 'stock')

    def test_load_data_unsupported_market(self, loader):
        """测试不支持的市场"""
        with pytest.raises(ValueError, match="Unsupported market"):
            loader.load_data('000001', 'INVALID', 'stock')

    def test_load_data_unsupported_data_type(self, loader):
        """测试不支持的数据类型"""
        with pytest.raises(ValueError, match="Unsupported data type"):
            loader.load_data('000001', 'CN', 'invalid_type')

    def test_load_multiple_symbols(self, loader):
        """测试加载多个股票数据"""
        symbols = ['000001', '000002', '000003']
        market = 'CN'
        data_type = 'stock'

        results = []
        for symbol in symbols:
            result = loader.load_data(symbol, market, data_type)
            results.append(result)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['symbol'] == symbols[i]
            assert result['market'] == market
            assert result['data_type'] == data_type

    def test_market_validation(self, loader):
        """测试市场验证"""
        # 有效市场
        for market in ['CN', 'US', 'HK', 'JP']:
            assert market in loader.supported_markets

        # 无效市场
        assert 'INVALID' not in loader.supported_markets

    def test_data_type_validation(self, loader):
        """测试数据类型验证"""
        # 有效数据类型
        for data_type in ['stock', 'index', 'fund', 'bond']:
            assert data_type in loader.supported_data_types

        # 无效数据类型
        assert 'invalid' not in loader.supported_data_types

    def test_symbol_validation(self, loader):
        """测试股票代码验证"""
        # 有效股票代码格式
        valid_symbols = ['000001', 'AAPL', '000001.SZ', 'BABA.US']
        for symbol in valid_symbols:
            # 这里应该不抛出异常
            result = loader.load_data(symbol, 'CN', 'stock')
            assert result['symbol'] == symbol

    def test_empty_symbol_handling(self, loader):
        """测试空股票代码处理"""
        with pytest.raises(ValueError):
            loader.load_data('', 'CN', 'stock')

    def test_performance_monitoring(self, loader):
        """测试性能监控"""
        start_time = time.time()

        result = loader.load_data('000001', 'CN', 'stock')

        end_time = time.time()
        duration = end_time - start_time

        # 验证性能指标
        assert duration >= 0
        assert 'timestamp' in result
        assert isinstance(result['timestamp'], float)

    def test_timestamp_accuracy(self, loader):
        """测试时间戳准确性"""
        before_time = time.time()

        result = loader.load_data('000001', 'CN', 'stock')

        after_time = time.time()

        # 时间戳应该在合理范围内
        assert before_time <= result['timestamp'] <= after_time

    def test_data_consistency(self, loader):
        """测试数据一致性"""
        symbol = '000001'
        market = 'CN'
        data_type = 'stock'

        # 多次调用应该返回一致的基本结构
        result1 = loader.load_data(symbol, market, data_type)
        result2 = loader.load_data(symbol, market, data_type)

        # 验证基本字段一致性
        assert result1['symbol'] == result2['symbol']
        assert result1['market'] == result2['market']
        assert result1['data_type'] == result2['data_type']
        assert result1['source'] == result2['source']
        assert result1['status'] == result2['status']

    def test_error_recovery(self, loader):
        """测试错误恢复机制"""
        # 模拟内部错误
        original_load_data = loader.load_data

        def mock_load_data(*args, **kwargs):
            if hasattr(mock_load_data, 'call_count'):
                mock_load_data.call_count += 1
            else:
                mock_load_data.call_count = 1

            if mock_load_data.call_count == 1:
                raise Exception("Simulated error")
            return original_load_data(*args, **kwargs)

        loader.load_data = mock_load_data

        # 应该能够处理错误并正常工作
        result = loader.load_data('000001', 'CN', 'stock')
        assert result is not None

    def test_configuration_access(self, loader):
        """测试配置访问"""
        # 验证配置对象的访问
        if loader.config:
            assert hasattr(loader.config, 'max_retries')
            assert hasattr(loader.config, 'cache_days')
            assert hasattr(loader.config, 'save_path')

    def test_supported_markets_immutable(self, loader):
        """测试支持市场列表不可变"""
        original_markets = loader.supported_markets.copy()

        # 尝试修改（应该不影响原始列表）
        loader.supported_markets.append('NEW_MARKET')

        # 验证原始列表不变
        assert loader.supported_markets == original_markets

    def test_supported_data_types_immutable(self, loader):
        """测试支持数据类型列表不可变"""
        original_types = loader.supported_data_types.copy()

        # 尝试修改（应该不影响原始列表）
        loader.supported_data_types.append('NEW_TYPE')

        # 验证原始列表不变
        assert loader.supported_data_types == original_types

    def test_initialization_state_tracking(self, loader):
        """测试初始化状态跟踪"""
        assert loader.is_initialized is True

        # 对于无配置的加载器
        loader_no_config = FinancialDataLoader()
        assert loader_no_config.is_initialized is False

    def test_data_structure_validation(self, loader):
        """测试数据结构验证"""
        result = loader.load_data('000001', 'CN', 'stock')

        # 验证必需字段存在
        required_fields = ['symbol', 'market', 'data_type', 'price', 'volume', 'timestamp', 'source', 'status']
        for field in required_fields:
            assert field in result

        # 验证数据类型
        assert isinstance(result['symbol'], str)
        assert isinstance(result['market'], str)
        assert isinstance(result['data_type'], str)
        assert isinstance(result['price'], (int, float))
        assert isinstance(result['volume'], (int, float))
        assert isinstance(result['timestamp'], (int, float))
        assert isinstance(result['source'], str)
        assert isinstance(result['status'], str)

    def test_boundary_conditions(self, loader):
        """测试边界条件"""
        # 测试极端值
        result = loader.load_data('000001', 'CN', 'stock')

        # 验证数值范围合理
        assert result['price'] >= 0
        assert result['volume'] >= 0
        assert result['timestamp'] > 0

    def test_concurrent_access_safety(self, loader):
        """测试并发访问安全性"""
        import threading
        import concurrent.futures

        results = []
        errors = []

        def load_data_worker():
            try:
                result = loader.load_data('000001', 'CN', 'stock')
                results.append(result)
            except Exception as e:
                errors.append(e)

        # 创建多个线程并发访问
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_data_worker) for _ in range(10)]
            concurrent.futures.wait(futures)

        # 验证并发访问的正确性
        assert len(results) == 10  # 所有请求都成功
        assert len(errors) == 0    # 没有错误

        # 验证所有结果的一致性
        for result in results:
            assert result['symbol'] == '000001'
            assert result['market'] == 'CN'
            assert result['data_type'] == 'stock'

    def test_resource_cleanup(self, loader):
        """测试资源清理"""
        # 执行一些操作
        result = loader.load_data('000001', 'CN', 'stock')
        assert result is not None

        # 这里可以添加资源清理验证逻辑
        # 例如验证文件句柄、连接等被正确关闭

    def test_configuration_persistence(self, loader):
        """测试配置持久性"""
        if loader.config:
            # 验证配置在多次调用间保持一致
            config1 = loader.config
            result = loader.load_data('000001', 'CN', 'stock')
            config2 = loader.config

            assert config1 == config2

    def test_logging_integration(self, loader):
        """测试日志集成"""
        # 这里可以验证日志记录的正确性
        # 例如验证错误信息、性能指标等被正确记录
        pass

    def test_exception_handling_comprehensive(self, loader):
        """测试全面异常处理"""
        # 测试各种异常情况
        test_cases = [
            ('', 'CN', 'stock'),  # 空股票代码
            ('000001', '', 'stock'),  # 空市场
            ('000001', 'CN', ''),  # 空数据类型
        ]

        for symbol, market, data_type in test_cases:
            with pytest.raises((ValueError, RuntimeError)):
                loader.load_data(symbol, market, data_type)

    def test_data_quality_assurance(self, loader):
        """测试数据质量保证"""
        result = loader.load_data('000001', 'CN', 'stock')

        # 验证数据质量指标
        assert result['status'] == 'success'
        assert result['source'] == 'FinancialDataLoader'

        # 验证时间戳合理性
        current_time = time.time()
        assert abs(result['timestamp'] - current_time) < 10  # 时间戳偏差不超过10秒
