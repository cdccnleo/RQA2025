# -*- coding: utf-8 -*-
"""
中国股票验证器测试
测试股票数据验证逻辑
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
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock
from src.data.validation.china_stock_validator import ChinaStockValidator


class TestChinaStockValidator:
    """测试中国股票验证器"""

    def setup_method(self):
        """测试前准备"""
        self.validator = ChinaStockValidator()

    def test_validate_data_empty_dataframe(self):
        """测试空DataFrame验证"""
        empty_df = pd.DataFrame()

        result = self.validator.validate_data(empty_df)

        assert result['is_valid'] == False
        assert '数据为空或格式错误' in result['errors']
        assert result['warnings'] == []

    def test_validate_data_none_input(self):
        """测试None输入验证"""
        result = self.validator.validate_data(None)

        assert result['is_valid'] == False
        assert '数据为空或格式错误' in result['errors']

    def test_validate_data_invalid_type(self):
        """测试无效类型输入验证"""
        result = self.validator.validate_data("invalid_data")

        assert result['is_valid'] == False
        assert '数据为空或格式错误' in result['errors']

    def test_validate_data_valid_dataframe(self):
        """测试有效DataFrame验证"""
        valid_data = pd.DataFrame({
            'open': [10.0, 11.0, 12.0],
            'high': [11.0, 12.0, 13.0],
            'low': [9.0, 10.0, 11.0],
            'close': [10.5, 11.5, 12.5],
            'volume': [1000, 1100, 1200]
        })

        result = self.validator.validate_data(valid_data)

        assert result['is_valid'] == True
        assert result['errors'] == []
        assert result['warnings'] == []

    def test_validate_data_model_none_model(self):
        """测试None数据模型验证"""
        result = self.validator.validate_data_model(None)

        assert result['is_valid'] == False
        assert '数据模型为空' in result['errors']

    def test_validate_data_model_missing_data_attribute(self):
        """测试缺少data属性的数据模型"""
        mock_model = Mock()
        del mock_model.data  # 确保没有data属性

        result = self.validator.validate_data_model(mock_model)

        assert result['is_valid'] == False
        assert '数据模型缺少data属性' in result['errors']

    def test_validate_data_model_empty_data(self):
        """测试空数据的数据模型"""
        mock_model = Mock()
        mock_model.data = pd.DataFrame()

        result = self.validator.validate_data_model(mock_model)

        assert result['is_valid'] == False
        assert '数据为空或格式错误' in result['errors']

    def test_validate_data_model_missing_required_columns(self):
        """测试缺少必要列的数据模型"""
        mock_model = Mock()
        mock_model.data = pd.DataFrame({
            'open': [10.0, 11.0],
            'high': [11.0, 12.0]
            # 缺少 low, close, volume
        })

        result = self.validator.validate_data_model(mock_model)

        assert result['is_valid'] == False
        assert '缺少必要的列' in result['errors'][0]
        assert 'low' in result['errors'][0]
        assert 'close' in result['errors'][0]
        assert 'volume' in result['errors'][0]

    def test_validate_data_model_invalid_column_types(self):
        """测试列类型不正确的数据模型"""
        mock_model = Mock()
        mock_model.data = pd.DataFrame({
            'open': ['10.0', '11.0'],  # 应该是数值类型
            'high': [11.0, 12.0],
            'low': [9.0, 10.0],
            'close': [10.5, 11.5],
            'volume': [1000, 1100]
        })

        result = self.validator.validate_data_model(mock_model)

        assert result['is_valid'] == False
        assert '列 open 不是数值类型' in result['errors'][0]

    def test_validate_data_model_with_null_values(self):
        """测试包含空值的数据模型"""
        mock_model = Mock()
        mock_model.data = pd.DataFrame({
            'open': [10.0, np.nan, 12.0],
            'high': [11.0, 12.0, 13.0],
            'low': [9.0, 10.0, 11.0],
            'close': [10.5, 11.5, 12.5],
            'volume': [1000, 1100, 1200]
        })

        result = self.validator.validate_data_model(mock_model)

        assert result['is_valid'] == True
        assert result['errors'] == []
        assert len(result['warnings']) > 0
        assert 'open' in result['warnings'][0]  # 应该有空值警告

    def test_validate_data_model_complete_valid_data(self):
        """测试完全有效的数据模型"""
        mock_model = Mock()
        mock_model.data = pd.DataFrame({
            'open': [10.0, 11.0, 12.0],
            'high': [11.0, 12.0, 13.0],
            'low': [9.0, 10.0, 11.0],
            'close': [10.5, 11.5, 12.5],
            'volume': [1000, 1100, 1200],
            'symbol': ['000001.SZ', '000002.SZ', '000003.SZ'],
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
        })

        result = self.validator.validate_data_model(mock_model)

        assert result['is_valid'] == True
        assert result['errors'] == []
        assert result['warnings'] == []

    def test_validate_data_model_invalid_data_types(self):
        """测试无效数据类型的验证"""
        mock_model = Mock()
        mock_model.data = "invalid_data_type"  # 字符串而不是DataFrame

        result = self.validator.validate_data_model(mock_model)

        assert result['is_valid'] == False
        assert '数据为空或格式错误' in result['errors'][0]

    def test_validator_initialization(self):
        """测试验证器初始化"""
        validator = ChinaStockValidator()
        assert validator is not None
        assert hasattr(validator, 'validate_data')
        assert hasattr(validator, 'validate_data_model')
