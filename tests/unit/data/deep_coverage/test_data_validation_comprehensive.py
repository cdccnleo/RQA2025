"""
数据验证模块深度测试
全面测试数据验证系统的各种功能和边界条件
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os
import json

# 导入实际的类
from src.data.validation.validator import DataValidator
from src.data.validation.china_stock_validator import ChinaStockValidator
from src.data.validation.validator_components import ValidatorComponent, ValidatorComponentFactory


class TestDataValidationComprehensive:
    """数据验证综合深度测试"""

    @pytest.fixture
    def sample_stock_data(self):
        """创建样本股票数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'symbol': ['000001'] * 100,
            'date': dates,
            'open': np.random.uniform(10, 20, 100),
            'high': np.random.uniform(15, 25, 100),
            'low': np.random.uniform(8, 18, 100),
            'close': np.random.uniform(10, 20, 100),
            'volume': np.random.randint(100000, 10000000, 100),
            'amount': np.random.uniform(1000000, 10000000, 100)
        })

    @pytest.fixture
    def invalid_stock_data(self):
        """创建无效的股票数据用于测试"""
        return pd.DataFrame({
            'symbol': ['INVALID'] * 10,
            'date': ['invalid_date'] * 10,
            'open': ['not_a_number'] * 10,
            'high': np.random.uniform(15, 25, 10),
            'low': np.random.uniform(8, 18, 10),
            'close': np.random.uniform(10, 20, 10),
            'volume': np.random.randint(100000, 10000000, 10),
            'amount': np.random.uniform(1000000, 10000000, 10)
        })

    @pytest.fixture
    def data_validator(self):
        """创建数据验证器实例"""
        return DataValidator()

    @pytest.fixture
    def china_stock_validator(self):
        """创建中国股票验证器实例"""
        return ChinaStockValidator()

    @pytest.fixture
    def validator_component(self):
        """创建验证器组件实例"""
        return ValidatorComponent()

    def test_data_validator_initialization(self, data_validator):
        """测试数据验证器初始化"""
        assert data_validator is not None
        assert hasattr(data_validator, 'config')

    def test_china_stock_validator_initialization(self, china_stock_validator):
        """测试中国股票验证器初始化"""
        assert china_stock_validator is not None
        assert hasattr(china_stock_validator, 'stock_code_pattern')

    def test_validator_component_initialization(self, validator_component):
        """测试验证器组件初始化"""
        assert validator_component is not None

    def test_basic_data_type_validation(self, data_validator, sample_stock_data):
        """测试基本数据类型验证"""
        # 测试数值类型验证
        result = data_validator.validate_numeric_columns(sample_stock_data, ['close', 'volume'])
        assert result['valid'] is True

        # 测试无效数据
        invalid_data = sample_stock_data.copy()
        invalid_data['close'] = invalid_data['close'].astype(str)
        result = data_validator.validate_numeric_columns(invalid_data, ['close'])
        assert result['valid'] is False

    def test_date_format_validation(self, data_validator, sample_stock_data):
        """测试日期格式验证"""
        # 测试有效日期
        result = data_validator.validate_date_columns(sample_stock_data, ['date'])
        assert result['valid'] is True

        # 测试无效日期
        invalid_data = sample_stock_data.copy()
        invalid_data['date'] = ['invalid_date'] * len(invalid_data)
        result = data_validator.validate_date_columns(invalid_data, ['date'])
        assert result['valid'] is False

    def test_range_validation(self, data_validator, sample_stock_data):
        """测试范围验证"""
        # 测试价格范围
        price_ranges = {
            'close': {'min': 0, 'max': 1000},
            'volume': {'min': 0, 'max': 100000000}
        }

        result = data_validator.validate_value_ranges(sample_stock_data, price_ranges)
        assert result['valid'] is True

        # 测试超出范围的值
        out_of_range_data = sample_stock_data.copy()
        out_of_range_data.loc[0, 'close'] = 10000  # 超出最大值
        result = data_validator.validate_value_ranges(out_of_range_data, price_ranges)
        assert result['valid'] is False

    def test_missing_value_validation(self, data_validator, sample_stock_data):
        """测试缺失值验证"""
        # 测试无缺失值数据
        result = data_validator.validate_missing_values(sample_stock_data, ['close', 'volume'])
        assert result['valid'] is True

        # 测试包含缺失值的数据
        missing_data = sample_stock_data.copy()
        missing_data.loc[0, 'close'] = np.nan
        result = data_validator.validate_missing_values(missing_data, ['close'])
        assert result['valid'] is False

    def test_china_stock_code_validation(self, china_stock_validator):
        """测试中国股票代码验证"""
        # 测试有效股票代码
        valid_codes = ['000001', '600000', '300001', '000001.SZ', '600000.SH']
        for code in valid_codes:
            assert china_stock_validator.validate_stock_code(code) is True

        # 测试无效股票代码
        invalid_codes = ['123456', 'abc123', '000001.BB', '12345678']
        for code in invalid_codes:
            assert china_stock_validator.validate_stock_code(code) is False

    def test_china_stock_data_validation(self, china_stock_validator, sample_stock_data):
        """测试中国股票数据验证"""
        # 修改数据以符合中国股票格式
        china_data = sample_stock_data.copy()
        china_data['symbol'] = '000001'

        result = china_stock_validator.validate_stock_data(china_data)
        assert 'valid' in result
        assert 'errors' in result

    def test_business_logic_validation(self, data_validator, sample_stock_data):
        """测试业务逻辑验证"""
        # 测试价格逻辑：最高价 >= 收盘价 >= 最低价
        business_rules = [
            lambda df: (df['high'] >= df['close']).all(),
            lambda df: (df['close'] >= df['low']).all(),
            lambda df: (df['volume'] > 0).all()
        ]

        result = data_validator.validate_business_rules(sample_stock_data, business_rules)
        assert result['valid'] is True

        # 测试违反业务规则的数据
        invalid_data = sample_stock_data.copy()
        invalid_data.loc[0, 'high'] = invalid_data.loc[0, 'low'] - 1  # 最高价低于最低价
        result = data_validator.validate_business_rules(invalid_data, business_rules)
        assert result['valid'] is False

    def test_schema_validation(self, data_validator, sample_stock_data):
        """测试模式验证"""
        # 定义数据模式
        schema = {
            'symbol': {'type': 'string', 'required': True},
            'date': {'type': 'datetime', 'required': True},
            'close': {'type': 'number', 'required': True, 'min': 0},
            'volume': {'type': 'integer', 'required': True, 'min': 0}
        }

        result = data_validator.validate_schema(sample_stock_data, schema)
        assert result['valid'] is True

        # 测试违反模式的数据
        invalid_schema_data = sample_stock_data.copy()
        invalid_schema_data['close'] = invalid_schema_data['close'].astype(str)
        result = data_validator.validate_schema(invalid_schema_data, schema)
        assert result['valid'] is False

    def test_cross_field_validation(self, data_validator, sample_stock_data):
        """测试跨字段验证"""
        # 测试成交金额与成交量关系（大致合理性检查）
        def volume_amount_consistency(df):
            # 计算平均价格
            avg_price = (df['open'] + df['close']) / 2
            # 估算成交金额
            estimated_amount = avg_price * df['volume']
            # 检查实际成交金额与估算金额的合理性（允许20%误差）
            return ((df['amount'] / estimated_amount - 1).abs() < 0.2).all()

        result = data_validator.validate_cross_field_rules(sample_stock_data, [volume_amount_consistency])
        assert 'valid' in result

    def test_data_consistency_validation(self, data_validator, sample_stock_data):
        """测试数据一致性验证"""
        # 测试时间序列一致性
        result = data_validator.validate_time_series_consistency(sample_stock_data, 'date')
        assert result['valid'] is True

        # 测试重复数据
        duplicate_data = pd.concat([sample_stock_data, sample_stock_data.head(5)])
        result = data_validator.validate_duplicates(duplicate_data, ['symbol', 'date'])
        assert result['valid'] is False

    def test_validation_error_reporting(self, data_validator, invalid_stock_data):
        """测试验证错误报告"""
        # 验证无效数据
        result = data_validator.validate_comprehensive(invalid_stock_data)

        # 检查错误报告结构
        assert 'errors' in result
        assert 'warnings' in result
        assert 'summary' in result

        # 检查是否报告了错误
        assert len(result['errors']) > 0

    def test_validation_performance(self, data_validator, sample_stock_data):
        """测试验证性能"""
        import time

        # 创建大数据集
        large_data = pd.concat([sample_stock_data] * 100)

        start_time = time.time()
        result = data_validator.validate_comprehensive(large_data)
        end_time = time.time()

        # 检查性能（应该在合理时间内完成）
        validation_time = end_time - start_time
        assert validation_time < 30  # 30秒内完成

        # 检查结果仍然有效
        assert 'errors' in result

    def test_validation_configurability(self, data_validator):
        """测试验证可配置性"""
        # 测试不同配置
        configs = [
            {'strict_mode': True, 'fail_fast': True},
            {'strict_mode': False, 'fail_fast': False},
            {'custom_rules': ['rule1', 'rule2']}
        ]

        for config in configs:
            validator = DataValidator(config)
            assert validator.config == config

    def test_partial_validation(self, data_validator, sample_stock_data):
        """测试部分验证"""
        # 只验证特定字段
        result = data_validator.validate_partial(sample_stock_data, columns=['close', 'volume'])
        assert 'close' in str(result)
        assert 'volume' in str(result)

    def test_validation_rule_customization(self, data_validator):
        """测试验证规则定制"""
        # 添加自定义验证规则
        def custom_price_rule(value):
            return 0 <= value <= 1000

        data_validator.add_custom_rule('price_range', custom_price_rule)

        # 测试自定义规则
        test_data = pd.DataFrame({'price': [100, 500, 1500]})  # 1500超出范围
        result = data_validator.validate_with_custom_rules(test_data, ['price_range'])
        assert result['valid'] is False

    def test_validation_statistics(self, data_validator, sample_stock_data):
        """测试验证统计信息"""
        result = data_validator.validate_with_statistics(sample_stock_data)

        # 检查统计信息
        assert 'total_records' in result
        assert 'validated_fields' in result
        assert 'validation_time' in result
        assert result['total_records'] == len(sample_stock_data)

    def test_batch_validation(self, data_validator, sample_stock_data):
        """测试批量验证"""
        # 创建数据批次
        batches = [sample_stock_data] * 5

        results = data_validator.validate_batch(batches)

        # 检查批量结果
        assert len(results) == 5
        for result in results:
            assert 'valid' in result

    def test_validation_error_recovery(self, data_validator):
        """测试验证错误恢复"""
        # 创建会导致验证失败的数据
        bad_data = pd.DataFrame({
            'column1': [None, None, None],
            'column2': ['invalid'] * 3
        })

        # 测试错误恢复
        result = data_validator.validate_with_error_recovery(bad_data)

        # 检查错误恢复机制
        assert result is not None
        assert 'recovered_errors' in result

    def test_validator_component_factory(self):
        """测试验证器组件工厂"""
        factory = ValidatorComponentFactory()

        # 创建不同类型的验证器组件
        component_types = ['basic', 'advanced', 'custom']

        for comp_type in component_types:
            try:
                component = factory.create_component(comp_type)
                assert component is not None
            except NotImplementedError:
                # 如果组件类型未实现，跳过
                continue

    def test_validation_chain_of_responsibility(self, data_validator, sample_stock_data):
        """测试验证责任链模式"""
        # 创建验证链
        validation_chain = [
            'type_check',
            'range_check',
            'consistency_check',
            'business_rule_check'
        ]

        result = data_validator.validate_with_chain(sample_stock_data, validation_chain)

        # 检查链式验证结果
        assert 'chain_results' in result
        assert len(result['chain_results']) == len(validation_chain)

    def test_validation_report_generation(self, data_validator, sample_stock_data):
        """测试验证报告生成"""
        result = data_validator.validate_and_generate_report(sample_stock_data)

        # 检查报告内容
        assert 'summary' in result
        assert 'details' in result
        assert 'recommendations' in result

        # 检查报告可以序列化
        report_json = json.dumps(result, default=str)
        assert len(report_json) > 0

    def test_real_time_validation(self, data_validator):
        """测试实时验证"""
        # 模拟实时数据流
        real_time_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'price': [150.0],
            'timestamp': [datetime.now()]
        })

        # 实时验证
        result = data_validator.validate_real_time(real_time_data)

        # 检查实时验证结果
        assert 'is_valid' in result
        assert 'processing_time' in result
        assert result['processing_time'] < 1.0  # 实时验证应很快

    def test_validation_rule_engine(self, data_validator):
        """测试验证规则引擎"""
        # 定义规则引擎配置
        rule_config = {
            'rules': [
                {
                    'name': 'price_positive',
                    'condition': 'price > 0',
                    'action': 'pass'
                },
                {
                    'name': 'volume_reasonable',
                    'condition': 'volume > 0 and volume < 100000000',
                    'action': 'pass'
                }
            ]
        }

        # 创建规则引擎验证器
        rule_validator = data_validator.create_rule_engine_validator(rule_config)

        # 测试数据
        test_data = pd.DataFrame({
            'price': [100, -5, 200],  # -5为无效价格
            'volume': [1000000, 2000000, 50000000]
        })

        result = rule_validator.validate(test_data)
        assert result['valid'] is False  # 因为有负数价格

    def test_validation_threshold_configuration(self, data_validator):
        """测试验证阈值配置"""
        # 配置不同的验证阈值
        thresholds = {
            'max_errors': 10,
            'error_tolerance': 0.1,  # 10%的错误容忍度
            'warning_threshold': 5
        }

        validator = DataValidator({'thresholds': thresholds})

        # 创建包含一些错误的数据
        error_data = pd.DataFrame({
            'value': list(range(100)) + ['error'] * 5  # 5个错误，占5%
        })

        result = validator.validate_with_thresholds(error_data)

        # 检查阈值处理
        assert 'within_threshold' in result
        # 由于错误率5%低于10%的容忍度，应该在阈值内
        assert result['within_threshold'] is True

    def test_validation_audit_trail(self, data_validator, sample_stock_data):
        """测试验证审计跟踪"""
        # 启用审计跟踪
        audit_validator = DataValidator({'enable_audit': True})

        result = audit_validator.validate_with_audit(sample_stock_data)

        # 检查审计信息
        assert 'audit_trail' in result
        assert len(result['audit_trail']) > 0

        # 检查审计条目结构
        audit_entry = result['audit_trail'][0]
        assert 'timestamp' in audit_entry
        assert 'operation' in audit_entry
        assert 'result' in audit_entry
