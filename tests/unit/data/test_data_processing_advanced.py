#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data层 - 数据处理高级测试

测试数据加载、转换、验证、存储
"""

import pytest
import pandas as pd
import numpy as np


class TestDataLoading:
    """测试数据加载"""
    
    def test_load_csv_data(self):
        """测试加载CSV数据"""
        # 模拟CSV数据
        data = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=5),
            'price': [100, 102, 101, 105, 103]
        })
        
        assert len(data) == 5
        assert 'price' in data.columns
    
    def test_load_with_schema_validation(self):
        """测试带模式验证的加载"""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10.5, 20.3, 30.1]
        })
        
        # 验证列类型
        assert pd.api.types.is_integer_dtype(data['id'])
        assert pd.api.types.is_float_dtype(data['value'])
    
    def test_incremental_data_load(self):
        """测试增量数据加载"""
        existing_data = pd.DataFrame({'id': [1, 2, 3]})
        new_data = pd.DataFrame({'id': [4, 5]})
        
        # 合并数据
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        
        assert len(combined) == 5


class TestDataTransformation:
    """测试数据转换"""
    
    def test_filter_data(self):
        """测试过滤数据"""
        data = pd.DataFrame({
            'value': [10, 20, 30, 40, 50]
        })
        
        filtered = data[data['value'] > 25]
        
        assert len(filtered) == 3
    
    def test_aggregate_data(self):
        """测试聚合数据"""
        data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40]
        })
        
        aggregated = data.groupby('category')['value'].sum()
        
        assert aggregated['A'] == 40
        assert aggregated['B'] == 60
    
    def test_join_datasets(self):
        """测试连接数据集"""
        df1 = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        df2 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
        
        merged = pd.merge(df1, df2, on='id')
        
        assert len(merged) == 2
        assert 'name' in merged.columns
        assert 'value' in merged.columns


class TestDataValidation:
    """测试数据验证"""
    
    def test_validate_required_fields(self):
        """测试验证必需字段"""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        required_fields = ['id', 'name']
        has_all_fields = all(field in data.columns for field in required_fields)
        
        assert has_all_fields is True
    
    def test_validate_data_types(self):
        """测试验证数据类型"""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'price': [10.5, 20.3, 30.1]
        })
        
        assert pd.api.types.is_integer_dtype(data['id'])
        assert pd.api.types.is_float_dtype(data['price'])
    
    def test_validate_value_ranges(self):
        """测试验证值范围"""
        data = pd.Series([0.1, 0.5, 0.8, 0.95])
        
        in_range = data.between(0, 1).all()
        
        assert in_range == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

