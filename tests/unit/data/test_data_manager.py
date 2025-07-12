"""
测试数据管理器
"""
import unittest
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import re
from datetime import datetime, timedelta

from src.data.data_manager import DataManager, DataModel
from src.infrastructure.utils.exceptions import DataLoaderError


@pytest.fixture
def sample_data():
    """创建示例数据"""
    # 创建股票数据
    dates = pd.date_range('2023-01-01', '2023-01-10')
    symbols = ['000001.SZ', '000002.SZ', '000003.SZ']

    data = []
    for date in dates:
        for symbol in symbols:
            data.append({
                'date': date,
                'symbol': symbol,
                'close': np.random.randint(10, 100),
                'volume': np.random.randint(1000, 10000)
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_metadata():
    """创建示例元数据"""
    return {
        'type': 'stock',
        'start_date': '2023-01-01',
        'end_date': '2023-01-10',
        'symbols': ['000001.SZ', '000002.SZ', '000003.SZ'],
        'source': 'test'
    }


@pytest.fixture
def sample_config():
    """创建示例配置"""
    return {
        'version_dir': './test_versions',
        'stock_config': {'api_key': 'test_key'},
        'index_config': {'source': 'test'},
        'financial_config': {'db_url': 'test_db'},
        'news_config': {'api_url': 'test_url'}
    }


@pytest.fixture
def data_manager(sample_config):
    """创建数据管理器实例"""
    with patch('src.data.loader.stock_loader.StockDataLoader'), \
         patch('src.data.loader.stock_loader.StockListLoader'), \
         patch('src.data.loader.stock_loader.IndustryLoader'), \
         patch('src.data.loader.index_loader.IndexDataLoader'), \
         patch('src.data.loader.financial_loader.FinancialDataLoader'), \
         patch('src.data.loader.news_loader.FinancialNewsLoader'), \
         patch('src.data.loader.news_loader.SentimentNewsLoader'):
        manager = DataManager(config_dict=sample_config)
        return manager


@pytest.fixture
def temp_version_dir():
    """创建临时版本目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestDataManager:
    """测试数据管理器"""

    @pytest.mark.parametrize("data_type", ["stock", "index", "financial", "news"])
    def test_load_data(self, data_manager, sample_data, sample_metadata, data_type):
        """测试加载数据"""
        # Mock 加载器的 load_data 方法
        loader_map = {
            'stock': data_manager.stock_loader,
            'index': data_manager.index_loader,
            'financial': data_manager.financial_loader,
            'news': data_manager.news_loader
        }
        loader_map[data_type].load_data = Mock(return_value=sample_data)

        # 加载数据
        model = data_manager.load_data(
            data_type=data_type,
            start_date='2023-01-01',
            end_date='2023-01-10',
            symbols=['000001.SZ']
        )

        # 验证结果
        assert isinstance(model, DataModel)
        assert isinstance(model.data, pd.DataFrame)
        pd.testing.assert_frame_equal(model.data, sample_data)

        # 验证元数据
        metadata = model.get_metadata()
        assert metadata['type'] == data_type
        assert re.match(r'^\d{4}-\d{2}-\d{2}$', metadata['start_date'])
        assert re.match(r'^\d{4}-\d{2}-\d{2}$', metadata['end_date'])
        assert isinstance(metadata['symbols'], list)

    def test_load_data_invalid_type(self, data_manager):
        """测试加载无效数据源类型"""
        print(f"Testing with data_manager: {data_manager}")
        with pytest.raises(DataLoaderError, match=r"无效的数据源类型: invalid"):
            data_manager.load_data(
                data_type='invalid',
                start_date='2023-01-01',
                end_date='2023-01-10'
            )

    @pytest.mark.parametrize("start_date,end_date", [
        ('2023-01-01', '2023-01-10'),
        ('2023-01-01', '2023-01-01'),
        ('2023-12-31', '2024-01-01')
    ])
    def test_load_data_date_range(self, data_manager, sample_data, start_date, end_date):
        """测试不同日期范围的数据加载"""
        data_manager.stock_loader.load_data = Mock(return_value=sample_data)

        model = data_manager.load_data(
            data_type='stock',
            start_date=start_date,
            end_date=end_date,
            symbols=['000001.SZ']
        )

        metadata = model.get_metadata()
        assert metadata['start_date'] == start_date
        assert metadata['end_date'] == end_date

    def test_version_management(self, data_manager, sample_data, sample_metadata):
        """测试版本管理功能"""
        # Mock 加载器
        data_manager.stock_loader.load_data = Mock(return_value=sample_data)

        # 创建初始版本
        model1 = data_manager.load_data(
            data_type='stock',
            start_date='2023-01-01',
            end_date='2023-01-10',
            symbols=['000001.SZ']
        )
        version1 = data_manager.version_manager.current_version

        # 修改数据创建新版本
        modified_data = sample_data.copy()
        modified_data['close'] = modified_data['close'] * 1.1
        data_manager.stock_loader.load_data = Mock(return_value=modified_data)

        model2 = data_manager.load_data(
            data_type='stock',
            start_date='2023-01-01',
            end_date='2023-01-10',
            symbols=['000001.SZ']
        )
        version2 = data_manager.version_manager.current_version

        # 比较版本
        diff = data_manager.compare_versions(version1, version2)
        assert 'metadata_diff' in diff
        assert 'data_diff' in diff

        # 回滚到第一个版本
        rollback_version = data_manager.rollback(version1)
        rollback_model = data_manager.get_version(rollback_version)
        pd.testing.assert_frame_equal(rollback_model.data, model1.data)

    @pytest.mark.parametrize("data_types", [
        ['stock', 'index'],
        ['stock', 'financial'],
        ['stock', 'news'],
        ['stock', 'index', 'financial']
    ])
    def test_merge_data(self, data_manager, sample_data, data_types):
        """测试数据合并"""
        # Mock 各个加载器
        for data_type in data_types:
            loader_map = {
                'stock': data_manager.stock_loader,
                'index': data_manager.index_loader,
                'financial': data_manager.financial_loader,
                'news': data_manager.news_loader
            }
            # 为每个数据类型创建略微不同的数据
            modified_data = sample_data.copy()
            if data_type != 'stock':
                modified_data[f'{data_type}_value'] = np.random.rand(len(modified_data))
            loader_map[data_type].load_data = Mock(return_value=modified_data)

        # 合并数据
        merged_model = data_manager.merge_data(
            data_types=data_types,
            start_date='2023-01-01',
            end_date='2023-01-10',
            symbols=['000001.SZ']
        )

        # 验证结果
        assert isinstance(merged_model, DataModel)
        assert isinstance(merged_model.data, pd.DataFrame)

        # 验证合并后的数据包含所有必要的列
        expected_columns = {'date', 'symbol', 'close', 'volume'}
        for data_type in data_types:
            if data_type != 'stock':
                expected_columns.add(f'{data_type}_value')

        assert set(merged_model.data.columns).issuperset(expected_columns)

        # 验证元数据
        metadata = merged_model.get_metadata()
        assert metadata['type'] == 'merged'
        assert set(metadata['data_types']) == set(data_types)

    def test_get_lineage(self, data_manager, sample_data):
        """测试获取版本血缘关系"""
        # Mock 加载器
        data_manager.stock_loader.load_data = Mock(return_value=sample_data)

        # 创建多个版本
        model1 = data_manager.load_data(
            data_type='stock',
            start_date='2023-01-01',
            end_date='2023-01-10',
            symbols=['000001.SZ']
        )
        version1 = data_manager.version_manager.current_version

        modified_data = sample_data.copy()
        modified_data['close'] = modified_data['close'] * 1.1
        data_manager.stock_loader.load_data = Mock(return_value=modified_data)

        model2 = data_manager.load_data(
            data_type='stock',
            start_date='2023-01-01',
            end_date='2023-01-10',
            symbols=['000001.SZ']
        )
        version2 = data_manager.version_manager.current_version

        # 获取血缘关系
        lineage = data_manager.get_lineage(version2)

        # 验证血缘关系
        assert lineage['version_id'] == version2
        assert isinstance(lineage['ancestors'], list)
        assert any(ancestor['version_id'] == version1 for ancestor in lineage['ancestors'])

    def test_list_versions(self, data_manager, sample_data):
        """测试列出版本"""
        # Mock 加载器
        data_manager.stock_loader.load_data = Mock(return_value=sample_data)

        # 创建多个版本
        for i in range(3):
            modified_data = sample_data.copy()
            modified_data['close'] = modified_data['close'] * (1 + i * 0.1)
            data_manager.stock_loader.load_data = Mock(return_value=modified_data)

            data_manager.load_data(
                data_type='stock',
                start_date='2023-01-01',
                end_date='2023-01-10',
                symbols=['000001.SZ'],
                tags=[f'v{i+1}']
            )

        # 列出所有版本
        versions = data_manager.list_versions()
        assert len(versions) == 3

        # 按标签筛选
        versions = data_manager.list_versions(tags=['v1'])
        assert len(versions) == 1

        # 限制数量
        versions = data_manager.list_versions(limit=2)
        assert len(versions) == 2


if __name__ == '__main__':
    pytest.main(['-v', 'test_data_manager.py'])
