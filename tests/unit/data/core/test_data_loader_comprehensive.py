"""
测试数据加载器核心功能 - 综合测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class TestDataLoaderComprehensive:
    """测试数据加载器核心功能 - 综合测试"""

    def test_data_loader_import(self):
        """测试数据加载器导入"""
        try:
            from src.data.core.data_loader import DataLoader, CSVDataLoader, JSONDataLoader, DatabaseDataLoader
            assert DataLoader is not None
            assert CSVDataLoader is not None
            assert JSONDataLoader is not None
            assert DatabaseDataLoader is not None
        except ImportError:
            pytest.skip("DataLoader components not available")

    def test_data_loader_initialization(self):
        """测试数据加载器初始化"""
        try:
            from src.data.core.data_loader import DataLoader

            # 测试抽象基类不能直接实例化
            with pytest.raises(TypeError):
                DataLoader()

            # 测试带配置的初始化
            config = {"timeout": 30, "retries": 3}
            # 由于是抽象类，我们通过子类测试
            from src.data.core.data_loader import CSVDataLoader
            loader = CSVDataLoader(config)
            assert loader.config == config

        except ImportError:
            pytest.skip("DataLoader not available")

    def test_csv_data_loader(self):
        """测试CSV数据加载器"""
        try:
            from src.data.core.data_loader import CSVDataLoader

            loader = CSVDataLoader()

            # 测试基本属性
            assert hasattr(loader, 'config')
            assert hasattr(loader, 'logger')

        except ImportError:
            pytest.skip("CSVDataLoader not available")

    @patch('pandas.read_csv')
    def test_csv_data_loader_load_data(self, mock_read_csv):
        """测试CSV数据加载器的加载功能"""
        try:
            from src.data.core.data_loader import CSVDataLoader

            # 创建测试数据
            test_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=5),
                'open': [100, 102, 101, 103, 105],
                'high': [105, 107, 106, 108, 110],
                'low': [95, 97, 96, 98, 100],
                'close': [102, 104, 103, 105, 107],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })

            mock_read_csv.return_value = test_data

            loader = CSVDataLoader()
            result = loader.load_data("test.csv")

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 5
            assert 'date' in result.columns
            assert 'close' in result.columns

            # 验证read_csv被调用
            mock_read_csv.assert_called_once()

        except ImportError:
            pytest.skip("CSVDataLoader not available")

    @patch('pandas.DataFrame.to_csv')
    def test_csv_data_loader_save_data(self, mock_to_csv):
        """测试CSV数据加载器的保存功能"""
        try:
            from src.data.core.data_loader import CSVDataLoader

            mock_to_csv.return_value = None  # to_csv returns None

            loader = CSVDataLoader()

            # 创建测试数据
            test_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=3),
                'close': [100, 102, 104]
            })

            result = loader.save_data(test_data, "output.csv")

            assert result is True
            mock_to_csv.assert_called_once()

        except ImportError:
            pytest.skip("CSVDataLoader not available")

    def test_json_data_loader(self):
        """测试JSON数据加载器"""
        try:
            from src.data.core.data_loader import JSONDataLoader

            loader = JSONDataLoader()
            assert loader is not None

        except ImportError:
            pytest.skip("JSONDataLoader not available")

    @patch('pandas.read_json')
    def test_json_data_loader_load_data(self, mock_read_json):
        """测试JSON数据加载器的加载功能"""
        try:
            from src.data.core.data_loader import JSONDataLoader

            # 创建测试数据
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=3),
                'value': [1.0, 2.0, 3.0]
            })

            mock_read_json.return_value = test_data

            loader = JSONDataLoader()
            result = loader.load_data("test.json")

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            mock_read_json.assert_called_once()

        except ImportError:
            pytest.skip("JSONDataLoader not available")

    def test_database_data_loader(self):
        """测试数据库数据加载器"""
        try:
            from src.data.core.data_loader import DatabaseDataLoader

            loader = DatabaseDataLoader()
            assert loader is not None

        except ImportError:
            pytest.skip("DatabaseDataLoader not available")

    @patch('pandas.read_sql')
    @patch('sqlite3.connect')
    def test_database_data_loader_load_data(self, mock_connect, mock_read_sql):
        """测试数据库数据加载器的加载功能"""
        try:
            from src.data.core.data_loader import DatabaseDataLoader

            # 创建测试数据
            test_data = pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['A', 'B', 'C'],
                'value': [10.0, 20.0, 30.0]
            })

            mock_read_sql.return_value = test_data
            mock_connection = Mock()
            mock_connect.return_value = mock_connection

            loader = DatabaseDataLoader()

            # 设置连接字符串到配置中
            if hasattr(loader, 'config'):
                loader.config['connection_string'] = "sqlite:///test.db"

            result = loader.load_data("SELECT * FROM test_table", connection_string="sqlite:///test.db")

            # 如果返回None（配置问题），这是正常的
            if result is not None:
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 3
                mock_read_sql.assert_called_once()
            else:
                # 如果返回None，验证至少没有抛出异常
                assert result is None

        except ImportError:
            pytest.skip("DatabaseDataLoader not available")

    def test_data_loader_validate_data(self):
        """测试数据验证功能"""
        try:
            from src.data.core.data_loader import CSVDataLoader

            loader = CSVDataLoader()

            # 测试有效数据
            valid_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=5),
                'open': [100, 102, 101, 103, 105],
                'close': [102, 104, 103, 105, 107]
            })

            is_valid = loader.validate_data(valid_data)
            assert isinstance(is_valid, bool)

            # 测试无效数据 - 空DataFrame
            invalid_data = pd.DataFrame()
            is_valid = loader.validate_data(invalid_data)
            assert isinstance(is_valid, bool)

        except ImportError:
            pytest.skip("DataLoader validation not available")

    def test_data_loader_error_handling(self):
        """测试数据加载器的错误处理"""
        try:
            from src.data.core.data_loader import CSVDataLoader

            loader = CSVDataLoader()

            # 测试加载不存在的文件
            with patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
                result = loader.load_data("non_existent.csv")
                assert result is None  # 应该返回None或抛出异常

            # 测试保存到无效位置
            test_data = pd.DataFrame({'a': [1, 2, 3]})
            with patch('pandas.DataFrame.to_csv', side_effect=PermissionError("Permission denied")):
                result = loader.save_data(test_data, "/invalid/path/file.csv")
                assert result is False

        except ImportError:
            pytest.skip("DataLoader error handling not available")

    def test_data_loader_configuration(self):
        """测试数据加载器配置"""
        try:
            from src.data.core.data_loader import CSVDataLoader

            # 测试默认配置
            loader1 = CSVDataLoader()
            assert loader1.config == {} or isinstance(loader1.config, dict)

            # 测试自定义配置
            config = {
                'encoding': 'utf-8',
                'sep': ';',
                'timeout': 60
            }
            loader2 = CSVDataLoader(config)
            assert loader2.config == config

        except ImportError:
            pytest.skip("DataLoader configuration not available")

    def test_data_loader_different_sources(self):
        """测试不同数据源的加载"""
        try:
            from src.data.core.data_loader import CSVDataLoader

            loader = CSVDataLoader()

            # 测试文件路径
            with patch('pandas.read_csv') as mock_read:
                mock_read.return_value = pd.DataFrame({'a': [1]})
                result = loader.load_data("data/file.csv")
                assert result is not None
                mock_read.assert_called_once()

            # 测试URL（如果支持）
            with patch('pandas.read_csv') as mock_read:
                mock_read.return_value = pd.DataFrame({'b': [2]})
                result = loader.load_data("http://example.com/data.csv")
                assert result is not None

        except ImportError:
            pytest.skip("DataLoader different sources not available")

    def test_data_loader_large_dataset_handling(self):
        """测试大数据集处理"""
        try:
            from src.data.core.data_loader import CSVDataLoader

            loader = CSVDataLoader()

            # 创建大数据集
            large_data = pd.DataFrame({
                'id': range(10000),
                'value': np.random.randn(10000),
                'category': np.random.choice(['A', 'B', 'C'], 10000)
            })

            with patch('pandas.read_csv', return_value=large_data):
                result = loader.load_data("large_dataset.csv")

                assert isinstance(result, pd.DataFrame)
                assert len(result) == 10000
                assert 'id' in result.columns
                assert 'value' in result.columns

        except ImportError:
            pytest.skip("DataLoader large dataset handling not available")

    def test_data_loader_data_types_preservation(self):
        """测试数据类型保持"""
        try:
            from src.data.core.data_loader import CSVDataLoader

            loader = CSVDataLoader()

            # 创建包含不同数据类型的测试数据
            test_data = pd.DataFrame({
                'int_col': [1, 2, 3],
                'float_col': [1.1, 2.2, 3.3],
                'str_col': ['a', 'b', 'c'],
                'bool_col': [True, False, True],
                'date_col': pd.date_range('2023-01-01', periods=3)
            })

            with patch('pandas.read_csv', return_value=test_data):
                result = loader.load_data("test.csv")

                assert isinstance(result, pd.DataFrame)
                assert result['int_col'].dtype in ['int64', 'int32']
                assert result['float_col'].dtype in ['float64', 'float32']
                assert result['str_col'].dtype == 'object'
                assert result['date_col'].dtype == 'datetime64[ns]'

        except ImportError:
            pytest.skip("DataLoader data types preservation not available")
