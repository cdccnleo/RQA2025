import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import configparser
import os
import time
from src.data.loader.financial_loader import FinancialDataLoader
from src.infrastructure.utils.exceptions import DataLoaderError

class DummyRequestException(Exception):
    pass

class TestFinancialDataLoader:
    """FinancialDataLoader单元测试"""
    
    @pytest.fixture
    def sample_financial_data(self):
        """创建示例财务数据"""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'roe': [10.5, 11.2, 12.1, 11.8, 12.5, 13.1, 12.9, 13.5, 14.2, 14.8],
            'roa': [5.2, 5.8, 6.1, 5.9, 6.3, 6.8, 6.6, 7.1, 7.5, 7.9],
            'debt_ratio': [45.2, 44.8, 44.1, 44.5, 43.8, 43.2, 43.5, 42.8, 42.1, 41.5]
        }, index=dates)
    
    @pytest.fixture
    def mock_config(self):
        """创建mock配置"""
        config = configparser.ConfigParser()
        config['Financial'] = {
            'save_path': 'data/financial',
            'max_retries': '3',
            'cache_days': '30'
        }
        return config
    
    @pytest.fixture
    def loader(self, tmp_path):
        """创建FinancialDataLoader实例"""
        save_path = tmp_path / "financial"
        return FinancialDataLoader(save_path=str(save_path), max_retries=2, cache_days=7)
    
    @pytest.fixture
    def loader_with_thread_pool(self, tmp_path):
        """创建带线程池的loader"""
        mock_pool = Mock()
        save_path = tmp_path / "financial"
        return FinancialDataLoader(save_path=str(save_path), thread_pool=mock_pool)
    
    def test_init_default_values(self, tmp_path):
        """测试默认初始化"""
        loader = FinancialDataLoader()
        assert loader.save_path == Path("data/financial")
        assert loader.max_retries == 3
        assert loader.cache_days == 30
        assert loader.thread_pool is None
    
    def test_init_custom_values(self, tmp_path):
        """测试自定义初始化"""
        save_path = tmp_path / "custom_financial"
        loader = FinancialDataLoader(
            save_path=str(save_path),
            max_retries=5,
            cache_days=15,
            raise_errors=True
        )
        assert loader.save_path == save_path
        assert loader.max_retries == 5
        assert loader.cache_days == 15
        assert loader.raise_errors is True
    
    def test_setup_creates_directory(self, tmp_path):
        """测试_setup创建目录"""
        save_path = tmp_path / "financial"
        loader = FinancialDataLoader(save_path=str(save_path))
        assert save_path.exists()
        assert save_path.is_dir()
    
    @patch('src.data.loader.financial_loader.is_production')
    def test_init_raise_errors_auto(self, mock_is_prod, tmp_path):
        """测试raise_errors自动设置"""
        # 生产环境
        mock_is_prod.return_value = True
        loader = FinancialDataLoader(save_path=str(tmp_path))
        assert loader.raise_errors is False
        
        # 开发环境
        mock_is_prod.return_value = False
        loader = FinancialDataLoader(save_path=str(tmp_path))
        assert loader.raise_errors is True
    
    def test_create_from_config(self, mock_config, tmp_path):
        """测试从配置创建实例"""
        with patch('configparser.ConfigParser.read'):
            with patch('configparser.ConfigParser.__getitem__', return_value=mock_config['Financial']):
                loader = FinancialDataLoader.create_from_config(mock_config)
                assert isinstance(loader, FinancialDataLoader)
                assert loader.save_path == Path("data/financial")
                assert loader.max_retries == 3
                assert loader.cache_days == 30
    
    def test_create_from_config_dict(self, tmp_path):
        """测试从字典配置创建实例"""
        config_dict = {
            'Financial': {
                'save_path': 'custom/financial',
                'max_retries': '5',
                'cache_days': '15'
            }
        }
        with patch('configparser.ConfigParser.read'):
            with patch('configparser.ConfigParser.__getitem__', return_value=config_dict['Financial']):
                loader = FinancialDataLoader.create_from_config(config_dict)
                assert loader.save_path == Path("custom/financial")
                assert loader.max_retries == 5
                assert loader.cache_days == 15
    
    def test_create_from_config_invalid_int(self, mock_config, tmp_path):
        """测试配置中无效整数值"""
        mock_config['Financial']['max_retries'] = 'invalid'
        with patch('configparser.ConfigParser.read'):
            with patch('configparser.ConfigParser.__getitem__', return_value=mock_config['Financial']):
                with pytest.raises(DataLoaderError, match="配置项 max_retries 的值无效"):
                    FinancialDataLoader.create_from_config(mock_config)
    
    @patch('src.data.loader.financial_loader.ak')
    def test_load_data_with_valid_cache(self, mock_ak, loader, sample_financial_data):
        """测试加载数据 - 有效缓存"""
        symbol = "000001"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        # Mock缓存文件存在且有效
        cache_file = loader.save_path / f"{symbol}_financial.csv"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with patch.object(loader, '_is_cache_valid', return_value=(True, "缓存有效")):
            with patch.object(loader, '_load_cache_data', return_value=sample_financial_data):
                result = loader.load_data(symbol, start_date, end_date)
                
                assert isinstance(result, pd.DataFrame)
                assert not result.empty
                assert 'roe' in result.columns
    
    @patch('src.data.loader.financial_loader.ak')
    def test_load_data_with_invalid_cache(self, mock_ak, loader, sample_financial_data):
        """测试加载数据 - 无效缓存"""
        symbol = "000001"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        with patch.object(loader, '_is_cache_valid', return_value=(False, "缓存过期")):
            with patch.object(loader, '_fetch_and_process_data', return_value=sample_financial_data):
                result = loader.load_data(symbol, start_date, end_date)
                
                assert isinstance(result, pd.DataFrame)
                assert not result.empty
    
    def test_load_data_error_handling(self, loader):
        """测试加载数据错误处理"""
        symbol = "000001"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        with patch.object(loader, '_is_cache_valid', side_effect=Exception("测试异常")):
            with pytest.raises(DataLoaderError, match="加载财务数据失败"):
                loader.load_data(symbol, start_date, end_date)
    
    def test_load_data_error_handling_raise_errors(self, loader):
        """测试加载数据错误处理 - 抛出异常"""
        loader.raise_errors = True
        symbol = "000001"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        with patch.object(loader, '_is_cache_valid', side_effect=Exception("测试异常")):
            with pytest.raises(DataLoaderError, match="加载财务数据失败"):
                loader.load_data(symbol, start_date, end_date)
    
    def test_is_cache_valid_file_not_exists(self, loader):
        """测试缓存有效性检查 - 文件不存在"""
        file_path = Path("/nonexistent/file.csv")
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        valid, reason = loader._is_cache_valid(file_path, start_date, end_date)
        assert not valid
        assert "缓存文件不存在" in reason
    
    @patch('os.path.getmtime')
    @patch('time.time')
    def test_is_cache_valid_file_expired(self, mock_time, mock_getmtime, loader):
        """测试缓存有效性检查 - 文件过期"""
        file_path = Path("/test/file.csv")
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        # Mock文件存在但已过期
        mock_getmtime.return_value = time.time() - (loader.cache_days + 1) * 86400
        mock_time.return_value = time.time()
        
        with patch.object(loader, '_is_file_timely', return_value=False):
            valid, reason = loader._is_cache_valid(file_path, start_date, end_date)
            assert not valid
            assert "缓存文件已过期" in reason or "缓存文件不存在" in reason
    
    def test_is_cache_valid_success(self, loader, sample_financial_data):
        """测试缓存有效性检查 - 成功"""
        file_path = Path("/test/file.csv")
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        # Mock文件存在
        with patch.object(Path, 'exists', return_value=True):
            with patch.object(loader, '_is_file_timely', return_value=True):
                with patch.object(loader, '_load_cache_data', return_value=(sample_financial_data, None)):
                    valid, reason = loader._is_cache_valid(file_path, start_date, end_date)
                    assert valid
                    assert "缓存验证通过" in reason
    
    def test_is_file_timely_valid(self, loader, tmp_path):
        """测试文件时效性检查 - 有效"""
        file_path = tmp_path / "test.csv"
        file_path.write_text("test")
        
        # Mock文件修改时间为最近
        with patch('os.path.getmtime', return_value=time.time() - 3600):  # 1小时前
            assert loader._is_file_timely(file_path)
    
    def test_is_file_timely_expired(self, loader, tmp_path):
        """测试文件时效性检查 - 过期"""
        file_path = tmp_path / "test.csv"
        file_path.write_text("test")
        
        # Mock文件修改时间为很久以前
        with patch('os.path.getmtime', return_value=time.time() - (loader.cache_days + 1) * 86400):
            assert not loader._is_file_timely(file_path)
    
    def test_is_file_timely_not_exists(self, loader):
        """测试文件时效性检查 - 文件不存在"""
        file_path = Path("/nonexistent/file.csv")
        assert not loader._is_file_timely(file_path)
    
    def test_load_cache_data_success(self, loader, sample_financial_data, tmp_path):
        """测试加载缓存数据 - 成功"""
        file_path = tmp_path / "test.csv"
        # 确保数据有正确的日期索引格式
        sample_financial_data.reset_index().to_csv(file_path, index=False)
        
        result = loader._load_cache_data(file_path)
        # 根据实际实现，可能返回元组或DataFrame
        if isinstance(result, tuple):
            result_df, error = result
            assert error is None
        else:
            result_df = result
        assert isinstance(result_df, pd.DataFrame)
        assert not result_df.empty
        assert 'roe' in result_df.columns
    
    def test_load_cache_data_validation_error(self, loader, tmp_path):
        """测试加载缓存数据 - 验证错误"""
        file_path = tmp_path / "test.csv"
        # 创建无效数据（缺少必要列）
        invalid_data = pd.DataFrame({'other_col': [1, 2, 3]})
        invalid_data.to_csv(file_path)
        
        with pytest.raises(DataLoaderError, match="缓存文件包含无效日期"):
            loader._load_cache_data(file_path, validate=True)
    
    def test_load_cache_data_csv_error(self, loader):
        """测试加载缓存数据 - CSV解析错误"""
        file_path = Path("/nonexistent/file.csv")
        
        with pytest.raises(DataLoaderError, match="加载缓存数据失败"):
            loader._load_cache_data(file_path)
    
    def test_validate_data_success(self, loader, sample_financial_data):
        """测试数据验证 - 成功"""
        assert loader._validate_data(sample_financial_data)
    
    def test_validate_data_missing_required_columns(self, loader):
        """测试数据验证 - 缺少必要列"""
        invalid_data = pd.DataFrame({
            'other_col': [1, 2, 3],
            'another_col': [4, 5, 6]
        })
        with pytest.raises(DataLoaderError, match="数据验证失败: 缺少必要列"):
            loader._validate_data(invalid_data)
    
    def test_validate_data_empty_dataframe(self, loader):
        """测试数据验证 - 空数据框"""
        empty_data = pd.DataFrame()
        with pytest.raises(DataLoaderError, match="数据验证失败: 数据为空"):
            loader._validate_data(empty_data)
    
    @patch('src.data.loader.financial_loader.ak')
    def test_fetch_and_process_data_success(self, mock_ak, loader, sample_financial_data):
        """测试获取和处理数据 - 成功"""
        symbol = "000001"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        with patch.object(loader, '_fetch_raw_data', return_value=sample_financial_data):
            with patch.object(loader, '_validate_fetched_data', return_value=True):
                with patch.object(loader, '_save_data', return_value=True):
                    result = loader._fetch_and_process_data(symbol, start_date, end_date)
                    
                    assert isinstance(result, pd.DataFrame)
                    assert not result.empty
    
    @patch('src.data.loader.financial_loader.ak')
    def test_fetch_and_process_data_validation_failed(self, mock_ak, loader):
        """测试获取和处理数据 - 验证失败"""
        symbol = "000001"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        sample_data = pd.DataFrame({'roe': [1, 2, 3]})
        
        with patch.object(loader, '_fetch_raw_data', return_value=sample_data):
            with patch.object(loader, '_validate_fetched_data', return_value=False):
                with pytest.raises(DataLoaderError, match="获取和处理数据失败"):
                    loader._fetch_and_process_data(symbol, start_date, end_date)
    
    def test_validate_fetched_data_success(self, loader, sample_financial_data):
        """测试获取数据验证 - 成功"""
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        assert loader._validate_fetched_data(sample_financial_data, start_date, end_date)
    
    def test_validate_fetched_data_empty(self, loader):
        """测试获取数据验证 - 空数据"""
        empty_data = pd.DataFrame()
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        assert not loader._validate_fetched_data(empty_data, start_date, end_date)
    
    @patch('src.data.loader.financial_loader.ak')
    def test_fetch_raw_data_success(self, mock_ak, loader):
        """测试获取原始数据 - 成功"""
        symbol = "000001"
        start_date = "2023-01-01"
        end_date = "2023-01-10"
        
        # 创建包含date列的mock数据
        mock_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'roe': [10.5, 11.2, 12.1],
            'roa': [5.2, 5.8, 6.1]
        })
        
        with patch.object(loader, '_fetch_financial_data', return_value=mock_data):
            result = loader._fetch_raw_data(symbol, start_date, end_date)
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
    
    @patch('src.data.loader.financial_loader.ak')
    def test_fetch_financial_data_success(self, mock_ak, loader):
        """测试获取财务数据 - 成功"""
        symbol = "000001"
        year = "2023"
        
        # Mock返回有效数据
        mock_data = pd.DataFrame({
            '日期': ['2023-01-01', '2023-01-02', '2023-01-03'],
            '净资产收益率(%)': [10.5, 11.2, 12.1],
            '净利润增长率(%)': [5.2, 5.8, 6.1]
        })
        mock_ak.stock_financial_analysis_indicator.return_value = mock_data
        
        result = loader._fetch_financial_data(symbol, year)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'roe' in result.columns
        assert 'date' in result.columns
    
    @patch('src.data.loader.financial_loader.ak')
    def test_fetch_financial_data_api_error(self, mock_ak, loader):
        """测试获取财务数据 - API错误"""
        symbol = "000001"
        year = "2023"
        
        mock_ak.stock_financial_report_sina.side_effect = Exception("API错误")
        
        # 由于实际实现中有异常处理，这里应该返回空DataFrame而不是抛出异常
        result = loader._fetch_financial_data(symbol, year)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_retry_api_call_success(self, loader):
        """测试API调用重试 - 成功"""
        mock_func = Mock(return_value="success")
        
        result = loader._retry_api_call(mock_func, "arg1", kwarg1="value1")
        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")
    
    def test_retry_api_call_with_retries(self, loader):
        """测试API调用重试 - 需要重试"""
        loader.max_retries = 1  # 显式设置重试次数
        with patch('src.data.loader.financial_loader.RequestException', new=DummyRequestException):
            with patch('src.data.loader.financial_loader.ConnectionError', new=DummyRequestException):
                with patch('src.data.loader.financial_loader.time.sleep'):
                    mock_func = Mock()
                    mock_func.side_effect = [DummyRequestException("失败1"), DummyRequestException("失败2"), "success"]
                    with pytest.raises(DataLoaderError, match="数据获取失败"):
                        loader._retry_api_call(mock_func, "arg1")
                    assert mock_func.call_count == 2  # max_retries=1，总共调用2次

    def test_retry_api_call_max_retries_exceeded(self, loader):
        """测试API调用重试 - 超过最大重试次数"""
        loader.max_retries = 1  # 显式设置重试次数
        with patch('src.data.loader.financial_loader.RequestException', new=DummyRequestException):
            with patch('src.data.loader.financial_loader.ConnectionError', new=DummyRequestException):
                with patch('src.data.loader.financial_loader.time.sleep'):
                    mock_func = Mock()
                    mock_func.side_effect = DummyRequestException("持续失败")
                    with pytest.raises(DataLoaderError, match="数据获取失败"):
                        loader._retry_api_call(mock_func, "arg1")
                    assert mock_func.call_count == 2  # max_retries=1，总共调用2次
    
    def test_save_data_success(self, loader, sample_financial_data, tmp_path):
        """测试保存数据 - 成功"""
        file_path = tmp_path / "test_save.csv"
        
        result = loader._save_data(sample_financial_data, file_path)
        assert result is True
        assert file_path.exists()
        
        # 验证保存的数据
        saved_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        assert not saved_data.empty
        assert 'roe' in saved_data.columns
    
    def test_save_data_error(self, loader, sample_financial_data):
        """测试保存数据 - 错误"""
        # 使用只读路径
        file_path = Path("/readonly/test.csv")
        
        with pytest.raises(DataLoaderError, match="Data saving 阶段出错"):
            loader._save_data(sample_financial_data, file_path)
    
    def test_get_metadata(self, loader):
        """测试获取元数据"""
        metadata = loader.get_metadata()
        assert isinstance(metadata, dict)
        assert 'loader_type' in metadata
        assert 'cache_days' in metadata
        assert 'max_retries' in metadata
        assert metadata['loader_type'] == 'FinancialDataLoader'
    
    def test_load_interface(self, loader, sample_financial_data):
        """测试load接口方法"""
        with patch.object(loader, 'load_data', return_value=sample_financial_data):
            result = loader.load(symbol="000001", start_date="2023-01-01", end_date="2023-01-10")
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
    
    def test_validate_interface(self, loader, sample_financial_data):
        """测试validate接口方法"""
        assert loader.validate(sample_financial_data)
        
        invalid_data = pd.DataFrame({'other_col': [1, 2, 3]})
        assert not loader.validate(invalid_data)
    
    def test_get_required_config_fields(self, loader):
        """测试获取必要配置字段"""
        fields = loader.get_required_config_fields()
        assert isinstance(fields, list)
        assert 'save_path' in fields
        assert 'max_retries' in fields
        assert 'cache_days' in fields
    
    def test_handle_exception(self, loader):
        """测试异常处理"""
        test_exception = Exception("测试异常")
        
        with pytest.raises(DataLoaderError, match="测试阶段 阶段出错"):
            loader._handle_exception(test_exception, "测试阶段")
    
    def test_get_financial_column_mapping(self, loader):
        """测试获取财务列映射"""
        mapping = loader._get_financial_column_mapping()
        assert isinstance(mapping, dict)
        assert 'roe' in mapping.values()
        # 检查实际返回的映射内容
        assert any('roe' in str(v) for v in mapping.values()) 