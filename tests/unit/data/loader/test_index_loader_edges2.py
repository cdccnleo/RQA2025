"""
index_loader.py 边界测试补充
目标：将覆盖率从 69% 提升到 80%+
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
import pickle
import configparser
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.data.loader.index_loader import IndexDataLoader
try:
    from src.infrastructure.error import DataLoaderError
except ImportError:
    from src.infrastructure.utils.exceptions import DataLoaderError


@pytest.fixture
def index_loader(tmp_path):
    """创建 IndexDataLoader 实例"""
    loader = IndexDataLoader(
        save_path=str(tmp_path),
        max_retries=3,
        cache_days=30
    )
    return loader


def test_index_loader_create_from_config_section_object(tmp_path, monkeypatch):
    """测试 IndexDataLoader（create_from_config，Section 对象，覆盖 83 行）"""
    # 创建一个 Section 对象
    config = configparser.ConfigParser()
    config.read_string("""
    [Index]
    save_path = test_path
    max_retries = 5
    cache_days = 60
    """)
    section = config['Index']
    
    with patch('configparser.ConfigParser.read'):
        loader = IndexDataLoader.create_from_config(section)
        assert loader is not None


def test_index_loader_create_from_config_safe_getint_exception(tmp_path, monkeypatch):
    """测试 IndexDataLoader（create_from_config，safe_getint 异常，覆盖 99, 104 行）"""
    config = {
        'Index': {
            'save_path': str(tmp_path),
            'max_retries': 'invalid_int',  # 无效的整数值
            'cache_days': '30'
        }
    }
    
    with patch('configparser.ConfigParser.read'):
        with pytest.raises(DataLoaderError, match="配置项 max_retries 的值无效"):
            IndexDataLoader.create_from_config(config)


def test_index_loader_load_data_empty_data(index_loader, monkeypatch):
    """测试 IndexDataLoader（load_data，API 返回空数据，覆盖 175 行）"""
    # Mock _fetch_raw_data 返回空 DataFrame
    monkeypatch.setattr(index_loader, '_fetch_raw_data', Mock(return_value=pd.DataFrame()))
    
    with pytest.raises(DataLoaderError, match="API返回的数据为空"):
        index_loader.load_data("HS300", "2020-01-01", "2020-01-31")


def test_index_loader_load_data_processed_empty(index_loader, monkeypatch):
    """测试 IndexDataLoader（load_data，处理后数据为空，覆盖 187 行）"""
    # Mock _fetch_raw_data 返回有效数据
    raw_df = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': [1000]
    })
    
    # Mock _process_raw_data 返回空 DataFrame
    monkeypatch.setattr(index_loader, '_fetch_raw_data', Mock(return_value=raw_df))
    monkeypatch.setattr(index_loader, '_process_raw_data', Mock(return_value=pd.DataFrame()))
    monkeypatch.setattr(index_loader, '_get_file_path', Mock(return_value=Path('test.csv')))
    
    with pytest.raises(DataLoaderError, match="处理后的指数数据为空"):
        index_loader.load_data("HS300", "2020-01-01", "2020-01-31")


def test_index_loader_load_data_retry_empty_data(index_loader, monkeypatch):
    """测试 IndexDataLoader（load_data，重试空数据，覆盖 196-199 行）"""
    # Mock _fetch_raw_data 第一次返回空，第二次返回有效数据
    raw_df = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': [1000]
    })
    
    call_count = [0]
    def mock_fetch(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return pd.DataFrame()  # 第一次返回空
        return raw_df  # 第二次返回有效数据
    
    monkeypatch.setattr(index_loader, '_fetch_raw_data', Mock(side_effect=mock_fetch))
    monkeypatch.setattr(index_loader, '_process_raw_data', Mock(return_value=raw_df))
    monkeypatch.setattr(index_loader, '_get_file_path', Mock(return_value=Path('test.csv')))
    monkeypatch.setattr(index_loader, '_save_data', Mock())
    
    result = index_loader.load_data("HS300", "2020-01-01", "2020-01-31")
    assert result is not None


def test_index_loader_load_data_connection_error(index_loader, monkeypatch):
    """测试 IndexDataLoader（load_data，连接错误，覆盖 202-207 行）"""
    # Mock _fetch_raw_data 抛出 ConnectionError
    monkeypatch.setattr(index_loader, '_fetch_raw_data', Mock(side_effect=ConnectionError("Connection failed")))
    monkeypatch.setattr(index_loader, '_get_file_path', Mock(return_value=Path('test.csv')))
    
    with pytest.raises(DataLoaderError, match="超过最大重试次数"):
        index_loader.load_data("HS300", "2020-01-01", "2020-01-31")


def test_index_loader_load_data_timeout_error(index_loader, monkeypatch):
    """测试 IndexDataLoader（load_data，超时错误，覆盖 202-207 行）"""
    # Mock _fetch_raw_data 抛出 TimeoutError
    monkeypatch.setattr(index_loader, '_fetch_raw_data', Mock(side_effect=TimeoutError("Timeout")))
    monkeypatch.setattr(index_loader, '_get_file_path', Mock(return_value=Path('test.csv')))
    
    with pytest.raises(DataLoaderError, match="超过最大重试次数"):
        index_loader.load_data("HS300", "2020-01-01", "2020-01-31")


def test_index_loader_load_data_general_exception(index_loader, monkeypatch):
    """测试 IndexDataLoader（load_data，一般异常，覆盖 208-210 行）"""
    # Mock _fetch_raw_data 抛出一般异常
    monkeypatch.setattr(index_loader, '_fetch_raw_data', Mock(side_effect=ValueError("General error")))
    monkeypatch.setattr(index_loader, '_get_file_path', Mock(return_value=Path('test.csv')))
    
    with pytest.raises(DataLoaderError, match="加载指数数据失败"):
        index_loader.load_data("HS300", "2020-01-01", "2020-01-31")


def test_index_loader_load_single_index_validation_failed(index_loader, monkeypatch):
    """测试 IndexDataLoader（load_single_index，验证失败，覆盖 244 行）"""
    # Mock _fetch_raw_data 和 _process_raw_data
    raw_df = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': [1000]
    })
    
    processed_df = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': [1000]
    })
    
    # Mock _validate_index_data 返回验证失败
    monkeypatch.setattr(index_loader, '_fetch_raw_data', Mock(return_value=raw_df))
    monkeypatch.setattr(index_loader, '_process_raw_data', Mock(return_value=processed_df))
    monkeypatch.setattr(index_loader, '_validate_index_data', Mock(return_value=(False, ["验证错误"])))
    monkeypatch.setattr(index_loader, '_get_cache_file_path', Mock(return_value=Path('test.pkl')))
    monkeypatch.setattr(index_loader, '_load_cache_payload', Mock(return_value=None))
    
    with pytest.raises(DataLoaderError, match="数据验证失败"):
        index_loader.load_single_index("HS300", "2020-01-01", "2020-01-31", force_refresh=True)


def test_index_loader_load_multiple_indexes_with_thread_pool(index_loader, monkeypatch):
    """测试 IndexDataLoader（load_multiple_indexes，使用线程池，覆盖 294-302 行）"""
    # 创建一个真实的线程池
    from concurrent.futures import ThreadPoolExecutor
    
    thread_pool = ThreadPoolExecutor(max_workers=2)
    index_loader.thread_pool = thread_pool
    
    # Mock _load_single_index_with_cache
    def mock_load(name):
        return {
            'data': pd.DataFrame(),
            'metadata': {}
        }
    
    monkeypatch.setattr(index_loader, '_load_single_index_with_cache', mock_load)
    
    result = index_loader.load_multiple_indexes(["HS300", "SZ50"], max_workers=2)
    
    assert len(result) == 2
    thread_pool.shutdown(wait=True)


def test_index_loader_is_cache_valid_empty_file(index_loader, tmp_path):
    """测试 IndexDataLoader（_is_cache_valid，空文件，覆盖 344-345 行）"""
    # 创建一个空文件
    cache_file = tmp_path / "cache" / "test.csv"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("")
    
    result = index_loader._is_cache_valid(cache_file)
    assert result is False


def test_index_loader_is_cache_valid_exception(index_loader, tmp_path):
    """测试 IndexDataLoader（_is_cache_valid，异常处理，覆盖 347-349 行）"""
    # 创建一个无效的文件路径
    invalid_path = tmp_path / "nonexistent" / "test.csv"
    
    result = index_loader._is_cache_valid(invalid_path)
    assert result is False


def test_index_loader_fetch_raw_data_empty(index_loader, monkeypatch):
    """测试 IndexDataLoader（_fetch_raw_data，空数据，覆盖 360 行）"""
    # Mock _retry_api_call 返回 None
    monkeypatch.setattr(index_loader, '_retry_api_call', Mock(return_value=None))
    
    with pytest.raises(DataLoaderError, match="API返回的数据为空或不存在"):
        index_loader._fetch_raw_data("000300", "20200101", "20200131")


def test_index_loader_process_raw_data_empty(index_loader):
    """测试 IndexDataLoader（_process_raw_data，空数据，覆盖 366 行）"""
    with pytest.raises(DataLoaderError, match="原始数据为空"):
        index_loader._process_raw_data(pd.DataFrame())


def test_index_loader_process_raw_data_missing_cols(index_loader):
    """测试 IndexDataLoader（_process_raw_data，缺少列，覆盖 393-394 行）"""
    # 创建一个缺少必要列的 DataFrame
    df = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100]
        # 缺少 high, low, close, volume
    })
    
    with pytest.raises(DataLoaderError, match="原始数据缺少必要列"):
        index_loader._process_raw_data(df)


def test_index_loader_process_raw_data_date_parse_error(index_loader, monkeypatch):
    """测试 IndexDataLoader（_process_raw_data，日期解析错误，覆盖 399-401 行）"""
    df = pd.DataFrame({
        'date': ['invalid_date'],
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': [1000]
    })
    
    # Mock pd.to_datetime 抛出异常
    def mock_to_datetime(*args, **kwargs):
        raise ValueError("Invalid date format")
    
    monkeypatch.setattr('pandas.to_datetime', mock_to_datetime)
    
    with pytest.raises(DataLoaderError, match="日期格式解析失败"):
        index_loader._process_raw_data(df)


def test_index_loader_retry_api_call_request_exception(index_loader, monkeypatch):
    """测试 IndexDataLoader（_retry_api_call，RequestException，覆盖 420-424 行）"""
    from requests import RequestException
    
    def mock_func(*args, **kwargs):
        raise RequestException("Request failed")
    
    monkeypatch.setattr('time.sleep', Mock())
    
    with pytest.raises(DataLoaderError):
        index_loader._retry_api_call(mock_func)


def test_index_loader_retry_api_call_connection_error(index_loader, monkeypatch):
    """测试 IndexDataLoader（_retry_api_call，ConnectionError，覆盖 420-424 行）"""
    def mock_func(*args, **kwargs):
        raise ConnectionError("Connection failed")
    
    monkeypatch.setattr('time.sleep', Mock())
    
    with pytest.raises(DataLoaderError):
        index_loader._retry_api_call(mock_func)


def test_index_loader_retry_api_call_general_exception(index_loader, monkeypatch):
    """测试 IndexDataLoader（_retry_api_call，一般异常，覆盖 425-429 行）"""
    def mock_func(*args, **kwargs):
        raise ValueError("General error")
    
    monkeypatch.setattr('time.sleep', Mock())
    
    with pytest.raises(DataLoaderError):
        index_loader._retry_api_call(mock_func)


def test_index_loader_merge_with_cache_missing_cols(index_loader, tmp_path):
    """测试 IndexDataLoader（_merge_with_cache，缺少列，覆盖 441-442 行）"""
    # 创建一个缺少必要列的缓存文件
    cache_file = tmp_path / "test.csv"
    cache_df = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100]
        # 缺少 high, low, close, volume
    })
    cache_df.to_csv(cache_file, index=False)
    
    new_df = pd.DataFrame({
        'date': ['2020-01-02'],
        'open': [110],
        'high': [120],
        'low': [100],
        'close': [115],
        'volume': [2000]
    })
    new_df.set_index('date', inplace=True)
    
    result = index_loader._merge_with_cache(cache_file, new_df)
    assert result is not None


def test_index_loader_merge_with_cache_read_error(index_loader, tmp_path):
    """测试 IndexDataLoader（_merge_with_cache，读取错误，覆盖 443-445 行）"""
    # 创建一个无效的缓存文件
    cache_file = tmp_path / "test.csv"
    cache_file.write_text("invalid csv content")
    
    new_df = pd.DataFrame({
        'date': ['2020-01-02'],
        'open': [110],
        'high': [120],
        'low': [100],
        'close': [115],
        'volume': [2000]
    })
    new_df.set_index('date', inplace=True)
    
    result = index_loader._merge_with_cache(cache_file, new_df)
    assert result is not None


def test_index_loader_load_cache_payload_exception(index_loader, tmp_path):
    """测试 IndexDataLoader（_load_cache_payload，异常处理，覆盖 482-484 行）"""
    # 创建一个无效的缓存文件
    cache_file = tmp_path / "test.pkl"
    cache_file.write_text("invalid pickle content")
    
    result = index_loader._load_cache_payload(cache_file)
    assert result is None


def test_index_loader_validate_index_data_volume_errors(index_loader):
    """测试 IndexDataLoader（_validate_index_data，volume 错误，覆盖 523 行）"""
    # 创建一个包含非数字 volume 的数据
    data = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': ['invalid']  # 非数字
    })
    
    is_valid, errors = index_loader._validate_index_data(data)
    assert is_valid is False
    assert len(errors) > 0


def test_index_loader_validate_index_data_negative_volume(index_loader):
    """测试 IndexDataLoader（_validate_index_data，负 volume，覆盖 533 行）"""
    # 创建一个包含负 volume 的数据
    data = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': [-1000]  # 负数
    })
    
    is_valid, errors = index_loader._validate_index_data(data)
    assert is_valid is False
    assert len(errors) > 0


def test_index_loader_cleanup_exception(index_loader, tmp_path, monkeypatch):
    """测试 IndexDataLoader（cleanup，异常处理，覆盖 537-538 行）"""
    # 创建一个缓存文件
    cache_file = tmp_path / "cache" / "test.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("test")
    
    # Mock unlink 抛出异常
    def mock_unlink(*args, **kwargs):
        raise PermissionError("Cannot delete file")
    
    monkeypatch.setattr('pathlib.Path.unlink', mock_unlink)
    
    # 应该不会抛出异常，只是记录警告
    index_loader.cleanup()


def test_index_loader_normalize_data_type_error(index_loader):
    """测试 IndexDataLoader（normalize_data，类型错误，覆盖 562 行）"""
    # normalize_data 会捕获 TypeError 并转换为 DataLoaderError
    with pytest.raises(DataLoaderError, match="数据标准化失败"):
        index_loader.normalize_data("not a dataframe")


def test_index_loader_normalize_data_empty(index_loader):
    """测试 IndexDataLoader（normalize_data，空数据，覆盖 565 行）"""
    # normalize_data 会捕获 ValueError 并转换为 DataLoaderError
    with pytest.raises(DataLoaderError, match="数据标准化失败"):
        index_loader.normalize_data(pd.DataFrame())


def test_index_loader_normalize_data_missing_cols(index_loader):
    """测试 IndexDataLoader（normalize_data，缺少列，覆盖 571 行）"""
    df = pd.DataFrame({
        'open': [100],
        'high': [110]
        # 缺少 low, close, volume
    })
    
    with pytest.raises(DataLoaderError, match="数据缺少必要列"):
        index_loader.normalize_data(df)


def test_index_loader_normalize_data_exception(index_loader, monkeypatch):
    """测试 IndexDataLoader（normalize_data，异常处理，覆盖 594-596 行）"""
    df = pd.DataFrame({
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': [1000]
    })
    
    # Mock StandardScaler 抛出异常
    def mock_standard_scaler(*args, **kwargs):
        raise Exception("Scaler error")
    
    monkeypatch.setattr('sklearn.preprocessing.StandardScaler', mock_standard_scaler)
    
    with pytest.raises(DataLoaderError, match="数据标准化失败"):
        index_loader.normalize_data(df)


def test_index_loader_save_data_missing_cols(index_loader, tmp_path):
    """测试 IndexDataLoader（_save_data，缺少列，覆盖 650-651 行）"""
    df = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100]
        # 缺少必要列
    })
    
    file_path = tmp_path / "test.csv"
    result = index_loader._save_data(df, file_path)
    assert result is False


def test_index_loader_save_data_date_parse_error(index_loader, tmp_path):
    """测试 IndexDataLoader（_save_data，日期解析错误，覆盖 655-659 行）"""
    df = pd.DataFrame({
        'date': ['invalid_date'],
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': [1000]
    })
    
    file_path = tmp_path / "test.csv"
    result = index_loader._save_data(df, file_path)
    assert result is False


def test_index_loader_save_data_exception(index_loader, tmp_path, monkeypatch):
    """测试 IndexDataLoader（_save_data，异常处理，覆盖 664-666 行）"""
    df = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100],
        'high': [110],
        'low': [90],
        'close': [105],
        'volume': [1000]
    })
    
    file_path = tmp_path / "test.csv"
    
    # Mock to_csv 抛出异常
    def mock_to_csv(*args, **kwargs):
        raise IOError("Cannot write file")
    
    monkeypatch.setattr('pandas.DataFrame.to_csv', mock_to_csv)
    
    result = index_loader._save_data(df, file_path)
    assert result is False

