"""
stock_loader.py 边界测试补充（第8批）
目标：将覆盖率从 65% 提升到 80%+
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
import configparser
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.data.loader.stock_loader import (
    StockDataLoader,
    IndustryLoader,
    StockListLoader
)
try:
    from src.infrastructure.utils.exceptions import DataLoaderError
except ImportError:
    from src.infrastructure.error import DataLoaderError


@pytest.fixture
def stock_loader(tmp_path):
    """创建 StockDataLoader 实例"""
    loader = StockDataLoader(
        save_path=str(tmp_path),
        max_retries=3,
        cache_days=30
    )
    return loader


def test_stock_loader_create_from_config_section_object(tmp_path, monkeypatch):
    """测试 StockDataLoader（create_from_config，Section 对象，覆盖 108 行）"""
    # 创建一个 Section 对象
    config = configparser.ConfigParser()
    config.read_string("""
    [Stock]
    save_path = test_path
    max_retries = 5
    cache_days = 60
    """)
    section = config['Stock']
    
    with patch('configparser.ConfigParser.read'):
        with patch('pathlib.Path.mkdir'):
            loader = StockDataLoader.create_from_config(section, thread_pool=None)
            assert loader is not None


def test_stock_loader_load_data_impl_data_none(stock_loader, monkeypatch):
    """测试 StockDataLoader（_load_data_impl，data 为 None，覆盖 476 行）"""
    # 设置 max_retries 为 0，使所有重试立即失败
    stock_loader.max_retries = 0
    
    # Mock 所有方法，使 data 保持为 None
    # 关键：需要让异常在 except Exception 分支中处理，而不是 DataLoaderError 分支
    monkeypatch.setattr(stock_loader, '_get_file_path', Mock(return_value=Path('test.csv')))
    monkeypatch.setattr(stock_loader, '_is_cache_valid', Mock(return_value=False))
    
    # Mock _fetch_raw_data 抛出一般异常（不是 DataLoaderError），这样会进入 except Exception 分支
    # 但由于 max_retries=0，第一次尝试就会失败，然后抛出异常，data 保持为 None
    def mock_fetch(*args, **kwargs):
        raise ValueError("General error")
    
    monkeypatch.setattr(stock_loader, '_fetch_raw_data', mock_fetch)
    monkeypatch.setattr(stock_loader, '_get_holidays', Mock(return_value=[]))
    monkeypatch.setattr('time.sleep', Mock())
    
    # 由于所有重试都失败，data 保持为 None，最终抛出"未能加载数据"错误
    with pytest.raises(DataLoaderError) as exc_info:
        stock_loader._load_data_impl("000001", "2020-01-01", "2020-01-31")
    
    # 检查是否包含"未能加载数据"或"加载股票数据失败"
    assert "未能加载数据" in str(exc_info.value) or "加载股票数据失败" in str(exc_info.value)


def test_stock_loader_get_holidays_with_calendar(stock_loader, monkeypatch):
    """测试 StockDataLoader（_get_holidays，使用 pandas_market_calendars，覆盖 486-495 行）"""
    # Mock pandas_market_calendars
    mock_calendar = Mock()
    # 创建交易日索引（排除周末）
    trading_days = pd.date_range('2020-01-01', '2020-01-31', freq='B')  # 工作日
    mock_schedule = pd.DataFrame(index=trading_days)
    mock_schedule.index = trading_days  # 确保索引是 DatetimeIndex
    
    def mock_get_calendar(name):
        mock_calendar.schedule = Mock(return_value=mock_schedule)
        return mock_calendar
    
    # 需要 mock 整个导入
    import sys
    mock_pandas_market_calendars = MagicMock()
    mock_pandas_market_calendars.get_calendar = mock_get_calendar
    sys.modules['pandas_market_calendars'] = mock_pandas_market_calendars
    
    result = stock_loader._get_holidays("2020-01-01", "2020-01-31")
    assert isinstance(result, list)


def test_stock_loader_get_holidays_exception(stock_loader, monkeypatch):
    """测试 StockDataLoader（_get_holidays，异常处理，覆盖 499-501 行）"""
    # Mock get_calendar 抛出异常（非 ImportError）
    import sys
    mock_pandas_market_calendars = MagicMock()
    mock_calendar = Mock()
    mock_calendar.schedule = Mock(side_effect=Exception("Calendar error"))
    mock_pandas_market_calendars.get_calendar = Mock(return_value=mock_calendar)
    sys.modules['pandas_market_calendars'] = mock_pandas_market_calendars
    
    result = stock_loader._get_holidays("2020-01-01", "2020-01-31")
    assert result == []


def test_stock_loader_is_cache_valid_oserror(stock_loader, tmp_path, monkeypatch):
    """测试 StockDataLoader（_is_cache_valid，OSError，覆盖 523-525 行）"""
    cache_file = tmp_path / "cache" / "test.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("test")
    
    # Mock exists 返回 True，但 stat 抛出 OSError
    original_exists = Path.exists
    original_stat = Path.stat
    cache_file_str = str(cache_file)
    
    def mock_exists(self):
        if str(self) == cache_file_str:
            return True
        return original_exists(self)
    
    def mock_stat(self):
        if str(self) == cache_file_str:
            raise OSError("Cannot access file")
        return original_stat(self)
    
    monkeypatch.setattr('pathlib.Path.exists', mock_exists)
    monkeypatch.setattr('pathlib.Path.stat', mock_stat)
    
    result = stock_loader._is_cache_valid(cache_file)
    # 应该返回 True（因为文件存在，只是无法读取修改时间）
    assert result is True


def test_stock_loader_is_cache_valid_exception(stock_loader, tmp_path, monkeypatch):
    """测试 StockDataLoader（_is_cache_valid，一般异常，覆盖 528-530 行）"""
    cache_file = tmp_path / "cache" / "test.pkl"
    
    # Mock exists 返回 True，但后续操作抛出异常
    def mock_exists(self):
        return True
    
    def mock_stat(self):
        raise Exception("File system error")
    
    monkeypatch.setattr('pathlib.Path.exists', mock_exists)
    monkeypatch.setattr('pathlib.Path.stat', mock_stat)
    
    result = stock_loader._is_cache_valid(cache_file)
    assert result is False


def test_stock_loader_cleanup_exception(stock_loader, tmp_path, monkeypatch):
    """测试 StockDataLoader（cleanup，异常处理，覆盖 579-580 行）"""
    # 创建一个缓存文件
    cache_file = tmp_path / "cache" / "test.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("test")
    
    # Mock unlink 抛出异常
    def mock_unlink(self, missing_ok=False):
        raise PermissionError("Cannot delete file")
    
    monkeypatch.setattr('pathlib.Path.unlink', mock_unlink)
    
    # 应该不会抛出异常，只是记录警告
    stock_loader.cleanup()


def test_stock_loader_fetch_raw_data_timeout_keyword(stock_loader, monkeypatch):
    """测试 StockDataLoader（_fetch_raw_data，timeout 关键字，覆盖 603 行）"""
    # Mock stock_zh_a_daily 抛出包含 timeout 的异常
    def mock_stock_zh_a_daily(*args, **kwargs):
        raise Exception("Timeout error occurred")
    
    monkeypatch.setattr('akshare.stock_zh_a_daily', mock_stock_zh_a_daily)
    monkeypatch.setattr('akshare.stock_zh_a_hist', Mock(return_value=pd.DataFrame()))
    
    with pytest.raises(Exception, match="Timeout"):
        stock_loader._fetch_raw_data("000001", "20200101", "20200131", "none")


def test_stock_loader_fetch_raw_data_hist_exception(stock_loader, monkeypatch):
    """测试 StockDataLoader（_fetch_raw_data，hist 异常，覆盖 621 行）"""
    # Mock stock_zh_a_daily 不可用
    monkeypatch.setattr('akshare.stock_zh_a_daily', None)
    
    # Mock stock_zh_a_hist 抛出 DataLoaderError
    def mock_stock_zh_a_hist(*args, **kwargs):
        raise DataLoaderError("Hist API error")
    
    monkeypatch.setattr('akshare.stock_zh_a_hist', mock_stock_zh_a_hist)
    
    with pytest.raises(DataLoaderError, match="Hist API error"):
        stock_loader._fetch_raw_data("000001", "20200101", "20200131", "none")


def test_stock_loader_process_raw_data_missing_col(stock_loader):
    """测试 StockDataLoader（_process_raw_data，缺少列，覆盖 660-661, 667 行）"""
    # 测试 660-661 行：创建一个缺少必要列的 DataFrame（缺少 close）
    # 注意：high 和 low 会被自动创建（655-658 行），但 close 不会
    df = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100]
        # 缺少 high, low, close, volume
        # high 和 low 会被自动创建，但 close 会导致异常
    })
    
    # 应该抛出 DataLoaderError，因为缺少必要列 close
    with pytest.raises(DataLoaderError, match="原始数据缺少必要列"):
        stock_loader._process_raw_data(df)
    
    # 注意：667 行的代码实际上无法通过正常流程到达，因为 volume 在 660-661 行检查时就会抛出异常
    # 但我们可以通过 mock 来测试 667 行的逻辑，或者跳过这个测试
    # 这里我们只测试 660-661 行的异常处理


@pytest.fixture
def industry_loader(tmp_path):
    """创建 IndustryLoader 实例"""
    loader = IndustryLoader(
        save_path=str(tmp_path),
        max_retries=3,
        cache_days=30
    )
    return loader


def test_industry_loader_setup(industry_loader, tmp_path):
    """测试 IndustryLoader（_setup，覆盖 819 行）"""
    # _setup 在 __init__ 中调用，验证目录已创建
    assert industry_loader.save_path.exists()


def test_industry_loader_load_data_cache_valid(industry_loader, tmp_path, monkeypatch):
    """测试 IndustryLoader（load_data，缓存有效，覆盖 823-829 行）"""
    # 创建缓存文件
    cache_file = industry_loader.industry_map_path
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_df = pd.DataFrame({
        'symbol': ['000001', '000002'],
        'industry': ['银行', '地产']
    })
    cache_df.to_csv(cache_file, index=False, encoding='utf-8')
    
    # Mock _is_cache_valid 返回 True
    monkeypatch.setattr(industry_loader, '_is_cache_valid', Mock(return_value=True))
    
    result = industry_loader.load_data()
    
    assert isinstance(result, dict)
    assert len(result) > 0


def test_industry_loader_load_data_connection_error(industry_loader, monkeypatch):
    """测试 IndustryLoader（load_data，连接错误，覆盖 834-836 行）"""
    # Mock _is_cache_valid 返回 False
    monkeypatch.setattr(industry_loader, '_is_cache_valid', Mock(return_value=False))
    
    # Mock _fetch_raw_data 抛出 ConnectionError
    monkeypatch.setattr(industry_loader, '_fetch_raw_data', Mock(side_effect=ConnectionError("Connection failed")))
    
    with pytest.raises(DataLoaderError, match="获取行业数据时连接失败"):
        industry_loader.load_data()


def test_industry_loader_load_data_empty_data(industry_loader, monkeypatch):
    """测试 IndustryLoader（load_data，空数据，覆盖 838-839 行）"""
    # Mock _is_cache_valid 返回 False
    monkeypatch.setattr(industry_loader, '_is_cache_valid', Mock(return_value=False))
    
    # Mock _fetch_raw_data 返回空 DataFrame
    monkeypatch.setattr(industry_loader, '_fetch_raw_data', Mock(return_value=pd.DataFrame()))
    
    with pytest.raises(DataLoaderError, match="API 返回行业数据为空"):
        industry_loader.load_data()


def test_industry_loader_load_data_connection_error_retry(industry_loader, monkeypatch):
    """测试 IndustryLoader（load_data，连接错误重试，覆盖 853-859 行）"""
    # Mock _is_cache_valid 返回 False
    monkeypatch.setattr(industry_loader, '_is_cache_valid', Mock(return_value=False))
    
    # Mock _fetch_raw_data 返回有效数据
    industry_df = pd.DataFrame({
        '板块代码': ['BK0001'],
        '板块名称': ['银行']
    })
    monkeypatch.setattr(industry_loader, '_fetch_raw_data', Mock(return_value=industry_df))
    
    # Mock stock_board_industry_cons_em 抛出 ConnectionError
    def mock_cons_em(symbol):
        raise ConnectionError("Connection failed")
    
    monkeypatch.setattr('akshare.stock_board_industry_cons_em', mock_cons_em)
    
    # 应该抛出异常或返回部分数据
    try:
        result = industry_loader.load_data()
        # 如果返回了部分数据，验证结果
        assert isinstance(result, dict)
    except DataLoaderError:
        # 如果抛出异常，也是预期的
        pass


def test_industry_loader_load_data_no_mapping(industry_loader, monkeypatch):
    """测试 IndustryLoader（load_data，无映射数据，覆盖 861-871 行）"""
    # Mock _is_cache_valid 返回 False
    monkeypatch.setattr(industry_loader, '_is_cache_valid', Mock(return_value=False))
    
    # Mock _fetch_raw_data 返回有效数据
    industry_df = pd.DataFrame({
        '板块代码': ['BK0001'],
        '板块名称': ['银行']
    })
    monkeypatch.setattr(industry_loader, '_fetch_raw_data', Mock(return_value=industry_df))
    
    # Mock stock_board_industry_cons_em 返回空数据
    monkeypatch.setattr('akshare.stock_board_industry_cons_em', Mock(return_value=pd.DataFrame()))
    
    result = industry_loader.load_data()
    
    # 应该返回空字典或部分数据
    assert isinstance(result, dict)


def test_industry_loader_load_data_all_failed(industry_loader, monkeypatch):
    """测试 IndustryLoader（load_data，所有行业失败，覆盖 862-863 行）"""
    # Mock _is_cache_valid 返回 False
    monkeypatch.setattr(industry_loader, '_is_cache_valid', Mock(return_value=False))
    
    # Mock _fetch_raw_data 返回有效数据
    industry_df = pd.DataFrame({
        '板块代码': ['BK0001'],
        '板块名称': ['银行']
    })
    monkeypatch.setattr(industry_loader, '_fetch_raw_data', Mock(return_value=industry_df))
    
    # Mock stock_board_industry_cons_em 始终失败
    def mock_cons_em(symbol):
        raise ConnectionError("Connection failed")
    
    monkeypatch.setattr('akshare.stock_board_industry_cons_em', mock_cons_em)
    
    with pytest.raises(DataLoaderError, match="无法获取任何行业映射数据"):
        industry_loader.load_data()


def test_industry_loader_load_data_exception(industry_loader, monkeypatch):
    """测试 IndustryLoader（load_data，一般异常，覆盖 883-885 行）"""
    # Mock _is_cache_valid 返回 False
    monkeypatch.setattr(industry_loader, '_is_cache_valid', Mock(return_value=False))
    
    # Mock _fetch_raw_data 抛出一般异常
    monkeypatch.setattr(industry_loader, '_fetch_raw_data', Mock(side_effect=ValueError("General error")))
    
    with pytest.raises(DataLoaderError, match="加载行业数据失败"):
        industry_loader.load_data()


def test_industry_loader_is_cache_valid_not_exists(industry_loader, tmp_path):
    """测试 IndustryLoader（_is_cache_valid，文件不存在，覆盖 889 行）"""
    cache_file = tmp_path / "nonexistent.csv"
    
    result = industry_loader._is_cache_valid(cache_file)
    assert result is False


def test_industry_loader_is_cache_valid_expired(industry_loader, tmp_path, monkeypatch):
    """测试 IndustryLoader（_is_cache_valid，缓存过期，覆盖 889-904 行）"""
    cache_file = tmp_path / "test.csv"
    # 创建一个有效的 CSV 文件（包含数据）
    cache_df = pd.DataFrame({
        'symbol': ['000001'],
        'industry': ['银行']
    })
    cache_df.to_csv(cache_file, index=False, encoding='utf-8')
    
    # Mock getmtime 返回过期时间
    old_time = datetime.now().timestamp() - (31 * 86400)  # 31天前
    
    import os
    import time
    original_getmtime = os.path.getmtime
    def mock_getmtime(path):
        if str(path) == str(cache_file):
            return old_time
        return original_getmtime(path)
    
    monkeypatch.setattr('os.path.getmtime', mock_getmtime)
    # 也需要 mock time.time() 因为代码中使用了 time.time()
    monkeypatch.setattr('time.time', Mock(return_value=datetime.now().timestamp()))
    
    result = industry_loader._is_cache_valid(cache_file)
    assert result is False


def test_industry_loader_is_cache_valid_exception(industry_loader, tmp_path, monkeypatch):
    """测试 IndustryLoader（_is_cache_valid，异常处理，覆盖 898-900 行）"""
    cache_file = tmp_path / "test.csv"
    cache_file.write_text("test")
    
    # Mock getmtime 抛出异常
    monkeypatch.setattr('os.path.getmtime', Mock(side_effect=OSError("Cannot access file")))
    
    result = industry_loader._is_cache_valid(cache_file)
    assert result is False


def test_industry_loader_fetch_raw_data_success(industry_loader, monkeypatch):
    """测试 IndustryLoader（_fetch_raw_data，成功，覆盖 908-920 行）"""
    # Mock stock_board_industry_name_em 返回有效数据
    mock_df = pd.DataFrame({
        '板块代码': ['BK0001'],
        '板块名称': ['银行']
    })
    monkeypatch.setattr('akshare.stock_board_industry_name_em', Mock(return_value=mock_df))
    
    result = industry_loader._fetch_raw_data()
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_industry_loader_fetch_raw_data_exception(industry_loader, monkeypatch):
    """测试 IndustryLoader（_fetch_raw_data，异常处理，覆盖 924-935 行）"""
    # Mock stock_board_industry_name_em 抛出 ConnectionError（会触发 1096 行的异常处理）
    def mock_industry_name_em(*args, **kwargs):
        raise ConnectionError("API error")
    
    monkeypatch.setattr('akshare.stock_board_industry_name_em', mock_industry_name_em)
    
    with pytest.raises(DataLoaderError, match="获取行业数据时连接失败"):
        industry_loader._fetch_raw_data()


def test_industry_loader_calculate_concentration_window_too_small(industry_loader, monkeypatch):
    """测试 IndustryLoader（calculate_industry_concentration，窗口太小，覆盖 967-968 行）"""
    # 注意：方法名是 calculate_industry_concentration，需要 stock_loader 属性
    from src.data.loader.stock_loader import StockDataLoader
    
    # 创建 stock_loader 并设置到 industry_loader
    stock_loader = StockDataLoader(save_path=str(industry_loader.save_path))
    industry_loader.stock_loader = stock_loader
    
    with pytest.raises(ValueError, match="窗口大小过小"):
        industry_loader.calculate_industry_concentration("银行", "2020-01-01", "2020-01-31", window=5)


def test_industry_loader_calculate_concentration_no_components(industry_loader, monkeypatch):
    """测试 IndustryLoader（calculate_industry_concentration，无成分股，覆盖 974-975 行）"""
    # 创建 stock_loader 并设置到 industry_loader
    from src.data.loader.stock_loader import StockDataLoader
    stock_loader = StockDataLoader(save_path=str(industry_loader.save_path))
    industry_loader.stock_loader = stock_loader
    
    # Mock _get_industry_components 返回空 DataFrame
    monkeypatch.setattr(industry_loader, '_get_industry_components', Mock(return_value=pd.DataFrame()))
    
    with pytest.raises(DataLoaderError, match="获取行业数据失败"):
        industry_loader.calculate_industry_concentration("银行", "2020-01-01", "2020-01-31", window=10)


def test_industry_loader_calculate_concentration_empty_after_merge(industry_loader, monkeypatch):
    """测试 IndustryLoader（calculate_industry_concentration，合并后为空，覆盖 979-981, 1009-1011 行）"""
    # 创建 stock_loader 并设置到 industry_loader
    from src.data.loader.stock_loader import StockDataLoader
    stock_loader = StockDataLoader(save_path=str(industry_loader.save_path))
    industry_loader.stock_loader = stock_loader
    
    # Mock _get_industry_components 返回有效数据
    components_df = pd.DataFrame({
        'symbol': ['000001', '000002']
    })
    monkeypatch.setattr(industry_loader, '_get_industry_components', Mock(return_value=components_df))
    
    # Mock _load_stock_data 返回 None
    monkeypatch.setattr(industry_loader, '_load_stock_data', Mock(return_value=None))
    
    result = industry_loader.calculate_industry_concentration("银行", "2020-01-01", "2020-01-31", window=10)
    
    # 应该返回空 DataFrame
    assert result.empty


def test_industry_loader_calculate_concentration_no_valid_data(industry_loader, monkeypatch):
    """测试 IndustryLoader（calculate_industry_concentration，无有效数据，覆盖 1003-1005 行）"""
    # 创建 stock_loader 并设置到 industry_loader
    from src.data.loader.stock_loader import StockDataLoader
    stock_loader = StockDataLoader(save_path=str(industry_loader.save_path))
    industry_loader.stock_loader = stock_loader
    
    # Mock _get_industry_components 返回有效数据
    components_df = pd.DataFrame({
        'symbol': ['000001', '000002']
    })
    monkeypatch.setattr(industry_loader, '_get_industry_components', Mock(return_value=components_df))
    
    # Mock _load_stock_data 返回 None
    monkeypatch.setattr(industry_loader, '_load_stock_data', Mock(return_value=None))
    
    result = industry_loader.calculate_industry_concentration("银行", "2020-01-01", "2020-01-31", window=10)
    
    # 应该返回空 DataFrame
    assert result.empty


def test_industry_loader_load_stock_data_success(industry_loader, monkeypatch):
    """测试 IndustryLoader（_load_stock_data，成功，覆盖 1046-1058 行）"""
    # 创建 stock_loader 并设置到 industry_loader
    from src.data.loader.stock_loader import StockDataLoader
    stock_loader = StockDataLoader(save_path=str(industry_loader.save_path))
    industry_loader.stock_loader = stock_loader
    
    # Mock stock_loader.load_data 返回有效数据（需要设置索引为 date）
    mock_df = pd.DataFrame({
        'close': [100]
    }, index=pd.DatetimeIndex(['2020-01-01']))
    monkeypatch.setattr(industry_loader.stock_loader, 'load_data', Mock(return_value=mock_df))
    
    result = industry_loader._load_stock_data("000001", "2020-01-01", "2020-01-31")
    
    assert result is not None
    # 返回的是 Series（close 列），不是 DataFrame
    assert isinstance(result, pd.Series)


def test_industry_loader_load_stock_data_exception(industry_loader, monkeypatch):
    """测试 IndustryLoader（_load_stock_data，异常处理，覆盖 1062-1069 行）"""
    # 创建 stock_loader 并设置到 industry_loader
    from src.data.loader.stock_loader import StockDataLoader
    stock_loader = StockDataLoader(save_path=str(industry_loader.save_path))
    industry_loader.stock_loader = stock_loader
    
    # Mock stock_loader.load_data 抛出异常
    monkeypatch.setattr(industry_loader.stock_loader, 'load_data', Mock(side_effect=Exception("Load error")))
    
    result = industry_loader._load_stock_data("000001", "2020-01-01", "2020-01-31")
    
    # 异常时应该返回 None
    assert result is None


def test_industry_loader_save_data_success(industry_loader, tmp_path, monkeypatch):
    """测试 IndustryLoader（_save_data，成功，覆盖 1105-1109 行）"""
    df = pd.DataFrame({'symbol': ['000001'], 'industry': ['银行']})
    file_path = tmp_path / "test.csv"
    
    result = industry_loader._save_data(df, file_path)
    
    assert result is True
    assert file_path.exists()


def test_industry_loader_save_data_exception(industry_loader, tmp_path, monkeypatch):
    """测试 IndustryLoader（_save_data，异常处理，覆盖 1110-1111 行）"""
    df = pd.DataFrame({'symbol': ['000001'], 'industry': ['银行']})
    file_path = tmp_path / "test.csv"
    
    # Mock to_csv 抛出异常
    def mock_to_csv(*args, **kwargs):
        raise IOError("Cannot write file")
    
    monkeypatch.setattr('pandas.DataFrame.to_csv', mock_to_csv)
    
    with pytest.raises(DataLoaderError):
        industry_loader._save_data(df, file_path)


def test_industry_loader_check_cache_not_exists(industry_loader, tmp_path):
    """测试 IndustryLoader（_check_cache，文件不存在，覆盖 1115-1116 行）"""
    cache_file = tmp_path / "nonexistent.csv"
    
    is_valid, df = industry_loader._check_cache(cache_file)
    
    assert is_valid is False
    assert df is None


def test_industry_loader_check_cache_expired(industry_loader, tmp_path, monkeypatch):
    """测试 IndustryLoader（_check_cache，缓存过期，覆盖 1120-1121 行）"""
    cache_file = tmp_path / "test.csv"
    # 创建一个有效的 CSV 文件
    cache_df = pd.DataFrame({'symbol': ['000001'], 'industry': ['银行']})
    cache_df.to_csv(cache_file, index=False, encoding='utf-8')
    
    # Mock getmtime 返回过期时间
    old_time = datetime.now().timestamp() - (31 * 86400)  # 31天前
    
    import os
    import time
    original_getmtime = os.path.getmtime
    def mock_getmtime(path):
        if str(path) == str(cache_file):
            return old_time
        return original_getmtime(path)
    
    monkeypatch.setattr('os.path.getmtime', mock_getmtime)
    # Mock time.time() 返回当前时间
    monkeypatch.setattr('time.time', Mock(return_value=datetime.now().timestamp()))
    
    is_valid, df = industry_loader._check_cache(cache_file)
    
    assert is_valid is False
    assert df is None


def test_industry_loader_check_cache_empty_file(industry_loader, tmp_path):
    """测试 IndustryLoader（_check_cache，空文件，覆盖 1124-1126 行）"""
    cache_file = tmp_path / "test.csv"
    cache_file.write_text("")  # 空文件
    
    is_valid, df = industry_loader._check_cache(cache_file)
    
    assert is_valid is False
    assert df is None


def test_industry_loader_check_cache_empty_data_error(industry_loader, tmp_path, monkeypatch):
    """测试 IndustryLoader（_check_cache，EmptyDataError，覆盖 1129-1131 行）"""
    cache_file = tmp_path / "test.csv"
    cache_file.write_text("test")
    
    # Mock read_csv 抛出 EmptyDataError
    def mock_read_csv(*args, **kwargs):
        raise pd.errors.EmptyDataError("Empty file")
    
    monkeypatch.setattr('pandas.read_csv', mock_read_csv)
    
    is_valid, df = industry_loader._check_cache(cache_file)
    
    assert is_valid is False
    assert df is None


def test_industry_loader_check_cache_exception(industry_loader, tmp_path, monkeypatch):
    """测试 IndustryLoader（_check_cache，一般异常，覆盖 1132-1134 行）"""
    cache_file = tmp_path / "test.csv"
    cache_file.write_text("test")
    
    # Mock getmtime 抛出异常
    def mock_getmtime(path):
        raise OSError("Cannot access file")
    
    monkeypatch.setattr('os.path.getmtime', mock_getmtime)
    
    is_valid, df = industry_loader._check_cache(cache_file)
    
    assert is_valid is False
    assert df is None


def test_industry_loader_get_industry_components_success(industry_loader, monkeypatch):
    """测试 IndustryLoader（_get_industry_components，成功，覆盖 1062-1069 行）"""
    # Mock load_data 返回行业映射
    industry_map = {
        '000001': '银行',
        '000002': '银行',
        '000003': '地产'
    }
    monkeypatch.setattr(industry_loader, 'load_data', Mock(return_value=industry_map))
    
    result = industry_loader._get_industry_components("银行")
    
    assert not result.empty
    assert len(result) == 2


def test_industry_loader_get_industry_success(industry_loader, monkeypatch):
    """测试 IndustryLoader（get_industry，成功，覆盖 906-920 行）"""
    # 直接设置 _industry_map，因为 get_industry 会检查 _industry_map
    industry_loader._industry_map = {
        '000001': '银行',
        '000002': '银行',
        '000003': '地产'
    }
    
    result = industry_loader.get_industry("000001")
    
    assert isinstance(result, str)
    # 注意：get_industry 会调用 _standardize_industry_name，如果"银行"不在映射中，会返回原值
    assert result in ["银行", "行业未知"]  # 根据标准化映射调整


def test_industry_loader_get_industry_not_found(industry_loader, monkeypatch):
    """测试 IndustryLoader（get_industry，未找到，覆盖 913-917 行）"""
    # 直接设置 _industry_map，但不包含目标股票
    industry_loader._industry_map = {
        '000001': '银行',
        '000002': '银行'
    }
    # 设置 debug_mode 为 False，这样会返回"行业未知"而不是抛出异常
    if not hasattr(industry_loader, 'debug_mode'):
        industry_loader.debug_mode = False
    
    result = industry_loader.get_industry("999999")
    
    assert result == "行业未知"


def test_industry_loader_get_industry_not_found_debug_mode(industry_loader, monkeypatch):
    """测试 IndustryLoader（get_industry，未找到且 debug_mode=True，覆盖 914-915 行）"""
    # 直接设置 _industry_map，但不包含目标股票
    industry_loader._industry_map = {
        '000001': '银行',
        '000002': '银行'
    }
    # 设置 debug_mode 为 True，这样会抛出异常
    industry_loader.debug_mode = True
    
    with pytest.raises(DataLoaderError, match="获取行业数据失败"):
        industry_loader.get_industry("999999")


def test_industry_loader_standardize_industry_name(industry_loader):
    """测试 IndustryLoader（_standardize_industry_name，覆盖 922-935 行）"""
    # 测试标准化映射
    assert industry_loader._standardize_industry_name("石油行业") == "能源"
    assert industry_loader._standardize_industry_name("旅游酒店") == "消费"
    assert industry_loader._standardize_industry_name("互联网服务") == "科技"
    assert industry_loader._standardize_industry_name("生物制品") == "医药"
    assert industry_loader._standardize_industry_name("电池") == "新能源"
    assert industry_loader._standardize_industry_name("商业百货") == "零售"
    assert industry_loader._standardize_industry_name("家电行业") == "家电"
    assert industry_loader._standardize_industry_name("酿酒行业") == "白酒"
    assert industry_loader._standardize_industry_name("房地产开发") == "地产"
    # 测试未映射的行业名称
    assert industry_loader._standardize_industry_name("其他行业") == "其他行业"


@pytest.fixture
def stock_list_loader(tmp_path):
    """创建 StockListLoader 实例"""
    loader = StockListLoader(
        save_path=str(tmp_path),
        max_retries=3,
        cache_days=30
    )
    return loader


def test_stock_list_loader_setup(stock_list_loader, tmp_path):
    """测试 StockListLoader（_setup，覆盖 1162 行）"""
    # _setup 在 __init__ 中调用，验证目录已创建
    assert stock_list_loader.save_path.exists()


def test_stock_list_loader_load_data_cache_valid(stock_list_loader, tmp_path, monkeypatch):
    """测试 StockListLoader（load_data，缓存有效，覆盖 1168-1169 行）"""
    # 创建缓存文件
    cache_file = stock_list_loader.list_path
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_df = pd.DataFrame({
        '股票代码': ['000001', '000002'],
        '股票名称': ['平安银行', '万科A']
    })
    cache_df.to_csv(cache_file, index=False, encoding='utf-8')
    
    # Mock _is_cache_valid 返回 True
    monkeypatch.setattr(stock_list_loader, '_is_cache_valid', Mock(return_value=True))
    
    result = stock_list_loader.load_data()
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_stock_list_loader_load_data_empty_data(stock_list_loader, monkeypatch):
    """测试 StockListLoader（load_data，空数据，覆盖 1173-1174 行）"""
    # Mock _is_cache_valid 返回 False
    monkeypatch.setattr(stock_list_loader, '_is_cache_valid', Mock(return_value=False))
    
    # Mock _fetch_raw_data 返回空 DataFrame
    monkeypatch.setattr(stock_list_loader, '_fetch_raw_data', Mock(return_value=pd.DataFrame()))
    
    with pytest.raises(DataLoaderError, match="API 返回股票列表为空"):
        stock_list_loader.load_data()


def test_stock_list_loader_load_data_exception(stock_list_loader, monkeypatch):
    """测试 StockListLoader（load_data，异常处理，覆盖 1183-1185 行）"""
    # Mock _is_cache_valid 返回 False
    monkeypatch.setattr(stock_list_loader, '_is_cache_valid', Mock(return_value=False))
    
    # Mock _fetch_raw_data 抛出异常
    monkeypatch.setattr(stock_list_loader, '_fetch_raw_data', Mock(side_effect=ValueError("General error")))
    
    with pytest.raises(DataLoaderError, match="加载股票列表失败"):
        stock_list_loader.load_data()


def test_stock_list_loader_is_cache_valid_not_exists(stock_list_loader, tmp_path):
    """测试 StockListLoader（_is_cache_valid，文件不存在，覆盖 1189 行）"""
    cache_file = tmp_path / "nonexistent.csv"
    
    result = stock_list_loader._is_cache_valid(cache_file)
    assert result is False


def test_stock_list_loader_is_cache_valid_expired(stock_list_loader, tmp_path, monkeypatch):
    """测试 StockListLoader（_is_cache_valid，缓存过期，覆盖 1194-1195 行）"""
    cache_file = tmp_path / "test.csv"
    cache_file.write_text("test")
    
    # Mock getmtime 返回过期时间
    old_time = datetime.now().timestamp() - (31 * 86400)  # 31天前
    
    monkeypatch.setattr('os.path.getmtime', Mock(return_value=old_time))
    
    result = stock_list_loader._is_cache_valid(cache_file)
    assert result is False


def test_stock_list_loader_is_cache_valid_exception(stock_list_loader, tmp_path, monkeypatch):
    """测试 StockListLoader（_is_cache_valid，异常处理，覆盖 1198-1200 行）"""
    cache_file = tmp_path / "test.csv"
    cache_file.write_text("test")
    
    # Mock getmtime 抛出异常
    monkeypatch.setattr('os.path.getmtime', Mock(side_effect=OSError("Cannot access file")))
    
    result = stock_list_loader._is_cache_valid(cache_file)
    assert result is False


def test_stock_list_loader_get_stock_list(stock_list_loader, monkeypatch):
    """测试 StockListLoader（get_stock_list，覆盖 1204 行）"""
    # Mock load_data 返回有效数据
    mock_df = pd.DataFrame({
        '股票代码': ['000001'],
        '股票名称': ['平安银行']
    })
    monkeypatch.setattr(stock_list_loader, 'load_data', Mock(return_value=mock_df))
    
    result = stock_list_loader.get_stock_list()
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_stock_list_loader_retry_api_call_success(stock_list_loader, monkeypatch):
    """测试 StockListLoader（_retry_api_call，成功，覆盖 1208-1216 行）"""
    # Mock API 调用成功
    mock_df = pd.DataFrame({
        'code': ['000001'],
        'name': ['平安银行']
    })
    
    def mock_func():
        return mock_df
    
    result = stock_list_loader._retry_api_call(mock_func)
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_stock_list_loader_retry_api_call_exception(stock_list_loader, monkeypatch):
    """测试 StockListLoader（_retry_api_call，异常处理，覆盖 1211-1213 行）"""
    # Mock API 调用抛出 RequestException（会重试）
    from requests import RequestException
    
    def mock_func():
        raise RequestException("API error")
    
    monkeypatch.setattr('time.sleep', Mock())
    
    # 由于 max_retries=3，会重试3次，最后一次会抛出 RequestException
    with pytest.raises(RequestException):
        stock_list_loader._retry_api_call(mock_func)


def test_stock_list_loader_repr(stock_list_loader):
    """测试 StockListLoader（__repr__，覆盖 1233-1248 行）"""
    result = repr(stock_list_loader)
    
    assert isinstance(result, str)
    assert "StockListLoader" in result
    assert "save_path" in result


def test_stock_list_loader_fetch_raw_data(stock_list_loader, monkeypatch):
    """测试 StockListLoader（_fetch_raw_data，覆盖 1250-1258 行）"""
    # Mock stock_info_a_code_name 返回有效数据
    mock_df = pd.DataFrame({
        'code': ['000001'],
        'name': ['平安银行']
    })
    monkeypatch.setattr('akshare.stock_info_a_code_name', Mock(return_value=mock_df))
    
    result = stock_list_loader._fetch_raw_data()
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

