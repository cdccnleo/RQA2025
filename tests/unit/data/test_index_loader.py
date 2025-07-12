# tests/data/test_index_loader.py
import numpy as np
import pytest
import pandas as pd
from unittest.mock import patch, Mock
import os
import time
from pathlib import Path

from requests import RequestException

from src.infrastructure.utils.logger import get_logger
from src.data.loader.index_loader import IndexDataLoader
from src.infrastructure.utils.exceptions import DataLoaderError
import akshare as ak

# 配置测试日志
logger = get_logger(__name__)

# 测试常量
TEST_INDEX = "HS300"  # 测试指数代码
START_DATE = "2022-01-01"
END_DATE = "2022-01-31"


@pytest.fixture(scope="function")
def index_loader(tmp_path):
    """初始化指数数据加载器"""
    return IndexDataLoader(
        save_path=tmp_path / "test_index",
        max_retries=2,
        cache_days=7
    )


def test_index_loader_invalid_index(index_loader):
    """使用无效指数代码"""
    invalid_index = "INVALID_INDEX"
    with pytest.raises(ValueError) as excinfo:
        index_loader.load_data(invalid_index, START_DATE, END_DATE)
    assert "Invalid index code" in str(excinfo.value)


def test_index_loader_invalid_date_range(index_loader):
    """使用无效日期范围（开始日期大于结束日期）"""
    start = "2022-02-01"
    end = "2022-01-01"
    with pytest.raises(ValueError) as excinfo:
        index_loader.load_data(TEST_INDEX, start, end)
    assert "开始日期不能大于结束日期" in str(excinfo.value)
    logger.debug("日期范围校验正常")


def test_index_loader_empty_cache(index_loader, monkeypatch):
    """测试空缓存加载"""

    # 模拟 _fetch_raw_data 返回空数据
    def mock_empty_api(*args, **kwargs):
        return pd.DataFrame()

    # 使用 monkeypatch 模拟 _fetch_raw_data 方法
    monkeypatch.setattr(index_loader, "_fetch_raw_data", mock_empty_api)

    # 清空缓存文件，确保加载数据时从API获取
    file_path = index_loader._get_file_path(TEST_INDEX, START_DATE, END_DATE)
    if Path(file_path).exists():
        Path(file_path).unlink()

    # 预期抛出 DataLoaderError
    with pytest.raises(DataLoaderError) as excinfo:
        index_loader.load_data(TEST_INDEX, START_DATE, END_DATE)
    assert "API返回的数据为空" in str(excinfo.value)


def test_index_loader_expired_cache(index_loader, monkeypatch):
    """测试缓存过期验证"""
    file_path = index_loader._get_file_path(TEST_INDEX, START_DATE, END_DATE)

    # 确保缓存文件存在
    if not file_path.exists():
        # 确保 load_data 能够成功创建缓存文件
        valid_data = pd.DataFrame({
            "date": pd.date_range(start=START_DATE, end=END_DATE),
            "open": [3000] * len(pd.date_range(start=START_DATE, end=END_DATE)),
            "high": [3010] * len(pd.date_range(start=START_DATE, end=END_DATE)),
            "low": [2990] * len(pd.date_range(start=START_DATE, end=END_DATE)),
            "close": [3005] * len(pd.date_range(start=START_DATE, end=END_DATE)),
            "volume": [150000000] * len(pd.date_range(start=START_DATE, end=END_DATE))
        })
        valid_data.to_csv(file_path, index=False, encoding='utf-8')

    # 模拟API返回空数据
    monkeypatch.setattr(index_loader, "_fetch_raw_data", lambda *args, **kwargs: pd.DataFrame())

    # 修改文件时间为过期
    expired_time = time.time() - index_loader.cache_days * 86400 - 1
    os.utime(file_path, (expired_time, expired_time))

    # 预期抛出 DataLoaderError
    with pytest.raises(DataLoaderError):
        index_loader.load_data(TEST_INDEX, START_DATE, END_DATE)


def test_index_loader_empty_api_response(index_loader, monkeypatch):
    """测试API返回空数据"""

    def mock_empty_api(*args, **kwargs):
        return pd.DataFrame(columns=[])  # 完全空的DataFrame

    monkeypatch.setattr(index_loader, "_fetch_raw_data", mock_empty_api)

    with pytest.raises(DataLoaderError) as excinfo:
        index_loader.load_data(TEST_INDEX, START_DATE, END_DATE)
    assert "API返回的数据为空" in str(excinfo.value)


def test_index_loader_single_day_load(index_loader):
    """边界测试：单日数据加载"""
    test_raw_data = pd.DataFrame({
        '日期': ['2023-01-05'],  # 保持为列
        '开盘': [3000],
        '最高': [3010],
        '最低': [2990],
        '收盘': [3005],
        '成交量': [150000000]
    })
    test_raw_data['日期'] = pd.to_datetime(test_raw_data['日期'])

    with patch.object(index_loader, '_fetch_raw_data', return_value=test_raw_data):
        df = index_loader.load_data("HS300", "2023-01-05", "2023-01-05")
        assert not df.empty, "单日数据不应为空"
        assert len(df.index) == 1, "应返回单日数据"


def test_index_loader_cross_year_load(index_loader):
    """边界测试：跨年数据加载"""
    test_raw_data = pd.DataFrame({
        '日期': ['2022-12-31', '2023-01-01'],  # 保持为列
        '开盘': [3000, 3100],
        '收盘': [3005, 3105],
        '最高': [3010, 3110],
        '最低': [2990, 3090],
        '成交量': [150000000, 160000000]
    })
    test_raw_data['日期'] = pd.to_datetime(test_raw_data['日期'])

    with patch.object(index_loader, '_fetch_raw_data', return_value=test_raw_data):
        df = index_loader.load_data("HS300", "2022-12-31", "2023-01-01")
        assert not df.empty, "跨年数据不应为空"
        assert len(df.index) == 2, "应返回两天数据"


# 清理测试文件
@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_files():
    """会话级fixture: 测试结束后清理临时文件"""
    yield
    temp_dir = Path("data/test_index")
    if temp_dir.exists():
        for f in temp_dir.glob("*.csv"):
            f.unlink()
        temp_dir.rmdir()
    logger.info("已清理所有临时测试文件")


def test_missing_columns_processing(index_loader, monkeypatch):
    """测试原始数据缺失必要列"""

    def mock_fetch_invalid_data(*args, **kwargs):
        return pd.DataFrame({"错误列": [1, 2, 3]})

    monkeypatch.setattr(index_loader, "_fetch_raw_data", mock_fetch_invalid_data)

    with pytest.raises(DataLoaderError) as excinfo:
        index_loader.load_data("HS300", "2023-01-01", "2023-01-05")
    assert "原始数据缺少必要列" in str(excinfo.value)  # 保持原断言，错误信息应已修正


def test_duplicate_index_handling(index_loader, monkeypatch):
    """测试重复索引数据处理"""
    test_data = pd.DataFrame({
        "日期": ["2023-01-01", "2023-01-01"],
        "开盘": [3000, 3010],
        "收盘": [3005, 3015],
        "最高": [3010, 3020],  # 确保包含所有必要列
        "最低": [2990, 3000],
        "成交量": [1e6, 1.2e6]
    })

    with patch.object(index_loader, "_fetch_raw_data", return_value=test_data):
        df = index_loader.load_data("HS300", "2023-01-01", "2023-01-01")
        assert len(df) == 1


def test_index_retry_mechanism(index_loader: IndexDataLoader, caplog):
    """测试指数加载器重试机制"""
    # 强制删除缓存文件，确保触发API调用
    file_path = index_loader._get_file_path("HS300", "2023-01-01", "2023-01-05")
    if file_path.exists():
        file_path.unlink()
    assert not file_path.exists(), "缓存文件未被正确删除"

    # 模拟连续抛出异常（次数 = max_retries + 1）
    with patch.object(
            index_loader,
            "_fetch_raw_data",
            side_effect=[ConnectionError] * (index_loader.max_retries + 1)  # 初始调用 + 重试次数
    ) as mock_api:
        with pytest.raises(DataLoaderError):
            index_loader.load_data("HS300", "2023-01-01", "2023-01-05")


def test_index_loader_invalid_scaler(index_loader: IndexDataLoader):
    """测试无效标准化器路径处理"""
    test_data = pd.DataFrame({"close": [3000]}, index=[pd.Timestamp("2023-01-01")])

    with pytest.raises(DataLoaderError):
        index_loader.normalize_data(test_data, scaler_path="/invalid/path/scaler.pkl")


def test_index_loader_partial_columns(index_loader: IndexDataLoader, monkeypatch):
    """测试部分列缺失处理"""
    test_data = pd.DataFrame({
        "日期": ["2023-01-01"],  # 存在
        "开盘": [3000],  # 存在
        "收盘": [3010]  # 存在
        # 明确缺失 "最高"、"最低"、"成交量"
    })
    print("测试数据原始列名:", test_data.columns.tolist())

    # 强制 Mock 返回测试数据
    monkeypatch.setattr(index_loader, "_fetch_raw_data", lambda *args: test_data)

    # 清空缓存文件，确保加载数据时从API获取
    file_path = index_loader._get_file_path("HS300", "2023-01-01", "2023-01-01")
    if Path(file_path).exists():
        Path(file_path).unlink()

    # 捕获异常并检查异常信息
    with pytest.raises(DataLoaderError) as excinfo:
        index_loader.load_data("HS300", "2023-01-01", "2023-01-01")
    assert "原始数据缺少必要列" in str(excinfo.value)


def test_index_loader_mixed_frequency_data(index_loader: IndexDataLoader, monkeypatch):
    """测试混合频率数据对齐处理"""
    test_data = pd.DataFrame({
        "日期": ["2023-01-01 09:30", "2023-01-01 15:00"],
        "开盘": [3000, 3005],
        "收盘": [3010, 3015],
        "最高": [3020, 3025],
        "最低": [2995, 3000],
        "成交量": [1e6, 2e6]
    })

    monkeypatch.setattr(index_loader, "_fetch_raw_data", lambda *args: test_data)
    data = index_loader.load_data("HS300", "2023-01-01", "2023-01-01")

    # 验证聚合后的数据
    assert len(data) == 1  # 只有一天的数据
    assert data.index[0] == pd.Timestamp("2023-01-01")
    assert data.iloc[0]['open'] == 3000  # 开盘价取第一个值
    assert data.iloc[0]['close'] == 3015  # 收盘价取最后一个值
    assert data.iloc[0]['high'] == 3025  # 最高价取最大值
    assert data.iloc[0]['low'] == 2995  # 最低价取最小值
    assert data.iloc[0]['volume'] == 3e6  # 成交量求和


# 指数数据标准化异常
def test_index_normalization_exception(index_loader: IndexDataLoader):
    """测试指数数据标准化异常处理"""
    invalid_data = pd.DataFrame({"invalid": [1, 2, 3]})
    with pytest.raises(DataLoaderError):
        index_loader.normalize_data(invalid_data)


def test_index_normalize_data(index_loader: IndexDataLoader, tmp_path: Path):
    """测试指数数据标准化处理"""
    test_data = pd.DataFrame({
        "open": [3000, 3100],
        "high": [3050, 3150],
        "low": [2980, 3080],
        "close": [3000, 3100],
        "volume": [1e6, 2e6]
    }, index=pd.date_range("2023-01-01", periods=2))

    # 使用临时路径存储标准化器
    scaler_path = tmp_path / "test_scaler.pkl"

    normalized = index_loader.normalize_data(test_data, scaler_path=scaler_path)
    assert "close" in normalized.columns
    assert abs(normalized["close"].mean()) < 1e-6  # 标准化后均值为0


def test_index_loader_custom_scaler(index_loader: IndexDataLoader, tmp_path: Path):
    """测试自定义标准化器路径处理"""
    test_data = pd.DataFrame({
        "open": [3000, 3100],
        "high": [3050, 3150],
        "low": [2980, 3080],
        "close": [3000, 3100],
        "volume": [1e6, 2e6]
    }, index=pd.date_range("2023-01-01", periods=2))

    scaler_path = tmp_path / "custom_scaler.pkl"

    normalized = index_loader.normalize_data(test_data, scaler_path)
    assert scaler_path.exists()
    assert abs(normalized.mean()).sum() < 1e-6


# 新增测试用例 - 指数数据加载器网络重试
def test_index_loader_network_retry(index_loader: IndexDataLoader, caplog):
    """测试指数加载器网络异常重试机制"""
    with patch.object(index_loader, "_fetch_raw_data",
                      side_effect=ConnectionError("模拟网络错误")) as mock_fetch:
        with pytest.raises(DataLoaderError):
            index_loader.load_data("HS300", "2023-01-01", "2023-01-05")
        assert mock_fetch.call_count == index_loader.max_retries + 1


def test_index_loader_special_cases(index_loader):
    # 测试空缓存文件
    file_path = index_loader.save_path / "HS300_20230101_20230105.csv"
    pd.DataFrame().to_csv(file_path, encoding='utf-8')

    # 模拟API返回空数据
    with patch.object(index_loader, "_fetch_raw_data", return_value=pd.DataFrame()):
        with pytest.raises(DataLoaderError) as excinfo:
            index_loader.load_data("HS300", "2023-01-01", "2023-01-05")
        assert "API返回的数据为空" in str(excinfo.value)

    # 测试日期格式转换失败
    raw_data = pd.DataFrame({
        "日期": ["invalid-date"],
        "开盘": [3000],
        "收盘": [3010],
        "最高": [3020],
        "最低": [2990],
        "成交量": [1e6]
    })
    with patch.object(index_loader, "_fetch_raw_data", return_value=raw_data):
        with pytest.raises(DataLoaderError) as excinfo:
            index_loader.load_data("HS300", "2023-01-01", "2023-01-05")
        assert "日期格式解析失败" in str(excinfo.value)


# 测试标准化过程中输入无效数据
def test_normalize_invalid_data(index_loader):
    invalid_data = pd.DataFrame({"invalid": [1, 2]})
    with pytest.raises(DataLoaderError):
        index_loader.normalize_data(invalid_data)


# 测试网络请求重试机制
def test_retry_on_connection_error(index_loader, mocker):
    # 模拟连续抛出ConnectionError
    mocker.patch.object(
        index_loader,
        "_fetch_raw_data",
        side_effect=ConnectionError("模拟网络错误")
    )
    with pytest.raises(DataLoaderError) as excinfo:
        index_loader.load_data("HS300", "2023-01-01", "2023-01-05")
    assert "超过最大重试次数" in str(excinfo.value)
    assert index_loader._fetch_raw_data.call_count == index_loader.max_retries + 1


def test_index_loader_edge_cases():
    # 测试 API 返回空数据
    try:
        with patch.object(IndexDataLoader, '_fetch_raw_data', return_value=pd.DataFrame()):
            index_loader = IndexDataLoader(save_path="temp", cache_days=7)
            data = index_loader.load_data("HS300", "2023-01-01", "2023-01-05")
    except DataLoaderError as e:
        data = pd.DataFrame()
    assert data.empty

    # 测试无效指数代码
    with pytest.raises(ValueError):
        index_loader.load_data("INVALID_INDEX", "2023-01-01", "2023-01-05")

    # 测试日期范围外数据
    with pytest.raises(DataLoaderError):
        data = index_loader.load_data("HS300", "2023-01-01", "2023-01-01")


def test_column_whitespace_handling(index_loader, monkeypatch):
    """测试列名包含全角空格的处理"""
    test_data = pd.DataFrame({
        "　日期　": ["2023-01-01"],  # 全角空格
        "　开盘　": [3000],
        "收盘": [3010],
        "最高": [3020],
        "最低": [2990],
        "成交量": [1e6]
    })

    logger.debug(f"测试数据原始列名: {test_data.columns.tolist()}")

    with patch.object(index_loader, "_fetch_raw_data", return_value=test_data):
        data = index_loader.load_data("HS300", "2023-01-01", "2023-01-01")

    # 检查索引名称是否为 'date'，并验证数据存在
    assert data.index.name == 'date', "数据框索引名称应为 'date'"
    assert not data.empty, "数据不应为空"
    # 确保其他必要列存在
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])


def test_mixed_frequency_data(index_loader, monkeypatch):
    """测试混合频率数据处理"""
    test_data = pd.DataFrame({
        "日期": ["2023-01-01 09:30", "2023-01-01 15:00"],
        "开盘": [3000, 3005],
        "收盘": [3010, 3015],
        "最高": [3020, 3025],
        "最低": [2995, 3000],
        "成交量": [1e6, 2e6]
    })

    # 使用 Mock 设置返回值
    mock_fetch = Mock(return_value=test_data)
    monkeypatch.setattr(index_loader, "_fetch_raw_data", mock_fetch)
    data = index_loader.load_data("HS300", "2023-01-01", "2023-01-01")
    assert len(data) == 1


def test_index_loader_cross_quarter_load(index_loader):
    start_date = "2023-03-31"
    end_date = "2023-06-30"
    test_raw_data = pd.DataFrame({
        '日期': ['2023-03-31', '2023-06-30'],
        '开盘': [3000, 3100],
        '收盘': [3005, 3105],
        '最高': [3010, 3110],
        '最低': [2990, 3090],
        '成交量': [150000000, 160000000]
    })
    test_raw_data['日期'] = pd.to_datetime(test_raw_data['日期'])
    with patch.object(index_loader, "_fetch_raw_data", return_value=test_raw_data):
        df = index_loader.load_data("HS300", start_date, end_date)
        assert not df.empty
        assert len(df) == 2


def test_index_loader_data_conflict_handling(index_loader, tmp_path):
    # 创建旧缓存
    cache_path = tmp_path / "HS300_20230101_20230131.csv"
    existing_data = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01"]),
        "open": [3000],
        "high": [3010],
        "low": [2990],
        "close": [3005],
        "volume": [150000000]
    })
    existing_data.to_csv(cache_path, index=False)
    # 新数据与缓存数据冲突
    new_data = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01"]),
        "open": [3005],
        "high": [3015],
        "low": [2995],
        "close": [3010],
        "volume": [160000000]
    })
    merged = index_loader._merge_with_cache(cache_path, new_data)
    assert merged is not None


def test_index_data_denormalization(index_loader, tmp_path):
    """测试标准化后反标准化数据完整性"""
    scaler_path = tmp_path / "test_scaler.pkl"  # 使用临时路径存储标准化器

    test_data = pd.DataFrame({
        "open": [3000, 3100],
        "high": [3050, 3150],
        "low": [2950, 3050],
        "close": [3010, 3110],
        "volume": [1e6, 2e6]
    }, index=pd.date_range("2023-01-01", periods=2))

    # 标准化数据
    normalized = index_loader.normalize_data(test_data, scaler_path=scaler_path)
    # 反标准化数据
    denormalized = index_loader.normalize_data(normalized, scaler_path=scaler_path, inverse=True)

    assert np.allclose(test_data.values, denormalized.values, rtol=1e-3), "反标准化数据应与原始数据一致"


def test_real_time_index_update(index_loader, mocker):
    """测试实时指数更新场景"""
    mocker.patch.object(index_loader, "_is_cache_valid", return_value=False)
    # 模拟第一次调用返回空数据，第二次调用返回有效数据
    valid_data = pd.DataFrame({
        "日期": pd.date_range("2023-01-01", periods=1),
        "开盘": [3000],
        "最高": [3010],
        "最低": [2990],
        "收盘": [3005],
        "成交量": [150000000]
    })
    mocker.patch.object(
        index_loader,
        "_fetch_raw_data",
        side_effect=[
            pd.DataFrame(),  # 第一次调用返回空数据
            valid_data  # 第二次调用返回有效数据
        ]
    )
    # 捕获可能的异常
    try:
        data = index_loader.load_data("HS300", "2023-01-01", "2023-01-01")
        assert not data.empty, "返回的数据不应为空"
    except DataLoaderError as e:
        pytest.fail(f"实时指数更新测试失败: {str(e)}")


def test_invalid_index_code():
    """测试无效指数代码处理"""
    loader = IndexDataLoader()
    with pytest.raises(ValueError, match="Invalid index code"):
        loader.load_data("INVALID", "2022-01-01", "2022-12-31")


def test_date_conversion_logic():
    loader = IndexDataLoader()
    mock_data = pd.DataFrame({
        '日期': pd.date_range('2022-01-01', periods=5),
        '开盘': [100, 101, 102, 103, 104],
        '最高': [110, 111, 112, 113, 114],
        '最低': [90, 91, 92, 93, 94],
        '收盘': [105, 106, 107, 108, 109],
        '成交量': [100000, 150000, 200000, 250000, 300000]
    })
    with patch.object(loader, '_is_cache_valid', return_value=False), \
            patch.object(loader, '_fetch_raw_data', return_value=mock_data):
        result = loader.load_data("HS300", "2022-01-01", "2022-12-31")
        assert not result.empty


def test_normalization_failure():
    """测试数据标准化失败场景"""
    loader = IndexDataLoader()
    invalid_data = pd.DataFrame()  # 空数据

    with pytest.raises(DataLoaderError, match="数据标准化失败: 输入数据为空"):
        loader.normalize_data(invalid_data)


def test_save_data_failure(mocker, tmp_path):
    """测试数据保存失败处理"""
    loader = IndexDataLoader(save_path=tmp_path)
    mocker.patch("pandas.DataFrame.to_csv", side_effect=Exception("IO error"))

    valid_data = pd.DataFrame({
        'date': ['2020-01-01'],
        'open': [100.0],
        'high': [105.0],
        'low': [95.0],
        'close': [102.0],
        'volume': [100000]
    })

    assert not loader._save_data(valid_data, tmp_path / "test.csv")


@pytest.mark.parametrize("start,end", [
    ("2023-01-01", "2023-01-05"),
    ("2023-01-05", "2023-01-01"),  # 无效日期范围
    ("2023-13-01", "2023-01-05")   # 无效日期格式
])
def test_date_validation(index_loader, start, end):
    if start > end or "13" in start:
        with pytest.raises((ValueError, DataLoaderError)):
            index_loader.load_data("HS300", start, end)


def test_api_data_processing(index_loader):
    """测试API数据处理异常"""
    # 模拟API返回无效数据
    with patch('akshare.index_zh_a_hist') as mock_api:
        mock_api.return_value = pd.DataFrame()  # 空数据
        with pytest.raises(DataLoaderError):
            index_loader.load_data("HS300", "2020-01-01", "2023-01-01")

        # 缺少必要列
        mock_api.return_value = pd.DataFrame({'wrong': [1, 2, 3]})
        with pytest.raises(DataLoaderError):
            index_loader.load_data("HS300", "2020-01-01", "2023-01-01")


def test_index_cache_missing_columns(index_loader, tmp_path):
    # 创建缺少必要列的缓存文件
    invalid_data = pd.DataFrame({"date": ["2023-01-01"], "price": [100]})
    file_path = tmp_path / "HS300_20230101_20230105.csv"
    invalid_data.to_csv(file_path, index=False)

    assert not index_loader._is_cache_valid(file_path)


def test_index_data_normalization(index_loader):
    test_data = pd.DataFrame({
        "open": [100, 102],
        "high": [105, 106],
        "low": [98, 101],
        "close": [103, 104],
        "volume": [10000, 15000]
    })
    normalized = index_loader.normalize_data(test_data)
    assert not normalized.empty
    # 验证标准化后数据范围
    assert abs(normalized.mean().mean()) < 0.1  # 均值接近0


# 测试用例1: 验证无效指数代码处理
def test_index_loader_invalid_code():
    loader = IndexDataLoader()
    with pytest.raises(ValueError):
        loader.load_data("INVALID_INDEX", "2023-01-01", "2023-12-31")


# 测试用例2: 验证缓存文件列名缺失
def test_index_loader_missing_columns_in_cache(tmp_path):
    loader = IndexDataLoader(save_path=tmp_path)
    file_path = tmp_path / "HS300_20230101_20231231.csv"

    # 创建缺少必要列的缓存文件
    pd.DataFrame({"open": [3000], "close": [3100]}).to_csv(file_path)

    # 应检测到列缺失并返回False
    assert not loader._is_cache_valid(file_path)


# 测试用例3: 验证数据标准化失败处理
def test_index_loader_normal(index_loader, monkeypatch):
    """正常加载指数数据"""
    # 模拟返回的指数数据
    mock_data = pd.DataFrame({
        "date": pd.date_range("2022-01-01", "2022-01-31"),
        "open": [100.0] * 31,
        "high": [101.0] * 31,
        "low": [99.0] * 31,
        "close": [100.5] * 31,
        "volume": [1000] * 31
    })

    # 模拟 _fetch_raw_data 方法
    monkeypatch.setattr(index_loader, "_fetch_raw_data", Mock(return_value=mock_data))

    # 加载数据
    data = index_loader.load_data(TEST_INDEX, START_DATE, END_DATE)

    # 验证数据正确性
    assert not data.empty
    assert 'open' in data.columns
    assert 'high' in data.columns
    assert 'low' in data.columns
    assert 'close' in data.columns
    assert 'volume' in data.columns
    assert isinstance(data.index, pd.DatetimeIndex)  # 确保索引是日期类型

    # 验证缓存文件是否生成
    file_path = index_loader._get_file_path(TEST_INDEX, START_DATE, END_DATE)
    assert file_path.exists()

    # 验证文件内容
    saved_df = pd.read_csv(file_path, parse_dates=["date"])
    assert not saved_df.empty
    assert saved_df.shape == (31, 6)  # 5列 + date列
    assert 'date' in saved_df.columns

# 测试验证索引处理逻辑
def test_save_data_with_index(tmp_path):
    """测试带索引数据的保存逻辑"""
    loader = IndexDataLoader(save_path=tmp_path)

    # 创建带日期索引的数据框
    dates = pd.date_range("2023-01-01", periods=5)
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 2000, 3000, 4000, 5000]
    }, index=dates)
    data.index.name = 'date'

    # 测试保存
    file_path = tmp_path / "test_data.csv"
    result = loader._save_data(data, file_path)

    assert result is True
    assert file_path.exists()

    # 验证保存的内容
    saved_df = pd.read_csv(file_path, parse_dates=["date"])
    assert 'date' in saved_df.columns
    assert saved_df.shape == (5, 6)  # 5列 + date列
    assert saved_df['date'].tolist() == dates.tolist()

# 测试用例4: 验证数据保存失败处理
def test_index_loader_save_failure(tmp_path, mocker):
    loader = IndexDataLoader(save_path=tmp_path)
    test_data = pd.DataFrame({
        "open": [3000], "high": [3100], "low": [2900],
        "close": [3050], "volume": [1000000]
    })

    # 模拟保存失败
    mocker.patch("pandas.DataFrame.to_csv", side_effect=Exception("IO error"))
    assert not loader._save_data(test_data, tmp_path / "test.csv")


def test_index_loader_cache_missing_columns(tmp_path):
    """测试缓存文件缺少必要列"""
    loader = IndexDataLoader(save_path=tmp_path)
    file_path = tmp_path / "HS300_20230101_20230102.csv"
    file_path.write_text("date,open\n2023-01-01,1000")  # 缺少必要列

    assert not loader._is_cache_valid(file_path)


# 测试网络重试机制
def test_network_retry(index_loader, mocker):
    """测试指数加载器网络异常重试机制"""
    mocker.patch.object(index_loader, "_fetch_raw_data",
                        side_effect=ConnectionError("模拟网络错误"))
    with pytest.raises(DataLoaderError):
        index_loader.load_data("HS300", "2023-01-01", "2023-01-05")
    assert index_loader._fetch_raw_data.call_count == index_loader.max_retries + 1


def test_akshare_failure(index_loader, monkeypatch):
    monkeypatch.setattr(ak, "index_zh_a_hist",
                       Mock(side_effect=RequestException))
    with pytest.raises(DataLoaderError):
        index_loader.load_data("HS300", "2023-01-01", "2023-01-05")


def test_single_day_data(index_loader, monkeypatch):
    """测试单日数据加载"""
    # 模拟返回的指数数据
    mock_data = pd.DataFrame({
        "date": ["2023-01-01"],
        "open": [3000.0],
        "high": [3100.0],
        "low": [2900.0],
        "close": [3050.0],
        "volume": [1000000]
    })

    # 模拟 _fetch_raw_data 方法
    monkeypatch.setattr(index_loader, "_fetch_raw_data", Mock(return_value=mock_data))

    # 加载数据
    data = index_loader.load_data("HS300", "2023-01-01", "2023-01-01")

    # 验证数据正确性
    assert not data.empty
    assert 'open' in data.columns
    assert 'high' in data.columns
    assert 'low' in data.columns
    assert 'close' in data.columns
    assert 'volume' in data.columns

    # 验证数据值
    assert data.loc["2023-01-01", "open"] == 3000.0
    assert data.loc["2023-01-01", "high"] == 3100.0
    assert data.loc["2023-01-01", "low"] == 2900.0
    assert data.loc["2023-01-01", "close"] == 3050.0
    assert data.loc["2023-01-01", "volume"] == 1000000
