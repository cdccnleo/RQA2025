# tests/data/test_news_loader.py
import time
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import pytest

from src.features.processors.sentiment import SentimentAnalyzer
from src.infrastructure.utils.datetime_parser import DateTimeParser
from src.infrastructure.utils.logger import get_logger
from src.data.loader.news_loader import FinancialNewsLoader, SentimentNewsLoader
from src.infrastructure.utils.exceptions import DataLoaderError
import os

# 配置测试日志
logger = get_logger(__name__)

# 测试常量
TEST_START = "2023-01-01"
TEST_END = "2023-01-07"
INVALID_SOURCE = "invalid_source"
TEST_SOURCE = "cls"
TEMP_SAVE_PATH = "tests/data/temp_news"


@pytest.fixture(scope="function")
def news_loader(tmp_path) -> FinancialNewsLoader:
    """初始化基础新闻加载器fixture"""
    return FinancialNewsLoader(
        source=TEST_SOURCE,
        save_path=tmp_path / TEMP_SAVE_PATH,
        cache_days=1
    )


@pytest.fixture(scope="function")
def sentiment_loader(tmp_path) -> SentimentNewsLoader:
    """初始化情感分析新闻加载器fixture"""
    return SentimentNewsLoader(
        source=TEST_SOURCE,
        save_path=tmp_path / TEMP_SAVE_PATH,
        cache_days=1,
        debug_mode=True
    )


def test_news_loader_init_invalid_source():
    """异常测试: 使用无效数据源初始化"""
    with pytest.raises(ValueError) as excinfo:
        FinancialNewsLoader(source=INVALID_SOURCE)
    assert "Unsupported news source" in str(excinfo.value)
    logger.debug("无效数据源测试通过")


def test_load_normal_news_data(news_loader: FinancialNewsLoader):
    # 使用明确的时间格式
    test_raw_data = pd.DataFrame({
        'title': ['新闻标题1', '新闻标题2'],
        'content': ['<div>内容1</div>', '<p>内容2</p>'],
        'publish_date': ['2023-01-01', '2023-01-02'],
        'publish_time': ['10:00:00', '14:30:00']
    })

    with patch.object(news_loader, '_fetch_raw_data', return_value=test_raw_data):
        # 确保传入的时间是本地时间格式
        df = news_loader.load_data("2023-01-01 00:00:00", "2023-01-05 23:59:59")
        assert not df.empty, "返回的 DataFrame 不应为空"
        assert "publish_time" in df.columns, "应包含 publish_time 列"
        assert "content" in df.columns, "应包含 content 列"


def test_sentiment_loader_adds_sentiment(sentiment_loader: SentimentNewsLoader):
    """正常流程测试: 情感分析功能验证"""
    # 确保传入的数据有效
    test_data = pd.DataFrame({
        '标题': ['新闻标题1', '新闻标题2'],
        '内容': ['内容1', '内容2'],
        '发布日期': ['2023-01-01', '2023-01-02'],
        '发布时间': ['10:00:00', '14:30:00']
    })

    with patch.object(sentiment_loader, '_fetch_raw_data', return_value=test_data):
        df = sentiment_loader.load_data(TEST_START, TEST_END)
        assert 'sentiment' in df.columns, "缺少情感分数列"
        assert df['sentiment'].notna().any(), "至少有一条情感分析成功"


def test_invalid_date_range(news_loader: FinancialNewsLoader):
    """异常测试: 无效日期范围处理"""
    with pytest.raises(ValueError) as excinfo:
        news_loader.load_data("2023-02-01", "2023-01-01")
    assert "开始日期不能大于结束日期" in str(excinfo.value)
    logger.debug("日期范围校验正常")


def test_empty_api_response(news_loader: FinancialNewsLoader, monkeypatch):
    """测试: 模拟API返回空数据"""

    def mock_empty_api(*args, **kwargs):
        return pd.DataFrame()

    # 使用monkeypatch模拟空响应
    monkeypatch.setattr(news_loader, '_fetch_raw_data', mock_empty_api)

    # 预期返回空的 DataFrame
    data = news_loader.load_data(TEST_START, TEST_END)
    assert data.empty


def test_expired_cache(news_loader: FinancialNewsLoader):
    """边界测试: 缓存过期验证"""
    test_raw_data = pd.DataFrame({
        '标题': ['过期测试'],
        '内容': ['过期内容'],
        '发布日期': ['2023-01-01'],
        '发布时间': ['08:00:00']
    })

    with patch.object(news_loader, '_fetch_raw_data', return_value=test_raw_data):
        # 生成初始缓存
        # 使用news_loader.save_path获取正确的缓存路径
        cache_path = news_loader.save_path / f"news_{TEST_START}_{TEST_END}.csv"
        news_loader.load_data(TEST_START, TEST_END)

        # 修改缓存时间为过期
        expired_time = os.path.getmtime(cache_path) - 2 * 86400
        os.utime(cache_path, (expired_time, expired_time))

        # 应重新获取数据
        with patch.object(news_loader, '_fetch_raw_data') as mock_fetch:
            mock_fetch.return_value = test_raw_data
            news_loader.load_data(TEST_START, TEST_END)
            mock_fetch.assert_called_once()


def test_cross_year_load(news_loader: FinancialNewsLoader):
    """边界测试: 跨年数据加载"""
    test_raw_data = pd.DataFrame({
        '标题': ['跨年新闻1', '跨年新闻2'],
        '内容': ['内容1', '内容2'],
        '发布日期': ['2022-12-31', '2023-01-01'],
        '发布时间': ['23:59:59', '00:00:00']
    })

    with patch.object(news_loader, '_fetch_raw_data', return_value=test_raw_data):
        df = news_loader.load_data("2022-12-25", "2023-01-05")
        years = df['publish_time'].dt.year.unique()
        assert len(years) >= 2


def test_content_cleaning(news_loader: FinancialNewsLoader):
    """数据质量测试: HTML内容清洗验证"""
    test_html = "<div>Hello <b>World</b></div>"
    cleaned = news_loader._clean_single_html(test_html)

    assert "<" not in cleaned, "应移除HTML标签"
    assert "Hello World" in cleaned, "应保留文本内容"
    logger.debug(f"HTML清洗结果: {cleaned}")


def test_duplicate_removal(news_loader: FinancialNewsLoader):
    """数据质量测试: 重复数据删除"""
    # 构造包含重复项的数据
    test_data = pd.DataFrame({
        "title": ["测试新闻"] * 2,
        "content": ["相同内容"] * 2,
        "publish_date": ["2023-01-01"] * 2,
        "publish_time": ["10:00:00"] * 2
    })

    processed = news_loader._remove_duplicates(test_data)
    assert len(processed) == 1, "应删除重复项"


# 清理测试文件
@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_files():
    """会话级fixture: 测试结束后清理临时文件"""
    yield
    temp_dir = Path(TEMP_SAVE_PATH)
    if temp_dir.exists():
        for f in temp_dir.glob("*.csv"):
            f.unlink()
        temp_dir.rmdir()
    logger.info("已清理所有临时测试文件")


def test_news_missing_datetime_columns(news_loader: FinancialNewsLoader):
    """测试原始数据缺少日期时间列的处理"""
    test_raw = pd.DataFrame({"错误列": [1, 2]})

    with patch.object(news_loader, "_fetch_raw_data", return_value=test_raw):
        with pytest.raises(DataLoaderError) as excinfo:
            news_loader.load_data("2023-01-01", "2023-01-05")
        assert "缺失必要列" in str(excinfo.value)


def test_news_invalid_html_cleaning(news_loader: FinancialNewsLoader):
    """测试非字符串类型的HTML清洗"""
    assert news_loader._clean_single_html(123) == ""
    assert news_loader._clean_single_html(None) == ""


def test_news_loader_invalid_aggregation(news_loader: FinancialNewsLoader, monkeypatch):
    """测试无效聚合字段处理"""
    # 构造映射后的列包含'title'和'publish_time'，但删除'content'
    test_raw = pd.DataFrame({
        "标题": ["测试"],
        "内容": ["内容"],
        "发布日期": ["2023-01-01"],
        "发布时间": ["10:00:00"]
    })
    # 列映射后的DataFrame
    mapped_df = test_raw.rename(columns={
        '标题': 'title',
        '内容': 'content',
        '发布日期': 'publish_date',
        '发布时间': 'publish_time'
    })
    # 删除'content'列以模拟聚合阶段缺少该列
    mapped_df = mapped_df.drop(columns=['content'])

    # 模拟_process_raw_data返回处理后的数据，绕过早期错误
    def mock_process_raw(raw_df):
        return mapped_df

    with patch.object(news_loader, '_fetch_raw_data', return_value=test_raw), \
            patch.object(news_loader, '_process_raw_data', mock_process_raw):
        with pytest.raises(DataLoaderError) as excinfo:
            news_loader.load_data("2023-01-01", "2023-01-05")
        assert "聚合所需字段缺失" in str(excinfo.value)


def test_news_loader_timezone_handling(news_loader: FinancialNewsLoader):
    """测试时区敏感时间解析"""
    test_data = pd.DataFrame({
        "publish_date": ["2023-01-01"],
        "publish_time": ["23:59:59+08:00"]  # 带时区信息
    })

    processed = news_loader._parse_datetime(test_data)
    assert pd.api.types.is_datetime64_any_dtype(processed["publish_time"])


def test_news_loader_time_parsing_edge_cases(news_loader: FinancialNewsLoader):
    """测试极端时间格式解析"""
    test_data = pd.DataFrame({
        "publish_date": ["2023-12-31"],  # 使用映射后的列名
        "publish_time": ["23:59:60"]  # 闰秒格式
    })

    processed = news_loader._parse_datetime(test_data)
    assert pd.api.types.is_datetime64_any_dtype(processed["publish_time"])


def test_news_loader_multilingual_processing(news_loader: FinancialNewsLoader):
    """测试多语言HTML清洗处理"""
    test_html = """
    <div>
        <p>中文内容</p>
        <script>console.log("test")</script>
        <div class="ad">广告内容</div>
    </div>
    """
    cleaned = news_loader._clean_single_html(test_html)
    assert "中文内容" in cleaned
    assert "console.log" not in cleaned
    assert "广告内容" not in cleaned


# 新闻加载器无效HTML处理
def test_news_loader_invalid_html_processing(news_loader: FinancialNewsLoader):
    """测试新闻加载器处理损坏HTML内容"""
    test_html = "<div>Broken<html><body>Unclosed tags"
    cleaned = news_loader._clean_single_html(test_html)
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


def test_sentiment_loader_error_handling(sentiment_loader, mocker):
    # 启用调试模式
    sentiment_loader.debug_mode = True

    # 模拟情感分析失败
    mocker.patch("src.features.processors.sentiment.SentimentAnalyzer.snownlp_sentiment",
                 side_effect=Exception("模拟情感分析失败"))
    df = pd.DataFrame({"content": ["正常内容"] * 100})

    # 使用 pytest.raises 捕获预期异常
    with pytest.raises(DataLoaderError, match="情感分析失败: 模拟情感分析失败"):
        sentiment_loader._add_sentiment_scores(df)


# 测试时区解析
def test_timezone_parsing(news_loader):
    test_data = pd.DataFrame({
        "publish_date": ["2023-01-01"],  # 正确列名
        "publish_time": ["12:00:00+08:00"]
    })
    processed = news_loader._parse_datetime(test_data)

    # 验证解析结果不为空
    assert not processed.empty, "解析结果不应为空"

    # 获取解析后的时间
    parsed_time = processed["publish_time"].iloc[0]

    # 验证时间值正确（北京时间）
    expected_time = pd.Timestamp("2023-01-01 12:00:00")
    assert parsed_time == expected_time, f"时间解析错误: 期望 {expected_time}, 实际 {parsed_time}"

    # 验证无时区信息
    assert parsed_time.tzinfo is None, "时间应无时区信息"


def test_sentiment_analyzer_exception_handling(sentiment_loader, mocker):
    # 模拟情感分析失败
    mocker.patch("src.features.processors.sentiment.SentimentAnalyzer.snownlp_sentiment",
                 side_effect=Exception("模拟情感分析失败"))
    df = pd.DataFrame({"content": ["测试内容"] * 3})

    # 使用 pytest.raises 捕获预期异常
    with pytest.raises(DataLoaderError, match=r".*情感分析失败.*"):
        sentiment_loader._add_sentiment_scores(df)


def test_sentiment_loader_edge_cases(sentiment_loader):
    """测试空内容和无效值的处理"""
    sentiment_loader.debug_mode = True  # 启用调试模式
    df = pd.DataFrame({"content": ["", None, "   "]})

    # 使用 pytest.raises 捕获预期异常
    with pytest.raises(DataLoaderError, match=r".*情感分析失败.*"):
        sentiment_loader._add_sentiment_scores(df)


def test_news_loader_edge_cases():
    # 测试 API 返回空数据
    with patch.object(FinancialNewsLoader, '_fetch_raw_data', return_value=pd.DataFrame()):
        news_loader = FinancialNewsLoader(source="cls", save_path="temp", cache_days=7)
        data = news_loader.load_data("2023-01-01", "2023-01-05")
        assert data.empty

    # 测试 HTML 清洗失败
    with patch.object(FinancialNewsLoader, '_clean_single_html', return_value=""):
        data = news_loader.load_data(TEST_START, TEST_END)
        assert data.empty

    # 测试重复新闻删除
    duplicate_data = pd.DataFrame({
        "publish_time": ["2023-01-01 10:00:00", "2023-01-01 10:00:00"],
        "title": ["重复新闻", "重复新闻"],
        "content": ["相同内容", "相同内容"]  # content 相同
    })
    cleaned = news_loader._remove_duplicates(duplicate_data)
    assert len(cleaned) == 1


def test_sentiment_model_failure(sentiment_loader, mocker):
    """测试情感分析模型异常"""
    sentiment_loader.debug_mode = True  # 启用调试模式

    # 模拟情感分析失败
    mocker.patch("src.features.processors.sentiment.SentimentAnalyzer.snownlp_sentiment",
                 side_effect=Exception("Model error"))
    test_data = pd.DataFrame({
        "content": ["测试内容"],
        "publish_time": ["2023-01-01"]
    })

    # 使用 pytest.raises 捕获预期异常
    with pytest.raises(DataLoaderError, match=r".*情感分析失败.*"):
        sentiment_loader._add_sentiment_scores(test_data)


def test_html_cleaning_with_script(news_loader):
    """测试包含脚本标签的HTML清洗"""
    dirty_html = "<script>alert(1)</script><div>真实内容</div>"
    cleaned = news_loader._clean_single_html(dirty_html)
    assert "alert" not in cleaned
    assert "真实内容" in cleaned


def test_news_loader_different_sources_compatibility(news_loader: FinancialNewsLoader):
    """测试不同新闻源的兼容性"""
    # 模拟 sina 数据源的数据
    sina_data = pd.DataFrame({
        'content': ['新闻内容1'],
        'time': ['2023-01-01 10:00:00'],
        'title': ['新闻标题1']  # 添加 title 列
    })

    # 模拟 em 数据源的数据
    em_data = pd.DataFrame({
        'em_content': ['新闻内容2'],
        'em_time': ['2023-01-01 11:00:00'],
        'em_title': ['新闻标题2']  # 添加 em_title 列
    })

    # 测试 sina 数据源
    news_loader.source = "sina"
    with patch("akshare.stock_info_global_sina", return_value=sina_data):
        data_sina = news_loader._fetch_raw_data("2023-01-01", "2023-01-01")
        assert not data_sina.empty

    # 测试 em 数据源
    news_loader.source = "em"
    with patch("akshare.stock_info_global_em", return_value=em_data):
        data_em = news_loader._fetch_raw_data("2023-01-01", "2023-01-01")
        assert not data_em.empty


def test_sentiment_loader_model_unavailable(sentiment_loader, mocker):
    """测试情感分析模型不可用时的处理"""
    # 启用调试模式
    sentiment_loader.debug_mode = True

    # 模拟情感分析模型不可用
    mocker.patch("src.features.processors.sentiment.SentimentAnalyzer.snownlp_sentiment",
                 side_effect=Exception("情感分析模型不可用"))

    test_data = pd.DataFrame({"content": ["测试内容"]})

    # 使用 pytest.raises 捕获预期异常
    with pytest.raises(DataLoaderError, match=r".*情感分析失败.*"):
        sentiment_loader._add_sentiment_scores(test_data)


def test_multilingual_sentiment(sentiment_loader):
    """测试多语言情感分析"""
    test_data = pd.DataFrame({
        "content": ["Excellent earnings report!", "利润大幅下滑"],
        "publish_time": ["2023-01-01", "2023-01-02"]
    })
    # 模拟 SentimentAnalyzer 的 snownlp_sentiment 方法
    with patch("src.features.processors.sentiment.SentimentAnalyzer.snownlp_sentiment") as mock:
        mock.side_effect = [0.8, -0.5]
        df = sentiment_loader._add_sentiment_scores(test_data)
        assert df["sentiment"].tolist() == [0.8, -0.5]


def test_news_loader_rate_limiting(news_loader, mocker):
    """测试API速率限制处理"""
    mocker.patch.object(news_loader, "_fetch_raw_data",
                        side_effect=ConnectionError("API rate limit exceeded"))
    data = news_loader.load_data("2023-01-01", "2023-01-05")
    assert data.empty  # 验证返回的数据是否为空


def test_em_source_adaptation():
    """测试EM数据源的特殊处理逻辑"""
    loader = FinancialNewsLoader(source="em")
    test_data = pd.DataFrame({
        'em_title': ["Test Title"],
        'em_content': ["Test Content"],
        'em_time': ["2023-01-01 12:00:00"]
    })

    # 模拟 akshare API 调用
    with patch("akshare.stock_info_global_em", return_value=test_data):
        processed = loader._fetch_raw_data("2020-01-01", "2023-01-01")
        assert "标题" in processed.columns
        assert "内容" in processed.columns
        assert "发布日期" in processed.columns
        assert "发布时间" in processed.columns


def test_sentiment_analysis_failure():
    """测试情感分析全面失败场景"""
    loader = SentimentNewsLoader()
    test_data = pd.DataFrame({
        'publish_time': ['2023-01-01'],
        'content': ["@#$%^&*"]  # 无法分析的文本
    })

    with patch.object(SentimentAnalyzer, 'snownlp_sentiment', return_value=None):
        result = loader._add_sentiment_scores(test_data)
        assert result["sentiment"].isna().all()


@pytest.mark.parametrize("source", ["cls", "sina", "em"])
def test_news_sources(source):
    """测试不同新闻源适配 - 使用模拟数据"""
    # 创建模拟数据
    if source == "cls":
        mock_data = pd.DataFrame({
            "标题": [f"{source}新闻标题"],
            "内容": [f"{source}新闻内容"],
            "发布日期": ["2023-01-05"],
            "发布时间": ["10:00:00"]
        })
    elif source == "sina":
        mock_data = pd.DataFrame({
            "content": [f"{source}新闻内容"],
            "time": ["2023-01-05 10:00:00"]
        })
    elif source == "em":
        mock_data = pd.DataFrame({
            "em_content": [f"{source}新闻内容"],
            "em_time": ["2023-01-05 10:00:00"],
            "em_title": [f"{source}新闻标题"]
        })

    # 确保 time 列是字符串类型
    if "time" in mock_data.columns:
        mock_data["time"] = mock_data["time"].astype(str)

    # 模拟 akshare API 调用
    with patch(f"akshare.stock_info_global_{source}", return_value=mock_data) as mock_func:
        loader = FinancialNewsLoader(source=source)
        df = loader.load_data("2023-01-01", "2023-01-10")

        # 验证结果
        assert not df.empty


def test_sentiment_analysis():
    """测试情感分析"""
    loader = SentimentNewsLoader()
    test_data = pd.DataFrame({
        "publish_time": ["2023-01-01"],
        "content": ["这家公司表现非常好，业绩超出预期"]
    })
    result = loader._add_sentiment_scores(test_data)
    assert "sentiment" in result.columns
    assert result["sentiment"].iloc[0] > 0.5  # 正面情感


def test_datetime_parser_with_various_formats():
    """测试日期时间解析器处理不同格式的能力"""
    test_data = pd.DataFrame({
        'publish_date': ['2023/01/01', '2023-02-01', '20230301'],
        'publish_time': ['10:00', '15:30:45', '']
    })

    result = DateTimeParser.parse_datetime(test_data, "publish_date", "publish_time")

    # 验证结果
    assert not result.empty
    assert 'publish_time' in result.columns
    assert result['publish_time'].dtype == 'datetime64[ns]'
    # 验证具体日期值
    assert result['publish_time'].iloc[0] == pd.Timestamp('2023-01-01 10:00:00')
    assert result['publish_time'].iloc[1] == pd.Timestamp('2023-02-01 15:30:45')
    assert result['publish_time'].iloc[2] == pd.Timestamp('2023-03-01 00:00:00')


def test_fallback_datetime_processing(news_loader):
    """测试备选日期时间处理方案"""
    test_data = pd.DataFrame({
        'title': ['测试新闻'],
        'content': ['测试内容'],
        'publish_date': ['2023-01-01'],
        'publish_time': ['10:00:00']
    })

    # 模拟DateTimeParser失败
    with patch('src.infrastructure.utils.datetime_parser.DateTimeParser.parse_datetime',
               side_effect=Exception("模拟失败")):
        result = news_loader._process_raw_data(test_data)

    assert not result.empty
    assert 'publish_time' in result.columns
    assert result['publish_time'].iloc[0] == pd.Timestamp('2023-01-01 10:00:00')


def test_invalid_source():
    """测试无效数据源"""
    with pytest.raises(ValueError):
        FinancialNewsLoader(source="invalid")


def test_empty_news_processing(news_loader):
    """测试空数据处理"""
    with patch('akshare.stock_info_global_cls') as mock_api:
        mock_api.return_value = pd.DataFrame()
        df = news_loader.load_data("2020-01-01", "2023-01-01")
        assert df.empty


def test_date_parsing_failure(news_loader):
    """测试日期解析失败"""
    # 创建无效日期数据
    raw_data = pd.DataFrame({
        '标题': ['测试新闻'],
        '内容': ['测试内容'],
        '发布日期': ['2020-01-01'],
        '发布时间': ['invalid-time']
    })

    with patch.object(news_loader, '_fetch_raw_data', return_value=raw_data):
        with pytest.raises(DataLoaderError):
            news_loader._parse_datetime(raw_data)


def test_news_loader_source_not_supported():
    with pytest.raises(ValueError):
        loader = FinancialNewsLoader(source="invalid_source")


def test_news_loader_empty_api_response(news_loader, mocker):
    mocker.patch("akshare.stock_info_global_cls", return_value=pd.DataFrame())
    df = news_loader.load_data("2023-01-01", "2023-01-05")
    assert df.empty


def test_news_loader_html_clean_error(news_loader, mocker):
    # 模拟 BeautifulSoup 抛出异常
    mocker.patch("src.data.loader.news_loader.BeautifulSoup",
                 side_effect=Exception("HTML parsing error"))

    test_data = pd.DataFrame({"content": ["<html>Test</html>"]})
    result = news_loader._clean_html_content(test_data)
    # 期望当解析失败时返回空字符串
    assert result.iloc[0]["content"] == ""


def test_sentiment_analyzer_failure(sentiment_loader, mocker):
    """测试情感分析器失败场景"""
    # 启用调试模式
    sentiment_loader.debug_mode = True

    # Mock 情感分析方法抛出异常
    mocker.patch("src.features.processors.sentiment.SentimentAnalyzer.snownlp_sentiment",
                 side_effect=Exception("Analysis failed"))

    test_data = pd.DataFrame({
        "content": ["This is a test news"],
        "publish_time": [pd.Timestamp.now()]
    })

    # 使用正则表达式匹配包含时间戳的异常信息
    with pytest.raises(DataLoaderError, match=r".*情感分析失败.*"):
        sentiment_loader._add_sentiment_scores(test_data)


def test_empty_content_sentiment(sentiment_loader):
    """测试空内容和无效值的处理"""
    # 启用调试模式
    sentiment_loader.debug_mode = True

    test_data = pd.DataFrame({
        "content": ["", None, "   "],
        "publish_time": [pd.Timestamp.now()] * 3
    })

    # 使用正则表达式匹配包含时间戳的异常信息
    with pytest.raises(DataLoaderError, match=r".*情感分析失败.*"):
        sentiment_loader._add_sentiment_scores(test_data)


# 测试用例1: 验证情感分析模型完全失败处理
def test_sentiment_loader_model_failure():
    loader = SentimentNewsLoader()
    test_data = pd.DataFrame({
        "content": ["正常内容1", "正常内容2"],
        "publish_time": ["2023-01-01", "2023-01-02"]
    })

    with patch.object(SentimentAnalyzer, 'snownlp_sentiment', side_effect=Exception("Model error")):
        result = loader._add_sentiment_scores(test_data)
        assert result["sentiment"].isna().all()


# 测试用例2: 验证多源数据适配处理
def test_news_loader_em_source_adaptation():
    loader = FinancialNewsLoader(source="em")
    test_data = pd.DataFrame({
        'em_title': ["Test Title"],
        'em_content': ["Test Content"],
        'em_time': ["2023-01-01 12:00:00"]
    })

    with patch("akshare.stock_info_global_em", return_value=test_data):
        processed = loader._fetch_raw_data("2020-01-01", "2023-01-01")
        assert "标题" in processed.columns
        assert "内容" in processed.columns


# 测试用例3: 验证HTML清洗异常处理
def test_news_loader_html_cleaning_failure():
    loader = FinancialNewsLoader()
    test_html = "<div>Broken<html><body>Unclosed tags"
    cleaned = loader._clean_single_html(test_html)
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


def test_news_loader_cache_invalid_due_to_age(tmp_path, mocker):
    """测试缓存过期时重新获取数据"""
    loader = FinancialNewsLoader(save_path=tmp_path, cache_days=0)
    file_path = tmp_path / "news_2023-01-01_2023-01-02.csv"
    file_path.write_text("publish_time,content,title\n2023-01-01,Test,Title")

    # 修改文件时间使其过期
    past_time = time.time() - 86400 * 8
    os.utime(file_path, (past_time, past_time))

    mocker.patch.object(loader, '_fetch_raw_data', return_value=pd.DataFrame({
        '标题': ['Test'],
        '内容': ['Content'],
        '发布日期': ['2023-01-01'],
        '发布时间': ['12:00:00']
    }))

    result = loader.load_data("2023-01-01", "2023-01-02")
    assert not result.empty
    assert 'publish_time' in result.columns


def test_news_sentiment_analysis_exception(sentiment_loader, mocker):
    """测试情感分析异常处理"""
    # 启用调试模式
    sentiment_loader.debug_mode = True

    # 模拟情感分析器返回 NaN
    mocker.patch("src.features.processors.sentiment.SentimentAnalyzer.snownlp_sentiment",
                 return_value=np.nan)

    test_data = pd.DataFrame({
        "content": ["Test content"],
        "publish_time": [pd.Timestamp('2023-01-01')]
    })

    # 使用 pytest.raises 捕获预期异常
    with pytest.raises(DataLoaderError, match=r".*情感分析失败.*"):
        sentiment_loader._add_sentiment_scores(test_data)


# 测试多语言HTML清洗
def test_multilingual_html_cleaning(news_loader):
    """测试多语言HTML清洗处理"""
    test_html = "<div>中文内容<script>alert(1)</script></div>"
    cleaned = news_loader._clean_single_html(test_html)
    assert "中文内容" in cleaned
    assert "alert" not in cleaned


# 测试时区处理
def test_timezone_handling(news_loader):
    """测试时区敏感时间解析"""
    test_data = pd.DataFrame({
        "publish_date": ["2023-01-01"],
        "publish_time": ["23:59:59+08:00"]  # 带时区信息
    })

    processed = news_loader._parse_datetime(test_data)
    assert pd.api.types.is_datetime64_any_dtype(processed["publish_time"])


# 测试不同新闻源适配
@pytest.mark.parametrize("source", ["cls", "sina", "em"])
def test_news_sources(source, tmp_path):
    """测试不同新闻源适配"""
    loader = FinancialNewsLoader(source=source, save_path=tmp_path)

    # 提供完整的模拟数据
    mock_data = {
        "cls": pd.DataFrame({
            "标题": ["Test Title"],
            "内容": ["Test Content"],
            "发布日期": ["2023-01-01"],
            "发布时间": ["10:00:00"]
        }),
        "sina": pd.DataFrame({
            "content": ["Test Content"],  # sina源使用英文字段名
            "time": ["2023-01-01 10:00:00"]
        }),
        "em": pd.DataFrame({
            "em_title": ["Test Title"],
            "em_content": ["Test Content"],
            "em_time": ["2023-01-01 10:00:00"]
        })
    }[source]

    with patch(f"akshare.stock_info_global_{source}", return_value=mock_data):
        df = loader.load_data("2023-01-01", "2023-01-05")
        assert not df.empty
        assert "publish_time" in df.columns
        assert "content" in df.columns


