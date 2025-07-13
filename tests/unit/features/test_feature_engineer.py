import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
from src.features.feature_engineer import FeatureEngineer

@pytest.fixture
def temp_cache_dir(tmp_path):
    """临时缓存目录fixture"""
    cache_dir = tmp_path / "feature_cache"
    cache_dir.mkdir()
    return cache_dir

@pytest.fixture
def mock_processors():
    """模拟特征处理器"""
    mock_tech_processor = MagicMock()
    mock_sentiment_analyzer = MagicMock()

    return {
        "tech_processor": mock_tech_processor,
        "sentiment_analyzer": mock_sentiment_analyzer
    }

@pytest.fixture
def feature_engineer(temp_cache_dir, mock_processors):
    """特征工程测试实例"""
    with patch('src.feature.feature_engineer.TechnicalProcessor', return_value=mock_processors["tech_processor"]), \
         patch('src.feature.feature_engineer.SentimentAnalyzer', return_value=mock_processors["sentiment_analyzer"]):

        fe = FeatureEngineer(cache_dir=str(temp_cache_dir))
        yield fe

        # 清理
        fe.executor.shutdown()

def test_technical_feature_calculation(feature_engineer, mock_processors):
    """测试技术指标特征计算"""
    # 准备测试数据
    symbol = "600000"
    price_data = pd.DataFrame({
        'open': [10, 11, 12],
        'high': [11, 12, 13],
        'low': [9, 10, 11],
        'close': [10.5, 11.5, 12.5],
        'volume': [1000, 2000, 3000]
    })
    indicators = ["SMA", "RSI"]

    # 设置模拟返回值
    mock_processors["tech_processor"].calculate.side_effect = [
        pd.DataFrame({"SMA": [10.2, 11.2, 12.2]}),
        pd.DataFrame({"RSI": [60, 65, 70]})
    ]

    # 计算特征
    features = feature_engineer.calculate_technical_features(
        symbol, price_data, indicators
    )

    # 验证结果
    assert not features.empty
    assert "SMA" in features.columns
    assert "RSI" in features.columns
    assert len(features) == 3

    # 验证处理器调用
    assert mock_processors["tech_processor"].calculate.call_count == 2

def test_sentiment_feature_calculation(feature_engineer, mock_processors):
    """测试情感特征计算"""
    # 准备测试数据
    text_data = pd.DataFrame({
        'text': ["good news", "bad news", "neutral"],
        'timestamp': pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    })
    models = ["BERT", "TextBlob"]

    # 设置模拟返回值
    mock_processors["sentiment_analyzer"].analyze.side_effect = [
        pd.DataFrame({"BERT_score": [0.8, 0.2, 0.5]}),
        pd.DataFrame({"TextBlob_score": [0.7, 0.3, 0.4]})
    ]

    # 计算特征
    features = feature_engineer.calculate_sentiment_features(
        text_data, models
    )

    # 验证结果
    assert not features.empty
    assert "BERT_score" in features.columns
    assert "TextBlob_score" in features.columns
    assert len(features) == 3

    # 验证处理器调用
    assert mock_processors["sentiment_analyzer"].analyze.call_count == 2

def test_feature_caching(feature_engineer, temp_cache_dir):
    """测试特征缓存功能"""
    symbol = "000001"
    test_features = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    cache_key = feature_engineer._get_cache_key("test_func", symbol, "arg1")

    # 保存到缓存
    feature_engineer._save_to_cache(cache_key, test_features)

    # 验证缓存文件存在
    cache_file = temp_cache_dir / f"{cache_key}.pkl"
    assert cache_file.exists()

    # 验证元数据更新
    assert cache_key in feature_engineer.cache_metadata

    # 从缓存读取
    retrieved = feature_engineer._get_from_cache(cache_key)
    pd.testing.assert_frame_equal(retrieved, test_features)

def test_batch_feature_calculation(feature_engineer, mock_processors):
    """测试批量特征计算"""
    symbols = ["600000", "000001", "601318"]
    price_data = {
        "600000": pd.DataFrame({'close': [10, 11, 12]}),
        "000001": pd.DataFrame({'close': [20, 21, 22]}),
        "601318": pd.DataFrame({'close': [30, 31, 32]})
    }
    indicators = ["SMA"]

    # 设置模拟返回值
    mock_processors["tech_processor"].calculate.side_effect = [
        pd.DataFrame({"SMA": [10.1, 11.1, 12.1]}),
        pd.DataFrame({"SMA": [20.1, 21.1, 22.1]}),
        pd.DataFrame({"SMA": [30.1, 31.1, 32.1]})
    ]

    # 批量计算特征
    results = feature_engineer.batch_calculate_technical_features(
        symbols, price_data, indicators
    )

    # 验证结果
    assert len(results) == 3
    for symbol in symbols:
        assert symbol in results
        assert not results[symbol].empty
        assert "SMA" in results[symbol].columns

    # 验证每个标的都计算了特征
    assert mock_processors["tech_processor"].calculate.call_count == 3

def test_cache_expiration(feature_engineer, temp_cache_dir):
    """测试缓存过期机制"""
    test_features = pd.DataFrame({"feature": [1, 2, 3]})
    cache_key = feature_engineer._get_cache_key("test_func", "test", "arg1")

    # 保存过期缓存(设置1小时TTL)
    feature_engineer._save_to_cache(cache_key, test_features, ttl=1)

    # 修改元数据中的时间戳为过期
    feature_engineer.cache_metadata[cache_key]["timestamp"] = datetime.now() - timedelta(hours=2)

    # 尝试获取应返回None
    assert feature_engineer._get_from_cache(cache_key) is None

    # 验证缓存文件被删除
    cache_file = temp_cache_dir / f"{cache_key}.pkl"
    assert not cache_file.exists()
    assert cache_key not in feature_engineer.cache_metadata

def test_clear_cache(feature_engineer, temp_cache_dir):
    """测试清理缓存"""
    # 创建一些测试缓存
    for i in range(5):
        cache_key = f"test_key_{i}"
        test_features = pd.DataFrame({"feature": [i]})
        feature_engineer._save_to_cache(cache_key, test_features)

    # 修改部分缓存为过期
    old_keys = ["test_key_0", "test_key_1"]
    for key in old_keys:
        feature_engineer.cache_metadata[key]["timestamp"] = datetime.now() - timedelta(days=8)

    # 执行清理(保留7天内)
    feature_engineer.clear_cache(older_than_days=7)

    # 验证过期缓存被删除
    for key in old_keys:
        assert key not in feature_engineer.cache_metadata
        assert not (temp_cache_dir / f"{key}.pkl").exists()

    # 验证有效缓存保留
    assert len(feature_engineer.cache_metadata) == 3
