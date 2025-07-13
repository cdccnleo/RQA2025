import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from pathlib import Path
import tempfile
import os

# 正常特征数据
@pytest.fixture
def sample_features():
    return pd.DataFrame({
        'close': [100, 101, 99, 102, 103],
        'volume': [1e6, 1.2e6, 0.8e6, 1.1e6, 1.3e6],
        'sentiment': [0.5, 0.6, -0.2, 0.8, 0.1],
        'trend': [1, 1, -1, 1, -1]
    })

# 含异常值的数据
@pytest.fixture
def broken_features(sample_features):
    df = sample_features.copy()
    df.iloc[2:3] = np.nan
    return df

# 模拟特征生成器
@pytest.fixture
def mock_engineer(sample_features):
    mock = Mock()
    mock.generate.return_value = sample_features
    return mock

# 模拟特征选择器
@pytest.fixture
def mock_selector(sample_features):
    mock = Mock()
    mock.select.return_value = sample_features[['close', 'sentiment']]
    return mock

# 模拟标准化器
@pytest.fixture
def mock_standardizer(sample_features):
    mock = Mock()
    mock.transform.return_value = sample_features
    return mock

# 临时模型路径
@pytest.fixture
def tmp_model_path():
    """创建临时模型路径"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "test_model"
        model_path.mkdir(exist_ok=True)
        yield model_path

# Mock ModelManager
@pytest.fixture
def mock_model_manager():
    """模拟ModelManager"""
    mock = Mock()
    mock.validate_model = Mock(return_value=True)
    mock.retrain_model = Mock()
    return mock

# Mock FeatureManager
@pytest.fixture
def mock_feature_manager(tmp_model_path, mock_model_manager):
    """模拟FeatureManager"""
    mock = Mock()
    mock.register = Mock()
    mock.process = Mock()
    mock.model_path = tmp_model_path
    mock.stock_code = "000001"
    mock.model_manager = mock_model_manager
    return mock

# Mock FeatureConfig
@pytest.fixture
def mock_feature_config():
    """模拟FeatureConfig"""
    mock = Mock()
    mock.name = "test_feature"
    mock.feature_type = "TECHNICAL"
    mock.params = {"window": 14}
    mock.dependencies = ["close"]
    return mock

# 技术指标测试数据
@pytest.fixture
def technical_test_data():
    """技术指标测试数据"""
    return pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    })

# 情感分析测试数据
@pytest.fixture
def sentiment_test_data():
    """情感分析测试数据"""
    return pd.DataFrame({
        'content': [
            "公司业绩超出预期，股价大涨！",
            "The company's performance is outstanding!",
            "产品极其糟糕，服务态度差！",
            "This product is terrible and a waste of money."
        ],
        'date': pd.date_range(start="2023-01-01", periods=4)
    })

# Mock BERT模型路径
@pytest.fixture
def tmp_bert_model_path():
    """创建临时BERT模型路径"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        bert_path = Path(tmp_dir) / "bert_model"
        bert_path.mkdir(exist_ok=True)
        # 创建一些假文件
        (bert_path / "config.json").touch()
        (bert_path / "pytorch_model.bin").touch()
        yield bert_path

# 创建真实的FeatureManager实例
@pytest.fixture
def real_feature_manager(tmp_model_path, mock_model_manager):
    """创建真实的FeatureManager实例"""
    from src.features.feature_manager import FeatureManager
    return FeatureManager(
        model_path=str(tmp_model_path),
        stock_code="000001",
        model_manager=mock_model_manager
    )
