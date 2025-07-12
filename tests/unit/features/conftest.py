import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

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
