import pandas as pd
import pytest

from src.ml.deep_learning.core.data_preprocessor import DataPreprocessor
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def test_data_preprocessor_dropna():
    data = pd.DataFrame({"a": [1, None, 2], "b": [3, 4, None]})
    preprocessor = DataPreprocessor({"dropna": True})
    result = preprocessor.preprocess(data)
    assert result.isna().sum().sum() == 0


def test_data_preprocessor_raises_on_empty():
    preprocessor = DataPreprocessor()
    with pytest.raises(ValueError):
        preprocessor.preprocess(pd.DataFrame())

