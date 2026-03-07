import pandas as pd
import pytest

from src.ml.deep_learning.core.data_pipeline import DataPipeline, PipelineSplit
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def test_data_pipeline_split_basic():
    data = pd.DataFrame({"feature": range(10)})
    pipeline = DataPipeline({"test_size": 0.2, "val_size": 0.3})

    split = pipeline.split(data)

    assert isinstance(split, PipelineSplit)
    assert len(split.test) >= 1
    assert len(split.validation) >= 1
    assert len(split.train) + len(split.validation) + len(split.test) == len(data)


def test_data_pipeline_split_small_dataset():
    data = pd.DataFrame({"feature": [1, 2, 3]})
    pipeline = DataPipeline({"test_size": 0.5, "val_size": 0.3})

    split = pipeline.split(data)

    assert len(split.test) == 1  # 至少一条测试记录
    assert len(split.validation) >= 1


def test_data_pipeline_split_raises_on_empty():
    pipeline = DataPipeline()
    with pytest.raises(ValueError):
        pipeline.split(pd.DataFrame())
