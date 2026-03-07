import queue
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.ml.deep_learning.core import integration_tests
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class DummyBatch:
    def __init__(self, data, metadata=None):
        self.data = data
        self.metadata = metadata or {}


class DummyPipeline:
    def __init__(self):
        self.started = False
        self.data_source = SimpleNamespace(connect=lambda: True, disconnect=lambda: None)

    def create_data_source(self, config):
        return self.data_source

    def set_data_source(self, source):
        pass

    def start_pipeline(self):
        self.started = True
        return True

    def process_data_stream(self, q, max_batches=1):
        yield DummyBatch(
            data=np.ones((5, 2)),
            metadata={"features_engineered": True}
        )

    def stop_pipeline(self):
        self.started = False


class DummyManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir


class DummyPreprocessor:
    pass


class DummyService:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def stop_service(self):
        pass


def test_end_to_end_pipeline_simplified(monkeypatch, tmp_path):
    monkeypatch.setenv("ML_CORE_FORCE_FALLBACK", "1")
    import src.ml.deep_learning.core.integration_tests as module

    monkeypatch.setattr(
        module,
        "create_financial_data_pipeline",
        lambda: DummyPipeline(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "create_test_financial_data",
        lambda n=200: pd.DataFrame(
            {"timestamp": [0], "close": [1.0], "open": [1.0], "high": [1.0], "low": [1.0], "volume": [1.0]}
        ),
    )
    monkeypatch.setattr(
        module,
        "DeepLearningManager",
        DummyManager,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "DataPreprocessor",
        DummyPreprocessor,
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "ModelService",
        DummyService,
        raising=False,
    )

    test_case = module.TestIntegration()
    test_case.setUp()
    try:
        module.create_financial_data_pipeline = lambda: DummyPipeline()
        test_case.test_end_to_end_pipeline()
    finally:
        test_case.tearDown()

