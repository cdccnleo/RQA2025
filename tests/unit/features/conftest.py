import importlib
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

# 设置Python路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 移除有问题的导入，避免在conftest.py中触发重型库依赖
# 让测试文件自己处理情感分析模块的导入和Mock
# if "src.features.sentiment.analyzer" in sys.modules:
#     sys.modules.pop("src.features.sentiment.analyzer", None)
# try:
#     importlib.invalidate_caches()
#     importlib.import_module("src.features.sentiment.analyzer")
# except Exception:
#     # 兼容：在某些环境下情感分析子模块可能依赖未就绪，避免阻塞其它用例收集
#     pass

# 使用Mock对象处理导入问题，完全避免在conftest.py中触发重型库依赖
# 让测试文件自己处理真实类的导入，这样可以更好地控制导入时机和错误处理
from unittest.mock import Mock

# 强制导入真实类，如果失败则抛出异常
from src.features.core.config import FeatureConfig, FeatureType
from src.features.core.engine import BaseFeatureProcessor
from src.features.processors.base_processor import ProcessorConfig


@pytest.fixture(scope="session", autouse=True)
def feature_test_log_dir() -> Path:
    """
    创建并返回特征层测试日志目录。

    所有特征层测试统一输出到 ``test_logs/features``，避免污染项目根目录。
    """
    log_dir = Path("test_logs") / "features"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


@pytest.fixture
def sample_price_frame() -> pd.DataFrame:
    """
    提供标准行情数据集，包含 open/high/low/close/volume 列。

    数据使用可复现的线性序列，覆盖正序、逆序与波动场景，可用于大部分技术指标测试。
    """
    index = pd.date_range("2024-01-01 09:30:00", periods=64, freq="15min")
    base = np.linspace(100, 120, num=64)
    noise = np.sin(np.linspace(0, math.pi * 4, num=64)) * 1.5

    data = {
        "open": np.round(base + noise, 2),
        "high": np.round(base + noise + 1.2, 2),
        "low": np.round(base + noise - 1.2, 2),
        "close": np.round(base + np.cos(np.linspace(0, math.pi * 4, num=64)), 2),
        "volume": np.linspace(1_000, 5_000, num=64, dtype=int),
    }

    return pd.DataFrame(data, index=index)


@pytest.fixture
def sparse_price_frame() -> pd.DataFrame:
    """
    提供较小样本集用于边界和异常测试，例如窗口长度大于样本数的情况。
    """
    data = {
        "open": [101.0, 102.5, 103.2],
        "high": [102.0, 103.7, 104.5],
        "low": [100.5, 101.8, 102.9],
        "close": [101.8, 103.1, 103.7],
        "volume": [1100, 1250, 1320],
    }
    index = pd.date_range("2024-02-01", periods=3, freq="D")
    return pd.DataFrame(data, index=index)


@pytest.fixture
def feature_config_basic() -> FeatureConfig:
    """
    返回仅启用技术指标的基础配置，默认开启特征选择与标准化，关闭特征持久化。
    """
    return FeatureConfig(
        feature_types=[FeatureType.TECHNICAL],
        enable_feature_selection=True,
        enable_standardization=True,
        enable_feature_saving=False,
    )


@pytest.fixture
def feature_config_with_sentiment() -> FeatureConfig:
    """
    返回同时启用技术指标与情感特征的配置，用于多处理器协同场景。
    """
    return FeatureConfig(
        feature_types=[FeatureType.TECHNICAL, FeatureType.SENTIMENT],
        enable_feature_selection=True,
        enable_standardization=True,
        enable_feature_saving=False,
        sentiment_types=["news_sentiment", "social_sentiment"],
    )


@pytest.fixture
def feature_config_without_selection(feature_config_basic: FeatureConfig) -> FeatureConfig:
    """
    返回关闭特征选择与标准化的配置，用于验证配置分支逻辑。
    """
    config = FeatureConfig.from_dict(feature_config_basic.to_dict())
    config.enable_feature_selection = False
    config.enable_standardization = False
    return config


@pytest.fixture
def feature_config_parallel(feature_config_basic: FeatureConfig) -> FeatureConfig:
    """
    返回启用并行处理并调整 worker/chunk 配置的 FeatureConfig，用于性能模块测试。
    """
    config = FeatureConfig.from_dict(feature_config_basic.to_dict())
    config.parallel_processing = True
    config.max_workers = 2
    config.chunk_size = 16
    return config


@pytest.fixture
def feature_names_map() -> Dict[str, List[str]]:
    """
    提供常见特征集合，用于参数化测试选择不同指标组合。
    """
    return {
        "trend": ["sma", "ema", "macd"],
        "momentum": ["rsi", "stoch"],
        "volatility": ["atr", "bbands"],
    }


@pytest.fixture
def passthrough_processor():
    """
    返回一个轻量化的示例处理器，用于测试引擎注册流程和处理器调度。

    该处理器直接回传输入数据，并附加 ``feature_passthrough`` 列，避免依赖真实基础设施。
    """
    try:
        # 尝试导入真实的BaseFeatureProcessor
        from src.features.processors.base_processor import BaseFeatureProcessor, ProcessorConfig

        class PassthroughProcessor(BaseFeatureProcessor):
            def __init__(self):
                super().__init__(
                    ProcessorConfig(
                        processor_type="passthrough",
                        feature_params={"passthrough": {}},
                    )
                )

            def process(self, request):  # type: ignore[override]
                data = getattr(request, "data", None)
                if data is None:
                    raise ValueError("请求对象缺少 data 属性")
                result = data.copy()
                if "feature_passthrough" not in result:
                    result["feature_passthrough"] = result["close"].astype(float)
                return result

            def _compute_feature(self, data, feature_name, params):  # type: ignore[override]
                return data["close"].astype(float)

            def _get_feature_metadata(self, feature_name):  # type: ignore[override]
                return {"name": feature_name, "category": "utility"}

            def _get_available_features(self):  # type: ignore[override]
                return ["passthrough"]

        return PassthroughProcessor()
    except ImportError:
        # 如果导入失败，返回Mock对象
        mock_processor = BaseFeatureProcessor()
        mock_processor.process = Mock(return_value=pd.DataFrame())
        return mock_processor


@pytest.fixture
def feature_engine_factory(monkeypatch):
    """
    提供可注入处理器的 FeatureEngine 工厂，自动跳过默认处理器注册。
    """
    from src.features.core.engine import FeatureEngine

    monkeypatch.setattr(FeatureEngine, "_register_default_processors", lambda self: None)

    def _factory(config: Optional[FeatureConfig] = None, processors: Optional[Dict[str, BaseFeatureProcessor]] = None):
        engine = FeatureEngine(config=config)
        if not hasattr(engine, "stats"):
            engine.stats = {
                "processed_features": 0,
                "processing_time": 0.0,
                "errors": 0,
            }
        if processors:
            for name, processor in processors.items():
                engine.processors[name] = processor
        return engine

    return _factory
