"""
ml_core.py 异常分支测试补充
覆盖 load_model、create_feature_processor、fit_feature_processor、transform_features 等方法的异常处理
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import sys

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.ml.core.ml_core import MLCore


@pytest.fixture
def ml_core_factory(monkeypatch):
    """创建MLCore实例的工厂函数"""
    def failing_adapter():
        raise RuntimeError("adapter unavailable")

    monkeypatch.setattr("src.ml.core.ml_core._get_models_adapter", failing_adapter, raising=False)

    def factory(config=None):
        cfg = {"random_state": 0}
        if config:
            cfg.update(config)
        return MLCore(cfg)

    return factory


def test_load_model_file_not_found(ml_core_factory, tmp_path):
    """测试 load_model - 文件不存在（覆盖 408-410 行）"""
    core = ml_core_factory()
    non_existent_file = tmp_path / "non_existent_model.pkl"
    
    result = core.load_model(str(non_existent_file))
    
    assert result is None


def test_load_model_joblib_load_failure(ml_core_factory, tmp_path, monkeypatch):
    """测试 load_model - joblib.load 失败（覆盖 408-410 行）"""
    core = ml_core_factory()
    model_file = tmp_path / "model.pkl"
    model_file.write_bytes(b"invalid pickle data")
    
    result = core.load_model(str(model_file))
    
    assert result is None


def test_load_model_success(ml_core_factory, tmp_path):
    """测试 load_model - 成功加载（覆盖 396-406 行）"""
    core = ml_core_factory()
    
    # 使用train_model创建并训练一个模型
    import pandas as pd
    import numpy as np
    X = pd.DataFrame({'feature1': np.random.randn(100), 'feature2': np.random.randn(100)})
    y = pd.Series(np.random.randn(100))
    model_id = core.train_model(X, y, model_type='linear', model_params={})
    
    # 保存模型
    model_file = tmp_path / "model.pkl"
    core.save_model(model_id, str(model_file))
    
    # 创建新的core实例并加载模型
    core2 = ml_core_factory()
    loaded_id = core2.load_model(str(model_file))
    
    assert loaded_id is not None
    assert loaded_id in core2.models


def test_create_feature_processor_import_error(ml_core_factory, monkeypatch):
    """测试 create_feature_processor - sklearn 导入失败（覆盖 453-455 行）"""
    core = ml_core_factory()
    
    # Mock sklearn.preprocessing 导入失败
    original_import = __import__
    def mock_import(name, *args, **kwargs):
        if name == "sklearn.preprocessing":
            raise ImportError("sklearn not available")
        return original_import(name, *args, **kwargs)
    
    monkeypatch.setattr("builtins.__import__", mock_import)
    
    with pytest.raises(Exception):  # 应该抛出异常
        core.create_feature_processor("standard")


def test_create_feature_processor_invalid_type(ml_core_factory):
    """测试 create_feature_processor - 不支持的处理器类型（覆盖 438-439 行）"""
    core = ml_core_factory()
    
    with pytest.raises(ValueError, match="不支持的处理器类型"):
        core.create_feature_processor("invalid_type")


def test_create_feature_processor_exception_during_creation(ml_core_factory, monkeypatch):
    """测试 create_feature_processor - 创建过程中异常（覆盖 453-455 行）"""
    core = ml_core_factory()
    
    # Mock StandardScaler 构造函数抛出异常
    with patch("sklearn.preprocessing.StandardScaler", side_effect=RuntimeError("Creation failed")):
        with pytest.raises(RuntimeError):
            core.create_feature_processor("standard")


def test_fit_feature_processor_not_found(ml_core_factory):
    """测试 fit_feature_processor - 处理器不存在（覆盖 460-461 行）"""
    core = ml_core_factory()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    with pytest.raises(ValueError, match="特征处理器.*不存在"):
        core.fit_feature_processor("non_existent_processor", X)


def test_fit_feature_processor_fit_failure(ml_core_factory, monkeypatch):
    """测试 fit_feature_processor - fit 方法失败（覆盖 470-472 行）"""
    core = ml_core_factory()
    
    # 创建一个处理器
    processor_id = core.create_feature_processor("standard")
    
    # Mock fit 方法抛出异常
    processor = core.feature_processors[processor_id]["processor"]
    original_fit = processor.fit
    
    def failing_fit(X, y=None):
        raise RuntimeError("Fit failed")
    
    monkeypatch.setattr(processor, "fit", failing_fit)
    
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(RuntimeError, match="Fit failed"):
        core.fit_feature_processor(processor_id, X)


def test_fit_feature_processor_success(ml_core_factory):
    """测试 fit_feature_processor - 成功拟合（覆盖 457-468 行）"""
    core = ml_core_factory()
    
    processor_id = core.create_feature_processor("standard")
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    # 应该不抛出异常
    core.fit_feature_processor(processor_id, X)
    
    # 验证处理器已拟合
    processor = core.feature_processors[processor_id]["processor"]
    assert hasattr(processor, "mean_")  # StandardScaler 拟合后应该有 mean_ 属性


def test_transform_features_not_found(ml_core_factory):
    """测试 transform_features - 处理器不存在（覆盖 477-478 行）"""
    core = ml_core_factory()
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    with pytest.raises(ValueError, match="特征处理器.*不存在"):
        core.transform_features("non_existent_processor", X)


def test_transform_features_transform_failure(ml_core_factory, monkeypatch):
    """测试 transform_features - transform 方法失败（覆盖 488-490 行）"""
    core = ml_core_factory()
    
    # 创建并拟合一个处理器
    processor_id = core.create_feature_processor("standard")
    X_fit = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    core.fit_feature_processor(processor_id, X_fit)
    
    # Mock transform 方法抛出异常
    processor = core.feature_processors[processor_id]["processor"]
    original_transform = processor.transform
    
    def failing_transform(X):
        raise RuntimeError("Transform failed")
    
    monkeypatch.setattr(processor, "transform", failing_transform)
    
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(RuntimeError, match="Transform failed"):
        core.transform_features(processor_id, X)


def test_transform_features_success(ml_core_factory):
    """测试 transform_features - 成功转换（覆盖 474-486 行）"""
    core = ml_core_factory()
    
    processor_id = core.create_feature_processor("standard")
    X_fit = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    core.fit_feature_processor(processor_id, X_fit)
    
    X_transform = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = core.transform_features(processor_id, X_transform)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == X_transform.shape


def test_transform_features_with_dataframe(ml_core_factory):
    """测试 transform_features - 使用 DataFrame 输入（覆盖 483 行）"""
    core = ml_core_factory()
    
    processor_id = core.create_feature_processor("standard")
    X_fit = pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], columns=["a", "b"])
    core.fit_feature_processor(processor_id, X_fit)
    
    X_transform = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"])
    result = core.transform_features(processor_id, X_transform)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_load_model_missing_model_id(ml_core_factory, tmp_path, monkeypatch):
    """测试 load_model - 加载的模型信息缺少 model_id（覆盖 402 行）"""
    core = ml_core_factory()
    
    # 使用train_model创建并训练一个模型
    import pandas as pd
    import numpy as np
    X = pd.DataFrame({'feature1': np.random.randn(100), 'feature2': np.random.randn(100)})
    y = pd.Series(np.random.randn(100))
    model_id = core.train_model(X, y, model_type='linear', model_params={})
    
    model_file = tmp_path / "model.pkl"
    core.save_model(model_id, str(model_file))
    
    # Mock joblib.load 返回缺少 model_id 的字典
    import joblib
    original_load = joblib.load
    
    def mock_load(filepath):
        data = original_load(filepath)
        data.pop("model_id", None)  # 移除 model_id
        return data
    
    monkeypatch.setattr(joblib, "load", mock_load)
    
    # 创建新的core实例并加载模型
    core2 = ml_core_factory()
    loaded_id = core2.load_model(str(model_file))
    
    # 应该生成新的 model_id
    assert loaded_id is not None
    assert loaded_id in core2.models


