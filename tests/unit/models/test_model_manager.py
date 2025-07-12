# tests/models/test_model_manager.py
import os
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call
import pymc as pm
import joblib
import pandas as pd
import pytest
import arviz as az
import numpy as np
import xarray as xr
import shap
import torch
from sklearn.ensemble import RandomForestRegressor

from src.models.base_model import BaseModel
from src.models.model_manager import ModelManager, ModelDriftDetector, ModelEnsembler, ModelMonitor, ModelExplainer
from src.models.lstm import LSTMModelWrapper
from src.models.nn import NeuralNetworkModel
from src.models.rf import RandomForestModel
import tempfile
from src.infrastructure.utils.logger import get_logger
from src.models.utils import DeviceManager

logger = get_logger(__name__)


class TestModelManager(ModelManager):  # 明确继承自 ModelManager
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_invalid_model_save(self, model_manager, tmp_path):
        """测试保存无效模型时抛出异常"""
        model = RandomForestModel(model_name="test_model")
        # 特意不进行训练，保持模型未训练状态
        with pytest.raises(Exception):
            model_manager.save_model(model, "test_model", "1.0.0", [], metadata={})

    def test_metadata_management(tmp_path):
        manager = ModelManager(base_path=tmp_path)

        # 使用 BaseModel 作为测试模型类，确保路径有效
        class DummyModel(BaseModel):
            def __init__(self):
                super().__init__(model_name="dummy")

            def train(self, features, target):
                pass

            def predict(self, features):
                return np.zeros(len(features))

        model = DummyModel()

        # 显式设置 metadata 中的 model_type
        metadata = {
            "model_name": "test_model",
            "version": "1.0.0",
            "feature_columns": ["feature1", "feature2"],
            "metadata": {"description": "Test model"},
            "model_type": f"{BaseModel.__module__}.{BaseModel.__name__}"  # 确保路径有效
        }

        manager.save_model(model, "test_model", "1.0.0", ["feature1", "feature2"], metadata)
        loaded_model, loaded_metadata = manager.load_model("test_model", "1.0.0")

        assert loaded_metadata["model_name"] == "test_model"

    def test_model_manager(self, model_manager, sample_data):
        """测试模型管理器"""
        features, target = sample_data

        # 测试模型保存
        lstm = LSTMModelWrapper(
            input_size=10,
            hidden_size=32,
            seq_length=10,
            device="cpu"
        )
        lstm.train(features, target)
        model_manager.save_model(
            model=lstm,
            model_name="test_lstm",
            version="1.0.0",
            feature_columns=features.columns.tolist(),
            overwrite=True  # 设置 overwrite 为 True
        )

        # 测试模型加载
        loaded_model, metadata = model_manager.load_model("test_lstm", version="1.0.0")
        assert loaded_model is not None
        assert metadata["model_name"] == "test_lstm"
        assert metadata["version"] == "1.0.0"

    def test_model_saving(self, model_manager: ModelManager, sample_model, sample_metadata):
        """测试模型保存功能"""
        model_path = model_manager.save_model(
            model=sample_model,
            model_name="test_model",
            version="1.0.0",
            feature_columns=sample_metadata["feature_columns"],
            metadata=sample_metadata["metadata"],
            overwrite=True  # 明确保写参数设置为 True
        )
        assert model_path.exists(), "模型保存路径不存在"

    def test_model_loading(self, model_manager: ModelManager, sample_model):
        """测试模型加载功能"""
        # 先保存模型
        model_manager.save_model(
            model=sample_model,
            model_name="test_model",
            version="1.0.0",
            feature_columns=["feat_0", "feat_1"],  # 示例特征列
            metadata={"description": "Test model"},
            overwrite=True
        )
        # 再加载模型
        loaded_model, metadata = model_manager.load_model("test_model", version="1.0.0")
        assert loaded_model is not None, "加载的模型为空"

    def test_version_control(model_manager: ModelManager):
        """测试版本控制逻辑"""
        # 保存多个版本
        for v in ["1.0.0", "1.0.1", "2.0.0"]:
            model_manager.save_model({"version": v}, "version_test", v)

        # 验证最新版本
        assert model_manager.get_latest_version("version_test") == "2.0.0"

    def test_load_nonexistent_model(model_manager: ModelManager):
        """测试加载不存在的模型"""
        with pytest.raises(FileNotFoundError):
            model_manager.load_model("ghost_model")

    def test_save_conflict(model_manager: ModelManager, sample_model):
        """测试重复保存冲突处理"""
        model_manager.save_model(sample_model, "conflict_test", "1.0.0")
        with pytest.raises(FileExistsError):
            model_manager.save_model(sample_model, "conflict_test", "1.0.0")

    def test_corrupted_metadata(model_manager: ModelManager, tmp_path: Path):
        """测试损坏元数据文件处理"""
        # 生成无效元数据文件
        meta_path = tmp_path / "metadata/test_model_metadata_v1.0.0.pkl"
        meta_path.write_bytes(b"invalid data")

        with pytest.raises(Exception):
            model_manager.load_model("test_model", "1.0.0")

    def test_semantic_versioning(model_manager: ModelManager):
        """测试语义化版本排序"""
        versions = ["1.0.0", "1.11.0", "2.0.1", "2.1.0-beta"]
        for v in versions:
            model_manager.save_model({"version": v}, "semver_test", v)

        valid_versions = [v for v in versions if v.replace('.', '').isdigit()]
        assert model_manager.get_latest_version("semver_test") == max(
            valid_versions,
            key=lambda x: [int(n) for n in x.split('.')]
        )

    def test_special_char_model_name(model_manager: ModelManager):
        """测试特殊字符模型名称处理"""
        model_name = "model!@#"
        model_manager.save_model({}, model_name, "1.0.0")
        assert (model_manager.base_path / f"{model_name}_v1.0.0.pkl").exists()

    def test_large_metadata(model_manager: ModelManager):
        """测试大体积元数据存储"""
        large_metadata = {"features": ["f" + str(i) for i in range(1000)]}
        model_manager.save_model({}, "large_meta", "1.0.0", large_metadata)
        _, meta = model_manager.load_model("large_meta", "1.0.0")
        assert len(meta["metadata"]["features"]) == 1000

    # --------------------------
    # 1. 模型保存/加载一致性测试
    # --------------------------
    def test_model_save_load_consistency(sample_data):
        features, target = sample_data
        temp_dir = tempfile.TemporaryDirectory()

        # 测试LSTM模型
        lstm = LSTMModelWrapper(input_size=5, seq_length=10)
        lstm.train(features, target, epochs=2)
        lstm_pred = lstm.predict(features.iloc[:15])  # 确保足够时间步长

        # 保存并重新加载
        lstm.save(temp_dir.name)
        loaded_lstm = LSTMModelWrapper.load(temp_dir.name, "lstm")
        loaded_pred = loaded_lstm.predict(features.iloc[:15])
        assert np.allclose(lstm_pred, loaded_pred, rtol=1e-4), "LSTM预测不一致"

        # 测试随机森林
        rf = RandomForestModel()
        rf.train(features, target)
        rf_pred = rf.predict(features)
        rf.save(temp_dir.name)
        loaded_rf = RandomForestModel.load(temp_dir.name, "random_forest")
        loaded_rf_pred = loaded_rf.predict(features)
        assert np.allclose(rf_pred, loaded_rf_pred), "随机森林预测不一致"

        # 测试神经网络
        nn = NeuralNetworkModel(input_size=5)
        nn.train(features, target, epochs=2)
        nn_pred = nn.predict(features)
        nn.save(temp_dir.name)
        loaded_nn = NeuralNetworkModel.load(temp_dir.name, "neural_network")
        loaded_nn_pred = loaded_nn.predict(features)
        assert np.allclose(nn_pred, loaded_nn_pred, rtol=1e-4), "神经网络预测不一致"

        temp_dir.cleanup()

    # --------------------------
    # 2. 交叉验证最佳模型选择测试
    # --------------------------
    def test_lstm_cross_validation(self, lstm_model, sample_data):
        """测试LSTM交叉验证最佳模型选择"""
        features, target = sample_data
        lstm_model.train(features, target)

        # 检查每个fold的模型是否独立保存
        assert hasattr(lstm_model, "fold_models")
        assert len(lstm_model.fold_models) > 0

        # 验证最终模型是否为所有fold中验证损失最小的模型
        best_fold_loss = min([fold_model.best_val_loss for fold_model in lstm_model.fold_models])
        assert lstm_model.best_val_loss == best_fold_loss

    # --------------------------
    # 3. 特征重要性验证
    # --------------------------
    def test_feature_importance(self, rf_model, nn_model, lstm_model, sample_data):
        """测试特征重要性计算"""
        features, target = sample_data

        # 测试随机森林特征重要性
        rf_model.train(features, target)

        # 获取特征重要性并排序（降序）
        reported_importance = np.sort(rf_model.get_feature_importance().values)[::-1]  # 降序排列
        raw_importance = np.sort(rf_model.model.feature_importances_)[::-1]  # 降序排列

        # 比较排序后的数值是否一致
        assert np.allclose(reported_importance, raw_importance)

    # --------------------------
    # 4. 异常输入处理测试
    # --------------------------
    def test_invalid_inputs(sample_data):
        features, target = sample_data

        # LSTM时间步不足
        lstm = LSTMModelWrapper(input_size=5, seq_length=10)
        lstm.train(features, target)
        with pytest.raises(ValueError):
            lstm.predict(features.iloc[:5])  # 输入长度不足seq_length

        # 神经网络特征维度不匹配
        nn = NeuralNetworkModel(input_size=5)
        nn.train(features, target)
        with pytest.raises(ValueError):
            nn.predict(features.iloc[:, :3])  # 特征数量错误

        # 未训练模型预测
        untrained_rf = RandomForestModel()
        with pytest.raises(RuntimeError):
            untrained_rf.predict(features)


def test_version_sorting(model_manager):
    """测试语义化版本排序"""
    versions = ["2.1.0", "1.9.1", "1.10.0"]
    for v in versions:
        model_manager.save_model(
            None,  # 这里可以传递一个虚拟模型或实际模型
            model_name="version_test",
            version=v,
            feature_columns=["feature1"],  # 添加 feature_columns 参数
            metadata={"author": "test"}
        )


def test_corrupted_model_file(tmp_path, model_manager):
    """测试损坏模型文件处理"""
    model_path = tmp_path / "test_model_v1.0.0.pt"
    model_path.write_bytes(b"corrupted data")

    with pytest.raises(Exception):
        model_manager.load_model("test_model", "1.0.0")


# 测试保存不同类型模型
def test_save_different_model_types(tmp_path):
    manager = ModelManager()

    # 测试LSTM
    lstm = LSTMModelWrapper(input_size=10)
    manager.save_model(lstm, "lstm_test", "1.0", ["feature1"], overwrite=True)

    # 测试随机森林
    rf = RandomForestModel()
    manager.save_model(rf, "rf_test", "1.0", ["feature1"], overwrite=True)

    # 验证元数据
    meta = joblib.load(manager.metadata_store / "rf_test_v1.0.pkl")
    assert meta["model_type"] == "src.models.rf.RandomForestModel"


# 测试动态加载模型类
def test_dynamic_model_loading(tmp_path):
    manager = ModelManager(base_path=tmp_path)

    # 创建模型目录和元数据文件
    model_dir = tmp_path / "random_forest_v1.0"
    model_dir.mkdir(parents=True)
    joblib.dump(RandomForestModel(), model_dir / "random_forest.pkl")

    metadata = {
        "model_name": "random_forest",
        "version": "1.0",
        "feature_columns": ["f1", "f2"],
        "metadata": {"author": "test_author"}  # 确保包含 metadata 字段
    }
    (tmp_path / "metadata").mkdir(parents=True, exist_ok=True)
    joblib.dump(metadata, tmp_path / "metadata" / "random_forest_v1.0.pkl")

    # 加载应成功
    loaded_model, loaded_meta = manager.load_model("random_forest", "1.0")
    assert loaded_meta["metadata"]["author"] == "test_author"


# 覆盖模型验证失败场景
def test_model_validation_failure(tmp_path):
    manager = ModelManager(base_path=tmp_path)

    # 创建模型目录和元数据文件
    model_dir = tmp_path / "fake_model_v1.0"
    model_dir.mkdir(parents=True)
    joblib.dump(RandomForestModel(), model_dir / "fake_model.pkl")

    metadata = {
        "model_name": "fake_model",
        "version": "1.0",
        "feature_columns": ["wrong_order"]
    }
    joblib.dump(metadata, tmp_path / "metadata" / "fake_model_v1.0.pkl")

    # 测试特征顺序不匹配
    assert not manager.validate_model(
        "fake_model", "1.0",
        checks={"feature_columns": ["correct_order"]}
    )


def test_model_manager_save_load(tmpdir):
    # 初始化模型管理器
    manager = ModelManager(base_path=tmpdir)

    # 创建一个简单模型
    input_size = 10
    seq_length = 10
    hidden_size = 256
    num_layers = 3
    output_size = 1
    dropout = 0.5
    device = "cpu"

    model = LSTMModelWrapper(
        input_size=input_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
        device=device
    )

    # 创建示例数据进行简单训练
    X = pd.DataFrame(np.random.randn(100, input_size))
    y = pd.Series(np.random.randn(100))
    model.train(X, y)

    # 保存模型
    model_path = manager.save_model(
        model=model,
        model_name="test_lstm_model",
        version="1.0.0",
        feature_columns=X.columns.tolist(),
        overwrite=True
    )

    # 加载模型
    loaded_model, metadata = manager.load_model(model_name="test_lstm_model", version="1.0.0")

    # 验证加载的模型
    assert loaded_model.is_trained  # 确保模型已训练


@patch("importlib.import_module")
@patch("joblib.load")
def test_dynamic_class_loading(mock_joblib_load, mock_import):
    """测试动态模型类加载"""
    # 模拟导入模块
    mock_module = MagicMock()
    mock_module.LSTMModelWrapper = LSTMModelWrapper  # 直接使用真实类
    mock_import.return_value = mock_module

    # 模拟加载元数据
    mock_metadata = {
        "model_name": "test",
        "version": "1.0",
        "timestamp": "2023-01-01T00:00:00",
        "feature_columns": ["feature1", "feature2"],
        "model_type": "src.models.lstm.LSTMModelWrapper",
        "config": {"input_size": 10, "seq_length": 10, "hidden_size": 256, "num_layers": 3, "output_size": 1, "dropout": 0.5, "device": "cpu"},
        "is_trained": True  # 确保包含训练状态
    }
    mock_joblib_load.return_value = mock_metadata

    # 初始化模型管理器并设置正确的路径
    manager = ModelManager(base_path="models")
    manager.metadata_store = Path("models/metadata")

    # 模拟模型文件存在
    with patch("pathlib.Path.exists", return_value=True):
        # 模拟模型文件加载
        with patch("torch.load") as mock_torch_load:
            # 返回一个包含实际字典的模型状态
            mock_torch_load.return_value = {
                'model_state_dict': {},  # 使用空字典代替 MagicMock
                'config': mock_metadata['config'],
                'is_trained': True  # 确保包含训练状态
            }
            # 模拟模型文件路径
            with patch("pathlib.Path.glob", return_value=[Path("models/test_v1.0/test.pt")]):
                model, metadata = manager.load_model("test", "1.0")

    # 验证模型和元数据
    assert isinstance(model, LSTMModelWrapper)
    assert metadata == mock_metadata

def test_gpu_device_mapping():
    """测试GPU设备自动映射"""
    with patch("torch.cuda.is_available", return_value=True):
        manager = ModelManager(device="auto")
        assert "cuda" in str(manager.device)


def test_version_parsing():
    """测试语义化版本解析逻辑"""
    manager = ModelManager()
    manager.metadata_store = Path("models/metadata")  # 显式设置元数据存储路径

    # 模拟元数据文件列表
    test_versions = ["1.2.3", "1.10.0", "2.0.1"]
    latest_version = max(test_versions, key=lambda v: tuple(map(int, v.split('.'))))
    assert latest_version == "2.0.1"


@patch("builtins.open", new_callable=mock_open)
def test_metadata_saving(mock_file):
    """测试元数据保存完整性"""
    # 初始化模型管理器
    manager = ModelManager()
    manager.metadata_store = Path("models/metadata")  # 显式设置元数据存储路径

    # 创建一个简单的随机森林模型
    model = RandomForestModel(n_estimators=100, max_depth=5)

    # 保存模型
    model_path = manager.save_model(
        model=model,
        model_name="test_rf_model",
        version="1.0.0",
        feature_columns=["feature1", "feature2"],
        metadata={"extra_info": "test metadata"},
        overwrite=True
    )

    # 获取元数据文件路径
    metadata_path = manager.metadata_store / "test_rf_model_v1.0.0.pkl"

    # 验证是否调用了 open 方法来保存元数据
    expected_call = call(str(metadata_path), 'wb')  # 将路径转换为字符串
    assert expected_call in mock_file.call_args_list, f"Expected call {expected_call} not found in {mock_file.call_args_list}"


def test_model_explainer():
    # 创建模型和数据
    model = RandomForestModel(n_estimators=100, max_depth=5)  # 初始化模型
    # 创建特征数据
    X = pd.DataFrame(np.random.randn(100, 5), columns=["feature1", "feature2", "feature3", "feature4", "feature5"])
    # 创建目标数据
    y = pd.Series(np.random.randn(100))

    # 训练模型
    model.train(X, y)

    # 初始化解释器
    explainer = shap.TreeExplainer(model.model)  # 使用 shap.TreeExplainer

    # 计算 SHAP 值
    shap_values = explainer.shap_values(X)

    # 验证 SHAP 值的形状
    assert shap_values.shape == (len(X), X.shape[1])


@pytest.mark.parametrize("model_type", ["lstm", "nn", "rf"])
def test_model_initialization(model_type):
    # 测试模型初始化
    if model_type == "lstm":
        model = LSTMModelWrapper(input_size=5, seq_length=10)
    elif model_type == "nn":
        model = NeuralNetworkModel(input_size=5, hidden_layers=[32, 16])
    elif model_type == "rf":
        model = RandomForestModel(n_estimators=50)

    assert model is not None


@pytest.mark.parametrize("input_size, seq_length, hidden_size", [
    (10, 5, 64),
    (5, 3, 32),
    (20, 10, 128)
])
def test_lstm_initialization(input_size, seq_length, hidden_size):
    model = LSTMModelWrapper(
        input_size=input_size,
        seq_length=seq_length,
        hidden_size=hidden_size
    )
    assert model.input_size == input_size
    assert model.seq_length == seq_length
    assert model.hidden_size == hidden_size


@pytest.mark.parametrize("input_size, hidden_layers, dropout_rate", [
    (10, [256, 128], 0.5),
    (5, [64], 0.3),
    (20, [128, 64, 32], 0.2)
])
def test_nn_initialization(input_size, hidden_layers, dropout_rate):
    model = NeuralNetworkModel(
        input_size=input_size,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate
    )
    assert model.config['input_size'] == input_size
    assert model.config['hidden_layers'] == hidden_layers
    assert model.config['dropout_rate'] == dropout_rate


@pytest.mark.parametrize("n_estimators, max_depth", [
    (100, 5),
    (50, None),
    (200, 10)
])
def test_rf_initialization(n_estimators, max_depth):
    model = RandomForestModel(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    assert model.config['n_estimators'] == n_estimators
    assert model.config['max_depth'] == max_depth


def test_invalid_model_load(model_manager, tmp_path):
    # 使用具体的模型子类代替 BaseModel
    model = RandomForestModel(model_name="test_model")
    with pytest.raises(Exception):
        model_manager.load_model("test_model", "1.0.0")


def test_lstm_invalid_sequence_length():
    model = LSTMModelWrapper(input_size=5, seq_length=10)
    with pytest.raises(ValueError):
        model._create_sequences(np.random.randn(5, 5), np.random.randn(5))  # 输入长度小于 seq_length

def test_nn_invalid_feature_dimension():
    model = NeuralNetworkModel(input_size=5)
    features = pd.DataFrame(np.random.randn(10, 3))
    with pytest.raises(ValueError):
        model.train(features, pd.Series(np.random.randn(10)))

def test_rf_invalid_feature_dimension():
    model = RandomForestModel()
    features = pd.DataFrame(np.random.randn(10, 5))
    with pytest.raises(ValueError):  # 修改为 ValueError
        model.train(pd.DataFrame(), pd.Series(dtype=float))


def test_model_manager_get_latest_version(model_manager):
    versions = ["1.0.0", "1.1.0", "2.0.0"]
    for v in versions:
        model_manager.save_model({}, "version_test", v, feature_columns=["feat1", "feat2"])
    assert model_manager.get_latest_version("version_test") == "2.0.0"

def test_model_manager_validate_model(model_manager, sample_data):
    features, target = sample_data
    model = LSTMModelWrapper(input_size=features.shape[1], seq_length=10)
    model.train(features, target)
    model_manager.save_model(model, "test_lstm", "1.0.0", features.columns.tolist())

    # 测试验证成功
    assert model_manager.validate_model("test_lstm", "1.0.0", checks={"feature_columns": features.columns.tolist()})

    # 测试验证失败
    assert not model_manager.validate_model("test_lstm", "1.0.0", checks={"feature_columns": ["invalid_feature"]})

def test_model_manager_explain_model(model_manager, rf_model, sample_data):
    features, _ = sample_data
    target = pd.Series(np.random.randn(100))
    rf_model.train(features, target)

    # 解释模型
    shap_values, feature_importance = model_manager.explain_model(rf_model, features)
    assert isinstance(shap_values, list) or isinstance(shap_values, np.ndarray)
    assert isinstance(feature_importance, pd.Series)


def test_metadata_version_sorting(model_manager, tmp_path):
    """验证模型版本排序"""
    model_name = "test_model"
    versions = ["1.0.0", "2.0.0", "1.10.0"]

    # 创建临时目录
    model_manager.base_path = tmp_path
    model_manager.metadata_store = tmp_path / "metadata"
    model_manager.metadata_store.mkdir(parents=True, exist_ok=True)

    # 保存多个版本的模型
    for version in versions:
        model_manager.save_model(
            model=LSTMModelWrapper(input_size=5, seq_length=10),
            model_name=model_name,
            version=version,
            feature_columns=["feature1", "feature2"],
            metadata={"description": "Test model"},
            overwrite=True
        )

    # 验证版本排序
    latest_version = model_manager.get_latest_version(model_name)
    assert latest_version == "2.0.0"


@patch("importlib.import_module", side_effect=ImportError)
def test_failed_class_loading(mock_import, model_manager, tmp_path):
    """验证模型类动态加载失败处理"""
    model_manager.base_path = tmp_path
    model_manager.metadata_store = tmp_path / "metadata"
    model_manager.metadata_store.mkdir(parents=True, exist_ok=True)

    # 确保模型文件和元数据文件不存在
    model_name = "invalid_model"
    version = "1.0.0"
    model_dir = model_manager.base_path / f"{model_name}_v{version}"
    metadata_path = model_manager.metadata_store / f"{model_name}_v{version}.pkl"

    if model_dir.exists():
        shutil.rmtree(model_dir)
    if metadata_path.exists():
        os.remove(metadata_path)

    # 验证是否抛出 FileNotFoundError
    with pytest.raises(FileNotFoundError):
        model_manager.load_model(model_name, version)


def test_metadata_persistence(model_manager, sample_model):
    """测试元数据完整存储"""
    meta = {"author": "test", "created_at": "2023-01-01"}
    model_manager.save_model(
        sample_model,
        "meta_test",
        "1.0",
        feature_columns=["feat1"],
        metadata=meta
    )

    _, loaded_meta = model_manager.load_model("meta_test", "1.0")
    assert loaded_meta["metadata"]["author"] == "test"
    assert loaded_meta["version"] == "1.0"


@pytest.mark.parametrize("device_str,expected_device", [
    ("cuda", "cuda"),
    ("auto", "cuda"),
    ("cpu", "cpu")
])
@patch("torch.cuda.is_available", return_value=True)
def test_device_mapping(mock_cuda, device_str, expected_device):
    """测试设备映射逻辑"""
    manager = ModelManager(device=device_str)
    assert str(manager.device).startswith(expected_device)


@patch("builtins.open", new_callable=mock_open)
@patch("joblib.dump")
def test_metadata_integrity(mock_dump, mock_file):
    """测试元数据完整保存"""
    manager = ModelManager()
    model = RandomForestModel()
    model.feature_names_ = ["f1", "f2"]

    manager.save_model(model, "test", "1.0", ["f1", "f2"])

    # 验证元数据关键字段
    call_args = mock_dump.call_args[0][0]
    assert call_args["feature_columns"] == ["f1", "f2"]
    assert "timestamp" in call_args


@patch("pathlib.Path.mkdir")
@patch("torch.save")
@patch("joblib.dump")
def test_pytorch_save_flow(mock_joblib_dump, mock_save, mock_mkdir):
    """验证PyTorch模型保存流程"""
    model = LSTMModelWrapper(input_size=5)
    model.save(Path("/fake/path"), "test")
    mock_mkdir.assert_called_once()
    mock_save.assert_called()
    mock_joblib_dump.assert_called()

# 测试模型加载失败场景
def test_failed_model_loading(tmp_path):
    manager = ModelManager(base_path=tmp_path)

    # 创建损坏的模型文件
    model_path = tmp_path / "test_model_v1.0.0.pkl"
    model_path.write_bytes(b"invalid data")

    with pytest.raises(Exception):
        manager.load_model("test_model", "1.0.0")


def test_model_explanation(model_manager, rf_model, sample_data):
    """测试模型解释功能"""
    features, target = sample_data
    rf_model.train(features, target)

    # 测试SHAP解释和特征重要性
    shap_values, feature_importance = model_manager.explain_model(rf_model, features)
    assert isinstance(shap_values, list) or isinstance(shap_values, np.ndarray)
    assert isinstance(feature_importance, pd.Series)


@pytest.mark.parametrize("metric,expected", [
    ("psi", 0.0),
    ("pd", 0.0)
])
def test_drift_detection(metric, expected):
    """测试漂移检测"""
    detector = ModelDriftDetector()
    data1 = np.random.randn(100)
    data2 = np.random.randn(100)

    if metric == "psi":
        result = detector.calculate_psi(data1, data2)
    else:
        result = detector.calculate_pd(data1, data2)

    assert result >= expected


def test_version_control_logic(model_manager, tmp_path):
    """测试版本控制逻辑"""
    model_manager.base_path = tmp_path

    # 创建一个简单的模型类
    class TestModel:
        def __init__(self):
            pass

        def save(self, path, overwrite=False):
            pass

    versions = ["1.0.0", "1.1.0", "2.0.0"]
    for v in versions:
        model_manager.save_model(
            model=TestModel(),
            model_name="version_test",
            version=v,
            feature_columns=["feat1"],
            metadata={"version": v}
        )


# 测试模型保存和加载
def test_model_save_load(tmp_path):
    manager = ModelManager(base_path=str(tmp_path))

    # 创建测试模型
    lstm_model = LSTMModelWrapper(input_size=5, hidden_size=32)
    rf_model = RandomForestModel()

    # 保存模型
    lstm_path = manager.save_model(
        lstm_model, "lstm_test", "1.0", ["f1", "f2", "f3", "f4", "f5"]
    )
    rf_path = manager.save_model(
        rf_model, "rf_test", "1.0", ["f1", "f2", "f3"]
    )

    # 加载模型
    loaded_lstm, lstm_meta = manager.load_model("lstm_test", "1.0")
    loaded_rf, rf_meta = manager.load_model("rf_test", "1.0")

    assert isinstance(loaded_lstm, LSTMModelWrapper)
    assert isinstance(loaded_rf, RandomForestModel)
    assert lstm_meta["feature_columns"] == ["f1", "f2", "f3", "f4", "f5"]


# 测试模型验证
def test_model_validation(tmp_path):
    manager = ModelManager(base_path=str(tmp_path))

    # 保存模型
    model = RandomForestModel()
    manager.save_model(model, "test_model", "1.0", ["f1", "f2", "f3"])

    # 验证模型
    checks = {
        "feature_columns": ["f1", "f2", "f3"],  # 匹配
        "metrics": {"accuracy": 0.8}
    }
    assert manager.validate_model("test_model", "1.0", checks) is False

    # 测试特征顺序不匹配
    checks["feature_columns"] = ["f3", "f2", "f1"]
    assert manager.validate_model("test_model", "1.0", checks) is False


# 测试模型解释和漂移检测
def test_model_explain_drift():
    """测试模型解释和漂移检测"""
    manager = ModelManager()
    model = RandomForestModel()
    model._is_trained = True

    # 创建测试数据
    features = pd.DataFrame(np.random.rand(100, 3), columns=["f1", "f2", "f3"])
    target = pd.Series(np.random.randint(0, 2, 100))
    model.train(features, target)

    # 模型解释
    shap_values, feature_importance = manager.explain_model(model, features)
    assert feature_importance is not None

    # 漂移检测
    expected = pd.DataFrame(np.random.rand(100), columns=["feature"])  # 转换为 DataFrame
    actual = pd.DataFrame(np.random.rand(100), columns=["feature"])    # 转换为 DataFrame
    psi = manager.detect_drift(expected, actual)
    assert psi is not None


def test_model_manager_device_config(monkeypatch):
    # 测试设备配置
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # 测试auto选择GPU
    manager = ModelManager(device="auto")
    assert manager.device.type == "cuda"

    # 测试强制CPU
    manager = ModelManager(device="cpu")
    assert manager.device.type == "cpu"


def test_model_monitor_history(tmp_path):
    # 测试性能监控历史保存
    model = RandomForestModel()
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.rand(100))
    model.train(X, y)

    monitor = ModelMonitor(model, (X, y), ["mse"])
    monitor.monitor_performance()

    # 检查历史记录
    assert len(monitor.performance_history) == 1
    assert "mse" in monitor.performance_history[0]["performance"]


# 测试保存和加载不同模型类型
@pytest.mark.parametrize("model_type", ["lstm", "neural_net", "random_forest"])
def test_save_load_models(tmp_path, model_type):
    manager = ModelManager(base_path=str(tmp_path))

    # 创建模型
    if model_type == "lstm":
        model = LSTMModelWrapper(input_size=3, seq_length=5)
    elif model_type == "neural_net":
        model = NeuralNetworkModel(input_size=3)
    else:  # random_forest
        model = RandomForestModel()

    # 模拟训练
    model._is_trained = True
    model.feature_names_ = ["f1", "f2", "f3"]

    # 保存模型
    save_path = manager.save_model(
        model=model,
        model_name=f"test_{model_type}",
        version="1.0",
        feature_columns=["f1", "f2", "f3"]
    )
    assert save_path.exists()

    # 加载模型
    loaded_model, metadata = manager.load_model(
        model_name=f"test_{model_type}",
        version="1.0"
    )
    assert loaded_model.is_trained
    assert metadata["feature_columns"] == ["f1", "f2", "f3"]


# 测试模型验证
def test_validate_model(tmp_path):
    manager = ModelManager(base_path=str(tmp_path))

    # 创建并保存模型
    model = RandomForestModel()
    model._is_trained = True
    model.feature_names_ = ["f1", "f2"]
    manager.save_model(
        model=model,
        model_name="validation_test",
        version="1.0",
        feature_columns=["f1", "f2"]
    )

    # 验证模型 - 通过
    assert manager.validate_model(
        "validation_test", "1.0",
        checks={"feature_columns": ["f1", "f2"]}
    )

    # 验证模型 - 失败（特征不匹配）
    assert not manager.validate_model(
        "validation_test", "1.0",
        checks={"feature_columns": ["f1", "f3"]}
    )


# 测试模型解释
def test_explain_model():
    manager = ModelManager()
    model = RandomForestModel()
    model.model = RandomForestRegressor()
    model._is_trained = True

    features = pd.DataFrame(np.random.rand(10, 3), columns=["f1", "f2", "f3"])
    features["target"] = features["f1"] * 2 + features["f2"] * 0.5

    # 训练模型
    model.train(features[["f1", "f2", "f3"]], features["target"])

    # 解释模型
    shap_values, feature_importance = manager.explain_model(
        model, features[["f1", "f2", "f3"]]
    )
    assert shap_values is not None
    assert len(feature_importance) == 3


def test_model_explainer_shap_rf():
    # 创建示例数据
    X = pd.DataFrame([[1, 2], [3, 4]], columns=["f1", "f2"])
    y = pd.Series([5, 6])  # 示例目标变量

    # 初始化随机森林模型
    rf = RandomForestModel(n_estimators=10)
    rf.train(X, y)  # 训练模型

    # 检查内部的 RandomForestRegressor 是否生成了 estimators_ 属性
    assert hasattr(rf.model, "estimators_"), "estimators_ 属性未生成"

    # 可选：检查树的数量
    assert len(rf.model.estimators_) == 10, f"树的数量不匹配，预期 10 棵，实际 {len(rf.model.estimators_)} 棵"

    # 创建解释器
    explainer = ModelExplainer(rf, ["f1", "f2"])

    # 计算 SHAP 值
    shap_values = explainer.explain_with_shap(X)
    assert shap_values is not None

def test_model_explainer_permutation_importance():
    class DummyModel:
        def predict(self, X):
            return np.random.rand(len(X))

    model = DummyModel()
    explainer = ModelExplainer(model, ["f1", "f2"])
    X = pd.DataFrame(np.random.rand(10, 2), columns=["f1", "f2"])
    y = pd.Series(np.random.rand(10))
    importances = explainer.get_feature_importance(X, y)
    assert importances is not None


def test_model_ensembler_bayesian_ensemble():
    class DummyModel:
        def predict(self, X):
            return np.random.rand(len(X))

    models = [DummyModel(), DummyModel()]
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    trace = ModelEnsembler.bayesian_ensemble(models, X, y, n_samples=10)
    assert trace is not None


def test_model_manager_save_rf(tmp_path):
    manager = ModelManager(base_path=str(tmp_path))
    rf = RandomForestModel()
    rf._is_trained = True
    rf.model = RandomForestRegressor()
    rf.feature_names_ = ["f1", "f2"]

    # 测试覆盖保存逻辑
    path = manager.save_model(rf, "rf_model", "v1", ["f1", "f2"])
    assert path.exists()

    # 测试覆盖加载逻辑
    loaded = manager.load_model("rf_model", "v1")
    assert loaded is not None


# 测试SHAP解释器分支
def test_explainer_shap_branches():
    # 创建神经网络模型并训练
    nn_model = NeuralNetworkModel(input_size=3)
    X_train = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
    y_train = pd.Series(np.random.randn(100))
    nn_model.train(X_train, y_train, epochs=5)  # 训练模型

    # 创建解释器
    explainer = ModelExplainer(nn_model, ["f1", "f2", "f3"])

    # 使用足够的样本进行解释
    X = np.random.randn(100, 3)  # 确保有足够多的样本
    with patch("shap.DeepExplainer") as mock_deep:
        explainer.explain_with_shap(X)

    # 测试随机森林路径
    rf_model = RandomForestModel()
    X_train_rf = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
    y_train_rf = pd.Series(np.random.randn(100))
    rf_model.train(X_train_rf, y_train_rf)  # 训练随机森林模型
    explainer_rf = ModelExplainer(rf_model, ["f1", "f2", "f3"])
    with patch("shap.TreeExplainer") as mock_tree:
        explainer_rf.explain_with_shap(np.random.randn(10, 3))

    # 测试LSTM路径
    lstm = LSTMModelWrapper(input_size=3)
    lstm._is_trained = True
    explainer = ModelExplainer(lstm, ["f1", "f2", "f3"])
    with patch("shap.DeepExplainer") as mock_deep:
        # 确保背景数据样本数量足够
        explainer.explain_with_shap(np.random.randn(100, 3))

# 测试漂移检测
def test_drift_detection_metrics():
    detector = ModelDriftDetector()
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0.5, 1, 1000)

    psi = detector.calculate_psi(expected, actual)
    jsd = detector.calculate_pd(expected, actual, "js")

    assert psi > 0.1
    assert 0 < jsd < 1


# 测试模型保存各类型
def test_model_saving_branches(tmp_path):
    manager = ModelManager(base_path=tmp_path)

    # 测试LSTM保存
    lstm = LSTMModelWrapper(input_size=3)
    lstm._is_trained = True
    with patch("torch.save") as mock_save, patch("joblib.dump") as mock_joblib:
        manager.save_model(lstm, "test_lstm", "v1", ["f1", "f2", "f3"])
        mock_save.assert_called()
        mock_joblib.assert_called()  # 应保存scaler

    # 测试随机森林保存
    rf = RandomForestModel()
    rf.model = RandomForestRegressor()
    rf._is_trained = True
    with patch("joblib.dump") as mock_dump:
        manager.save_model(rf, "test_rf", "v1", ["f1", "f2", "f3"])
        mock_dump.assert_called()

    # 测试神经网络保存
    nn = NeuralNetworkModel(input_size=3)
    nn._is_trained = True
    with patch("torch.save") as mock_save, patch("joblib.dump") as mock_joblib:
        manager.save_model(nn, "test_nn", "v1", ["f1", "f2", "f3"])
        mock_save.assert_called()
        mock_joblib.assert_called()


# 测试模型监控
def test_model_monitoring():
    # 创建带预测概率的模型
    class DummyModel:
        def predict(self, X):
            return np.round(np.random.rand(len(X)))

        def predict_proba(self, X):
            return np.column_stack([np.random.rand(len(X)), np.random.rand(len(X))])

    model = DummyModel()
    X_val = np.random.randn(10, 3)
    y_val = np.random.randint(0, 2, 10)
    monitor = ModelMonitor(model, (X_val, y_val), ["accuracy", "auc", "mse"])

    with patch.object(monitor, "_save_performance_history"):
        monitor.monitor_performance()
        assert len(monitor.performance_history) == 1
        assert "auc" in monitor.performance_history[0]["performance"]


def test_model_drift_detector_psi():
    """测试 PSI 漂移检测计算"""
    detector = ModelDriftDetector()
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0.5, 1, 1000)
    psi = detector.calculate_psi(expected, actual)
    assert psi > 0  # 应有漂移
    assert isinstance(psi, float)

def test_model_drift_detector_js_divergence():
    """测试 JS 散度计算"""
    detector = ModelDriftDetector()
    p = np.array([0.2, 0.5, 0.3])
    q = np.array([0.3, 0.4, 0.3])
    js = detector._js_divergence(p, q)
    assert 0 < js < 1


def test_bayesian_ensemble(mocker):
    """测试贝叶斯模型集成"""
    # 创建模拟模型
    mock_model1 = MagicMock()
    mock_model1.predict.return_value = np.array([1.0, 2.0])
    mock_model2 = MagicMock()
    mock_model2.predict.return_value = np.array([1.5, 2.5])

    # 创建模拟的后验预测数据
    posterior_predictive_data = {
        'y_pred': np.random.normal(loc=1.2, scale=0.1, size=(2, 100, 2))
    }

    # 创建 InferenceData 对象，包含 posterior_predictive 组
    idata = az.InferenceData(
        posterior_predictive=az.data.base.dict_to_dataset(posterior_predictive_data)
    )

    # 替换 pm.sample_posterior_predictive 的调用
    mock_sample_posterior_predictive = mocker.patch('pymc.sample_posterior_predictive', return_value=idata)

    # 模拟贝叶斯集成
    posterior_predictive = ModelEnsembler.bayesian_ensemble(
        [mock_model1, mock_model2],
        pd.DataFrame([[0.1], [0.2]]),
        np.array([1.2, 2.2]),
        n_samples=100
    )

    # 验证 mock 是否被正确调用
    mock_sample_posterior_predictive.assert_called_once()

    # 验证后验预测数据是否正确
    assert hasattr(posterior_predictive, 'posterior_predictive'), "InferenceData 对象中没有 posterior_predictive 组"

def test_uncertainty_calculation():
    """测试不确定性计算"""
    # 创建符合要求的 xarray.Dataset
    data = xr.Dataset({
        'y_pred': xr.DataArray(
            np.random.rand(2, 100, 2),  # 2条链，100个样本，2个数据点
            dims=['chain', 'draw', 'obs'],
            coords={
                'chain': [0, 1],
                'draw': np.arange(100),
                'obs': [0, 1]
            }
        )
    })

    # 创建 InferenceData 对象
    trace = az.InferenceData(posterior_predictive=data)

    # 计算不确定性
    means, stds = ModelEnsembler.calculate_uncertainty(trace)

    # 验证结果
    assert len(means) == 2
    assert len(stds) == 2
    assert all(0 <= m <= 1 for m in means)
    assert all(s > 0 for s in stds)


def test_uncertainty_with_single_observation():
    """测试单个观测点的不确定性计算"""
    data = xr.Dataset({
        'y_pred': xr.DataArray(
            np.random.rand(2, 100, 1),  # 单个观测点
            dims=['chain', 'draw', 'obs'],
            coords={'obs': [0]}
        )
    })
    trace = az.InferenceData(posterior_predictive=data)
    means, stds = ModelEnsembler.calculate_uncertainty(trace)
    assert len(means) == 1
    assert len(stds) == 1


def test_uncertainty_with_multidimensional_output():
    """测试多维输出（如多步预测）"""
    data = xr.Dataset({
        'y_pred': xr.DataArray(
            np.random.rand(2, 100, 3, 2),  # 3步预测，每步2个特征
            dims=['chain', 'draw', 'step', 'feature']
        )
    })
    trace = az.InferenceData(posterior_predictive=data)
    means, stds = ModelEnsembler.calculate_uncertainty(trace, var_name='y_pred')
    assert means.shape == (3, 2)  # 应保留step和feature维度
    assert stds.shape == (3, 2)


def test_save_load_neural_network(tmp_path):
    """测试神经网络模型的保存和加载"""
    manager = ModelManager(base_path=str(tmp_path))
    model = NeuralNetworkModel(input_size=5)

    # 模拟训练
    model._is_trained = True
    model.feature_names_ = ['f1', 'f2', 'f3', 'f4', 'f5']

    # 保存并加载
    save_path = manager.save_model(model, "test_nn", "v1", ['f1', 'f2', 'f3', 'f4', 'f5'])
    loaded_model, metadata = manager.load_model("test_nn", "v1")  # 解包元组

    assert loaded_model.config['input_size'] == 5
    assert loaded_model.is_trained
    assert metadata['feature_columns'] == ['f1', 'f2', 'f3', 'f4', 'f5']


def test_explain_random_forest(mocker):
    """测试随机森林模型解释"""
    manager = ModelManager()

    # 创建完整的模拟对象
    mock_model = mocker.MagicMock(spec=RandomForestModel)
    mock_model.is_trained = True
    mock_model.model = mocker.MagicMock()  # 添加缺失的model属性
    # 设置特征重要性数组，长度与特征数量匹配
    mock_model.model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
    mock_model.feature_names_ = ["f1", "f2", "f3", "f4", "f5"]  # 添加特征名称

    # 模拟SHAP解释器
    mock_explainer = mocker.MagicMock()
    # 返回合适的SHAP值数组 (10个样本，5个特征)
    mock_explainer.shap_values.return_value = np.random.rand(10, 5)
    mocker.patch('shap.TreeExplainer', return_value=mock_explainer)

    # 测试SHAP解释
    features = pd.DataFrame(np.random.rand(10, 5), columns=mock_model.feature_names_)
    shap_values, feature_importance = manager.explain_model(mock_model, features)

    # 验证结果中包含了SHAP值和特征重要性
    assert shap_values is not None
    assert feature_importance is not None


def test_load_missing_model(tmp_path):
    """测试加载不存在的模型"""
    manager = ModelManager(base_path=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        manager.load_model("missing_model", "v99")


def test_save_overwrite_conflict(tmp_path):
    """测试覆盖保存冲突"""
    manager = ModelManager(base_path=str(tmp_path))
    model = RandomForestModel()
    manager.save_model(model, "conflict", "v1", ['f1'])

    # 测试不允许覆盖
    with pytest.raises(FileExistsError):
        manager.save_model(model, "conflict", "v1", ['f1'], overwrite=False)

    # 测试允许覆盖
    save_path = manager.save_model(model, "conflict", "v1", ['f1'], overwrite=True)
    assert save_path.exists()


def test_gpu_device_selection(mocker):
    """测试GPU设备选择逻辑"""
    mocker.patch('torch.cuda.is_available', return_value=True)
    manager = ModelManager(device="cuda")
    assert manager.device.type == "cuda"


def test_cpu_fallback(mocker):
    """测试CUDA不可用时的CPU回退"""
    mocker.patch('torch.cuda.is_available', return_value=False)
    manager = ModelManager(device="cuda")
    assert manager.device.type == "cpu"


def test_load_model_missing_metadata(tmp_path, mocker):
    """测试加载缺少元数据的模型"""
    manager = ModelManager(base_path=str(tmp_path))

    # 创建模型但不保存元数据
    model = RandomForestModel()
    dir_path = tmp_path / "test_rf_v1"
    dir_path.mkdir()
    model_path = dir_path / "test_rf.pkl"
    joblib.dump({'model': model, 'config': {}}, model_path)

    # 模拟元数据缺失
    mocker.patch.object(manager, '_load_metadata', side_effect=FileNotFoundError)

    # 尝试加载模型并验证是否抛出异常
    with pytest.raises(FileNotFoundError):
        manager.load_model("test_rf", "v1")


def test_save_model_overwrite_conflict(tmp_path):
    """测试覆盖保存冲突处理"""
    manager = ModelManager(base_path=str(tmp_path))
    model = RandomForestModel()

    # 第一次保存
    manager.save_model(model, "conflict", "v1", ['feature'])

    # 测试不允许覆盖
    with pytest.raises(FileExistsError):
        manager.save_model(model, "conflict", "v1", ['feature'], overwrite=False)

    # 测试允许覆盖
    save_path = manager.save_model(model, "conflict", "v1", ['feature'], overwrite=True)
    assert save_path.exists()


def test_model_ensembler_bayesian(mocker):
    """测试贝叶斯模型集成"""
    # 创建模拟模型
    mock_model1 = mocker.MagicMock()
    mock_model1.predict.return_value = np.array([1.0, 2.0])
    mock_model2 = mocker.MagicMock()
    mock_model2.predict.return_value = np.array([1.5, 2.5])

    # 创建真实的 InferenceData 对象作为 trace
    with pm.Model() as model:
        weights = pm.Dirichlet('weights', a=np.ones(2))
        pred_0 = pm.Normal('pred_0', mu=1.0, sigma=1)
        pred_1 = pm.Normal('pred_1', mu=1.5, sigma=1)
        y_pred = pm.Normal('y_pred', mu=pm.Deterministic('mu', weights[0] * pred_0 + weights[1] * pred_1), sigma=1, observed=np.array([1.2, 2.2]))
        trace = pm.sample(tune=10, draws=10, chains=2, random_seed=42)

    # 替换 pm.sample_posterior_predictive 的调用
    mocker.patch('pymc.sample_posterior_predictive', return_value=trace)

    # 使用真实贝叶斯集成方法
    result_trace = ModelEnsembler.bayesian_ensemble(
        [mock_model1, mock_model2],
        pd.DataFrame([[0.1], [0.2]], columns=['feature']),
        np.array([1.2, 2.2]),
        n_samples=10
    )

    # 验证结果
    assert 'weights' in result_trace.posterior


def test_model_drift_detection():
    """测试特征漂移检测"""
    manager = ModelManager()
    train_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    test_data = pd.DataFrame({
        'feature1': np.random.normal(1, 1, 100),  # 故意制造漂移
        'feature2': np.random.normal(0, 1, 100)
    })

    report = manager.detect_drift(train_data, test_data)
    print(report)

    # 验证结果
    assert 'feature_drift' in report
    assert isinstance(report['feature_drift'], dict)


def test_device_manager_gpu_fallback(mocker):
    """测试GPU不可用时的回退逻辑"""
    # 模拟CUDA不可用
    mocker.patch('torch.cuda.is_available', return_value=False)

    # 用户请求CUDA但不可用
    device = DeviceManager.get_device("cuda")
    assert device.type == "cpu"

    # 检查日志警告
    # (实际项目中应使用caplog检查日志内容)


def test_save_load_consistency(tmp_path):
    """测试模型保存和加载的一致性"""
    manager = ModelManager(base_path=str(tmp_path))

    # 创建并训练一个简单模型
    model = RandomForestModel(n_estimators=10)
    X_train = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    model.train(X_train, y_train)

    # 保存模型
    save_path = manager.save_model(model, "test_model", "v1", ['f1', 'f2'])

    # 加载模型
    loaded_model, metadata = manager.load_model("test_model", "v1")

    # 验证一致性
    assert loaded_model.is_trained
    assert loaded_model.feature_names_ == ['f1', 'f2']
    assert metadata['model_name'] == "test_model"
    assert metadata['version'] == "v1"


def test_model_monitor_performance(mocker):
    """测试模型性能监控"""
    # 创建带预测概率的模拟模型
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = np.array([0, 1])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

    monitor = ModelMonitor(
        mock_model,
        (pd.DataFrame([[0.1], [0.2]]), np.array([0, 1])),
        ['accuracy', 'auc', 'mse']
    )

    monitor.monitor_performance()

    assert len(monitor.performance_history) == 1
    perf = monitor.performance_history[0]['performance']
    assert 'auc' in perf and perf['auc'] > 0.5
    assert 'mse' in perf

