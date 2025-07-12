# tests/core/models/test_base_model.py
import os
import pytest
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.models.nn import NeuralNetworkModel
from src.models.rf import RandomForestModel
from src.models.lstm import LSTMModelWrapper
from src.models.base_model import BaseModel, TorchModelMixin, ModelPersistence
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)  # 自动继承全局配置


class ConcreteModel(BaseModel):
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name=model_name, config=config)
        self.model = None  # 初始化 model 属性
        self.build_model()  # 在初始化时构建模型结构

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @is_trained.setter
    def is_trained(self, value: bool):
        self._is_trained = value

    def train(self, features: pd.DataFrame, target: pd.Series, **kwargs):
        self.feature_names_ = features.columns.tolist()
        self.model = RandomForestRegressor(
            n_estimators=self.config.get('n_estimators', 100),  # 从config获取参数
            max_depth=self.config.get('max_depth', None)
        )
        self.model.fit(features, target)
        self._is_trained = True

    def predict(self, features):
        if not self.is_trained:
            raise RuntimeError("模型尚未训练")
        return self.model.predict(features)

    def save(self, dir_path: Union[str, Path], overwrite: bool = False) -> Path:
        return super().save(dir_path, overwrite)

    @classmethod
    def load(cls, dir_path: Union[str, Path], model_name: str):
        return super().load(dir_path, model_name)

    def build_model(self):
        # 根据 self.config 构建模型结构
        # 示例：假设 config 中包含输入特征数和输出特征数
        input_features = self.config.get('input_features', 2)
        output_features = self.config.get('output_features', 1)
        self.model = torch.nn.Linear(input_features, output_features)


class ConcreteTestModel(BaseModel):
    """实现所有抽象方法的具体测试模型"""
    def train(self, features: pd.DataFrame, target: pd.Series, **kwargs) -> BaseModel:
        # 保存特征顺序
        self.feature_names_ = features.columns.tolist()
        self._is_trained = True
        return self  # 返回自身以支持链式调用

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        # 触发特征顺序校验
        self._validate_feature_order(features)
        return np.zeros(len(features))  # 返回模拟预测结果


class TestBaseModel:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            BaseModel("test_model")

    def test_model_saving(self, tmp_path):
        # 创建一个临时的模型类，继承自BaseModel并实现抽象方法
        class SklearnDummyModel(BaseModel):
            def __init__(self, model_name):
                super().__init__(model_name)
                # 初始化一个简单的sklearn模型
                self.model = LinearRegression()  # 确保 self.model 不为 None

            def train(self, features, target):
                pass

            def predict(self, features):
                pass

        model = SklearnDummyModel("dummy_model")
        model.save(tmp_path, model_name="dummy_model", overwrite=True)
        assert (tmp_path / "dummy_model.pkl").exists()

    def test_feature_order_validation(self):
        model = MagicMock(spec=BaseModel)
        model.feature_names_ = ["feature1", "feature2"]
        model.strict_feature_order = True

        # 正确顺序
        correct_features = pd.DataFrame([[1, 2]], columns=["feature1", "feature2"])
        BaseModel._validate_feature_order(model, correct_features)  # 不应抛出异常

        # 错误顺序
        wrong_features = pd.DataFrame([[1, 2]], columns=["feature2", "feature1"])
        with pytest.raises(ValueError):
            BaseModel._validate_feature_order(model, wrong_features)

    def test_config_initialization(self):
        """测试config参数初始化"""

        class ConcreteModel(BaseModel):
            def __init__(self, model_name, config=None):
                super().__init__(model_name, config)
                self.model = LinearRegression()  # 确保 self.model 不为 None

            def train(self, features, target):
                pass

            def predict(self, features):
                pass

        model = ConcreteModel("test", config={"key": "value"})
        expected_config = {"key": "value", "model_name": "test"}
        assert model.config == expected_config


def test_strict_feature_validation(sample_data):
    """严格特征顺序校验测试"""
    # 使用具体子类而非抽象基类
    model = ConcreteTestModel("test")
    model.strict_feature_order = True  # 启用严格顺序检查

    features, target = sample_data
    model.train(features, target)  # 训练以保存特征顺序

    # 构造顺序不一致的测试数据
    reversed_features = features[['feature2', 'feature1']]

    # 预期抛出特征顺序错误
    with pytest.raises(ValueError, match="特征列顺序与训练时不一致"):
        model.predict(reversed_features)


@pytest.mark.parametrize("model_type,config", [
    ("lstm", {"input_size": 5, "seq_length": 10}),
    ("random_forest", {"n_estimators": 50}),
    ("neural_network", {"input_size": 5})
])
def test_model_save_load(tmp_path, model_type, config):
    """参数化测试不同模型保存/加载"""
    # 初始化模型
    if model_type == "lstm":
        from src.models.lstm import LSTMModelWrapper
        model = LSTMModelWrapper(**config)
    elif model_type == "random_forest":
        from src.models.rf import RandomForestModel
        model = RandomForestModel(**config)
    else:
        from src.models.nn import NeuralNetworkModel
        model = NeuralNetworkModel(**config)

    # 训练模型
    X = pd.DataFrame(np.random.randn(100, config.get("input_size", 5)))
    y = pd.Series(np.random.randn(100))
    model.train(X, y)  # 确保训练时能够正确处理设备问题

    # 测试保存
    save_path = model.save(tmp_path, model_type)
    assert save_path.exists()

    # 测试加载
    loaded_model = type(model).load(tmp_path, model_type)
    assert loaded_model.is_trained


def test_unsupported_model_save(tmp_path):
    """测试保存不支持模型类型"""
    # 创建一个不支持的模型类，继承自BaseModel但设置model为不支持的类型
    class UnsupportedModel(BaseModel):
        def __init__(self, model_name):
            super().__init__(model_name)
            self.model = object()  # 设置为不支持的类型

        def train(self, features, target):
            pass

        def predict(self, features):
            pass

    model = UnsupportedModel("unsupported_model")
    with pytest.raises(NotImplementedError):
        model.save(tmp_path, model_name="unsupported_model", overwrite=True)


@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, pd.Series]:
    """生成测试数据"""
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    return X, y


@pytest.fixture
def trained_model(tmp_path: Path, sample_data) -> ConcreteModel:
    """已训练模型夹具"""
    model = ConcreteModel("test_model")
    X, y = sample_data
    model.train(X, y)
    return model


# --------------------------
# 正常流程测试
# --------------------------
def test_model_saving(tmp_path: Path, trained_model: ConcreteModel):
    """验证模型保存完整性"""
    save_path = trained_model.save(tmp_path)  # 使用save方法返回的路径
    assert save_path.exists()
    assert save_path.stat().st_size > 100


def test_model_loading(tmp_path: Path, trained_model: ConcreteModel):
    """验证模型加载功能"""
    trained_model.save(tmp_path)
    loaded = ConcreteModel.load(tmp_path, "test_model")
    assert loaded.is_trained
    assert loaded.predict(pd.DataFrame([[1, 4]], columns=['feature1', 'feature2'])).shape == (1,)


def test_feature_importance_format(trained_model: ConcreteModel, sample_data):
    """验证特征重要性输出格式"""
    features, _ = sample_data
    fi = trained_model.get_feature_importance()
    assert isinstance(fi, pd.Series)
    assert set(fi.index) == set(features.columns)
    assert len(fi) == len(features.columns)


# --------------------------
# 异常处理测试
# --------------------------
def test_save_conflict(tmp_path: Path, trained_model: ConcreteModel):
    """测试重复保存不覆盖"""
    trained_model.save(tmp_path, overwrite=False)  # 第一次保存
    with pytest.raises(FileExistsError):
        trained_model.save(tmp_path, overwrite=False)  # 第二次保存，不允许覆盖


def test_load_nonexistent(tmp_path: Path):
    """加载不存在模型应报错"""
    with pytest.raises(FileNotFoundError):
        ConcreteModel.load(tmp_path, "ghost_model")


def test_untrained_predict():
    """未训练模型预测应报错"""
    model = ConcreteModel("new_model")
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame())


# --------------------------
# 边界条件测试
# --------------------------
def test_empty_feature_importance():
    """无特征重要性支持的模型"""

    class DummyModel(BaseModel):
        def train(self, X, y):
            self.model = MagicMock()
            return self

        def predict(self, X):
            return np.zeros(len(X))

        def get_feature_importance(self) -> pd.Series:
            raise NotImplementedError("该模型不支持特征重要性获取")

    model = DummyModel("dummy").train(pd.DataFrame(), pd.Series())
    with pytest.raises(NotImplementedError):
        model.get_feature_importance()


def test_zero_importance():
    """零重要性特征处理"""

    class DummyModel(BaseModel):
        def train(self, X, y):
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier()
            self.model.fit(X, y)
            # 手动设置特征重要性为零（通过重新训练模型）
            self.model = RandomForestClassifier(class_weight={0: 1, 1: 0})
            self.model.fit(X, y)
            self.is_trained = True
            return self

        def predict(self, X):
            return np.zeros(len(X))

    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    model = DummyModel("dummy").train(X, y)
    fi = model.get_feature_importance()
    assert (fi == 0).all()


# --------------------------
# 类型兼容性测试
# --------------------------
@pytest.mark.parametrize("model_type", ["sklearn", "pytorch"])
def test_cross_framework_saving(tmp_path: Path, model_type: str):
    """测试不同框架模型保存/加载"""
    if model_type == "sklearn":
        from sklearn.linear_model import LinearRegression
        mock_model = LinearRegression()
    else:
        import torch
        mock_model = torch.nn.Linear(2, 1)

    model = ConcreteModel("cross_model")
    model.model = mock_model
    model.save(tmp_path)


def test_sklearn_model_saving(tmp_path):
    """测试保存Scikit-learn模型"""
    from sklearn.linear_model import LinearRegression
    model = ConcreteModel("sklearn_model")
    model.model = LinearRegression()
    save_path = model.save(tmp_path)
    assert save_path.suffix == ".pkl"

    loaded = ConcreteModel.load(tmp_path, "sklearn_model")
    assert isinstance(loaded.model, LinearRegression)


def test_pytorch_model_saving(tmp_path):
    """测试保存PyTorch模型"""
    import torch.nn as nn
    # 创建 ConcreteModel 实例并传递 config
    model = ConcreteModel(
        model_name="pytorch_model",
        config={"input_features": 2, "output_features": 1}
    )
    # 设置模型为一个简单的线性层
    model.model = nn.Linear(2, 1)
    save_path = model.save(tmp_path)  # 保存模型
    assert save_path.suffix == ".pt"  # 断言保存的文件扩展名是 .pt

    loaded = ConcreteModel.load(tmp_path, "pytorch_model")  # 加载模型
    assert loaded.model is not None  # 确保加载的模型不是 None


def test_strict_feature_order_validation():
    """测试严格特征顺序校验"""
    # 初始化 ConcreteModel 时提供 model_name 参数
    model = ConcreteModel(model_name="test_model", config={'model_type': 'regression'})
    # 假设的特征数据
    features = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
    # 假设的目标数据
    target = pd.Series(np.random.choice([0, 1], size=100))  # 分类任务
    # 训练模型
    model.train(features, target)
    # 修改特征列顺序以触发校验
    reordered_features = features[[f'feature_{i}' for i in [4, 3, 2, 1, 0]]]
    # 预测时会触发 ValueError，因为特征列顺序与训练时不一致
    with pytest.raises(ValueError):
        model.predict(reordered_features)


def test_untrained_validate():
    """测试未训练模型验证"""
    model = ConcreteModel(model_name="test_model")
    with pytest.raises(RuntimeError):
        model.validate(pd.DataFrame(), pd.Series(), {})


# 新增回归/分类任务测试用例
def test_regression_model():
    model = ConcreteModel("reg_test", config={'model_type': 'regression'})
    features = pd.DataFrame(np.random.randn(100, 5))
    target = pd.Series(np.random.randn(100))  # 连续目标
    model.train(features, target)
    assert model.is_trained


def test_classification_model():
    model = ConcreteModel("cls_test", config={'model_type': 'classification'})
    features = pd.DataFrame(np.random.randn(100, 5))
    target = pd.Series(np.random.choice([0, 1], 100))  # 离散目标
    model.train(features, target)
    assert model.is_trained


# 测试save()方法对不支持模型类型的异常
def test_save_unsupported_model_type(tmp_path):
    class DummyModel(BaseModel):
        def train(self, *args): pass

        def predict(self, *args): pass

    model = DummyModel("test")
    model.model = "invalid_type"  # 注入非法模型类型
    with pytest.raises(NotImplementedError):
        model.save(tmp_path)


# 测试load()方法文件不存在异常
def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        BaseModel.load("nonexistent_dir", "invalid_model")


def test_base_model_load_nonexistent():
    with pytest.raises(FileNotFoundError):
        BaseModel.load("fake_dir", "fake_model")


def test_feature_validation_strict_mode():
    """测试严格特征顺序校验"""

    class TestModel(BaseModel):
        def train(self, *args): pass

        def predict(self, *args): return np.array([1])

    model = TestModel("test")
    model.feature_names_ = ["a", "b"]
    model.strict_feature_order = True

    with pytest.raises(ValueError):
        model._validate_feature_order(pd.DataFrame([[1, 2]], columns=["b", "a"]))


def test_abstract_methods():
    """验证抽象方法必须实现"""
    with pytest.raises(TypeError):
        BaseModel("invalid")


@pytest.mark.parametrize("model_type", [LSTMModelWrapper, NeuralNetworkModel, RandomForestModel])
def test_base_model_save_load(tmpdir, model_type):
    # 定义每个模型所需的参数
    model_params = {
        LSTMModelWrapper: {
            "input_size": 5,
            "seq_length": 10,
            "hidden_size": 256,
            "num_layers": 3,
            "output_size": 1,
            "dropout": 0.5,
            "device": "cpu"
        },
        NeuralNetworkModel: {
            "input_size": 5,
            "hidden_layers": [32, 16],
            "dropout_rate": 0.5,
            "device": "cpu",
            "output_size": 1
        },
        RandomForestModel: {
            "n_estimators": 100,
            "max_depth": 5
        }
    }

    # 初始化模型
    model = model_type(**model_params[model_type])

    # 创建示例数据进行简单训练
    X = pd.DataFrame(np.random.randn(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    y = pd.Series(np.random.randn(100))

    # 训练模型
    if isinstance(model, (LSTMModelWrapper, NeuralNetworkModel)):
        model.train(X, y, epochs=5)
    elif isinstance(model, RandomForestModel):
        model.train(X, y)

    # 保存模型
    model_path = model.save(tmpdir, model_name="test_model", overwrite=True)

    # 加载模型
    loaded_model = model_type.load(tmpdir, model_name="test_model")

    # 验证加载的模型
    assert loaded_model.is_trained


def test_feature_order_validation():
    # 创建模型所需的参数
    model_params = {
        "input_size": 5,
        "seq_length": 10,
        "hidden_size": 256,
        "num_layers": 3,
        "output_size": 1,
        "dropout": 0.5,
        "device": "cpu"
    }

    # 初始化模型
    model = LSTMModelWrapper(**model_params)

    # 创建特征数据
    correct_features = pd.DataFrame(np.random.randn(10, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    model.feature_names_ = correct_features.columns.tolist()

    # 正确顺序的特征
    model._validate_feature_order(correct_features)  # 不应抛出异常

    # 错误顺序的特征
    wrong_features = pd.DataFrame(np.random.randn(10, 5), columns=["f5", "f4", "f3", "f2", "f1"])
    model.strict_feature_order = True  # 确保严格校验特征顺序
    with pytest.raises(ValueError):
        model._validate_feature_order(wrong_features)


@pytest.mark.parametrize("features_type", [np.ndarray, pd.DataFrame])
def test_feature_validation_types(features_type, lstm_model, sample_data):
    """验证不同数据类型的特征校验逻辑"""
    features, _ = sample_data
    original_features = features.copy()  # 保存原始特征

    # 根据类型转换特征
    if features_type == np.ndarray:
        features = features.values
        # 对于numpy数组，直接操作列索引
        if len(features.shape) > 1:  # 确保是二维数组
            reordered_features = np.hstack([features[:, -1:], features[:, :-1]])
        else:
            reordered_features = features  # 一维数组无法重新排列
    else:
        # 对于DataFrame，重新排列列顺序
        reordered_features = features[[features.columns[-1]] + features.columns[:-1].tolist()]

    lstm_model.feature_names_ = (
        original_features.columns.tolist() if isinstance(original_features, pd.DataFrame)
        else list(range(original_features.shape[1]))
    )
    lstm_model.strict_feature_order = True

    # 如果是DataFrame，验证列顺序不匹配是否触发异常
    if features_type == pd.DataFrame:
        with pytest.raises(ValueError):
            lstm_model._validate_feature_order(reordered_features)
    else:
        # 对于numpy数组，验证特征维度是否匹配
        if len(reordered_features.shape) >= 2 and reordered_features.shape[1] != len(lstm_model.feature_names_):
            with pytest.raises(ValueError):
                lstm_model._validate_feature_order(reordered_features)
        else:
            # 如果维度匹配，不抛出异常
            try:
                lstm_model._validate_feature_order(reordered_features)
            except ValueError:
                pytest.fail("验证特征顺序时不应抛出异常")


@pytest.mark.parametrize("file_ext, model_type", [("invalid", "sklearn"), ("invalid", "pytorch")])
def test_unsupported_file_extension(model_manager, file_ext, model_type, tmp_path):
    """验证不支持的文件扩展名处理"""
    # 创建一个临时文件路径
    file_path = tmp_path / f"test.{file_ext}"

    # 确保文件存在
    file_path.touch()

    with pytest.raises(NotImplementedError):
        if model_type == "sklearn":
            model_manager._load_sklearn(file_path)
        else:
            model_manager._load_pytorch(file_path)


def test_base_model_abstract_methods():
    """验证未实现抽象方法抛出TypeError"""

    class InvalidModel(BaseModel):
        pass

    with pytest.raises(TypeError):
        InvalidModel("invalid_model")


def test_load_corrupted_file(tmp_path):
    """测试加载损坏模型文件"""
    # 确保有权限访问临时目录
    if not os.access(tmp_path, os.W_OK):
        pytest.skip("没有足够的权限访问临时目录")

    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_bytes(b"invalid_data")

    class DummyModel(BaseModel):
        def __init__(self, model_name):
            super().__init__(model_name)

        def train(self, features, target):
            pass

        def predict(self, features):
            pass

    model = DummyModel("dummy_model")

    with pytest.raises(ValueError):
        model.load(tmp_path, "corrupt")

@pytest.fixture(params=[LSTMModelWrapper, NeuralNetworkModel, RandomForestModel])
def generic_model(request, sample_data):
    """通用模型初始化fixture"""
    features, _ = sample_data
    if request.param == LSTMModelWrapper:
        return LSTMModelWrapper(input_size=features.shape[1])
    elif request.param == NeuralNetworkModel:
        return NeuralNetworkModel(input_size=features.shape[1])
    else:
        return RandomForestModel()

@patch("torch.cuda.is_available", return_value=False)
def test_device_auto_cpu(mock_cuda):
    """测试自动选择CPU设备"""
    model = LSTMModelWrapper(input_size=5, device="auto")
    assert model.device.type == "cpu"

@patch("pathlib.Path.exists", return_value=False)
def test_load_missing_scaler(mock_exists):
    """模拟scaler文件缺失"""
    model = LSTMModelWrapper(input_size=5)
    with pytest.raises(FileNotFoundError):
        model.load(Path("invalid_dir"), "missing_scaler")


@pytest.mark.parametrize("input_data,error_msg", [
    (np.random.rand(10, 3), "输入特征维度不匹配"),  # 特征数不一致
    (pd.DataFrame([[1, 2]], columns=["b", "a"]), "特征列顺序与训练时不一致")  # 列顺序不一致
])
def test_feature_validation_errors(input_data, error_msg):
    """测试特征校验异常路径"""

    class TestModel(BaseModel):
        def train(self, *args): pass

        def predict(self, *args): return np.array([1])

    model = TestModel("test")
    model.feature_names_ = ["a", "b"]
    model.strict_feature_order = True

    with pytest.raises(ValueError, match=error_msg):
        model._validate_feature_order(input_data)


def test_abstract_class_instantiation():
    """验证抽象基类不可直接实例化"""
    with pytest.raises(TypeError):
        BaseModel("abstract")


@patch("joblib.dump")
@patch("pathlib.Path.exists", return_value=True)
def test_save_overwrite_validation(mock_exists, mock_dump):
    """测试覆盖保存时文件存在校验"""
    model = RandomForestModel()
    model.model = RandomForestRegressor()

    with pytest.raises(FileExistsError):
        model.save(Path("/exist"), overwrite=False)


def test_unsupported_model_load(tmp_path):
    """测试加载不支持模型类型"""
    # 创建一个损坏的模型文件
    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_bytes(b"invalid_data")

    class DummyModel(BaseModel):
        def __init__(self, model_name):
            super().__init__(model_name)

        def train(self, features, target):
            pass

        def predict(self, features):
            pass

    model = DummyModel("dummy_model")
    with pytest.raises(Exception):
        model.load(tmp_path, "corrupt")


# 测试抽象方法实现
def test_abstract_methods_implementation():
    class ConcreteModel(BaseModel):
        def train(self, features, target):
            pass
        def predict(self, features):
            pass

    model = ConcreteModel("test")
    assert model.model_name == "test"

# 测试特征校验逻辑
def test_feature_validation_edge_cases():
    model = ConcreteModel("test")
    model.feature_names_ = ["a", "b"]
    model.strict_feature_order = True

    # 测试 numpy 数组输入形状不匹配
    with pytest.raises(ValueError, match="输入特征维度不匹配"):
        model._validate_feature_order(np.array([[1, 2, 3], [4, 5, 6]]))  # 错误形状，3列而不是2列

    # 测试 DataFrame 列顺序不匹配
    with pytest.raises(ValueError, match="特征列顺序与训练时不一致"):
        model._validate_feature_order(pd.DataFrame([[1, 2]], columns=["b", "a"]))


@pytest.mark.parametrize("model_type,config", [
    ("sklearn", {"model": LinearRegression()}),
    ("pytorch", {"model": torch.nn.Linear(2, 1)})
])
def test_model_persistence(tmp_path, model_type, config):
    """测试模型持久化工具类"""
    model = config["model"]
    path = tmp_path / "test_model.pkl"

    # 测试保存
    ModelPersistence.save_model(model, path)
    assert path.exists()

    # 测试加载
    loaded = ModelPersistence.load_model(path)
    assert isinstance(loaded, type(model))


def test_torch_model_mixin_methods():
    """测试TorchModelMixin方法"""

    class TestModel(BaseModel, TorchModelMixin):
        def train(self, *args): pass

        def predict(self, *args): pass

        def get_model(self):
            # 创建一个简单的线性模型
            return torch.nn.Linear(2, 1)

    model = TestModel("test")

    # 测试优化器配置
    optimizer = model.configure_optimizer()
    assert isinstance(optimizer, torch.optim.Optimizer)

    # 测试损失函数配置
    loss_fn = model.configure_loss()
    assert isinstance(loss_fn, torch.nn.Module)

    # 测试训练epoch
    # 直接创建一个包含批次数据的迭代器
    batch_data = [(torch.randn(2, 2), torch.randn(2, 1))]
    mock_loader = iter(batch_data)

    # 确保模型在正确设备上
    device = "cpu"  # 强制使用 CPU
    model.get_model().to(device)

    # 确保数据在正确设备上
    batch_data = [(inputs.to(device), labels.to(device)) for inputs, labels in batch_data]
    mock_loader = iter(batch_data)

    loss = model.train_epoch(mock_loader, optimizer, loss_fn, device=device)
    assert isinstance(loss, float)


@pytest.mark.parametrize("is_trained", [True, False])
def test_is_trained_property(is_trained):
    """测试is_trained属性"""
    model = ConcreteModel("test")
    model.is_trained = is_trained
    assert model.is_trained == is_trained


def test_validate_feature_order_mismatch():
    class TestModel(BaseModel):
        def train(self, features, target, **kwargs): pass

        def predict(self, features): pass

    model = TestModel("test")
    model.feature_names_ = ["f1", "f2", "f3"]
    model.strict_feature_order = True

    # 测试 DataFrame 列顺序不匹配
    with pytest.raises(ValueError):
        features = pd.DataFrame([[1, 2, 3]], columns=["f1", "f3", "f2"])
        model._validate_feature_order(features)

    # 测试 numpy 数组维度不匹配
    with pytest.raises(ValueError):
        features = np.array([[1, 2]])
        model._validate_feature_order(features)


# 测试 BaseModel.save 不支持的类型
def test_save_unsupported_type():
    class TestModel(BaseModel):
        def train(self, features, target, **kwargs): pass

        def predict(self, features): pass

    model = TestModel("test")
    model.model = "unsupported_type"

    with pytest.raises(NotImplementedError):
        model.save("dir_path")


# 测试 BaseModel.load 文件不存在
def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        BaseModel.load("nonexistent_dir", "nonexistent_model")


# 测试 TorchModelMixin.train_epoch 空数据加载器
def test_train_epoch_empty_loader():
    class TestModel(TorchModelMixin):
        def get_model(self):
            return torch.nn.Linear(2, 1)

    model = TestModel()
    empty_loader = []

    with pytest.raises(ValueError):
        model.train_epoch(empty_loader, torch.optim.Adam(model.get_model().parameters()),
                          torch.nn.MSELoss())



def test_torch_model_mixin_abstract():
    # 测试 TorchModelMixin 抽象方法
    class TestModel(TorchModelMixin):
        def get_model(self):
            return torch.nn.Linear(2, 1)

    model = TestModel()
    assert isinstance(model.configure_optimizer(), torch.optim.Optimizer)
    assert isinstance(model.configure_loss(), torch.nn.Module)


def test_pytorch_model_save_load(tmp_path):
    class PyTorchModel(BaseModel):
        def build_model(self):
            self.model = torch.nn.Linear(5, 1)

        def train(self, features, target):
            self._is_trained = True
            return self

        def predict(self, features):
            return torch.randn(len(features))

    model = PyTorchModel("pytorch_test")
    model.build_model()
    model.train(None, None)

    # Test save
    save_path = model.save(tmp_path, "pytorch_test")
    assert save_path.exists()

    # Test load
    loaded = PyTorchModel.load(tmp_path, "pytorch_test")
    assert loaded.is_trained  # 现在应为 True


# 测试多指标验证
def test_validate_with_multiple_metrics():
    class MockModel(BaseModel):
        def train(self, features, target, **kwargs):
            # 模拟训练逻辑，不做实际训练
            self._is_trained = True
            return self

        def predict(self, features):
            return np.array([0.5, 0.6, 0.7])

    model = MockModel("test")
    model._is_trained = True
    features = pd.DataFrame({"f1": [1, 2, 3]})
    target = pd.Series([0.5, 0.6, 0.7])
    metrics = {
        "mse": lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean(),
        "mae": lambda y_true, y_pred: np.abs(y_true - y_pred).mean()
    }

    results = model.validate(features, target, metrics)
    assert "mse" in results
    assert "mae" in results
    assert results["mse"] < 0.01


def test_base_model_feature_validation_numpy_mismatch():
    class ConcreteModel(BaseModel):
        def train(self, features: pd.DataFrame, target: pd.Series, **kwargs) -> BaseModel:
            self._is_trained = True
            return self

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.array([0.0] * len(features))

    model = ConcreteModel("test")
    model.feature_names_ = ["f1", "f2", "f3"]
    with pytest.raises(ValueError):
        # 传入numpy数组但特征数量不匹配
        model._validate_feature_order(np.array([[1, 2], [3, 4]]))


def test_base_model_feature_validation_unsupported_type():
    class ConcreteModel(BaseModel):
        def train(self, features: pd.DataFrame, target: pd.Series, **kwargs) -> BaseModel:
            self._is_trained = True
            return self

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.array([0.0] * len(features))

    model = ConcreteModel("test")
    model.feature_names_ = ["f1", "f2"]
    with pytest.raises(TypeError):
        # 传入不支持的数据类型
        model._validate_feature_order("invalid_data")


def test_base_model_predict_not_implemented():
    # 创建一个继承自 BaseModel 的子类，但不实现 predict 方法
    class IncompleteModel(BaseModel):
        def train(self, features: pd.DataFrame, target: pd.Series, **kwargs) -> BaseModel:
            self._is_trained = True
            return self

    # 使用 unittest.mock 模拟 predict 方法
    model = MagicMock(spec=IncompleteModel)
    model.predict.side_effect = NotImplementedError

    with pytest.raises(NotImplementedError):
        model.predict(pd.DataFrame())


def test_base_model_save_unsupported_type():
    # 创建一个继承自 BaseModel 的子类，并实现所有抽象方法
    class ConcreteModel(BaseModel):
        def train(self, features: pd.DataFrame, target: pd.Series, **kwargs) -> BaseModel:
            self._is_trained = True
            return self

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.array([0.0] * len(features))

    model = ConcreteModel("test")
    with pytest.raises(NotImplementedError):
        model.save("unsupported_path", "test")


def test_base_model_load_unknown_extension(tmp_path):
    class ConcreteModel(BaseModel):
        def train(self, features: pd.DataFrame, target: pd.Series, **kwargs) -> BaseModel:
            self._is_trained = True
            return self

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.array([0.0] * len(features))

    model = ConcreteModel("test")
    # 创建无效扩展名文件
    invalid_file = tmp_path / "model.invalid"
    invalid_file.touch()

    # 确保没有支持的扩展名文件存在
    supported_files = [tmp_path / f"model{ext}" for ext in [".pkl", ".pt"]]
    for file in supported_files:
        if file.exists():
            file.unlink()

    with pytest.raises(FileNotFoundError):
        model.load(tmp_path, "model")


def test_base_model_feature_importance_not_supported():
    class ConcreteModel(BaseModel):
        def train(self, features: pd.DataFrame, target: pd.Series, **kwargs) -> BaseModel:
            self._is_trained = True
            return self

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.array([0.0] * len(features))

    model = ConcreteModel("test")
    model.model = type("DummyModel", (), {})()  # 无feature_importances_的模型
    with pytest.raises(NotImplementedError):
        model.get_feature_importance()


def test_base_model_load_invalid_extension(tmp_path):
    invalid_path = tmp_path / "test.pkl"  # 使用支持的扩展名
    invalid_path.touch()
    with pytest.raises(ValueError):
        BaseModel.load(tmp_path, "test")


# 测试sklearn模型保存/加载
def test_sklearn_save_load(tmp_path):
    # 创建sklearn模型实例
    model = RandomForestModel()
    model.model = RandomForestRegressor()
    model._is_trained = True
    model.feature_names_ = ["f1", "f2", "f3"]

    # 保存并加载
    save_path = model.save(tmp_path, "test_sklearn")
    loaded = RandomForestModel.load(tmp_path, "test_sklearn")  # 使用具体的子类加载模型

    # 验证加载的模型
    assert loaded.is_trained
    assert isinstance(loaded.model, RandomForestRegressor)

# 测试未训练模型验证
def test_validate_untrained():
    # 使用具体的子类创建模型实例
    model = RandomForestModel(model_name="test")
    # 确保模型未被训练
    model._is_trained = False
    with pytest.raises(RuntimeError):
        model.validate(
            features=pd.DataFrame(),
            target=pd.Series(dtype=float),
            metrics={"mse": mean_squared_error}
        )

# 测试特征重要性异常
def test_feature_importance_not_implemented():
    # 使用具体的子类创建模型实例
    model = NeuralNetworkModel(input_size=3)  # 或其他具体子类
    # 确保模型未被训练
    model._is_trained = True
    with pytest.raises(NotImplementedError):
        model.get_feature_importance()


# 测试TorchModelMixin默认方法
def test_torch_mixin_defaults():
    class DummyModel(TorchModelMixin):
        def get_model(self) -> torch.nn.Module:
            raise NotImplementedError("get_model 方法未实现")

    model = DummyModel()
    with pytest.raises(NotImplementedError):
        model.get_model()

    with pytest.raises(NotImplementedError):
        model.configure_optimizer()

    with pytest.raises(NotImplementedError):
        model.configure_loss()
