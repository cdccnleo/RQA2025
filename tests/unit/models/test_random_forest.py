# tests/models/test_random_forest.py

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.tree import DecisionTreeRegressor

from src.models.rf import RandomForestModel
from sklearn.ensemble import RandomForestRegressor


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """生成测试数据：100样本，5特征，正态分布目标"""
    features = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    target = pd.Series(np.random.randn(100))
    return features, target


@pytest.fixture
def trained_model(sample_data) -> RandomForestModel:
    """训练好的模型夹具"""
    features, target = sample_data
    model = RandomForestModel(n_estimators=10)  # 减少树数以加速测试
    model.train(features, target)
    return model


def test_training_process(trained_model: RandomForestModel):
    """验证模型训练后状态和属性"""
    assert trained_model.is_trained
    assert trained_model.model is not None
    assert isinstance(trained_model.model, RandomForestRegressor)
    assert len(trained_model.model.estimators_) == 10  # 与n_estimators=10一致


def test_prediction_shape(trained_model: RandomForestModel, sample_data):
    """测试预测结果形状匹配输入"""
    features, _ = sample_data
    predictions = trained_model.predict(features)
    assert predictions.shape == (100,)


def test_prediction_range(trained_model: RandomForestModel, sample_data):
    """验证预测值在合理范围内（基于正态分布数据）"""
    features, _ = sample_data
    predictions = trained_model.predict(features)
    assert predictions.min() >= -5 and predictions.max() <= 5


def test_empty_input_handling():
    """测试空数据输入训练时的异常处理"""
    model = RandomForestModel()
    with pytest.raises(ValueError):  # 修改为捕获 ValueError
        model.train(pd.DataFrame(), pd.Series(dtype=float))


def test_untrained_prediction():
    """测试未训练模型预测时抛出异常"""
    model = RandomForestModel()
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame(np.random.randn(10, 5)))


def test_feature_importance_output(trained_model: RandomForestModel):
    """验证特征重要性输出格式和值范围"""
    importance = trained_model.get_feature_importance()
    assert isinstance(importance, pd.Series)
    assert len(importance) == 5  # 特征数
    assert (importance >= 0).all() and (importance <= 1).all()
    assert np.isclose(importance.sum(), 1.0, atol=1e-3)


def test_model_saving_loading(tmp_path: Path, trained_model: RandomForestModel):
    """测试模型保存和加载功能"""
    save_path = tmp_path / "saved_model"
    trained_model.save(save_path, model_name="random_forest")

    loaded_model = RandomForestModel.load(save_path, model_name="random_forest")
    assert loaded_model.is_trained
    assert loaded_model.config['n_estimators'] == trained_model.config['n_estimators']


def test_extreme_parameters(sample_data):
    """测试极端参数（n_estimators=1）下的模型行为"""
    features, target = sample_data
    model = RandomForestModel(n_estimators=1)
    model.train(features, target)
    assert len(model.model.estimators_) == 1
    assert model.predict(features).shape == (100,)


def test_numpy_input(trained_model, sample_data):
    """测试numpy数组输入"""
    features, _ = sample_data
    np_features = features.values
    predictions = trained_model.predict(np_features)
    assert predictions is not None


def test_random_forest_feature_importance():
    # 创建模型
    model = RandomForestModel()

    # 创建数据
    X = pd.DataFrame(np.random.randn(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    y = pd.Series(np.random.randn(100))

    # 训练模型
    model.train(X, y)

    # 提取特征重要性
    importance = model.get_feature_importance()
    assert len(importance) == 5
    assert importance.index.tolist() == ["f1", "f2", "f3", "f4", "f5"]  # 验证列名


def test_rf_extreme_parameters():
    """验证极端参数处理"""
    with pytest.raises(ValueError):
        RandomForestModel(n_estimators=0)


def test_feature_interaction_matrix(rf_model, sample_data):
    """验证特征交互矩阵"""
    features, target = sample_data
    rf_model.train(features, target)  # 训练模型
    interaction_df = rf_model.analyze_feature_interactions(features)
    assert interaction_df.shape == (features.shape[1], features.shape[1])
    assert not interaction_df.isnull().any().any()


def test_rf_initialization():
    model = RandomForestModel(n_estimators=50)
    assert model.config["n_estimators"] == 50


def test_feature_importance():
    model = RandomForestModel(n_estimators=10)
    X = pd.DataFrame(np.random.randn(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    y = pd.Series(np.random.randn(100))
    model.train(X, y)
    importance = model.get_feature_importance()
    assert len(importance) == 5


def test_feature_interaction():
    model = RandomForestModel(n_estimators=10)
    X = pd.DataFrame(np.random.randn(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    model.train(X, pd.Series(np.random.randn(100)))
    interaction_df = model.analyze_feature_interactions(X)
    assert interaction_df.shape == (5, 5)


# 测试特征交互分析
def test_feature_interaction_analysis():
    model = RandomForestModel(n_estimators=10)
    X = pd.DataFrame(np.random.randn(100, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    y = pd.Series(np.random.randn(100))
    model.train(X, y)
    interaction_df = model.analyze_feature_interactions(X)
    assert interaction_df.shape == (5, 5)


def test_extreme_parameter_values():
    """测试极端参数值"""
    with pytest.raises(ValueError):
        RandomForestModel(n_estimators=0)
    with pytest.raises(ValueError):
        RandomForestModel(max_depth=0)


def test_feature_interaction_edge_cases(rf_model, sample_data):
    """测试特征交互边界情况"""
    features, target = sample_data
    rf_model.train(features, target)

    # 测试单特征情况
    single_feature = features.iloc[:, :1]
    with pytest.raises(ValueError, match="特征交互分析需要至少两列特征"):
        rf_model.analyze_feature_interactions(single_feature)


# 测试特征交互分析
def test_feature_interactions():
    model = RandomForestModel()
    model.model = RandomForestRegressor()
    model._is_trained = True

    # 创建有交互特征的数据
    features = pd.DataFrame({
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "f3": np.random.rand(100)
    })
    features["target"] = features["f1"] * features["f2"] + features["f3"]

    # 训练模型
    model.train(features[["f1", "f2", "f3"]], features["target"])

    # 分析特征交互
    interaction_df = model.analyze_feature_interactions(
        features[["f1", "f2", "f3"]]
    )
    assert not interaction_df.empty
    assert "f1" in interaction_df.columns
    assert "f2" in interaction_df.index

    # 测试f1和f2应有强交互
    assert interaction_df.loc["f1", "f2"] > interaction_df.loc["f1", "f3"]


def test_rf_save_file_exists(tmp_path):
    rf = RandomForestModel()
    rf._is_trained = True
    path = tmp_path / "random_forest.pkl"  # 使用模型默认的文件名
    path.touch()  # 创建空文件

    with pytest.raises(FileExistsError):
        rf.save(tmp_path, overwrite=False)


def test_rf_load_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        RandomForestModel.load(tmp_path, "missing_model")


def test_rf_feature_importance_untrained():
    rf = RandomForestModel()
    with pytest.raises(RuntimeError):
        rf.get_feature_importance()


def test_rf_feature_interaction_insufficient_features():
    rf = RandomForestModel()
    rf._is_trained = True
    rf.model = RandomForestRegressor()
    # 单列特征
    X = pd.DataFrame(np.random.rand(10, 1))
    with pytest.raises(ValueError):
        rf.analyze_feature_interactions(X)


def test_rf_predict_numpy_input():
    rf = RandomForestModel()
    rf.model = RandomForestRegressor()
    rf._is_trained = True

    # 模拟训练过程以生成必要的属性
    rf.model.estimators_ = []
    for _ in range(10):
        tree = DecisionTreeRegressor()
        # 生成简单的训练数据
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        tree.fit(X_train, y_train)
        rf.model.estimators_.append(tree)
    rf.model.n_features_in_ = 2
    rf.model.n_outputs_ = 1

    rf.feature_names_ = ["f1", "f2"]

    # 测试 numpy 数组输入
    predictions = rf.predict(np.array([[1, 2], [3, 4]]))
    assert predictions is not None


def __post_init__(self):
    assert self.model is None or hasattr(self.model, 'fit'), \
           "模型实例必须实现fit方法"


# 测试空输入训练
def test_train_empty_input():
    model = RandomForestModel()
    with pytest.raises(ValueError):
        model.train(pd.DataFrame(), pd.Series())

# 测试特征交互异常
def test_feature_interaction_error():
    # 创建并训练随机森林模型
    model = RandomForestModel()
    X_train = pd.DataFrame(np.random.randn(100, 2), columns=["f1", "f2"])
    y_train = pd.Series(np.random.randn(100))
    model.train(X_train, y_train)  # 训练模型

    # 测试特征交互分析时的错误处理（特征数量不足）
    with pytest.raises(ValueError):
        model.analyze_feature_interactions(pd.DataFrame([[1], [2]], columns=["single"]))

# 测试numpy交互分析
def test_numpy_feature_interaction():
    model = RandomForestModel()
    model.model = RandomForestRegressor()
    model.model.fit(np.random.rand(10, 3), np.random.rand(10))
    model._is_trained = True  # 显式设置训练状态为 True
    result = model.analyze_feature_interactions(np.random.rand(5, 3))
    assert isinstance(result, pd.DataFrame)