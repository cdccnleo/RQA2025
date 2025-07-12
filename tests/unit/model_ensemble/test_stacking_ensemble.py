import pytest
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from models.stacking_ensemble import (
    StackingEnsemble,
    WeightedAverageEnsemble
)

@pytest.fixture
def sample_data():
    """创建测试数据"""
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 100)  # binary target
    return X, y

@pytest.fixture
def base_models():
    """创建基模型列表"""
    return [
        DecisionTreeClassifier(max_depth=3),
        RandomForestClassifier(n_estimators=10)
    ]

@pytest.fixture
def meta_model():
    """创建元模型"""
    return LogisticRegression()

@pytest.fixture
def stacking_ensemble(base_models, meta_model):
    """创建堆叠集成实例"""
    return StackingEnsemble(
        base_models=base_models,
        meta_model=meta_model,
        n_folds=3,
        verbose=False
    )

@pytest.fixture
def weighted_ensemble(base_models):
    """创建加权平均集成实例"""
    return WeightedAverageEnsemble(
        models=base_models,
        weights=[0.7, 0.3],
        use_probas=False
    )

def test_stacking_fit(stacking_ensemble, sample_data):
    """测试堆叠集成训练"""
    X, y = sample_data

    # 模拟基模型训练
    with patch.object(DecisionTreeClassifier, 'fit') as mock_dt_fit, \
         patch.object(RandomForestClassifier, 'fit') as mock_rf_fit, \
         patch.object(LogisticRegression, 'fit') as mock_meta_fit:

        stacking_ensemble.fit(X, y)

        # 验证基模型训练调用
        assert mock_dt_fit.call_count == 3 + 1  # 3 folds + final
        assert mock_rf_fit.call_count == 3 + 1  # 3 folds + final

        # 验证元模型训练调用
        mock_meta_fit.assert_called_once()

def test_stacking_predict(stacking_ensemble, sample_data):
    """测试堆叠集成预测"""
    X, y = sample_data

    # 训练模型
    stacking_ensemble.fit(X, y)

    # 模拟基模型预测
    with patch.object(DecisionTreeClassifier, 'predict',
                     return_value=np.zeros(X.shape[0])) as mock_dt_pred, \
         patch.object(RandomForestClassifier, 'predict',
                     return_value=np.ones(X.shape[0])) as mock_rf_pred, \
         patch.object(LogisticRegression, 'predict') as mock_meta_pred:

        # 设置元模型预测返回值
        mock_meta_pred.return_value = np.ones(X.shape[0])

        preds = stacking_ensemble.predict(X)

        # 验证预测调用
        assert mock_dt_pred.call_count == 4  # 3 folds + final
        assert mock_rf_pred.call_count == 4  # 3 folds + final
        mock_meta_pred.assert_called_once()

        # 验证预测结果
        assert preds.shape == (X.shape[0],)
        assert np.all(preds == 1)

def test_weighted_average_predict(weighted_ensemble, sample_data):
    """测试加权平均预测"""
    X, y = sample_data

    # 训练模型
    weighted_ensemble.fit(X, y)

    # 模拟模型预测
    with patch.object(DecisionTreeClassifier, 'predict',
                     return_value=np.ones(X.shape[0])) as mock_dt_pred, \
         patch.object(RandomForestClassifier, 'predict',
                     return_value=np.zeros(X.shape[0])) as mock_rf_pred:

        preds = weighted_ensemble.predict(X)

        # 验证预测调用
        mock_dt_pred.assert_called_once()
        mock_rf_pred.assert_called_once()

        # 验证加权结果 (0.7*1 + 0.3*0)
        assert preds.shape == (X.shape[0],)
        assert np.allclose(preds, 0.7)

def test_weighted_average_proba(weighted_ensemble, sample_data):
    """测试加权平均概率预测"""
    X, y = sample_data

    # 修改配置以使用概率
    weighted_ensemble.use_probas = True

    # 训练模型
    weighted_ensemble.fit(X, y)

    # 模拟概率预测
    mock_proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    with patch.object(DecisionTreeClassifier, 'predict_proba',
                     return_value=mock_proba) as mock_dt_proba, \
         patch.object(RandomForestClassifier, 'predict_proba',
                     return_value=mock_proba) as mock_rf_proba:

        probas = weighted_ensemble.predict_proba(X[:2])

        # 验证调用
        mock_dt_proba.assert_called_once()
        mock_rf_proba.assert_called_once()

        # 验证加权概率 (0.7*mock_proba + 0.3*mock_proba)
        assert probas.shape == (2, 2)
        assert np.allclose(probas, mock_proba)

def test_weight_update(weighted_ensemble):
    """测试权重更新"""
    # 初始权重
    assert weighted_ensemble.weights == [0.7, 0.3]

    # 更新权重
    weighted_ensemble.update_weights([0.4, 0.6])
    assert weighted_ensemble.weights == [0.4, 0.6]

    # 无效权重测试
    with pytest.raises(ValueError, match="Number of weights must match"):
        weighted_ensemble.update_weights([0.5, 0.4, 0.1])

def test_stacking_model_weights(stacking_ensemble, sample_data):
    """测试获取模型权重"""
    X, y = sample_data

    # 训练模型
    stacking_ensemble.fit(X, y)

    # 模拟元模型系数
    stacking_ensemble.meta_model.coef_ = np.array([[0.5, -0.3]])

    weights = stacking_ensemble.get_model_weights()

    # 验证权重
    assert weights == {"model_0": 0.5, "model_1": -0.3}

def test_stacking_error_handling(stacking_ensemble, sample_data):
    """测试堆叠集成错误处理"""
    X, y = sample_data

    # 未训练时预测
    with pytest.raises(RuntimeError, match="Please fit the model first"):
        stacking_ensemble.predict(X)

    # 不支持概率预测
    stacking_ensemble.use_probas = True
    stacking_ensemble.fit(X, y)
    with pytest.raises(NotImplementedError, match="does not support predict_proba"):
        stacking_ensemble.predict_proba(X)

def test_weighted_average_error_handling(weighted_ensemble, sample_data):
    """测试加权平均错误处理"""
    X, y = sample_data

    # 未训练时预测
    with pytest.raises(RuntimeError, match="Please fit the model first"):
        weighted_ensemble.predict(X)

    # 未配置概率预测
    with pytest.raises(NotImplementedError, match="not configured to use probabilities"):
        weighted_ensemble.predict_proba(X)
