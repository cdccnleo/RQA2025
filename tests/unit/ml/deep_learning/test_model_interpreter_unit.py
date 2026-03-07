import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import os

from src.ml.deep_learning.model_interpreter import (
    ModelInterpreter,
    SHAPInterpreter,
    LIMEInterpreter,
    explain_model_prediction,
    get_model_feature_importance,
    generate_model_explanation_report
)


def test_model_interpreter_explain_returns_importance():
    interpreter = ModelInterpreter()
    result = interpreter.explain(model={"weights": [1]}, data={"x": [1, 2]})
    assert result == {"importance": 1.0}


def test_shap_interpreter_init():
    """测试SHAP解释器初始化"""
    interpreter = SHAPInterpreter()
    assert interpreter.background_data is None

    background_data = pd.DataFrame({"x": [1, 2, 3]})
    interpreter_with_data = SHAPInterpreter(background_data=background_data)
    assert interpreter_with_data.background_data is background_data


def test_shap_interpreter_explain_with_dataframe():
    """测试SHAP解释器使用DataFrame数据"""
    interpreter = SHAPInterpreter()
    model = Mock()
    data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [0.1, 0.2, 0.3]
    })

    result = interpreter.explain(model, data)
    assert isinstance(result, dict)
    assert "feature1" in result
    assert "feature2" in result
    assert all(isinstance(v, (int, float)) for v in result.values())


def test_shap_interpreter_explain_with_array():
    """测试SHAP解释器使用数组数据"""
    interpreter = SHAPInterpreter()
    model = Mock()
    data = np.array([[1, 2], [3, 4]])

    result = interpreter.explain(model, data)
    assert isinstance(result, dict)
    assert result == {"feature_0": 0.5, "feature_1": 0.3}


def test_lime_interpreter_init():
    """测试LIME解释器初始化"""
    interpreter = LIMEInterpreter()
    assert interpreter.training_data is None

    training_data = np.array([[1, 2], [3, 4]])
    interpreter_with_data = LIMEInterpreter(training_data=training_data)
    assert interpreter_with_data.training_data is training_data


def test_lime_interpreter_explain_with_dataframe():
    """测试LIME解释器使用DataFrame数据"""
    interpreter = LIMEInterpreter()
    model = Mock()
    data = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [0.1, 0.2, 0.3]
    })

    result = interpreter.explain(model, data, instance_idx=0)
    assert isinstance(result, dict)
    assert "feature1" in result
    assert "feature2" in result
    assert all(isinstance(v, (int, float)) for v in result.values())


def test_lime_interpreter_explain_with_array():
    """测试LIME解释器使用数组数据"""
    interpreter = LIMEInterpreter()
    model = Mock()
    data = np.array([[1, 2], [3, 4]])

    result = interpreter.explain(model, data, instance_idx=1)
    assert isinstance(result, dict)
    assert result == {"feature_0": 0.5, "feature_1": 0.3}


def test_explain_model_prediction_shap():
    """测试使用SHAP方法解释模型预测"""
    model = Mock()
    data = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [0.1, 0.2]
    })

    result = explain_model_prediction(model, data, method="shap")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_explain_model_prediction_lime():
    """测试使用LIME方法解释模型预测"""
    model = Mock()
    data = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [0.1, 0.2]
    })

    result = explain_model_prediction(model, data, method="lime", instance_idx=0)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_explain_model_prediction_default():
    """测试使用默认方法解释模型预测"""
    model = Mock()
    data = {"x": [1, 2]}

    result = explain_model_prediction(model, data, method="unknown")
    assert isinstance(result, dict)
    assert result == {"importance": 1.0}


def test_get_model_feature_importance_with_importances():
    """测试获取具有feature_importances_属性的模型特征重要性"""
    model = Mock()
    model.feature_importances_ = np.array([0.3, 0.7])
    data = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [0.1, 0.2]
    })

    result = get_model_feature_importance(model, data)
    assert isinstance(result, dict)
    assert result["feature1"] == 0.3
    assert result["feature2"] == 0.7


def test_get_model_feature_importance_with_array_importances():
    """测试获取数组形式feature_importances_的模型特征重要性"""
    model = Mock()
    model.feature_importances_ = np.array([0.2, 0.5, 0.3])
    data = np.array([[1, 2, 3], [4, 5, 6]])

    result = get_model_feature_importance(model, data)
    assert isinstance(result, dict)
    assert result["feature_0"] == 0.2
    assert result["feature_1"] == 0.5
    assert result["feature_2"] == 0.3


def test_get_model_feature_importance_without_importances():
    """测试获取没有feature_importances_属性的模型特征重要性"""
    model = Mock()
    del model.feature_importances_  # 确保没有这个属性
    data = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [0.1, 0.2]
    })

    result = get_model_feature_importance(model, data)
    assert isinstance(result, dict)
    assert len(result) > 0  # 应该使用SHAP解释器


def test_generate_model_explanation_report():
    """测试生成模型解释报告"""
    model = Mock()
    model.feature_importances_ = np.array([0.4, 0.6])
    data = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [0.1, 0.2]
    })

    report = generate_model_explanation_report(model, data)
    assert isinstance(report, dict)
    assert "model_type" in report
    assert "feature_importance" in report
    assert "top_features" in report
    assert "explanation_method" in report
    assert report["explanation_method"] == "shap"


def test_generate_model_explanation_report_with_save():
    """测试生成模型解释报告并保存到文件"""
    model = Mock()
    model.feature_importances_ = np.array([0.3, 0.7])
    data = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [0.1, 0.2]
    })

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "report.json")
        report = generate_model_explanation_report(model, data, output_path)

        assert isinstance(report, dict)
        assert os.path.exists(output_path)

        # 验证文件内容
        import json
        with open(output_path, 'r') as f:
            saved_report = json.load(f)
        assert saved_report["model_type"] == "Mock"
        assert "feature_importance" in saved_report


def test_generate_model_explanation_report_save_error():
    """测试生成模型解释报告保存失败的情况"""
    model = Mock()
    model.feature_importances_ = np.array([0.5, 0.5])
    data = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [0.1, 0.2]
    })

    # 使用一个会失败的路径（通过mock json.dump来模拟）
    with patch('json.dump', side_effect=Exception("Mock save error")):
        report = generate_model_explanation_report(model, data, output_path="dummy_path.json")
        assert isinstance(report, dict)
        assert "save_error" in report
        assert "Mock save error" in report["save_error"]

