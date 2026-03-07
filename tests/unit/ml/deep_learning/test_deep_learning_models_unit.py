"""
测试深度学习模型
"""

import pytest
from src.ml.deep_learning.deep_learning_models import DeepLearningModel


class TestDeepLearningModel:
    """测试深度学习模型"""

    def test_deep_learning_model_creation(self):
        """测试深度学习模型创建"""
        model = DeepLearningModel(
            name="test_model",
            parameters={"learning_rate": 0.01, "epochs": 100}
        )
        assert model.name == "test_model"
        assert model.parameters["learning_rate"] == 0.01
        assert model.parameters["epochs"] == 100

    def test_deep_learning_model_predict(self):
        """测试深度学习模型预测"""
        model = DeepLearningModel(
            name="simple_model",
            parameters={"param1": "value1"}
        )

        # 测试预测方法
        inputs = [1, 2, 3, 4]
        result = model.predict(inputs)
        assert result == inputs  # 当前实现直接返回输入

    def test_deep_learning_model_with_empty_parameters(self):
        """测试深度学习模型空参数"""
        model = DeepLearningModel(
            name="empty_model",
            parameters={}
        )
        assert model.name == "empty_model"
        assert model.parameters == {}

    def test_deep_learning_model_predict_with_dict_input(self):
        """测试深度学习模型字典输入预测"""
        model = DeepLearningModel(
            name="dict_model",
            parameters={"type": "dict_input"}
        )

        input_dict = {"data": [1, 2, 3], "shape": (3,)}
        result = model.predict(input_dict)
        assert result == input_dict

    def test_deep_learning_model_predict_with_none_input(self):
        """测试深度学习模型None输入预测"""
        model = DeepLearningModel(
            name="none_model",
            parameters={"handle_none": True}
        )

        result = model.predict(None)
        assert result is None
