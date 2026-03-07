# -*- coding: utf-8 -*-
"""
机器学习层覆盖率提升测试（简化版）

针对机器学习层的关键边界条件创建测试用例
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.ml.core.ml_core import MLCore
from src.ml.core.exceptions import MLException, ModelNotFoundError


class TestMLCoverageBoostSimple:
    """机器学习层覆盖率提升测试（简化版）"""

    @pytest.fixture
    def ml_core(self):
        """创建ML核心实例"""
        return MLCore()

    def test_ml_core_large_dataset_handling(self, ml_core):
        """测试大数据集处理"""
        # 创建大数据集
        np.random.seed(42)
        X_large = pd.DataFrame(np.random.randn(2000, 10))
        y_large = pd.Series(np.random.randint(0, 2, 2000))

        # 测试大规模训练
        model_id = ml_core.train_model(X_large, y_large, model_type="rf",
                                      model_params={"n_estimators": 5, "max_depth": 3})
        assert isinstance(model_id, str)

        # 测试大规模预测
        pred = ml_core.predict(model_id, X_large[:50])
        assert len(pred) == 50

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_cross_validation_boundary(self, ml_core):
        """测试交叉验证边界情况"""
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))

        # 训练模型用于交叉验证
        model_id = ml_core.train_model(X, y, model_type="rf",
                                      model_params={"n_estimators": 3})
        assert isinstance(model_id, str)

        # 测试交叉验证
        cv_result = ml_core.cross_validate(X, y, model_type="rf",
                                          model_params={"n_estimators": 3})
        assert cv_result is not None

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_feature_importance_edge_cases(self, ml_core):
        """测试特征重要性边界情况"""
        X = pd.DataFrame(np.random.randn(100, 4))
        y = pd.Series(np.random.randint(0, 2, 100))

        # 训练模型
        model_id = ml_core.train_model(X, y, model_type="rf",
                                      model_params={"n_estimators": 5})
        assert isinstance(model_id, str)

        # 测试特征重要性
        importance = ml_core.get_feature_importance(model_id)
        assert isinstance(importance, dict)
        assert len(importance) > 0

        # 清理
        ml_core.delete_model(model_id)

    def test_ml_core_error_recovery_scenarios(self, ml_core):
        """测试错误恢复场景"""
        # 测试无效输入的错误处理
        X_invalid = pd.DataFrame()
        y_invalid = pd.Series([])

        with pytest.raises(MLException):
            ml_core.train_model(X_invalid, y_invalid, model_type="rf")

        # 测试不存在的模型预测
        X_test = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(ModelNotFoundError):
            ml_core.predict("nonexistent_model", X_test)

    def test_ml_core_model_lifecycle_management(self, ml_core):
        """测试模型生命周期管理"""
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))

        # 创建和训练模型
        model_id = ml_core.train_model(X, y, model_type="rf",
                                      model_params={"n_estimators": 3})
        assert isinstance(model_id, str)

        # 检查模型信息
        info = ml_core.get_model_info(model_id)
        assert info is not None

        # 列出模型
        models = ml_core.list_models()
        assert model_id in models

        # 删除模型
        result = ml_core.delete_model(model_id)
        assert result is True

        # 验证删除成功
        models_after = ml_core.list_models()
        assert model_id not in models_after

    def test_ml_core_evaluation_comprehensive(self, ml_core):
        """测试模型评估综合功能"""
        X = pd.DataFrame(np.random.randn(100, 4))
        y = pd.Series(np.random.randint(0, 2, 100))

        # 训练模型
        model_id = ml_core.train_model(X, y, model_type="rf",
                                      model_params={"n_estimators": 5})
        assert isinstance(model_id, str)

        # 评估模型
        eval_result = ml_core.evaluate_model(model_id, X, y)
        assert eval_result is not None
        assert isinstance(eval_result, dict)

        # 清理
        ml_core.delete_model(model_id)
