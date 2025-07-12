import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from src.models.evaluation.model_evaluator import ModelEvaluator
from src.models.evaluation.cross_validator import CrossValidator

class TestEvaluationIntegration:
    @pytest.fixture
    def classification_data(self):
        X, y = make_classification(
            n_samples=100,
            n_classes=2,
            n_features=5,
            random_state=42
        )
        return X, y

    def test_full_evaluation_flow(self, classification_data):
        X, y = classification_data
        model = LogisticRegression(random_state=42)

        # 交叉验证阶段
        validator = CrossValidator(model, n_splits=5)
        cv_results = validator.k_fold_validate(X, y)

        assert 'accuracy' in cv_results
        assert 0 <= cv_results['accuracy'] <= 1

        # 最终评估阶段
        model.fit(X, y)  # 在全量数据上训练
        evaluator = ModelEvaluator(model, 'classification')
        final_metrics = evaluator.evaluate(X, y)

        assert 'accuracy' in final_metrics
        assert abs(cv_results['accuracy'] - final_metrics['accuracy']) < 0.2

    def test_modules_integration(self, classification_data):
        X, y = classification_data
        model = LogisticRegression(random_state=42)

        # 使用交叉验证结果指导最终评估
        validator = CrossValidator(model, n_splits=5)
        validator.k_fold_validate(X, y)

        # 从交叉验证中选择最佳参数
        model.set_params(C=1.0)  # 模拟参数调整
        evaluator = ModelEvaluator(model, 'classification')
        final_metrics = evaluator.evaluate(X, y)

        assert final_metrics['accuracy'] > 0.7  # 确保模型基本可用

    @pytest.mark.performance
    def test_evaluation_performance(self, classification_data):
        X, y = classification_data
        model = LogisticRegression(random_state=42)
        evaluator = ModelEvaluator(model, 'classification')

        import time
        start_time = time.time()
        for _ in range(10):
            evaluator.evaluate(X, y)
        duration = time.time() - start_time

        assert duration < 1.0  # 10次评估应在1秒内完成

    def test_report_generation_flow(self, classification_data):
        X, y = classification_data
        model = LogisticRegression(random_state=42)

        # 完整报告生成流程
        validator = CrossValidator(model, n_splits=3)
        validator.k_fold_validate(X, y)

        model.fit(X, y)
        evaluator = ModelEvaluator(model, 'classification')
        evaluator.evaluate(X, y)

        report = evaluator.generate_report(include_cv=True, cv_results=validator)
        assert "Cross Validation Results" in report
        assert "Final Evaluation Metrics" in report
        assert "Model Information" in report
