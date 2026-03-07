import sys
import builtins
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.features.processors.feature_importance import FeatureImportanceAnalyzer, permutation_importance


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=2, random_state=42
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return X, y, feature_names


def test_calculate_permutation_importance(classification_data):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=42).fit(X, y)

    analyzer = FeatureImportanceAnalyzer(model)
    scores = analyzer.calculate_permutation_importance(X, y, feature_names, n_repeats=5)

    assert set(scores.keys()) == set(feature_names)
    assert analyzer.importance_std is not None
    assert len(analyzer.importance_std) == len(feature_names)


def test_get_top_features_requires_prior_calculation(classification_data):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=0).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model)
    analyzer.calculate_permutation_importance(X, y, feature_names, n_repeats=3)

    top = analyzer.get_top_features(top_n=2)
    assert len(top) == 2
    assert top[0] in feature_names


def test_plot_importance_returns_figure(classification_data):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=0).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model)
    analyzer.calculate_permutation_importance(X, y, feature_names, n_repeats=3)

    fig = analyzer.plot_importance(top_n=3)
    assert fig is not None


def test_calculate_shap_values_requires_shap(classification_data, monkeypatch):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=0).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model)

    monkeypatch.delitem(sys.modules, "shap", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("missing shap")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # 如果 shap 未安装，应抛出 ImportError，与实际环境保持一致
    with pytest.raises(ImportError):
        analyzer.calculate_shap_values(X, feature_names)


def test_calculate_permutation_importance_mismatch_features(classification_data):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=0).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model)
    with pytest.raises(ValueError):
        analyzer.calculate_permutation_importance(X, y, feature_names[:-1])


def test_calculate_permutation_importance_handles_import_error(monkeypatch, classification_data):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=0).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model)

    monkeypatch.setattr(
        "src.features.processors.feature_importance.permutation_importance",
        lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("no sklearn")),
    )

    scores = analyzer.calculate_permutation_importance(X, y, feature_names, n_repeats=3)

    assert set(scores.keys()) == set(feature_names)
    assert all(value >= 0 for value in scores.values())


def test_get_top_features_without_calculation_raises(classification_data):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=0).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model)
    with pytest.raises(RuntimeError):
        analyzer.get_top_features()


def test_plot_importance_without_calculation_raises(classification_data):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=0).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model)
    with pytest.raises(RuntimeError):
        analyzer.plot_importance()


def test_calculate_shap_values_tree_path(monkeypatch, classification_data):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=0).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model)

    class TreeExplainerStub:
        def __init__(self, model):
            self.model = model

        def shap_values(self, data):
            # 返回多分类列表以覆盖平均逻辑
            base = np.arange(data.shape[0] * data.shape[1]).reshape(data.shape[0], data.shape[1])
            return [base, base * 0.5]

    class KernelExplainerStub:
        def __init__(self, predict, background):
            self.predict = predict
            self.background = background

        def shap_values(self, data):
            return np.ones((data.shape[0], data.shape[1]))

    def summary_plot(values, data, feature_names=None, plot_type=None, show=None, **kwargs):
        plt.figure()

    stub = SimpleNamespace(
        TreeExplainer=TreeExplainerStub,
        KernelExplainer=KernelExplainerStub,
        summary_plot=summary_plot,
    )

    monkeypatch.setitem(sys.modules, "shap", stub)

    results = analyzer.calculate_shap_values(X, feature_names, explainer_type="tree")
    assert set(results.keys()) == set(feature_names)
    for arr in results.values():
        assert arr.shape[0] == X.shape[0]


def test_calculate_shap_values_kernel_path(monkeypatch, classification_data):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=0).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model)

    class KernelExplainerStub:
        def __init__(self, predict, background):
            self.predict = predict
            self.background = background

        def shap_values(self, data):
            return np.arange(data.shape[0] * data.shape[1]).reshape(data.shape[0], data.shape[1])

    stub = SimpleNamespace(
        TreeExplainer=None,
        KernelExplainer=KernelExplainerStub,
        summary_plot=lambda *args, **kwargs: plt.figure(),
    )

    monkeypatch.setitem(sys.modules, "shap", stub)

    values = analyzer.calculate_shap_values(X, feature_names, explainer_type="kernel")
    assert set(values.keys()) == set(feature_names)


def test_plot_shap_summary_returns_figure(monkeypatch, classification_data):
    X, y, feature_names = classification_data
    model = RandomForestClassifier(random_state=0).fit(X, y)
    analyzer = FeatureImportanceAnalyzer(model)

    class TreeExplainerStub:
        def __init__(self, model):
            self.model = model

        def shap_values(self, data):
            return np.ones((data.shape[0], data.shape[1]))

    def summary_plot(values, data, feature_names=None, plot_type=None, show=None, **kwargs):
        plt.figure()

    stub = SimpleNamespace(
        TreeExplainer=TreeExplainerStub,
        KernelExplainer=None,
        summary_plot=summary_plot,
    )

    monkeypatch.setitem(sys.modules, "shap", stub)

    fig = analyzer.plot_shap_summary(X, feature_names)
    assert fig is not None

