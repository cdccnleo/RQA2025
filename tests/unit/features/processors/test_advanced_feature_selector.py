import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.processors.advanced_feature_selector import (
    AdvancedFeatureSelector,
    FeatureImportance,
    SelectionMethod,
    SelectionResult,
    TaskType,
)


@pytest.fixture(autouse=True)
def silence_logger(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.advanced_feature_selector.logger",
        logging.getLogger(__name__),
    )


@pytest.fixture
def selector(tmp_path):
    cache_dir = tmp_path / "selector_cache"
    cache_dir.mkdir()
    return AdvancedFeatureSelector(
        task_type=TaskType.REGRESSION,
        random_state=42,
        n_jobs=1,
        cache_dir=str(cache_dir),
    )


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(100, 6)),
        columns=[f"f{i}" for i in range(6)],
    )
    y = X["f0"] * 0.8 + rng.normal(scale=0.1, size=100)
    return X, pd.Series(y, name="target")


def test_select_features_kbest(selector, regression_data):
    X, y = regression_data
    results = selector.select_features(
        X,
        y,
        methods=[SelectionMethod.K_BEST],
        max_features=3,
        min_features=2,
        cv_folds=3,
        use_cache=False,
    )
    result = results[SelectionMethod.K_BEST.value]
    assert len(result.selected_features) <= 3
    assert result.selection_method == SelectionMethod.K_BEST.value


def test_select_features_mutual_info(selector, regression_data):
    X, y = regression_data
    results = selector.select_features(
        X,
        y,
        methods=[SelectionMethod.MUTUAL_INFO],
        max_features=4,
        use_cache=False,
    )
    result = results[SelectionMethod.MUTUAL_INFO.value]
    assert result.feature_importances is not None
    assert result.selection_method == SelectionMethod.MUTUAL_INFO.value


def test_select_features_caching(selector, regression_data):
    X, y = regression_data
    selector.select_features(
        X,
        y,
        methods=[SelectionMethod.K_BEST],
        max_features=3,
        use_cache=True,
    )
    cache_dir = Path(selector.cache_dir)
    if cache_dir.exists():
        assert any(cache_dir.iterdir())
    second_run = selector.select_features(
        X,
        y,
        methods=[SelectionMethod.K_BEST],
        max_features=3,
        use_cache=True,
    )
    assert SelectionMethod.K_BEST.value in second_run


def test_select_features_invalid_method(selector, regression_data):
    X, y = regression_data
    with pytest.raises(ValueError):
        selector.select_features(X, y, methods=["unknown"], use_cache=False)


def test_validate_input_errors(selector):
    with pytest.raises(ValueError):
        selector._validate_input(pd.DataFrame(), pd.Series())
    with pytest.raises(ValueError):
        selector._validate_input(pd.DataFrame({"a": [1]}), pd.Series(dtype=float))
    with pytest.raises(ValueError):
        selector._validate_input(pd.DataFrame({"a": [1, 2]}), pd.Series([1]))


def test_select_with_method_routes(monkeypatch, selector, regression_data):
    X, y = regression_data

    def make_result(name):
        return SelectionResult(
            selected_features=["f0"],
            feature_importances=[FeatureImportance("f0", 1.0, 1, name)],
            selection_method=name,
            selection_time=0.0,
        )

    methods_to_patch = {
        SelectionMethod.K_BEST: "_select_k_best",
        SelectionMethod.RFECV: "_select_rfecv",
        SelectionMethod.SELECT_FROM_MODEL: "_select_from_model",
        SelectionMethod.MUTUAL_INFO: "_select_mutual_info",
        SelectionMethod.PERMUTATION: "_select_permutation",
        SelectionMethod.CORRELATION: "_select_correlation",
        SelectionMethod.VARIANCE: "_select_variance",
        SelectionMethod.PCA: "_select_pca",
    }

    for method, attr in methods_to_patch.items():
        monkeypatch.setattr(
            selector,
            attr,
            lambda *args, _method=method: make_result(_method.value),
        )
        result = selector._select_with_method(
            X, y, method, max_features=2, min_features=1, cv_folds=2, scoring=None
        )
        assert result.selection_method == method.value
        assert result.selection_time >= 0.0

    with pytest.raises(ImportError):
        selector._select_with_method(
            X,
            y,
            SelectionMethod.BORUTA,
            max_features=2,
            min_features=1,
            cv_folds=2,
            scoring=None,
        )


def test_select_boruta_raises(selector, regression_data):
    X, y = regression_data
    with pytest.raises(ImportError):
        selector._select_boruta(X, y)


def test_cache_save_and_load(monkeypatch, selector, regression_data):
    X, y = regression_data
    results = {"dummy": SelectionResult([], [], "dummy", 0.0)}
    key = selector._generate_cache_key(X, y, [SelectionMethod.K_BEST])

    def fake_dump(*args, **kwargs):
        raise RuntimeError("dump failed")

    monkeypatch.setattr(
        "src.features.processors.advanced_feature_selector.joblib.dump",
        fake_dump,
    )
    selector._save_to_cache(X, y, [SelectionMethod.K_BEST], results)

    cache_file = Path(selector.cache_dir) / f"{key}.pkl"
    cache_file.write_text("corrupted")

    def fake_load(*args, **kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        "src.features.processors.advanced_feature_selector.joblib.load",
        fake_load,
    )
    assert selector._load_from_cache(X, y, [SelectionMethod.K_BEST]) is None


def test_feature_importance_summary(selector):
    summary = selector.get_feature_importance_summary({})
    assert summary.empty

    result = SelectionResult(
        selected_features=["f0"],
        feature_importances=[
            FeatureImportance("f0", 0.8, 1, "k_best", p_value=0.01),
            FeatureImportance("f0", 0.6, 2, "mutual_info"),
        ],
        selection_method="k_best",
        selection_time=0.1,
    )
    summary = selector.get_feature_importance_summary({"k_best": result})
    assert "avg_rank" in summary.columns
    assert summary.iloc[0]["feature_name"] == "f0"


def test_evaluate_selection_performance(monkeypatch, selector, regression_data):
    X, y = regression_data

    class DummyModel:
        def fit(self, X, y):
            return self

    monkeypatch.setattr(
        "src.features.processors.advanced_feature_selector.RandomForestRegressor",
        lambda *args, **kwargs: DummyModel(),
    )
    monkeypatch.setattr(
        "src.features.processors.advanced_feature_selector.cross_val_score",
        lambda model, X, y, cv, scoring, n_jobs: np.array([0.9, 0.8]),
    )

    result = SelectionResult(
        selected_features=["f0", "f1"],
        feature_importances=[],
        selection_method="k_best",
        selection_time=0.0,
    )
    scores = selector.evaluate_selection_performance(X, y, {"k_best": result}, cv_folds=2)
    assert scores["k_best"] < 0.0  # neg MSE expected for regression

    empty_result = SelectionResult(
        selected_features=[],
        feature_importances=[],
        selection_method="mutual_info",
        selection_time=0.0,
    )
    scores = selector.evaluate_selection_performance(X, y, {"mutual_info": empty_result})
    assert scores["mutual_info"] == 0.0



