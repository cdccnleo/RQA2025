from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.exceptions import NotFittedError

from src.features.core.config_integration import ConfigScope
from src.features.processors.feature_selector import FeatureSelector


@pytest.fixture(autouse=True)
def mock_config_manager(monkeypatch):
    stub = SimpleNamespace(
        get_config=lambda scope: {},
        register_config_watcher=lambda scope, cb: None,
    )
    monkeypatch.setattr(
        "src.features.processors.feature_selector.get_config_integration_manager",
        lambda: stub,
    )
    return stub


@pytest.fixture
def sample_features():
    return pd.DataFrame(
        {
            "trend": [1.0, 2.0, 3.0, 4.0],
            "momentum": [2.0, 4.0, 6.0, 8.0],
            "noise": [10.0, 9.0, 8.0, 7.0],
        }
    )


@pytest.fixture
def sample_target():
    return pd.Series([0.1, 0.2, 0.3, 0.4])


def test_kbest_selector_preserves_features(tmp_path, sample_features, sample_target):
    selector = FeatureSelector(
        selector_type="kbest",
        n_features=1,
        preserve_features=["noise"],
        model_path=tmp_path,
    )

    selector.fit(sample_features, sample_target)
    assert selector.is_fitted is True
    assert "noise" in selector.selected_features

    transformed = selector.transform(sample_features)
    assert set(transformed.columns) == set(selector.selected_features)


def test_select_variance_strategy(sample_features):
    selector = FeatureSelector(selector_type="variance", threshold=0.0)
    result = selector.select_features(sample_features)
    assert result.shape[1] == sample_features.shape[1]


def test_select_correlation_strategy(sample_features, sample_target):
    selector = FeatureSelector(selector_type="correlation", threshold=0.5)
    selected = selector.select_features(sample_features, sample_target)
    assert not selected.empty
    assert "trend" in selected.columns


def test_transform_without_fit_returns_original(sample_features):
    selector = FeatureSelector(selector_type="kbest", n_features=2)
    transformed = selector.transform(sample_features)
    pd.testing.assert_frame_equal(transformed, sample_features)


def test_fit_handles_empty_features(sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    empty = pd.DataFrame()
    selector.fit(empty, sample_target, is_training=True)
    assert selector.selected_features == []
    assert selector.is_fitted is False


def test_update_selector_params_changes_type(sample_features, sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=2)
    selector.update_selector_params(selector_type="variance")
    assert selector.selector_type == "variance"
    result = selector.select_features(sample_features, sample_target)
    assert not result.empty


def test_update_selector_params_unknown_key_logs_warning(caplog):
    selector = FeatureSelector(selector_type="kbest", n_features=2)
    with caplog.at_level("WARNING"):
        selector.update_selector_params(non_existing=1)
    assert any("未知参数" in message for message in caplog.messages)


def test_fit_with_missing_target(sample_features):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    with pytest.raises(ValueError):
        selector.fit(sample_features, None)


def test_select_features_without_target_returns_input():
    selector = FeatureSelector(selector_type="correlation", threshold=0.3)
    features = pd.DataFrame({"a": [1, 2, 3]})
    selected = selector.select_features(features)
    pd.testing.assert_frame_equal(selected, features)


def test_transform_after_fit_handles_new_columns(sample_features, sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    selector.fit(sample_features, sample_target)
    expanded = sample_features.assign(extra=1)
    transformed = selector.transform(expanded)
    assert set(transformed.columns) == set(selector.selected_features)


def test_fit_with_zero_variance_feature(sample_target):
    selector = FeatureSelector(selector_type="variance", threshold=0.1)
    features = pd.DataFrame({"constant": [1, 1, 1, 1], "varying": [1, 2, 3, 4]})
    selected = selector.select_features(features, sample_target)
    assert "constant" not in selected.columns


def test_transform_returns_empty_when_no_selected_features(sample_features, sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    selector.fit(pd.DataFrame(), sample_target, is_training=True)
    transformed = selector.transform(sample_features)
    pd.testing.assert_frame_equal(transformed, sample_features)


def test_transform_raises_on_missing_columns(sample_features, sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=2)
    selector.fit(sample_features, sample_target)
    reduced = sample_features.drop(columns=["trend"])
    transformed = selector.transform(reduced)
    assert set(transformed.columns) == set(selector.selected_features)


def test_fit_accepts_numpy_input(sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    selector.fit(features, sample_target, is_training=True)
    assert selector.is_fitted is True


def test_fit_with_numpy_target(sample_features):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    target = np.array([0.1, 0.2, 0.3, 0.4])
    selector.fit(sample_features, target, is_training=True)
    assert selector.is_fitted is True


def test_load_selector_raises_without_model_path():
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    selector.model_path = None
    with pytest.raises(RuntimeError):
        selector._load_selector()


def test_load_selector_file_not_found(tmp_path, caplog):
    selector = FeatureSelector(selector_type="kbest", n_features=1, model_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        selector._load_selector()
    assert any("特征选择器文件未找到" in message for message in caplog.messages)


def test_load_selector_success(tmp_path):
    selector = FeatureSelector(selector_type="kbest", n_features=1, model_path=tmp_path)
    selector.selector = SelectKBest(f_regression, k=1)
    selector.selected_features = ["a"]
    selector._save_selector()

    new_selector = FeatureSelector(selector_type="kbest", n_features=1, model_path=tmp_path)
    new_selector._load_selector()
    assert isinstance(new_selector.selector, SelectKBest)
    assert new_selector.selected_features == ["a"]


def test_get_selected_features_handles_missing_support(sample_features, sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    selector.fit(sample_features, sample_target)
    selector.selector = SimpleNamespace()
    with pytest.raises(NotFittedError):
        selector._get_selected_features(sample_features)


def test_update_selector_params_resets_state(sample_features, sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    selector.fit(sample_features, sample_target)
    selector.update_selector_params(n_features=2)
    assert selector.n_features == 2
    assert selector.is_fitted is False


def test_on_config_change_reinitializes_selector(sample_features, sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    selector.fit(sample_features, sample_target)
    selector._on_config_change(ConfigScope.PROCESSING, "selector_type", "kbest", "rfecv")
    assert selector.selector_type == "rfecv"
    assert selector.selector.__class__.__name__.lower().startswith("rfecv")


def test_select_by_strategy_importance_requires_target(sample_features):
    selector = FeatureSelector(selector_type="importance")
    result = selector.select_features(sample_features, target=None)
    pd.testing.assert_frame_equal(result, sample_features)


def test_select_by_strategy_importance_with_target(sample_features, sample_target):
    selector = FeatureSelector(selector_type="importance")
    result = selector.select_features(sample_features, sample_target)
    assert not result.empty


def test_invalid_selector_type_raises(monkeypatch):
    with pytest.raises(ValueError):
        FeatureSelector(selector_type="unsupported")


def test_fit_with_invalid_target_type(sample_features):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    with pytest.raises(TypeError):
        selector.fit(sample_features, target={"bad": "type"})


def test_transform_with_empty_features_returns_empty(sample_features, sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    selector.fit(sample_features, sample_target)
    empty = pd.DataFrame()
    transformed = selector.transform(empty)
    assert transformed.empty


def test_get_selected_features_handles_exception(sample_features, sample_target, monkeypatch, caplog):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    selector.fit(sample_features, sample_target)

    class BrokenSelector:
        support_ = True

        def get_support(self):
            raise RuntimeError("boom")

    selector.selector = BrokenSelector()
    with caplog.at_level("ERROR"):
        selected = selector._get_selected_features(sample_features)
    assert selected == []
    assert any("获取选中特征失败" in msg for msg in caplog.messages)


def test_on_config_change_updates_threshold(sample_features, sample_target):
    selector = FeatureSelector(selector_type="correlation", threshold=0.3)
    selector._on_config_change(ConfigScope.PROCESSING, "threshold", 0.3, 0.8)
    assert selector.threshold == 0.8


def test_init_raises_when_min_features_invalid():
    with pytest.raises(ValueError):
        FeatureSelector(selector_type="kbest", min_features_to_select=0)


def test_transform_missing_columns_detects_difference(sample_features, sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=2)
    selector.fit(sample_features, sample_target)
    selector.original_feature_names = list(sample_features.columns)
    reduced = sample_features.drop(columns=["momentum"])
    with pytest.raises(ValueError, match="输入特征缺少列"):
        selector.transform(reduced)


def test_transform_returns_empty_when_selected_features_cleared(sample_features, sample_target):
    selector = FeatureSelector(selector_type="kbest", n_features=1)
    selector.fit(sample_features, sample_target)
    selector.selected_features = []
    transformed = selector.transform(sample_features)
    assert transformed.empty


def test_load_selector_failure_raises_runtime_error(tmp_path, monkeypatch):
    selector = FeatureSelector(selector_type="kbest", n_features=1, model_path=tmp_path)
    file_path = tmp_path / "feature_selector.pkl"
    file_path.write_text("invalid")
    features_path = tmp_path / "selected_features.pkl"
    features_path.write_text("invalid")

    def boom_load(_path):
        raise ValueError("broken load")

    monkeypatch.setattr("joblib.load", boom_load)
    with pytest.raises(RuntimeError, match="加载特征选择器失败"):
        selector._load_selector()

