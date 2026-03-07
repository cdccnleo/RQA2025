import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.features.processors.feature_selector import FeatureSelector


def _make_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = np.linspace(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = x1 * 0.1 + rng.normal(0, 0.01, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


def test_select_variance_returns_top_variance() -> None:
    df = _make_df()
    fs = FeatureSelector(selector_type="variance")
    out = fs.select(df)
    # 方差最大的列应被保留（至少不是空）
    assert not out.empty
    assert set(out.columns).issubset(set(df.columns))


def test_select_correlation_threshold_with_target() -> None:
    df = _make_df()
    target = df["x1"] + 0.001  # 与 x1 高相关
    fs = FeatureSelector(selector_type="correlation", threshold=0.9)
    out = fs.select(df, target=target)
    assert "x1" in out.columns


def test_select_importance_with_target() -> None:
    df = _make_df()
    target = (df["x1"] * 2 + df["x2"] * 0.1 + 0.01).to_numpy()
    fs = FeatureSelector(selector_type="importance")
    out = fs.select(df, target=pd.Series(target, index=df.index))
    # 重要性选择返回非空且是子集
    assert not out.empty
    assert set(out.columns).issubset(set(df.columns))


def test_transform_unfitted_returns_original() -> None:
    df = _make_df()
    fs = FeatureSelector(selector_type="rfecv", n_features=3, min_features_to_select=3, cv=2)
    out = fs.transform(df)
    # 未拟合返回原始数据
    pd.testing.assert_frame_equal(out, df)


def test_load_selector_file_not_found(tmp_path: Path) -> None:
    fs = FeatureSelector(selector_type="rfecv", model_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        fs._load_selector()


def test_update_selector_params_unknown_key_logs_warning(caplog) -> None:
    fs = FeatureSelector(selector_type="kbest", n_features=2)
    with caplog.at_level("WARNING"):
        fs.update_selector_params(non_existing=1)
    assert any("未知参数" in rec.message for rec in caplog.records)





