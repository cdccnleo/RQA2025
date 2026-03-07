import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.features.processors.feature_standardizer import FeatureStandardizer


def _make_df(rows: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(rows, dtype=float),
            "y": np.arange(rows, dtype=float) * 2.0,
        }
    )


def test_fit_transform_training_saves_and_transform_roundtrip(tmp_path: Path) -> None:
    df = _make_df()
    std = FeatureStandardizer(model_path=tmp_path, method="standard")

    out = std.fit_transform(df, is_training=True)
    assert not out.empty
    assert std.is_fitted
    assert std.scaler_path.exists()

    out2 = std.transform(df)
    assert out2.equals(out)

    inv = std.inverse_transform(out)
    # 允许数值误差
    pd.testing.assert_frame_equal(inv.round(6), df.round(6))


def test_fit_transform_inference_missing_file_returns_original(tmp_path: Path) -> None:
    df = _make_df()
    std = FeatureStandardizer(model_path=tmp_path, method="standard")
    out = std.fit_transform(df, is_training=False)
    # 文件不存在时直接返回原始数据
    pd.testing.assert_frame_equal(out, df.select_dtypes(include=[np.number]))


def test_transform_and_inverse_require_fitted(tmp_path: Path) -> None:
    df = _make_df()
    std = FeatureStandardizer(model_path=tmp_path, method="standard")
    with pytest.raises(RuntimeError):
        std.transform(df)
    with pytest.raises(RuntimeError):
        std.inverse_transform(df)


def test_fit_transform_persist_failure_warns_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_df()
    std = FeatureStandardizer(model_path=tmp_path, method="standard")

    import src.features.processors.feature_standardizer as mod

    def _boom(*args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(mod.joblib, "dump", _boom)
    out = std.fit_transform(df, is_training=True)
    assert not out.empty

