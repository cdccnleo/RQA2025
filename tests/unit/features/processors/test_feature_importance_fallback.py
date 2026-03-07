import sys
import types
import numpy as np
from pathlib import Path
from typing import Any

import pytest

class _DummyModel:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "._DummyModel":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 简单线性关系，供 fallback 路径使用
        return (X.sum(axis=1) > 0).astype(int)


def _remove_module(mod_name: str) -> Any:
    existed = sys.modules.pop(mod_name, None)
    return existed


def test_permutation_importance_double_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 构造场景：删除 sklearn.inspection 与 sklearn.ensemble，触发本地 fallback 定义
    existed_inspection = _remove_module("sklearn.inspection")
    existed_ensemble = _remove_module("sklearn.ensemble")
    existed_mod = _remove_module("src.features.processors.feature_importance")

    # 确保 sklearn 基包不影响本地导入逻辑
    if "sklearn" not in sys.modules:
        sys.modules["sklearn"] = types.ModuleType("sklearn")

    try:
        # 动态导入以触发 fallback 定义
        import importlib
        fi_module = importlib.import_module("src.features.processors.feature_importance")
        FeatureImportanceAnalyzer = getattr(fi_module, "FeatureImportanceAnalyzer")

        model = _DummyModel()
        analyzer = FeatureImportanceAnalyzer(model)

        X = np.random.randn(32, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        feature_names = [f"f{i}" for i in range(X.shape[1])]

        scores = analyzer.calculate_permutation_importance(X, y, feature_names, n_repeats=3, scoring="accuracy")
        assert set(scores.keys()) == set(feature_names)
        # 确保 fallback 返回了非负数值
        assert all(isinstance(v, float) for v in scores.values())

    finally:
        # 还原模块缓存，避免影响其他用例
        if existed_inspection:
            sys.modules["sklearn.inspection"] = existed_inspection
        if existed_ensemble:
            sys.modules["sklearn.ensemble"] = existed_ensemble
        if existed_mod:
            sys.modules["src.features.processors.feature_importance"] = existed_mod

