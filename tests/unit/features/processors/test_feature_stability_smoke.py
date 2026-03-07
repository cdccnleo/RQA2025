import numpy as np
import pandas as pd
from pathlib import Path

from src.features.processors.feature_stability import FeatureStabilityAnalyzer


def _make_df(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": np.linspace(0, 1, n),
            "b": np.linspace(1, 0, n) + np.random.randn(n) * 0.01,
        }
    )


def test_analyze_and_export_stability_report(tmp_path: Path) -> None:
    df = _make_df(60)
    time_index = pd.date_range("2024-01-01", periods=len(df), freq="D")

    analyzer = FeatureStabilityAnalyzer()
    out = analyzer.analyze_feature_stability(df, time_index=time_index)
    assert "combined_stability" in out and out["combined_stability"]

    # 导出报告分支（先设置内部评分状态）
    analyzer.stability_scores = out["combined_stability"]
    outfile = tmp_path / "stability.json"
    analyzer.export_stability_report(str(outfile))
    assert outfile.exists()




