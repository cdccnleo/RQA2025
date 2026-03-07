import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.processors.feature_correlation import FeatureCorrelationAnalyzer


def test_export_and_plot_correlation_report(tmp_path: Path) -> None:
    # 构造简单数据，确保相关性矩阵可用
    df = pd.DataFrame(
        {
            "a": np.arange(10, dtype=float),
            "b": np.arange(10, dtype=float) * 2.0,
            "c": np.arange(10, dtype=float) + 1.0,
        }
    )
    analyzer = FeatureCorrelationAnalyzer()
    analyzer.analyze_feature_correlation(df)

    # 导出报告分支
    out_json = tmp_path / "corr_report.json"
    analyzer.export_correlation_report(str(out_json))
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "correlation_matrix" in data
    assert "vif_scores" in data

    # 保存热力图分支
    out_png = tmp_path / "corr_heatmap.png"
    analyzer.plot_correlation_heatmap(save_path=str(out_png))
    assert out_png.exists()


def test_group_correlated_features_empty_and_pair() -> None:
    analyzer = FeatureCorrelationAnalyzer()

    # 空输入返回空列表
    assert analyzer._group_correlated_features([]) == []

    # 单对构成2元素组（>1才计入）
    pairs = [{"feature1": "x", "feature2": "y", "correlation": 0.9}]
    groups = analyzer._group_correlated_features(pairs)
    assert groups and {"x", "y"} == set(groups[0])


def test_get_feature_recommendations_coverage() -> None:
    analyzer = FeatureCorrelationAnalyzer()
    # 先设置必要的状态以覆盖 keep/remove/merge 三类
    analyzer.vif_scores = {"k1": 100.0, "k2": 1.0, "k3": analyzer.config["vif_threshold"] / 1.5}
    analyzer.multicollinearity_groups = [["k1", "k2"], ["z"]]

    rec = analyzer.get_feature_recommendations()
    assert any(rec.values())

