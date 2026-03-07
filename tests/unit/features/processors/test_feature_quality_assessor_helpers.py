#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.processors.feature_quality_assessor import FeatureQualityAssessor


def test_assess_feature_quality_returns_empty_report_for_blank_frame():
    assessor = FeatureQualityAssessor()
    result = assessor.assess_feature_quality(pd.DataFrame())
    assert result["quality_scores"] == {}
    assert result["importance_results"]["combined_importance"] == {}


def test_get_feature_recommendations_handles_quality_buckets():
    assessor = FeatureQualityAssessor()
    assessor.quality_scores = {"strong": 0.9, "mid": 0.6, "weak": 0.2}
    recommendations = assessor.get_feature_recommendations(threshold=0.8)
    assert recommendations == {"keep": ["strong"], "improve": ["mid"], "remove": ["weak"]}


def test_export_quality_report_persists_json(tmp_path):
    assessor = FeatureQualityAssessor()
    assessor.quality_scores = {"f1": 0.9, "f2": 0.3}
    output = tmp_path / "quality_report.json"

    assessor.export_quality_report(str(output))

    data = json.loads(output.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"quality_scores", "quality_summary", "recommendations"}
    assert data["quality_scores"]["f1"] == 0.9


def test_batch_evaluate_runs_per_column():
    assessor = FeatureQualityAssessor()
    df = pd.DataFrame({"a": np.arange(5), "b": np.linspace(0, 1, 5)})

    results = assessor.batch_evaluate(df)

    assert set(results.keys()) == {"a", "b"}
    assert all("quality_scores" in report for report in results.values())


def test_get_feature_quality_summary_statistics():
    assessor = FeatureQualityAssessor()
    assessor.quality_scores = {"x": 0.2, "y": 0.8}

    summary = assessor.get_feature_quality_summary()

    assert summary["total_features"] == 2
    assert summary["min_quality"] == 0.2
    assert summary["max_quality"] == 0.8

