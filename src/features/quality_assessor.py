#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高层质量评估入口

封装 `FeatureQualityAssessor`，提供数据质量检测、改进与评分接口，
供 tests/unit/features/test_quality_assessor.py 等用例调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import logging

from .processors.feature_quality_assessor import FeatureQualityAssessor


@dataclass
class QualityAssessorConfig:
    """质量评估与改进配置"""

    outlier_zscore: float = 3.0
    missing_value_strategy: str = "median"  # median / mean / zero
    clip_quantiles: float = 0.01


class QualityAssessor:
    """面向业务的质量评估器"""

    def __init__(self, config: Optional[QualityAssessorConfig] = None):
        self.config = config or QualityAssessorConfig()
        self.logger = logging.getLogger(__name__)
        self.feature_assessor = FeatureQualityAssessor()

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def assess_quality(self, features: pd.DataFrame) -> Dict[str, Any]:
        """对整组特征进行质量评估"""
        df = self._ensure_dataframe(features)
        issues = self._detect_issues(df)

        quality_detail = self.feature_assessor.assess_feature_quality(df)
        quality_scores = quality_detail.get("quality_scores", {})

        overall_score = (
            float(np.mean(list(quality_scores.values())))
            if quality_scores else 0.0
        )

        report = {
            "score": round(overall_score, 4),
            "issues": issues,
            "quality_scores": quality_scores,
            "comprehensive_report": quality_detail.get("comprehensive_report", {}),
        }
        return report

    def improve_quality(self, features: pd.DataFrame) -> pd.DataFrame:
        """对数据进行简单清洗，用于测试场景"""
        df = self._ensure_dataframe(features).copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]

        if len(numeric_cols) > 0:
            if self.config.missing_value_strategy == "median":
                fill_values = df[numeric_cols].median()
            elif self.config.missing_value_strategy == "mean":
                fill_values = df[numeric_cols].mean()
            else:
                fill_values = 0
            df[numeric_cols] = df[numeric_cols].fillna(fill_values)

            # Clip extreme outliers
            lower = df[numeric_cols].quantile(self.config.clip_quantiles)
            upper = df[numeric_cols].quantile(1 - self.config.clip_quantiles)
            df[numeric_cols] = df[numeric_cols].clip(lower, upper, axis=1)

        if non_numeric_cols:
            df[non_numeric_cols] = df[non_numeric_cols].fillna(
                df[non_numeric_cols].mode().iloc[0]
            )

        return df

    # ------------------------------------------------------------------ #
    # 辅助方法
    # ------------------------------------------------------------------ #
    def _ensure_dataframe(self, features: Any) -> pd.DataFrame:
        if isinstance(features, pd.Series):
            return features.to_frame()
        if isinstance(features, pd.DataFrame):
            return features.copy()
        raise TypeError("features 必须是 pandas DataFrame 或 Series")

    def _detect_issues(self, df: pd.DataFrame) -> List[str]:
        issues: List[str] = []

        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            issues.append(f"存在缺失值列: {missing_cols}")

        constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
        if constant_cols:
            issues.append(f"检测到常量列: {constant_cols}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            zscore = np.abs(
                (df[numeric_cols] - df[numeric_cols].mean())
                / (df[numeric_cols].std(ddof=0).replace(0, np.nan))
            )
            outlier_ratio = (zscore > self.config.outlier_zscore).sum().sum() / (
                df[numeric_cols].shape[0] * max(len(numeric_cols), 1)
            )
            if outlier_ratio > 0.01:
                issues.append(f"异常值比例为 {outlier_ratio:.2%}")

        return issues

