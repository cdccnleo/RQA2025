#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自包含的特征质量评估器实现
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureQualityConfig:
    """特征质量评估配置"""

    importance_weight: float = 0.4
    correlation_weight: float = 0.3
    stability_weight: float = 0.3
    quality_threshold: float = 0.7
    min_samples: int = 5


class FeatureQualityAssessor:
    """轻量级特征质量评估器"""

    def __init__(self, config: Optional[FeatureQualityConfig] = None):
        self.config = config or FeatureQualityConfig()
        self.logger = logging.getLogger(f"{__name__}.FeatureQualityAssessor")
        self.quality_scores: Dict[str, float] = {}
        self.feature_rankings: Dict[str, List[Tuple[str, float]]] = {}
        self.importance_analyzer = self._load_optional_analyzer("feature_importance")
        self.correlation_analyzer = self._load_optional_analyzer("feature_correlation")
        self.stability_analyzer = self._load_optional_analyzer("feature_stability")

    # ------------------------------------------------------------------ #
    # 核心评估流程
    # ------------------------------------------------------------------ #
    def assess_feature_quality(
        self,
        features: Union[pd.DataFrame, pd.Series],
        target: Optional[pd.Series] = None,
        time_index: Optional[pd.DatetimeIndex] = None,
        target_type: str = "regression",
    ) -> Dict[str, Any]:
        df = self._ensure_dataframe(features)
        if df.empty:
            # 空输入时返回结构化的空报告，保障调用方键访问安全
            return self._empty_report()

        if target is not None and len(target) != len(df):
            raise ValueError("features 与 target 的长度不匹配")

        df = self._prepare_features(df)
        target_series = self._prepare_target(df, target)

        importance_results = (
            self.importance_analyzer.analyze_feature_importance(df, target_series, target_type)
            if self.importance_analyzer
            else self._assess_importance(df, target_series)
        )
        correlation_results = (
            self.correlation_analyzer.analyze_feature_correlation(df)
            if self.correlation_analyzer
            else self._assess_correlation(df)
        )
        correlation_results = self._ensure_correlation_payload(correlation_results)

        stability_results = (
            self.stability_analyzer.analyze_feature_stability(df, time_index)
            if self.stability_analyzer
            else self._assess_stability(df, time_index)
        )

        quality_scores = self._combine_quality_scores(
            importance_results["combined_importance"],
            correlation_results["correlation_scores"],
            stability_results["combined_stability"],
        )
        comprehensive_report = self._generate_comprehensive_report(
            importance_results,
            correlation_results,
            stability_results,
            quality_scores,
        )

        self.quality_scores = quality_scores
        redundant_features = self._extract_redundant_features(correlation_results)
        feature_score_payload = self._build_feature_scores(
            df.columns,
            importance_results,
            correlation_results,
            stability_results,
            quality_scores,
        )

        self.feature_rankings = {
            "importance": sorted(
                importance_results["combined_importance"].items(),
                key=lambda item: item[1],
                reverse=True,
            ),
            "correlation": sorted(
                correlation_results["correlation_scores"].items(),
                key=lambda item: item[1],
                reverse=True,
            ),
            "stability": sorted(
                stability_results["combined_stability"].items(),
                key=lambda item: item[1],
                reverse=True,
            ),
            "quality": sorted(quality_scores.items(), key=lambda item: item[1], reverse=True),
        }

        summary = comprehensive_report["summary"]

        return {
            "importance_results": importance_results,
            "correlation_results": correlation_results,
            "stability_results": stability_results,
            "quality_scores": quality_scores,
            "feature_scores": feature_score_payload,
            "redundant_features": redundant_features,
            "overall_quality": summary.get("average_quality_score", 0.0),
            "recommendations": comprehensive_report["recommendations"],
            "summary": summary,
            "comprehensive_report": comprehensive_report,
        }

    # ------------------------------------------------------------------ #
    # 公开辅助方法（供测试调用）
    # ------------------------------------------------------------------ #
    def evaluate_feature(self, feature: Union[pd.Series, pd.DataFrame], target: Optional[pd.Series] = None) -> Dict[str, Any]:
        return self.assess_feature_quality(feature, target)

    def batch_evaluate(self, features: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for column in features.columns:
            column_result = self.assess_feature_quality(features[[column]], target)
            results[column] = column_result
        return results

    def get_feature_recommendations(self, threshold: float = 0.7) -> Dict[str, List[str]]:
        if not self.quality_scores:
            return {"keep": [], "improve": [], "remove": []}

        recommendations = {"keep": [], "improve": [], "remove": []}
        for feature, score in self.quality_scores.items():
            if score >= threshold:
                recommendations["keep"].append(feature)
            elif score >= threshold * 0.7:
                recommendations["improve"].append(feature)
            else:
                recommendations["remove"].append(feature)
        return recommendations

    def get_feature_quality_summary(self) -> Dict[str, Any]:
        if not self.quality_scores:
            return {}
        values = list(self.quality_scores.values())
        return {
            "total_features": len(values),
            "average_quality": float(np.mean(values)),
            "median_quality": float(np.median(values)),
            "std_quality": float(np.std(values)),
            "min_quality": float(np.min(values)),
            "max_quality": float(np.max(values)),
        }

    def export_quality_report(self, filepath: str) -> None:
        if not self.quality_scores:
            self.logger.warning("没有质量评分，跳过导出")
            return
        report = {
            "quality_scores": self.quality_scores,
            "quality_summary": self.get_feature_quality_summary(),
            "recommendations": self.get_feature_recommendations(),
        }
        with open(filepath, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.quality_scores.items(), key=lambda item: item[1], reverse=True)[:n]

    def get_low_quality_features(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        return [
            (feature, score)
            for feature, score in sorted(self.quality_scores.items(), key=lambda item: item[1])
            if score < threshold
        ]

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _ensure_dataframe(self, features: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            return features.copy()
        if isinstance(features, pd.Series):
            return features.to_frame()
        raise TypeError("features 必须是 pandas DataFrame 或 Series")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        for column in df.columns:
            if not np.issubdtype(df[column].dtype, np.number):
                df[column] = pd.Categorical(df[column]).codes.astype(float)
        return df

    def _prepare_target(self, df: pd.DataFrame, target: Optional[pd.Series]) -> pd.Series:
        if target is None:
            return df.sum(axis=1)
        aligned = target.reindex(df.index)
        return aligned.ffill().bfill()

    def _assess_importance(self, df: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        scores: Dict[str, float] = {}
        for column in df.columns:
            corr = abs(df[column].corr(target))
            if np.isnan(corr):
                corr = 0.0
            scores[column] = corr

        normalized = self._normalize_scores(scores)
        return {
            "combined_importance": normalized,
            "analysis_report": {
                "summary": {
                    "high_importance_features": [f for f, v in normalized.items() if v >= 0.8],
                    "low_importance_features": [f for f, v in normalized.items() if v < 0.3],
                },
                "detail": normalized,
            },
        }

    def _assess_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        corr_matrix = df.corr().abs().fillna(0)
        avg_corr: Dict[str, float] = {}
        for column in df.columns:
            others = corr_matrix[column].drop(labels=column, errors="ignore")
            avg_corr[column] = float(others.mean()) if not others.empty else 0.0

        normalized = self._normalize_scores(avg_corr)
        return {
            "correlation_matrix": corr_matrix,
            "correlation_scores": normalized,
            "analysis_report": {
                "summary": {
                    "high_correlation_features": [f for f, v in normalized.items() if v > 0.7],
                    "multicollinearity_groups": int((corr_matrix > 0.9).sum().sum()),
                }
            },
        }

    def _assess_stability(
        self,
        df: pd.DataFrame,
        time_index: Optional[pd.DatetimeIndex],
    ) -> Dict[str, Any]:
        stability_scores: Dict[str, float] = {}
        for column in df.columns:
            series = df[column].dropna()
            if series.empty:
                stability_scores[column] = 0.0
                continue
            mean_val = series.mean()
            std_val = series.std(ddof=0)
            if mean_val == 0:
                stability_scores[column] = 1.0
            else:
                cv = abs(std_val / mean_val)
                stability_scores[column] = max(0.0, 1 - cv)

        normalized = self._normalize_scores(stability_scores)
        return {
            "combined_stability": normalized,
            "analysis_report": {
                "summary": {
                    "high_stability_features": [f for f, v in normalized.items() if v >= 0.8],
                    "low_stability_features": [f for f, v in normalized.items() if v < 0.3],
                }
            },
        }

    def _combine_quality_scores(
        self,
        importance_scores: Dict[str, float],
        correlation_scores: Dict[str, float],
        stability_scores: Dict[str, float],
    ) -> Dict[str, float]:
        weights = {
            "importance": self.config.importance_weight,
            "correlation": self.config.correlation_weight,
            "stability": self.config.stability_weight,
        }

        all_features = set(importance_scores) | set(correlation_scores) | set(stability_scores)
        combined: Dict[str, float] = {}
        for feature in all_features:
            importance = importance_scores.get(feature, 0.0)
            correlation = 1 - correlation_scores.get(feature, 0.0)  # 相关性越低越好
            stability = stability_scores.get(feature, 0.0)
            combined[feature] = (
                importance * weights["importance"]
                + correlation * weights["correlation"]
                + stability * weights["stability"]
            )
        return combined

    def _generate_comprehensive_report(
        self,
        importance_results: Dict[str, Any],
        correlation_results: Dict[str, Any],
        stability_results: Dict[str, Any],
        quality_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        scores = list(quality_scores.values())
        summary = {
            "total_features": len(scores),
            "average_quality_score": float(np.mean(scores)) if scores else 0.0,
            "high_quality_features": [f for f, s in quality_scores.items() if s >= self.config.quality_threshold],
            "low_quality_features": [f for f, s in quality_scores.items() if s < 0.5],
        }

        return {
            "summary": summary,
            "quality_ranking": sorted(quality_scores.items(), key=lambda item: item[1], reverse=True),
            "recommendations": self._generate_quality_recommendations(
                importance_results, correlation_results, stability_results, quality_scores
            ),
        }

    def _build_feature_scores(
        self,
        columns: pd.Index,
        importance_results: Dict[str, Any],
        correlation_results: Dict[str, Any],
        stability_results: Dict[str, Any],
        quality_scores: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        payload: Dict[str, Dict[str, float]] = {}
        for feature in columns:
            payload[feature] = {
                "importance_score": importance_results["combined_importance"].get(feature, 0.0),
                "correlation_penalty": correlation_results["correlation_scores"].get(feature, 0.0),
                "stability_score": stability_results["combined_stability"].get(feature, 0.0),
                "quality_score": quality_scores.get(feature, 0.0),
            }
        return payload

    def _extract_redundant_features(self, correlation_results: Dict[str, Any]) -> List[str]:
        summary = correlation_results["analysis_report"]["summary"]
        redundant = summary.get("high_correlation_features", [])
        return redundant.copy()

    def _generate_quality_recommendations(
        self,
        importance_results: Dict[str, Any],
        correlation_results: Dict[str, Any],
        stability_results: Dict[str, Any],
        quality_scores: Dict[str, float],
    ) -> List[str]:
        recommendations: List[str] = []
        low_importance = importance_results["analysis_report"]["summary"]["low_importance_features"]
        if low_importance:
            recommendations.append(f"重要性较低的特征: {low_importance}")

        high_corr = correlation_results["analysis_report"]["summary"]["high_correlation_features"]
        if high_corr:
            recommendations.append(f"存在多重共线性特征: {high_corr}")

        low_stability = stability_results["analysis_report"]["summary"]["low_stability_features"]
        if low_stability:
            recommendations.append(f"稳定性较差的特征: {low_stability}")

        low_quality = [f for f, score in quality_scores.items() if score < 0.5]
        if low_quality:
            recommendations.append(f"总体质量偏弱的特征: {low_quality}")

        if not recommendations:
            recommendations.append("特征质量整体稳定，保持当前配置。")

        return recommendations

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = np.array(list(scores.values()))
        max_value = values.max()
        if max_value == 0:
            return {feature: 0.0 for feature in scores}
        return {feature: float(value / max_value) for feature, value in scores.items()}

    def _ensure_correlation_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "correlation_scores" in payload:
            return payload
        vif_scores = payload.get("analysis_results", {}).get("vif_analysis", {})
        if vif_scores:
            normalized = self._normalize_scores({feature: 1 - score for feature, score in vif_scores.items()})
        else:
            normalized = {}
        payload["correlation_scores"] = normalized
        payload.setdefault("analysis_report", {}).setdefault("summary", {})
        summary = payload["analysis_report"]["summary"]
        summary.setdefault("high_correlation_features", [f for f, v in normalized.items() if v > 0.7])
        summary.setdefault("multicollinearity_groups", len(summary["high_correlation_features"]))
        payload.setdefault("correlation_matrix", pd.DataFrame())
        return payload

    def _load_optional_analyzer(self, module_name: str) -> Optional[Any]:
        try:
            module = importlib.import_module(module_name)
            class_name = "".join(part.title() for part in module_name.split("_")) + "Analyzer"
            analyzer_cls = getattr(module, class_name)
            return analyzer_cls()
        except (ImportError, AttributeError):
            return None

    def _empty_report(self) -> Dict[str, Any]:
        return {
            "importance_results": {"combined_importance": {}, "analysis_report": {"summary": {}, "detail": {}}},
            "correlation_results": {"correlation_scores": {}, "analysis_report": {"summary": {}}},
            "stability_results": {"combined_stability": {}, "analysis_report": {"summary": {}}},
            "quality_scores": {},
            "feature_scores": {},
            "redundant_features": [],
            "overall_quality": 0.0,
            "summary": {},
            "recommendations": [],
            "comprehensive_report": {"summary": {}, "recommendations": []},
        }
