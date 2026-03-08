#!/usr/bin/env python3
"""轻量级特征工程实现，覆盖单测所需的核心行为。"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover
    from src.infrastructure.integration import get_models_adapter as _get_models_adapter
except ImportError:  # pragma: no cover
    class _FallbackAdapter:
        def get_models_logger(self):
            return logging.getLogger(__name__)

    def _get_models_adapter():
        return _FallbackAdapter()


def get_models_adapter():
    return _get_models_adapter()


logger = get_models_adapter().get_models_logger()


class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    TIME_SERIES = "time_series"


@dataclass
class FeatureDefinition:
    name: str
    feature_type: Optional[FeatureType]
    data_type: str
    description: str = ""
    nullable: bool = True


@dataclass
class FeaturePipeline:
    name: str
    steps: List[Dict[str, Any]]
    input_features: List[str]
    output_features: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"


class FeatureEngineer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_max_size = self.config.get("cache_max_size", 100)
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.pipelines: Dict[str, FeaturePipeline] = {}
        self._pipeline_cache: Dict[str, pd.DataFrame] = {}
        self.stats = {
            "pipelines_executed": 0,
            "cache_hits": 0,
        }

    # ------------------------------------------------------------------ #
    def define_feature(
        self,
        name: str,
        feature_type: Optional[FeatureType],
        data_type: str,
        **kwargs,
    ) -> FeatureDefinition:
        definition = FeatureDefinition(name, feature_type, data_type, **kwargs)
        self.feature_definitions[name] = definition
        return definition

    def create_pipeline(
        self, name: str, steps: List[Dict[str, Any]], input_features: List[str]
    ) -> FeaturePipeline:
        pipeline = FeaturePipeline(
            name=name,
            steps=steps,
            input_features=input_features,
            output_features=input_features.copy(),
        )
        self.pipelines[name] = pipeline
        return pipeline

    def get_pipeline_info(self, name: str) -> Optional[Dict[str, Any]]:
        pipeline = self.pipelines.get(name)
        if not pipeline:
            return None
        return {
            "name": pipeline.name,
            "steps": len(pipeline.steps),
            "step_types": [step.get("type") for step in pipeline.steps],
            "input_features": len(pipeline.input_features),
            "output_features": len(pipeline.output_features),
            "version": pipeline.version,
        }

    # ------------------------------------------------------------------ #
    def process_data(self, data: pd.DataFrame, pipeline_name: str) -> pd.DataFrame:
        if pipeline_name not in self.pipelines:
            raise ValueError(f"管道 {pipeline_name} 不存在")

        pipeline = self.pipelines[pipeline_name]

        if self.enable_caching:
            cache_key = self._get_cache_key(data, pipeline_name)
            cached = self._pipeline_cache.get(cache_key)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return cached.copy()

        result = data.copy()
        for step in pipeline.steps:
            result = self._apply_step(result, step)

        if self.enable_caching:
            if len(self._pipeline_cache) >= self.cache_max_size:
                oldest_key = next(iter(self._pipeline_cache))
                self._pipeline_cache.pop(oldest_key, None)
            self._pipeline_cache[cache_key] = result.copy()

        self.stats["pipelines_executed"] += 1
        return result

    # ------------------------------------------------------------------ #
    def _apply_step(self, data: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
        step_type = step.get("type")
        if step_type == "handle_missing":
            method = step.get("method", "fill")
            if method == "fill":
                fill_value = step.get("fill_value", 0)
                return data.fillna(fill_value)
            if method == "drop":
                return data.dropna()
        elif step_type == "custom_transformation":
            func_name = step.get("function")
            if func_name == "log":
                return np.log1p(data.select_dtypes(include=[np.number])).join(
                    data.select_dtypes(exclude=[np.number])
                )
        elif step_type == "create_temporal_features":
            return self._create_temporal_features(data)

        return data

    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in data:
            return data
        result = data.copy()
        ts = pd.to_datetime(result["timestamp"])
        result["hour"] = ts.dt.hour
        result["day_of_week"] = ts.dt.dayofweek
        result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
        result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
        return result

    def _get_cache_key(self, data: pd.DataFrame, pipeline_name: str) -> str:
        payload = json.dumps(data.to_dict("list"), sort_keys=True, default=str)
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return f"{pipeline_name}:{digest}"


__all__ = [
    "FeatureEngineer",
    "FeatureType",
    "FeatureDefinition",
    "FeaturePipeline",
    "get_models_adapter",
]

