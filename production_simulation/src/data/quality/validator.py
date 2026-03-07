"""
数据质量验证模块

提供数据质量验证、指标计算以及自定义规则扩展。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class ValidationResult:
    """数据质量验证结果."""

    is_valid: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data_type: str = "unknown"
    checks: Dict[str, Any] = field(default_factory=dict)


class ValidationError(Exception):
    """数据质量验证异常."""

    def __init__(self, message: str, result: Optional[ValidationResult] = None):
        super().__init__(message)
        self.result = result


class DataValidator:
    """数据质量验证器."""

    DEFAULT_NUMERIC_COLUMNS = {"price", "volume", "amount", "quantity", "open", "close"}
    DEFAULT_ALLOWED_RANGES: Dict[str, Tuple[float, float]] = {
        "price": (0.0, 1e5),
        "volume": (0.0, 1e9),
        "amount": (0.0, 1e12),
        "quantity": (0.0, 1e8),
    }
    SUPPORTED_TYPE_CHECKS: Dict[str, Callable[[Any], bool]] = {
        "string": lambda value: isinstance(value, str),
        "integer": lambda value: isinstance(value, int),
        "float": lambda value: isinstance(value, (int, float)),
        "number": lambda value: isinstance(value, (int, float)),
        "boolean": lambda value: isinstance(value, bool),
        "dict": lambda value: isinstance(value, dict),
        "list": lambda value: isinstance(value, list),
    }

    def __init__(self, validator_type: str = "basic", config: Optional[Dict[str, Any]] = None):
        self.validator_type = validator_type
        self.config = config or {}
        self.strict_mode = bool(self.config.get("strict_mode", False))
        self.rules: List[Callable[[Any], Sequence[str]]] = list(self.config.get("custom_rules", []))
        self.validation_history: List[ValidationResult] = []

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def validate(self, data: Any, data_type: str = "generic") -> Dict[str, Any]:
        """执行数据质量验证."""

        start = time.perf_counter()
        metrics: Dict[str, float] = {}
        errors: List[str] = []
        checks: Dict[str, Any] = {}

        if isinstance(data, pd.DataFrame):
            metrics.update(self._calculate_completeness_metrics(data))
            errors.extend(self._validate_missing_values(data))
            errors.extend(self._validate_numeric_types(data))
            errors.extend(self._validate_value_ranges(data))

            duplicate_issues = self._validate_duplicates(data)
            if duplicate_issues:
                errors.append(duplicate_issues)

            checks["row_count"] = int(data.shape[0])
            checks["column_count"] = int(data.shape[1])
        elif isinstance(data, dict):
            errors.extend(self._validate_dict_payload(data))
            metrics["field_count"] = float(len(data))
        else:
            errors.append(f"unsupported data type: {type(data).__name__}")

        # 执行自定义规则
        for rule in self.rules:
            try:
                rule_errors = rule(data)
                if rule_errors:
                    errors.extend(str(err) for err in rule_errors)
            except Exception as exc:  # pragma: no cover - 防御性
                errors.append(f"custom rule execution failed: {exc}")

        metrics.setdefault("error_count", float(len(errors)))
        result = ValidationResult(
            is_valid=not errors,
            metrics=metrics,
            errors=errors,
            data_type=data_type,
            checks=checks,
        )

        # 记录历史
        self.validation_history.append(result)

        duration = time.perf_counter() - start
        response = {
            "is_valid": result.is_valid,
            "errors": result.errors,
            "metrics": result.metrics,
            "timestamp": result.timestamp,
            "data_type": data_type,
            "checks": checks,
            "duration_seconds": duration,
        }
        return response

    def add_rule(self, rule: Callable[[Any], Sequence[str]]) -> None:
        """添加自定义验证规则."""

        if not callable(rule):
            raise TypeError("rule must be callable")
        self.rules.append(rule)

    def clear_rules(self) -> None:
        """清空自定义规则."""

        self.rules.clear()

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _calculate_completeness_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        total_cells = int(data.shape[0] * data.shape[1]) or 1
        missing_cells = int(data.isna().sum().sum())
        completeness = 1.0 - missing_cells / total_cells
        completeness = max(0.0, min(1.0, completeness))

        return {
            "total_cells": float(total_cells),
            "missing_cells": float(missing_cells),
            "missing_rate": missing_cells / total_cells,
            "completeness": completeness,
        }

    def _validate_missing_values(self, data: pd.DataFrame) -> List[str]:
        issues: List[str] = []
        missing_mask = data.isna()
        if missing_mask.any().any():
            columns_with_missing = missing_mask.any()[missing_mask.any()].index.tolist()
            issues.append(
                f"missing values detected in columns: {', '.join(columns_with_missing)}"
            )
        return issues

    def _validate_numeric_types(self, data: pd.DataFrame) -> List[str]:
        issues: List[str] = []

        for column in data.columns:
            series = data[column]
            if self._is_probably_numeric_column(column, series):
                coerced = pd.to_numeric(series, errors="coerce")
                if coerced.isna().sum() > series.isna().sum():
                    issues.append(f"type mismatch detected in column '{column}'")
        return issues

    def _validate_value_ranges(self, data: pd.DataFrame) -> List[str]:
        issues: List[str] = []
        for column, (lower, upper) in self.DEFAULT_ALLOWED_RANGES.items():
            if column in data.columns:
                series = pd.to_numeric(data[column], errors="coerce")
                if series.notna().any():
                    if (series < lower).any() or (series > upper).any():
                        issues.append(
                            f"value out of expected range in column '{column}' "
                            f"(allowed {lower}~{upper})"
                        )
        return issues

    def _validate_duplicates(self, data: pd.DataFrame) -> Optional[str]:
        if data.duplicated().any():
            return "duplicate rows detected in dataset"
        return None

    def _validate_dict_payload(self, payload: Dict[str, Any]) -> List[str]:
        issues: List[str] = []
        if not payload:
            issues.append("missing data payload")
            return issues

        for key, value in payload.items():
            if value is None:
                issues.append(f"missing value for field '{key}'")

        return issues

    def _is_probably_numeric_column(self, column: str, series: pd.Series) -> bool:
        if column.lower() in self.DEFAULT_NUMERIC_COLUMNS:
            return True

        sample = series.dropna().head(5)
        if sample.empty:
            return False

        numeric_count = sum(self._looks_like_number(value) for value in sample)
        return numeric_count / len(sample) >= 0.6

    @staticmethod
    def _looks_like_number(value: Any) -> bool:
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                return False
            else:
                return True
        return False


__all__ = ["DataValidator", "ValidationResult", "ValidationError"]


