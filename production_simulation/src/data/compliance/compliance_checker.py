"""
合规校验器

负责根据策略校验数据的字段完整性、类型、范围以及交易合规性。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .data_policy_manager import DataPolicyManager


SUPPORTED_TYPE_CHECKS: Dict[str, Tuple[str, ...]] = {
    "string": ("str",),
    "integer": ("int",),
    "float": ("int", "float"),
    "number": ("int", "float"),
    "boolean": ("bool",),
    "dict": ("dict",),
    "list": ("list", "tuple"),
}


@dataclass
class ComplianceIssue:
    field: str
    message: str

    def as_text(self) -> str:
        return f"{self.field}: {self.message}"


@dataclass
class ComplianceResult:
    compliance: bool
    issues: List[str] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())
    check_duration_seconds: float = 0.0
    check_type: str = "standard"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compliance": self.compliance,
            "issues": self.issues,
            "checked_at": self.checked_at,
            "check_duration_seconds": self.check_duration_seconds,
            "check_type": self.check_type,
        }


class ComplianceChecker:
    """数据合规校验器."""

    def __init__(self, policy_manager: DataPolicyManager):
        self.policy_manager = policy_manager

    def check(self, data: Any, policy_id: Optional[str] = None) -> Dict[str, Any]:
        """校验单条数据."""

        start = time.perf_counter()
        issues: List[ComplianceIssue] = []

        if policy_id:
            policy = self.policy_manager.get_policy(policy_id)
            if not policy:
                issues.append(ComplianceIssue("policy", f"Policy '{policy_id}' not found"))
            else:
                issues.extend(self._validate_required_fields(data, policy))
                issues.extend(self._validate_field_types(data, policy))
                issues.extend(self._validate_max_lengths(data, policy))
                issues.extend(self._validate_value_ranges(data, policy))
                issues.extend(self._validate_enum_values(data, policy))
        else:
            issues.append(ComplianceIssue("policy", "Policy id is required for compliance check"))

        result = ComplianceResult(
            compliance=not issues,
            issues=[issue.as_text() for issue in issues],
            check_duration_seconds=time.perf_counter() - start,
            check_type="policy" if policy_id else "standard",
        )
        return result.to_dict()

    def check_bulk_data(self, data_list: Iterable[Dict[str, Any]], policy_id: Optional[str]) -> Dict[str, Any]:
        """批量数据合规校验."""

        start = time.perf_counter()
        total = 0
        compliant = 0
        all_issues: List[str] = []

        for item in data_list:
            total += 1
            result = self.check(item, policy_id)
            if result["compliance"]:
                compliant += 1
            else:
                all_issues.extend(result["issues"])

        compliance_rate = compliant / total if total else 0.0

        return {
            "total_records": total,
            "compliant_records": compliant,
            "non_compliant_records": total - compliant,
            "compliance_rate": compliance_rate,
            "all_issues": all_issues,
            "checked_at": datetime.now().isoformat(),
            "check_duration_seconds": time.perf_counter() - start,
        }

    def check_trading_compliance(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """交易数据合规校验."""

        start = time.perf_counter()
        issues: List[ComplianceIssue] = []

        required = ["amount", "trade_type", "timestamp"]
        for field in required:
            if field not in trade_data:
                issues.append(ComplianceIssue(field, "missing required field"))

        amount = trade_data.get("amount")
        if amount is not None and (not isinstance(amount, (int, float)) or amount <= 0):
            issues.append(ComplianceIssue("amount", "must be a positive number"))

        trade_type = trade_data.get("trade_type")
        allowed_types = {"buy", "sell", "short", "cover"}
        if trade_type is not None and trade_type not in allowed_types:
            issues.append(ComplianceIssue("trade_type", f"invalid trade type '{trade_type}'"))

        timestamp = trade_data.get("timestamp")
        if timestamp:
            try:
                datetime.fromisoformat(str(timestamp))
            except ValueError:
                issues.append(ComplianceIssue("timestamp", "invalid ISO timestamp"))

        result = ComplianceResult(
            compliance=not issues,
            issues=[issue.as_text() for issue in issues],
            check_type="trading",
            check_duration_seconds=time.perf_counter() - start,
        )
        return result.to_dict()

    def _validate_required_fields(self, data: Any, policy: Dict[str, Any]) -> List[ComplianceIssue]:
        issues: List[ComplianceIssue] = []
        required_fields = policy.get("required_fields", [])
        for field in required_fields:
            if not self._has_field(data, field):
                issues.append(ComplianceIssue(field, "缺失字段"))
        return issues

    def _validate_field_types(self, data: Any, policy: Dict[str, Any]) -> List[ComplianceIssue]:
        issues: List[ComplianceIssue] = []
        field_types = policy.get("field_types", {})

        for field, expected in field_types.items():
            if not self._has_field(data, field):
                continue

            value = self._get_field_value(data, field)
            if value is None:
                issues.append(ComplianceIssue(field, "字段值为空"))
                continue

            allowed_python_types = SUPPORTED_TYPE_CHECKS.get(expected.lower())
            if not allowed_python_types:
                continue

            if type(value).__name__ not in allowed_python_types:
                issues.append(
                    ComplianceIssue(field, f"类型错误，期望 {expected}, 实际为 {type(value).__name__}")
                )
        return issues

    def _validate_max_lengths(self, data: Any, policy: Dict[str, Any]) -> List[ComplianceIssue]:
        issues: List[ComplianceIssue] = []
        max_lengths = policy.get("max_field_lengths", {})

        for field, max_length in max_lengths.items():
            if not self._has_field(data, field):
                continue

            value = self._get_field_value(data, field)
            if isinstance(value, str) and len(value) > max_length:
                issues.append(ComplianceIssue(field, f"超出限制，最大长度 {max_length}"))
        return issues

    def _validate_value_ranges(self, data: Any, policy: Dict[str, Any]) -> List[ComplianceIssue]:
        issues: List[ComplianceIssue] = []
        ranges = policy.get("business_rules", {}).get("value_ranges", {})

        for field, rules in ranges.items():
            if not self._has_field(data, field):
                continue

            value = self._get_field_value(data, field)
            if not isinstance(value, (int, float)):
                continue

            minimum = rules.get("min")
            maximum = rules.get("max")

            if minimum is not None and value < minimum:
                issues.append(ComplianceIssue(field, f"值低于最小值 {minimum}"))
            if maximum is not None and value > maximum:
                issues.append(ComplianceIssue(field, f"值超过最大值 {maximum}"))
        return issues

    def _validate_enum_values(self, data: Any, policy: Dict[str, Any]) -> List[ComplianceIssue]:
        issues: List[ComplianceIssue] = []
        enum_rules = policy.get("business_rules", {}).get("enum_values", {})

        for field, allowed_values in enum_rules.items():
            if not self._has_field(data, field):
                continue

            value = self._get_field_value(data, field)
            if value not in allowed_values:
                issues.append(
                    ComplianceIssue(field, f"取值不在允许列表 {allowed_values}")
                )
        return issues

    @staticmethod
    def _has_field(data: Any, field: str) -> bool:
        if isinstance(data, dict):
            return field in data
        if isinstance(data, pd.DataFrame):
            return field in data.columns
        return hasattr(data, field)

    @staticmethod
    def _get_field_value(data: Any, field: str) -> Any:
        if isinstance(data, dict):
            return data.get(field)
        if isinstance(data, pd.DataFrame):
            if field not in data.columns:
                return None
            column = data[field]
            value = column.dropna()
            return value.iloc[0] if not value.empty else None
        return getattr(data, field, None)


__all__ = ["ComplianceChecker", "ComplianceResult", "ComplianceIssue"]


