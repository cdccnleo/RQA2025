"""
Data Quality Checks Automation Module
数据质量检查自动化模块

This module provides automated data quality checking capabilities for quantitative trading
此模块为量化交易提供自动化数据质量检查能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
from collections import deque

logger = logging.getLogger(__name__)


class QualityCheckType(Enum):

    """Data quality check types"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


class QualityCheckStatus(Enum):

    """Quality check status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class QualityCheckResult:

    """
    Quality check result data class
    质量检查结果数据类
    """
    check_id: str
    check_type: str
    status: str
    timestamp: datetime
    total_records: int = 0
    passed_records: int = 0
    failed_records: int = 0
    warning_records: int = 0
    score: float = 0.0
    details: Dict[str, Any] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class QualityCheckRule:

    """
    Quality Check Rule Class
    质量检查规则类

    Represents a single data quality check rule
    表示单个数据质量检查规则
    """

    def __init__(self,


                 rule_id: str,
                 name: str,
                 check_type: QualityCheckType,
                 config: Dict[str, Any],
                 severity: str = "error"):
        """
        Initialize quality check rule
        初始化质量检查规则

        Args:
            rule_id: Unique rule identifier
                    唯一规则标识符
            name: Human - readable rule name
                 人类可读的规则名称
            check_type: Type of quality check
                       质量检查类型
            config: Rule configuration
                   规则配置
            severity: Rule severity ('error', 'warning')
                     规则严重程度
        """
        self.rule_id = rule_id
        self.name = name
        self.check_type = check_type
        self.config = config
        self.severity = severity

        # Runtime statistics
        self.execution_count = 0
        self.pass_count = 0
        self.fail_count = 0
        self.last_executed: Optional[datetime] = None

    def execute(self, data: pd.DataFrame) -> QualityCheckResult:
        """
        Execute the quality check rule
        执行质量检查规则

        Args:
            data: Data to check
                 要检查的数据

        Returns:
            QualityCheckResult: Check result
                               检查结果
        """
        self.execution_count += 1
        self.last_executed = datetime.now()

        result = QualityCheckResult(
            check_id=self.rule_id,
            check_type=self.check_type.value,
            status=QualityCheckStatus.RUNNING.value,
            timestamp=self.last_executed,
            total_records=len(data)
        )

        try:
            if self.check_type == QualityCheckType.COMPLETENESS:
                result = self._check_completeness(data, result)
            elif self.check_type == QualityCheckType.ACCURACY:
                result = self._check_accuracy(data, result)
            elif self.check_type == QualityCheckType.CONSISTENCY:
                result = self._check_consistency(data, result)
            elif self.check_type == QualityCheckType.TIMELINESS:
                result = self._check_timeliness(data, result)
            elif self.check_type == QualityCheckType.VALIDITY:
                result = self._check_validity(data, result)
            elif self.check_type == QualityCheckType.UNIQUENESS:
                result = self._check_uniqueness(data, result)

            # Calculate score
            result.score = result.passed_records / max(result.total_records, 1) * 100

            # Determine final status
            if result.failed_records > 0:
                if self.severity == "error":
                    result.status = QualityCheckStatus.FAILED.value
                    self.fail_count += 1
                else:
                    result.status = QualityCheckStatus.WARNING.value
            else:
                result.status = QualityCheckStatus.PASSED.value
                self.pass_count += 1

        except Exception as e:
            result.status = QualityCheckStatus.FAILED.value
            result.error_message = str(e)
            self.fail_count += 1
            logger.error(f"Quality check rule {self.rule_id} failed: {str(e)}")

        return result

    def _check_completeness(self, data: pd.DataFrame, result: QualityCheckResult) -> QualityCheckResult:
        """
        Check data completeness
        检查数据完整性

        Args:
            data: Data to check
                 要检查的数据
            result: Result object to update
                   要更新的结果对象

        Returns:
            QualityCheckResult: Updated result
                               更新的结果
        """
        fields_to_check = self.config.get('fields', data.columns.tolist())
        null_threshold = self.config.get('null_threshold', 0.05)  # 5% max nulls

        result.details = {
            'fields_checked': [],
            'null_counts': {},
            'null_percentages': {}
        }

        for field in fields_to_check:
            if field in data.columns:
                null_count = data[field].isnull().sum()
                null_percentage = null_count / len(data) * 100

                result.details['fields_checked'].append(field)
                result.details['null_counts'][field] = int(null_count)
                result.details['null_percentages'][field] = round(null_percentage, 2)

                if null_percentage <= null_threshold:
                    result.passed_records += len(data)
                else:
                    result.failed_records += len(data)

        return result

    def _check_accuracy(self, data: pd.DataFrame, result: QualityCheckResult) -> QualityCheckResult:
        """
        Check data accuracy
        检查数据准确性

        Args:
            data: Data to check
                 要检查的数据
            result: Result object to update
                   要更新的结果对象

        Returns:
            QualityCheckResult: Updated result
                               更新的结果
        """
        accuracy_rules = self.config.get('rules', [])

        result.details = {
            'rules_checked': [],
            'violations': {}
        }

        total_violations = 0

        for rule in accuracy_rules:
            rule_type = rule.get('type', '')
            field = rule.get('field', '')

            if rule_type == 'range':
                min_val = rule.get('min')
                max_val = rule.get('max')
                violations = data[~data[field].between(min_val, max_val, inclusive='both')]
                violation_count = len(violations)

            elif rule_type == 'pattern':
                pattern = rule.get('pattern')
                violations = data[~data[field].astype(str).str.match(pattern)]
                violation_count = len(violations)

            elif rule_type == 'reference':
                ref_field = rule.get('reference_field')
                violations = data[data[field] != data[ref_field]]
                violation_count = len(violations)

            else:
                continue

            result.details['rules_checked'].append(f"{rule_type}_{field}")
            result.details['violations'][f"{rule_type}_{field}"] = violation_count
            total_violations += violation_count

        result.passed_records = len(data) - total_violations
        result.failed_records = total_violations

        return result

    def _check_consistency(self, data: pd.DataFrame, result: QualityCheckResult) -> QualityCheckResult:
        """
        Check data consistency
        检查数据一致性

        Args:
            data: Data to check
                 要检查的数据
            result: Result object to update
                   要更新的结果对象

        Returns:
            QualityCheckResult: Updated result
                               更新的结果
        """
        consistency_rules = self.config.get('rules', [])

        result.details = {
            'rules_checked': [],
            'inconsistencies': {}
        }

        total_inconsistencies = 0

        for rule in consistency_rules:
            rule_type = rule.get('type', '')

            if rule_type == 'cross_field':
                field1 = rule.get('field1')
                field2 = rule.get('field2')
                operator = rule.get('operator', '==')

                if operator == '==':
                    inconsistencies = data[data[field1] != data[field2]]
                elif operator == '>=':
                    inconsistencies = data[data[field1] < data[field2]]

                inconsistency_count = len(inconsistencies)

            elif rule_type == 'business_rule':
                # Custom business rule check
                rule_func = rule.get('function')
                if rule_func:
                    inconsistencies = data[~data.apply(rule_func, axis=1)]
                    inconsistency_count = len(inconsistencies)
                else:
                    inconsistency_count = 0

            else:
                continue

            rule_name = f"{rule_type}_{rule.get('name', 'unknown')}"
            result.details['rules_checked'].append(rule_name)
            result.details['inconsistencies'][rule_name] = inconsistency_count
            total_inconsistencies += inconsistency_count

        result.passed_records = len(data) - total_inconsistencies
        result.failed_records = total_inconsistencies

        return result

    def _check_timeliness(self, data: pd.DataFrame, result: QualityCheckResult) -> QualityCheckResult:
        """
        Check data timeliness
        检查数据及时性

        Args:
            data: Data to check
                 要检查的数据
            result: Result object to update
                   要更新的结果对象

        Returns:
            QualityCheckResult: Updated result
                               更新的结果
        """
        timestamp_field = self.config.get('timestamp_field', 'timestamp')
        max_age_hours = self.config.get('max_age_hours', 24)
        current_time = datetime.now()

        result.details = {
            'timestamp_field': timestamp_field,
            'max_age_hours': max_age_hours,
            'stale_records': 0,
            'oldest_record': None,
            'newest_record': None
        }

        if timestamp_field in data.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[timestamp_field]):
                data[timestamp_field] = pd.to_datetime(data[timestamp_field])

            # Check timeliness
            max_age = timedelta(hours=max_age_hours)
            stale_records = data[data[timestamp_field] < (current_time - max_age)]

            result.details['stale_records'] = len(stale_records)
            result.details['oldest_record'] = data[timestamp_field].min(
            ).isoformat() if len(data) > 0 else None
            result.details['newest_record'] = data[timestamp_field].max(
            ).isoformat() if len(data) > 0 else None

            result.passed_records = len(data) - len(stale_records)
            result.failed_records = len(stale_records)
        else:
            result.failed_records = len(data)
            result.details['error'] = f"Timestamp field '{timestamp_field}' not found"

        return result

    def _check_validity(self, data: pd.DataFrame, result: QualityCheckResult) -> QualityCheckResult:
        """
        Check data validity
        检查数据有效性

        Args:
            data: Data to check
                 要检查的数据
            result: Result object to update
                   要更新的结果对象

        Returns:
            QualityCheckResult: Updated result
                               更新的结果
        """
        validity_rules = self.config.get('rules', [])

        result.details = {
            'rules_checked': [],
            'invalid_records': {}
        }

        total_invalid = 0

        for rule in validity_rules:
            rule_type = rule.get('type', '')
            field = rule.get('field', '')

            if rule_type == 'data_type':
                expected_type = rule.get('expected_type')
                if expected_type == 'numeric':
                    invalid = data[~pd.api.types.is_numeric_dtype(
                        data[field]) & ~data[field].isnull()]
                elif expected_type == 'string':
                    invalid = data[~data[field].astype(str).str.match(
                        r'^[a - zA - Z\s]+$') & ~data[field].isnull()]
                else:
                    invalid = data[pd.isna(data[field])]

            elif rule_type == 'allowed_values':
                allowed_values = rule.get('values', [])
                invalid = data[~data[field].isin(allowed_values) & ~data[field].isnull()]

            else:
                continue

            invalid_count = len(invalid)
            result.details['rules_checked'].append(f"{rule_type}_{field}")
            result.details['invalid_records'][f"{rule_type}_{field}"] = invalid_count
            total_invalid += invalid_count

        result.passed_records = len(data) - total_invalid
        result.failed_records = total_invalid

        return result

    def _check_uniqueness(self, data: pd.DataFrame, result: QualityCheckResult) -> QualityCheckResult:
        """
        Check data uniqueness
        检查数据唯一性

        Args:
            data: Data to check
                 要检查的数据
            result: Result object to update
                   要更新的结果对象

        Returns:
            QualityCheckResult: Updated result
                               更新的结果
        """
        uniqueness_fields = self.config.get('fields', [])

        result.details = {
            'fields_checked': uniqueness_fields,
            'duplicate_counts': {},
            'total_duplicates': 0
        }

        total_duplicates = 0

        if uniqueness_fields:
            # Check for duplicates based on specified fields
            duplicate_mask = data.duplicated(subset=uniqueness_fields, keep=False)
            duplicates = data[duplicate_mask]

            if len(duplicates) > 0:
                # Count duplicates per group
                duplicate_groups = duplicates.groupby(uniqueness_fields).size()
                result.details['duplicate_counts'] = duplicate_groups.to_dict()
                total_duplicates = len(duplicates)

        result.details['total_duplicates'] = total_duplicates
        result.passed_records = len(data) - total_duplicates
        result.failed_records = total_duplicates

        return result

    def get_rule_stats(self) -> Dict[str, Any]:
        """
        Get rule execution statistics
        获取规则执行统计信息

        Returns:
            dict: Rule statistics
                  规则统计信息
        """
        total_executions = self.pass_count + self.fail_count
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'check_type': self.check_type.value,
            'severity': self.severity,
            'total_executions': total_executions,
            'pass_count': self.pass_count,
            'fail_count': self.fail_count,
            'pass_rate': self.pass_count / max(total_executions, 1) * 100,
            'last_executed': self.last_executed.isoformat() if self.last_executed else None
        }


class DataQualityChecker:

    """
    Data Quality Checker Class
    数据质量检查器类

    Orchestrates multiple quality check rules and provides comprehensive quality assessment
    协调多个质量检查规则并提供全面的质量评估
    """

    def __init__(self, checker_name: str = "default_data_quality_checker"):
        """
        Initialize data quality checker
        初始化数据质量检查器

        Args:
            checker_name: Name of the quality checker
                        质量检查器名称
        """
        self.checker_name = checker_name
        self.rules: Dict[str, QualityCheckRule] = {}

        # Statistics
        self.check_history: deque = deque(maxlen=1000)
        self.stats = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'warning_checks': 0,
            'average_score': 0.0
        }

        logger.info(f"Data quality checker {checker_name} initialized")

    def add_rule(self, rule: QualityCheckRule) -> None:
        """
        Add a quality check rule
        添加质量检查规则

        Args:
            rule: Quality check rule to add
                 要添加的质量检查规则
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"Added quality check rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a quality check rule
        移除质量检查规则

        Args:
            rule_id: Rule identifier
                    规则标识符

        Returns:
            bool: True if removed successfully, False otherwise
                  移除成功返回True，否则返回False
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed quality check rule: {rule_id}")
            return True
        return False

    def run_quality_checks(self,


                           data: pd.DataFrame,
                           rule_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run quality checks on data
        对数据运行质量检查

        Args:
            data: Data to check
                 要检查的数据
            rule_ids: Specific rule IDs to run (optional)
                     要运行的特定规则ID（可选）

        Returns:
            dict: Quality check results
                  质量检查结果
        """
        rules_to_run = []
        if rule_ids:
            rules_to_run = [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
        else:
            rules_to_run = list(self.rules.values())

        if not rules_to_run:
            return {'error': 'No rules to execute'}

        check_results = []
        overall_score = 0.0
        total_passed = 0
        total_failed = 0
        total_warnings = 0

        start_time = datetime.now()

        for rule in rules_to_run:
            try:
                result = rule.execute(data)
                check_results.append(result.to_dict())

                if result.status == QualityCheckStatus.PASSED.value:
                    total_passed += 1
                elif result.status == QualityCheckStatus.FAILED.value:
                    total_failed += 1
                elif result.status == QualityCheckStatus.WARNING.value:
                    total_warnings += 1

                overall_score += result.score

            except Exception as e:
                logger.error(f"Failed to execute rule {rule.rule_id}: {str(e)}")
                error_result = QualityCheckResult(
                    check_id=rule.rule_id,
                    check_type=rule.check_type.value,
                    status=QualityCheckStatus.FAILED.value,
                    timestamp=datetime.now(),
                    error_message=str(e)
                )
                check_results.append(error_result.to_dict())
                total_failed += 1

        # Calculate overall metrics
        total_checks = len(check_results)
        overall_score = overall_score / max(total_checks, 1)

        quality_assessment = {
            'overall_score': round(overall_score, 2),
            'quality_level': self._assess_quality_level(overall_score),
            'total_checks': total_checks,
            'passed_checks': total_passed,
            'failed_checks': total_failed,
            'warning_checks': total_warnings,
            'pass_rate': round(total_passed / max(total_checks, 1) * 100, 2),
            'execution_time': (datetime.now() - start_time).total_seconds(),
            'check_results': check_results
        }

        # Update statistics
        self._update_stats(quality_assessment)

        # Store in history
        self.check_history.append({
            'timestamp': datetime.now(),
            'results': quality_assessment
        })

        return quality_assessment

    def _assess_quality_level(self, score: float) -> str:
        """
        Assess overall quality level based on score
        根据分数评估整体质量水平

        Args:
            score: Quality score (0 - 100)
                  质量分数（0 - 100）

        Returns:
            str: Quality level
                 质量水平
        """
        if score >= 95:
            return "Excellent"
        elif score >= 85:
            return "Good"
        elif score >= 75:
            return "Acceptable"
        elif score >= 60:
            return "Poor"
        else:
            return "Critical"

    def _update_stats(self, assessment: Dict[str, Any]) -> None:
        """
        Update checker statistics
        更新检查器统计信息

        Args:
            assessment: Quality assessment results
                       质量评估结果
        """
        self.stats['total_checks'] += 1

        if assessment['failed_checks'] == 0 and assessment['warning_checks'] == 0:
            self.stats['passed_checks'] += 1
        elif assessment['failed_checks'] > 0:
            self.stats['failed_checks'] += 1
        else:
            self.stats['warning_checks'] += 1

        # Update average score
        total_checks = self.stats['total_checks']
        current_avg = self.stats['average_score']
        new_score = assessment['overall_score']
        self.stats['average_score'] = (
            (current_avg * (total_checks - 1)) + new_score
        ) / total_checks

    def get_checker_stats(self) -> Dict[str, Any]:
        """
        Get checker statistics
        获取检查器统计信息

        Returns:
            dict: Checker statistics
                  检查器统计信息
        """
        return {
            'checker_name': self.checker_name,
            'total_rules': len(self.rules),
            'stats': self.stats,
            'rules_summary': {
                rule_id: rule.get_rule_stats()
                for rule_id, rule in self.rules.items()
            }
        }

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent quality check history
        获取最近的质量检查历史

        Args:
            limit: Maximum number of records to return
                  返回的最大记录数

        Returns:
            list: Recent check history
                  最近检查历史
        """
        return list(self.check_history)[-limit:]

    def create_completeness_rule(self,


                                 rule_id: str,
                                 name: str,
                                 fields: List[str],
                                 null_threshold: float = 0.05) -> str:
        """
        Create a completeness check rule
        创建完整性检查规则

        Args:
            rule_id: Rule identifier
                    规则标识符
            name: Rule name
                 规则名称
            fields: Fields to check for completeness
                   要检查完整性的字段
            null_threshold: Maximum allowed null percentage
                           允许的最大空值百分比

        Returns:
            str: Created rule ID
                 创建的规则ID
        """
        config = {
            'fields': fields,
            'null_threshold': null_threshold
        }

        rule = QualityCheckRule(
            rule_id=rule_id,
            name=name,
            check_type=QualityCheckType.COMPLETENESS,
            config=config
        )

        self.add_rule(rule)
        return rule_id

    def create_accuracy_rule(self,


                             rule_id: str,
                             name: str,
                             rules: List[Dict[str, Any]]) -> str:
        """
        Create an accuracy check rule
        创建准确性检查规则

        Args:
            rule_id: Rule identifier
                    规则标识符
            name: Rule name
                 规则名称
            rules: List of accuracy validation rules
                  准确性验证规则列表

        Returns:
            str: Created rule ID
                 创建的规则ID
        """
        config = {'rules': rules}

        rule = QualityCheckRule(
            rule_id=rule_id,
            name=name,
            check_type=QualityCheckType.ACCURACY,
            config=config
        )

        self.add_rule(rule)
        return rule_id

    def create_timeliness_rule(self,


                               rule_id: str,
                               name: str,
                               timestamp_field: str,
                               max_age_hours: int = 24) -> str:
        """
        Create a timeliness check rule
        创建及时性检查规则

        Args:
            rule_id: Rule identifier
                    规则标识符
            name: Rule name
                 规则名称
            timestamp_field: Field containing timestamps
                           包含时间戳的字段
            max_age_hours: Maximum allowed age in hours
                          允许的最大时长（小时）

        Returns:
            str: Created rule ID
                 创建的规则ID
        """
        config = {
            'timestamp_field': timestamp_field,
            'max_age_hours': max_age_hours
        }

        rule = QualityCheckRule(
            rule_id=rule_id,
            name=name,
            check_type=QualityCheckType.TIMELINESS,
            config=config
        )

        self.add_rule(rule)
        return rule_id


# Global data quality checker instance
# 全局数据质量检查器实例
data_quality_checker = DataQualityChecker()

__all__ = [
    'QualityCheckType',
    'QualityCheckStatus',
    'QualityCheckResult',
    'QualityCheckRule',
    'DataQualityChecker',
    'data_quality_checker'
]
