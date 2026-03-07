"""
validator_base 模块

提供 validator_base 相关功能和接口。
"""

import re

# ==================== 枚举定义 ====================
import datetime

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable
#!/usr/bin/env python3
"""
配置验证器基础组件

包含验证器相关的枚举、数据类、接口和基类定义
拆分自validators.py，提高代码组织性和可维护性
"""


class ValidationSeverity(Enum):
    """验证严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def __lt__(self, other):
        """定义比较顺序"""
        order = {'info': 0, 'warning': 1, 'error': 2, 'critical': 3}
        return order[self.value] < order[other.value]


class ValidationType(Enum):
    """验证类型"""
    REQUIRED = "required"
    TYPE = "type"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"
    DEPENDENCY = "dependency"

# ==================== 数据类定义 ====================


class ValidationResult(dict):
    """验证结果"""

    def __init__(self,
                 is_valid: bool = True,
                 errors: Optional[List[str]] = None,
                 warnings: Optional[List[str]] = None,
                 severity: ValidationSeverity = ValidationSeverity.INFO,
                 field: str = "",
                 value: Any = None,
                 suggestions: Optional[List[str]] = None,
                 successes: Optional[List[str]] = None,  # 成功消息列表
                 success: Optional[bool] = None,  # 向后兼容性参数
                 message: Optional[str] = None):  # 向后兼容性参数
        """初始化验证结果

        Args:
            is_valid: 验证是否有效
            errors: 错误列表
            warnings: 警告列表
            severity: 验证严重程度
            field: 验证字段
            value: 验证值
            suggestions: 修复建议
            success: 向后兼容性参数 (等同于is_valid)
            message: 向后兼容性参数 (错误消息)
        """
        super().__init__()

        # 处理向后兼容性参数
        if success is not None:
            is_valid = success
        if message is not None and not errors:
            if is_valid:
                successes = (successes or []) + [message]
            else:
                errors = [message]

        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.successes = successes or []  # 成功消息列表
        self.severity = severity
        self.field = field
        self.value = value
        self.suggestions = suggestions or []
        self.timestamp = datetime.datetime.now()

        # 向后兼容性属性
        self.success = self.is_valid
        self.valid = self.is_valid  # 添加valid属性作为is_valid的别名
        if self.errors:
            self.message = str(self.errors[0])
        elif self.successes:
            self.message = str(self.successes[0])
        else:
            self.message = ""
        self._sync_mapping()

    def __bool__(self) -> bool:
        """布尔值转换"""
        return self.is_valid

    def __str__(self) -> str:
        """字符串表示"""
        status_symbol = "✓" if self.is_valid else "✗"
        status_text = "OK" if self.is_valid else "ERROR"
        errors_str = ", ".join(self.errors) if self.errors else ""
        warnings_str = ", ".join(self.warnings) if self.warnings else ""
        message = f"errors: {errors_str}" if errors_str else ""
        if warnings_str:
            message += f" warnings: {warnings_str}" if message else f"warnings: {warnings_str}"

        field_text = self.field or ""
        if message:
            field_segment = f" {field_text}" if field_text else ""
            return f"{status_symbol}{field_segment}: {message}".strip()

        if field_text:
            return f"{status_symbol} {field_text} ({status_text})"

        return f"{status_symbol} {status_text}"

    def add_error(self, message: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        """添加错误"""
        self.errors.append(message)
        self.is_valid = False
        if severity > self.severity:
            self.severity = severity
        self.success = self.is_valid
        self.valid = self.is_valid
        self.message = str(self.errors[0]) if self.errors else ""
        self._sync_mapping()

    def add_warning(self, message: str):
        """添加警告"""
        self.warnings.append(message)
        self._sync_mapping()

    def add_success(self, message: str):
        """添加成功消息"""
        self.successes.append(message)
        self._sync_mapping()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        self._sync_mapping()
        return dict(self)

    def merge(self, other: 'ValidationResult'):
        """合并另一个验证结果"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
        if other.severity > self.severity:
            self.severity = other.severity
        self.success = self.is_valid
        self.valid = self.is_valid
        if self.errors:
            self.message = str(self.errors[0])
        self._sync_mapping()

    # 内部方法：保持字典视图与属性同步
    def _sync_mapping(self):
        dict.clear(self)
        dict.update(self, {
            'is_valid': self.is_valid,
            'success': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'successes': self.successes,
            'severity': self.severity.value if isinstance(self.severity, ValidationSeverity) else self.severity,
            'field': self.field,
            'value': self.value,
            'suggestions': self.suggestions,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'message': self.message,
            'timestamp': self.timestamp.isoformat()
        })


class ValidationRule:
    """验证规则"""

    def __init__(
        self,
        rule_type_or_name,
        field_or_check=None,
        required: bool = True,
        validator: Optional[Callable] = None,
        **kwargs,
    ):
        """初始化验证规则，兼容简化与高级两种使用方式"""

        # 简化模式：ValidationRule("rule_name", lambda value: ...)
        if not isinstance(rule_type_or_name, ValidationType):
            name = kwargs.get('name') or (str(rule_type_or_name) if rule_type_or_name is not None else "")
            check_func = field_or_check if callable(field_or_check) else None
            if validator and callable(validator):
                check_func = validator
            if check_func is None:
                check_func = kwargs.get('check_func')
            if check_func is None:
                raise ValueError("ValidationRule requires a callable check function in simplified mode")

            self.rule_type = kwargs.get('rule_type', ValidationType.CUSTOM)
            self.field = kwargs.get('field', name)
            self.required = kwargs.get('required', required)
            self.params = {k: v for k, v in kwargs.items() if k not in {'name', 'field', 'rule_type', 'required', 'check_func'}}
            self.name = name
            self.check_func = check_func
            self.validator = self._wrap_check_callable(check_func, kwargs.get('failure_message'))
            return

        # 高级模式：ValidationRule(ValidationType.TYPE, "field", required=True, validator=...)
        self.rule_type = rule_type_or_name
        self.field = field_or_check if isinstance(field_or_check, str) else kwargs.get('field', '')
        self.required = required
        self.validator = validator
        self.params = kwargs

        # 添加测试需要的属性
        self.check_func = validator  # check_func是validator的别名
        self.name = kwargs.get('name', self.field)

    def validate(self, value: Any) -> ValidationResult:
        """执行验证

        Args:
            value: 要验证的值

        Returns:
            验证结果
        """
        try:
            if self.required and value is None:
                return ValidationResult(is_valid=False, errors=[f"字段 '{self.field}' 为必需字段"],
                                        severity=ValidationSeverity.ERROR,
                                        field=self.field,
                                        value=value
                                        )

            if self.validator:
                return self.validator(value, **self.params)

            # 根据规则类型执行内置验证
            return self._validate_by_type(value)

        except Exception as e:
            return ValidationResult(is_valid=False, errors=[f"验证过程异常: {str(e)}"],
                                    severity=ValidationSeverity.ERROR,
                                    field=self.field,
                                    value=value
                                    )

    def _validate_by_type(self, value: Any) -> ValidationResult:
        """根据类型执行验证"""
        if self.rule_type == ValidationType.TYPE:
            expected_type = self.params.get('type')
            if expected_type:
                # 支持Union类型（元组）或单个类型
                if isinstance(expected_type, tuple):
                    # Union类型检查
                    if not any(isinstance(value, t) for t in expected_type):
                        type_names = [t.__name__ for t in expected_type]
                        return ValidationResult(is_valid=False, errors=[f"字段 '{self.field}' 类型错误，期望 {type_names} 之一，实际 {type(value).__name__}"],
                                                severity=ValidationSeverity.ERROR,
                                                field=self.field,
                                                value=value
                                                )
                elif not isinstance(value, expected_type):
                    return ValidationResult(is_valid=False, errors=[f"字段 '{self.field}' 类型错误，期望 {expected_type.__name__}，实际 {type(value).__name__}"],
                                            severity=ValidationSeverity.ERROR,
                                            field=self.field,
                                            value=value
                                            )

        elif self.rule_type == ValidationType.RANGE:
            min_val = self.params.get('min')
            max_val = self.params.get('max')

            if isinstance(value, (int, float)):
                if min_val is not None and value < min_val:
                    return ValidationResult(is_valid=False, errors=[f"字段 '{self.field}' 值 {value} 小于最小值 {min_val}"],
                                            severity=ValidationSeverity.ERROR,
                                            field=self.field,
                                            value=value
                                            )
                if max_val is not None and value > max_val:
                    return ValidationResult(is_valid=False, errors=[f"字段 '{self.field}' 值 {value} 大于最大值 {max_val}"],
                                            severity=ValidationSeverity.ERROR,
                                            field=self.field,
                                            value=value
                                            )

        elif self.rule_type == ValidationType.PATTERN:
            pattern = self.params.get('pattern')
            if pattern and isinstance(value, str):
                if not re.match(pattern, value):
                    return ValidationResult(is_valid=False, errors=[f"字段 '{self.field}' 值 '{value}' 不匹配模式 '{pattern}'"],
                                            severity=ValidationSeverity.ERROR,
                                            field=self.field,
                                            value=value
                                            )

        result = ValidationResult(is_valid=True,
                                  field=self.field,
                                  value=value
                                  )
        result.add_success(f"字段 '{self.field}' 验证通过")
        return result

    def _wrap_check_callable(self, func: Callable, failure_message: Optional[str] = None) -> Callable:
        """将简单检查函数包装为返回ValidationResult的验证器"""

        default_failure = failure_message or "验证未通过"

        def _wrapped(value: Any, **_kwargs) -> ValidationResult:
            try:
                outcome = func(value)
                if isinstance(outcome, ValidationResult):
                    return outcome
                if outcome:
                    result = ValidationResult(is_valid=True, field=self.field or self.name, value=value)
                    result.add_success(f"规则 '{self.name}' 验证通过")
                    return result
                return ValidationResult(
                    is_valid=False,
                    errors=[f"规则 '{self.name}' 验证失败: {default_failure}"],
                    severity=ValidationSeverity.ERROR,
                    field=self.field or self.name,
                    value=value,
                )
            except Exception as exc:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"验证过程异常: {exc}"],
                    severity=ValidationSeverity.ERROR,
                    field=self.field or self.name,
                    value=value,
                )

        return _wrapped

# ==================== 接口定义 ====================


class IConfigValidator(ABC):
    """配置验证器接口"""

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """验证配置

        Args:
            config: 要验证的配置字典

        Returns:
            验证结果
        """

    @abstractmethod
    def validate_field(self, field: str, value: Any) -> ValidationResult:
        """验证单个字段

        Args:
            field: 字段名
            value: 字段值

        Returns:
            验证结果
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """验证器名称"""

    @property
    @abstractmethod
    def description(self) -> str:
        """验证器描述"""

# ==================== 基类定义 ====================


class BaseConfigValidator(IConfigValidator):
    """配置验证器基类

    提供通用的验证框架和辅助方法
    """

    def __init__(self, name: str = "", description: str = ""):
        """初始化验证器

        Args:
            name: 验证器名称
            description: 验证器描述
        """
        self._name = name or self.__class__.__name__
        self._description = description or f"{self._name} 验证器"
        self._rules: List[ValidationRule] = []

    @property
    def name(self) -> str:
        """验证器名称"""
        return self._name

    @property
    def description(self) -> str:
        """验证器描述"""
        return self._description

    def add_rule(self, rule: ValidationRule):
        """添加验证规则"""
        self._rules.append(rule)

    def clear_rules(self):
        """清空验证规则"""
        self._rules.clear()

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """验证配置

        Args:
            config: 要验证的配置字典

        Returns:
            验证结果
        """
        # 执行规则验证
        rule_results = self._validate_rules(config)

        # 执行自定义验证
        custom_results = self._validate_custom(config)

        # 检查是否有严重的自定义验证错误
        critical_custom_result = self._check_critical_custom_validation(custom_results)
        if critical_custom_result:
            return critical_custom_result

        # 合并所有验证结果
        all_results = rule_results + custom_results
        return self._merge_validation_results(all_results)

    def _validate_rules(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """验证所有规则"""
        results = []

        for rule in self._rules:
            result = self._validate_single_rule(config, rule)
            results.append(result)

        return results

    def _validate_single_rule(self, config: Dict[str, Any], rule) -> ValidationResult:
        """验证单个规则"""
        # 导航到嵌套字段
        field_value, found = self._navigate_nested_field(config, rule.field)

        if found and field_value is not None:
            # 字段存在，执行规则验证
            return rule.validate(field_value)
        elif rule.required:
            # 必需字段缺失
            return ValidationResult(
                is_valid=False,
                errors=[f"必需字段 '{rule.field}' 缺失"],
                severity=ValidationSeverity.ERROR,
                field=rule.field
            )
        else:
            # 非必需字段缺失，添加警告
            return ValidationResult(
                is_valid=True,
                warnings=[f"建议配置字段 '{rule.field}'"],
                severity=ValidationSeverity.WARNING,
                field=rule.field
            )

    def _navigate_nested_field(self, config: Dict[str, Any], field_path: str) -> Tuple[Any, bool]:
        """导航到嵌套字段并返回值

        Returns:
            (field_value, found): 字段值和是否找到的标志
        """
        field_parts = field_path.split('.')
        current_config = config

        # 遍历嵌套结构
        for part in field_parts[:-1]:
            if isinstance(current_config, dict) and part in current_config:
                current_config = current_config[part]
            else:
                return None, False

        # 获取最终字段值
        field_name = field_parts[-1]
        if isinstance(current_config, dict) and field_name in current_config:
            return current_config[field_name], True

        return None, False

    def _check_critical_custom_validation(self, custom_results: List[ValidationResult]) -> Optional[ValidationResult]:
        """检查是否有严重的自定义验证错误"""
        if not custom_results:
            return None

        all_errors = []
        all_warnings = []
        is_valid = True
        max_severity = ValidationSeverity.INFO

        for result in custom_results:
            if isinstance(result, ValidationResult):
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
                if not result.is_valid:
                    is_valid = False
                if result.severity > max_severity:
                    max_severity = result.severity
            elif isinstance(result, list):
                for sub_result in result:
                    if isinstance(sub_result, ValidationResult):
                        all_errors.extend(sub_result.errors)
                        all_warnings.extend(sub_result.warnings)
                        if not sub_result.is_valid:
                            is_valid = False
                        if sub_result.severity > max_severity:
                            max_severity = sub_result.severity

        if all_errors or not is_valid:
            return ValidationResult(
                is_valid=is_valid,
                errors=all_errors,
                warnings=all_warnings,
                severity=max_severity
            )

        return None

    def _merge_validation_results(self, results: List[ValidationResult]) -> ValidationResult:
        """合并验证结果"""
        if not results:
            return ValidationResult(is_valid=True)

        combined_result = ValidationResult(is_valid=True)

        for result in results:
            if isinstance(result, ValidationResult):
                combined_result.errors.extend(result.errors)
                combined_result.warnings.extend(result.warnings)
                if not result.is_valid:
                    combined_result.is_valid = False
                if result.severity > combined_result.severity:
                    combined_result.severity = result.severity
            elif isinstance(result, list):
                # 处理返回列表的情况
                for sub_result in result:
                    if isinstance(sub_result, ValidationResult):
                        combined_result.errors.extend(sub_result.errors)
                        combined_result.warnings.extend(sub_result.warnings)
                        if not sub_result.is_valid:
                            combined_result.is_valid = False
                        if sub_result.severity > combined_result.severity:
                            combined_result.severity = sub_result.severity

        return combined_result

    def validate_field(self, field: str, value: Any) -> ValidationResult:
        """验证单个字段"""
        for rule in self._rules:
            if rule.field == field:
                return rule.validate(value)

        # 如果没有找到对应规则，执行自定义验证
        return self._validate_field_custom(field, value)

    def _validate_custom(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """自定义验证逻辑

        子类可以重写此方法添加额外的验证逻辑

        Args:
            config: 配置字典

        Returns:
            验证结果列表
        """
        return []

    def _validate_field_custom(self, field: str, value: Any) -> ValidationResult:
        """自定义字段验证逻辑

        子类可以重写此方法添加字段验证逻辑

        Args:
            field: 字段名
            value: 字段值

        Returns:
            验证结果
        """
        return ValidationResult(is_valid=True,
                                successes=[f"字段 '{field}' 验证通过"],
                                field=field,
                                value=value
                                )

    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """获取验证结果摘要

        Args:
            results: 验证结果列表

        Returns:
            摘要信息字典
        """
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed

        # 按严重程度统计
        severity_count = {}
        for severity in ValidationSeverity:
            severity_count[severity.value] = sum(1 for r in results if r.severity == severity)

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': passed / total if total > 0 else 0,
            'severity_breakdown': severity_count,
            'errors': [r.message for r in results if not r.success]
        }




