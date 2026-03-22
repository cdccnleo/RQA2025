#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征验证模块

提供统一的特征验证接口，解耦验证逻辑与业务逻辑。
支持配置验证、数据验证、参数验证等多种验证类型。
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from .exceptions import (
    FeatureDataValidationError,
    FeatureConfigValidationError,
    FeatureProcessingError
)

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证级别枚举"""
    ERROR = "error"       # 验证失败抛出异常
    WARNING = "warning"   # 验证失败记录警告
    INFO = "info"         # 验证失败仅记录信息


class ValidationResult:
    """验证结果类"""

    def __init__(self, is_valid: bool = True, message: str = "",
                 level: ValidationLevel = ValidationLevel.ERROR,
                 details: Optional[Dict[str, Any]] = None):
        self.is_valid = is_valid
        self.message = message
        self.level = level
        self.details = details or {}
        self.timestamp = pd.Timestamp.now()

    def __bool__(self):
        return self.is_valid

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_valid": self.is_valid,
            "message": self.message,
            "level": self.level.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ValidationRule:
    """验证规则定义"""
    name: str
    validator: Callable[[Any], ValidationResult]
    level: ValidationLevel = ValidationLevel.ERROR
    message: str = ""
    condition: Optional[Callable[[], bool]] = None  # 条件函数，返回True时执行验证


class BaseValidator(ABC):
    """验证器基类"""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._rules: List[ValidationRule] = []
        self._validation_history: List[Dict[str, Any]] = []

    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """执行验证"""

    def add_rule(self, rule: ValidationRule) -> 'BaseValidator':
        """添加验证规则"""
        self._rules.append(rule)
        return self

    def remove_rule(self, rule_name: str) -> bool:
        """移除验证规则"""
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                self._rules.pop(i)
                return True
        return False

    def _execute_rules(self, data: Any) -> List[ValidationResult]:
        """执行所有验证规则"""
        results = []
        for rule in self._rules:
            # 检查条件
            if rule.condition and not rule.condition():
                continue

            try:
                result = rule.validator(data)
                result.level = rule.level
                results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    is_valid=False,
                    message=f"验证规则 '{rule.name}' 执行失败: {str(e)}",
                    level=ValidationLevel.ERROR
                ))
        return results

    def _record_validation(self, data_type: str, result: ValidationResult):
        """记录验证历史"""
        self._validation_history.append({
            "validator": self.name,
            "data_type": data_type,
            "result": result.to_dict(),
            "timestamp": pd.Timestamp.now().isoformat()
        })

    def get_validation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取验证历史"""
        return self._validation_history[-limit:]

    def clear_history(self):
        """清除验证历史"""
        self._validation_history.clear()


class ConfigValidator(BaseValidator):
    """配置验证器"""

    def __init__(self):
        super().__init__("ConfigValidator")
        self._setup_default_rules()

    def _setup_default_rules(self):
        """设置默认验证规则"""
        # 验证配置不为空
        self.add_rule(ValidationRule(
            name="not_empty",
            validator=lambda x: ValidationResult(
                is_valid=bool(x),
                message="配置不能为空"
            ),
            level=ValidationLevel.ERROR
        ))

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """验证配置"""
        results = self._execute_rules(config)

        # 合并结果
        errors = [r for r in results if not r.is_valid and r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if not r.is_valid and r.level == ValidationLevel.WARNING]

        if errors:
            result = ValidationResult(
                is_valid=False,
                message=f"配置验证失败: {'; '.join(e.message for e in errors)}",
                level=ValidationLevel.ERROR,
                details={"errors": [e.to_dict() for e in errors]}
            )
        elif warnings:
            result = ValidationResult(
                is_valid=True,
                message=f"配置验证通过，但有警告: {'; '.join(w.message for w in warnings)}",
                level=ValidationLevel.WARNING,
                details={"warnings": [w.to_dict() for w in warnings]}
            )
        else:
            result = ValidationResult(is_valid=True)

        self._record_validation("config", result)
        return result


class DataValidator(BaseValidator):
    """数据验证器"""

    def __init__(self, required_columns: Optional[List[str]] = None,
                 min_rows: int = 1, min_cols: int = 1):
        super().__init__("DataValidator")
        self.required_columns = required_columns or []
        self.min_rows = min_rows
        self.min_cols = min_cols
        self._setup_default_rules()

    def _setup_default_rules(self):
        """设置默认验证规则"""
        # 验证数据不为空
        self.add_rule(ValidationRule(
            name="not_empty",
            validator=self._validate_not_empty,
            level=ValidationLevel.ERROR
        ))

        # 验证数据类型
        self.add_rule(ValidationRule(
            name="is_dataframe",
            validator=self._validate_is_dataframe,
            level=ValidationLevel.ERROR
        ))

        # 验证最小行数
        self.add_rule(ValidationRule(
            name="min_rows",
            validator=self._validate_min_rows,
            level=ValidationLevel.ERROR
        ))

        # 验证最小列数
        self.add_rule(ValidationRule(
            name="min_cols",
            validator=self._validate_min_cols,
            level=ValidationLevel.ERROR
        ))

        # 验证必需列
        if self.required_columns:
            self.add_rule(ValidationRule(
                name="required_columns",
                validator=self._validate_required_columns,
                level=ValidationLevel.ERROR
            ))

    def _validate_not_empty(self, data: Any) -> ValidationResult:
        """验证数据不为空"""
        is_valid = data is not None
        if isinstance(data, pd.DataFrame):
            is_valid = is_valid and not data.empty
        return ValidationResult(
            is_valid=is_valid,
            message="数据不能为空" if not is_valid else ""
        )

    def _validate_is_dataframe(self, data: Any) -> ValidationResult:
        """验证数据是DataFrame"""
        is_valid = isinstance(data, pd.DataFrame)
        return ValidationResult(
            is_valid=is_valid,
            message="数据必须是pandas DataFrame" if not is_valid else ""
        )

    def _validate_min_rows(self, data: pd.DataFrame) -> ValidationResult:
        """验证最小行数"""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=True)
        is_valid = len(data) >= self.min_rows
        return ValidationResult(
            is_valid=is_valid,
            message=f"数据行数必须大于等于 {self.min_rows}，当前: {len(data)}" if not is_valid else ""
        )

    def _validate_min_cols(self, data: pd.DataFrame) -> ValidationResult:
        """验证最小列数"""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=True)
        is_valid = len(data.columns) >= self.min_cols
        return ValidationResult(
            is_valid=is_valid,
            message=f"数据列数必须大于等于 {self.min_cols}，当前: {len(data.columns)}" if not is_valid else ""
        )

    def _validate_required_columns(self, data: pd.DataFrame) -> ValidationResult:
        """验证必需列"""
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(is_valid=True)
        missing = [col for col in self.required_columns if col not in data.columns]
        is_valid = len(missing) == 0
        return ValidationResult(
            is_valid=is_valid,
            message=f"缺少必需列: {missing}" if not is_valid else "",
            details={"missing_columns": missing}
        )

    def validate(self, data: Any) -> ValidationResult:
        """验证数据"""
        results = self._execute_rules(data)

        # 合并结果
        errors = [r for r in results if not r.is_valid and r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if not r.is_valid and r.level == ValidationLevel.WARNING]

        if errors:
            result = ValidationResult(
                is_valid=False,
                message=f"数据验证失败: {'; '.join(e.message for e in errors)}",
                level=ValidationLevel.ERROR,
                details={
                    "errors": [e.to_dict() for e in errors],
                    "data_shape": data.shape if isinstance(data, pd.DataFrame) else None
                }
            )
        elif warnings:
            result = ValidationResult(
                is_valid=True,
                message=f"数据验证通过，但有警告: {'; '.join(w.message for w in warnings)}",
                level=ValidationLevel.WARNING,
                details={"warnings": [w.to_dict() for w in warnings]}
            )
        else:
            result = ValidationResult(is_valid=True)

        self._record_validation("data", result)
        return result


class FeatureParamsValidator(BaseValidator):
    """特征参数验证器"""

    def __init__(self, param_specs: Optional[Dict[str, Dict[str, Any]]] = None):
        super().__init__("FeatureParamsValidator")
        self.param_specs = param_specs or {}
        self._setup_default_rules()

    def _setup_default_rules(self):
        """设置默认验证规则"""
        self.add_rule(ValidationRule(
            name="valid_params",
            validator=self._validate_params,
            level=ValidationLevel.ERROR
        ))

    def _validate_params(self, params: Dict[str, Any]) -> ValidationResult:
        """验证参数"""
        errors = []

        for param_name, spec in self.param_specs.items():
            if spec.get("required", False) and param_name not in params:
                errors.append(f"缺少必需参数: {param_name}")
                continue

            if param_name in params:
                value = params[param_name]

                # 类型验证
                if "type" in spec:
                    expected_type = spec["type"]
                    if not isinstance(value, expected_type):
                        errors.append(f"参数 {param_name} 类型错误，期望: {expected_type.__name__}")

                # 范围验证
                if "min" in spec and value < spec["min"]:
                    errors.append(f"参数 {param_name} 小于最小值 {spec['min']}")
                if "max" in spec and value > spec["max"]:
                    errors.append(f"参数 {param_name} 大于最大值 {spec['max']}")

                # 枚举验证
                if "choices" in spec and value not in spec["choices"]:
                    errors.append(f"参数 {param_name} 必须是以下之一: {spec['choices']}")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            message="; ".join(errors) if errors else "",
            details={"errors": errors}
        )

    def validate(self, params: Dict[str, Any]) -> ValidationResult:
        """验证参数"""
        results = self._execute_rules(params)

        errors = [r for r in results if not r.is_valid and r.level == ValidationLevel.ERROR]

        if errors:
            result = ValidationResult(
                is_valid=False,
                message=f"参数验证失败: {'; '.join(e.message for e in errors)}",
                level=ValidationLevel.ERROR,
                details={"errors": [e.to_dict() for e in errors]}
            )
        else:
            result = ValidationResult(is_valid=True)

        self._record_validation("params", result)
        return result


class ValidationPipeline:
    """验证管道"""

    def __init__(self, name: str = ""):
        self.name = name or "ValidationPipeline"
        self._validators: List[Tuple[BaseValidator, str]] = []  # (validator, data_key)
        self._stop_on_error = True

    def add_validator(self, validator: BaseValidator, data_key: str = "") -> 'ValidationPipeline':
        """添加验证器"""
        self._validators.append((validator, data_key))
        return self

    def set_stop_on_error(self, stop: bool) -> 'ValidationPipeline':
        """设置是否在错误时停止"""
        self._stop_on_error = stop
        return self

    def validate(self, data: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """执行验证管道"""
        results = {}

        for validator, data_key in self._validators:
            # 获取要验证的数据
            if data_key:
                validation_data = data.get(data_key)
            else:
                validation_data = data

            # 执行验证
            result = validator.validate(validation_data)
            results[validator.name] = result

            # 错误时停止
            if self._stop_on_error and not result.is_valid and result.level == ValidationLevel.ERROR:
                logger.error(f"验证管道在 {validator.name} 处停止: {result.message}")
                break

        return results

    def validate_all(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, ValidationResult]]:
        """验证所有，返回整体结果和详细结果"""
        results = self.validate(data)
        is_valid = all(r.is_valid for r in results.values())
        return is_valid, results


# 全局验证器实例
_config_validator = None
_data_validator = None


def get_config_validator() -> ConfigValidator:
    """获取全局配置验证器"""
    global _config_validator
    if _config_validator is None:
        _config_validator = ConfigValidator()
    return _config_validator


def get_data_validator(required_columns: Optional[List[str]] = None,
                       min_rows: int = 1, min_cols: int = 1) -> DataValidator:
    """获取数据验证器"""
    return DataValidator(required_columns, min_rows, min_cols)


def create_validation_pipeline(name: str = "") -> ValidationPipeline:
    """创建验证管道"""
    return ValidationPipeline(name)


# 便捷函数

def validate_config(config: Dict[str, Any], raise_on_error: bool = False) -> ValidationResult:
    """便捷函数：验证配置"""
    validator = get_config_validator()
    result = validator.validate(config)

    if not result.is_valid and raise_on_error:
        raise FeatureConfigValidationError(
            message=result.message,
            config_dict=config
        )

    return result


def validate_data(data: Any, required_columns: Optional[List[str]] = None,
                  raise_on_error: bool = False) -> ValidationResult:
    """便捷函数：验证数据"""
    validator = get_data_validator(required_columns)
    result = validator.validate(data)

    if not result.is_valid and raise_on_error:
        missing_cols = result.details.get("missing_columns", [])
        raise FeatureDataValidationError(
            message=result.message,
            missing_columns=missing_cols,
            data_shape=data.shape if isinstance(data, pd.DataFrame) else None
        )

    return result


def validate_params(params: Dict[str, Any],
                    param_specs: Dict[str, Dict[str, Any]],
                    raise_on_error: bool = False) -> ValidationResult:
    """便捷函数：验证参数"""
    validator = FeatureParamsValidator(param_specs)
    result = validator.validate(params)

    if not result.is_valid and raise_on_error:
        raise FeatureProcessingError(
            message=result.message,
            step="parameter_validation"
        )

    return result
