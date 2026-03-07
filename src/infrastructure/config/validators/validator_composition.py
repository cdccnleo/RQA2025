"""
validator_composition 模块

提供 validator_composition 相关功能和接口。
"""

import logging

from typing import Dict, Any, List, Optional, Callable
from .validator_base import (
    IConfigValidator, ValidationRule, ValidationType, ValidationResult, ValidationSeverity, BaseConfigValidator
)

"""
配置验证器组合和工厂

包含验证器的组合、工厂模式和通用验证器实现
拆分自validators.py，提高代码组织性和可维护性
"""

logger = logging.getLogger(__name__)

# ==================== 验证器组合 ====================


class ConfigValidators(IConfigValidator):
    """配置验证器组合

    将多个验证器组合在一起，按顺序执行验证
    """

    def __init__(self, validators: Optional[List[IConfigValidator]] = None, name: str = "CompositeValidator"):
        """初始化验证器组合

        Args:
            validators: 验证器列表
            name: 组合验证器名称
        """
        self._name = name
        self._validators: List[IConfigValidator] = validators or []
        self._fail_fast = False  # 是否在第一次失败时停止

    @property
    def name(self) -> str:
        """验证器名称"""
        return self._name

    @property
    def description(self) -> str:
        """验证器描述"""
        validator_names = [v.name for v in self._validators]
        return f"组合验证器，包含: {', '.join(validator_names)}"

    @property
    def validators(self) -> List[IConfigValidator]:
        """验证器列表"""
        return self._validators

    def add_validator(self, validator: IConfigValidator):
        """添加验证器

        Args:
            validator: 要添加的验证器
        """
        if validator not in self._validators:
            self._validators.append(validator)
            logger.debug(f"添加验证器: {validator.name}")

    def remove_validator(self, validator: IConfigValidator):
        """移除验证器

        Args:
            validator: 要移除的验证器
        """
        if validator in self._validators:
            self._validators.remove(validator)
            logger.debug(f"移除验证器: {validator.name}")

    def clear_validators(self):
        """清空所有验证器"""
        self._validators.clear()
        logger.debug("清空所有验证器")

    def set_fail_fast(self, fail_fast: bool = True):
        """设置快速失败模式

        Args:
            fail_fast: 是否在第一次验证失败时停止后续验证
        """
        self._fail_fast = fail_fast

    def validate(self, config: Dict[str, Any]) -> tuple:
        """验证配置

        Args:
            config: 要验证的配置字典

        Returns:
            (is_valid, results) 元组，其中is_valid表示整体验证是否通过
        """
        all_results = []

        for validator in self._validators:
            try:
                logger.debug(f"执行验证器: {validator.name}")
                result = validator.validate(config)
                # 单个验证器返回单个ValidationResult对象
                all_results.append(result)

                # 检查是否需要快速失败
                if self._fail_fast:
                    if not result.is_valid:
                        logger.debug(f"验证器 {validator.name} 失败，停止后续验证")
                        break

            except Exception as e:
                # 验证器执行异常
                error_result = ValidationResult(is_valid=False, errors=[f"验证器 {validator.name} 执行异常: {str(e)}"],
                                                severity=ValidationSeverity.ERROR,
                                                field="validator_execution"
                                                )
                all_results.append(error_result)
                logger.error(f"验证器 {validator.name} 执行失败: {e}")

                if self._fail_fast:
                    break

        # 计算整体验证结果
        is_valid = all(r.is_valid for r in all_results)

        # 合并所有验证结果
        combined_errors = []
        combined_warnings = []
        combined_suggestions = []

        for r in all_results:
            if not r.is_valid:
                combined_errors.extend(r.errors)
            combined_warnings.extend(r.warnings)
            combined_suggestions.extend(r.suggestions)

        aggregated_result = ValidationResult(
            is_valid=is_valid,
            errors=combined_errors,
            warnings=combined_warnings,
            severity=ValidationSeverity.ERROR if not is_valid else ValidationSeverity.INFO,
            field="combined_validation",
            value=config,
            suggestions=combined_suggestions,
        )

        if aggregated_result.is_valid and not aggregated_result.errors and not aggregated_result.warnings and not aggregated_result.suggestions:
            return True, None

        return aggregated_result.is_valid, aggregated_result

    def validate_field(self, field: str, value: Any) -> ValidationResult:
        """验证单个字段

        按顺序使用各个验证器进行字段验证，返回第一个非成功的验证结果
        如果所有验证器都通过，返回成功的验证结果

        Args:
            field: 字段名
            value: 字段值

        Returns:
            验证结果
        """
        for validator in self._validators:
            try:
                result = validator.validate_field(field, value)
                if not result.success:
                    return result
            except Exception as e:
                return ValidationResult(is_valid=False, errors=[f"验证器 {validator.name} 执行异常: {str(e)}"],
                                        severity=ValidationSeverity.ERROR,
                                        field=field,
                                        value=value
                                        )

        return ValidationResult(is_valid=True,
                                message=f"字段 '{field}' 通过所有验证器",
                                field=field,
                                value=value
                                )

    def get_validator_info(self) -> List[Dict[str, Any]]:
        """获取验证器信息"""
        return [
            {
                'name': validator.name,
                'description': validator.description,
                'type': validator.__class__.__name__
            }
            for validator in self._validators
        ]

    def __len__(self) -> int:
        """返回验证器数量"""
        return len(self._validators)

    def __iter__(self):
        """迭代验证器"""
        return iter(self._validators)

    def __contains__(self, validator: IConfigValidator) -> bool:
        """检查是否包含指定的验证器"""
        return validator in self._validators

    def validate_type(self, value: Any, expected_type: type) -> bool:
        """验证类型（兼容性方法）

        Args:
            value: 要验证的值
            expected_type: 期望的类型

        Returns:
            bool: 类型是否匹配
        """
        return isinstance(value, expected_type)

    def validate_range(self, value: Any, min_val: Optional[Any] = None,
                       max_val: Optional[Any] = None) -> bool:
        """验证范围（兼容性方法）

        Args:
            value: 要验证的值
            min_val: 最小值
            max_val: 最大值

        Returns:
            bool: 是否在范围内
        """
        try:
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True
        except TypeError:
            return False

# ==================== 验证器工厂 ====================


class UnifiedValidatorFactory:
    """统一验证器工厂

    提供创建各种验证器的工厂方法
    """

    def __init__(self):
        """初始化工厂"""
        self._validator_types = {}
        self._validator_classes = self._validator_types  # 向后兼容性
        self._register_builtin_validators()

    def _register_builtin_validators(self):
        """注册内置验证器"""
        from .specialized_validators import (
            TradingHoursValidator,
            DatabaseConfigValidator,
            LoggingConfigValidator,
            NetworkConfigValidator
        )

        self.register_validator_type('trading_hours', TradingHoursValidator)
        self.register_validator_type('database', DatabaseConfigValidator)
        self.register_validator_type('logging', LoggingConfigValidator)
        self.register_validator_type('network', NetworkConfigValidator)

    def register_validator_type(self, validator_type: str, validator_class: type):
        """注册验证器类型

        Args:
            validator_type: 验证器类型名称
            validator_class: 验证器类
        """
        self._validator_types[validator_type] = validator_class
        logger.debug(f"注册验证器类型: {validator_type} -> {validator_class.__name__}")

    def create_validator(self, validator_type: str, *args, **kwargs) -> Optional[IConfigValidator]:
        """创建验证器实例

        Args:
            validator_type: 验证器类型
            *args, **kwargs: 传递给验证器构造函数的参数

        Returns:
            验证器实例

        Raises:
            ValueError: 如果验证器类型不存在
        """
        validator_class = self._validator_types.get(validator_type)
        if validator_class is None:
            logger.warning(f"未知的验证器类型: {validator_type}")
            raise ValueError(f"未知的验证器类型: {validator_type}")

        try:
            validator = validator_class(*args, **kwargs)
            logger.debug(f"创建验证器: {validator_type} -> {validator.name}")
            return validator
        except Exception as e:
            logger.error(f"创建验证器失败 {validator_type}: {e}")
            return None

    def create_composite_validator(self, validator_specs, name: str = "CompositeValidator") -> ConfigValidators:
        """创建组合验证器

        Args:
            validator_specs: 验证器规格列表，可以是字符串列表或字典列表
            name: 组合验证器名称

        Returns:
            组合验证器实例
        """
        composite = ConfigValidators(name=name)

        for spec in validator_specs:
            if isinstance(spec, str):
                # 如果是字符串，直接作为验证器类型
                validator_type = spec
                init_kwargs = {}
            elif isinstance(spec, dict):
                # 如果是字典，提取type和其他参数
                validator_type = spec.get('type')
                init_kwargs = {k: v for k, v in spec.items() if k != 'type'}
            else:
                logger.warning(f"无效的验证器规格: {spec}")
                continue

            if validator_type:
                try:
                    validator = self.create_validator(validator_type, **init_kwargs)

                    if validator:
                        composite.add_validator(validator)
                    else:
                        logger.warning(f"无法创建验证器: {spec}")
                except ValueError as e:
                    # 忽略未知验证器类型，记录警告日志
                    logger.warning(f"跳过未知验证器类型: {validator_type}")
                    continue
                except Exception as e:
                    logger.warning(f"创建验证器失败: {spec}, 错误: {e}")
                    continue

        logger.debug(f"创建组合验证器: {name}，包含 {len(composite)} 个验证器")
        return composite

    def register_validator(self, name: str, validator_class):
        """注册验证器类型"""
        # 检查验证器类是否实现了IConfigValidator接口
        if not issubclass(validator_class, IConfigValidator):
            raise ValueError("必须实现IConfigValidator接口")

        self._validator_types[name] = validator_class
        logger.debug(f"注册验证器类型: {name}")

    def get_available_validators(self) -> List[str]:
        """获取可用验证器列表"""
        return list(self._validator_types.keys())

    def create_validator_suite(self, suite_spec) -> ConfigValidators:
        """创建验证器套件"""
        if isinstance(suite_spec, dict):
            name = suite_spec.get('name', 'ValidatorSuite')
            validators_spec = suite_spec.get('validators', [])
        elif isinstance(suite_spec, list):
            # 如果传入的是列表，直接使用
            name = 'ValidatorSuite'
            validators_spec = suite_spec
        else:
            raise ValueError("suite_spec must be dict or list")

        return self.create_composite_validator(validators_spec, name)

    def get_available_types(self) -> List[str]:
        """获取可用的验证器类型"""
        return list(self._validator_types.keys())

    def get_validator_info(self, validator_type: str) -> Optional[Dict[str, Any]]:
        """获取验证器类型信息

        Args:
            validator_type: 验证器类型

        Returns:
            验证器信息字典，包含名称、描述等
        """
        validator_class = self._validator_types.get(validator_type)
        if validator_class is None:
            return None

        try:
            # 创建临时实例获取信息
            temp_validator = validator_class()
            return {
                'type': validator_type,
                'class': validator_class.__name__,
                'name': temp_validator.name,
                'description': temp_validator.description
            }
        except Exception as e:
            logger.warning(f"获取验证器信息失败 {validator_type}: {e}")
            return {
                'type': validator_type,
                'class': validator_class.__name__,
                'error': str(e)
            }

# ==================== 通用配置验证器 ====================


class ConfigValidator(BaseConfigValidator):
    """通用配置验证器

    提供灵活的配置验证功能，支持规则配置和自定义验证
    """

    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None,
                 name: str = "ConfigValidator"):
        """初始化通用配置验证器

        Args:
            rules: 验证规则列表
            name: 验证器名称
        """
        super().__init__(name, "简单配置验证器")

        self._custom_validators: Dict[str, Callable] = {}

        # 添加默认规则
        if rules:
            self._load_rules(rules)

    def add_custom_validator(self, field: str, validator_func: Callable):
        """添加自定义验证器

        Args:
            field: 字段名
            validator_func: 验证函数，接受(value, **kwargs)参数，返回ValidationResult
        """
        self._custom_validators[field] = validator_func
        logger.debug(f"添加自定义验证器: {field}")

    def add_validation_rule(self, field: str, rule_func: Callable):
        """添加验证规则

        Args:
            field: 字段名
            rule_func: 验证函数
        """
        # 创建一个简单的验证规则
        rule = ValidationRule(
            ValidationType.CUSTOM,
            field,
            validator=rule_func
        )
        self._rules.append(rule)

    @property
    def rules(self):
        """验证规则字典（向后兼容性）"""
        rules_dict = {}
        for rule in self._rules:
            rules_dict[rule.field] = rule.validator if hasattr(
                rule, 'validator') and rule.validator else rule
        return rules_dict

    def validate(self, config: Dict[str, Any]) -> bool:
        """验证配置（简化版本，返回布尔值）

        Args:
            config: 配置字典

        Returns:
            是否验证通过
        """
        if not self._rules:
            # 没有规则时返回True
            return True

        # 首先调用父类的验证方法
        results = super().validate(config)
        if not results.is_valid:
            return False

        # 然后检查自定义验证规则
        for rule in self._rules:
            if hasattr(rule, 'validator') and rule.validator:
                try:
                    field_value = self._get_nested_value(config, rule.field)
                    if not rule.validator(field_value):
                        return False
                except KeyError:
                    # 如果字段不存在，根据规则的required属性决定
                    if getattr(rule, 'required', True):
                        return False

        return True

    def _get_nested_value(self, config: Dict[str, Any], field_path: str):
        """获取嵌套配置值

        Args:
            config: 配置字典
            field_path: 字段路径，如 'database.host'

        Returns:
            字段值

        Raises:
            KeyError: 如果键不存在
        """
        keys = field_path.split('.')
        current = config

        for key in keys:
            if isinstance(current, dict):
                current = current[key]
            else:
                raise KeyError(f"Cannot access '{key}' in non-dict value")
        return current

    def remove_custom_validator(self, field: str):
        """移除自定义验证器"""
        if field in self._custom_validators:
            del self._custom_validators[field]
            logger.debug(f"移除自定义验证器: {field}")

    def _load_rules(self, rules: List[Dict[str, Any]]):
        """加载验证规则

        Args:
            rules: 规则配置列表
        """
        for rule_config in rules:
            try:
                rule_type = ValidationType[rule_config['type'].upper()]
                field = rule_config['field']

                params = rule_config.get('params', {})
                rule = ValidationRule(
                    rule_type,
                    field,
                    required=rule_config.get('required', True),
                    validator=rule_config.get('validator'),
                    **params
                )

                self.add_rule(rule)
                logger.debug(f"加载验证规则: {field} ({rule_type.value})")

            except (KeyError, ValueError) as e:
                logger.warning(f"无效的规则配置 {rule_config}: {e}")

    def _validate_custom(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """执行自定义验证"""
        results = []

        # 执行自定义验证器
        for field, validator_func in self._custom_validators.items():
            if field in config:
                try:
                    result = validator_func(config[field])
                    if isinstance(result, ValidationResult):
                        results.append(result)
                    elif isinstance(result, list):
                        results.extend(result)
                except Exception as e:
                    results.append(ValidationResult(is_valid=False, errors=[f"自定义验证器执行失败: {str(e)}"],
                                                    severity=ValidationSeverity.ERROR,
                                                    field=field,
                                                    value=config[field]
                                                    ))

        return results

    def validate_with_context(self, config: Dict[str, Any],
                              context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """在上下文中验证配置

        Args:
            config: 配置字典
            context: 验证上下文

        Returns:
            验证结果列表
        """
        # 保存当前上下文（如果需要）
        if context:
            self._validation_context = context

        try:
            return self.validate(config)
        finally:
            # 清理上下文
            if hasattr(self, '_validation_context'):
                delattr(self, '_validation_context')

    def get_validation_stats(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """获取验证统计信息"""
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed

        # 按严重程度统计
        severity_stats = {}
        for severity in ValidationSeverity:
            severity_stats[severity.value] = sum(1 for r in results if r.severity == severity)

        return {
            'total_validations': total,
            'passed': passed,
            'failed': failed,
            'success_rate': passed / total if total > 0 else 0,
            'severity_distribution': severity_stats,
            'field_errors': [r.field for r in results if not r.success],
            'error_messages': [r.message for r in results if not r.success]
        }




