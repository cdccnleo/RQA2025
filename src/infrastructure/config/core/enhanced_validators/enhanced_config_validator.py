from typing import Callable, Dict, Any

from ...validators.enhanced_validators import ConfigValidationResult


class EnhancedConfigValidator:
    """增强配置验证器"""

    def __init__(self):
        self.validators: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def add_validator(self, name: str, validator: Callable[[Dict[str, Any]], Any]):
        self.validators[name] = validator

    def validate(self, config: Dict[str, Any]) -> ConfigValidationResult:
        result = ConfigValidationResult()

        for name, validator in self.validators.items():
            try:
                validator_result = validator(config)

                if isinstance(validator_result, ConfigValidationResult):
                    if not validator_result.is_valid:
                        result.is_valid = False
                    result.errors.extend(validator_result.errors)
                    result.warnings.extend(validator_result.warnings)
                    result.recommendations.extend(validator_result.recommendations)
                elif validator_result is False:
                    result.is_valid = False
                    result.errors.append(f"Validator {name} returned False")
                elif isinstance(validator_result, str) and validator_result:
                    result.warnings.append(validator_result)

            except Exception as exc:  # noqa: BLE001 - 保持向后兼容的宽泛捕获
                result.is_valid = False
                result.errors.append(f"{name}: {exc}")

        return result
