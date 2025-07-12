from typing import Dict, Any, List, Optional
from enum import Enum
import datetime
import json

class RiskLevel(Enum):
    """风险等级枚举"""
    SAFE = 0
    WARNING = 1
    DANGER = 2
    BLOCKED = 3

class ValidationResult:
    """验证结果类"""

    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.risk_level = RiskLevel.SAFE
        self.metadata = {}

    def add_error(self, rule: str, message: str, risk: RiskLevel):
        """添加验证错误"""
        self.is_valid = False
        self.errors.append({
            'rule': rule,
            'message': message,
            'timestamp': datetime.datetime.now().isoformat(),
            'risk_level': risk.name
        })
        if risk.value > self.risk_level.value:
            self.risk_level = risk

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'is_valid': self.is_valid,
            'risk_level': self.risk_level.name,
            'errors': self.errors,
            'metadata': self.metadata
        }

class RuleValidator:
    """基础规则验证器"""

    def __init__(self):
        self.required_fields = []
        self.type_checks = {}
        self.range_checks = {}

    def validate(self, data: Dict) -> ValidationResult:
        """执行基础验证"""
        result = ValidationResult()

        # 必填字段检查
        for field in self.required_fields:
            if field not in data:
                result.add_error(
                    'required_field',
                    f"缺少必填字段: {field}",
                    RiskLevel.BLOCKED
                )

        # 类型检查
        for field, expected_type in self.type_checks.items():
            if field in data and not isinstance(data[field], expected_type):
                result.add_error(
                    'type_check',
                    f"字段类型错误: {field} 应为 {expected_type}",
                    RiskLevel.BLOCKED
                )

        # 范围检查
        for field, (min_val, max_val) in self.range_checks.items():
            if field in data and not (min_val <= data[field] <= max_val):
                result.add_error(
                    'range_check',
                    f"字段值超出范围: {field} 应在 {min_val}-{max_val}",
                    RiskLevel.WARNING
                )

        return result

class RiskEngine:
    """风险控制引擎"""

    def __init__(self):
        self.validators = {
            'pre_check': RuleValidator(),
            'business_rules': {},
            'risk_checks': []
        }
        self.audit_log = []

    def add_business_rule(self, name: str, validator):
        """添加业务规则验证器"""
        self.validators['business_rules'][name] = validator

    def add_risk_check(self, check_func):
        """添加风险检查函数"""
        self.validators['risk_checks'].append(check_func)

    def validate(self, data: Dict) -> Dict:
        """执行完整风险验证流程"""
        # 预验证
        pre_result = self.validators['pre_check'].validate(data)
        if not pre_result.is_valid:
            self._log_validation(data, pre_result)
            return pre_result.to_dict()

        # 业务规则验证
        for name, validator in self.validators['business_rules'].items():
            result = validator.validate(data)
            if not result.is_valid:
                self._log_validation(data, result)
                return result.to_dict()

        # 风险检查
        final_result = ValidationResult()
        for check in self.validators['risk_checks']:
            check_result = check(data)
            if not check_result.is_valid:
                for error in check_result.errors:
                    final_result.add_error(
                        error['rule'],
                        error['message'],
                        RiskLevel(error['risk_level'])
                    )

        if not final_result.is_valid:
            self._log_validation(data, final_result)

        return final_result.to_dict()

    def _log_validation(self, data: Dict, result: ValidationResult):
        """记录审计日志"""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'input_data': data,
            'validation_result': result.to_dict(),
            'decision': 'REJECTED' if not result.is_valid else 'APPROVED'
        }
        self.audit_log.append(log_entry)

    def get_audit_log(self, limit: int = 100) -> List[Dict]:
        """获取审计日志"""
        return self.audit_log[-limit:]

# 示例业务规则验证器
class MarketRuleValidator(RuleValidator):
    """市场规则验证器"""

    def validate(self, data: Dict) -> ValidationResult:
        result = super().validate(data)

        # 示例: 检查交易时间
        if 'trade_time' in data:
            hour = data['trade_time'].hour
            if not (9 <= hour < 16):
                result.add_error(
                    'market_hours',
                    "非交易时间段",
                    RiskLevel.BLOCKED
                )

        return result

# 示例风险检查函数
def liquidity_risk_check(data: Dict) -> ValidationResult:
    """流动性风险检查"""
    result = ValidationResult()

    if 'amount' in data and data['amount'] > 1_000_000:
        result.add_error(
            'liquidity_risk',
            "大额交易需人工审核",
            RiskLevel.DANGER
        )

    return result
