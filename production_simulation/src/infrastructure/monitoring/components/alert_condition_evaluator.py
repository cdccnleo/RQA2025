"""
告警条件评估器组件

负责评估告警触发条件。
"""

import re
from typing import Dict, Any


class AlertConditionEvaluator:
    """告警条件评估器"""
    
    def __init__(self):
        """初始化条件评估器"""
        self.operator_handlers = {
            'eq': self._evaluate_equals,
            'ne': self._evaluate_not_equals,
            'gt': self._evaluate_greater_than,
            'gte': self._evaluate_greater_than_or_equal,
            'lt': self._evaluate_less_than,
            'lte': self._evaluate_less_than_or_equal,
            'contains': self._evaluate_contains,
            'regex': self._evaluate_regex
        }
    
    def evaluate_condition(self, condition: Dict[str, Any], data: Dict[str, Any]) -> bool:
        """评估触发条件"""
        try:
            operator = condition.get('operator', 'eq')
            field = condition.get('field')
            expected_value = condition.get('value')

            if field not in data:
                return False

            actual_value = data[field]
            
            # 使用操作符处理器
            handler = self.operator_handlers.get(operator)
            if handler:
                return handler(actual_value, expected_value)
            
            print(f"不支持的操作符: {operator}")
            return False

        except Exception as e:
            print(f"条件评估失败: {e}")
            return False
    
    def _evaluate_equals(self, actual_value: Any, expected_value: Any) -> bool:
        """评估等于操作"""
        return actual_value == expected_value
    
    def _evaluate_not_equals(self, actual_value: Any, expected_value: Any) -> bool:
        """评估不等于操作"""
        return actual_value != expected_value
    
    def _evaluate_greater_than(self, actual_value: Any, expected_value: Any) -> bool:
        """评估大于操作"""
        try:
            return float(actual_value) > float(expected_value)
        except (ValueError, TypeError):
            return False
    
    def _evaluate_greater_than_or_equal(self, actual_value: Any, expected_value: Any) -> bool:
        """评估大于等于操作"""
        try:
            return float(actual_value) >= float(expected_value)
        except (ValueError, TypeError):
            return False
    
    def _evaluate_less_than(self, actual_value: Any, expected_value: Any) -> bool:
        """评估小于操作"""
        try:
            return float(actual_value) < float(expected_value)
        except (ValueError, TypeError):
            return False
    
    def _evaluate_less_than_or_equal(self, actual_value: Any, expected_value: Any) -> bool:
        """评估小于等于操作"""
        try:
            return float(actual_value) <= float(expected_value)
        except (ValueError, TypeError):
            return False
    
    def _evaluate_contains(self, actual_value: Any, expected_value: Any) -> bool:
        """评估包含操作"""
        try:
            return str(expected_value) in str(actual_value)
        except Exception:
            return False
    
    def _evaluate_regex(self, actual_value: Any, expected_value: Any) -> bool:
        """评估正则表达式匹配"""
        try:
            return bool(re.search(str(expected_value), str(actual_value)))
        except Exception:
            return False
    
    def validate_condition(self, condition: Dict[str, Any]) -> bool:
        """验证条件格式是否正确"""
        required_fields = ['operator', 'field', 'value']
        
        # 检查必需字段
        for field in required_fields:
            if field not in condition:
                print(f"条件缺少必需字段: {field}")
                return False
        
        # 检查操作符是否支持
        operator = condition.get('operator')
        if operator not in self.operator_handlers:
            print(f"不支持的操作符: {operator}")
            return False
        
        return True
    
    def get_supported_operators(self) -> list:
        """获取支持的操作符列表"""
        return list(self.operator_handlers.keys())

