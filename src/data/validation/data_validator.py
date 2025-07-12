"""数据验证流水线模块"""
import logging
from typing import Dict, Any, Optional
from enum import Enum, auto

import pandas as pd

logger = logging.getLogger(__name__)

class DataType(Enum):
    """支持的数据类型枚举"""
    MARGIN_TRADING = auto()  # 融资融券
    DRAGON_BOARD = auto()    # 龙虎榜
    LEVEL2 = auto()          # Level2行情
    CORPORATE_ACTION = auto()  # 公司行为

class ValidationRule(Enum):
    """验证规则枚举"""
    NOT_NULL = auto()        # 非空检查
    VALUE_RANGE = auto()     # 值范围检查
    CONSISTENCY = auto()     # 一致性检查
    BUSINESS_RULE = auto()   # 业务规则检查

class DataValidator:
    """A股数据验证器"""

    def __init__(self):
        self._rules = {
            DataType.MARGIN_TRADING: [
                (ValidationRule.NOT_NULL, ['date', 'symbol', 'margin_balance']),
                (ValidationRule.VALUE_RANGE, {
                    'margin_balance': (0, None),  # 融资余额必须>=0
                    'short_balance': (0, None)    # 融券余额必须>=0
                }),
                (ValidationRule.BUSINESS_RULE, self._check_margin_ratio)
            ],
            DataType.DRAGON_BOARD: [
                (ValidationRule.NOT_NULL, ['symbol', 'buy_seats', 'sell_seats']),
                (ValidationRule.CONSISTENCY, self._check_dragon_board_consistency)
            ],
            DataType.LEVEL2: [
                (ValidationRule.NOT_NULL, ['symbol', 'price', 'bids', 'asks']),
                (ValidationRule.VALUE_RANGE, {
                    'price': (0, None),  # 价格必须>0
                    'volume': (0, None)  # 成交量必须>=0
                }),
                (ValidationRule.BUSINESS_RULE, self._check_level2_order_book)
            ]
        }

    def validate(self, data: Any, data_type: DataType) -> Dict[str, Any]:
        """
        验证数据
        Args:
            data: 要验证的数据(DataFrame或dict)
            data_type: 数据类型
        Returns:
            dict包含验证结果:
            - is_valid: 是否有效
            - errors: 错误列表
            - stats: 统计信息
        """
        if data_type not in self._rules:
            raise ValueError(f"Unsupported data type: {data_type}")

        result = {
            'is_valid': True,
            'errors': [],
            'stats': {}
        }

        # 应用所有规则
        for rule_type, rule_config in self._rules[data_type]:
            if rule_type == ValidationRule.NOT_NULL:
                self._apply_not_null_rule(data, rule_config, result)
            elif rule_type == ValidationRule.VALUE_RANGE:
                self._apply_value_range_rule(data, rule_config, result)
            elif rule_type == ValidationRule.CONSISTENCY:
                rule_config(data, result)
            elif rule_type == ValidationRule.BUSINESS_RULE:
                rule_config(data, result)

        return result

    def _apply_not_null_rule(self, data, columns, result):
        """应用非空检查规则"""
        if isinstance(data, pd.DataFrame):
            for col in columns:
                if col not in data.columns:
                    result['is_valid'] = False
                    result['errors'].append(f"Missing column: {col}")
                elif data[col].isnull().any():
                    result['is_valid'] = False
                    result['errors'].append(f"Null values found in column: {col}")
        else:  # dict
            for key in columns:
                if key not in data:
                    result['is_valid'] = False
                    result['errors'].append(f"Missing key: {key}")
                elif data[key] is None:
                    result['is_valid'] = False
                    result['errors'].append(f"Null value found for key: {key}")

    def _apply_value_range_rule(self, data, range_config, result):
        """应用值范围检查规则"""
        for field, (min_val, max_val) in range_config.items():
            if isinstance(data, pd.DataFrame):
                if field not in data.columns:
                    continue
                values = data[field]
            else:  # dict
                if field not in data:
                    continue
                values = [data[field]]

            for val in values:
                if val is None:
                    continue
                if min_val is not None and val < min_val:
                    result['is_valid'] = False
                    result['errors'].append(
                        f"Value {val} for {field} is below minimum {min_val}"
                    )
                if max_val is not None and val > max_val:
                    result['is_valid'] = False
                    result['errors'].append(
                        f"Value {val} for {field} is above maximum {max_val}"
                    )

    def _check_margin_ratio(self, data, result):
        """检查融资融券业务规则"""
        if isinstance(data, pd.DataFrame):
            if 'margin_balance' in data.columns and 'short_balance' in data.columns:
                ratio = data['short_balance'] / data['margin_balance']
                if (ratio > 0.5).any():  # 融券余额不应超过融资余额的50%
                    result['is_valid'] = False
                    result['errors'].append(
                        "Short balance exceeds 50% of margin balance"
                    )

    def _check_dragon_board_consistency(self, data, result):
        """检查龙虎榜数据一致性"""
        if isinstance(data, dict):
            if len(data.get('buy_seats', [])) == 0 and len(data.get('sell_seats', [])) == 0:
                result['is_valid'] = False
                result['errors'].append(
                    "Dragon board data has neither buy nor sell seats"
                )

    def _check_level2_order_book(self, data, result):
        """检查Level2订单簿业务规则"""
        if isinstance(data, dict):
            bids = data.get('bids', [])
            asks = data.get('asks', [])

            # 检查买卖价差
            if bids and asks:
                best_bid = max(b[0] for b in bids)
                best_ask = min(a[0] for a in asks)
                if best_bid >= best_ask:
                    result['is_valid'] = False
                    result['errors'].append(
                        "Order book has bid price >= ask price"
                    )
