#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数值计算边界处理工具
解决除零、溢出、精度等问题
"""

import math
import sys
from decimal import getcontext
from typing import Union, Optional, Tuple, Any


class NumericBoundaryHandler:
    """数值边界处理器"""

    def __init__(self):
        # 设置Decimal精度
        getcontext().prec = 28

        # 定义安全阈值
        self.MAX_FLOAT = sys.float_info.max
        self.MIN_FLOAT = sys.float_info.min
        self.MAX_INT = sys.maxsize
        self.MIN_INT = -sys.maxsize - 1

    def safe_divide(self, dividend: Union[int, float],
                    divisor: Union[int, float],
                    default: Union[int, float] = 0.0) -> Union[int, float]:
        """安全的除法运算"""
        try:
            # 检查除数类型
            if not isinstance(divisor, (int, float)):
                return default

            # 检查除数是否为零
            if divisor == 0:
                return default

            # 检查数值范围
            if abs(dividend) > self.MAX_FLOAT or abs(divisor) > self.MAX_FLOAT:
                return default

            result = dividend / divisor

            # 检查结果是否有效
            if not (self.MIN_FLOAT < result < self.MAX_FLOAT):
                return default

            return result

        except (ZeroDivisionError, OverflowError, ValueError):
            return default

    def safe_sqrt(self, value: Union[int, float],
                  default: Union[int, float] = 0.0) -> Union[int, float]:
        """安全的平方根计算"""
        try:
            if not isinstance(value, (int, float)):
                return default

            if value < 0:
                return default

            if value > self.MAX_FLOAT:
                return default

            result = math.sqrt(value)

            if not (self.MIN_FLOAT < result < self.MAX_FLOAT):
                return default

            return result

        except (ValueError, OverflowError):
            return default

    def safe_log(self, value: Union[int, float],
                 base: Union[int, float] = math.e,
                 default: Union[int, float] = 0.0) -> Union[int, float]:
        """安全的对数计算"""
        try:
            if not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
                return default

            if value <= 0 or base <= 0 or base == 1:
                return default

            if value > self.MAX_FLOAT or base > self.MAX_FLOAT:
                return default

            result = math.log(value, base)

            if not (self.MIN_FLOAT < result < self.MAX_FLOAT):
                return default

            return result

        except (ValueError, ZeroDivisionError, OverflowError):
            return default

    def safe_exp(self, value: Union[int, float],
                 default: Union[int, float] = 1.0) -> Union[int, float]:
        """安全指数计算"""
        try:
            if not isinstance(value, (int, float)):
                return default

            # 防止溢出
            if value > 700:  # ln(MAX_FLOAT) ≈ 709
                return self.MAX_FLOAT
            if value < -700:
                return 0.0

            result = math.exp(value)

            if not (self.MIN_FLOAT < result < self.MAX_FLOAT):
                return default

            return result

        except (OverflowError, ValueError):
            return default

    def safe_power(self, base: Union[int, float],
                   exponent: Union[int, float],
                   default: Union[int, float] = 0.0) -> Union[int, float]:
        """安全的幂运算"""
        try:
            if not isinstance(base, (int, float)) or not isinstance(exponent, (int, float)):
                return default

            # 特殊情况处理
            if base == 0:
                return 0.0 if exponent > 0 else default
            if exponent == 0:
                return 1.0

            # 检查数值范围
            if abs(base) > 1e100 or abs(exponent) > 100:
                return default

            result = math.pow(base, exponent)

            if not (self.MIN_FLOAT < abs(result) < self.MAX_FLOAT):
                return default

            return result

        except (ValueError, OverflowError, ZeroDivisionError):
            return default

    def safe_percentage(self, value: Union[int, float],
                        total: Union[int, float],
                        default: Union[int, float] = 0.0) -> Union[int, float]:
        """安全百分比计算"""
        if total == 0:
            return default
        return self.safe_divide(value * 100, total, default)

    def clamp_value(self, value: Union[int, float],
                    min_val: Union[int, float],
                    max_val: Union[int, float]) -> Union[int, float]:
        """限制数值范围"""
        if not isinstance(value, (int, float)):
            return min_val

        if value < min_val:
            return min_val
        if value > max_val:
            return max_val
        return value

    def safe_mean(self, values: list, default: Union[int, float] = 0.0) -> Union[int, float]:
        """安全平均值计算"""
        if not isinstance(values, list) or len(values) == 0:
            return default

        valid_values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]

        if len(valid_values) == 0:
            return default

        total = sum(valid_values)
        return self.safe_divide(total, len(valid_values), default)

    def safe_std(self, values: list, default: Union[int, float] = 0.0) -> Union[int, float]:
        """安全标准差计算"""
        if not isinstance(values, list) or len(values) < 2:
            return default

        valid_values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]

        if len(valid_values) < 2:
            return default

        mean = self.safe_mean(valid_values, default)
        if mean == default:
            return default

        variance = sum((v - mean) ** 2 for v in valid_values)
        return self.safe_sqrt(self.safe_divide(variance, len(valid_values) - 1, 0), default)

    def safe_array_access(self, array: list, index: int, default: Any = None) -> Any:
        """安全的数组访问"""
        if not isinstance(array, list):
            return default

        if not isinstance(index, int):
            return default

        if 0 <= index < len(array):
            return array[index]

        return default

    def safe_dict_access(self, dictionary: dict, key: str, default: Any = None) -> Any:
        """安全的字典访问"""
        if not isinstance(dictionary, dict):
            return default

        return dictionary.get(key, default)

    def validate_numeric_input(self, value: Any,
                               min_val: Optional[float] = None,
                               max_val: Optional[float] = None,
                               allow_none: bool = False) -> Tuple[bool, Union[int, float, None]]:
        """验证数值输入"""
        if value is None:
            return allow_none, None

        if not isinstance(value, (int, float)):
            return False, None

        if math.isnan(value) or math.isinf(value):
            return False, None

        if min_val is not None and value < min_val:
            return False, None

        if max_val is not None and value > max_val:
            return False, None

        return True, value


class FinancialCalculations:
    """金融计算边界处理"""

    def __init__(self):
        self.numeric_handler = NumericBoundaryHandler()

    def safe_return_calculation(self, current_price: float,
                                previous_price: float,
                                default: float = 0.0) -> float:
        """安全收益率计算"""
        if not isinstance(current_price, (int, float)) or not isinstance(previous_price, (int, float)):
            return default

        if previous_price <= 0:
            return default

        return self.numeric_handler.safe_divide(current_price - previous_price, previous_price, default)

    def safe_volatility_calculation(self, prices: list, default: float = 0.0) -> float:
        """安全波动率计算"""
        if not isinstance(prices, list) or len(prices) < 2:
            return default

        # 计算收益率
        returns = []
        for i in range(1, len(prices)):
            ret = self.safe_return_calculation(prices[i], prices[i-1], None)
            if ret is not None:
                returns.append(ret)

        if len(returns) < 2:
            return default

        # 计算波动率（收益率标准差）
        return self.numeric_handler.safe_std(returns, default)

    def safe_sharpe_ratio(self, returns: list, risk_free_rate: float = 0.0, default: float = 0.0) -> float:
        """安全夏普比率计算"""
        if not isinstance(returns, list) or len(returns) < 2:
            return default

        # 计算平均收益率
        avg_return = self.numeric_handler.safe_mean(returns, None)
        if avg_return is None:
            return default

        # 计算波动率
        volatility = self.numeric_handler.safe_std(returns, None)
        if volatility is None or volatility == 0:
            return default

        # 计算夏普比率
        excess_return = avg_return - risk_free_rate
        return self.numeric_handler.safe_divide(excess_return, volatility, default)

    def safe_drawdown_calculation(self, prices: list, default: float = 0.0) -> float:
        """安全最大回撤计算"""
        if not isinstance(prices, list) or len(prices) < 2:
            return default

        if not all(isinstance(p, (int, float)) and p > 0 for p in prices):
            return default

        max_price = max(prices)
        min_price_after_max = min(prices[prices.index(max_price):])

        if max_price == 0:
            return default

        drawdown = (max_price - min_price_after_max) / max_price
        return max(0, drawdown)  # 确保非负


class TradingCalculations:
    """交易计算边界处理"""

    def __init__(self):
        self.numeric_handler = NumericBoundaryHandler()
        self.financial_calc = FinancialCalculations()

    def safe_position_size(self, capital: float,
                           risk_per_trade: float,
                           stop_loss: float,
                           default: float = 0.0) -> float:
        """安全仓位大小计算"""
        if not isinstance(capital, (int, float)) or capital <= 0:
            return default

        if not isinstance(risk_per_trade, (int, float)) or risk_per_trade <= 0 or risk_per_trade >= 1:
            return default

        if not isinstance(stop_loss, (int, float)) or stop_loss <= 0:
            return default

        risk_amount = capital * risk_per_trade
        return self.numeric_handler.safe_divide(risk_amount, stop_loss, default)

    def safe_portfolio_value(self, positions: dict, prices: dict, default: float = 0.0) -> float:
        """安全投资组合价值计算"""
        if not isinstance(positions, dict) or not isinstance(prices, dict):
            return default

        total_value = 0.0

        for symbol, quantity in positions.items():
            if not isinstance(quantity, (int, float)):
                continue

            price = self.numeric_handler.safe_dict_access(prices, symbol, None)
            if price is None or not isinstance(price, (int, float)) or price <= 0:
                continue

            position_value = quantity * price
            if abs(position_value) > 1e12:  # 防止溢出
                continue

            total_value += position_value

        return total_value if abs(total_value) < 1e15 else default

    def safe_pnl_calculation(self, entry_price: float,
                             exit_price: float,
                             quantity: float,
                             commission: float = 0.0,
                             default: float = 0.0) -> float:
        """安全盈亏计算"""
        if not all(isinstance(x, (int, float)) for x in [entry_price, exit_price, quantity]):
            return default

        if entry_price <= 0 or exit_price <= 0:
            return default

        gross_pnl = (exit_price - entry_price) * abs(quantity)

        if abs(gross_pnl) > 1e12:  # 防止溢出
            return default

        return gross_pnl - commission


if __name__ == "__main__":
    # 测试数值边界处理器
    handler = NumericBoundaryHandler()

    print("测试安全除法:")
    print(f"10 / 2 = {handler.safe_divide(10, 2)}")
    print(f"10 / 0 = {handler.safe_divide(10, 0)}")
    print(f"'a' / 2 = {handler.safe_divide('a', 2)}")

    print("\n测试安全平方根:")
    print(f"sqrt(4) = {handler.safe_sqrt(4)}")
    print(f"sqrt(-1) = {handler.safe_sqrt(-1)}")
    print(f"sqrt('a') = {handler.safe_sqrt('a')}")

    print("\n测试安全对数:")
    print(f"log(10) = {handler.safe_log(10)}")
    print(f"log(0) = {handler.safe_log(0)}")
    print(f"log(-1) = {handler.safe_log(-1)}")

    # 测试金融计算
    financial = FinancialCalculations()

    print("\n测试收益率计算:")
    print(f"收益率 (100->110) = {financial.safe_return_calculation(110, 100)}")
    print(f"收益率 (100->0) = {financial.safe_return_calculation(100, 0)}")

    # 测试交易计算
    trading = TradingCalculations()

    print("\n测试仓位大小计算:")
    print(f"仓位大小 (10000, 0.02, 2) = {trading.safe_position_size(10000, 0.02, 2)}")
    print(f"仓位大小 (10000, 0, 2) = {trading.safe_position_size(10000, 0, 2)}")
