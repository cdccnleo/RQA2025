#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
订单管理器 - 处理交易订单的执行和管理
"""

from typing import Dict, Any
from datetime import datetime, time
import random

class OrderManager:
    """管理交易订单的执行和状态"""

    def __init__(self, config: Dict[str, Any]):
        """初始化订单管理器"""
        self.config = config
        self.mock_time = None  # 用于测试的时间模拟
        self.last_close_prices = {}  # 记录各标的的昨日收盘价
        self.fixed_prices = {}  # 记录固定价格(如科创板盘后定价)

    def execute(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """执行交易订单

        Args:
            order: 订单字典，包含symbol, price, quantity, side等字段

        Returns:
            执行结果字典，包含执行状态、成交均价、成交量等信息
        """
        # 模拟执行逻辑
        executed_price = order["price"]
        if order["side"] == "buy":
            executed_price *= 1.0002  # 模拟滑点
        else:
            executed_price *= 0.9998

        return {
            "status": "FILLED",
            "symbol": order["symbol"],
            "executed_price": round(executed_price, 4),
            "executed_quantity": order["quantity"],
            "timestamp": datetime.now().isoformat()
        }

    def get_last_close_price(self, symbol: str) -> float:
        """获取指定标的的昨日收盘价

        Args:
            symbol: 标的代码

        Returns:
            昨日收盘价
        """
        if symbol not in self.last_close_prices:
            # 模拟生成一个合理的收盘价
            base_price = 100.0 if symbol.startswith("6") else 50.0
            self.last_close_prices[symbol] = round(base_price * random.uniform(0.8, 1.2), 2)
        return self.last_close_prices[symbol]

    def set_mock_time(self, time_str: str):
        """设置模拟时间(仅用于测试)

        Args:
            time_str: 时间字符串，格式"HH:MM:SS"
        """
        hour, minute, second = map(int, time_str.split(":"))
        self.mock_time = time(hour, minute, second)

    def get_fixed_price(self, symbol: str) -> float:
        """获取固定价格(如科创板盘后定价)

        Args:
            symbol: 标的代码

        Returns:
            固定价格
        """
        if symbol not in self.fixed_prices:
            # 模拟生成固定价格为收盘价的1.0倍
            self.fixed_prices[symbol] = self.get_last_close_price(symbol)
        return self.fixed_prices[symbol]
