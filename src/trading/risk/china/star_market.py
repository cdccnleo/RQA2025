#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
科创板特殊交易规则检查模块
实现科创板特有的交易规则检查逻辑
"""

from typing import Dict, Any, Optional
from datetime import time
from src.utils.logger import get_logger
from src.trading.risk.china.price_limit import PriceLimitChecker
from src.trading.risk.china.t1_restriction import T1RestrictionChecker

logger = get_logger(__name__)

class STARMarketRuleChecker:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化科创板规则检查器
        :param config: 配置参数
        """
        self.config = config
        self.price_limit_checker = PriceLimitChecker(config)
        self.t1_checker = T1RestrictionChecker(config)

        # 科创板特殊参数
        self.star_market_price_limit = 0.20  # 20%涨跌停限制
        self.after_hours_trading_start = time(15, 5)  # 盘后交易开始时间
        self.after_hours_trading_end = time(15, 30)   # 盘后交易结束时间

    def check(self, order: Dict[str, Any], position: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查科创板特殊交易规则
        :param order: 订单信息
        :param position: 持仓信息
        :return: 检查结果
        """
        if not order['symbol'].startswith('688'):
            return {
                'passed': True,
                'reason': 'not_star_market',
                'message': '非科创板股票'
            }

        results = {
            'passed': True,
            'failed_checks': [],
            'warnings': []
        }

        # 检查20%涨跌停限制
        price_limit_result = self.price_limit_checker.check(
            order,
            limit_rate=self.star_market_price_limit
        )
        if not price_limit_result['passed']:
            results['passed'] = False
            results['failed_checks'].append({
                'rule': 'price_limit',
                'message': f"违反科创板20%涨跌停限制: {price_limit_result['message']}"
            })

        # 检查T+1限制
        t1_result = self.t1_checker.check(order, position)
        if not t1_result['passed']:
            results['passed'] = False
            results['failed_checks'].append({
                'rule': 't1_restriction',
                'message': f"违反科创板T+1交易限制: {t1_result['message']}"
            })

        # 检查盘后固定价格交易
        if self._is_after_hours_trading():
            if order['type'] != 'LIMIT':
                results['passed'] = False
                results['failed_checks'].append({
                    'rule': 'after_hours_trading',
                    'message': "盘后固定价格交易必须使用限价单"
                })

            if order['time_in_force'] != 'DAY':
                results['passed'] = False
                results['failed_checks'].append({
                    'rule': 'after_hours_trading',
                    'message': "盘后固定价格交易时间必须设置为当日有效"
                })

        # 检查单笔申报数量
        if order['quantity'] > 200000:
            results['warnings'].append({
                'rule': 'order_size',
                'message': "科创板单笔申报数量不得超过20万股"
            })

        return results

    def _is_after_hours_trading(self) -> bool:
        """
        判断当前是否处于盘后固定价格交易时段
        :return: 是否在盘后交易时段
        """
        from datetime import datetime
        now = datetime.now().time()
        return self.after_hours_trading_start <= now <= self.after_hours_trading_end

    def get_after_hours_fixed_price(self, symbol: str) -> Optional[float]:
        """
        获取盘后固定价格交易的参考价格
        :param symbol: 股票代码
        :return: 固定价格，如果无法获取则返回None
        """
        # TODO: 实现从交易所API获取盘后固定价格
        return None

    def check_circuit_breaker(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查科创板熔断机制
        :param market_data: 市场数据
        :return: 检查结果
        """
        # 科创板不设熔断机制
        return {
            'passed': True,
            'message': '科创板不设熔断机制'
        }

    def check_price_limits(self, symbol: str, price: float) -> Dict[str, Any]:
        """
        检查科创板价格限制
        :param symbol: 股票代码
        :param price: 待检查价格
        :return: 检查结果
        """
        if not symbol.startswith('688'):
            return {
                'passed': True,
                'message': '非科创板股票'
            }

        # 获取参考价格（通常是前收盘价）
        reference_price = self._get_reference_price(symbol)
        if reference_price is None:
            return {
                'passed': False,
                'message': '无法获取参考价格'
            }

        upper_limit = reference_price * (1 + self.star_market_price_limit)
        lower_limit = reference_price * (1 - self.star_market_price_limit)

        if price > upper_limit:
            return {
                'passed': False,
                'message': f'价格超过科创板涨停限制: {upper_limit:.2f}'
            }
        elif price < lower_limit:
            return {
                'passed': False,
                'message': f'价格低于科创板跌停限制: {lower_limit:.2f}'
            }
        else:
            return {
                'passed': True,
                'message': '价格在科创板涨跌停限制范围内'
            }

    def _get_reference_price(self, symbol: str) -> Optional[float]:
        """
        获取参考价格（前收盘价）
        :param symbol: 股票代码
        :return: 参考价格，如果无法获取则返回None
        """
        # TODO: 实现从数据层获取前收盘价
        return None
