"""A股结算引擎"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from ..features.feature_engine import FeatureEngine
from ..trading.order_executor import Order, Trade

logger = logging.getLogger(__name__)

@dataclass
class SettlementConfig:
    """结算配置"""
    t1_settlement: bool = True  # 是否启用T+1结算
    freeze_ratio: float = 1.0  # 资金冻结比例
    settlement_time: str = "16:00"  # 结算时间
    a_share_fees: Dict[str, float] = {  # A股费用标准
        "commission": 0.00025,  # 佣金万2.5
        "stamp_duty": 0.001,    # 印花税千1
        "transfer_fee": 0.00002 # 过户费万0.2
    }
    margin_rules: Dict[str, float] = {  # 融资融券规则
        "maintenance_ratio": 1.3,  # 维持担保比例130%
        "concentration_limit": 0.3  # 单票集中度限制30%
    }

class SettlementEngine:
    """A股结算引擎"""

    def __init__(self, feature_engine: FeatureEngine, config: Optional[SettlementConfig] = None):
        self.engine = feature_engine
        self.config = config or SettlementConfig()

        # 结算状态
        self.frozen_cash = 0.0
        self.settled_positions = {}
        self.last_settlement_time = None

    def process_t1_settlement(self, trades: List[Trade]) -> float:
        """处理T+1结算"""
        if not self.config.t1_settlement:
            return 0.0

        total_amount = 0.0

        # 1. 计算结算金额
        for trade in trades:
            # 计算交易金额
            amount = trade.price * trade.quantity

            # 计算费用
            fees = self._calculate_a_share_fees(trade)

            # 累计冻结金额
            total_amount += (amount + fees) * self.config.freeze_ratio

            # 记录待结算持仓
            if trade.side == "BUY":
                self.settled_positions[trade.symbol] = \
                    self.settled_positions.get(trade.symbol, 0) + trade.quantity
            else:
                self.settled_positions[trade.symbol] = \
                    self.settled_positions.get(trade.symbol, 0) - trade.quantity

        # 2. 冻结资金
        self.frozen_cash = total_amount
        self.last_settlement_time = datetime.now()

        logger.info(f"T+1结算完成，冻结资金: {self.frozen_cash:.2f}元")
        return total_amount

    def release_settlement(self) -> float:
        """释放结算资金"""
        if not self.config.t1_settlement:
            return 0.0

        # 检查结算时间是否已过
        if not self._is_settlement_time_passed():
            logger.warning("结算时间未到，不能释放资金")
            return 0.0

        # 释放冻结资金
        released_amount = self.frozen_cash
        self.frozen_cash = 0.0

        # 更新持仓
        self.settled_positions = {
            k: v for k, v in self.settled_positions.items() if v > 0
        }

        logger.info(f"释放结算资金: {released_amount:.2f}元")
        return released_amount

    def _is_settlement_time_passed(self) -> bool:
        """检查结算时间是否已过"""
        if not self.last_settlement_time:
            return False

        now = datetime.now()
        settlement_time = datetime.strptime(self.config.settlement_time, "%H:%M").time()

        # 检查是否已过结算时间
        if now.time() >= settlement_time:
            # 检查是否已经是第二天
            if now.date() > self.last_settlement_time.date():
                return True

        return False

    def _calculate_a_share_fees(self, trade: Trade) -> float:
        """计算A股交易费用"""
        amount = trade.price * trade.quantity

        # 计算佣金(双向收取)
        commission = amount * self.config.a_share_fees["commission"]

        # 计算印花税(卖出收取)
        stamp_duty = 0
        if trade.side == "SELL":
            stamp_duty = amount * self.config.a_share_fees["stamp_duty"]

        # 计算过户费(双向收取)
        transfer_fee = amount * self.config.a_share_fees["transfer_fee"]

        return commission + stamp_duty + transfer_fee

    def process_margin_settlement(self, positions: Dict[str, float]) -> Dict[str, float]:
        """处理融资融券结算"""
        # 1. 检查维持担保比例
        maintenance_ratio = self._calculate_maintenance_ratio(positions)
        if maintenance_ratio < self.config.margin_rules["maintenance_ratio"]:
            logger.warning(f"维持担保比例不足: {maintenance_ratio:.2f}")
            # 实际实现中会触发追保或平仓

        # 2. 检查单票集中度
        total_value = sum(positions.values())
        adjustments = {}
        for symbol, value in positions.items():
            if total_value > 0 and value / total_value > self.config.margin_rules["concentration_limit"]:
                adjustments[symbol] = total_value * self.config.margin_rules["concentration_limit"]
                logger.warning(f"单票集中度超标: {symbol} {value/total_value:.2%}")

        return adjustments

    def _calculate_maintenance_ratio(self, positions: Dict[str, float]) -> float:
        """计算维持担保比例"""
        # 简化为1.5
        # 实际实现中会计算(现金+证券市值)/(融资余额+融券市值)
        return 1.5

    def reconcile_with_broker(self, broker_data: Dict[str, float]) -> Dict[str, float]:
        """与券商系统对账"""
        discrepancies = {}

        # 检查持仓差异
        for symbol, quantity in self.settled_positions.items():
            if symbol in broker_data:
                if abs(quantity - broker_data[symbol]) > 1e-6:  # 考虑浮点误差
                    discrepancies[symbol] = {
                        "local": quantity,
                        "broker": broker_data[symbol]
                    }
            else:
                discrepancies[symbol] = {
                    "local": quantity,
                    "broker": 0
                }

        # 检查资金差异
        if abs(self.frozen_cash - broker_data.get("frozen_cash", 0)) > 1e-6:
            discrepancies["cash"] = {
                "local": self.frozen_cash,
                "broker": broker_data.get("frozen_cash", 0)
            }

        return discrepancies

class ChinaSettlementEngine(SettlementEngine):
    """A股特定结算引擎"""

    def __init__(self, feature_engine: FeatureEngine, config: Optional[SettlementConfig] = None):
        super().__init__(feature_engine, config)
        self.config.t1_settlement = True  # 强制启用T+1

        # A股特定结算参数
        self._init_a_share_params()

    def _init_a_share_params(self):
        """初始化A股特定结算参数"""
        self.stock_types = {
            "ST": {"min_price": 1.0, "lot_size": 100},
            "688": {"min_price": 1.0, "lot_size": 200},
            "normal": {"min_price": 0.01, "lot_size": 100}
        }

        # 科创板特定规则
        self.star_market_rules = {
            "after_hours_trading": True,  # 盘后固定价格交易
            "price_limit": 0.2  # 涨跌幅限制20%
        }

    def process_t1_settlement(self, trades: List[Trade]) -> float:
        """处理A股T+1结算"""
        total_amount = 0.0

        for trade in trades:
            # A股特定检查
            if trade.symbol.startswith("688"):
                # 科创板特定处理
                if not self._check_star_market_rules(trade):
                    continue
            elif trade.symbol.startswith(("ST", "*ST")):
                # ST股票特定处理
                if not self._check_st_stock_rules(trade):
                    continue

            # 计算交易金额
            amount = trade.price * trade.quantity

            # 计算A股特定费用
            fees = self._calculate_a_share_fees(trade)

            # 累计冻结金额
            total_amount += (amount + fees) * self.config.freeze_ratio

            # 记录待结算持仓
            if trade.side == "BUY":
                self.settled_positions[trade.symbol] = \
                    self.settled_positions.get(trade.symbol, 0) + trade.quantity
            else:
                self.settled_positions[trade.symbol] = \
                    self.settled_positions.get(trade.symbol, 0) - trade.quantity

        # 冻结资金
        self.frozen_cash = total_amount
        self.last_settlement_time = datetime.now()

        logger.info(f"A股T+1结算完成，冻结资金: {self.frozen_cash:.2f}元")
        return total_amount

    def _check_star_market_rules(self, trade: Trade) -> bool:
        """检查科创板特定规则"""
        # 检查涨跌幅限制
        if trade.price > trade.prev_close * (1 + self.star_market_rules["price_limit"]):
            return False
        if trade.price < trade.prev_close * (1 - self.star_market_rules["price_limit"]):
            return False

        return True

    def _check_st_stock_rules(self, trade: Trade) -> bool:
        """检查ST股票特定规则"""
        # 检查涨跌幅限制
        if trade.price > trade.prev_close * 1.05:
            return False
        if trade.price < trade.prev_close * 0.95:
            return False

        return True

    def process_after_hours_trading(self, trades: List[Trade]) -> float:
        """处理科创板盘后固定价格交易"""
        if not self.star_market_rules["after_hours_trading"]:
            return 0.0

        total_amount = 0.0

        for trade in trades:
            if trade.symbol.startswith("688"):
                # 盘后交易使用收盘价
                trade.price = trade.close_price

                # 计算交易金额
                amount = trade.price * trade.quantity

                # 计算费用
                fees = self._calculate_a_share_fees(trade)

                # 累计冻结金额
                total_amount += (amount + fees) * self.config.freeze_ratio

                # 记录待结算持仓
                if trade.side == "BUY":
                    self.settled_positions[trade.symbol] = \
                        self.settled_positions.get(trade.symbol, 0) + trade.quantity
                else:
                    self.settled_positions[trade.symbol] = \
                        self.settled_positions.get(trade.symbol, 0) - trade.quantity

        # 冻结资金
        self.frozen_cash += total_amount
        logger.info(f"科创板盘后交易结算完成，冻结资金: {total_amount:.2f}元")
        return total_amount
