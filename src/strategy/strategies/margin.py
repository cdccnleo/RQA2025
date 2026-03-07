import logging
"""融资融券策略实现"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from .basic_strategy import ChinaMarketStrategy

logger = logging.getLogger(__name__)


class MarginStrategy(ChinaMarketStrategy):

    """融资融券策略

    策略逻辑：
    1. 监控融资融券数据
    2. 分析融资融券余额变化
    3. 评估杠杆风险
    4. 控制仓位和风险
    5. 在合适时机进行融资融券操作
    """

    def __init__(self, config: Dict[str, Any]):

        super().__init__(config)

        # 融资融券策略特有参数
        self.max_leverage_ratio = config.get('max_leverage_ratio', 1.5)  # 最大杠杆率
        self.min_margin_ratio = config.get('min_margin_ratio', 0.3)  # 最小保证金比例
        self.max_position_ratio = config.get('max_position_ratio', 0.2)  # 最大仓位比例
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.10)  # 止损比例
        self.take_profit_ratio = config.get('take_profit_ratio', 0.25)  # 止盈比例
        self.margin_call_threshold = config.get('margin_call_threshold', 0.25)  # 平仓线

        # 策略状态
        self.positions = {}  # 当前持仓
        self.margin_account = {
            'cash': 1000000,  # 现金
            'margin_balance': 0,  # 融资余额
            'short_balance': 0,  # 融券余额
            'available_margin': 1000000  # 可用保证金
        }
        self.margin_history = {}  # 融资融券历史数据

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成融资融券交易信号"""
        if data.empty:
            return {'signals': [], 'confidence': 0.0}

        signals = []
        confidence = 0.0

        try:
            # 获取融资融券数据
            margin_data = self._get_margin_data(data)

            # 分析融资融券信号
            for stock in margin_data:
                signal = self._analyze_margin_stock(stock, data)
                if signal:
                    signals.append(signal)
                    confidence = max(confidence, signal.get('confidence', 0))

        except Exception as e:
            logger.error(f"生成融资融券信号时出错: {e}")

        return {
            'signals': signals,
            'confidence': confidence,
            'timestamp': datetime.now()
        }

    def _get_margin_data(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """获取融资融券数据"""
        margin_stocks = []

        # 模拟融资融券数据
        for _, row in data.iterrows():
            symbol = row.get('symbol', '')
            current_price = row.get('close', 0)
            amount = row.get('amount', 0)

            # 检查数据有效性
            if not symbol or amount is None:
                continue

            # 模拟融资融券数据
            margin_stocks.append({
                'symbol': symbol,
                'current_price': current_price,
                'margin_balance': amount * 0.05,  # 模拟融资余额
                'short_balance': amount * 0.02,  # 模拟融券余额
                'margin_ratio': 0.3,  # 模拟保证金比例
                'leverage_ratio': 1.2,  # 模拟杠杆率
                'interest_rate': 0.08,  # 模拟利率
                'timestamp': datetime.now()
            })

        return margin_stocks

    def _analyze_margin_stock(self, stock: Dict[str, Any], data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """分析融资融券股票"""
        symbol = stock['symbol']

        # 分析融资融券风险
        risk_analysis = self._analyze_margin_risk(stock)

        # 分析融资融券机会
        opportunity_analysis = self._analyze_margin_opportunity(stock)

        # 评估交易信号
        signal = self._evaluate_margin_signal(stock, risk_analysis, opportunity_analysis)

        if signal:
            return {
                'symbol': symbol,
                'signal_type': signal['type'],
                'price': stock['current_price'],
                'confidence': signal['confidence'],
                'reason': signal['reason'],
                'target_price': signal['target_price'],
                'stop_loss': signal['stop_loss'],
                'position_size': signal['position_size'],
                'leverage_ratio': signal['leverage_ratio'],
                'risk_level': risk_analysis['risk_level']
            }

        return None

    def _analyze_margin_risk(self, stock: Dict[str, Any]) -> Dict[str, Any]:
        """分析融资融券风险"""
        margin_balance = stock['margin_balance']
        short_balance = stock['short_balance']
        margin_ratio = stock['margin_ratio']
        leverage_ratio = stock['leverage_ratio']

        # 计算风险指标
        total_exposure = margin_balance + short_balance
        net_exposure = margin_balance - short_balance

        # 评估风险等级
        if leverage_ratio > self.max_leverage_ratio:
            risk_level = 'high'
        elif margin_ratio < self.min_margin_ratio:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'total_exposure': total_exposure,
            'net_exposure': net_exposure,
            'risk_level': risk_level,
            'leverage_ratio': leverage_ratio,
            'margin_ratio': margin_ratio
        }

    def _analyze_margin_opportunity(self, stock: Dict[str, Any]) -> Dict[str, Any]:
        """分析融资融券机会"""
        margin_balance = stock['margin_balance']
        short_balance = stock['short_balance']
        interest_rate = stock['interest_rate']

        # 计算融资融券机会
        margin_opportunity = margin_balance > short_balance * 2  # 融资机会
        short_opportunity = short_balance > margin_balance * 2  # 融券机会

        # 计算利率优势
        rate_advantage = interest_rate < 0.1  # 利率低于10%

        return {
            'margin_opportunity': margin_opportunity,
            'short_opportunity': short_opportunity,
            'rate_advantage': rate_advantage,
            'interest_rate': interest_rate
        }

    def _evaluate_margin_signal(self, stock: Dict[str, Any],


                                risk_analysis: Dict[str, Any],
                                opportunity_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """评估融资融券信号"""
        symbol = stock['symbol']

        # 检查风险等级
        if risk_analysis['risk_level'] == 'high':
            return None

        # 检查是否已有持仓
        if symbol in self.positions:
            return None

        # 检查可用保证金
        if self.margin_account['available_margin'] < 100000:  # 最少10万可用保证金
            return None

        # 确定交易类型
        if opportunity_analysis['margin_opportunity'] and opportunity_analysis['rate_advantage']:
            signal_type = 'margin_buy'
        elif opportunity_analysis['short_opportunity']:
            signal_type = 'short_sell'
        else:
            return None

        # 计算信号置信度
        confidence = self._calculate_margin_confidence(stock, risk_analysis, opportunity_analysis)

        if confidence < 0.7:  # 融资融券策略需要更高置信度
            return None

        # 计算目标价格和止损价格
        current_price = stock['current_price']
        if signal_type == 'margin_buy':
            target_price = current_price * (1 + self.take_profit_ratio)
            stop_loss = current_price * (1 - self.stop_loss_ratio)
        else:  # short_sell
            target_price = current_price * (1 - self.take_profit_ratio)
            stop_loss = current_price * (1 + self.stop_loss_ratio)

        # 计算仓位大小和杠杆率
        available_capital = self._get_available_capital()
        position_size = min(
            available_capital * self.max_position_ratio,
            available_capital * confidence * 0.3  # 融资融券策略相对保守
        )

        leverage_ratio = min(1.5, 1.0 + confidence * 0.5)  # 根据置信度调整杠杆

        return {
            'type': signal_type,
            'confidence': confidence,
            'reason': f"风险等级: {risk_analysis['risk_level']}, 杠杆率: {leverage_ratio:.2f}, 利率: {stock['interest_rate']:.1%}",
            'target_price': target_price,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'leverage_ratio': leverage_ratio
        }

    def _calculate_margin_confidence(self, stock: Dict[str, Any],


                                     risk_analysis: Dict[str, Any],
                                     opportunity_analysis: Dict[str, Any]) -> float:
        """计算融资融券信号置信度"""
        confidence = 0.0

        # 风险等级权重 30%
        if risk_analysis['risk_level'] == 'low':
            confidence += 0.3
        elif risk_analysis['risk_level'] == 'medium':
            confidence += 0.15

        # 杠杆率权重 25%
        leverage_weight = max(0, 1 - risk_analysis['leverage_ratio'] / self.max_leverage_ratio)
        confidence += leverage_weight * 0.25

        # 保证金比例权重 20%
        margin_weight = min(risk_analysis['margin_ratio'] / self.min_margin_ratio, 1.0)
        confidence += margin_weight * 0.2

        # 利率优势权重 15%
        if opportunity_analysis['rate_advantage']:
            confidence += 0.15

        # 机会权重 10%
        if opportunity_analysis['margin_opportunity'] or opportunity_analysis['short_opportunity']:
            confidence += 0.1

        return min(confidence, 1.0)

    def _get_available_capital(self) -> float:
        """获取可用资金"""
        return self.margin_account['available_margin']

    def execute_strategy(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行融资融券策略"""
        executed_trades = []

        try:
            for signal in signals.get('signals', []):
                if signal['signal_type'] in ['margin_buy', 'short_sell']:
                    trade = self._execute_margin_order(signal)
                    if trade:
                        executed_trades.append(trade)

            # 检查现有持仓是否需要调整
            adjustment_trades = self._check_margin_adjustments()
            executed_trades.extend(adjustment_trades)

        except Exception as e:
            logger.error(f"执行融资融券策略时出错: {e}")

        return executed_trades

    def _execute_margin_order(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """执行融资融券订单"""
        symbol = signal['symbol']
        price = signal['price']
        position_size = signal['position_size']
        leverage_ratio = signal['leverage_ratio']

        # 计算实际购买股数（考虑杠杆）
        actual_shares = int(position_size * leverage_ratio / price)
        if actual_shares <= 0:
            return None

        # 计算所需保证金
        required_margin = position_size * (1 / leverage_ratio)

        # 检查保证金是否充足
        if required_margin > self.margin_account['available_margin']:
            logger.warning(
                f"保证金不足: 需要 {required_margin}, 可用 {self.margin_account['available_margin']}")
            return None

        # 创建订单
        order = {
            'symbol': symbol,
            'side': 'buy' if signal['signal_type'] == 'margin_buy' else 'sell',
            'quantity': actual_shares,
            'price': price,
            'order_type': 'margin',
            'leverage_ratio': leverage_ratio,
            'required_margin': required_margin,
            'timestamp': datetime.now(),
            'signal': signal
        }

        # 更新保证金账户
        self.margin_account['available_margin'] -= required_margin
        if signal['signal_type'] == 'margin_buy':
            self.margin_account['margin_balance'] += position_size
        else:
            self.margin_account['short_balance'] += position_size

        # 记录持仓
        self.positions[symbol] = {
            'quantity': actual_shares,
            'avg_price': price,
            'target_price': signal['target_price'],
            'stop_loss': signal['stop_loss'],
            'entry_time': datetime.now(),
            'leverage_ratio': leverage_ratio,
            'required_margin': required_margin,
            'position_type': signal['signal_type']
        }

        # 记录交易
        self.log_trade(order)

        logger.info(
            f"执行融资融券订单: {symbol}, 类型: {signal['signal_type']}, 价格: {price}, 数量: {actual_shares}, 杠杆: {leverage_ratio}")

        return order

    def _check_margin_adjustments(self) -> List[Dict[str, Any]]:
        """检查融资融券持仓调整"""
        adjustment_trades = []

        for symbol, position in list(self.positions.items()):
            # 获取当前价格
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue

            # 检查止盈止损
            if self._should_exit_position(symbol, position, current_price):
                trade = self._execute_margin_exit(symbol, position, current_price)
                if trade:
                    adjustment_trades.append(trade)
                    del self.positions[symbol]

            # 检查保证金预警
            elif self._check_margin_call(symbol, position, current_price):
                trade = self._execute_margin_exit(symbol, position, current_price, 'margin_call')
                if trade:
                    adjustment_trades.append(trade)
                    del self.positions[symbol]

        return adjustment_trades

    def _should_exit_position(self, symbol: str, position: Dict[str, Any], current_price: float) -> bool:
        """检查是否应该退出持仓"""
        position_type = position.get('position_type', 'margin_buy')

        if position_type == 'margin_buy':
            return (current_price >= position['target_price']
                    or current_price <= position['stop_loss'])
        else:  # short_sell
            return (current_price <= position['target_price']
                    or current_price >= position['stop_loss'])

    def _check_margin_call(self, symbol: str, position: Dict[str, Any], current_price: float) -> bool:
        """检查是否触发平仓线"""
        position_type = position.get('position_type', 'margin_buy')
        avg_price = position['avg_price']

        if position_type == 'margin_buy':
            # 融资买入，价格下跌触发平仓
            loss_ratio = (avg_price - current_price) / avg_price
        else:
            # 融券卖出，价格上涨触发平仓
            loss_ratio = (current_price - avg_price) / avg_price

        return loss_ratio > self.margin_call_threshold

    def _execute_margin_exit(self, symbol: str, position: Dict[str, Any],


                             current_price: float, reason: str = 'normal') -> Optional[Dict[str, Any]]:
        """执行融资融券退出"""
        position_type = position.get('position_type', 'margin_buy')

        order = {
            'symbol': symbol,
            'side': 'sell' if position_type == 'margin_buy' else 'buy',
            'quantity': position['quantity'],
            'price': current_price,
            'order_type': 'market',
            'timestamp': datetime.now(),
            'reason': reason
        }

        # 更新保证金账户
        required_margin = position.get('required_margin', 0)
        self.margin_account['available_margin'] += required_margin

        if position_type == 'margin_buy':
            self.margin_account['margin_balance'] -= position['quantity'] * position['avg_price']
        else:
            self.margin_account['short_balance'] -= position['quantity'] * position['avg_price']

        # 记录交易
        self.log_trade(order)

        logger.info(f"执行融资融券退出: {symbol}, 类型: {position_type}, 价格: {current_price}, 原因: {reason}")

        return order

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        # 这里应该从市场数据模块获取
        # 模拟返回价格
        return 10.0  # 模拟价格

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """获取策略指标"""
        total_positions = len(self.positions)
        total_value = sum(
            pos['quantity'] * self._get_current_price(symbol)
            for symbol, pos in self.positions.items()
        )

        # 计算杠杆率统计
        avg_leverage = sum(
            pos.get('leverage_ratio', 1.0)
            for pos in self.positions.values()
        ) / total_positions if total_positions > 0 else 1.0

        return {
            'total_positions': total_positions,
            'total_value': total_value,
            'avg_leverage': avg_leverage,
            'available_margin': self.margin_account['available_margin'],
            'margin_balance': self.margin_account['margin_balance'],
            'short_balance': self.margin_account['short_balance']
        }
