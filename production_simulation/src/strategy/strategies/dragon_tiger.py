import logging
"""龙虎榜策略实现"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from .basic_strategy import ChinaMarketStrategy

logger = logging.getLogger(__name__)


class DragonTigerStrategy(ChinaMarketStrategy):

    """龙虎榜策略

    策略逻辑：
    1. 监控龙虎榜数据
    2. 分析机构资金流向
    3. 识别游资行为模式
    4. 跟踪主力资金动向
    5. 在合适时机跟随或反向操作
    """

    def __init__(self, config: Dict[str, Any]):

        super().__init__(config)

        # 龙虎榜策略特有参数
        self.min_net_amount = config.get('min_net_amount', 5000000)  # 最小净流入金额
        self.min_buy_amount = config.get('min_buy_amount', 10000000)  # 最小买入金额
        self.max_position_ratio = config.get('max_position_ratio', 0.25)  # 最大仓位比例
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.08)  # 止损比例
        self.take_profit_ratio = config.get('take_profit_ratio', 0.20)  # 止盈比例
        self.follow_days = config.get('follow_days', 3)  # 跟踪天数

        # 策略状态
        self.positions = {}  # 当前持仓
        self.dragon_tiger_history = {}  # 龙虎榜历史数据
        self.institution_flow = {}  # 机构资金流向

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成龙虎榜交易信号"""
        if data.empty:
            return {'signals': [], 'confidence': 0.0}

        signals = []
        confidence = 0.0

        try:
            # 获取龙虎榜数据
            dragon_tiger_data = self._get_dragon_tiger_data(data)

            # 分析龙虎榜信号
            for stock in dragon_tiger_data:
                signal = self._analyze_dragon_tiger_stock(stock, data)
                if signal:
                    signals.append(signal)
                    confidence = max(confidence, signal.get('confidence', 0))

        except Exception as e:
            logger.error(f"生成龙虎榜信号时出错: {e}")

        return {
            'signals': signals,
            'confidence': confidence,
            'timestamp': datetime.now()
        }

    def _get_dragon_tiger_data(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """获取龙虎榜数据"""
        dragon_tiger_stocks = []

        # 模拟龙虎榜数据
        for _, row in data.iterrows():
            symbol = row.get('symbol', '')
            current_price = row.get('close', 0)
            amount = row.get('amount', 0)

            # 检查数据有效性
            if not symbol or amount is None:
                continue

            # 模拟龙虎榜上榜条件
        if self._is_dragon_tiger_candidate(symbol, row):
            dragon_tiger_stocks.append({
                'symbol': symbol,
                'current_price': current_price,
                'net_amount': amount * 0.1,  # 模拟净流入
                'buy_amount': amount * 0.6,  # 模拟买入金额
                'sell_amount': amount * 0.4,  # 模拟卖出金额
                'institution_ratio': 0.3,  # 模拟机构占比
                'retail_ratio': 0.7,  # 模拟散户占比
                'timestamp': datetime.now()
            })

        return dragon_tiger_stocks

    def _is_dragon_tiger_candidate(self, symbol: str, row: pd.Series) -> bool:
        """判断是否为龙虎榜候选股票"""
        # 模拟龙虎榜上榜条件
        volume = row.get('volume', 0)
        amount = row.get('amount', 0)
        turnover_rate = row.get('turnover_rate', 0)

        # 检查数据有效性
        if volume is None or amount is None or turnover_rate is None:
            return False

        # 条件：成交量、成交额、换手率达到一定标准
        return (volume > 1000000
                and amount > 50000000
                and turnover_rate > 0.05)

    def _analyze_dragon_tiger_stock(self, stock: Dict[str, Any], data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """分析龙虎榜股票"""
        symbol = stock['symbol']

        # 分析机构资金流向
        institution_analysis = self._analyze_institution_flow(stock)

        # 分析游资行为
        retail_analysis = self._analyze_retail_behavior(stock)

        # 评估买入信号
        buy_signal = self._evaluate_dragon_tiger_signal(
            stock, institution_analysis, retail_analysis)

        if buy_signal:
            return {
                'symbol': symbol,
                'signal_type': 'buy',
                'price': stock['current_price'],
                'confidence': buy_signal['confidence'],
                'reason': buy_signal['reason'],
                'target_price': buy_signal['target_price'],
                'stop_loss': buy_signal['stop_loss'],
                'position_size': buy_signal['position_size'],
                'institution_flow': institution_analysis,
                'retail_behavior': retail_analysis
            }

        return None

    def _analyze_institution_flow(self, stock: Dict[str, Any]) -> Dict[str, Any]:
        """分析机构资金流向"""
        net_amount = stock['net_amount']
        buy_amount = stock['buy_amount']
        sell_amount = stock['sell_amount']
        institution_ratio = stock['institution_ratio']

        # 计算机构净流入
        institution_net = net_amount * institution_ratio

        # 判断机构态度
        if institution_net > self.min_net_amount:
            attitude = 'bullish'
        elif institution_net < -self.min_net_amount:
            attitude = 'bearish'
        else:
            attitude = 'neutral'

        return {
            'net_amount': net_amount,
            'institution_net': institution_net,
            'attitude': attitude,
            'buy_strength': buy_amount / (buy_amount + sell_amount) if (buy_amount + sell_amount) > 0 else 0.5
        }

    def _analyze_retail_behavior(self, stock: Dict[str, Any]) -> Dict[str, Any]:
        """分析游资行为"""
        retail_ratio = stock['retail_ratio']
        net_amount = stock['net_amount']

        # 计算游资净流入
        retail_net = net_amount * retail_ratio

        # 判断游资行为模式
        if retail_net > 0 and retail_net > net_amount * 0.5:
            behavior = 'following'
        elif retail_net < 0 and abs(retail_net) > net_amount * 0.5:
            behavior = 'contrarian'
        else:
            behavior = 'neutral'

        return {
            'retail_net': retail_net,
            'behavior': behavior,
            'momentum': 'strong' if abs(retail_net) > self.min_net_amount else 'weak'
        }

    def _evaluate_dragon_tiger_signal(self, stock: Dict[str, Any],


                                      institution_analysis: Dict[str, Any],
                                      retail_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """评估龙虎榜信号"""
        symbol = stock['symbol']

        # 检查是否已有持仓
        if symbol in self.positions:
            return None

        # 检查机构资金流向
        if institution_analysis['attitude'] == 'bearish':
            return None

        # 检查游资行为
        if retail_analysis['behavior'] == 'contrarian' and retail_analysis['momentum'] == 'strong':
            return None

        # 计算信号置信度
        confidence = self._calculate_dragon_tiger_confidence(
            stock, institution_analysis, retail_analysis)

        if confidence < 0.65:  # 龙虎榜策略需要更高置信度
            return None

        # 计算目标价格和止损价格
        current_price = stock['current_price']
        target_price = current_price * (1 + self.take_profit_ratio)
        stop_loss = current_price * (1 - self.stop_loss_ratio)

        # 计算仓位大小
        available_capital = self._get_available_capital()
        position_size = min(
            available_capital * self.max_position_ratio,
            available_capital * confidence * 0.4  # 龙虎榜策略相对保守
        )

        return {
            'confidence': confidence,
            'reason': f"机构态度: {institution_analysis['attitude']}, 游资行为: {retail_analysis['behavior']}, 净流入: {stock['net_amount']/10000:.0f}万",
            'target_price': target_price,
            'stop_loss': stop_loss,
            'position_size': position_size
        }

    def _calculate_dragon_tiger_confidence(self, stock: Dict[str, Any],


                                           institution_analysis: Dict[str, Any],
                                           retail_analysis: Dict[str, Any]) -> float:
        """计算龙虎榜信号置信度"""
        confidence = 0.0

        # 机构态度权重 40%
        if institution_analysis['attitude'] == 'bullish':
            confidence += 0.4
        elif institution_analysis['attitude'] == 'neutral':
            confidence += 0.2

        # 机构买入强度权重 25%
        buy_strength = institution_analysis['buy_strength']
        confidence += buy_strength * 0.25

        # 游资行为权重 20%
        if retail_analysis['behavior'] == 'following':
            confidence += 0.2
        elif retail_analysis['behavior'] == 'neutral':
            confidence += 0.1

        # 净流入金额权重 15%
        net_amount_ratio = min(stock['net_amount'] / self.min_net_amount, 2.0)
        confidence += min(net_amount_ratio * 0.15, 0.15)

        return min(confidence, 1.0)

    def _get_available_capital(self) -> float:
        """获取可用资金"""
        # 这里应该从账户管理模块获取
        return 1000000  # 模拟100万可用资金

    def execute_strategy(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行龙虎榜策略"""
        executed_trades = []

        try:
            for signal in signals.get('signals', []):
                if signal['signal_type'] == 'buy':
                    trade = self._execute_buy_order(signal)
                    if trade:
                        executed_trades.append(trade)

            # 检查现有持仓是否需要调整
            adjustment_trades = self._check_position_adjustments()
            executed_trades.extend(adjustment_trades)

        except Exception as e:
            logger.error(f"执行龙虎榜策略时出错: {e}")

        return executed_trades

    def _execute_buy_order(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """执行买入订单"""
        symbol = signal['symbol']
        price = signal['price']
        position_size = signal['position_size']

        # 计算购买股数
        shares = int(position_size / price)
        if shares <= 0:
            return None

        # 创建订单
        order = {
            'symbol': symbol,
            'side': 'buy',
            'quantity': shares,
            'price': price,
            'order_type': 'limit',
            'timestamp': datetime.now(),
            'signal': signal
        }

        # 记录持仓
        self.positions[symbol] = {
            'quantity': shares,
            'avg_price': price,
            'target_price': signal['target_price'],
            'stop_loss': signal['stop_loss'],
            'entry_time': datetime.now(),
            'institution_flow': signal.get('institution_flow', {}),
            'retail_behavior': signal.get('retail_behavior', {})
        }

        # 记录交易
        self.log_trade(order)

        logger.info(f"执行龙虎榜买入: {symbol}, 价格: {price}, 数量: {shares}")

        return order

    def _check_position_adjustments(self) -> List[Dict[str, Any]]:
        """检查持仓调整"""
        adjustment_trades = []

        for symbol, position in list(self.positions.items()):
            # 获取当前价格
            current_price = self._get_current_price(symbol)
            if current_price is None:
                continue

            # 检查止盈止损
        if current_price >= position['target_price']:
            # 止盈
            trade = self._execute_sell_order(symbol, position, 'take_profit')
            if trade:
                adjustment_trades.append(trade)
                del self.positions[symbol]

            elif current_price <= position['stop_loss']:
                # 止损
                trade = self._execute_sell_order(symbol, position, 'stop_loss')
            if trade:
                adjustment_trades.append(trade)
                del self.positions[symbol]

            # 检查龙虎榜跟踪
            elif self._should_exit_dragon_tiger(symbol, position):
                trade = self._execute_sell_order(symbol, position, 'dragon_tiger_exit')
        if trade:
            adjustment_trades.append(trade)
            del self.positions[symbol]

        return adjustment_trades

    def _should_exit_dragon_tiger(self, symbol: str, position: Dict[str, Any]) -> bool:
        """检查是否应该退出龙虎榜跟踪"""
        entry_time = position['entry_time']
        current_time = datetime.now()

        # 检查是否超过跟踪天数
        if (current_time - entry_time).days > self.follow_days:
            return True

        # 检查机构态度是否改变
        institution_flow = position.get('institution_flow', {})
        if institution_flow.get('attitude') == 'bearish':
            return True

        return False

    def _execute_sell_order(self, symbol: str, position: Dict[str, Any], reason: str) -> Optional[Dict[str, Any]]:
        """执行卖出订单"""
        current_price = self._get_current_price(symbol)
        if current_price is None:
            return None

        order = {
            'symbol': symbol,
            'side': 'sell',
            'quantity': position['quantity'],
            'price': current_price,
            'order_type': 'market',
            'timestamp': datetime.now(),
            'reason': reason
        }

        # 记录交易
        self.log_trade(order)

        logger.info(f"执行龙虎榜卖出: {symbol}, 价格: {current_price}, 原因: {reason}")

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

        # 计算机构资金流向统计
        bullish_count = sum(
            1 for pos in self.positions.values()
            if pos.get('institution_flow', {}).get('attitude') == 'bullish'
        )

        return {
            'total_positions': total_positions,
            'total_value': total_value,
            'bullish_institution_count': bullish_count,
            'institution_ratio': bullish_count / total_positions if total_positions > 0 else 0
        }
