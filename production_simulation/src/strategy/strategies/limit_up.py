import logging
"""涨停板策略实现"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from .basic_strategy import ChinaMarketStrategy

logger = logging.getLogger(__name__)


class LimitUpStrategy(ChinaMarketStrategy):

    """涨停板策略

    策略逻辑：
    1. 监控涨停板股票
    2. 分析涨停板强度（封单量、封单金额、封单时间）
    3. 评估后续上涨概率
    4. 在合适时机介入
    """

    def __init__(self, config: Dict[str, Any]):

        super().__init__(config)

        # 涨停板策略特有参数
        self.limit_up_threshold = config.get('limit_up_threshold', 0.095)  # 涨停阈值
        self.min_seal_amount = config.get('min_seal_amount', 1000000)  # 最小封单金额
        self.min_seal_ratio = config.get('min_seal_ratio', 0.1)  # 最小封单比例
        self.max_position_ratio = config.get('max_position_ratio', 0.3)  # 最大仓位比例
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.05)  # 止损比例
        self.take_profit_ratio = config.get('take_profit_ratio', 0.15)  # 止盈比例

        # 策略状态
        self.positions = {}  # 当前持仓
        self.signal_history = []  # 信号历史

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成涨停板交易信号"""
        if data.empty:
            return {'signals': [], 'confidence': 0.0}

        signals = []
        confidence = 0.0

        try:
            # 计算涨停板指标
            limit_up_stocks = self._identify_limit_up_stocks(data)

            for stock in limit_up_stocks:
                signal = self._analyze_limit_up_stock(stock, data)
                if signal:
                    signals.append(signal)
                    confidence = max(confidence, signal.get('confidence', 0))

        except Exception as e:
            logger.error(f"生成涨停板信号时出错: {e}")

        return {
            'signals': signals,
            'confidence': confidence,
            'timestamp': datetime.now()
        }

    def _identify_limit_up_stocks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """识别涨停板股票"""
        limit_up_stocks = []

        for _, row in data.iterrows():
            symbol = row.get('symbol', '')
            current_price = row.get('close', 0)
            prev_close = row.get('pre_close', 0)

            if prev_close is None or prev_close <= 0:
                continue

            # 计算涨跌幅
            price_change_ratio = (current_price - prev_close) / prev_close

            # 判断是否涨停
            if price_change_ratio >= self.limit_up_threshold:
                limit_up_stocks.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'prev_close': prev_close,
                    'price_change_ratio': price_change_ratio,
                    'volume': row.get('volume', 0),
                    'amount': row.get('amount', 0),
                    'turnover_rate': row.get('turnover_rate', 0)
                })

        return limit_up_stocks

    def _analyze_limit_up_stock(self, stock: Dict[str, Any], data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """分析涨停板股票"""
        symbol = stock['symbol']

        # 获取该股票的详细数据
        stock_data = data[data['symbol'] == symbol]
        if stock_data.empty:
            return None

        # 确保stock_data是DataFrame类型
        if not isinstance(stock_data, pd.DataFrame):
            stock_data = pd.DataFrame([stock_data])

        # 计算涨停板强度指标
        strength_metrics = self._calculate_limit_up_strength(stock, stock_data)

        # 评估买入信号
        buy_signal = self._evaluate_buy_signal(stock, strength_metrics)

        if buy_signal:
            return {
                'symbol': symbol,
                'signal_type': 'buy',
                'price': stock['current_price'],
                'confidence': buy_signal['confidence'],
                'reason': buy_signal['reason'],
                'target_price': buy_signal['target_price'],
                'stop_loss': buy_signal['stop_loss'],
                'position_size': buy_signal['position_size']
            }

        return None

    def _calculate_limit_up_strength(self, stock: Dict[str, Any], stock_data: pd.DataFrame) -> Dict[str, float]:
        """计算涨停板强度指标"""
        symbol = stock['symbol']

        # 获取封单数据（模拟）
        seal_amount = stock.get('amount', 0) * 0.8  # 假设80 % 为封单金额
        seal_volume = stock.get('volume', 0) * 0.8  # 假设80 % 为封单量

        # 计算封单比例
        total_amount = stock.get('amount', 0)
        seal_ratio = seal_amount / total_amount if total_amount > 0 else 0

        # 计算封单强度
        strength_score = min(seal_ratio * 100, 100)

        # 计算历史涨停次数
        historical_limit_ups = self._count_historical_limit_ups(symbol, stock_data)

        return {
            'seal_amount': seal_amount,
            'seal_volume': seal_volume,
            'seal_ratio': seal_ratio,
            'strength_score': strength_score,
            'historical_limit_ups': historical_limit_ups
        }

    def _count_historical_limit_ups(self, symbol: str, stock_data: pd.DataFrame) -> int:
        """统计历史涨停次数"""
        if stock_data.empty:
            return 0

        limit_up_count = 0
        for _, row in stock_data.iterrows():
            current_price = row.get('close', 0)
            prev_close = row.get('pre_close', 0)

        if prev_close is not None and prev_close > 0:
            price_change_ratio = (current_price - prev_close) / prev_close
        if price_change_ratio >= self.limit_up_threshold:
            limit_up_count += 1

        return limit_up_count

    def _evaluate_buy_signal(self, stock: Dict[str, Any], strength_metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """评估买入信号"""
        symbol = stock['symbol']

        # 检查是否已有持仓
        if symbol in self.positions:
            return None

        # 检查封单强度
        if strength_metrics['seal_amount'] < self.min_seal_amount:
            return None

        if strength_metrics['seal_ratio'] < self.min_seal_ratio:
            return None

        # 计算信号置信度
        confidence = self._calculate_signal_confidence(stock, strength_metrics)

        if confidence < 0.6:  # 最低置信度要求
            return None

        # 计算目标价格和止损价格
        current_price = stock['current_price']
        target_price = current_price * (1 + self.take_profit_ratio)
        stop_loss = current_price * (1 - self.stop_loss_ratio)

        # 计算仓位大小
        available_capital = self._get_available_capital()
        position_size = min(
            available_capital * self.max_position_ratio,
            available_capital * confidence * 0.5  # 根据置信度调整仓位
        )

        return {
            'confidence': confidence,
            'reason': f"涨停板强度: {strength_metrics['strength_score']:.2f}, 封单比例: {strength_metrics['seal_ratio']:.2%}",
            'target_price': target_price,
            'stop_loss': stop_loss,
            'position_size': position_size
        }

    def _calculate_signal_confidence(self, stock: Dict[str, Any], strength_metrics: Dict[str, float]) -> float:
        """计算信号置信度"""
        confidence = 0.0

        # 封单强度权重 40%
        seal_weight = min(strength_metrics['strength_score'] / 100, 1.0)
        confidence += seal_weight * 0.4

        # 封单比例权重 30%
        ratio_weight = min(strength_metrics['seal_ratio'] / 0.5, 1.0)  # 假设50 % 为满分
        confidence += ratio_weight * 0.3

        # 历史涨停次数权重 20%
        historical_weight = min(strength_metrics['historical_limit_ups'] / 5, 1.0)  # 假设5次为满分
        confidence += historical_weight * 0.2

        # 成交量权重 10%
        volume_weight = min(stock.get('turnover_rate', 0) / 0.1, 1.0)  # 假设10 % 换手率为满分
        confidence += volume_weight * 0.1

        return min(confidence, 1.0)

    def _get_available_capital(self) -> float:
        """获取可用资金"""
        # 这里应该从账户管理模块获取
        return 1000000  # 模拟100万可用资金

    def execute_strategy(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行涨停板策略"""
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
            logger.error(f"执行涨停板策略时出错: {e}")

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
            'entry_time': datetime.now()
        }

        # 记录交易
        self.log_trade(order)

        logger.info(f"执行涨停板买入: {symbol}, 价格: {price}, 数量: {shares}")

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

        return adjustment_trades

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

        logger.info(f"执行涨停板卖出: {symbol}, 价格: {current_price}, 原因: {reason}")

        return order

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        # 这里应该从市场数据模块获取
        # 模拟返回价格
        return 10.0  # 模拟价格

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """获取策略指标"""
        total_positions = len(self.positions)
        if total_positions == 0:
            total_value = 0.0
        else:
            total_value = sum(
                pos['quantity'] * self._get_current_price(symbol)
                for symbol, pos in self.positions.items()
            )

        return {
            'total_positions': total_positions,
            'total_value': float(total_value),
            'signal_count': len(self.signal_history),
            'last_signal_time': self.signal_history[-1] if self.signal_history else None
        }
