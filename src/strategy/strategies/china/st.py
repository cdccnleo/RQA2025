import logging
"""ST股票策略实现"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from .basic_strategy import ChinaMarketStrategy

logger = logging.getLogger(__name__)


class STStrategy(ChinaMarketStrategy):

    """ST股票策略

    策略逻辑：
    1. 识别ST股票
    2. 分析ST股票风险特征
    3. 评估摘帽可能性
    4. 控制仓位和风险
    5. 在合适时机进行ST股票操作
    """

    def __init__(self, config: Dict[str, Any]):

        super().__init__(config)

        # ST股票策略特有参数
        self.max_position_ratio = config.get('max_position_ratio', 0.15)  # 最大仓位比例（ST股票风险较高）
        self.stop_loss_ratio = config.get('stop_loss_ratio', 0.15)  # 止损比例
        self.take_profit_ratio = config.get('take_profit_ratio', 0.30)  # 止盈比例
        self.max_st_ratio = config.get('max_st_ratio', 0.2)  # 最大ST股票占比
        self.min_volume = config.get('min_volume', 500000)  # 最小成交量要求

        # 策略状态
        self.positions = {}  # 当前持仓
        self.st_stocks = set()  # ST股票池
        self.st_risk_levels = {}  # ST股票风险等级

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成ST股票交易信号"""
        if data.empty:
            return {'signals': [], 'confidence': 0.0}

        signals = []
        confidence = 0.0

        try:
            # 获取ST股票数据
            st_data = self._get_st_stocks_data(data)

            # 分析ST股票信号
            for stock in st_data:
                signal = self._analyze_st_stock(stock, data)
                if signal:
                    signals.append(signal)
                    confidence = max(confidence, signal.get('confidence', 0))

        except Exception as e:
            logger.error(f"生成ST股票信号时出错: {e}")

        return {
            'signals': signals,
            'confidence': confidence,
            'timestamp': datetime.now()
        }

    def _get_st_stocks_data(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """获取ST股票数据"""
        st_stocks = []

        # 模拟ST股票数据
        for _, row in data.iterrows():
            symbol = row.get('symbol', '')
            current_price = row.get('close', 0)
            volume = row.get('volume', 0)
            amount = row.get('amount', 0)

            # 检查数据有效性
            if not symbol or volume is None or amount is None:
                continue

            # 判断是否为ST股票
        if self._is_st_stock(symbol, row):
            st_stocks.append({
                'symbol': symbol,
                'current_price': current_price,
                'volume': volume,
                'amount': amount,
                'st_type': self._get_st_type(symbol),  # ST类型
                'risk_level': self._assess_st_risk(symbol, row),  # 风险等级
                'delisting_risk': self._assess_delisting_risk(symbol, row),  # 退市风险
                'recovery_potential': self._assess_recovery_potential(symbol, row),  # 恢复潜力
                'timestamp': datetime.now()
            })

        return st_stocks

    def _is_st_stock(self, symbol: str, row: pd.Series) -> bool:
        """判断是否为ST股票"""
        # 模拟ST股票识别逻辑
        # 实际应该从股票基本信息中获取
        return symbol.startswith('*ST') or symbol.startswith('ST')

    def _get_st_type(self, symbol: str) -> str:
        """获取ST类型"""
        if symbol.startswith('*ST'):
            return '*ST'  # 退市风险警示
        elif symbol.startswith('ST'):
            return 'ST'   # 特别处理
        else:
            return 'NORMAL'

    def _assess_st_risk(self, symbol: str, row: pd.Series) -> str:
        """评估ST股票风险等级"""
        volume = row.get('volume', 0)
        amount = row.get('amount', 0)

        if volume is None or amount is None:
            return 'high'

        # 根据成交量和成交额评估风险
        if volume < self.min_volume:
            return 'high'
        elif amount < 10000000:  # 成交额小于1000万
            return 'medium'
        else:
            return 'low'

    def _assess_delisting_risk(self, symbol: str, row: pd.Series) -> float:
        """评估退市风险"""
        # 模拟退市风险评估
        # 实际应该基于财务指标、监管信息等
        base_risk = 0.3  # 基础退市风险

        # 根据ST类型调整风险
        if symbol.startswith('*ST'):
            base_risk += 0.4
        elif symbol.startswith('ST'):
            base_risk += 0.2

        return min(base_risk, 1.0)

    def _assess_recovery_potential(self, symbol: str, row: pd.Series) -> float:
        """评估恢复潜力"""
        # 模拟恢复潜力评估
        # 实际应该基于基本面分析、行业前景等
        base_potential = 0.2  # 基础恢复潜力

        # 根据成交活跃度调整潜力
        volume = row.get('volume', 0)
        if volume is not None and volume > 1000000:
            base_potential += 0.3

        return min(base_potential, 1.0)

    def _analyze_st_stock(self, stock: Dict[str, Any], data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """分析ST股票"""
        symbol = stock['symbol']

        # 分析ST股票风险
        risk_analysis = self._analyze_st_risk(stock)

        # 分析ST股票机会
        opportunity_analysis = self._analyze_st_opportunity(stock)

        # 评估交易信号
        signal = self._evaluate_st_signal(stock, risk_analysis, opportunity_analysis)

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
                'risk_level': risk_analysis['risk_level'],
                'st_type': stock['st_type']
            }

        return None

    def _analyze_st_risk(self, stock: Dict[str, Any]) -> Dict[str, Any]:
        """分析ST股票风险"""
        risk_level = stock['risk_level']
        delisting_risk = stock['delisting_risk']
        st_type = stock['st_type']

        # 计算综合风险评分
        risk_score = 0.0

        # 风险等级权重 40%
        if risk_level == 'high':
            risk_score += 0.4
        elif risk_level == 'medium':
            risk_score += 0.2

        # 退市风险权重 35%
        risk_score += delisting_risk * 0.35

        # ST类型权重 25%
        if st_type == '*ST':
            risk_score += 0.25
        elif st_type == 'ST':
            risk_score += 0.15

        return {
            'risk_level': risk_level,
            'delisting_risk': delisting_risk,
            'st_type': st_type,
            'risk_score': risk_score
        }

    def _analyze_st_opportunity(self, stock: Dict[str, Any]) -> Dict[str, Any]:
        """分析ST股票机会"""
        recovery_potential = stock['recovery_potential']
        volume = stock['volume']
        amount = stock['amount']

        # 计算机会评分
        opportunity_score = 0.0

        # 恢复潜力权重 50%
        opportunity_score += recovery_potential * 0.5

        # 成交活跃度权重 30%
        volume_score = min(volume / 1000000, 1.0)  # 100万为满分
        opportunity_score += volume_score * 0.3

        # 成交额权重 20%
        amount_score = min(amount / 50000000, 1.0)  # 5000万为满分
        opportunity_score += amount_score * 0.2

        return {
            'recovery_potential': recovery_potential,
            'opportunity_score': opportunity_score,
            'volume_active': volume > self.min_volume,
            'amount_sufficient': amount > 10000000
        }

    def _evaluate_st_signal(self, stock: Dict[str, Any],


                            risk_analysis: Dict[str, Any],
                            opportunity_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """评估ST股票信号"""
        symbol = stock['symbol']

        # 检查风险等级
        if risk_analysis['risk_score'] > 0.7:  # 风险过高
            return None

        # 检查是否已有持仓
        if symbol in self.positions:
            return None

        # 检查ST股票占比限制
        if self._get_st_position_ratio() >= self.max_st_ratio:
            return None

        # 检查成交活跃度
        if not opportunity_analysis['volume_active']:
            return None

        # 确定交易类型
        if opportunity_analysis['opportunity_score'] > 0.6 and risk_analysis['risk_score'] < 0.5:
            signal_type = 'buy'
        else:
            return None

        # 计算信号置信度
        confidence = self._calculate_st_confidence(stock, risk_analysis, opportunity_analysis)

        if confidence < 0.75:  # ST股票策略需要更高置信度
            return None

        # 计算目标价格和止损价格
        current_price = stock['current_price']
        target_price = current_price * (1 + self.take_profit_ratio)
        stop_loss = current_price * (1 - self.stop_loss_ratio)

        # 计算仓位大小（ST股票仓位较小）
        available_capital = self._get_available_capital()
        position_size = min(
            available_capital * self.max_position_ratio,
            available_capital * confidence * 0.2  # ST股票策略非常保守
        )

        return {
            'type': signal_type,
            'confidence': confidence,
            'reason': f"ST类型: {stock['st_type']}, 风险等级: {risk_analysis['risk_level']}, 恢复潜力: {stock['recovery_potential']:.1%}",
            'target_price': target_price,
            'stop_loss': stop_loss,
            'position_size': position_size
        }

    def _calculate_st_confidence(self, stock: Dict[str, Any],


                                 risk_analysis: Dict[str, Any],
                                 opportunity_analysis: Dict[str, Any]) -> float:
        """计算ST股票信号置信度"""
        confidence = 0.0

        # 风险评分权重 40%（风险越低越好）
        risk_weight = max(0, 1 - risk_analysis['risk_score'])
        confidence += risk_weight * 0.4

        # 机会评分权重 35%
        opportunity_weight = opportunity_analysis['opportunity_score']
        confidence += opportunity_weight * 0.35

        # 成交活跃度权重 15%
        if opportunity_analysis['volume_active']:
            confidence += 0.15

        # 成交额充足度权重 10%
        if opportunity_analysis['amount_sufficient']:
            confidence += 0.1

        return min(confidence, 1.0)

    def _get_st_position_ratio(self) -> float:
        """获取ST股票持仓占比"""
        total_positions = len(self.positions)
        st_positions = sum(
            1 for symbol in self.positions.keys()
            if self._is_st_stock(symbol, pd.Series())
        )

        return st_positions / total_positions if total_positions > 0 else 0

    def _get_available_capital(self) -> float:
        """获取可用资金"""
        # 这里应该从账户管理模块获取
        return 1000000  # 模拟100万可用资金

    def execute_strategy(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行ST股票策略"""
        executed_trades = []

        try:
            for signal in signals.get('signals', []):
                if signal['signal_type'] == 'buy':
                    trade = self._execute_buy_order(signal)
                    if trade:
                        executed_trades.append(trade)

            # 检查现有持仓是否需要调整
            adjustment_trades = self._check_st_adjustments()
            executed_trades.extend(adjustment_trades)

        except Exception as e:
            logger.error(f"执行ST股票策略时出错: {e}")

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
            'st_type': signal.get('st_type', 'ST'),
            'risk_level': signal.get('risk_level', 'medium')
        }

        # 更新ST股票池
        self.st_stocks.add(symbol)

        # 记录交易
        self.log_trade(order)

        logger.info(f"执行ST股票买入: {symbol}, 价格: {price}, 数量: {shares}")

        return order

    def _check_st_adjustments(self) -> List[Dict[str, Any]]:
        """检查ST股票持仓调整"""
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
                    self.st_stocks.discard(symbol)

            elif current_price <= position['stop_loss']:
                # 止损
                trade = self._execute_sell_order(symbol, position, 'stop_loss')
                if trade:
                    adjustment_trades.append(trade)
                    del self.positions[symbol]
                    self.st_stocks.discard(symbol)

            # 检查ST股票特殊风险
            elif self._should_exit_st_position(symbol, position, current_price):
                trade = self._execute_sell_order(symbol, position, 'st_risk_exit')
        if trade:
            adjustment_trades.append(trade)
            del self.positions[symbol]
            self.st_stocks.discard(symbol)

        return adjustment_trades

    def _should_exit_st_position(self, symbol: str, position: Dict[str, Any], current_price: float) -> bool:
        """检查是否应该退出ST股票持仓"""
        # 检查是否仍为ST股票
        if not self._is_st_stock(symbol, pd.Series()):
            return True  # 已摘帽，可以考虑退出

        # 检查持仓时间（ST股票不宜长期持有）
        entry_time = position['entry_time']
        current_time = datetime.now()

        if (current_time - entry_time).days > 30:  # 超过30天
            return True

        # 检查风险等级变化
        risk_level = position.get('risk_level', 'medium')
        if risk_level == 'high':
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

        logger.info(f"执行ST股票卖出: {symbol}, 价格: {current_price}, 原因: {reason}")

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

        # 计算ST股票统计
        st_positions = sum(
            1 for symbol in self.positions.keys()
            if self._is_st_stock(symbol, pd.Series())
        )

        # 计算风险等级统计
        high_risk_count = sum(
            1 for pos in self.positions.values()
            if pos.get('risk_level') == 'high'
        )

        return {
            'total_positions': total_positions,
            'total_value': total_value,
            'st_positions': st_positions,
            'st_ratio': st_positions / total_positions if total_positions > 0 else 0,
            'high_risk_count': high_risk_count,
            'high_risk_ratio': high_risk_count / total_positions if total_positions > 0 else 0
        }
