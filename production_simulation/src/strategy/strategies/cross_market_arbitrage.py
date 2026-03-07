"""
跨市场套利策略
支持A股、港股、美股之间的套利交易
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum
from src.trading.execution.multi_market_adapter import (
    MarketType, MarketOrder, MultiMarketManager
)
from ...interfaces.strategy_interfaces import BaseTradingStrategy
from src.trading.execution.multi_market_adapter import MultiMarketManager
from src.trading.risk.risk_controller import RiskController
from src.trading.portfolio.portfolio_manager import PortfolioManager


logger = logging.getLogger(__name__)


class ArbitrageType(Enum):

    """套利类型枚举"""
    PAIR_TRADING = "pair_trading"      # 配对交易
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"  # 统计套利
    CONVERGENCE_TRADING = "convergence_trading"  # 收敛交易
    MOMENTUM_ARBITRAGE = "momentum_arbitrage"  # 动量套利


@dataclass
class ArbitrageOpportunity:

    """套利机会"""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    symbol_pair: Tuple[str, str]  # (symbol1, symbol2)
    market_pair: Tuple[MarketType, MarketType]  # (market1, market2)
    spread: float  # 价差
    z_score: float  # Z分数
    confidence: float  # 置信度
    expected_return: float  # 预期收益
    risk_score: float  # 风险评分
    created_at: datetime
    expiry_time: datetime

    def is_expired(self) -> bool:
        """检查是否过期"""
        return datetime.now() > self.expiry_time

    def is_valid(self) -> bool:
        """检查是否有效"""
        return not self.is_expired() and self.confidence > 0.7


@dataclass
class ArbitrageSignal:

    """套利信号"""
    signal_id: str
    opportunity: ArbitrageOpportunity
    action: str  # 'long_short', 'short_long', 'close'
    quantity: int
    price1: float
    price2: float
    expected_profit: float
    stop_loss: float
    take_profit: float
    created_at: datetime


class CrossMarketArbitrageStrategy(BaseTradingStrategy):

    """跨市场套利策略"""

    def __init__(self, config: Dict[str, Any]):

        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.multi_market_manager = MultiMarketManager(config)
        self.risk_controller = RiskController(config)
        self.portfolio_manager = PortfolioManager(config)
        self.min_spread_threshold = config.get('min_spread_threshold', 0.01)
        self.max_position_size = config.get('max_position_size', 100000)
        self.execution_timeout = config.get('execution_timeout', 30)

        # 套利参数
        self.min_spread = config.get('min_spread', 0.02)  # 最小价差
        self.max_position_size = config.get('max_position_size', 10000)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.05)
        self.take_profit_pct = config.get('take_profit_pct', 0.10)
        self.max_holding_days = config.get('max_holding_days', 30)

        # 统计套利参数
        self.lookback_period = config.get('lookback_period', 60)
        self.z_score_threshold = config.get('z_score_threshold', 2.0)
        self.mean_reversion_strength = config.get('mean_reversion_strength', 0.1)

        # 配对交易参数
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.cointegration_threshold = config.get('cointegration_threshold', 0.05)

        # 套利机会缓存
        self.opportunities: Dict[str, ArbitrageOpportunity] = {}
        self.active_positions: Dict[str, ArbitrageSignal] = {}

    def detect_arbitrage_opportunities(self, market_data: Dict[str, pd.DataFrame]) -> List[ArbitrageOpportunity]:
        """检测套利机会"""
        opportunities = []

        # 1. 配对交易机会检测
        pair_opportunities = self._detect_pair_trading_opportunities(market_data)
        opportunities.extend(pair_opportunities)

        # 2. 统计套利机会检测
        stat_opportunities = self._detect_statistical_arbitrage_opportunities(market_data)
        opportunities.extend(stat_opportunities)

        # 3. 收敛交易机会检测
        convergence_opportunities = self._detect_convergence_trading_opportunities(market_data)
        opportunities.extend(convergence_opportunities)

        # 4. 动量套利机会检测
        momentum_opportunities = self._detect_momentum_arbitrage_opportunities(market_data)
        opportunities.extend(momentum_opportunities)

        # 更新缓存
        for opp in opportunities:
            self.opportunities[opp.opportunity_id] = opp

        return opportunities

    def _detect_pair_trading_opportunities(self, market_data: Dict[str, pd.DataFrame]) -> List[ArbitrageOpportunity]:
        """检测配对交易机会"""
        opportunities = []

        # 定义配对股票
        pairs = [
            # A股 - 港股配对
            (('600519.SH', MarketType.A_SHARE), ('02318.HK', MarketType.H_SHARE)),  # 贵州茅台
            (('000858.SZ', MarketType.A_SHARE), ('02319.HK', MarketType.H_SHARE)),  # 五粮液
            # A股 - 美股配对
            (('600036.SH', MarketType.A_SHARE), ('CMB', MarketType.US_SHARE)),  # 招商银行
            # 港股 - 美股配对
            (('0700.HK', MarketType.H_SHARE), ('TCEHY', MarketType.US_SHARE)),  # 腾讯
        ]

        for (symbol1, market1), (symbol2, market2) in pairs:
            if symbol1 in market_data and symbol2 in market_data:
                df1 = market_data[symbol1]
                df2 = market_data[symbol2]

        if len(df1) > self.lookback_period and len(df2) > self.lookback_period:
            # 计算相关性
            correlation = df1['close'].corr(df2['close'])

        if correlation > self.correlation_threshold:
            # 计算价差
            spread = (df1['close'].iloc[-1] - df2['close'].iloc[-1]) / df2['close'].iloc[-1]

        if abs(spread) > self.min_spread:
            # 计算Z分数
            spread_series = (df1['close'] - df2['close']) / df2['close']
            z_score = (spread - spread_series.mean()) / spread_series.std()

        if abs(z_score) > self.z_score_threshold:
            opportunity = ArbitrageOpportunity(
                opportunity_id=f"pair_{symbol1}_{symbol2}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                arbitrage_type=ArbitrageType.PAIR_TRADING,
                symbol_pair=(symbol1, symbol2),
                market_pair=(market1, market2),
                spread=spread,
                z_score=z_score,
                confidence=min(abs(correlation), 0.95),
                expected_return=abs(spread) * 0.5,
                risk_score=1 - abs(correlation),
                created_at=datetime.now(),
                expiry_time=datetime.now() + timedelta(hours=1)
            )
            opportunities.append(opportunity)

        return opportunities

    def _detect_statistical_arbitrage_opportunities(self, market_data: Dict[str, pd.DataFrame]) -> List[ArbitrageOpportunity]:
        """检测统计套利机会"""
        opportunities = []

        # 分析单个股票的价格偏离
        for symbol, df in market_data.items():
            if len(df) > self.lookback_period:
                # 计算移动平均和标准差
                ma = df['close'].rolling(window=self.lookback_period).mean()
                std = df['close'].rolling(window=self.lookback_period).std()

                current_price = df['close'].iloc[-1]
                current_ma = ma.iloc[-1]
                current_std = std.iloc[-1]

        if not pd.isna(current_ma) and not pd.isna(current_std) and current_std > 0:
            # 计算Z分数
            z_score = (current_price - current_ma) / current_std

        if abs(z_score) > self.z_score_threshold:
            # 判断市场类型
            market_type = self._get_market_type(symbol)

            opportunity = ArbitrageOpportunity(
                opportunity_id=f"stat_{symbol}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                arbitrage_type=ArbitrageType.STATISTICAL_ARBITRAGE,
                symbol_pair=(symbol, symbol),
                market_pair=(market_type, market_type),
                spread=z_score,
                z_score=z_score,
                confidence=min(abs(z_score) / 3, 0.9),
                expected_return=abs(z_score) * self.mean_reversion_strength,
                risk_score=0.3,
                created_at=datetime.now(),
                expiry_time=datetime.now() + timedelta(hours=2)
            )
            opportunities.append(opportunity)

        return opportunities

    def _detect_convergence_trading_opportunities(self, market_data: Dict[str, pd.DataFrame]) -> List[ArbitrageOpportunity]:
        """检测收敛交易机会"""
        opportunities = []

        # 分析同行业股票的价格收敛
        industry_groups = {
            'banking': ['600036.SH', '000001.SZ', '601398.SH'],  # 银行股
            'technology': ['000002.SZ', '600519.SH', '000858.SZ'],  # 科技股
        }

        for industry, symbols in industry_groups.items():
            available_symbols = [s for s in symbols if s in market_data]

            if len(available_symbols) >= 2:
                # 计算行业平均价格
                prices = []
                for symbol in available_symbols:
                    if len(market_data[symbol]) > 0:
                        prices.append(market_data[symbol]['close'].iloc[-1])

                if len(prices) >= 2:
                    avg_price = np.mean(prices)

                    # 找出偏离最大的股票
                    deviations = [(abs(p - avg_price) / avg_price, symbol)
                                  for p, symbol in zip(prices, available_symbols)]
                    max_deviation, max_symbol = max(deviations)

                    if max_deviation > self.min_spread:
                        market_type = self._get_market_type(max_symbol)

                        opportunity = ArbitrageOpportunity(
                            opportunity_id=f"conv_{max_symbol}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                            arbitrage_type=ArbitrageType.CONVERGENCE_TRADING,
                            symbol_pair=(max_symbol, industry),
                            market_pair=(market_type, market_type),
                            spread=max_deviation,
                            z_score=max_deviation * 2,
                            confidence=0.8,
                            expected_return=max_deviation * 0.3,
                            risk_score=0.4,
                            created_at=datetime.now(),
                            expiry_time=datetime.now() + timedelta(hours=3)
                        )
                        opportunities.append(opportunity)

        return opportunities

    def _detect_momentum_arbitrage_opportunities(self, market_data: Dict[str, pd.DataFrame]) -> List[ArbitrageOpportunity]:
        """检测动量套利机会"""
        opportunities = []

        # 分析价格动量
        for symbol, df in market_data.items():
            if len(df) > self.lookback_period:
                # 计算动量指标
                returns = df['close'].pct_change()
                momentum = returns.rolling(window=20).mean().iloc[-1]
                volatility = returns.rolling(window=20).std().iloc[-1]

                if not pd.isna(momentum) and not pd.isna(volatility) and volatility > 0:
                    # 计算夏普比率
                    sharpe_ratio = momentum / volatility

                    if abs(sharpe_ratio) > 1.0:  # 动量足够强
                        market_type = self._get_market_type(symbol)

                        opportunity = ArbitrageOpportunity(
                            opportunity_id=f"mom_{symbol}_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                            arbitrage_type=ArbitrageType.MOMENTUM_ARBITRAGE,
                            symbol_pair=(symbol, symbol),
                            market_pair=(market_type, market_type),
                            spread=sharpe_ratio,
                            z_score=sharpe_ratio,
                            confidence=min(abs(sharpe_ratio) / 2, 0.85),
                            expected_return=abs(sharpe_ratio) * 0.02,
                            risk_score=0.5,
                            created_at=datetime.now(),
                            expiry_time=datetime.now() + timedelta(hours=4)
                        )
                        opportunities.append(opportunity)

        return opportunities

    def _get_market_type(self, symbol: str) -> MarketType:
        """根据股票代码判断市场类型"""
        if symbol.endswith('.SH') or symbol.endswith('.SZ'):
            return MarketType.A_SHARE
        elif symbol.endswith('.HK'):
            return MarketType.H_SHARE
        else:
            return MarketType.US_SHARE

    def generate_arbitrage_signals(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageSignal]:
        """生成套利信号"""
        signals = []

        for opportunity in opportunities:
            if not opportunity.is_valid():
                continue

            # 检查是否已有相同持仓
            if opportunity.opportunity_id in self.active_positions:
                continue

            # 生成信号
            signal = self._generate_signal_from_opportunity(opportunity)
        if signal:
            signals.append(signal)
            self.active_positions[opportunity.opportunity_id] = signal

        return signals

    def _generate_signal_from_opportunity(self, opportunity: ArbitrageOpportunity) -> Optional[ArbitrageSignal]:
        """从套利机会生成信号"""
        try:
            symbol1, symbol2 = opportunity.symbol_pair
            market1, market2 = opportunity.market_pair

            # 获取当前价格
            price1 = self._get_current_price(symbol1, market1)
            price2 = self._get_current_price(symbol2, market2)

            if price1 is None or price2 is None:
                return None

            # 确定交易方向
            if opportunity.arbitrage_type == ArbitrageType.PAIR_TRADING:
                if opportunity.z_score > 0:
                    action = 'short_long'  # 做空价高的，做多价低的
                else:
                    action = 'long_short'  # 做多价高的，做空价低的
            elif opportunity.arbitrage_type == ArbitrageType.STATISTICAL_ARBITRAGE:
                if opportunity.z_score > 0:
                    action = 'short_long'  # 价格过高，做空
                else:
                    action = 'long_short'  # 价格过低，做多
            else:
                # 其他套利类型
                action = 'long_short' if opportunity.spread > 0 else 'short_long'

            # 计算交易数量
            quantity = min(
                self.max_position_size,
                int(abs(opportunity.expected_return) * 10000)
            )

            # 计算止损和止盈
            stop_loss = opportunity.expected_return * (1 - self.stop_loss_pct)
            take_profit = opportunity.expected_return * (1 + self.take_profit_pct)

            signal = ArbitrageSignal(
                signal_id=f"signal_{opportunity.opportunity_id}",
                opportunity=opportunity,
                action=action,
                quantity=quantity,
                price1=price1,
                price2=price2,
                expected_profit=opportunity.expected_return,
                stop_loss=stop_loss,
                take_profit=take_profit,
                created_at=datetime.now()
            )

            return signal

        except Exception as e:
            self.logger.error(f"生成套利信号失败: {e}")
            return None

    def _get_current_price(self, symbol: str, market_type: MarketType) -> Optional[float]:
        """获取当前价格（模拟）"""
        # 这里应该从实际市场数据获取价格
        # 目前使用模拟价格
        base_prices = {
            '600519.SH': 1500.0,
            '000858.SZ': 200.0,
            '600036.SH': 50.0,
            '02318.HK': 1800.0,
            '02319.HK': 250.0,
            '0700.HK': 300.0,
            'CMB': 50.0,
            'TCEHY': 40.0,
            'AAPL': 150.0,
        }

        return base_prices.get(symbol, 100.0)

    def execute_arbitrage_signals(self, signals: List[ArbitrageSignal]) -> List[Dict[str, Any]]:
        """执行套利信号"""
        results = []

        for signal in signals:
            try:
                result = self._execute_single_signal(signal)
                results.append(result)

                if result['success']:
                    self.logger.info(f"套利信号执行成功: {signal.signal_id}")
                else:
                    self.logger.error(f"套利信号执行失败: {signal.signal_id}, 原因: {result.get('error')}")

            except Exception as e:
                self.logger.error(f"执行套利信号异常: {e}")
                results.append({
                    'success': False,
                    'signal_id': signal.signal_id,
                    'error': str(e)
                })

        return results

    def _execute_single_signal(self, signal: ArbitrageSignal) -> Dict[str, Any]:
        """执行单个套利信号"""
        opportunity = signal.opportunity
        symbol1, symbol2 = opportunity.symbol_pair
        market1, market2 = opportunity.market_pair

        orders = []

        if signal.action == 'long_short':
            # 做多第一个，做空第二个
            order1 = MarketOrder(
                order_id=f"order_{signal.signal_id}_1",
                symbol=symbol1,
                market_type=market1,
                order_type='market',
                side='buy',
                quantity=signal.quantity
            )

            order2 = MarketOrder(
                order_id=f"order_{signal.signal_id}_2",
                symbol=symbol2,
                market_type=market2,
                order_type='market',
                side='sell',
                quantity=signal.quantity
            )

            orders = [order1, order2]

        elif signal.action == 'short_long':
            # 做空第一个，做多第二个
            order1 = MarketOrder(
                order_id=f"order_{signal.signal_id}_1",
                symbol=symbol1,
                market_type=market1,
                order_type='market',
                side='sell',
                quantity=signal.quantity
            )

            order2 = MarketOrder(
                order_id=f"order_{signal.signal_id}_2",
                symbol=symbol2,
                market_type=market2,
                order_type='market',
                side='buy',
                quantity=signal.quantity
            )

            orders = [order1, order2]

        # 执行订单
        execution_results = []
        for order in orders:
            result = self.multi_market_manager.place_order(order)
            execution_results.append(result)

        # 检查执行结果
        all_success = all(r['success'] for r in execution_results)

        return {
            'success': all_success,
            'signal_id': signal.signal_id,
            'opportunity_id': opportunity.opportunity_id,
            'execution_results': execution_results,
            'expected_profit': signal.expected_profit,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit
        }

    def monitor_positions(self) -> List[Dict[str, Any]]:
        """监控持仓"""
        monitoring_results = []

        # 创建字典键的副本以避免迭代时修改字典
        signal_ids = list(self.active_positions.keys())

        for signal_id in signal_ids:
            signal = self.active_positions[signal_id]
            try:
                # 检查是否需要平仓
                should_close = self._check_close_conditions(signal)

                if should_close:
                    close_result = self._close_position(signal)
                    monitoring_results.append(close_result)

                    # 从活跃持仓中移除
                    del self.active_positions[signal_id]
                else:
                    # 继续监控
                    monitoring_results.append({
                        'signal_id': signal_id,
                        'action': 'monitor',
                        'status': 'active',
                        'current_pnl': self._calculate_current_pnl(signal)
                    })

            except Exception as e:
                self.logger.error(f"监控持仓异常: {e}")
                monitoring_results.append({
                    'signal_id': signal_id,
                    'action': 'error',
                    'error': str(e)
                })

        return monitoring_results

    def _check_close_conditions(self, signal: ArbitrageSignal) -> bool:
        """检查是否需要平仓"""
        # 检查止盈止损
        current_pnl = self._calculate_current_pnl(signal)

        if current_pnl <= signal.stop_loss or current_pnl >= signal.take_profit:
            return True

        # 检查持仓时间
        holding_days = (datetime.now() - signal.created_at).days
        if holding_days >= self.max_holding_days:
            return True

        # 检查套利机会是否消失
        if signal.opportunity.is_expired():
            return True

        return False

    def _calculate_current_pnl(self, signal: ArbitrageSignal) -> float:
        """计算当前盈亏（模拟）"""
        # 这里应该根据实际持仓计算盈亏
        # 目前使用模拟计算
        base_pnl = signal.expected_profit * 0.5  # 假设实现了一半的预期收益
        return base_pnl

    def _close_position(self, signal: ArbitrageSignal) -> Dict[str, Any]:
        """平仓"""
        try:
            opportunity = signal.opportunity
            symbol1, symbol2 = opportunity.symbol_pair
            market1, market2 = opportunity.market_pair

            # 生成平仓订单（反向操作）
            if signal.action == 'long_short':
                close_action = 'short_long'
            else:
                close_action = 'long_short'

            # 创建平仓订单
            orders = []
            if close_action == 'short_long':
                order1 = MarketOrder(
                    order_id=f"close_{signal.signal_id}_1",
                    symbol=symbol1,
                    market_type=market1,
                    order_type='market',
                    side='sell',
                    quantity=signal.quantity
                )

                order2 = MarketOrder(
                    order_id=f"close_{signal.signal_id}_2",
                    symbol=symbol2,
                    market_type=market2,
                    order_type='market',
                    side='buy',
                    quantity=signal.quantity
                )
            else:
                order1 = MarketOrder(
                    order_id=f"close_{signal.signal_id}_1",
                    symbol=symbol1,
                    market_type=market1,
                    order_type='market',
                    side='buy',
                    quantity=signal.quantity
                )

                order2 = MarketOrder(
                    order_id=f"close_{signal.signal_id}_2",
                    symbol=symbol2,
                    market_type=market2,
                    order_type='market',
                    side='sell',
                    quantity=signal.quantity
                )

            orders = [order1, order2]

            # 执行平仓订单
            execution_results = []
            for order in orders:
                result = self.multi_market_manager.place_order(order)
                execution_results.append(result)

            all_success = all(r['success'] for r in execution_results)

            return {
                'signal_id': signal.signal_id,
                'action': 'close',
                'success': all_success,
                'execution_results': execution_results,
                'final_pnl': self._calculate_current_pnl(signal)
            }

        except Exception as e:
            self.logger.error(f"平仓异常: {e}")
            return {
                'signal_id': signal.signal_id,
                'action': 'close',
                'success': False,
                'error': str(e)
            }

    def get_strategy_summary(self) -> Dict[str, Any]:
        """获取策略摘要"""
        return {
            'total_opportunities': len(self.opportunities),
            'active_positions': len(self.active_positions),
            'market_status': self.multi_market_manager.get_market_status(),
            'account_info': self.multi_market_manager.get_all_accounts_info(),
            'positions': self.multi_market_manager.get_all_positions()
        }
