import logging
"""
策略模拟器

from src.engine.logging.unified_logger import get_unified_logger
提供模拟交易环境用于策略测试，包括：
- 历史数据回测
- 实时模拟交易
- 风险控制模拟
- 交易成本计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from .visual_editor import VisualStrategyEditor

logger = logging.getLogger(__name__)


class SimulationMode(Enum):

    """模拟模式"""
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    LIVE_SIMULATION = "live_simulation"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"
    SCENARIO_ANALYSIS = "scenario_analysis"
    REAL_TIME = "real_time"


@dataclass
class SimulationConfig:

    """模拟配置"""
    mode: SimulationMode
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission_rate: float = 0.0003  # 手续费率
    slippage: float = 0.0001  # 滑点
    risk_free_rate: float = 0.03  # 无风险利率
    # 蒙特卡洛模拟参数
    monte_carlo_iterations: int = 1000
    monte_carlo_scenarios: List[Dict] = None
    # 压力测试参数
    stress_test_scenarios: List[Dict] = None
    stress_test_parameters: Dict = None
    # 场景分析参数
    scenario_configs: List[Dict] = None
    # 实时模拟参数
    real_time_interval: int = 1  # 秒
    real_time_duration: int = 3600  # 秒


@dataclass
class TradeRecord:

    """交易记录"""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL
    quantity: float
    price: float
    commission: float
    slippage: float
    total_cost: float


@dataclass
class Position:

    """持仓信息"""
    symbol: str
    quantity: float
    avg_price: float
    current_value: float
    unrealized_pnl: float


@dataclass
class SimulationResult:

    """模拟结果"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: List[TradeRecord]
    positions: List[Position]
    equity_curve: pd.DataFrame
    performance_metrics: Dict


class StrategySimulator:

    """策略模拟器"""

    def __init__(self):

        self.current_positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.equity_history: List[Tuple[datetime, float]] = []

    def simulate(self, strategy: VisualStrategyEditor,
                 market_data: pd.DataFrame,
                 config: SimulationConfig) -> SimulationResult:
        """执行策略模拟

        Args:
            strategy: 策略编辑器
            market_data: 市场数据
            config: 模拟配置

        Returns:
            SimulationResult: 模拟结果
        """
        try:
            logger.info(f"开始策略模拟: {config.mode.value}")

            if config.mode == SimulationMode.BACKTEST:
                return self._backtest_simulation(strategy, market_data, config)
            elif config.mode == SimulationMode.PAPER_TRADING:
                return self._paper_trading_simulation(strategy, market_data, config)
            elif config.mode == SimulationMode.LIVE_SIMULATION:
                return self._live_simulation(strategy, market_data, config)
            elif config.mode == SimulationMode.MONTE_CARLO:
                return self._monte_carlo_simulation(strategy, market_data, config)
            elif config.mode == SimulationMode.STRESS_TEST:
                return self._stress_test_simulation(strategy, market_data, config)
            elif config.mode == SimulationMode.SCENARIO_ANALYSIS:
                return self._scenario_analysis_simulation(strategy, market_data, config)
            elif config.mode == SimulationMode.REAL_TIME:
                return self._real_time_simulation(strategy, market_data, config)
            else:
                raise ValueError(f"不支持的模拟模式: {config.mode}")

        except Exception as e:
            logger.error(f"策略模拟失败: {e}")
            raise

    def _initialize_simulation(self, config: SimulationConfig):
        """初始化模拟环境"""
        self.current_capital = config.initial_capital
        self.initial_capital = config.initial_capital
        self.current_positions = {}
        self.trade_history = []
        self.equity_history = []

        logger.info(f"初始化模拟环境，初始资金: {config.initial_capital}")

    def _update_market_data(self, market_data: pd.DataFrame, timestamp: datetime) -> pd.DataFrame:
        """更新市场数据"""
        # 获取到当前时间点的所有数据
        current_data = market_data.loc[:timestamp].copy()
        return current_data

    def _execute_strategy(self, strategy: VisualStrategyEditor,
                          market_data: pd.DataFrame,
                          timestamp: datetime) -> List[Dict]:
        """执行策略生成信号"""
        signals = []

        try:
            # 这里应该根据策略节点生成信号
            # 简化实现，实际应该遍历策略节点
            for node in strategy.get_all_nodes():
                if node.node_type.value == 'trade':
                    # 生成交易信号
                    signal = self._generate_trading_signal(node, market_data, timestamp)
                    if signal:
                        signals.append(signal)
        except Exception as e:
            logger.warning(f"策略执行失败: {e}")

        return signals

    def _generate_trading_signal(self, node, market_data: pd.DataFrame,
                                 timestamp: datetime) -> Optional[Dict]:
        """生成交易信号"""
        try:
            # 简化的信号生成逻辑
            # 实际应该根据节点类型和参数生成具体信号

            if not market_data.empty:
                current_price = market_data.iloc[-1]['close']

                # 简单的移动平均策略示例
                if 'period' in node.params:
                    period = node.params['period']
                    if len(market_data) >= period:
                        ma = market_data['close'].rolling(period).mean().iloc[-1]

                        if current_price > ma * 1.02:  # 买入信号
                            return {
                                'symbol': market_data.index.name or 'UNKNOWN',
                                'action': 'BUY',
                                'quantity': 100,
                                'price': current_price,
                                'confidence': 0.8
                            }
                        elif current_price < ma * 0.98:  # 卖出信号
                            return {
                                'symbol': market_data.index.name or 'UNKNOWN',
                                'action': 'SELL',
                                'quantity': 100,
                                'price': current_price,
                                'confidence': 0.8
                            }

            return None

        except Exception as e:
            logger.warning(f"信号生成失败: {e}")
            return None

    def _execute_trades(self, signals: List[Dict], market_data: pd.DataFrame,
                        timestamp: datetime, config: SimulationConfig):
        """执行交易"""
        for signal in signals:
            try:
                symbol = signal['symbol']
                action = signal['action']
                quantity = signal['quantity']
                price = signal['price']

                # 计算交易成本
                commission = price * quantity * config.commission_rate
                slippage_cost = price * quantity * config.slippage
                total_cost = price * quantity + commission + slippage_cost

                # 检查资金是否足够
                if action == 'BUY' and total_cost > self.current_capital:
                    logger.warning(f"资金不足，无法买入 {symbol}")
                    continue

                # 执行交易
                if action == 'BUY':
                    self._execute_buy(symbol, quantity, price, commission, slippage_cost, timestamp)
                elif action == 'SELL':
                    self._execute_sell(symbol, quantity, price, commission,
                                       slippage_cost, timestamp)

            except Exception as e:
                logger.warning(f"交易执行失败: {e}")

    def _execute_buy(self, symbol: str, quantity: float, price: float,
                     commission: float, slippage: float, timestamp: datetime):
        """执行买入"""
        # 更新资金
        total_cost = price * quantity + commission + slippage
        self.current_capital -= total_cost

        # 更新持仓
        if symbol in self.current_positions:
            # 已有持仓，计算新的平均价格
            current_pos = self.current_positions[symbol]
            total_quantity = current_pos.quantity + quantity
            total_value = current_pos.quantity * current_pos.avg_price + quantity * price
            new_avg_price = total_value / total_quantity

            self.current_positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                avg_price=new_avg_price,
                current_value=total_quantity * price,
                unrealized_pnl=total_quantity * (price - new_avg_price)
            )
        else:
            # 新建持仓
            self.current_positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_value=quantity * price,
                unrealized_pnl=0
            )

        # 记录交易
        trade_record = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            action='BUY',
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            total_cost=total_cost
        )
        self.trade_history.append(trade_record)

        logger.info(f"买入 {symbol}: {quantity} @ {price}")

    def _execute_sell(self, symbol: str, quantity: float, price: float,
                      commission: float, slippage: float, timestamp: datetime):
        """执行卖出"""
        if symbol not in self.current_positions:
            logger.warning(f"没有 {symbol} 的持仓")
            return

        current_pos = self.current_positions[symbol]
        if current_pos.quantity < quantity:
            logger.warning(f"{symbol} 持仓不足")
            return

        # 计算收益
        revenue = price * quantity - commission - slippage
        self.current_capital += revenue

        # 更新持仓
        remaining_quantity = current_pos.quantity - quantity
        if remaining_quantity > 0:
            self.current_positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining_quantity,
                avg_price=current_pos.avg_price,
                current_value=remaining_quantity * price,
                unrealized_pnl=remaining_quantity * (price - current_pos.avg_price)
            )
        else:
            # 完全卖出
            del self.current_positions[symbol]

        # 记录交易
        trade_record = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            action='SELL',
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            total_cost=revenue
        )
        self.trade_history.append(trade_record)

        logger.info(f"卖出 {symbol}: {quantity} @ {price}")

    def _update_positions(self, market_data: pd.DataFrame, timestamp: datetime):
        """更新持仓价值"""
        for symbol, position in self.current_positions.items():
            if not market_data.empty:
                current_price = market_data.iloc[-1]['close']
                position.current_value = position.quantity * current_price
                position.unrealized_pnl = position.quantity * (current_price - position.avg_price)

    def _record_equity(self, timestamp: datetime):
        """记录权益"""
        total_equity = self.current_capital
        for position in self.current_positions.values():
            total_equity += position.current_value

        self.equity_history.append((timestamp, total_equity))

    def _calculate_results(self, config: SimulationConfig) -> SimulationResult:
        """计算结果"""
        if not self.equity_history:
            raise ValueError("没有权益历史数据")

        # 创建权益曲线
        equity_df = pd.DataFrame(self.equity_history, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)

        # 计算收益率
        initial_equity = self.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_equity) / initial_equity

        # 计算年化收益率
        days = (config.end_date - config.start_date).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # 计算夏普比率
        daily_returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() - config.risk_free_rate / 365) / \
            daily_returns.std() if daily_returns.std() > 0 else 0

        # 计算最大回撤
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 计算胜率
        if self.trade_history:
            profitable_trades = [t for t in self.trade_history if t.total_cost > 0]
            win_rate = len(profitable_trades) / len(self.trade_history) if self.trade_history else 0
        else:
            win_rate = 0

        # 计算盈亏比
        if self.trade_history:
            profits = [t.total_cost for t in self.trade_history if t.total_cost > 0]
            losses = [abs(t.total_cost) for t in self.trade_history if t.total_cost < 0]

            total_profit = sum(profits) if profits else 0
            total_loss = sum(losses) if losses else 0

            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        else:
            profit_factor = 0

        # 性能指标
        performance_metrics = {
            'total_trades': len(self.trade_history),
            'avg_trade_return': np.mean([t.total_cost for t in self.trade_history]) if self.trade_history else 0,
            'volatility': daily_returns.std() if len(daily_returns) > 0 else 0,
            'var_95': daily_returns.quantile(0.05) if len(daily_returns) > 0 else 0,
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }

        return SimulationResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades=self.trade_history,
            positions=list(self.current_positions.values()),
            equity_curve=equity_df,
            performance_metrics=performance_metrics
        )

    def get_position_summary(self) -> Dict:
        """获取持仓摘要"""
        summary = {
            'total_positions': len(self.current_positions),
            'total_value': sum(pos.current_value for pos in self.current_positions.values()),
            'total_unrealized_pnl': sum(pos.unrealized_pnl for pos in self.current_positions.values()),
            'positions': {}
        }

        for symbol, position in self.current_positions.items():
            summary['positions'][symbol] = {
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'current_value': position.current_value,
                'unrealized_pnl': position.unrealized_pnl,
                'return_pct': (position.current_value / (position.quantity * position.avg_price) - 1) * 100
            }

        return summary

    def _backtest_simulation(self, strategy: VisualStrategyEditor,
                             market_data: pd.DataFrame,
                             config: SimulationConfig) -> SimulationResult:
        """回测模拟"""
        logger.info("开始回测模拟")

        # 初始化
        self._initialize_simulation(config)

        # 按时间顺序处理数据
        for timestamp, row in market_data.iterrows():
            if timestamp < config.start_date or timestamp > config.end_date:
                continue

            # 更新市场数据
            current_data = self._update_market_data(market_data, timestamp)

            # 执行策略
            signals = self._execute_strategy(strategy, current_data, timestamp)

            # 执行交易
            self._execute_trades(signals, current_data, timestamp, config)

            # 更新持仓
            self._update_positions(current_data, timestamp)

            # 记录权益
            self._record_equity(timestamp)

        # 计算结果
        result = self._calculate_results(config)

        logger.info(f"回测模拟完成，总收益率: {result.total_return:.2%}")
        return result

    def _paper_trading_simulation(self, strategy: VisualStrategyEditor,


                                  market_data: pd.DataFrame,
                                  config: SimulationConfig) -> SimulationResult:
        """纸面交易模拟"""
        logger.info("开始纸面交易模拟")

        # 初始化
        self._initialize_simulation(config)

        # 模拟实时交易环境
        for timestamp, row in market_data.iterrows():
            if timestamp < config.start_date or timestamp > config.end_date:
                continue

            # 更新市场数据
            current_data = self._update_market_data(market_data, timestamp)

            # 执行策略
            signals = self._execute_strategy(strategy, current_data, timestamp)

            # 执行交易（考虑延迟和滑点）
            self._execute_trades_with_delay(signals, current_data, timestamp, config)

            # 更新持仓
            self._update_positions(current_data, timestamp)

            # 记录权益
            self._record_equity(timestamp)

        # 计算结果
        result = self._calculate_results(config)

        logger.info(f"纸面交易模拟完成，总收益率: {result.total_return:.2%}")
        return result

    def _live_simulation(self, strategy: VisualStrategyEditor,


                         market_data: pd.DataFrame,
                         config: SimulationConfig) -> SimulationResult:
        """实盘模拟"""
        logger.info("开始实盘模拟")

        # 初始化
        self._initialize_simulation(config)

        # 模拟实盘交易环境
        for timestamp, row in market_data.iterrows():
            if timestamp < config.start_date or timestamp > config.end_date:
                continue

            # 更新市场数据
            current_data = self._update_market_data(market_data, timestamp)

            # 执行策略
            signals = self._execute_strategy(strategy, current_data, timestamp)

            # 执行交易（考虑实盘限制）
            self._execute_trades_with_real_constraints(signals, current_data, timestamp, config)

            # 更新持仓
            self._update_positions(current_data, timestamp)

            # 记录权益
            self._record_equity(timestamp)

        # 计算结果
        result = self._calculate_results(config)

        logger.info(f"实盘模拟完成，总收益率: {result.total_return:.2%}")
        return result

    def _monte_carlo_simulation(self, strategy: VisualStrategyEditor,


                                market_data: pd.DataFrame,
                                config: SimulationConfig) -> SimulationResult:
        """蒙特卡洛模拟"""
        logger.info("开始蒙特卡洛模拟")

        results = []

        for i in range(config.monte_carlo_iterations):
            # 生成随机场景
            scenario_data = self._generate_monte_carlo_scenario(market_data, config)

            # 执行单次模拟
            result = self._backtest_simulation(strategy, scenario_data, config)
            results.append(result)

            if i % 100 == 0:
                logger.info(f"蒙特卡洛模拟进度: {i + 1}/{config.monte_carlo_iterations}")

        # 聚合结果
        aggregated_result = self._aggregate_monte_carlo_results(results, config)

        logger.info(f"蒙特卡洛模拟完成，平均收益率: {aggregated_result.total_return:.2%}")
        return aggregated_result

    def _stress_test_simulation(self, strategy: VisualStrategyEditor,
                                market_data: pd.DataFrame,
                                config: SimulationConfig) -> SimulationResult:
        """压力测试模拟"""
        logger.info("开始压力测试模拟")

        results = []

        if config.stress_test_scenarios:
            for scenario in config.stress_test_scenarios:
                # 应用压力测试场景
                stress_data = self._apply_stress_scenario(market_data, scenario)

                # 执行模拟
                result = self._backtest_simulation(strategy, stress_data, config)
                results.append(result)

        # 聚合结果
        aggregated_result = self._aggregate_stress_test_results(results, config)

        logger.info(f"压力测试模拟完成")
        return aggregated_result

    def _scenario_analysis_simulation(self, strategy: VisualStrategyEditor,
                                      market_data: pd.DataFrame,
                                      config: SimulationConfig) -> SimulationResult:
        """场景分析模拟"""
        logger.info("开始场景分析模拟")

        results = []

        if config.scenario_configs:
            for scenario_config in config.scenario_configs:
                # 应用场景配置
                scenario_data = self._apply_scenario_config(market_data, scenario_config)

                # 执行模拟
                result = self._backtest_simulation(strategy, scenario_data, config)
                results.append(result)

        # 聚合结果
        aggregated_result = self._aggregate_scenario_results(results, config)

        logger.info(f"场景分析模拟完成")
        return aggregated_result

    def _real_time_simulation(self, strategy: VisualStrategyEditor,
                              market_data: pd.DataFrame,
                              config: SimulationConfig) -> SimulationResult:
        """实时模拟"""
        logger.info("开始实时模拟")

        # 初始化
        self._initialize_simulation(config)

        # 模拟实时数据流
        for timestamp, row in market_data.iterrows():
            if timestamp < config.start_date or timestamp > config.end_date:
                continue

            # 更新市场数据
            current_data = self._update_market_data(market_data, timestamp)

            # 执行策略
            signals = self._execute_strategy(strategy, current_data, timestamp)

            # 执行交易（实时模式）
            self._execute_trades_real_time(signals, current_data, timestamp, config)

            # 更新持仓
            self._update_positions(current_data, timestamp)

            # 记录权益
            self._record_equity(timestamp)

        # 计算结果
        result = self._calculate_results(config)

        logger.info(f"实时模拟完成，总收益率: {result.total_return:.2%}")
        return result

    def _execute_trades_with_delay(self, signals: List[Dict], market_data: pd.DataFrame,
                                   timestamp: datetime, config: SimulationConfig):
        """执行交易（考虑延迟）"""
        for signal in signals:
            try:
                # 添加交易延迟
                execution_delay = np.secrets.uniform(0.1, 0.5)  # 100 - 500ms延迟

                symbol = signal['symbol']
                action = signal['action']
                quantity = signal['quantity']
                price = signal['price']

                # 计算交易成本（考虑延迟导致的滑点）
                commission = price * quantity * config.commission_rate
                slippage = price * quantity * config.slippage * (1 + execution_delay)

                if action == 'BUY':
                    self._execute_buy(symbol, quantity, price, commission, slippage, timestamp)
                elif action == 'SELL':
                    self._execute_sell(symbol, quantity, price, commission, slippage, timestamp)

            except Exception as e:
                logger.warning(f"延迟交易执行失败: {e}")

    def _execute_trades_with_real_constraints(self, signals: List[Dict], market_data: pd.DataFrame,
                                              timestamp: datetime, config: SimulationConfig):
        """执行交易（实盘限制）"""
        for signal in signals:
            try:
                symbol = signal['symbol']
                action = signal['action']
                quantity = signal['quantity']
                price = signal['price']

                # 实盘限制检查
                if not self._check_real_trading_constraints(signal, config):
                    continue

                # 计算交易成本
                commission = price * quantity * config.commission_rate
                slippage = price * quantity * config.slippage

                if action == 'BUY':
                    self._execute_buy(symbol, quantity, price, commission, slippage, timestamp)
                elif action == 'SELL':
                    self._execute_sell(symbol, quantity, price, commission, slippage, timestamp)

            except Exception as e:
                logger.warning(f"实盘交易执行失败: {e}")

    def _execute_trades_real_time(self, signals: List[Dict], market_data: pd.DataFrame,
                                  timestamp: datetime, config: SimulationConfig):
        """执行交易（实时模式）"""
        for signal in signals:
            try:
                symbol = signal['symbol']
                action = signal['action']
                quantity = signal['quantity']
                price = signal['price']

                # 实时模式下的交易执行
                commission = price * quantity * config.commission_rate
                slippage = price * quantity * config.slippage

                if action == 'BUY':
                    self._execute_buy(symbol, quantity, price, commission, slippage, timestamp)
                elif action == 'SELL':
                    self._execute_sell(symbol, quantity, price, commission, slippage, timestamp)

            except Exception as e:
                logger.warning(f"实时交易执行失败: {e}")

    def _generate_monte_carlo_scenario(self, market_data: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
        """生成蒙特卡洛场景"""
        # 基于历史数据生成随机场景
        scenario_data = market_data.copy()

        # 添加随机波动
        for column in ['open', 'high', 'low', 'close']:
            if column in scenario_data.columns:
                noise = np.secrets.normal(0, 0.01, len(scenario_data))
                scenario_data[column] = scenario_data[column] * (1 + noise)

        return scenario_data

    def _aggregate_monte_carlo_results(self, results: List[SimulationResult], config: SimulationConfig) -> SimulationResult:
        """聚合蒙特卡洛结果"""
        # 计算平均结果
        total_returns = [r.total_return for r in results]
        annualized_returns = [r.annualized_return for r in results]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        max_drawdowns = [r.max_drawdown for r in results]

        aggregated_result = SimulationResult(
            total_return=np.mean(total_returns),
            annualized_return=np.mean(annualized_returns),
            sharpe_ratio=np.mean(sharpe_ratios),
            max_drawdown=np.mean(max_drawdowns),
            win_rate=np.mean([r.win_rate for r in results]),
            profit_factor=np.mean([r.profit_factor for r in results]),
            trades=results[0].trades,  # 使用第一次模拟的交易记录
            positions=results[0].positions,
            equity_curve=results[0].equity_curve,
            performance_metrics={
                'monte_carlo_results': results,
                'confidence_intervals': {
                    'total_return': [np.percentile(total_returns, 5), np.percentile(total_returns, 95)],
                    'sharpe_ratio': [np.percentile(sharpe_ratios, 5), np.percentile(sharpe_ratios, 95)]
                }
            }
        )

        return aggregated_result

    def _apply_stress_scenario(self, market_data: pd.DataFrame, scenario: Dict) -> pd.DataFrame:
        """应用压力测试场景"""
        stress_data = market_data.copy()

        # 应用压力场景
        if 'price_shock' in scenario:
            shock = scenario['price_shock']
            stress_data['close'] = stress_data['close'] * (1 + shock)
            stress_data['open'] = stress_data['open'] * (1 + shock)
            stress_data['high'] = stress_data['high'] * (1 + shock)
            stress_data['low'] = stress_data['low'] * (1 + shock)

        if 'volatility_increase' in scenario:
            vol_increase = scenario['volatility_increase']
            # 增加波动率
            stress_data['close'] = stress_data['close'] * \
                (1 + np.secrets.normal(0, vol_increase, len(stress_data)))

        return stress_data

    def _aggregate_stress_test_results(self, results: List[SimulationResult], config: SimulationConfig) -> SimulationResult:
        """聚合压力测试结果"""
        # 计算最坏情况结果
        worst_result = min(results, key=lambda x: x.total_return)

        worst_result.performance_metrics.update({
            'stress_test_results': results,
            'worst_case_scenario': worst_result.total_return,
            'scenario_count': len(results)
        })

        return worst_result

    def _apply_scenario_config(self, market_data: pd.DataFrame, scenario_config: Dict) -> pd.DataFrame:
        """应用场景配置"""
        scenario_data = market_data.copy()

        # 应用场景配置
        if 'market_regime' in scenario_config:
            regime = scenario_config['market_regime']
            if regime == 'bull_market':
                # 牛市场景
                scenario_data['close'] = scenario_data['close'] * \
                    (1 + np.secrets.uniform(0.001, 0.005, len(scenario_data)))
            elif regime == 'bear_market':
                # 熊市场景
                scenario_data['close'] = scenario_data['close'] * \
                    (1 - np.secrets.uniform(0.001, 0.005, len(scenario_data)))

        return scenario_data

    def _aggregate_scenario_results(self, results: List[SimulationResult], config: SimulationConfig) -> SimulationResult:
        """聚合场景分析结果"""
        # 计算平均结果
        avg_result = SimulationResult(
            total_return=np.mean([r.total_return for r in results]),
            annualized_return=np.mean([r.annualized_return for r in results]),
            sharpe_ratio=np.mean([r.sharpe_ratio for r in results]),
            max_drawdown=np.mean([r.max_drawdown for r in results]),
            win_rate=np.mean([r.win_rate for r in results]),
            profit_factor=np.mean([r.profit_factor for r in results]),
            trades=results[0].trades,
            positions=results[0].positions,
            equity_curve=results[0].equity_curve,
            performance_metrics={
                'scenario_results': results,
                'scenario_count': len(results)
            }
        )

        return avg_result

    def _check_real_trading_constraints(self, signal: Dict, config: SimulationConfig) -> bool:
        """检查实盘交易限制"""
        # 检查资金是否足够
        if signal['action'] == 'BUY':
            cost = signal['price'] * signal['quantity']
        if cost > self.current_capital:
            return False

        # 检查持仓限制
        symbol = signal['symbol']
        if symbol in self.current_positions:
            position = self.current_positions[symbol]
        if signal['action'] == 'SELL' and position.quantity < signal['quantity']:
            return False

        return True
