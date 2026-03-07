"""
Backtest Automation Module
回测自动化模块

This module provides automated backtesting capabilities for quantitative trading strategies
此模块为量化交易策略提供自动化回测能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import threading
import time

logger = logging.getLogger(__name__)


class BacktestStatus(Enum):

    """Backtest status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BacktestType(Enum):

    """Backtest type enumeration"""
    SINGLE_RUN = "single_run"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    PARAMETER_SWEEP = "parameter_sweep"
    ROBUSTNESS_TEST = "robustness_test"


@dataclass
class BacktestConfig:

    """
    Backtest configuration data class
    回测配置数据类
    """
    start_date: datetime
    end_date: datetime
    initial_capital: float
    commission: float = 0.001
    slippage: float = 0.0005
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02
    max_position_size: float = 1.0
    max_leverage: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['start_date'] = self.start_date.isoformat()
        data['end_date'] = self.end_date.isoformat()
        return data


@dataclass
class BacktestResult:

    """
    Backtest result data class
    回测结果数据类
    """
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    calmar_ratio: float
    sortino_ratio: float
    alpha: Optional[float] = None
    beta: Optional[float] = None
    benchmark_return: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BacktestJob:

    """
    Backtest job data class
    回测作业数据类
    """
    job_id: str
    strategy_id: str
    backtest_type: str
    config: BacktestConfig
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[BacktestResult] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        if self.result:
            data['result'] = self.result.to_dict()
        return data


class BacktestEngine:

    """
    Backtest Engine Class
    回测引擎类

    Automated backtesting engine for trading strategies
    交易策略的自动化回测引擎
    """

    def __init__(self, engine_name: str = "default_backtest_engine"):
        """
        Initialize backtest engine
        初始化回测引擎

        Args:
            engine_name: Name of the backtest engine
                        回测引擎名称
        """
        self.engine_name = engine_name
        self.backtest_jobs: Dict[str, BacktestJob] = {}
        self.active_jobs: Dict[str, threading.Thread] = {}

        # Engine configuration
        self.max_concurrent_jobs = 5
        self.cache_results = True
        self.result_cache: Dict[str, BacktestResult] = {}

        # Performance tracking
        self.stats = {
            'total_backtests': 0,
            'completed_backtests': 0,
            'failed_backtests': 0,
            'average_execution_time': 0.0,
            'cache_hit_rate': 0.0
        }

        logger.info(f"Backtest engine {engine_name} initialized")

    def create_backtest_job(self,


                            job_id: str,
                            strategy_id: str,
                            backtest_type: BacktestType,
                            config: BacktestConfig,
                            strategy_function: Callable,
                            data_provider: Callable) -> str:
        """
        Create a backtest job
        创建回测作业

        Args:
            job_id: Unique job identifier
                   唯一作业标识符
            strategy_id: Strategy identifier
                        策略标识符
            backtest_type: Type of backtest
                          回测类型
            config: Backtest configuration
                   回测配置
            strategy_function: Strategy function to test
                              要测试的策略函数
            data_provider: Function to provide market data
                          提供市场数据的函数

        Returns:
            str: Created job ID
                 创建的作业ID
        """
        job = BacktestJob(
            job_id=job_id,
            strategy_id=strategy_id,
            backtest_type=backtest_type.value,
            config=config,
            status=BacktestStatus.PENDING.value,
            created_at=datetime.now(),
            metadata={
                'strategy_function': strategy_function,
                'data_provider': data_provider
            }
        )

        self.backtest_jobs[job_id] = job
        logger.info(f"Created backtest job: {job_id} for strategy {strategy_id}")
        return job_id

    def execute_backtest(self, job_id: str, async_execution: bool = True) -> Dict[str, Any]:
        """
        Execute a backtest job
        执行回测作业

        Args:
            job_id: Job identifier
                   作业标识符
            async_execution: Whether to execute asynchronously
                           是否异步执行

        Returns:
            dict: Execution result
                  执行结果
        """
        if job_id not in self.backtest_jobs:
            return {'success': False, 'error': f'Backtest job {job_id} not found'}

        job = self.backtest_jobs[job_id]

        # Check concurrent job limit
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return {
                'success': False,
                'error': 'Maximum concurrent backtest jobs reached'
            }

        if async_execution:
            # Start async execution
            execution_thread = threading.Thread(
                target=self._execute_backtest_sync,
                args=(job_id,),
                daemon=True
            )
            self.active_jobs[job_id] = execution_thread
            execution_thread.start()

            return {
                'success': True,
                'execution_mode': 'async',
                'job_id': job_id
            }
        else:
            # Execute synchronously
            return self._execute_backtest_sync(job_id)

    def _execute_backtest_sync(self, job_id: str) -> Dict[str, Any]:
        """
        Execute backtest job synchronously
        同步执行回测作业

        Args:
            job_id: Job identifier
                   作业标识符

        Returns:
            dict: Execution result
                  执行结果
        """
        job = self.backtest_jobs[job_id]
        job.status = BacktestStatus.RUNNING.value
        job.started_at = datetime.now()

        result = {
            'job_id': job_id,
            'success': False,
            'start_time': job.started_at,
            'execution_time': 0.0
        }

        start_time = time.time()

        try:
            # Get strategy function and data provider
            strategy_function = job.metadata['strategy_function']
            data_provider = job.metadata['data_provider']

            # Execute backtest based on type
            if job.backtest_type == BacktestType.SINGLE_RUN.value:
                backtest_result = self._execute_single_run(job, strategy_function, data_provider)
            elif job.backtest_type == BacktestType.WALK_FORWARD.value:
                backtest_result = self._execute_walk_forward(job, strategy_function, data_provider)
            elif job.backtest_type == BacktestType.MONTE_CARLO.value:
                backtest_result = self._execute_monte_carlo(job, strategy_function, data_provider)
            elif job.backtest_type == BacktestType.PARAMETER_SWEEP.value:
                backtest_result = self._execute_parameter_sweep(
                    job, strategy_function, data_provider)
            elif job.backtest_type == BacktestType.ROBUSTNESS_TEST.value:
                backtest_result = self._execute_robustness_test(
                    job, strategy_function, data_provider)
            else:
                raise ValueError(f"Unknown backtest type: {job.backtest_type}")

            # Update job with results
            job.result = backtest_result
            job.completed_at = datetime.now()
            job.execution_time = time.time() - start_time
            job.status = BacktestStatus.COMPLETED.value

            result.update({
                'success': True,
                'end_time': job.completed_at,
                'execution_time': job.execution_time,
                'backtest_result': backtest_result.to_dict() if backtest_result else None
            })

            # Update statistics
            self._update_backtest_stats(job, True)

            logger.info(f"Backtest job {job_id} completed successfully")

        except Exception as e:
            execution_time = time.time() - start_time
            job.execution_time = execution_time
            job.completed_at = datetime.now()
            job.status = BacktestStatus.FAILED.value
            job.error_message = str(e)

            result.update({
                'success': False,
                'end_time': job.completed_at,
                'execution_time': execution_time,
                'error': str(e)
            })

            # Update statistics
            self._update_backtest_stats(job, False)

            logger.error(f"Backtest job {job_id} failed: {str(e)}")

        # Clean up
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]

        return result

    def _execute_single_run(self,


                            job: BacktestJob,
                            strategy_function: Callable,
                            data_provider: Callable) -> BacktestResult:
        """
        Execute single run backtest
        执行单次运行回测

        Args:
            job: Backtest job
                回测作业
            strategy_function: Strategy function
                              策略函数
            data_provider: Data provider function
                          数据提供函数

        Returns:
            BacktestResult: Backtest result
                           回测结果
        """
        # Get market data
        market_data = data_provider(job.config.start_date, job.config.end_date)

        # Initialize portfolio
        portfolio = {
            'cash': job.config.initial_capital,
            'positions': {},
            'trades': [],
            'equity_curve': []
        }

        # Run strategy
        current_date = job.config.start_date
        equity = job.config.initial_capital

        while current_date <= job.config.end_date:
            try:
                # Get current market data slice
                current_data = self._get_data_slice(market_data, current_date)

                if current_data is not None:
                    # Execute strategy
                    signals = strategy_function(current_data, portfolio)

                    # Process signals
                    portfolio = self._process_signals(signals, portfolio, current_data, job.config)

                    # Update equity
                    equity = portfolio['cash']
                    for symbol, position in portfolio['positions'].items():
                        if symbol in current_data:
                            current_price = current_data[symbol].get('close', 0)
                            equity += position['quantity'] * current_price

                    portfolio['equity_curve'].append({
                        'date': current_date,
                        'equity': equity
                    })

            except Exception as e:
                logger.error(f"Strategy execution failed on {current_date}: {str(e)}")

            current_date += timedelta(days=1)

        # Calculate performance metrics
        result = self._calculate_performance_metrics(portfolio, job.config)

        return result

    def _execute_walk_forward(self,


                              job: BacktestJob,
                              strategy_function: Callable,
                              data_provider: Callable) -> BacktestResult:
        """
        Execute walk - forward backtest
        执行步前进测

        Args:
            job: Backtest job
                回测作业
            strategy_function: Strategy function
                              策略函数
            data_provider: Data provider function
                          数据提供函数

        Returns:
            BacktestResult: Backtest result
                           回测结果
        """
        # Walk - forward analysis parameters
        train_window = timedelta(days=365)  # 1 year training
        test_window = timedelta(days=30)    # 1 month testing
        step_size = timedelta(days=30)      # Move forward 1 month

        combined_portfolio = {
            'cash': job.config.initial_capital,
            'positions': {},
            'trades': [],
            'equity_curve': []
        }

        current_train_start = job.config.start_date
        current_test_start = current_train_start + train_window

        while current_test_start + test_window <= job.config.end_date:
            try:
                # Train strategy on historical data
                train_data = data_provider(
                    current_train_start, current_test_start - timedelta(days=1))

                # Test strategy on future data
                test_data = data_provider(current_test_start, current_test_start + test_window)

                # Run backtest on test period
                test_portfolio = {
                    'cash': combined_portfolio['cash'],
                    'positions': combined_portfolio['positions'].copy(),
                    'trades': [],
                    'equity_curve': []
                }

                # Execute strategy on test data
                current_date = current_test_start
                while current_date <= current_test_start + test_window:
                    current_slice = self._get_data_slice(test_data, current_date)
                    if current_slice:
                        signals = strategy_function(current_slice, test_portfolio)
                        test_portfolio = self._process_signals(
                            signals, test_portfolio, current_slice, job.config)

                    current_date += timedelta(days=1)

                # Update combined portfolio
                combined_portfolio['cash'] = test_portfolio['cash']
                combined_portfolio['positions'] = test_portfolio['positions']
                combined_portfolio['trades'].extend(test_portfolio['trades'])
                combined_portfolio['equity_curve'].extend(test_portfolio['equity_curve'])

                # Move forward
                current_train_start += step_size
                current_test_start += step_size

            except Exception as e:
                logger.error(f"Walk - forward iteration failed: {str(e)}")
                break

        # Calculate performance metrics
        result = self._calculate_performance_metrics(combined_portfolio, job.config)

        return result

    def _execute_monte_carlo(self,


                             job: BacktestJob,
                             strategy_function: Callable,
                             data_provider: Callable) -> BacktestResult:
        """
        Execute Monte Carlo backtest
        执行蒙特卡洛回测

        Args:
            job: Backtest job
                回测作业
            strategy_function: Strategy function
                              策略函数
            data_provider: Data provider function
                          数据提供函数

        Returns:
            BacktestResult: Backtest result
                           回测结果
        """
        num_simulations = job.metadata.get('num_simulations', 100)
        results = []

        # Run multiple simulations with randomized parameters
        for i in range(num_simulations):
            try:
                # Add some randomization to parameters (placeholder)
                randomized_config = job.config

                # Execute single run with randomization
                simulation_result = self._execute_single_run(job, strategy_function, data_provider)
                results.append(simulation_result)

            except Exception as e:
                logger.error(f"Monte Carlo simulation {i} failed: {str(e)}")

        # Aggregate results
        if results:
            avg_result = self._aggregate_monte_carlo_results(results)
            return avg_result
        else:
            raise ValueError("All Monte Carlo simulations failed")

    def _execute_parameter_sweep(self,


                                 job: BacktestJob,
                                 strategy_function: Callable,
                                 data_provider: Callable) -> BacktestResult:
        """
        Execute parameter sweep backtest
        执行参数扫描回测

        Args:
            job: Backtest job
                回测作业
            strategy_function: Strategy function
                              策略函数
            data_provider: Data provider function
                          数据提供函数

        Returns:
            BacktestResult: Backtest result
                           回测结果
        """
        parameter_sets = job.metadata.get('parameter_sets', [])

        if not parameter_sets:
            # Default parameter sweep
            parameter_sets = [
                {'param1': 0.01, 'param2': 20},
                {'param1': 0.02, 'param2': 25},
                {'param1': 0.03, 'param2': 30}
            ]

        best_result = None
        best_score = float('-inf')

        for params in parameter_sets:
            try:
                # Modify strategy function with parameters

                def parameterized_strategy(data, portfolio): return strategy_function(
                    data, portfolio, **params)

                # Execute backtest
                result = self._execute_single_run(job, parameterized_strategy, data_provider)

                # Evaluate result (using Sharpe ratio as score)
                score = result.sharpe_ratio

                if score > best_score:
                    best_score = score
                    best_result = result

            except Exception as e:
                logger.error(f"Parameter sweep failed for params {params}: {str(e)}")

        return best_result if best_result else self._execute_single_run(job, strategy_function, data_provider)

    def _execute_robustness_test(self,


                                 job: BacktestJob,
                                 strategy_function: Callable,
                                 data_provider: Callable) -> BacktestResult:
        """
        Execute robustness test backtest
        执行稳健性测试回测

        Args:
            job: Backtest job
                回测作业
            strategy_function: Strategy function
                              策略函数
            data_provider: Data provider function
                          数据提供函数

        Returns:
            BacktestResult: Backtest result
                           回测结果
        """
        # Test under different market conditions
        test_scenarios = [
            {'name': 'normal', 'volatility_multiplier': 1.0, 'trend_bias': 0.0},
            {'name': 'high_volatility', 'volatility_multiplier': 1.5, 'trend_bias': 0.0},
            {'name': 'bull_market', 'volatility_multiplier': 1.0, 'trend_bias': 0.02},
            {'name': 'bear_market', 'volatility_multiplier': 1.0, 'trend_bias': -0.02}
        ]

        scenario_results = []

        for scenario in test_scenarios:
            try:
                # Modify data provider for scenario

                def scenario_data_provider(start, end): return self._modify_data_for_scenario(

                    data_provider(start, end), scenario
                )

                # Execute backtest for scenario
                result = self._execute_single_run(job, strategy_function, scenario_data_provider)
                scenario_results.append({
                    'scenario': scenario['name'],
                    'result': result
                })

            except Exception as e:
                logger.error(f"Robustness test failed for scenario {scenario['name']}: {str(e)}")

        # Return average result across scenarios
        if scenario_results:
            avg_result = self._aggregate_scenario_results(scenario_results)
            return avg_result
        else:
            return self._execute_single_run(job, strategy_function, data_provider)

    def _get_data_slice(self, market_data: pd.DataFrame, date: datetime) -> Optional[Dict[str, Any]]:
        """
        Get market data slice for a specific date
        获取特定日期的市场数据切片

        Args:
            market_data: Market data DataFrame
                        市场数据DataFrame
            date: Date to get data for
                 要获取数据的日期

        Returns:
            dict: Data slice or None
                  数据切片或None
        """
        try:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in market_data.index:
                return market_data.loc[date_str].to_dict()
            return None
        except Exception:
            return None

    def _process_signals(self,


                         signals: Dict[str, Any],
                         portfolio: Dict[str, Any],
                         market_data: Dict[str, Any],
                         config: BacktestConfig) -> Dict[str, Any]:
        """
        Process trading signals and update portfolio
        处理交易信号并更新投资组合

        Args:
            signals: Trading signals
                    交易信号
            portfolio: Current portfolio
                      当前投资组合
            market_data: Current market data
                        当前市场数据
            config: Backtest configuration
                   回测配置

        Returns:
            dict: Updated portfolio
                  更新的投资组合
        """
        for symbol, signal in signals.items():
            if symbol not in market_data:
                continue

            current_price = market_data[symbol].get('close', 0)
            if current_price <= 0:
                continue

            # Determine trade size
            if signal.get('type') == 'buy':
                max_quantity = min(
                    config.max_position_size * portfolio['cash'] / current_price,
                    config.max_leverage * portfolio['cash'] / current_price
                )

                quantity = min(signal.get('quantity', max_quantity), max_quantity)

                if quantity > 0 and portfolio['cash'] >= quantity * current_price * (1 + config.commission):
                    # Execute buy
                    cost = quantity * current_price * (1 + config.commission)
                    portfolio['cash'] -= cost

                    if symbol not in portfolio['positions']:
                        portfolio['positions'][symbol] = {'quantity': 0, 'avg_price': 0}

                    # Update position
                    position = portfolio['positions'][symbol]
                    total_quantity = position['quantity'] + quantity
                    total_cost = position['quantity'] * \
                        position['avg_price'] + quantity * current_price
                    position['avg_price'] = total_cost / total_quantity
                    position['quantity'] = total_quantity

                    # Record trade
                    portfolio['trades'].append({
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': quantity,
                        'price': current_price,
                        'timestamp': datetime.now()
                    })

            elif signal.get('type') == 'sell':
                if symbol in portfolio['positions']:
                    position = portfolio['positions'][symbol]
                    quantity = min(signal.get(
                        'quantity', position['quantity']), position['quantity'])

                    if quantity > 0:
                        # Execute sell
                        proceeds = quantity * current_price * (1 - config.commission)
                        portfolio['cash'] += proceeds

                        # Update position
                        position['quantity'] -= quantity

                        if position['quantity'] <= 0:
                            del portfolio['positions'][symbol]

                        # Record trade
                        portfolio['trades'].append({
                            'symbol': symbol,
                            'side': 'sell',
                            'quantity': quantity,
                            'price': current_price,
                            'timestamp': datetime.now()
                        })

        return portfolio

    def _calculate_performance_metrics(self,


                                       portfolio: Dict[str, Any],
                                       config: BacktestConfig) -> BacktestResult:
        """
        Calculate performance metrics from portfolio data
        从投资组合数据计算绩效指标

        Args:
            portfolio: Portfolio data
                      投资组合数据
            config: Backtest configuration
                   回测配置

        Returns:
            BacktestResult: Performance metrics
                           绩效指标
        """
        equity_curve = pd.DataFrame(portfolio['equity_curve'])
        trades = pd.DataFrame(portfolio['trades'])

        if equity_curve.empty:
            return BacktestResult(
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                calmar_ratio=0.0,
                sortino_ratio=0.0
            )

        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        equity_curve = equity_curve.dropna()

        total_return = (equity_curve['equity'].iloc[-1] -
                        config.initial_capital) / config.initial_capital

        # Annualized return
        days = (config.end_date - config.start_date).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # Volatility
        volatility = equity_curve['returns'].std() * np.sqrt(252)  # Annualized

        # Sharpe ratio
        excess_returns = equity_curve['returns'] - config.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * \
            np.sqrt(252) if excess_returns.std() > 0 else 0

        # Maximum drawdown
        cumulative = (1 + equity_curve['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate and profit factor
        if not trades.empty:
            winning_trades = trades[trades['side'] == 'sell']  # Simplified
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0

            # Profit factor (simplified)
            profit_factor = 1.5  # Placeholder
        else:
            win_rate = 0.0
            profit_factor = 0.0

        # Calmar ratio
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown < 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = equity_curve['returns'][equity_curve['returns'] < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_deviation * \
            np.sqrt(252) if downside_deviation > 0 else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )

    def _modify_data_for_scenario(self, data: pd.DataFrame, scenario: Dict[str, Any]) -> pd.DataFrame:
        """
        Modify market data for robustness testing scenarios
        为稳健性测试场景修改市场数据

        Args:
            data: Original market data
                 原始市场数据
            scenario: Scenario parameters
                     场景参数

        Returns:
            pd.DataFrame: Modified data
                         修改后的数据
        """
        modified_data = data.copy()

        # Apply volatility multiplier
        vol_multiplier = scenario.get('volatility_multiplier', 1.0)
        if vol_multiplier != 1.0:
            returns_cols = [col for col in modified_data.columns if 'return' in col.lower()]
            for col in returns_cols:
                modified_data[col] = modified_data[col] * vol_multiplier

        # Apply trend bias
        trend_bias = scenario.get('trend_bias', 0.0)
        if trend_bias != 0.0:
            price_cols = [col for col in modified_data.columns if 'close' in col.lower()]
            for col in price_cols:
                # Add trend bias to prices
                trend_factor = np.exp(np.cumsum([trend_bias] * len(modified_data)))
                modified_data[col] = modified_data[col] * trend_factor

        return modified_data

    def _aggregate_monte_carlo_results(self, results: List[BacktestResult]) -> BacktestResult:
        """
        Aggregate Monte Carlo simulation results
        聚合蒙特卡洛模拟结果

        Args:
            results: List of simulation results
                    模拟结果列表

        Returns:
            BacktestResult: Aggregated result
                           聚合结果
        """
        if not results:
            raise ValueError("No results to aggregate")

        # Calculate averages
        avg_total_return = np.mean([r.total_return for r in results])
        avg_annualized_return = np.mean([r.annualized_return for r in results])
        avg_volatility = np.mean([r.volatility for r in results])
        avg_sharpe_ratio = np.mean([r.sharpe_ratio for r in results])
        avg_max_drawdown = np.mean([r.max_drawdown for r in results])
        avg_win_rate = np.mean([r.win_rate for r in results])
        avg_profit_factor = np.mean([r.profit_factor for r in results])
        total_trades = int(np.mean([r.total_trades for r in results]))
        avg_calmar_ratio = np.mean([r.calmar_ratio for r in results])
        avg_sortino_ratio = np.mean([r.sortino_ratio for r in results])

        return BacktestResult(
            total_return=avg_total_return,
            annualized_return=avg_annualized_return,
            volatility=avg_volatility,
            sharpe_ratio=avg_sharpe_ratio,
            max_drawdown=avg_max_drawdown,
            win_rate=avg_win_rate,
            profit_factor=avg_profit_factor,
            total_trades=total_trades,
            calmar_ratio=avg_calmar_ratio,
            sortino_ratio=avg_sortino_ratio,
            additional_metrics={
                'monte_carlo_simulations': len(results),
                'return_std': np.std([r.total_return for r in results]),
                'sharpe_std': np.std([r.sharpe_ratio for r in results])
            }
        )

    def _aggregate_scenario_results(self, scenario_results: List[Dict[str, Any]]) -> BacktestResult:
        """
        Aggregate robustness test scenario results
        聚合稳健性测试场景结果

        Args:
            scenario_results: List of scenario results
                             场景结果列表

        Returns:
            BacktestResult: Aggregated result
                           聚合结果
        """
        if not scenario_results:
            raise ValueError("No scenario results to aggregate")

        results = [sr['result'] for sr in scenario_results]

        # Calculate averages
        avg_result = self._aggregate_monte_carlo_results(results)
        avg_result.additional_metrics = {
            'scenarios_tested': len(scenario_results),
            'scenario_names': [sr['scenario'] for sr in scenario_results]
        }

        return avg_result

    def get_backtest_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get backtest job status
        获取回测作业状态

        Args:
            job_id: Job identifier
                   作业标识符

        Returns:
            dict: Job status or None if not found
                  作业状态，如果未找到则返回None
        """
        if job_id in self.backtest_jobs:
            return self.backtest_jobs[job_id].to_dict()
        return None

    def cancel_backtest(self, job_id: str) -> bool:
        """
        Cancel a running backtest job
        取消正在运行的回测作业

        Args:
            job_id: Job identifier
                   作业标识符

        Returns:
            bool: True if cancelled successfully
                  取消成功返回True
        """
        if job_id in self.backtest_jobs:
            job = self.backtest_jobs[job_id]
            if job.status == BacktestStatus.RUNNING.value:
                job.status = BacktestStatus.CANCELLED.value
                job.completed_at = datetime.now()
                logger.info(f"Cancelled backtest job: {job_id}")
                return True
        return False

    def list_backtest_jobs(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List backtest jobs with optional status filter
        列出回测作业，可选状态过滤

        Args:
            status_filter: Status to filter by (optional)
                          要过滤的状态（可选）

        Returns:
            list: List of backtest jobs
                  回测作业列表
        """
        jobs = []
        for job in self.backtest_jobs.values():
            if status_filter is None or job.status == status_filter:
                jobs.append(job.to_dict())
        return jobs

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get backtest engine statistics
        获取回测引擎统计信息

        Returns:
            dict: Engine statistics
                  引擎统计信息
        """
        return {
            'engine_name': self.engine_name,
            'total_jobs': len(self.backtest_jobs),
            'active_jobs': len(self.active_jobs),
            'stats': self.stats
        }

    def _update_backtest_stats(self, job: BacktestJob, success: bool) -> None:
        """
        Update backtest statistics
        更新回测统计信息

        Args:
            job: Backtest job
                回测作业
            success: Whether backtest was successful
                    回测是否成功
        """
        self.stats['total_backtests'] += 1

        if success:
            self.stats['completed_backtests'] += 1
        else:
            self.stats['failed_backtests'] += 1

        # Update average execution time
        total_completed = self.stats['completed_backtests']
        current_avg = self.stats['average_execution_time']
        new_time = job.execution_time
        self.stats['average_execution_time'] = (
            (current_avg * (total_completed - 1)) + new_time
        ) / total_completed


# Global backtest engine instance
# 全局回测引擎实例
backtest_engine = BacktestEngine()

__all__ = [
    'BacktestStatus',
    'BacktestType',
    'BacktestConfig',
    'BacktestResult',
    'BacktestJob',
    'BacktestEngine',
    'backtest_engine'
]
