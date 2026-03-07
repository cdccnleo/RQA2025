#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
回测服务实现
Backtest Service Implementation

提供完整的策略回测功能，支持多种回测模式和性能分析。
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from strategy.interfaces.backtest_interfaces import (
    IBacktestService, IBacktestEngine, IBacktestPersistence,
    BacktestConfig, BacktestResult, BacktestMode, BacktestStatus
)
from ..interfaces.strategy_interfaces import IStrategyService, StrategyConfig
from core.integration.business_adapters import get_unified_adapter_factory

logger = logging.getLogger(__name__)


class BacktestService(IBacktestService):

    """
    回测服务
    Backtest Service

    提供完整的策略回测功能，包括数据加载、交易执行、性能分析等。
    """

    def __init__(self, strategy_service: IStrategyService,


                 backtest_engine: IBacktestEngine,
                 persistence: IBacktestPersistence):
        """
        初始化回测服务

        Args:
            strategy_service: 策略服务实例
            backtest_engine: 回测引擎实例
            persistence: 持久化服务实例
        """
        self.strategy_service = strategy_service
        self.backtest_engine = backtest_engine
        self.persistence = persistence
        self.adapter_factory = get_unified_adapter_factory()

        # 运行中的回测任务
        self.running_backtests: Dict[str, asyncio.Task] = {}

        # 线程池用于并行处理
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info("回测服务初始化完成")

    async def create_backtest(self, config: BacktestConfig) -> str:
        """
        创建回测任务

        Args:
            config: 回测配置

        Returns:
            str: 回测任务ID
        """
        try:
            # 验证配置
            if not await self._validate_backtest_config(config):
                raise ValueError(f"回测配置验证失败: {config.backtest_id}")

            # 保存配置
            success = self.persistence.save_backtest_config(config)
            if not success:
                raise RuntimeError(f"回测配置保存失败: {config.backtest_id}")

            # 发布事件
            await self._publish_event("backtest_created", {
                "backtest_id": config.backtest_id,
                "strategy_id": config.strategy_id,
                "mode": config.mode.value,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"回测任务创建成功: {config.backtest_id}")
            return config.backtest_id

        except Exception as e:
            logger.error(f"回测任务创建失败: {e}")
            raise

    async def run_backtest(self, backtest_id: str) -> BacktestResult:
        """
        运行回测

        Args:
            backtest_id: 回测ID

        Returns:
            BacktestResult: 回测结果
        """
        try:
            # 获取配置
            config = self.persistence.load_backtest_config(backtest_id)
            if not config:
                raise ValueError(f"回测配置不存在: {backtest_id}")

            # 创建异步任务
            task = asyncio.create_task(self._execute_backtest(config))
            self.running_backtests[backtest_id] = task

            # 等待执行完成
            result = await task

            # 保存结果
            success = self.persistence.save_backtest_result(result)
            if not success:
                logger.warning(f"回测结果保存失败: {backtest_id}")

            # 清理任务
            del self.running_backtests[backtest_id]

            # 发布事件
            await self._publish_event("backtest_completed", {
                "backtest_id": backtest_id,
                "status": result.status.value,
                "execution_time": result.execution_time,
                "timestamp": datetime.now().isoformat()
            })

            logger.info(f"回测执行完成: {backtest_id}")
            return result

        except Exception as e:
            logger.error(f"回测执行失败: {backtest_id}, 错误: {e}")

            # 创建失败结果
            failed_result = BacktestResult(
                backtest_id=backtest_id,
                strategy_id=config.strategy_id if 'config' in locals() else "",
                returns=pd.Series(),
                positions=pd.DataFrame(),
                trades=pd.DataFrame(),
                metrics={},
                risk_metrics={},
                status=BacktestStatus.FAILED,
                execution_time=0.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=str(e)
            )

            # 清理任务
            if backtest_id in self.running_backtests:
                del self.running_backtests[backtest_id]

            return failed_result

    async def _execute_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        执行回测逻辑

        Args:
            config: 回测配置

        Returns:
            BacktestResult: 回测结果
        """
        start_time = datetime.now()

        try:
            # 加载历史数据
            market_data = await self._load_market_data(config)
            if not market_data:
                raise RuntimeError("无法加载市场数据")

            # 执行回测
            if config.mode == BacktestMode.SINGLE:
                result = await self._run_single_backtest(config, market_data)
            elif config.mode == BacktestMode.MULTI_STRATEGY:
                result = await self._run_multi_strategy_backtest(config, market_data)
            else:
                raise ValueError(f"不支持的回测模式: {config.mode}")

            result.execution_time = (datetime.now() - start_time).total_seconds()
            result.start_time = start_time
            result.end_time = datetime.now()

            return result

        except Exception as e:
            logger.error(f"回测执行异常: {e}")
            raise

    async def _load_market_data(self, config: BacktestConfig) -> Dict[str, pd.DataFrame]:
        """
        加载市场数据，如果缺失则自动采集
        
        量化交易系统要求：不使用模拟数据，必须使用真实数据

        Args:
            config: 回测配置

        Returns:
            Dict[str, pd.DataFrame]: 市场数据字典
        """
        try:
            # 确定股票代码列表
            symbols = []
            if hasattr(config, 'symbols') and config.symbols:
                symbols = config.symbols
            elif hasattr(config, 'benchmark_symbol') and config.benchmark_symbol:
                symbols = [config.benchmark_symbol]
            else:
                # 如果没有指定，尝试从策略配置获取
                if hasattr(config, 'strategy_id') and config.strategy_id:
                    strategy_config = self.strategy_service.get_strategy(config.strategy_id)
                    if strategy_config and hasattr(strategy_config, 'symbols'):
                        symbols = strategy_config.symbols
                
                # 如果仍然没有，使用默认（但不生成模拟数据）
                if not symbols:
                    logger.warning("未指定股票代码，无法加载市场数据")
                    return {}
            
            # 解析日期
            from datetime import datetime
            if isinstance(config.start_date, str):
                start_date = datetime.strptime(config.start_date, "%Y-%m-%d")
            else:
                start_date = config.start_date
                
            if isinstance(config.end_date, str):
                end_date = datetime.strptime(config.end_date, "%Y-%m-%d")
            else:
                end_date = config.end_date
            
            # 1. 先尝试从数据库加载数据
            market_data = await self._load_from_storage(symbols, start_date, end_date)
            
            # 2. 检查数据完整性
            missing_data_info = self._check_data_completeness(market_data, symbols, start_date, end_date)
            
            # 3. 如果有缺失数据，自动触发数据采集
            if missing_data_info['has_missing']:
                logger.info(f"检测到缺失数据，自动触发数据采集: {missing_data_info}")
                
                # 获取数据源配置
                source_config = await self._get_data_source_config()
                
                if source_config:
                    # 触发数据采集
                    await self._trigger_data_collection(
                        symbols=symbols,
                        start_date=missing_data_info['missing_start'],
                        end_date=missing_data_info['missing_end'],
                        source_config=source_config
                    )
                    
                    # 重新加载数据
                    market_data = await self._load_from_storage(symbols, start_date, end_date)
                else:
                    logger.warning("无法获取数据源配置，跳过自动采集")
            
            # 4. 验证最终数据
            if not market_data or all(df.empty for df in market_data.values()):
                logger.error("无法加载市场数据，回测无法继续")
                raise RuntimeError("无法加载市场数据：数据库无数据且自动采集失败")
            
            return market_data

        except Exception as e:
            logger.error(f"市场数据加载失败: {e}")
            raise RuntimeError(f"市场数据加载失败: {e}") from e
    
    async def _load_from_storage(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        从存储（PostgreSQL）加载市场数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[str, pd.DataFrame]: 市场数据字典
        """
        try:
            # 尝试从PostgreSQL加载数据
            from src.gateway.web.postgresql_persistence import query_stock_data_from_postgresql
            
            # 获取数据源ID（默认使用akshare_stock）
            source_id = "akshare_stock"
            
            market_data = query_stock_data_from_postgresql(
                source_id=source_id,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            if market_data:
                logger.info(f"从PostgreSQL加载了 {len(market_data)} 个股票的数据")
            else:
                logger.warning("PostgreSQL中无数据")
            
            return market_data
            
        except ImportError:
            logger.warning("postgresql_persistence模块不可用，无法从数据库加载数据")
            return {}
        except Exception as e:
            logger.warning(f"从存储加载数据失败: {e}")
            return {}
    
    def _check_data_completeness(
        self, 
        market_data: Dict[str, pd.DataFrame],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        检查数据完整性
        
        Args:
            market_data: 已加载的市场数据
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            缺失数据信息字典
        """
        from datetime import timedelta
        
        missing_info = {
            'has_missing': False,
            'missing_symbols': [],
            'missing_start': start_date,
            'missing_end': end_date,
            'missing_ranges': {}
        }
        
        # 计算期望的日期范围
        expected_dates = pd.date_range(start_date.date(), end_date.date(), freq='D')
        expected_count = len(expected_dates)
        
        for symbol in symbols:
            df = market_data.get(symbol, pd.DataFrame())
            
            if df.empty:
                # 完全没有数据
                missing_info['has_missing'] = True
                missing_info['missing_symbols'].append(symbol)
                missing_info['missing_ranges'][symbol] = [(start_date, end_date)]
                logger.warning(f"股票 {symbol} 完全缺失数据")
            else:
                # 检查日期范围完整性
                actual_dates = set(df.index.date) if hasattr(df.index, 'date') else set()
                expected_dates_set = set(expected_dates.date)
                missing_dates = expected_dates_set - actual_dates
                
                if missing_dates:
                    missing_info['has_missing'] = True
                    missing_info['missing_symbols'].append(symbol)
                    
                    # 计算缺失的日期范围
                    from src.gateway.web.postgresql_persistence import calculate_missing_date_ranges
                    missing_ranges = calculate_missing_date_ranges(
                        start_date,
                        end_date,
                        actual_dates
                    )
                    missing_info['missing_ranges'][symbol] = missing_ranges
                    
                    missing_count = len(missing_dates)
                    completeness = (expected_count - missing_count) / expected_count * 100
                    logger.warning(f"股票 {symbol} 数据不完整: {missing_count}/{expected_count} 缺失 ({completeness:.1f}% 完整)")
        
        return missing_info
    
    async def _get_data_source_config(self) -> Optional[Dict[str, Any]]:
        """
        获取数据源配置
        
        Returns:
            数据源配置字典，如果不存在则返回None
        """
        try:
            # 尝试从数据源配置管理器获取akshare_stock配置
            from src.gateway.web.data_source_config_manager import get_data_source_config_manager
            
            config_manager = get_data_source_config_manager()
            source_config = config_manager.get_data_source("akshare_stock")
            
            if source_config:
                logger.info(f"获取到数据源配置: {source_config.get('id')}")
                return source_config
            else:
                logger.warning("未找到akshare_stock数据源配置")
                return None
                
        except Exception as e:
            logger.error(f"获取数据源配置失败: {e}")
            return None
    
    async def _trigger_data_collection(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        source_config: Dict[str, Any]
    ) -> bool:
        """
        触发数据采集
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            source_config: 数据源配置
            
        Returns:
            是否成功
        """
        try:
            logger.info(f"自动触发数据采集: {len(symbols)} 个股票, {start_date.date()} 到 {end_date.date()}")
            
            # 调用数据采集API
            from src.gateway.web.api import collect_data_via_data_layer
            
            request_data = {
                "symbols": symbols,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "data_type": "daily",
                "incremental": True  # 使用增量采集模式
            }
            
            result = await collect_data_via_data_layer(source_config, request_data)
            
            if result and result.get("data"):
                data_count = len(result.get("data", []))
                logger.info(f"数据采集成功: {data_count} 条记录")
                return True
            else:
                logger.warning("数据采集返回空结果")
                return False
                
        except Exception as e:
            logger.error(f"触发数据采集失败: {e}")
            return False

    async def _run_single_backtest(self, config: BacktestConfig,
                                   market_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        运行单策略回测

        Args:
            config: 回测配置
            market_data: 市场数据

        Returns:
            BacktestResult: 回测结果
        """
        # 获取策略配置
        strategy_config = self.strategy_service.get_strategy(config.strategy_id)
        if not strategy_config:
            raise ValueError(f"策略不存在: {config.strategy_id}")

        # 执行策略
        execution_result = await self._execute_strategy_simulation(
            strategy_config, market_data, config
        )

        # 计算指标
        metrics = self._calculate_backtest_metrics(
            execution_result, market_data, config
        )

        # 创建交易记录
        trades_df = pd.DataFrame(execution_result.get('trades', []))

        # 创建持仓记录
        positions_df = pd.DataFrame(execution_result.get('positions', []))

        return BacktestResult(
            backtest_id=config.backtest_id,
            strategy_id=config.strategy_id,
            returns=pd.Series(execution_result.get('returns', [])),
            positions=positions_df,
            trades=trades_df,
            metrics=metrics,
            risk_metrics=self._calculate_risk_metrics(execution_result, market_data),
            status=BacktestStatus.COMPLETED,
            execution_time=0.0,  # 将在上级函数中设置
            start_time=config.start_date,
            end_time=config.end_date
        )

    async def _run_multi_strategy_backtest(self, config: BacktestConfig,
                                           market_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        运行多策略回测

        Args:
            config: 回测配置
            market_data: 市场数据

        Returns:
            BacktestResult: 回测结果
        """
        # 这里简化实现，实际应该支持多个策略的并行回测
        return await self._run_single_backtest(config, market_data)

    async def _execute_strategy_simulation(self, strategy_config: StrategyConfig,
                                           market_data: Dict[str, pd.DataFrame],
                                           backtest_config: BacktestConfig) -> Dict[str, Any]:
        """
        执行策略模拟

        Args:
            strategy_config: 策略配置
            market_data: 市场数据
            backtest_config: 回测配置

        Returns:
            Dict[str, Any]: 模拟结果
        """
        # 这里简化实现，实际应该调用策略服务执行模拟
        # 暂时返回模拟结果

        returns = []
        trades = []
        positions = []

        # 简单的模拟逻辑
        capital = backtest_config.initial_capital
        position = 0
        entry_price = 0.0

        for symbol, df in market_data.items():
            for idx, row in df.iterrows():
                price = row['close']

                # 简单的交易逻辑示例
                if position == 0 and price < df['close'].iloc[0] * 0.95:  # 价格下跌5 % 时买入
                    position = int(capital * 0.1 / price)  # 10 % 的资金买入
                    entry_price = price
                    capital -= position * price

                    trades.append({
                        'timestamp': idx,
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': position,
                        'price': price,
                        'commission': position * price * backtest_config.commission
                    })

                elif position > 0 and price > entry_price * 1.05:  # 盈利5 % 时卖出
                    capital += position * price
                    pnl = (price - entry_price) * position

                    trades.append({
                        'timestamp': idx,
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': position,
                        'price': price,
                        'pnl': pnl,
                        'commission': position * price * backtest_config.commission
                    })

                    returns.append(pnl / (entry_price * position))
                    position = 0
                    entry_price = 0.0

                # 记录持仓
        if position > 0:
            positions.append({
                'timestamp': idx,
                'symbol': symbol,
                'quantity': position,
                'price': price,
                'value': position * price
            })

        return {
            'returns': returns,
            'trades': trades,
            'positions': positions,
            'final_capital': capital
        }

    def _calculate_backtest_metrics(self, execution_result: Dict[str, Any],


                                    market_data: Dict[str, pd.DataFrame],
                                    config: BacktestConfig) -> Dict[str, float]:
        """
        计算回测指标

        Args:
            execution_result: 执行结果
            market_data: 市场数据
            config: 回测配置

        Returns:
            Dict[str, float]: 回测指标
        """
        returns = execution_result.get('returns', [])
        trades = execution_result.get('trades', [])

        if not returns:
            return {}

        # 基础指标
        total_return = sum(returns) if returns else 0.0
        annual_return = total_return * 252 / len(returns) if returns else 0.0

        # 胜率
        winning_trades = [r for r in returns if r > 0]
        win_rate = len(winning_trades) / len(returns) if returns else 0.0

        # 盈亏比
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
        losing_trades = [r for r in returns if r < 0]
        avg_loss = abs(sum(losing_trades) / len(losing_trades)) if losing_trades else 0.0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # 最大回撤 (简化计算)
        cumulative_returns = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # 夏普比率 (简化计算)
        volatility = np.std(returns) * np.sqrt(252) if returns else 0.0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_trade_return': total_return / len(returns) if returns else 0.0
        }

    def _calculate_risk_metrics(self, execution_result: Dict[str, Any],


                                market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        计算风险指标

        Args:
            execution_result: 执行结果
            market_data: 市场数据

        Returns:
            Dict[str, float]: 风险指标
        """
        returns = execution_result.get('returns', [])

        if not returns:
            return {}

        # VaR (95 % 置信度)
        returns_array = np.array(returns)
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0.0

        # CVaR (条件VaR)
        cvar_95 = np.mean(returns_array[returns_array <= var_95]) if len(returns_array) > 0 else 0.0

        # 最大回撤 (这里可以更精确计算)
        max_drawdown = 0.0

        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_drawdown,
            'volatility': np.std(returns_array) if len(returns_array) > 0 else 0.0
        }

    async def _validate_backtest_config(self, config: BacktestConfig) -> bool:
        """
        验证回测配置

        Args:
            config: 回测配置

        Returns:
            bool: 配置是否有效
        """
        # 基本验证
        if not config.backtest_id or not config.strategy_id:
            return False

        if config.start_date >= config.end_date:
            return False

        if config.initial_capital <= 0:
            return False

        # 验证策略是否存在
        strategy = self.strategy_service.get_strategy(config.strategy_id)
        if not strategy:
            return False

        return True

    async def _publish_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        发布事件

        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        try:
            event_bus_adapter = self.adapter_factory.get_adapter("event_bus")
            await event_bus_adapter.publish_event({
                "event_type": f"backtest_{event_type}",
                "data": event_data,
                "source": "backtest_service",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"事件发布异常: {e}")

    def get_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """
        获取回测结果

        Args:
            backtest_id: 回测ID

        Returns:
            Optional[BacktestResult]: 回测结果
        """
        return self.persistence.load_backtest_result(backtest_id)

    def cancel_backtest(self, backtest_id: str) -> bool:
        """
        取消回测

        Args:
            backtest_id: 回测ID

        Returns:
            bool: 取消是否成功
        """
        if backtest_id in self.running_backtests:
            task = self.running_backtests[backtest_id]
            task.cancel()
            del self.running_backtests[backtest_id]
            logger.info(f"回测任务已取消: {backtest_id}")
            return True

        return False

    def get_backtest_status(self, backtest_id: str) -> BacktestStatus:
        """
        获取回测状态

        Args:
            backtest_id: 回测ID

        Returns:
            BacktestStatus: 回测状态
        """
        if backtest_id in self.running_backtests:
            return BacktestStatus.RUNNING

        result = self.persistence.load_backtest_result(backtest_id)
        if result:
            return result.status

        config = self.persistence.load_backtest_config(backtest_id)
        if config:
            return BacktestStatus.CREATED

        return BacktestStatus.CREATED  # 默认状态

    def list_backtests(self, strategy_id: Optional[str] = None,


                       status: Optional[BacktestStatus] = None) -> List[BacktestConfig]:
        """
        列出回测任务

        Args:
            strategy_id: 策略ID过滤器
            status: 状态过滤器

        Returns:
            List[BacktestConfig]: 回测配置列表
        """
        # 这里简化实现，实际应该从持久化层查询
        # 暂时返回空列表
        return []


# 导出类
__all__ = [
    'BacktestService'
]

