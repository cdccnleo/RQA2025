#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
回测引擎 - 性能优化版

提供高性能的回测执行引擎，支持并行处理、内存优化和缓存机制。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import gc
import psutil
import time
from dataclasses import dataclass
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:

    """回测配置类"""
    initial_capital: float = 1000000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    benchmark: str = None
    risk_free_rate: float = 0.03
    max_workers: int = None
    enable_cache: bool = True
    cache_dir: str = "cache / backtest_results"
    memory_limit_gb: float = 4.0
    enable_parallel: bool = True


class OptimizedBacktestEngine:

    """
    性能优化版回测引擎

    特性：
    - 并行处理多个策略
    - 内存优化和垃圾回收
    - 结果缓存机制
    - 性能监控
    - 大数据量处理优化
    """

    def __init__(self, config: BacktestConfig = None):
        """
        初始化优化版回测引擎

        Args:
            config: 回测配置
        """
        # 处理配置参数
        if config is not None:
            if isinstance(config, BacktestConfig):
                self.config = config
            elif isinstance(config, dict):
                # 从字典创建配置对象
                self.config = BacktestConfig(
                    initial_capital=config.get('initial_capital', 1000000.0),
                    commission_rate=config.get('commission_rate', 0.0003),
                    slippage_rate=config.get('slippage_rate', 0.0001),
                    benchmark=config.get('benchmark'),
                    risk_free_rate=config.get('risk_free_rate', 0.03),
                    max_workers=config.get('max_workers'),
                    enable_cache=config.get('enable_cache', True),
                    cache_dir=config.get('cache_dir', "cache / backtest_results"),
                    memory_limit_gb=config.get('memory_limit_gb', 4.0),
                    enable_parallel=config.get('enable_parallel', True)
                )
            else:
                self.config = BacktestConfig()
        else:
            self.config = BacktestConfig()

        # 设置并行工作线程数
        if self.config.max_workers is None:
            self.config.max_workers = min(mp.cpu_count(), 8)

        # 初始化缓存目录
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 性能监控
        self.performance_stats = {
            'total_runs': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_execution_time': 0.0,
            'memory_usage_mb': 0.0
        }

        # 内存监控
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_gb)

        logger.info(f"优化版回测引擎初始化完成，最大工作线程数: {self.config.max_workers}")

    def run_backtest(self,


                     strategy: Callable,
                     data: Dict[str, pd.DataFrame],
                     config: BacktestConfig = None,
                     **kwargs) -> Dict[str, Any]:
        """
        运行回测

        Args:
            strategy: 策略函数
            data: 市场数据
            config: 回测配置
            **kwargs: 其他参数

        Returns:
            回测结果字典
        """
        start_time = time.time()

        # 使用提供的配置或默认配置
        run_config = config or self.config

        # 检查缓存
        cache_key = self._generate_cache_key(strategy, data, run_config)
        if run_config.enable_cache:
            cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            self.performance_stats['cache_hits'] += 1
            logger.info("从缓存加载回测结果")
            return cached_result

        self.performance_stats['cache_misses'] += 1

        # 选择执行方式
        if run_config.enable_parallel and self.config.max_workers > 1:
            result = self._run_parallel_backtest(strategy, data, run_config)
        else:
            result = self._run_sequential_backtest(strategy, data, run_config)

        # 计算性能指标
        result = self._calculate_performance_metrics(result)

        # 保存到缓存
        if run_config.enable_cache:
            self._save_to_cache(cache_key, result)

        # 更新性能统计
        execution_time = time.time() - start_time
        self._update_performance_stats(execution_time)

        # 内存清理
        self._cleanup_memory()

        logger.info(f"回测完成，执行时间: {execution_time:.2f}秒")
        return result

    def run_multiple_strategies(self,


                                strategies: List[Callable],
                                data: Dict[str, pd.DataFrame],
                                config: BacktestConfig = None) -> Dict[str, Dict[str, Any]]:
        """
        并行运行多个策略

        Args:
            strategies: 策略函数列表
            data: 市场数据
            config: 回测配置

        Returns:
            策略名称到结果的映射
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_strategy = {
                executor.submit(self.run_backtest, strategy, data, config): strategy
                for strategy in strategies
            }

            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    strategy_name = strategy.__name__ if hasattr(
                        strategy, '__name__') else str(strategy)
                    results[strategy_name] = result
                except Exception as e:
                    logger.error(f"策略 {strategy} 执行失败: {e}")

        return results

    def run_parameter_optimization(self,

                                   strategy_factory: Callable,
                                   data: Dict[str, pd.DataFrame],
                                   param_ranges: Dict[str, List],
                                   config: BacktestConfig = None) -> List[Dict[str, Any]]:
        """
        参数优化

        Args:
            strategy_factory: 策略工厂函数
            data: 市场数据
            param_ranges: 参数范围
            config: 回测配置

        Returns:
            优化结果列表
        """
        param_combinations = self._generate_param_combinations(param_ranges)
        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_params = {
                executor.submit(self._run_single_optimization, strategy_factory, data, params, config): params
                for params in param_combinations
            }

            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    result['parameters'] = params
                    results.append(result)
                except Exception as e:
                    logger.error(f"参数组合 {params} 优化失败: {e}")

        # 按收益率排序
        results.sort(key=lambda x: x.get('total_return', 0), reverse=True)

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.performance_stats.copy()
        stats['memory_usage_mb'] = self.memory_monitor.get_current_memory_usage()
        return stats

    def clear_cache(self, pattern: str = None):
        """
        清理缓存

        Args:
            pattern: 文件模式匹配
        """
        if pattern:
            cache_files = list(self.cache_dir.glob(pattern))
        else:
            cache_files = list(self.cache_dir.glob('*.pkl'))

        for cache_file in cache_files:
            try:
                cache_file.unlink()
                logger.info(f"已删除缓存文件: {cache_file}")
            except Exception as e:
                logger.error(f"删除缓存文件失败 {cache_file}: {e}")

    def _run_parallel_backtest(self, strategy: Callable, data: Dict[str, pd.DataFrame],


                               config: BacktestConfig) -> Dict[str, Any]:
        """并行执行回测"""
        # 数据预处理
        processed_data = self._preprocess_data(data)

        # 数据分块
        data_chunks = self._split_data(processed_data)

        # 并行处理数据块
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._run_backtest_chunk, strategy, chunk, config): chunk
                for chunk in data_chunks
            }

            chunk_results = []
            for future in as_completed(future_to_chunk):
                try:
                    result = future.result()
                    chunk_results.append(result)
                except Exception as e:
                    logger.error(f"数据块处理失败: {e}")

        # 合并结果
        return self._merge_chunk_results(chunk_results)

    def _run_sequential_backtest(self, strategy: Callable, data: Dict[str, pd.DataFrame],


                                 config: BacktestConfig) -> Dict[str, Any]:
        """顺序执行回测"""
        # 数据预处理
        processed_data = self._preprocess_data(data)

        # 初始化回测状态
        positions = {}
        cash = config.initial_capital
        trades = []
        portfolio_values = []

        # 获取交易日期
        all_dates = set()
        for df in processed_data.values():
            if not df.empty:
                all_dates.update(df.index)

        trading_dates = sorted(list(all_dates))

        # 逐日执行策略
        for date in trading_dates:
            # 获取当日数据
            daily_data = {}
            for symbol, df in processed_data.items():
                if date in df.index:
                    daily_data[symbol] = df.loc[date:date]

            if not daily_data:
                continue

            # 执行策略
            try:
                signals = strategy(daily_data, positions, cash, date)

                # 更新持仓和现金
                positions, cash, new_trades = self._update_positions(
                    positions, cash, signals, daily_data, config
                )

                trades.extend(new_trades)

                # 计算组合价值
                portfolio_value = cash
                for symbol, position in positions.items():
                    if symbol in daily_data:
                        price = daily_data[symbol].iloc[0]['close']
                        portfolio_value += position['quantity'] * price

                portfolio_values.append({
                    'date': date,
                    'value': portfolio_value,
                    'cash': cash
                })

            except Exception as e:
                logger.error(f"策略执行失败，日期: {date}, 错误: {e}")

        return {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'final_positions': positions,
            'final_cash': cash,
            'total_return': (portfolio_values[-1]['value'] / config.initial_capital - 1) if portfolio_values else 0
        }

    def _run_backtest_chunk(self, strategy: Callable, data_chunk: Dict[str, pd.DataFrame],

                            config: BacktestConfig) -> Dict[str, Any]:
        """执行数据块回测"""
        return self._run_sequential_backtest(strategy, data_chunk, config)

    def _run_single_optimization(self, strategy_factory: Callable, data: Dict[str, pd.DataFrame],


                                 params: Dict, config: BacktestConfig) -> Dict[str, Any]:
        """执行单次参数优化"""
        strategy = strategy_factory(params)
        return self.run_backtest(strategy, data, config)

    def _preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """数据预处理"""
        processed_data = {}

        for symbol, df in data.items():
            if df is None or df.empty:
                continue

            # 确保索引是datetime类型
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # 按时间排序
            df = df.sort_index()

            # 处理缺失值
            df = df.fillna(method='ffill').fillna(method='bfill')

            processed_data[symbol] = df

        return processed_data

    def _split_data(self, data: Dict[str, pd.DataFrame]) -> List[Dict[str, pd.DataFrame]]:
        """数据分块"""
        # 简单的按时间分块
        all_dates = set()
        for df in data.values():
            if not df.empty:
                all_dates.update(df.index)

        if not all_dates:
            return [data]

        sorted_dates = sorted(list(all_dates))
        chunk_size = max(1, len(sorted_dates) // self.config.max_workers)

        chunks = []
        for i in range(0, len(sorted_dates), chunk_size):
            chunk_dates = sorted_dates[i:i + chunk_size]
            chunk_data = {}

        for symbol, df in data.items():
            chunk_df = df[df.index.isin(chunk_dates)]
        if not chunk_df.empty:
            chunk_data[symbol] = chunk_df

        if chunk_data:
            chunks.append(chunk_data)

        return chunks if chunks else [data]

    def _merge_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并数据块结果"""
        if not chunk_results:
            return {}

        merged_result = {
            'trades': [],
            'portfolio_values': [],
            'final_positions': {},
            'final_cash': 0,
            'total_return': 0
        }

        for chunk_result in chunk_results:
            merged_result['trades'].extend(chunk_result.get('trades', []))
            merged_result['portfolio_values'].extend(chunk_result.get('portfolio_values', []))

            # 合并最终持仓
            for symbol, position in chunk_result.get('final_positions', {}).items():
                if symbol in merged_result['final_positions']:
                    # 合并持仓
                    existing = merged_result['final_positions'][symbol]
                    existing['quantity'] += position['quantity']
                    existing['avg_price'] = (existing['avg_price'] * existing['quantity']
                                             + position['avg_price'] * position['quantity']) / (existing['quantity'] + position['quantity'])
                else:
                    merged_result['final_positions'][symbol] = position.copy()

            merged_result['final_cash'] += chunk_result.get('final_cash', 0)

        # 按时间排序
        merged_result['trades'].sort(key=lambda x: x.get('date', ''))
        merged_result['portfolio_values'].sort(key=lambda x: x.get('date', ''))

        return merged_result

    def _update_positions(self, positions: Dict[str, float], cash: float,


                          signals: Dict[str, float], data: Dict[str, Any],
                          config: BacktestConfig) -> tuple:
        """更新持仓和现金"""
        new_trades = []

        for symbol, signal in signals.items():
            if symbol not in data:
                continue

            current_price = data[symbol].iloc[0]['close']

            if signal > 0:  # 买入信号
                # 计算可买入数量
                max_quantity = int(cash * 0.95 / current_price)  # 保留5 % 现金
                quantity = min(int(signal), max_quantity)

                if quantity > 0:
                    # 计算交易成本
                    cost = quantity * current_price
                    commission = cost * config.commission_rate
                    total_cost = cost + commission

                    if total_cost <= cash:
                        # 执行交易
                        cash -= total_cost

                        if symbol in positions:
                            # 更新现有持仓
                            existing = positions[symbol]
                            total_quantity = existing['quantity'] + quantity
                            existing['avg_price'] = (existing['avg_price'] * existing['quantity']
                                                     + current_price * quantity) / total_quantity
                            existing['quantity'] = total_quantity
                        else:
                            # 新建持仓
                            positions[symbol] = {
                                'quantity': quantity,
                                'avg_price': current_price
                            }

                        new_trades.append({
                            'date': data[symbol].index[0],
                            'symbol': symbol,
                            'direction': 'buy',
                            'quantity': quantity,
                            'price': current_price,
                            'cost': total_cost
                        })

            elif signal < 0:  # 卖出信号
                if symbol in positions:
                    existing = positions[symbol]
                    quantity = min(int(abs(signal)), existing['quantity'])

                    if quantity > 0:
                        # 计算收益
                        revenue = quantity * current_price
                        commission = revenue * config.commission_rate
                        net_revenue = revenue - commission

                        # 更新持仓
                        existing['quantity'] -= quantity
                        if existing['quantity'] <= 0:
                            del positions[symbol]

                        cash += net_revenue

                        new_trades.append({
                            'date': data[symbol].index[0],
                            'symbol': symbol,
                            'direction': 'sell',
                            'quantity': quantity,
                            'price': current_price,
                            'revenue': net_revenue
                        })

        return positions, cash, new_trades

    def _calculate_performance_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """计算性能指标"""
        if not result.get('portfolio_values'):
            return result

        portfolio_values = result['portfolio_values']
        if not portfolio_values:
            return result

        # 计算收益率序列
        initial_value = portfolio_values[0]['value']
        returns = []
        for pv in portfolio_values[1:]:
            prev_value = portfolio_values[portfolio_values.index(pv) - 1]['value']
            returns.append((pv['value'] - prev_value) / prev_value)

        if not returns:
            return result

        returns = np.array(returns)

        # 计算各种指标
        total_return = (portfolio_values[-1]['value'] / initial_value - 1)
        annual_return = total_return * 252 / len(returns) if len(returns) > 0 else 0

        # 夏普比率
        risk_free_rate = self.config.risk_free_rate
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * \
            np.sqrt(252) if np.std(excess_returns) > 0 else 0

        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # 波动率
        volatility = np.std(returns) * np.sqrt(252)

        result.update({
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        })

        return result

    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """生成参数组合"""
        import itertools

        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        combinations = []
        for values in itertools.product(*param_values):
            combination = dict(zip(param_names, values))
            combinations.append(combination)

        return combinations

    def _generate_cache_key(self, strategy: Callable, data: Dict[str, pd.DataFrame],


                            config: BacktestConfig) -> str:
        """生成缓存键"""
        # 简化的缓存键生成
        strategy_name = strategy.__name__ if hasattr(strategy, '__name__') else str(strategy)
        data_hash = hashlib.md5(str(sorted(data.keys())).encode()).hexdigest()[:8]
        config_hash = hashlib.md5(str(config.__dict__).encode()).hexdigest()[:8]

        return f"{strategy_name}_{data_hash}_{config_hash}"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存加载结果"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"读取缓存失败: {e}")

        return None

    def _save_to_cache(self, cache_key: str, result: Dict):
        """保存结果到缓存"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"结果已保存到缓存: {cache_key}")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _update_performance_stats(self, execution_time: float):
        """更新性能统计"""
        self.performance_stats['total_runs'] += 1
        self.performance_stats['avg_execution_time'] = (
            (self.performance_stats['avg_execution_time'] *
             (self.performance_stats['total_runs'] - 1) + execution_time)
            / self.performance_stats['total_runs']
        )

    def _cleanup_memory(self):
        """内存清理"""
        if self.memory_monitor.check_memory_usage():
            gc.collect()
            logger.info("执行内存清理")


class MemoryMonitor:

    """内存监控器"""

    def __init__(self, limit_gb: float):

        self.limit_mb = limit_gb * 1024

    def get_current_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        return psutil.Process().memory_info().rss / 1024 / 1024

    def check_memory_usage(self):
        """检查内存使用量是否超过限制"""
        current_usage = self.get_current_memory_usage()
        return current_usage > self.limit_mb


# 向后兼容的别名
BacktestEngine = OptimizedBacktestEngine
