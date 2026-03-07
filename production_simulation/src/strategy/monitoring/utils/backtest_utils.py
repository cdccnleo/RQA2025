import logging
#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
回测专用工具函数
提供策略验证、风险指标计算等回测专用功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StrategyValidationResult:

    """策略验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class BacktestUtils:

    """回测专用工具类"""

    @staticmethod
    def validate_strategy(strategy: Any) -> StrategyValidationResult:
        """
        验证策略的有效性

        Args:
            strategy: 策略对象

        Returns:
            StrategyValidationResult: 验证结果
        """
        errors = []
        warnings = []
        suggestions = []

        # 检查策略是否有必要的方法
        required_methods = ['generate_signals', 'on_init', 'on_day_start']
        for method in required_methods:
            if not hasattr(strategy, method):
                errors.append(f"策略缺少必要方法: {method}")

        # 检查策略参数
        if hasattr(strategy, 'params'):
            params = strategy.params
            if isinstance(params, dict):
                # 检查参数合理性
                for key, value in params.items():
                    if isinstance(value, (int, float)):
                        if value < 0:
                            warnings.append(f"参数 {key} 为负值: {value}")
                        if value > 1000:
                            warnings.append(f"参数 {key} 值过大: {value}")

        # 检查策略名称
        if hasattr(strategy, '__class__'):
            strategy_name = strategy.__class__.__name__
        if len(strategy_name) < 3:
            suggestions.append("策略名称过短，建议使用更具描述性的名称")

        is_valid = len(errors) == 0

        return StrategyValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    @staticmethod
    def calculate_risk_metrics(returns: pd.Series,


                               risk_free_rate: float = 0.03) -> Dict[str, float]:
        """
        计算风险指标

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率

        Returns:
            Dict[str, float]: 风险指标字典
        """
        if returns.empty:
            return {}

        # 基础指标
        total_return = (1 + returns).prod() - 1
        volatility = returns.std() * np.sqrt(252)

        # 夏普比率
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 索提诺比率
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_deviation * \
            np.sqrt(252) if downside_deviation > 0 else 0

        # 卡玛比率
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': returns.quantile(0.05),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean()
        }

    @staticmethod
    def calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
        """
        计算交易指标

        Args:
            trades: 交易记录DataFrame

        Returns:
            Dict[str, float]: 交易指标字典
        """
        if trades.empty or 'profit' not in trades.columns:
            return {}

        profits = trades['profit']
        winning_trades = profits[profits > 0]
        losing_trades = profits[profits < 0]

        win_rate = len(winning_trades) / len(profits) if len(profits) > 0 else 0
        profit_factor = winning_trades.sum() / abs(losing_trades.sum()
                                                   ) if len(losing_trades) > 0 and losing_trades.sum() != 0 else 0
        average_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        average_loss = losing_trades.mean() if len(losing_trades) > 0 else 0

        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'total_trades': len(profits),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }

    @staticmethod
    def validate_data(data: pd.DataFrame,


                      required_columns: List[str] = None) -> StrategyValidationResult:
        """
        验证数据的有效性

        Args:
            data: 数据DataFrame
            required_columns: 必需的列名列表

        Returns:
            StrategyValidationResult: 验证结果
        """
        errors = []
        warnings = []
        suggestions = []

        if data.empty:
            errors.append("数据为空")
            return StrategyValidationResult(False, errors, warnings, suggestions)

        # 检查必需列
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"缺少必需列: {missing_columns}")

        # 检查数据类型
        for col in data.columns:
            if data[col].dtype == 'object':
                warnings.append(f"列 {col} 为对象类型，可能影响性能")

        # 检查缺失值
        missing_counts = data.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                warnings.append(f"列 {col} 有 {count} 个缺失值")

        # 检查重复值
        if data.duplicated().any():
            warnings.append("数据中存在重复行")

        # 检查数据范围
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].min() < -1:
                warnings.append(f"列 {col} 存在异常负值")
        if data[col].max() > 1000:
            warnings.append(f"列 {col} 存在异常大值")

        is_valid = len(errors) == 0

        return StrategyValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    @staticmethod
    def calculate_portfolio_metrics(positions: pd.DataFrame,


                                    prices: pd.DataFrame) -> Dict[str, float]:
        """
        计算组合指标

        Args:
            positions: 仓位DataFrame
            prices: 价格DataFrame

        Returns:
            Dict[str, float]: 组合指标字典
        """
        if positions.empty or prices.empty:
            return {}

        # 计算组合价值
        portfolio_value = (positions * prices).sum(axis=1)

        # 计算组合权重
        weights = positions.div(portfolio_value, axis=0)

        # 计算集中度
        concentration = (weights ** 2).sum(axis=1).mean()

        # 计算换手率
        turnover = positions.diff().abs().sum(axis=1).mean()

        return {
            'portfolio_value': portfolio_value.mean(),
            'concentration': concentration,
            'turnover': turnover,
            'num_positions': (positions != 0).sum(axis=1).mean()
        }

    @staticmethod
    def generate_backtest_report(results: Dict[str, Any]) -> str:
        """
        生成回测报告

        Args:
            results: 回测结果字典

        Returns:
            str: 报告内容
        """
        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("回测报告")
        report_lines.append("=" * 50)

        # 基础信息
        if 'config' in results:
            config = results['config']
            report_lines.append(
                f"回测期间: {config.get('start_date', 'N / A')} - {config.get('end_date', 'N / A')}")
            report_lines.append(f"初始资金: {config.get('initial_capital', 'N / A'):,.0f}")

        # 绩效指标
        if 'metrics' in results:
            metrics = results['metrics']
            report_lines.append("\n绩效指标:")
            report_lines.append("-" * 20)
            report_lines.append(f"总收益率: {metrics.get('total_return', 0):.2%}")
            report_lines.append(f"年化收益率: {metrics.get('annual_return', 0):.2%}")
            report_lines.append(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
            report_lines.append(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
            report_lines.append(f"波动率: {metrics.get('volatility', 0):.2%}")

        # 交易统计
        if 'trades' in results:
            trades = results['trades']
        if not trades.empty:
            report_lines.append("\n交易统计:")
            report_lines.append("-" * 20)
            report_lines.append(f"总交易次数: {len(trades)}")
            report_lines.append(f"胜率: {trades.get('win_rate', 0):.2%}")
            report_lines.append(f"盈亏比: {trades.get('profit_factor', 0):.2f}")

        report_lines.append("\n" + "=" * 50)

        return "\n".join(report_lines)

    @staticmethod
    def save_backtest_results(results: Dict[str, Any],


                              filepath: str) -> bool:
        """
        保存回测结果

        Args:
            results: 回测结果字典
            filepath: 保存路径

        Returns:
            bool: 是否保存成功
        """
        try:
            import json
            from datetime import datetime

            # 添加时间戳
            results['timestamp'] = datetime.now().isoformat()

            # 处理DataFrame序列化
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    serializable_results[key] = value.to_dict()
                elif isinstance(value, pd.Series):
                    serializable_results[key] = value.to_dict()
                else:
                    serializable_results[key] = value

            with open(filepath, 'w', encoding='utf - 8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            logger.info(f"回测结果已保存到: {filepath}")
            return True

        except Exception as e:
            logger.error(f"保存回测结果失败: {str(e)}")
            return False
