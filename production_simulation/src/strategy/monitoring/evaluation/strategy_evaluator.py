"""
策略评估器

提供完整的策略评估功能，包括：
- 策略对比分析
- 稳定性测试
- 风险评估
- 绩效归因分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:

    """评估配置类"""
    risk_free_rate: float = 0.03
    benchmark_return: float = 0.08
    confidence_level: float = 0.95
    min_periods: int = 252  # 最小评估期
    max_drawdown_threshold: float = 0.2
    sharpe_threshold: float = 1.0
    volatility_threshold: float = 0.3


@dataclass
class StrategyMetrics:

    """策略指标数据类"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: float
    beta: float
    alpha: float
    treynor_ratio: float
    jensen_alpha: float


class StrategyEvaluator:

    """
    策略评估器

    提供全面的策略评估功能，包括：
    - 基础绩效指标计算
    - 风险指标分析
    - 策略对比
    - 稳定性测试
    - 绩效归因分析
    """

    def __init__(self, config: EvaluationConfig = None):
        """
        初始化策略评估器

        Args:
            config: 评估配置
        """
        self.config = config or EvaluationConfig()
        self.evaluation_results = {}

        logger.info("策略评估器初始化完成")

    def evaluate_strategy(self,


                          strategy_returns: pd.Series,
                          benchmark_returns: pd.Series = None,
                          risk_free_rate: float = None) -> StrategyMetrics:
        """
        评估单个策略

        Args:
            strategy_returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            risk_free_rate: 无风险利率

        Returns:
            策略指标
        """
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate

        # 计算基础指标
        total_return = self._calculate_total_return(strategy_returns)
        annual_return = self._calculate_annual_return(strategy_returns)
        volatility = self._calculate_volatility(strategy_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(strategy_returns, risk_free_rate)
        max_drawdown = self._calculate_max_drawdown(strategy_returns)
        calmar_ratio = self._calculate_calmar_ratio(annual_return, max_drawdown)
        sortino_ratio = self._calculate_sortino_ratio(strategy_returns, risk_free_rate)

        # 计算相对指标（如果有基准）
        information_ratio = 0.0
        beta = 1.0
        alpha = 0.0
        treynor_ratio = 0.0
        jensen_alpha = 0.0

        if benchmark_returns is not None:
            information_ratio = self._calculate_information_ratio(
                strategy_returns, benchmark_returns)
            beta = self._calculate_beta(strategy_returns, benchmark_returns)
            alpha = self._calculate_alpha(strategy_returns, benchmark_returns, risk_free_rate)
            treynor_ratio = self._calculate_treynor_ratio(annual_return, beta, risk_free_rate)
            jensen_alpha = self._calculate_jensen_alpha(
                strategy_returns, benchmark_returns, risk_free_rate)

        return StrategyMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            information_ratio=information_ratio,
            beta=beta,
            alpha=alpha,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha
        )

    def compare_strategies(self,


                           strategies_returns: Dict[str, pd.Series],
                           benchmark_returns: pd.Series = None) -> Dict[str, StrategyMetrics]:
        """
        对比多个策略

        Args:
            strategies_returns: 策略名称到收益率序列的映射
            benchmark_returns: 基准收益率序列

        Returns:
            策略名称到指标的映射
        """
        results = {}

        for strategy_name, returns in strategies_returns.items():
            try:
                metrics = self.evaluate_strategy(returns, benchmark_returns)
                results[strategy_name] = metrics
                logger.info(f"策略 {strategy_name} 评估完成")
            except Exception as e:
                logger.error(f"策略 {strategy_name} 评估失败: {e}")

        return results

    def stability_test(self,


                       strategy_returns: pd.Series,
                       window_size: int = 252,
                       min_periods: int = None) -> Dict[str, Any]:
        """
        策略稳定性测试

        Args:
            strategy_returns: 策略收益率序列
            window_size: 滚动窗口大小
            min_periods: 最小期数

        Returns:
            稳定性测试结果
        """
        if min_periods is None:
            min_periods = self.config.min_periods

        # 滚动计算关键指标
        rolling_sharpe = self._calculate_rolling_sharpe(strategy_returns, window_size)
        rolling_volatility = self._calculate_rolling_volatility(strategy_returns, window_size)
        rolling_drawdown = self._calculate_rolling_drawdown(strategy_returns, window_size)

        # 计算稳定性指标
        sharpe_stability = self._calculate_stability_metric(rolling_sharpe)
        volatility_stability = self._calculate_stability_metric(rolling_volatility)
        drawdown_stability = self._calculate_stability_metric(rolling_drawdown)

        # 计算趋势分析
        sharpe_trend = self._calculate_trend_analysis(rolling_sharpe)
        volatility_trend = self._calculate_trend_analysis(rolling_volatility)

        return {
            'rolling_metrics': {
                'sharpe_ratio': rolling_sharpe,
                'volatility': rolling_volatility,
                'max_drawdown': rolling_drawdown
            },
            'stability_scores': {
                'sharpe_stability': sharpe_stability,
                'volatility_stability': volatility_stability,
                'drawdown_stability': drawdown_stability
            },
            'trend_analysis': {
                'sharpe_trend': sharpe_trend,
                'volatility_trend': volatility_trend
            }
        }

    def risk_analysis(self,


                      strategy_returns: pd.Series,
                      confidence_level: float = None) -> Dict[str, Any]:
        """
        风险分析

        Args:
            strategy_returns: 策略收益率序列
            confidence_level: 置信水平

        Returns:
            风险分析结果
        """
        if confidence_level is None:
            confidence_level = self.config.confidence_level

        # 计算VaR和CVaR
        var_historical = self._calculate_var_historical(strategy_returns, confidence_level)
        var_parametric = self._calculate_var_parametric(strategy_returns, confidence_level)
        cvar = self._calculate_cvar(strategy_returns, confidence_level)

        # 计算下行风险
        downside_risk = self._calculate_downside_risk(strategy_returns)

        # 计算极端风险指标
        extreme_risk = self._calculate_extreme_risk(strategy_returns)

        # 计算风险分解
        risk_decomposition = self._calculate_risk_decomposition(strategy_returns)

        return {
            'var_historical': var_historical,
            'var_parametric': var_parametric,
            'cvar': cvar,
            'downside_risk': downside_risk,
            'extreme_risk': extreme_risk,
            'risk_decomposition': risk_decomposition
        }

    def performance_attribution(self,


                                strategy_returns: pd.Series,
                                factor_returns: Dict[str, pd.Series],
                                benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """
        绩效归因分析

        Args:
            strategy_returns: 策略收益率序列
            factor_returns: 因子收益率序列
            benchmark_returns: 基准收益率序列

        Returns:
            绩效归因结果
        """
        # 因子暴露分析
        factor_exposures = self._calculate_factor_exposures(strategy_returns, factor_returns)

        # 因子贡献分析
        factor_contributions = self._calculate_factor_contributions(
            strategy_returns, factor_returns)

        # 残差分析
        residual_analysis = self._calculate_residual_analysis(strategy_returns, factor_returns)

        # 基准归因（如果有基准）
        benchmark_attribution = None
        if benchmark_returns is not None:
            benchmark_attribution = self._calculate_benchmark_attribution(
                strategy_returns, benchmark_returns, factor_returns
            )

        return {
            'factor_exposures': factor_exposures,
            'factor_contributions': factor_contributions,
            'residual_analysis': residual_analysis,
            'benchmark_attribution': benchmark_attribution
        }

    def generate_evaluation_report(self,


                                   strategy_metrics: StrategyMetrics,
                                   stability_results: Dict[str, Any] = None,
                                   risk_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        生成评估报告

        Args:
            strategy_metrics: 策略指标
            stability_results: 稳定性测试结果
            risk_results: 风险分析结果

        Returns:
            评估报告
        """
        report = {
            'summary': {
                'total_return': strategy_metrics.total_return,
                'annual_return': strategy_metrics.annual_return,
                'sharpe_ratio': strategy_metrics.sharpe_ratio,
                'max_drawdown': strategy_metrics.max_drawdown,
                'volatility': strategy_metrics.volatility
            },
            'risk_metrics': {
                'sharpe_ratio': strategy_metrics.sharpe_ratio,
                'sortino_ratio': strategy_metrics.sortino_ratio,
                'calmar_ratio': strategy_metrics.calmar_ratio,
                'information_ratio': strategy_metrics.information_ratio,
                'beta': strategy_metrics.beta,
                'alpha': strategy_metrics.alpha
            },
            'stability_analysis': stability_results,
            'risk_analysis': risk_results,
            'recommendations': self._generate_recommendations(strategy_metrics)
        }

        return report

    def _calculate_total_return(self, returns: pd.Series) -> float:
        """计算总收益率"""
        return (1 + returns).prod() - 1

    def _calculate_annual_return(self, returns: pd.Series) -> float:
        """计算年化收益率"""
        total_return = self._calculate_total_return(returns)
        years = len(returns) / 252
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """计算年化波动率"""
        return returns.std() * np.sqrt(252)

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate / 252
        volatility = self._calculate_volatility(returns)
        return excess_returns.mean() * 252 / volatility if volatility > 0 else 0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """计算卡玛比率"""
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """计算索提诺比率"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        return excess_returns.mean() * 252 / downside_volatility if downside_volatility > 0 else 0

    def _calculate_information_ratio(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算信息比率"""
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        return excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

    def _calculate_beta(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """计算贝塔系数"""
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0

    def _calculate_alpha(self, strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> float:
        """计算阿尔法系数"""
        beta = self._calculate_beta(strategy_returns, benchmark_returns)
        strategy_return = strategy_returns.mean() * 252
        benchmark_return = benchmark_returns.mean() * 252
        return strategy_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))

    def _calculate_treynor_ratio(self, annual_return: float, beta: float, risk_free_rate: float) -> float:
        """计算特雷诺比率"""
        return (annual_return - risk_free_rate) / beta if beta != 0 else 0

    def _calculate_jensen_alpha(self, strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> float:
        """计算詹森阿尔法"""
        return self._calculate_alpha(strategy_returns, benchmark_returns, risk_free_rate)

    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        """计算滚动夏普比率"""

        def rolling_sharpe(x):

            if len(x) < 2:
                return np.nan
            return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan

        return returns.rolling(window=window).apply(rolling_sharpe)

    def _calculate_rolling_volatility(self, returns: pd.Series, window: int) -> pd.Series:
        """计算滚动波动率"""
        return returns.rolling(window=window).std() * np.sqrt(252)

    def _calculate_rolling_drawdown(self, returns: pd.Series, window: int) -> pd.Series:
        """计算滚动最大回撤"""

        def rolling_drawdown(x):

            if len(x) < 2:
                return np.nan
            cumulative = (1 + x).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()

        return returns.rolling(window=window).apply(rolling_drawdown)

    def _calculate_stability_metric(self, rolling_series: pd.Series) -> float:
        """计算稳定性指标"""
        if rolling_series.isna().all():
            return 0.0

        # 使用变异系数作为稳定性指标
        valid_series = rolling_series.dropna()
        if len(valid_series) < 2:
            return 0.0

        return 1 / (1 + valid_series.std() / abs(valid_series.mean())) if valid_series.mean() != 0 else 0

    def _calculate_trend_analysis(self, rolling_series: pd.Series) -> Dict[str, float]:
        """计算趋势分析"""
        if rolling_series.isna().all():
            return {'slope': 0.0, 'r_squared': 0.0}

        valid_series = rolling_series.dropna()
        if len(valid_series) < 2:
            return {'slope': 0.0, 'r_squared': 0.0}

        x = np.arange(len(valid_series))
        y = valid_series.values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        return {
            'slope': slope,
            'r_squared': r_squared,
            'p_value': p_value
        }

    def _calculate_var_historical(self, returns: pd.Series, confidence_level: float) -> float:
        """计算历史VaR"""
        return np.percentile(returns, (1 - confidence_level) * 100)

    def _calculate_var_parametric(self, returns: pd.Series, confidence_level: float) -> float:
        """计算参数VaR"""
        from scipy.stats import norm
        z_score = norm.ppf(confidence_level)
        return returns.mean() - z_score * returns.std()

    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """计算CVaR"""
        var = self._calculate_var_historical(returns, confidence_level)
        return returns[returns <= var].mean()

    def _calculate_downside_risk(self, returns: pd.Series) -> float:
        """计算下行风险"""
        downside_returns = returns[returns < 0]
        return downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

    def _calculate_extreme_risk(self, returns: pd.Series) -> Dict[str, float]:
        """计算极端风险指标"""
        return {
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'tail_risk': returns.quantile(0.01)
        }

    def _calculate_risk_decomposition(self, returns: pd.Series) -> Dict[str, float]:
        """计算风险分解"""
        # 简化的风险分解
        volatility = returns.std()
        return {
            'systematic_risk': volatility * 0.7,  # 假设70 % 为系统性风险
            'idiosyncratic_risk': volatility * 0.3  # 假设30 % 为特质风险
        }

    def _calculate_factor_exposures(self, strategy_returns: pd.Series, factor_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """计算因子暴露"""
        exposures = {}
        for factor_name, factor_returns_series in factor_returns.items():
            # 简化的因子暴露计算
            correlation = strategy_returns.corr(factor_returns_series)
            exposures[factor_name] = correlation if not pd.isna(correlation) else 0.0

        return exposures

    def _calculate_factor_contributions(self, strategy_returns: pd.Series, factor_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """计算因子贡献"""
        contributions = {}
        for factor_name, factor_returns_series in factor_returns.items():
            # 简化的因子贡献计算
            exposure = strategy_returns.corr(factor_returns_series)
            factor_volatility = factor_returns_series.std()
            contribution = exposure * factor_volatility if not pd.isna(exposure) else 0.0
            contributions[factor_name] = contribution

        return contributions

    def _calculate_residual_analysis(self, strategy_returns: pd.Series, factor_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """计算残差分析"""
        # 简化的残差分析
        total_factor_contribution = sum(self._calculate_factor_contributions(
            strategy_returns, factor_returns).values())
        strategy_volatility = strategy_returns.std()
        residual_volatility = max(0, strategy_volatility - total_factor_contribution)

        return {
            'residual_volatility': residual_volatility,
            'explained_variance': total_factor_contribution / strategy_volatility if strategy_volatility > 0 else 0
        }

    def _calculate_benchmark_attribution(self, strategy_returns: pd.Series, benchmark_returns: pd.Series,


                                         factor_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """计算基准归因"""
        # 简化的基准归因
        excess_returns = strategy_returns - benchmark_returns
        return {
            'excess_return': excess_returns.mean() * 252,
            'tracking_error': excess_returns.std() * np.sqrt(252),
            'information_ratio': self._calculate_information_ratio(strategy_returns, benchmark_returns)
        }

    def _generate_recommendations(self, metrics: StrategyMetrics) -> List[str]:
        """生成建议"""
        recommendations = []

        if metrics.sharpe_ratio < self.config.sharpe_threshold:
            recommendations.append("夏普比率较低，建议优化风险调整后收益")

        if metrics.max_drawdown > self.config.max_drawdown_threshold:
            recommendations.append("最大回撤过大，建议加强风险控制")

        if metrics.volatility > self.config.volatility_threshold:
            recommendations.append("波动率较高，建议分散投资")

        if metrics.beta > 1.5:
            recommendations.append("贝塔系数较高，市场风险暴露较大")

        if metrics.alpha < 0:
            recommendations.append("阿尔法为负，策略相对基准表现不佳")

        if not recommendations:
            recommendations.append("策略表现良好，建议继续监控")

        return recommendations


# 向后兼容的别名
Evaluator = StrategyEvaluator
