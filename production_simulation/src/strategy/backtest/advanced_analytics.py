#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
高级分析功能模块

提供多因子分析、风险归因、机器学习集成等高级分析功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class FactorData:

    """因子数据结构"""
    timestamp: datetime
    symbol: str
    factor_name: str
    factor_value: float
    source: str = "calculated"


@dataclass
class RiskAttribution:

    """风险归因结果"""
    factor_name: str
    contribution: float
    percentage: float
    risk_type: str  # 'systematic', 'idiosyncratic'


class MultiFactorAnalyzer:

    """多因子分析器"""

    def __init__(self, factors: List[str] = None):

        self.factors = factors or ['momentum', 'value', 'size', 'quality', 'volatility']
        self.factor_data = {}
        self.model = None
        self.scaler = StandardScaler()

    def add_factor_data(self, factor_data: FactorData):
        """添加因子数据"""
        if factor_data.symbol not in self.factor_data:
            self.factor_data[factor_data.symbol] = {}

        if factor_data.factor_name not in self.factor_data[factor_data.symbol]:
            self.factor_data[factor_data.symbol][factor_data.factor_name] = []

        self.factor_data[factor_data.symbol][factor_data.factor_name].append({
            'timestamp': factor_data.timestamp,
            'value': factor_data.factor_value
        })

    def calculate_factor_exposure(self, symbol: str, factor_name: str) -> float:
        """计算因子暴露度"""
        if symbol not in self.factor_data or factor_name not in self.factor_data[symbol]:
            return 0.0

        factor_history = self.factor_data[symbol][factor_name]
        if not factor_history:
            return 0.0

        # 计算最近N期的平均暴露度
        recent_values = [item['value'] for item in factor_history[-10:]]
        return np.mean(recent_values)

    def build_factor_model(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """构建因子模型"""
        try:
            # 准备数据
            X = []
            y = []

            for symbol in returns_data.columns:
                if symbol in self.factor_data:
                    factor_exposures = []
                    for factor in self.factors:
                        exposure = self.calculate_factor_exposure(symbol, factor)
                        factor_exposures.append(exposure)

                if len(factor_exposures) == len(self.factors):
                    X.append(factor_exposures)
                    # 使用平均收益率作为因变量
                    y.append(returns_data[symbol].mean())

            if len(X) > 0:
                X = np.array(X)
                y = np.array(y)

                # 标准化
                X_scaled = self.scaler.fit_transform(X)

                # 训练模型
                self.model = LinearRegression()
                self.model.fit(X_scaled, y)

                # 计算因子重要性
                factor_importance = dict(zip(self.factors, self.model.coef_))

                return {
                    'model': self.model,
                    'factor_importance': factor_importance,
                    'r_squared': self.model.score(X_scaled, y),
                    'n_assets': len(X)
                }

        except Exception as e:
            logger.error(f"构建因子模型失败: {e}")

        return {}

    def predict_returns(self, symbol: str) -> float:
        """预测收益率"""
        if not self.model:
            return 0.0

        factor_exposures = []
        for factor in self.factors:
            exposure = self.calculate_factor_exposure(symbol, factor)
            factor_exposures.append(exposure)

        if len(factor_exposures) == len(self.factors):
            X = np.array([factor_exposures])
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)[0]

        return 0.0


class RiskAttributionAnalyzer:

    """风险归因分析器"""

    def __init__(self):

        self.risk_factors = ['market', 'size', 'value', 'momentum', 'quality']
        self.factor_returns = {}
        self.portfolio_weights = {}

    def add_factor_return(self, factor_name: str, return_value: float, timestamp: datetime):
        """添加因子收益率"""
        if factor_name not in self.factor_returns:
            self.factor_returns[factor_name] = []

        self.factor_returns[factor_name].append({
            'timestamp': timestamp,
            'return': return_value
        })

    def set_portfolio_weights(self, weights: Dict[str, float]):
        """设置投资组合权重"""
        self.portfolio_weights = weights

    def calculate_risk_attribution(self) -> List[RiskAttribution]:
        """计算风险归因"""
        attributions = []

        if not self.factor_returns or not self.portfolio_weights:
            return attributions

        try:
            # 计算各因子的方差贡献
            total_variance = 0
            factor_contributions = {}

            for factor_name in self.risk_factors:
                if factor_name in self.factor_returns:
                    factor_returns = [item['return'] for item in self.factor_returns[factor_name]]
                    if len(factor_returns) > 1:
                        factor_variance = np.var(factor_returns)
                        factor_contributions[factor_name] = factor_variance
                        total_variance += factor_variance

            # 计算贡献度百分比
            if total_variance > 0:
                for factor_name, contribution in factor_contributions.items():
                    percentage = (contribution / total_variance) * 100
                    attributions.append(RiskAttribution(
                        factor_name=factor_name,
                        contribution=contribution,
                        percentage=percentage,
                        risk_type='systematic'
                    ))

        except Exception as e:
            logger.error(f"计算风险归因失败: {e}")

        return attributions


class MLIntegrationAnalyzer:

    """机器学习集成分析器"""

    def __init__(self):

        self.models = {}
        self.feature_importance = {}
        self.predictions = {}

    def train_price_prediction_model(self, symbol: str, features: pd.DataFrame,


                                     target: pd.Series, model_type: str = 'random_forest'):
        """训练价格预测模型"""
        try:
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = LinearRegression()

            # 训练模型
            model.fit(features, target)

            # 保存模型
            self.models[symbol] = model

            # 计算特征重要性
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[symbol] = dict(
                    zip(features.columns, model.feature_importances_))
            else:
                self.feature_importance[symbol] = dict(zip(features.columns, model.coef_))

            logger.info(f"价格预测模型训练完成: {symbol}")

        except Exception as e:
            logger.error(f"训练价格预测模型失败 {symbol}: {e}")

    def predict_price(self, symbol: str, features: pd.DataFrame) -> float:
        """预测价格"""
        if symbol in self.models:
            try:
                prediction = self.models[symbol].predict(features)[0]
                self.predictions[symbol] = prediction
                return prediction
            except Exception as e:
                logger.error(f"价格预测失败 {symbol}: {e}")

        return 0.0

    def get_feature_importance(self, symbol: str) -> Dict[str, float]:
        """获取特征重要性"""
        return self.feature_importance.get(symbol, {})

    def get_prediction(self, symbol: str) -> float:
        """获取预测结果"""
        return self.predictions.get(symbol, 0.0)


class CustomMetricsCalculator:

    """自定义指标计算器"""

    def __init__(self):

        self.metrics = {}

    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # 日化无风险利率

        if np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        if not portfolio_values:
            return 0.0

        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown)

    def calculate_calmar_ratio(self, returns: List[float], portfolio_values: List[float]) -> float:
        """计算卡玛比率"""
        sharpe = self.calculate_sharpe_ratio(returns)
        max_dd = abs(self.calculate_max_drawdown(portfolio_values))

        if max_dd == 0:
            return 0.0

        return sharpe / max_dd

    def calculate_information_ratio(self, strategy_returns: List[float],


                                    benchmark_returns: List[float]) -> float:
        """计算信息比率"""
        if len(strategy_returns) != len(benchmark_returns):
            return 0.0

        strategy_array = np.array(strategy_returns)
        benchmark_array = np.array(benchmark_returns)

        active_returns = strategy_array - benchmark_array

        if np.std(active_returns) == 0:
            return 0.0

        return np.mean(active_returns) / np.std(active_returns)

    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        if not returns:
            return 0.0

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252

        # 只考虑负收益
        negative_returns = excess_returns[excess_returns < 0]

        if len(negative_returns) == 0:
            return 0.0

        downside_deviation = np.std(negative_returns)

        if downside_deviation == 0:
            return 0.0

        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)


class AdvancedAnalyticsEngine:

    """高级分析引擎主类"""

    def __init__(self):

        self.multi_factor_analyzer = MultiFactorAnalyzer()
        self.risk_attribution_analyzer = RiskAttributionAnalyzer()
        self.ml_integration_analyzer = MLIntegrationAnalyzer()
        self.custom_metrics_calculator = CustomMetricsCalculator()

    def add_factor_data(self, factor_data: FactorData):
        """添加因子数据"""
        self.multi_factor_analyzer.add_factor_data(factor_data)

    def build_factor_model(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """构建因子模型"""
        return self.multi_factor_analyzer.build_factor_model(returns_data)

    def predict_returns(self, symbol: str) -> float:
        """预测收益率"""
        return self.multi_factor_analyzer.predict_returns(symbol)

    def calculate_risk_attribution(self) -> List[RiskAttribution]:
        """计算风险归因"""
        return self.risk_attribution_analyzer.calculate_risk_attribution()

    def train_ml_model(self, symbol: str, features: pd.DataFrame,


                       target: pd.Series, model_type: str = 'random_forest'):
        """训练机器学习模型"""
        self.ml_integration_analyzer.train_price_prediction_model(
            symbol, features, target, model_type)

    def predict_price(self, symbol: str, features: pd.DataFrame) -> float:
        """预测价格"""
        return self.ml_integration_analyzer.predict_price(symbol, features)

    def calculate_custom_metrics(self, returns: List[float], portfolio_values: List[float],


                                 benchmark_returns: List[float] = None) -> Dict[str, float]:
        """计算自定义指标"""
        metrics = {}

        metrics['sharpe_ratio'] = self.custom_metrics_calculator.calculate_sharpe_ratio(returns)
        metrics['max_drawdown'] = self.custom_metrics_calculator.calculate_max_drawdown(
            portfolio_values)
        metrics['calmar_ratio'] = self.custom_metrics_calculator.calculate_calmar_ratio(
            returns, portfolio_values)
        metrics['sortino_ratio'] = self.custom_metrics_calculator.calculate_sortino_ratio(returns)

        if benchmark_returns:
            metrics['information_ratio'] = self.custom_metrics_calculator.calculate_information_ratio(
                returns, benchmark_returns)

        return metrics


# 导出主要类
__all__ = [
    'AdvancedAnalyticsEngine',
    'MultiFactorAnalyzer',
    'RiskAttributionAnalyzer',
    'MLIntegrationAnalyzer',
    'CustomMetricsCalculator',
    'FactorData',
    'RiskAttribution'
]
