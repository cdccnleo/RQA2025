"""多维度筛选器实现"""

import pandas as pd
from typing import Dict, List, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseFilter(ABC):
    """筛选器基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def apply(self, universe: List[str], market_data: pd.DataFrame) -> List[str]:
        """应用筛选条件"""

    @abstractmethod
    def calculate_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算相关指标"""


class LiquidityFilter(BaseFilter):
    """流动性筛选器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_daily_volume = config.get('min_daily_volume', 1000000)
        self.min_turnover_rate = config.get('min_turnover_rate', 0.01)
        self.min_market_cap = config.get('min_market_cap', 1000000000)

    def apply(self, universe: List[str], market_data: pd.DataFrame) -> List[str]:
        """应用流动性筛选"""
        if market_data.empty:
            return []

        metrics = self.calculate_metrics(market_data)
        filtered_stocks = []

        for stock in universe:
            if stock in metrics.index:
                stock_metrics = metrics.loc[stock]
                if self._pass_filter(stock_metrics):
                    filtered_stocks.append(stock)

        logger.info(f"流动性筛选: {len(universe)} -> {len(filtered_stocks)}")
        return filtered_stocks

    def calculate_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算流动性指标"""
        metrics = pd.DataFrame()

        if 'volume' in market_data.columns:
            metrics['daily_volume'] = market_data['volume']

        if 'turnover_rate' in market_data.columns:
            metrics['turnover_rate'] = market_data['turnover_rate']

        if 'market_cap' in market_data.columns:
            metrics['market_cap'] = market_data['market_cap']

        return metrics

    def _pass_filter(self, metrics: pd.Series) -> bool:
        """通过流动性筛选"""
        return (
            metrics.get('daily_volume', 0) >= self.min_daily_volume and
            metrics.get('turnover_rate', 0) >= self.min_turnover_rate and
            metrics.get('market_cap', 0) >= self.min_market_cap
        )


class VolatilityFilter(BaseFilter):
    """波动率筛选器"""

    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.max_volatility = config.get('max_volatility', 0.5)
        self.max_beta = config.get('max_beta', 2.0)
        self.min_sharpe_ratio = config.get('min_sharpe_ratio', 0.1)

    def apply(self, universe: List[str], market_data: pd.DataFrame) -> List[str]:
        """应用波动率筛选"""
        if market_data.empty:
            return []

        metrics = self.calculate_metrics(market_data)
        filtered_stocks = []

        for stock in universe:
            if stock in metrics.index:
                stock_metrics = metrics.loc[stock]
                if self._pass_filter(stock_metrics):
                    filtered_stocks.append(stock)

        logger.info(f"波动率筛选: {len(universe)} -> {len(filtered_stocks)}")
        return filtered_stocks

    def calculate_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算波动率指标"""
        metrics = pd.DataFrame()

        if 'returns' in market_data.columns:
            # 计算历史波动率
            metrics['volatility'] = market_data['returns'].rolling(20).std()

        if 'beta' in market_data.columns:
            metrics['beta'] = market_data['beta']

        if 'sharpe_ratio' in market_data.columns:
            metrics['sharpe_ratio'] = market_data['sharpe_ratio']

        return metrics

    def _pass_filter(self, metrics: pd.Series) -> bool:
        """通过波动率筛选"""
        return (
            metrics.get('volatility', 0) <= self.max_volatility and
            metrics.get('beta', 1.0) <= self.max_beta and
            metrics.get('sharpe_ratio', 0) >= self.min_sharpe_ratio
        )


class FundamentalFilter(BaseFilter):
    """基本面筛选器"""

    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.min_roe = config.get('min_roe', 0.05)
        self.max_pe = config.get('max_pe', 50)
        self.max_pb = config.get('max_pb', 5)
        self.min_profit_growth = config.get('min_profit_growth', 0.05)

    def apply(self, universe: List[str], market_data: pd.DataFrame) -> List[str]:
        """应用基本面筛选"""
        if market_data.empty:
            return []

        metrics = self.calculate_metrics(market_data)
        filtered_stocks = []

        for stock in universe:
            if stock in metrics.index:
                stock_metrics = metrics.loc[stock]
                if self._pass_filter(stock_metrics):
                    filtered_stocks.append(stock)

        logger.info(f"基本面筛选: {len(universe)} -> {len(filtered_stocks)}")
        return filtered_stocks

    def calculate_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算基本面指标"""
        metrics = pd.DataFrame()

        if 'roe' in market_data.columns:
            metrics['roe'] = market_data['roe']

        if 'pe' in market_data.columns:
            metrics['pe'] = market_data['pe']

        if 'pb' in market_data.columns:
            metrics['pb'] = market_data['pb']

        if 'profit_growth' in market_data.columns:
            metrics['profit_growth'] = market_data['profit_growth']

        return metrics

    def _pass_filter(self, metrics: pd.Series) -> bool:
        """通过基本面筛选"""
        return (
            metrics.get('roe', 0) >= self.min_roe and
            metrics.get('pe', 0) <= self.max_pe and
            metrics.get('pb', 0) <= self.max_pb and
            metrics.get('profit_growth', 0) >= self.min_profit_growth
        )


class TechnicalFilter(BaseFilter):
    """技术面筛选器"""

    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.rsi_min = config.get('rsi_min', 30)
        self.rsi_max = config.get('rsi_max', 70)
        self.macd_threshold = config.get('macd_threshold', 0.1)
        self.ma_trend_threshold = config.get('ma_trend_threshold', 0)

    def apply(self, universe: List[str], market_data: pd.DataFrame) -> List[str]:
        """应用技术面筛选"""
        if market_data.empty:
            return []

        metrics = self.calculate_metrics(market_data)
        filtered_stocks = []

        for stock in universe:
            if stock in metrics.index:
                stock_metrics = metrics.loc[stock]
                if self._pass_filter(stock_metrics):
                    filtered_stocks.append(stock)

        logger.info(f"技术面筛选: {len(universe)} -> {len(filtered_stocks)}")
        return filtered_stocks

    def calculate_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        metrics = pd.DataFrame()

        if 'rsi' in market_data.columns:
            metrics['rsi'] = market_data['rsi']

        if 'macd' in market_data.columns:
            metrics['macd'] = market_data['macd']

        if 'ma_trend' in market_data.columns:
            metrics['ma_trend'] = market_data['ma_trend']

        return metrics

    def _pass_filter(self, metrics: pd.Series) -> bool:
        """通过技术面筛选"""
        return (
            self.rsi_min <= metrics.get('rsi', 50) <= self.rsi_max and
            abs(metrics.get('macd', 0)) <= self.macd_threshold and
            metrics.get('ma_trend', 0) > self.ma_trend_threshold
        )


class SentimentFilter(BaseFilter):
    """情感面筛选器"""

    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.min_sentiment_score = config.get('min_sentiment_score', 0.3)
        self.max_news_volume = config.get('max_news_volume', 100)

    def apply(self, universe: List[str], market_data: pd.DataFrame) -> List[str]:
        """应用情感面筛选"""
        if market_data.empty:
            return []

        metrics = self.calculate_metrics(market_data)
        filtered_stocks = []

        for stock in universe:
            if stock in metrics.index:
                stock_metrics = metrics.loc[stock]
                if self._pass_filter(stock_metrics):
                    filtered_stocks.append(stock)

        logger.info(f"情感面筛选: {len(universe)} -> {len(filtered_stocks)}")
        return filtered_stocks

    def calculate_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算情感指标"""
        metrics = pd.DataFrame()

        if 'sentiment_score' in market_data.columns:
            metrics['sentiment_score'] = market_data['sentiment_score']

        if 'news_volume' in market_data.columns:
            metrics['news_volume'] = market_data['news_volume']

        return metrics

    def _pass_filter(self, metrics: pd.Series) -> bool:
        """通过情感面筛选"""
        return (
            metrics.get('sentiment_score', 0.5) >= self.min_sentiment_score and
            metrics.get('news_volume', 0) <= self.max_news_volume
        )


class RiskFilter(BaseFilter):
    """风险面筛选器"""

    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.max_credit_risk = config.get('max_credit_risk', 0.3)
        self.max_concentration_risk = config.get('max_concentration_risk', 0.2)
        self.min_rating = config.get('min_rating', 'BBB')

    def apply(self, universe: List[str], market_data: pd.DataFrame) -> List[str]:
        """应用风险面筛选"""
        if market_data.empty:
            return []

        metrics = self.calculate_metrics(market_data)
        filtered_stocks = []

        for stock in universe:
            if stock in metrics.index:
                stock_metrics = metrics.loc[stock]
                if self._pass_filter(stock_metrics):
                    filtered_stocks.append(stock)

        logger.info(f"风险面筛选: {len(universe)} -> {len(filtered_stocks)}")
        return filtered_stocks

    def calculate_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算风险指标"""
        metrics = pd.DataFrame()

        if 'credit_risk' in market_data.columns:
            metrics['credit_risk'] = market_data['credit_risk']

        if 'concentration_risk' in market_data.columns:
            metrics['concentration_risk'] = market_data['concentration_risk']

        if 'rating' in market_data.columns:
            metrics['rating'] = market_data['rating']

        return metrics

    def _pass_filter(self, metrics: pd.Series) -> bool:
        """通过风险面筛选"""
        return (
            metrics.get('credit_risk', 0) <= self.max_credit_risk and
            metrics.get('concentration_risk', 0) <= self.max_concentration_risk and
            self._is_rating_acceptable(metrics.get('rating', 'CCC'))
        )

    def _is_rating_acceptable(self, rating: str) -> bool:
        """检查评级是否可接受"""
        rating_hierarchy = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
        min_rating_index = rating_hierarchy.index(self.min_rating)
        current_rating_index = rating_hierarchy.index(
            rating) if rating in rating_hierarchy else len(rating_hierarchy) - 1
        return current_rating_index <= min_rating_index


class MultiDimensionalFilter:
    """多维度筛选器"""

    def __init__(self, config: Dict[str, any]):
        self.filters = {
            'liquidity': LiquidityFilter(config.get('liquidity', {})),
            'volatility': VolatilityFilter(config.get('volatility', {})),
            'fundamental': FundamentalFilter(config.get('fundamental', {})),
            'technical': TechnicalFilter(config.get('technical', {})),
            'sentiment': SentimentFilter(config.get('sentiment', {})),
            'risk': RiskFilter(config.get('risk', {}))
        }

    def filter_stocks(self, universe: List[str], market_data: pd.DataFrame) -> Dict[str, List[str]]:
        """多维度筛选股票"""
        results = {}

        for filter_name, filter_obj in self.filters.items():
            results[filter_name] = filter_obj.apply(universe, market_data)

        return results

    def get_intersection(self, filter_results: Dict[str, List[str]]) -> List[str]:
        """获取所有筛选条件的交集"""
        if not filter_results:
            return []

        # 转换为集合进行交集运算
        sets = [set(result) for result in filter_results.values() if result]
        if not sets:
            return []

        intersection = sets[0]
        for s in sets[1:]:
            intersection = intersection.intersection(s)

        return list(intersection)

    def get_union(self, filter_results: Dict[str, List[str]]) -> List[str]:
        """获取所有筛选条件的并集"""
        if not filter_results:
            return []

        union = set()
        for result in filter_results.values():
            union.update(result)

        return list(union)
