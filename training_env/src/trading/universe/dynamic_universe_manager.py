"""动态股票池管理器"""

import pandas as pd
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """市场状态枚举"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class UniverseUpdate:
    """股票池更新结果"""
    updated: bool
    market_state: Optional[MarketState] = None
    active_factors: Optional[List[str]] = None
    changes: Optional[Dict[str, List[str]]] = None
    reason: str = ""


class DynamicUniverseManager:
    """动态股票池管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_universe = set()  # 基础股票池
        self.active_universe = set()  # 活跃股票池
        self.candidate_pool = set()  # 候选股票池
        self.removal_pool = set()  # 移除股票池

        # 动态调整参数
        self.update_frequency = config.get('update_frequency', 'daily')
        self.min_liquidity = config.get('min_liquidity', 1000000)
        self.max_volatility = config.get('max_volatility', 0.5)
        self.min_market_cap = config.get('min_market_cap', 1000000000)

        # 筛选器 - 暂时注释掉，避免导入问题
        # self.filters = {
        #     'liquidity': LiquidityFilter(config.get('liquidity', {})),
        #     'volatility': VolatilityFilter(config.get('volatility', {})),
        #     'fundamental': FundamentalFilter(config.get('fundamental', {})),
        #     'technical': TechnicalFilter(config.get('technical', {})),
        #     'sentiment': SentimentFilter(config.get('sentiment', {})),
        #     'risk': RiskFilter(config.get('risk', {}))
        # }

        # 更新历史
        self.update_history = []

    def update_universe(self, market_data: pd.DataFrame) -> Dict[str, List[str]]:
        """动态更新股票池"""
        logger.info("开始更新动态股票池")

        # 1. 流动性筛选
        liquid_stocks = self._filter_by_liquidity(market_data)
        logger.info(f"流动性筛选后股票数量: {len(liquid_stocks)}")

        # 2. 波动率筛选
        stable_stocks = self._filter_by_volatility(market_data)
        logger.info(f"波动率筛选后股票数量: {len(stable_stocks)}")

        # 3. 基本面筛选
        fundamental_stocks = self._filter_by_fundamentals(market_data)
        logger.info(f"基本面筛选后股票数量: {len(fundamental_stocks)}")

        # 4. 技术面筛选
        technical_stocks = self._filter_by_technical(market_data)
        logger.info(f"技术面筛选后股票数量: {len(technical_stocks)}")

        # 5. 综合评分
        scored_stocks = self._calculate_composite_score(
            liquid_stocks, stable_stocks, fundamental_stocks, technical_stocks
        )
        logger.info(f"综合评分后股票数量: {len(scored_stocks)}")

        # 6. 更新股票池
        universe_changes = self._update_pools(scored_stocks)

        # 7. 记录更新历史
        self._record_update(universe_changes)

        return universe_changes

    def _filter_by_liquidity(self, market_data: pd.DataFrame) -> Set[str]:
        """流动性筛选"""
        if market_data.empty:
            return set()

        # 计算流动性指标
        liquidity_metrics = self._calculate_liquidity_metrics(market_data)

        # 应用筛选条件
        filtered_stocks = set()
        for stock in liquidity_metrics.index:
            if self._pass_liquidity_filter(liquidity_metrics.loc[stock]):
                filtered_stocks.add(stock)

        return filtered_stocks

    def _filter_by_volatility(self, market_data: pd.DataFrame) -> Set[str]:
        """波动率筛选"""
        if market_data.empty:
            return set()

        # 计算波动率指标
        volatility_metrics = self._calculate_volatility_metrics(market_data)

        # 应用筛选条件
        filtered_stocks = set()
        for stock in volatility_metrics.index:
            if self._pass_volatility_filter(volatility_metrics.loc[stock]):
                filtered_stocks.add(stock)

        return filtered_stocks

    def _filter_by_fundamentals(self, market_data: pd.DataFrame) -> Set[str]:
        """基本面筛选"""
        if market_data.empty:
            return set()

        # 计算基本面指标
        fundamental_metrics = self._calculate_fundamental_metrics(market_data)

        # 应用筛选条件
        filtered_stocks = set()
        for stock in fundamental_metrics.index:
            if self._pass_fundamental_filter(fundamental_metrics.loc[stock]):
                filtered_stocks.add(stock)

        return filtered_stocks

    def _filter_by_technical(self, market_data: pd.DataFrame) -> Set[str]:
        """技术面筛选"""
        if market_data.empty:
            return set()

        # 计算技术指标
        technical_metrics = self._calculate_technical_metrics(market_data)

        # 应用筛选条件
        filtered_stocks = set()
        for stock in technical_metrics.index:
            if self._pass_technical_filter(technical_metrics.loc[stock]):
                filtered_stocks.add(stock)

        return filtered_stocks

    def _calculate_composite_score(self, *filtered_stocks) -> Dict[str, float]:
        """计算综合评分"""
        # 获取所有筛选条件的交集
        intersection = set.intersection(*filtered_stocks) if filtered_stocks else set()

        scores = {}
        for stock in intersection:
            score = (
                self._liquidity_score(stock) * 0.3 +
                self._volatility_score(stock) * 0.2 +
                self._fundamental_score(stock) * 0.3 +
                self._technical_score(stock) * 0.2
            )
            scores[stock] = score

        return scores

    def _update_pools(self, scored_stocks: Dict[str, float]) -> Dict[str, List[str]]:
        """更新股票池"""
        # 按评分排序
        sorted_stocks = sorted(scored_stocks.items(), key=lambda x: x[1], reverse=True)

        # 确定股票池大小
        max_universe_size = self.config.get('max_universe_size', 100)

        # 更新活跃股票池
        new_active_universe = set()
        for stock, score in sorted_stocks[:max_universe_size]:
            new_active_universe.add(stock)

        # 计算变化
        added_stocks = new_active_universe - self.active_universe
        removed_stocks = self.active_universe - new_active_universe

        # 更新股票池
        self.active_universe = new_active_universe

        return {
            'active_universe': list(self.active_universe),
            'added_stocks': list(added_stocks),
            'removed_stocks': list(removed_stocks),
            'candidate_pool': list(self.candidate_pool),
            'removal_pool': list(self.removal_pool)
        }

    def _record_update(self, changes: Dict[str, List[str]]):
        """记录更新历史"""
        # 根据changes更新active_universe
        if 'active_universe' in changes:
            self.active_universe = set(changes['active_universe'])

        update_record = {
            'timestamp': datetime.now(),
            'changes': changes,
            'active_universe_size': len(self.active_universe),
            'added_count': len(changes.get('added_stocks', [])),
            'removed_count': len(changes.get('removed_stocks', []))
        }

        self.update_history.append(update_record)

        # 保留最近100条记录
        if len(self.update_history) > 100:
            self.update_history.pop(0)

    def get_universe_statistics(self) -> Dict[str, Any]:
        """获取股票池统计信息"""
        return {
            'active_universe_size': len(self.active_universe),
            'candidate_pool_size': len(self.candidate_pool),
            'removal_pool_size': len(self.removal_pool),
            'last_update': self.update_history[-1]['timestamp'] if self.update_history else None,
            'update_count': len(self.update_history)
        }

    # 辅助方法
    def _calculate_liquidity_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算流动性指标"""
        # 这里应该实现具体的流动性指标计算
        # 简化实现
        return pd.DataFrame({
            'daily_volume': market_data['volume'] if 'volume' in market_data.columns else [0] * len(market_data),
            'turnover_rate': market_data['turnover_rate'] if 'turnover_rate' in market_data.columns else [0] * len(market_data),
            'market_cap': market_data['market_cap'] if 'market_cap' in market_data.columns else [0] * len(market_data)
        }, index=market_data.index)

    def _calculate_volatility_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算波动率指标"""
        # 简化实现
        return pd.DataFrame({
            'volatility': market_data['volatility'] if 'volatility' in market_data.columns else [0] * len(market_data),
            'beta': market_data['beta'] if 'beta' in market_data.columns else [1.0] * len(market_data),
            'sharpe_ratio': market_data['sharpe_ratio'] if 'sharpe_ratio' in market_data.columns else [0] * len(market_data)
        }, index=market_data.index)

    def _calculate_fundamental_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算基本面指标"""
        # 简化实现
        return pd.DataFrame({
            'roe': [0] * len(market_data),
            'pe': [0] * len(market_data),
            'pb': [0] * len(market_data),
            'profit_growth': [0] * len(market_data)
        }, index=market_data.index)

    def _calculate_technical_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 简化实现
        return pd.DataFrame({
            'rsi': market_data['rsi'] if 'rsi' in market_data.columns else [50] * len(market_data),
            'macd': market_data['macd'] if 'macd' in market_data.columns else [0] * len(market_data),
            'ma_trend': market_data['ma_trend'] if 'ma_trend' in market_data.columns else [0] * len(market_data)
        }, index=market_data.index)

    def _pass_liquidity_filter(self, metrics: pd.Series) -> bool:
        """通过流动性筛选"""
        daily_volume = metrics.get('daily_volume', 0) if 'daily_volume' in metrics else 0
        turnover_rate = metrics.get('turnover_rate', 0) if 'turnover_rate' in metrics else 0
        market_cap = metrics.get('market_cap', 0) if 'market_cap' in metrics else 0

        # 确保值是标量
        if hasattr(daily_volume, 'item'):
            try:
                daily_volume = daily_volume.item()
            except ValueError:
                daily_volume = float(daily_volume.iloc[0]) if hasattr(
                    daily_volume, 'iloc') else float(daily_volume)
        if hasattr(turnover_rate, 'item'):
            try:
                turnover_rate = turnover_rate.item()
            except ValueError:
                turnover_rate = float(turnover_rate.iloc[0]) if hasattr(
                    turnover_rate, 'iloc') else float(turnover_rate)
        if hasattr(market_cap, 'item'):
            try:
                market_cap = market_cap.item()
            except ValueError:
                market_cap = float(market_cap.iloc[0]) if hasattr(
                    market_cap, 'iloc') else float(market_cap)

        return (
            daily_volume >= self.min_liquidity and
            turnover_rate >= 0.01 and
            market_cap >= self.min_market_cap
        )

    def _pass_volatility_filter(self, metrics: pd.Series) -> bool:
        """通过波动率筛选"""
        volatility = metrics.get('volatility', 0) if 'volatility' in metrics else 0
        beta = metrics.get('beta', 1.0) if 'beta' in metrics else 1.0
        sharpe_ratio = metrics.get('sharpe_ratio', 0) if 'sharpe_ratio' in metrics else 0

        # 确保值是标量
        if hasattr(volatility, 'item'):
            try:
                volatility = volatility.item()
            except ValueError:
                volatility = float(volatility.iloc[0]) if hasattr(
                    volatility, 'iloc') else float(volatility)
        if hasattr(beta, 'item'):
            try:
                beta = beta.item()
            except ValueError:
                beta = float(beta.iloc[0]) if hasattr(beta, 'iloc') else float(beta)
        if hasattr(sharpe_ratio, 'item'):
            try:
                sharpe_ratio = sharpe_ratio.item()
            except ValueError:
                sharpe_ratio = float(sharpe_ratio.iloc[0]) if hasattr(
                    sharpe_ratio, 'iloc') else float(sharpe_ratio)

        return (
            volatility <= self.max_volatility and
            beta <= 2.0 and  # 恢复原来的beta阈值
            sharpe_ratio >= 0.1
        )

    def _pass_fundamental_filter(self, metrics: pd.Series) -> bool:
        """通过基本面筛选"""
        roe = metrics.get('roe', 0) if 'roe' in metrics else 0
        pe = metrics.get('pe', 0) if 'pe' in metrics else 0
        pb = metrics.get('pb', 0) if 'pb' in metrics else 0
        profit_growth = metrics.get('profit_growth', 0) if 'profit_growth' in metrics else 0

        # 确保值是标量
        if hasattr(roe, 'item'):
            try:
                roe = roe.item()
            except ValueError:
                roe = float(roe.iloc[0]) if hasattr(roe, 'iloc') else float(roe)
        if hasattr(pe, 'item'):
            try:
                pe = pe.item()
            except ValueError:
                pe = float(pe.iloc[0]) if hasattr(pe, 'iloc') else float(pe)
        if hasattr(pb, 'item'):
            try:
                pb = pb.item()
            except ValueError:
                pb = float(pb.iloc[0]) if hasattr(pb, 'iloc') else float(pb)
        if hasattr(profit_growth, 'item'):
            try:
                profit_growth = profit_growth.item()
            except ValueError:
                profit_growth = float(profit_growth.iloc[0]) if hasattr(
                    profit_growth, 'iloc') else float(profit_growth)

        return (
            roe >= 0.05 and
            pe <= 50 and
            pb <= 5 and
            profit_growth >= 0.05
        )

    def _pass_technical_filter(self, metrics: pd.Series) -> bool:
        """通过技术面筛选"""
        rsi = metrics.get('rsi', 50) if 'rsi' in metrics else 50
        macd = metrics.get('macd', 0) if 'macd' in metrics else 0
        ma_trend = metrics.get('ma_trend', 0) if 'ma_trend' in metrics else 0

        # 确保值是标量
        if hasattr(rsi, 'item'):
            try:
                rsi = rsi.item()
            except ValueError:
                rsi = float(rsi.iloc[0]) if hasattr(rsi, 'iloc') else float(rsi)
        if hasattr(macd, 'item'):
            try:
                macd = macd.item()
            except ValueError:
                macd = float(macd.iloc[0]) if hasattr(macd, 'iloc') else float(macd)
        if hasattr(ma_trend, 'item'):
            try:
                ma_trend = ma_trend.item()
            except ValueError:
                ma_trend = float(ma_trend.iloc[0]) if hasattr(ma_trend, 'iloc') else float(ma_trend)

        return (
            30 <= rsi <= 70 and
            abs(macd) <= 0.1 and
            ma_trend > 0
        )

    def _liquidity_score(self, stock: str) -> float:
        """计算流动性评分"""
        # 简化实现
        return 0.8

    def _volatility_score(self, stock: str) -> float:
        """计算波动率评分"""
        # 简化实现
        return 0.7

    def _fundamental_score(self, stock: str) -> float:
        """计算基本面评分"""
        # 简化实现
        return 0.9

    def _technical_score(self, stock: str) -> float:
        """计算技术面评分"""
        # 简化实现
        return 0.6
