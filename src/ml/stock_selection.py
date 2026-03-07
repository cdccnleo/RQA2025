"""
股票选择与质量筛选模块
提供股票选择策略和质量评估功能
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# 股票行业分类映射
STOCK_SECTORS = {
    # 银行
    '000001': '银行',  # 平安银行
    '600000': '银行',  # 浦发银行
    '600016': '银行',  # 民生银行
    
    # 电子
    '002837': '电子',  # 英维克
    '000725': '电子',  # 京东方A
    '002475': '电子',  # 立讯精密
    
    # 通信
    '688702': '通信',  # 盛科通信
    '600050': '通信',  # 中国联通
    '000063': '通信',  # 中兴通讯
    
    # 金融
    '000987': '金融',  # 越秀资本
    '600030': '金融',  # 中信证券
    '600837': '金融',  # 海通证券
    
    # 新能源
    '301327': '新能源',  # 华宝新能
    '300750': '新能源',  # 宁德时代
    '002594': '新能源',  # 比亚迪
    
    # 传媒
    '000917': '传媒',  # 电广传媒
    '002027': '传媒',  # 分众传媒
    '300413': '传媒',  # 芒果超媒
    
    # 医药
    '600276': '医药',  # 恒瑞医药
    '000538': '医药',  # 云南白药
    '300760': '医药',  # 迈瑞医疗
    
    # 消费
    '000858': '消费',  # 五粮液
    '600519': '消费',  # 贵州茅台
    '000333': '消费',  # 美的集团
    
    # 科技
    '688981': '科技',  # 中芯国际
    '603501': '科技',  # 韦尔股份
    '002371': '科技',  # 北方华创
    
    # 能源
    '601857': '能源',  # 中国石油
    '600028': '能源',  # 中国石化
    '601088': '能源',  # 中国神华
}


class StockQualityFilter:
    """股票质量筛选器"""
    
    def __init__(self, 
                 min_completeness: float = 0.95,
                 min_volume: float = 1e7,  # 1000万
                 min_data_points: int = 500,
                 min_listing_days: int = 730):  # 2年
        """
        初始化质量筛选器
        
        Args:
            min_completeness: 最小数据完整率
            min_volume: 最小平均成交量
            min_data_points: 最小数据点数
            min_listing_days: 最小上市天数
        """
        self.min_completeness = min_completeness
        self.min_volume = float(min_volume)  # 转换为float避免Decimal比较问题
        self.min_data_points = min_data_points
        self.min_listing_days = min_listing_days
    
    def _to_float(self, value):
        """安全转换为float，处理Decimal和其他数值类型"""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    
    def filter_stock(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, Dict[str, any]]:
        """
        筛选单只股票
        
        Args:
            df: 股票数据DataFrame
            symbol: 股票代码
            
        Returns:
            (是否通过筛选, 质量指标字典)
        """
        metrics = {
            'symbol': symbol,
            'passed': False,
            'reasons': []
        }
        
        if df is None or len(df) == 0:
            metrics['reasons'].append('无数据')
            return False, metrics
        
        # 1. 检查数据完整率
        completeness = df.notna().mean().mean()
        metrics['completeness'] = completeness
        if completeness < self.min_completeness:
            metrics['reasons'].append(f'数据完整率不足: {completeness:.2%} < {self.min_completeness:.2%}')
        
        # 2. 检查数据点数量
        n_points = len(df)
        metrics['data_points'] = n_points
        if n_points < self.min_data_points:
            metrics['reasons'].append(f'数据点不足: {n_points} < {self.min_data_points}')
        
        # 3. 检查平均成交量
        if 'volume' in df.columns:
            avg_volume = self._to_float(df['volume'].mean())
            metrics['avg_volume'] = avg_volume
            if avg_volume < self.min_volume:
                metrics['reasons'].append(f'成交量不足: {avg_volume:.0f} < {self.min_volume:.0f}')
        
        # 4. 检查数据时间跨度
        if len(df) > 0:
            date_range = (df.index[-1] - df.index[0]).days
            metrics['date_range_days'] = date_range
            if date_range < self.min_listing_days:
                metrics['reasons'].append(f'数据时间跨度不足: {date_range}天 < {self.min_listing_days}天')
        
        # 5. 检查价格波动性（排除长期停牌股票）
        if 'close' in df.columns and len(df) > 1:
            try:
                close_prices = self._to_float(df['close'])
                price_changes = pd.Series(close_prices).pct_change().dropna()
                if len(price_changes) > 0:
                    volatility = self._to_float(price_changes.std())
                    metrics['volatility'] = volatility
                    # 如果波动率接近0，可能是长期停牌
                    if volatility < 0.001:  # 日波动率小于0.1%
                        metrics['reasons'].append('价格波动性过低，可能长期停牌')
            except Exception:
                pass  # 跳过波动性检查
        
        # 判断是否通过筛选
        if len(metrics['reasons']) == 0:
            metrics['passed'] = True
            logger.info(f"股票 {symbol} 通过质量筛选: 完整率={completeness:.2%}, 数据点={n_points}")
        else:
            logger.warning(f"股票 {symbol} 未通过质量筛选: {', '.join(metrics['reasons'])}")
        
        return metrics['passed'], metrics
    
    def filter_stocks_batch(self, stocks_data: Dict[str, pd.DataFrame]) -> Tuple[List[str], Dict[str, Dict]]:
        """
        批量筛选股票
        
        Args:
            stocks_data: 股票数据字典 {symbol: df}
            
        Returns:
            (通过筛选的股票列表, 所有股票的质量指标)
        """
        passed_stocks = []
        all_metrics = {}
        
        for symbol, df in stocks_data.items():
            passed, metrics = self.filter_stock(df, symbol)
            all_metrics[symbol] = metrics
            if passed:
                passed_stocks.append(symbol)
        
        logger.info(f"批量筛选完成: {len(passed_stocks)}/{len(stocks_data)} 只股票通过筛选")
        return passed_stocks, all_metrics


class StockSelector:
    """股票选择器"""
    
    def __init__(self, quality_filter: Optional[StockQualityFilter] = None):
        """
        初始化股票选择器
        
        Args:
            quality_filter: 质量筛选器
        """
        self.quality_filter = quality_filter or StockQualityFilter()
    
    def select_by_sector(self, symbols: List[str], max_per_sector: int = 2,
                        sector_weights: Optional[Dict[str, float]] = None) -> List[str]:
        """
        按行业分散选择股票
        
        Args:
            symbols: 候选股票列表
            max_per_sector: 每行业最大股票数
            sector_weights: 行业权重配置
            
        Returns:
            选中的股票列表
        """
        if sector_weights is None:
            # 默认行业权重
            sector_weights = {
                '银行': 0.15,
                '电子': 0.20,
                '通信': 0.15,
                '金融': 0.10,
                '新能源': 0.20,
                '传媒': 0.10,
                '医药': 0.15,
                '消费': 0.15,
                '科技': 0.15,
                '能源': 0.10,
                '其他': 0.10
            }
        
        sector_count = {}
        selected = []
        
        # 按行业权重排序股票
        symbol_sectors = [(s, STOCK_SECTORS.get(s, '其他')) for s in symbols]
        symbol_sectors.sort(key=lambda x: sector_weights.get(x[1], 0.1), reverse=True)
        
        for symbol, sector in symbol_sectors:
            current_count = sector_count.get(sector, 0)
            target_count = max_per_sector
            
            if current_count < target_count:
                selected.append(symbol)
                sector_count[sector] = current_count + 1
        
        logger.info(f"行业分散选择: 从 {len(symbols)} 只中选出 {len(selected)} 只, "
                   f"覆盖 {len(sector_count)} 个行业")
        return selected
    
    def select_by_volatility(self, stocks_data: Dict[str, pd.DataFrame],
                           n_low: int = 3, n_mid: int = 3, n_high: int = 3) -> List[str]:
        """
        按波动性分层选择股票
        
        Args:
            stocks_data: 股票数据字典
            n_low: 低波动股票数量
            n_mid: 中波动股票数量
            n_high: 高波动股票数量
            
        Returns:
            选中的股票列表
        """
        volatility_scores = {}
        
        for symbol, df in stocks_data.items():
            if df is None or len(df) < 30:
                continue
            
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                # 年化波动率
                volatility = returns.std() * np.sqrt(252)
                volatility_scores[symbol] = volatility
        
        if not volatility_scores:
            logger.warning("没有可用的波动性数据")
            return []
        
        # 按波动性排序
        sorted_stocks = sorted(volatility_scores.items(), key=lambda x: x[1])
        
        n = len(sorted_stocks)
        
        # 分层选择
        low_vol = [s for s, _ in sorted_stocks[:max(n//3, n_low)]]
        mid_vol = [s for s, _ in sorted_stocks[n//3:2*n//3]]
        high_vol = [s for s, _ in sorted_stocks[2*n//3:]]
        
        selected = (low_vol[:n_low] + mid_vol[:n_mid] + high_vol[:n_high])
        
        logger.info(f"波动性分层选择: 低波动={len(low_vol[:n_low])}, "
                   f"中波动={len(mid_vol[:n_mid])}, 高波动={len(high_vol[:n_high])}")
        return selected
    
    def select_diverse_portfolio(self, stocks_data: Dict[str, pd.DataFrame],
                                n_stocks: int = 10,
                                use_quality_filter: bool = True) -> List[str]:
        """
        选择多样化的股票组合
        
        Args:
            stocks_data: 股票数据字典
            n_stocks: 目标股票数量
            use_quality_filter: 是否使用质量筛选
            
        Returns:
            选中的股票列表
        """
        available_symbols = list(stocks_data.keys())
        
        # 第一步：质量筛选
        if use_quality_filter:
            passed_symbols, _ = self.quality_filter.filter_stocks_batch(stocks_data)
            if len(passed_symbols) < n_stocks // 2:
                logger.warning(f"通过质量筛选的股票数量不足: {len(passed_symbols)}")
                passed_symbols = available_symbols  # 回退到全部股票
        else:
            passed_symbols = available_symbols
        
        # 第二步：行业分散选择
        sector_selected = self.select_by_sector(passed_symbols, max_per_sector=2)
        
        # 第三步：波动性分层选择（补充）
        if len(sector_selected) < n_stocks:
            remaining_data = {s: stocks_data[s] for s in passed_symbols 
                            if s not in sector_selected}
            if remaining_data:
                vol_selected = self.select_by_volatility(
                    remaining_data, 
                    n_low=max(1, (n_stocks - len(sector_selected)) // 3),
                    n_mid=max(1, (n_stocks - len(sector_selected)) // 3),
                    n_high=max(1, (n_stocks - len(sector_selected)) // 3)
                )
                sector_selected.extend([s for s in vol_selected if s not in sector_selected])
        
        # 限制数量
        final_selection = sector_selected[:n_stocks]
        
        logger.info(f"多样化组合选择完成: 选中 {len(final_selection)} 只股票")
        return final_selection
    
    def analyze_stock_correlation(self, stocks_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        分析股票之间的相关性
        
        Args:
            stocks_data: 股票数据字典
            
        Returns:
            相关性矩阵
        """
        # 提取收益率数据
        returns_data = {}
        for symbol, df in stocks_data.items():
            if df is not None and 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                if len(returns) > 0:
                    returns_data[symbol] = returns
        
        if not returns_data:
            logger.warning("没有可用的收益率数据")
            return pd.DataFrame()
        
        # 创建收益率DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # 计算相关性矩阵
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def get_stock_recommendations(self, stocks_data: Dict[str, pd.DataFrame],
                                 n_recommendations: int = 10) -> Dict[str, any]:
        """
        获取股票推荐
        
        Args:
            stocks_data: 股票数据字典
            n_recommendations: 推荐数量
            
        Returns:
            推荐结果字典
        """
        # 质量筛选
        passed_symbols, quality_metrics = self.quality_filter.filter_stocks_batch(stocks_data)
        
        # 多样化选择
        recommended = self.select_diverse_portfolio(
            {s: stocks_data[s] for s in passed_symbols},
            n_stocks=n_recommendations
        )
        
        # 相关性分析
        correlation_matrix = self.analyze_stock_correlation(
            {s: stocks_data[s] for s in recommended}
        )
        
        return {
            'recommended_stocks': recommended,
            'quality_metrics': quality_metrics,
            'correlation_matrix': correlation_matrix.to_dict() if not correlation_matrix.empty else {},
            'n_passed_quality': len(passed_symbols),
            'n_total': len(stocks_data)
        }


# 便捷函数
def get_stock_sector(symbol: str) -> str:
    """获取股票行业"""
    return STOCK_SECTORS.get(symbol, '其他')


def filter_stocks_by_quality(stocks_data: Dict[str, pd.DataFrame],
                            min_completeness: float = 0.95,
                            min_volume: float = 1e7,
                            min_data_points: int = 500) -> List[str]:
    """
    便捷函数：按质量筛选股票
    
    Args:
        stocks_data: 股票数据字典
        min_completeness: 最小数据完整率
        min_volume: 最小成交量
        min_data_points: 最小数据点数
        
    Returns:
        通过筛选的股票列表
    """
    quality_filter = StockQualityFilter(
        min_completeness=min_completeness,
        min_volume=min_volume,
        min_data_points=min_data_points
    )
    passed_stocks, _ = quality_filter.filter_stocks_batch(stocks_data)
    return passed_stocks


def select_diverse_stocks(stocks_data: Dict[str, pd.DataFrame],
                         n_stocks: int = 10) -> List[str]:
    """
    便捷函数：选择多样化股票组合
    
    Args:
        stocks_data: 股票数据字典
        n_stocks: 目标股票数量
        
    Returns:
        选中的股票列表
    """
    selector = StockSelector()
    return selector.select_diverse_portfolio(stocks_data, n_stocks=n_stocks)
