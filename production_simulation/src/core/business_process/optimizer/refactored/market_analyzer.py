"""
市场分析器

从 IntelligentBusinessProcessOptimizer 提取的市场分析功能
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from .models import ProcessContext

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """
    市场分析器
    
    职责：
    - 市场数据分析
    - 趋势预测
    - 流动性分析
    - 历史数据获取
    
    从 IntelligentBusinessProcessOptimizer 中提取的方法：
    - _analyze_market_intelligently → analyze_market
    - _get_historical_data → get_historical_data
    - _analyze_market_trend → analyze_trend
    - _analyze_market_liquidity → analyze_liquidity
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化市场分析器
        
        Args:
            config: 配置字典，包含分析参数
        """
        self.config = config or {}
        
        # 分析参数
        self.trend_window = self.config.get('trend_window', 20)
        self.liquidity_threshold = self.config.get('liquidity_threshold', 1000000)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
        # 状态管理
        self.cache: Dict[str, tuple] = {}
        self.last_analysis: Dict[str, Dict[str, Any]] = {}
        
        logger.info("MarketAnalyzer initialized with config: %s", self.config)
    
    async def analyze_market(self, context: ProcessContext) -> Dict[str, Any]:
        """
        智能市场分析（主入口）
        
        原方法: _analyze_market_intelligently
        
        Args:
            context: 流程上下文
            
        Returns:
            Dict: 市场分析结果
                - trend: 趋势分析
                - liquidity: 流动性分析
                - sentiment: 市场情绪
                - risk_level: 风险等级
        """
        logger.info("Starting market analysis for process %s", context.process_id)
        
        try:
            # 1. 获取市场数据
            market_data = context.market_data
            symbols = market_data.get('symbols', [])
            
            # 2. 并行分析各个维度
            analysis_tasks = []
            for symbol in symbols:
                # 获取历史数据
                historical_data = await self.get_historical_data(symbol)
                
                # 分析趋势
                trend_task = self.analyze_trend(
                    market_data={'symbol': symbol, 'data': historical_data},
                    predictions={}  # 从预测服务获取
                )
                analysis_tasks.append(trend_task)
            
            # 等待所有分析完成
            trend_results = await asyncio.gather(*analysis_tasks)
            
            # 3. 分析流动性
            liquidity_analysis = await self.analyze_liquidity(
                orders=context.orders
            )
            
            # 4. 综合分析结果
            analysis_result = {
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols,
                'trend_analysis': trend_results,
                'liquidity_analysis': liquidity_analysis,
                'market_sentiment': self._calculate_market_sentiment(trend_results),
                'risk_level': self._assess_market_risk_level(trend_results, liquidity_analysis),
                'recommendations': self._generate_market_recommendations(trend_results)
            }
            
            # 缓存结果
            self.last_analysis[context.process_id] = analysis_result
            
            logger.info("Market analysis completed for process %s", context.process_id)
            return analysis_result
            
        except Exception as e:
            logger.error("Market analysis failed: %s", str(e), exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }
    
    async def get_historical_data(
        self, 
        symbol: str, 
        period: int = 30
    ) -> pd.DataFrame:
        """
        获取历史数据
        
        原方法: _get_historical_data
        
        Args:
            symbol: 交易品种代码
            period: 历史周期（天数）
            
        Returns:
            pd.DataFrame: 历史数据
        """
        logger.debug("Fetching historical data for %s (period: %d days)", symbol, period)
        
        # 检查缓存
        cache_key = f"{symbol}_{period}"
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < 3600:  # 1小时缓存
                logger.debug("Using cached data for %s", symbol)
                return data
        
        try:
            # TODO: 从实际数据源获取
            # 这里用模拟数据示例
            dates = pd.date_range(end=datetime.now(), periods=period)
            data = pd.DataFrame({
                'date': dates,
                'open': np.random.randn(period).cumsum() + 100,
                'high': np.random.randn(period).cumsum() + 102,
                'low': np.random.randn(period).cumsum() + 98,
                'close': np.random.randn(period).cumsum() + 100,
                'volume': np.random.randint(1000000, 10000000, period)
            })
            
            # 更新缓存
            self.cache[cache_key] = (datetime.now(), data)
            
            logger.debug("Historical data fetched for %s: %d records", symbol, len(data))
            return data
            
        except Exception as e:
            logger.error("Failed to fetch historical data for %s: %s", symbol, str(e))
            return pd.DataFrame()
    
    async def analyze_trend(
        self, 
        market_data: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分析市场趋势
        
        原方法: _analyze_market_trend
        
        Args:
            market_data: 市场数据
            predictions: 预测数据
            
        Returns:
            Dict: 趋势分析结果
        """
        symbol = market_data.get('symbol')
        data = market_data.get('data')
        
        logger.debug("Analyzing trend for %s", symbol)
        
        if data is None or data.empty:
            return {'symbol': symbol, 'trend': 'unknown', 'confidence': 0.0}
        
        try:
            # 1. 计算移动平均
            data = data.copy()  # 避免修改原始数据
            data['ma_short'] = data['close'].rolling(window=5).mean()
            data['ma_long'] = data['close'].rolling(window=self.trend_window).mean()
            
            # 2. 判断趋势
            latest_short_ma = data['ma_short'].iloc[-1]
            latest_long_ma = data['ma_long'].iloc[-1]
            
            if latest_short_ma > latest_long_ma * 1.02:
                trend = 'uptrend'
                strength = 'strong'
            elif latest_short_ma > latest_long_ma:
                trend = 'uptrend'
                strength = 'weak'
            elif latest_short_ma < latest_long_ma * 0.98:
                trend = 'downtrend'
                strength = 'strong'
            elif latest_short_ma < latest_long_ma:
                trend = 'downtrend'
                strength = 'weak'
            else:
                trend = 'sideways'
                strength = 'neutral'
            
            # 3. 计算趋势置信度
            price_momentum = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5]
            volume_trend = data['volume'].iloc[-5:].mean() / data['volume'].mean()
            confidence = min(abs(price_momentum) * 10 + (volume_trend - 1), 1.0)
            
            # 4. 结合预测数据
            if predictions:
                predicted_direction = predictions.get('direction', trend)
                if predicted_direction == trend:
                    confidence = min(confidence * 1.2, 1.0)
            
            result = {
                'symbol': symbol,
                'trend': trend,
                'strength': strength,
                'confidence': round(confidence, 3),
                'short_ma': round(float(latest_short_ma), 2),
                'long_ma': round(float(latest_long_ma), 2),
                'price_momentum': round(float(price_momentum), 4),
                'volume_ratio': round(float(volume_trend), 2)
            }
            
            logger.debug("Trend analysis for %s: %s (%s, confidence: %.3f)", 
                        symbol, trend, strength, confidence)
            
            return result
            
        except Exception as e:
            logger.error("Trend analysis failed for %s: %s", symbol, str(e))
            return {
                'symbol': symbol,
                'trend': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def analyze_liquidity(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析市场流动性
        
        原方法: _analyze_market_liquidity
        
        Args:
            orders: 订单列表
            
        Returns:
            Dict: 流动性分析结果
        """
        logger.debug("Analyzing market liquidity for %d orders", len(orders))
        
        if not orders:
            return {
                'overall_liquidity': 'unknown',
                'liquidity_score': 0.0,
                'symbols': {}
            }
        
        try:
            # 按品种分组分析
            symbol_liquidity: Dict[str, Dict[str, Any]] = {}
            total_volume = 0
            
            for order in orders:
                symbol = order.get('symbol')
                volume = order.get('volume', 0)
                price = order.get('price', 0)
                
                if symbol not in symbol_liquidity:
                    symbol_liquidity[symbol] = {
                        'total_volume': 0,
                        'total_value': 0,
                        'order_count': 0
                    }
                
                symbol_liquidity[symbol]['total_volume'] += volume
                symbol_liquidity[symbol]['total_value'] += volume * price
                symbol_liquidity[symbol]['order_count'] += 1
                total_volume += volume
            
            # 计算流动性评分
            liquidity_scores = {}
            for symbol, data in symbol_liquidity.items():
                # 流动性评分 = f(交易量, 订单数)
                volume_score = min(data['total_volume'] / self.liquidity_threshold, 1.0)
                order_score = min(data['order_count'] / 10, 1.0)
                liquidity_scores[symbol] = (volume_score * 0.7 + order_score * 0.3)
            
            # 整体流动性
            overall_score = sum(liquidity_scores.values()) / len(liquidity_scores) if liquidity_scores else 0.0
            
            if overall_score > 0.7:
                overall_liquidity = 'high'
            elif overall_score > 0.4:
                overall_liquidity = 'medium'
            else:
                overall_liquidity = 'low'
            
            result = {
                'overall_liquidity': overall_liquidity,
                'liquidity_score': round(overall_score, 3),
                'total_volume': total_volume,
                'symbols': liquidity_scores,
                'analysis_time': datetime.now().isoformat()
            }
            
            logger.debug("Liquidity analysis: %s (score: %.3f)", 
                        overall_liquidity, overall_score)
            
            return result
            
        except Exception as e:
            logger.error("Liquidity analysis failed: %s", str(e))
            return {
                'overall_liquidity': 'error',
                'liquidity_score': 0.0,
                'error': str(e)
            }
    
    def _calculate_market_sentiment(self, trend_results: List[Dict]) -> str:
        """计算市场情绪"""
        if not trend_results:
            return 'neutral'
        
        bullish_count = sum(1 for r in trend_results if r.get('trend') == 'uptrend')
        bearish_count = sum(1 for r in trend_results if r.get('trend') == 'downtrend')
        
        total = len(trend_results)
        bullish_ratio = bullish_count / total if total > 0 else 0
        
        if bullish_ratio > 0.6:
            return 'bullish'
        elif bullish_ratio < 0.4:
            return 'bearish'
        else:
            return 'neutral'
    
    def _assess_market_risk_level(
        self, 
        trend_results: List[Dict],
        liquidity_analysis: Dict
    ) -> str:
        """评估市场风险等级"""
        # 基于趋势一致性和流动性评估风险
        if not trend_results:
            return 'medium'
        
        # 趋势一致性
        trends = [r.get('trend') for r in trend_results]
        trend_consistency = max(trends.count(t) for t in set(trends)) / len(trends)
        
        # 流动性
        liquidity_score = liquidity_analysis.get('liquidity_score', 0.5)
        
        # 综合风险评分
        risk_score = (1 - trend_consistency) * 0.6 + (1 - liquidity_score) * 0.4
        
        if risk_score > 0.7:
            return 'high'
        elif risk_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _generate_market_recommendations(self, trend_results: List[Dict]) -> List[str]:
        """生成市场建议"""
        recommendations = []
        
        for result in trend_results:
            symbol = result.get('symbol')
            trend = result.get('trend')
            confidence = result.get('confidence', 0)
            
            if confidence > 0.7:
                if trend == 'uptrend':
                    recommendations.append(f"考虑增加 {symbol} 多头仓位")
                elif trend == 'downtrend':
                    recommendations.append(f"考虑减少 {symbol} 仓位或做空")
            elif confidence < 0.3:
                recommendations.append(f"{symbol} 趋势不明确，建议观望")
        
        return recommendations

