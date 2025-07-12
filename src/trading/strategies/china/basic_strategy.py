"""A股基础交易策略实现"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from src.infrastructure.monitoring import MetricsCollector
from src.trading.strategies.base_strategy import BaseStrategy
from src.data.market_data import MarketData
from src.features.feature_engineer import FeatureEngineer
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ChinaMarketStrategy(BaseStrategy):
    """中国股市策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.metrics_collector = MetricsCollector()
        self.feature_engineer = FeatureEngineer()
        self.market_data = MarketData()
        
        # 中国股市特有配置
        self.trading_hours = config.get('trading_hours', {
            'morning': ('09:30', '11:30'),
            'afternoon': ('13:00', '15:00')
        })
        
        self.price_limit = config.get('price_limit', {
            'up_limit': 0.10,  # 涨停板
            'down_limit': -0.10  # 跌停板
        })
        
        self.min_tick_size = config.get('min_tick_size', 0.01)
        
    def is_trading_time(self, current_time: pd.Timestamp) -> bool:
        """检查是否为交易时间"""
        time_str = current_time.strftime('%H:%M')
        
        morning_start, morning_end = self.trading_hours['morning']
        afternoon_start, afternoon_end = self.trading_hours['afternoon']
        
        return (morning_start <= time_str <= morning_end or 
                afternoon_start <= time_str <= afternoon_end)
    
    def check_price_limit(self, current_price: float, reference_price: float) -> Dict[str, bool]:
        """检查价格限制"""
        price_change = (current_price - reference_price) / reference_price
        
        return {
            'up_limit_hit': price_change >= self.price_limit['up_limit'],
            'down_limit_hit': price_change <= self.price_limit['down_limit']
        }
    
    def calculate_position_size(self, available_capital: float, 
                              risk_per_trade: float, 
                              stop_loss_pct: float) -> float:
        """计算仓位大小"""
        if stop_loss_pct <= 0:
            return 0
        
        position_size = (available_capital * risk_per_trade) / stop_loss_pct
        return min(position_size, available_capital)
    
    def validate_order(self, order: Dict[str, Any]) -> bool:
        """验证订单"""
        required_fields = ['symbol', 'side', 'quantity', 'price']
        
        for field in required_fields:
            if field not in order:
                logger.error(f"Missing required field: {field}")
                return False
        
        # 检查价格精度
        if not self._is_valid_price(order['price']):
            logger.error(f"Invalid price precision: {order['price']}")
            return False
        
        # 检查数量
        if order['quantity'] <= 0:
            logger.error(f"Invalid quantity: {order['quantity']}")
            return False
        
        return True
    
    def _is_valid_price(self, price: float) -> bool:
        """检查价格是否符合最小变动单位"""
        return abs(price - round(price / self.min_tick_size) * self.min_tick_size) < 1e-6
    
    def calculate_commission(self, order_value: float, commission_rate: float = 0.0003) -> float:
        """计算手续费"""
        return order_value * commission_rate
    
    def calculate_slippage(self, order_quantity: int, market_impact: float = 0.0001) -> float:
        """计算滑点"""
        return order_quantity * market_impact
    
    def get_market_status(self, symbol: str) -> Dict[str, Any]:
        """获取市场状态"""
        try:
            market_data = self.market_data.get_latest_data(symbol)
            
            if market_data is None:
                return {'status': 'no_data'}
            
            return {
                'status': 'trading',
                'last_price': market_data.get('close', 0),
                'volume': market_data.get('volume', 0),
                'timestamp': market_data.get('timestamp', pd.Timestamp.now())
            }
            
        except Exception as e:
            logger.error(f"Error getting market status for {symbol}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        if data.empty:
            return data
        
        # 处理缺失值
        data = data.fillna(method='ffill')
        
        # 添加技术指标
        data = self.feature_engineer.add_technical_indicators(data)
        
        # 添加基本面指标
        data = self.feature_engineer.add_fundamental_indicators(data)
        
        return data
    
    def calculate_risk_metrics(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算风险指标"""
        if not positions:
            return {
                'total_exposure': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'sharpe_ratio': 0.0
            }
        
        total_value = sum(pos.get('market_value', 0) for pos in positions)
        total_cost = sum(pos.get('cost', 0) for pos in positions)
        
        # 计算总敞口
        total_exposure = total_value / total_cost if total_cost > 0 else 0
        
        # 计算最大回撤（简化版本）
        max_drawdown = 0.0
        if total_cost > 0:
            drawdown = (total_value - total_cost) / total_cost
            max_drawdown = min(drawdown, 0)
        
        return {
            'total_exposure': total_exposure,
            'max_drawdown': max_drawdown,
            'var_95': 0.0,  # 需要历史数据计算
            'sharpe_ratio': 0.0  # 需要历史数据计算
        }
    
    def should_rebalance(self, current_positions: List[Dict[str, Any]], 
                        target_positions: List[Dict[str, Any]], 
                        threshold: float = 0.05) -> bool:
        """判断是否需要再平衡"""
        if not current_positions or not target_positions:
            return False
        
        current_total = sum(pos.get('market_value', 0) for pos in current_positions)
        target_total = sum(pos.get('target_value', 0) for pos in target_positions)
        
        if current_total == 0 or target_total == 0:
            return False
        
        # 计算偏离度
        deviation = abs(current_total - target_total) / target_total
        return deviation > threshold
    
    def log_trade(self, trade: Dict[str, Any]):
        """记录交易"""
        self.metrics_collector.record_trade(
            symbol=trade.get('symbol'),
            side=trade.get('side'),
            quantity=trade.get('quantity'),
            price=trade.get('price'),
            timestamp=trade.get('timestamp', pd.Timestamp.now())
        )
        
        logger.info(f"Trade executed: {trade}")
    
    def log_strategy_metrics(self, metrics: Dict[str, Any]):
        """记录策略指标"""
        self.metrics_collector.record_strategy_metrics(metrics)
        logger.info(f"Strategy metrics: {metrics}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        pass
    
    @abstractmethod
    def execute_strategy(self, signals: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行策略"""
        pass
