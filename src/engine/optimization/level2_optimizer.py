"""Level2数据优化器模块"""

from typing import Dict, List, Any, Optional
from src.infrastructure.monitoring import MetricsCollector
from src.data.market_data import MarketData
import pandas as pd
import numpy as np
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

class Level2Optimizer:
    """Level2数据优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.market_data = MarketData()
        
        # Level2配置
        self.max_depth = config.get('max_depth', 10)
        self.update_interval = config.get('update_interval', 0.1)  # 100ms
        self.max_symbols = config.get('max_symbols', 100)
        
        # 性能监控
        self.performance_metrics = {
            'updates_processed': 0,
            'symbols_tracked': 0,
            'avg_processing_time_ms': 0,
            'memory_usage_mb': 0
        }
        
        # 初始化数据结构
        self.order_books = {}
        self.last_update_times = {}
        self.processing_times = []
        
    def update_order_book(self, symbol: str, order_book_data: Dict[str, Any]):
        """更新订单簿数据"""
        start_time = time.time()
        
        try:
            # 验证数据
            if not self._validate_order_book_data(order_book_data):
                logger.warning(f"Invalid order book data for {symbol}")
                return
            
            # 更新订单簿
            self.order_books[symbol] = self._process_order_book(order_book_data)
            self.last_update_times[symbol] = time.time()
            
            # 更新性能指标
            processing_time = (time.time() - start_time) * 1000  # 转换为毫秒
            self.processing_times.append(processing_time)
            
            # 保持最近100次的处理时间
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            self.performance_metrics['updates_processed'] += 1
            self.performance_metrics['avg_processing_time_ms'] = np.mean(self.processing_times)
            
            # 记录指标
            self.metrics_collector.record_level2_update(
                symbol=symbol,
                processing_time_ms=processing_time,
                order_book_size=len(self.order_books[symbol])
            )
            
            logger.debug(f"Updated order book for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating order book for {symbol}: {e}")
    
    def _validate_order_book_data(self, data: Dict[str, Any]) -> bool:
        """验证订单簿数据"""
        required_fields = ['bids', 'asks', 'timestamp']
        
        for field in required_fields:
            if field not in data:
                return False
        
        # 验证买卖盘数据
        if not isinstance(data['bids'], list) or not isinstance(data['asks'], list):
            return False
        
        # 验证价格和数量
        for bid in data['bids']:
            if len(bid) != 2 or not isinstance(bid[0], (int, float)) or not isinstance(bid[1], (int, float)):
                return False
        
        for ask in data['asks']:
            if len(ask) != 2 or not isinstance(ask[0], (int, float)) or not isinstance(ask[1], (int, float)):
                return False
        
        return True
    
    def _process_order_book(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理订单簿数据"""
        processed = {
            'bids': data['bids'][:self.max_depth],
            'asks': data['asks'][:self.max_depth],
            'timestamp': data['timestamp'],
            'spread': 0.0,
            'mid_price': 0.0,
            'bid_depth': 0,
            'ask_depth': 0
        }
        
        # 计算买卖价差
        if processed['bids'] and processed['asks']:
            best_bid = processed['bids'][0][0]
            best_ask = processed['asks'][0][0]
            processed['spread'] = best_ask - best_bid
            processed['mid_price'] = (best_bid + best_ask) / 2
        
        # 计算深度
        processed['bid_depth'] = sum(bid[1] for bid in processed['bids'])
        processed['ask_depth'] = sum(ask[1] for ask in processed['asks'])
        
        return processed
    
    def get_order_book(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取订单簿数据"""
        if symbol not in self.order_books:
            return None
        
        return self.order_books[symbol]
    
    def get_best_quotes(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最优报价"""
        order_book = self.get_order_book(symbol)
        if not order_book:
            return None
        
        return {
            'best_bid': order_book['bids'][0] if order_book['bids'] else None,
            'best_ask': order_book['asks'][0] if order_book['asks'] else None,
            'spread': order_book['spread'],
            'mid_price': order_book['mid_price'],
            'timestamp': order_book['timestamp']
        }
    
    def calculate_market_impact(self, symbol: str, order_size: float, side: str) -> float:
        """计算市场冲击成本"""
        order_book = self.get_order_book(symbol)
        if not order_book:
            return 0.0
        
        if side == 'buy':
            orders = order_book['asks']
        else:
            orders = order_book['bids']
        
        remaining_size = order_size
        total_cost = 0.0
        
        for price, quantity in orders:
            if remaining_size <= 0:
                break
            
            executed_size = min(remaining_size, quantity)
            total_cost += executed_size * price
            remaining_size -= executed_size
        
        if order_size > 0:
            return total_cost / order_size
        else:
            return 0.0
    
    def get_market_depth(self, symbol: str, levels: int = 5) -> Dict[str, Any]:
        """获取市场深度"""
        order_book = self.get_order_book(symbol)
        if not order_book:
            return {'bids': [], 'asks': []}
        
        return {
            'bids': order_book['bids'][:levels],
            'asks': order_book['asks'][:levels],
            'timestamp': order_book['timestamp']
        }
    
    def calculate_vwap(self, symbol: str, side: str, size: float) -> float:
        """计算成交量加权平均价格"""
        order_book = self.get_order_book(symbol)
        if not order_book:
            return 0.0
        
        if side == 'buy':
            orders = order_book['asks']
        else:
            orders = order_book['bids']
        
        remaining_size = size
        total_value = 0.0
        
        for price, quantity in orders:
            if remaining_size <= 0:
                break
            
            executed_size = min(remaining_size, quantity)
            total_value += executed_size * price
            remaining_size -= executed_size
        
        if size > 0:
            return total_value / size
        else:
            return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics = self.performance_metrics.copy()
        
        # 计算内存使用
        total_memory = 0
        for symbol, order_book in self.order_books.items():
            # 估算订单簿内存使用
            total_memory += len(str(order_book)) * 8  # 粗略估算
        
        metrics['memory_usage_mb'] = total_memory / (1024 * 1024)
        metrics['symbols_tracked'] = len(self.order_books)
        
        return metrics
    
    def cleanup_old_data(self, max_age_seconds: int = 300):
        """清理旧数据"""
        current_time = time.time()
        symbols_to_remove = []
        
        for symbol, last_update in self.last_update_times.items():
            if current_time - last_update > max_age_seconds:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            if symbol in self.order_books:
                del self.order_books[symbol]
            if symbol in self.last_update_times:
                del self.last_update_times[symbol]
        
        logger.info(f"Cleaned up {len(symbols_to_remove)} old symbols")
    
    def get_active_symbols(self) -> List[str]:
        """获取活跃的股票代码"""
        return list(self.order_books.keys())
    
    def get_order_book_stats(self) -> Dict[str, Any]:
        """获取订单簿统计信息"""
        if not self.order_books:
            return {
                'total_symbols': 0,
                'avg_spread': 0.0,
                'avg_depth': 0.0,
                'most_active': []
            }
        
        spreads = []
        depths = []
        
        for symbol, order_book in self.order_books.items():
            spreads.append(order_book['spread'])
            depths.append(order_book['bid_depth'] + order_book['ask_depth'])
        
        # 按更新频率排序
        most_active = sorted(
            self.last_update_times.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_symbols': len(self.order_books),
            'avg_spread': np.mean(spreads) if spreads else 0.0,
            'avg_depth': np.mean(depths) if depths else 0.0,
            'most_active': [symbol for symbol, _ in most_active]
        }
