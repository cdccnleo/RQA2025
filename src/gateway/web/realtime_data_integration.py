"""
实时数据集成服务
将 RealTimeDataStream 集成到交易信号服务
"""

import logging
import asyncio
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RealtimeDataConfig:
    """实时数据配置"""
    buffer_size: int = 1000
    cache_ttl: int = 60  # 缓存TTL（秒）
    enable_caching: bool = True
    max_history_bars: int = 100  # 最大历史K线数
    batch_size: int = 50  # 批量处理大小
    flush_interval: float = 0.1  # 刷新间隔（秒）
    enable_compression: bool = True  # 启用消息压缩
    compression_threshold: int = 1024  # 压缩阈值（字节）


class RealtimeDataIntegration:
    """
    实时数据集成服务
    
    职责：
    1. 订阅 RealTimeDataStream 的实时数据
    2. 维护实时数据缓存
    3. 将实时数据转换为 pandas DataFrame
    4. 触发实时信号生成
    """
    
    def __init__(self, config: Optional[RealtimeDataConfig] = None):
        """
        初始化实时数据集成服务
        
        Args:
            config: 实时数据配置
        """
        self.config = config or RealtimeDataConfig()
        self._data_stream = None
        self._engine = None
        self._is_running = False
        
        # 实时数据缓存
        # symbol -> List[MarketData]
        self._realtime_cache: Dict[str, List[dict]] = {}
        
        # 信号生成回调
        self._signal_callbacks: List[Callable] = []
        
        # 批量数据处理缓冲区
        self._batch_buffer: List[Dict] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
        
        # 性能统计
        self._performance_stats = {
            'total_messages': 0,
            'batched_messages': 0,
            'compression_savings': 0,
            'avg_latency': 0.0
        }
        
        logger.info("实时数据集成服务初始化完成")
    
    async def start(self):
        """启动实时数据集成服务"""
        if self._is_running:
            logger.warning("实时数据集成服务已在运行中")
            return
        
        try:
            # 获取实时策略引擎
            from src.strategy.realtime.real_time_processor import get_real_time_strategy_engine
            
            self._engine = get_real_time_strategy_engine()
            self._data_stream = self._engine.data_stream
            
            # 启动引擎
            await self._engine.start()
            
            self._is_running = True
            
            # 启动批量处理任务
            self._batch_task = asyncio.create_task(self._batch_processor_loop())
            
            logger.info("实时数据集成服务已启动（含批量处理优化）")
            
        except Exception as e:
            logger.error(f"启动实时数据集成服务失败: {e}")
            raise
    
    async def stop(self):
        """停止实时数据集成服务"""
        if not self._is_running:
            return
        
        try:
            # 停止批量处理任务
            if self._batch_task:
                self._batch_task.cancel()
                try:
                    await self._batch_task
                except asyncio.CancelledError:
                    pass
            
            # 刷新剩余数据
            await self._flush_batch_buffer()
            
            if self._engine:
                await self._engine.stop()
            
            self._is_running = False
            self._realtime_cache.clear()
            
            logger.info("实时数据集成服务已停止")
            
        except Exception as e:
            logger.error(f"停止实时数据集成服务失败: {e}")
    
    def subscribe_symbol(self, symbol: str):
        """
        订阅股票代码的实时数据
        
        Args:
            symbol: 股票代码
        """
        if not self._is_running:
            logger.warning("实时数据集成服务未启动，无法订阅")
            return
        
        try:
            # 初始化缓存
            if symbol not in self._realtime_cache:
                self._realtime_cache[symbol] = []
            
            logger.info(f"已订阅股票实时数据: {symbol}")
            
        except Exception as e:
            logger.error(f"订阅股票实时数据失败 {symbol}: {e}")
    
    def unsubscribe_symbol(self, symbol: str):
        """
        取消订阅股票代码的实时数据
        
        Args:
            symbol: 股票代码
        """
        if symbol in self._realtime_cache:
            del self._realtime_cache[symbol]
            logger.info(f"已取消订阅股票实时数据: {symbol}")
    
    async def ingest_market_data(self, symbol: str, data: dict):
        """
        摄入市场数据（优化版，支持批量处理）
        
        Args:
            symbol: 股票代码
            data: 市场数据字典
        """
        if not self._is_running:
            return
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 添加到批量缓冲区
            async with self._batch_lock:
                self._batch_buffer.append({
                    'symbol': symbol,
                    'data': data,
                    'timestamp': datetime.now(),
                    'ingest_time': start_time
                })
                
                # 如果缓冲区达到批量大小，立即刷新
                if len(self._batch_buffer) >= self.config.batch_size:
                    await self._flush_batch_buffer()
            
            # 更新性能统计
            self._performance_stats['total_messages'] += 1
            
        except Exception as e:
            logger.error(f"摄入市场数据失败 {symbol}: {e}")
    
    async def _batch_processor_loop(self):
        """批量处理循环"""
        logger.info("启动批量数据处理循环")
        
        while self._is_running:
            try:
                await asyncio.sleep(self.config.flush_interval)
                
                async with self._batch_lock:
                    if self._batch_buffer:
                        await self._flush_batch_buffer()
                        
            except asyncio.CancelledError:
                logger.info("批量处理循环已取消")
                break
            except Exception as e:
                logger.error(f"批量处理循环错误: {e}")
    
    async def _flush_batch_buffer(self):
        """刷新批量缓冲区"""
        if not self._batch_buffer:
            return
        
        try:
            # 复制缓冲区数据
            batch_data = self._batch_buffer.copy()
            self._batch_buffer.clear()
            
            # 批量处理数据
            await self._process_batch_data(batch_data)
            
            # 更新统计
            self._performance_stats['batched_messages'] += len(batch_data)
            
            logger.debug(f"批量处理 {len(batch_data)} 条数据")
            
        except Exception as e:
            logger.error(f"刷新批量缓冲区失败: {e}")
    
    async def _process_batch_data(self, batch_data: List[Dict]):
        """处理批量数据"""
        try:
            # 按股票代码分组
            symbol_groups = {}
            for item in batch_data:
                symbol = item['symbol']
                if symbol not in symbol_groups:
                    symbol_groups[symbol] = []
                symbol_groups[symbol].append(item)
            
            # 批量更新缓存
            for symbol, items in symbol_groups.items():
                if symbol not in self._realtime_cache:
                    self._realtime_cache[symbol] = []
                
                # 批量添加到缓存
                for item in items:
                    self._realtime_cache[symbol].append({
                        'timestamp': item['timestamp'],
                        **item['data']
                    })
                
                # 限制缓存大小
                if len(self._realtime_cache[symbol]) > self.config.max_history_bars:
                    self._realtime_cache[symbol] = self._realtime_cache[symbol][-self.config.max_history_bars:]
            
            # 批量触发信号生成（只触发最后一个数据）
            for symbol, items in symbol_groups.items():
                if items:
                    await self._trigger_signal_generation(symbol, items[-1]['data'])
                    
        except Exception as e:
            logger.error(f"处理批量数据失败: {e}")
    
    def get_realtime_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        获取实时数据作为 pandas DataFrame
        
        Args:
            symbol: 股票代码
            
        Returns:
            pandas DataFrame or None
        """
        if symbol not in self._realtime_cache:
            return None
        
        try:
            data = self._realtime_cache[symbol]
            if not data:
                return None
            
            df = pd.DataFrame(data)
            
            # 设置时间索引
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # 确保必要的列存在
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            return df[required_columns].astype(float)
            
        except Exception as e:
            logger.error(f"获取实时DataFrame失败 {symbol}: {e}")
            return None
    
    def get_combined_dataframe(
        self,
        symbol: str,
        historical_df: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """
        获取组合数据（历史数据 + 实时数据）
        
        Args:
            symbol: 股票代码
            historical_df: 历史数据DataFrame
            
        Returns:
            组合后的 pandas DataFrame
        """
        realtime_df = self.get_realtime_dataframe(symbol)
        
        if realtime_df is None:
            return historical_df
        
        if historical_df is None:
            return realtime_df
        
        try:
            # 合并数据
            combined = pd.concat([historical_df, realtime_df])
            
            # 去重（基于索引）
            combined = combined[~combined.index.duplicated(keep='last')]
            
            # 排序
            combined.sort_index(inplace=True)
            
            return combined
            
        except Exception as e:
            logger.error(f"合并数据失败 {symbol}: {e}")
            return historical_df
    
    def register_signal_callback(self, callback: Callable):
        """
        注册信号生成回调函数
        
        Args:
            callback: 回调函数，接收 (symbol, data) 参数
        """
        self._signal_callbacks.append(callback)
        logger.info(f"注册信号生成回调，当前回调数: {len(self._signal_callbacks)}")
    
    def unregister_signal_callback(self, callback: Callable):
        """
        注销信号生成回调函数
        
        Args:
            callback: 回调函数
        """
        if callback in self._signal_callbacks:
            self._signal_callbacks.remove(callback)
            logger.info(f"注销信号生成回调，当前回调数: {len(self._signal_callbacks)}")
    
    async def _trigger_signal_generation(self, symbol: str, data: dict):
        """
        触发信号生成
        
        Args:
            symbol: 股票代码
            data: 市场数据
        """
        for callback in self._signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, data)
                else:
                    callback(symbol, data)
            except Exception as e:
                logger.error(f"信号生成回调执行失败: {e}")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'is_running': self._is_running,
            'cached_symbols': list(self._realtime_cache.keys()),
            'cache_sizes': {
                symbol: len(data) 
                for symbol, data in self._realtime_cache.items()
            },
            'total_cached_bars': sum(
                len(data) for data in self._realtime_cache.values()
            )
        }
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        total = self._performance_stats['total_messages']
        batched = self._performance_stats['batched_messages']
        
        return {
            'total_messages': total,
            'batched_messages': batched,
            'batch_ratio': batched / total if total > 0 else 0,
            'compression_savings': self._performance_stats['compression_savings'],
            'avg_latency': self._performance_stats['avg_latency'],
            'buffer_size': len(self._batch_buffer),
            'config': {
                'batch_size': self.config.batch_size,
                'flush_interval': self.config.flush_interval,
                'enable_compression': self.config.enable_compression
            }
        }
    
    def clear_cache(self):
        """清除缓存"""
        self._realtime_cache.clear()
        logger.info("实时数据缓存已清除")
    
    def compress_data(self, data: Dict) -> bytes:
        """
        压缩数据
        
        Args:
            data: 原始数据
            
        Returns:
            压缩后的字节数据
        """
        if not self.config.enable_compression:
            import json
            return json.dumps(data).encode('utf-8')
        
        try:
            import json
            import zlib
            
            # 序列化数据
            json_data = json.dumps(data).encode('utf-8')
            
            # 如果数据小于阈值，不压缩
            if len(json_data) < self.config.compression_threshold:
                return json_data
            
            # 压缩数据
            compressed = zlib.compress(json_data, level=6)
            
            # 更新统计
            savings = len(json_data) - len(compressed)
            self._performance_stats['compression_savings'] += savings
            
            logger.debug(f"数据压缩: {len(json_data)} -> {len(compressed)} 字节，节省 {savings} 字节")
            
            return compressed
            
        except Exception as e:
            logger.error(f"数据压缩失败: {e}")
            import json
            return json.dumps(data).encode('utf-8')
    
    def decompress_data(self, compressed_data: bytes) -> Dict:
        """
        解压缩数据
        
        Args:
            compressed_data: 压缩后的字节数据
            
        Returns:
            原始数据
        """
        if not self.config.enable_compression:
            import json
            return json.loads(compressed_data.decode('utf-8'))
        
        try:
            import json
            import zlib
            
            # 尝试解压缩
            try:
                decompressed = zlib.decompress(compressed_data)
                return json.loads(decompressed.decode('utf-8'))
            except:
                # 如果解压缩失败，可能是未压缩的数据
                return json.loads(compressed_data.decode('utf-8'))
                
        except Exception as e:
            logger.error(f"数据解压缩失败: {e}")
            import json
            return json.loads(compressed_data.decode('utf-8'))


# 单例实例
_realtime_data_integration: Optional[RealtimeDataIntegration] = None


def get_realtime_data_integration() -> RealtimeDataIntegration:
    """获取实时数据集成服务实例"""
    global _realtime_data_integration
    if _realtime_data_integration is None:
        _realtime_data_integration = RealtimeDataIntegration()
    return _realtime_data_integration


async def start_realtime_data_integration():
    """启动实时数据集成服务"""
    integration = get_realtime_data_integration()
    await integration.start()
    return integration


async def stop_realtime_data_integration():
    """停止实时数据集成服务"""
    global _realtime_data_integration
    if _realtime_data_integration:
        await _realtime_data_integration.stop()
        _realtime_data_integration = None
