#!/usr/bin/env python3
"""
实时数据处理器
支持实时数据流处理、事件驱动架构和流式计算
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Set
import json
import websockets
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

from src.core.cache.redis_cache import RedisCache
from src.core.monitoring.data_collection_monitor import DataCollectionMonitor


@dataclass
class StreamConfig:
    """流配置"""
    stream_name: str
    data_source: str
    update_interval_seconds: float = 1.0  # 更新间隔
    batch_size: int = 100  # 批处理大小
    max_queue_size: int = 10000  # 最大队列大小
    enable_buffering: bool = True  # 启用缓冲
    buffer_timeout_seconds: float = 5.0  # 缓冲超时
    retry_attempts: int = 3  # 重试次数
    timeout_seconds: int = 30  # 超时时间


@dataclass
class StreamEvent:
    """流事件"""
    event_id: str
    stream_name: str
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    sequence_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """处理结果"""
    event: StreamEvent
    processed_data: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    transformed_records: int = 0


class StreamProcessor:
    """流处理器"""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.stream_name}")

        # 队列管理
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.processing_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)

        # 缓冲区
        self.buffer: List[StreamEvent] = []
        self.buffer_lock = asyncio.Lock()
        self.last_flush_time = datetime.now()

        # 处理统计
        self.stats = {
            'events_received': 0,
            'events_processed': 0,
            'events_failed': 0,
            'processing_time_total': 0.0,
            'batches_processed': 0,
            'buffer_flushes': 0
        }

        # 控制标志
        self.running = False
        self.processing_task: Optional[asyncio.Task] = None
        self.buffer_task: Optional[asyncio.Task] = None

    async def start(self):
        """启动流处理器"""
        self.running = True
        self.logger.info(f"启动流处理器: {self.config.stream_name}")

        # 启动处理任务
        self.processing_task = asyncio.create_task(self._processing_loop())
        self.buffer_task = asyncio.create_task(self._buffer_management_loop())

    async def stop(self):
        """停止流处理器"""
        self.running = False
        self.logger.info(f"正在停止流处理器: {self.config.stream_name}")

        # 取消任务
        tasks_to_cancel = [self.processing_task, self.buffer_task]
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()

        # 等待任务完成
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # 清空缓冲区
        await self._flush_buffer()

        self.logger.info(f"流处理器已停止: {self.config.stream_name}")

    async def submit_event(self, event: StreamEvent):
        """提交事件到流"""
        try:
            await self.event_queue.put(event)
            self.stats['events_received'] += 1

            # 检查队列大小
            if self.event_queue.qsize() > self.config.max_queue_size * 0.8:
                self.logger.warning(f"事件队列接近满载: {self.event_queue.qsize()}/{self.config.max_queue_size}")

        except asyncio.QueueFull:
            self.logger.error(f"事件队列已满，丢弃事件: {event.event_id}")
            self.stats['events_failed'] += 1

    async def _processing_loop(self):
        """处理循环"""
        while self.running:
            try:
                # 获取批次事件
                batch = await self._get_batch_events()

                if batch:
                    # 处理批次
                    await self._process_batch(batch)

                await asyncio.sleep(self.config.update_interval_seconds)

            except Exception as e:
                self.logger.error(f"处理循环异常: {e}")
                await asyncio.sleep(1)

    async def _get_batch_events(self) -> List[StreamEvent]:
        """获取批次事件"""
        batch = []
        batch_size = self.config.batch_size

        try:
            # 非阻塞获取事件
            for _ in range(batch_size):
                try:
                    event = self.event_queue.get_nowait()
                    batch.append(event)
                    self.event_queue.task_done()
                except asyncio.QueueEmpty:
                    break

        except Exception as e:
            self.logger.error(f"获取批次事件失败: {e}")

        return batch

    async def _process_batch(self, batch: List[StreamEvent]):
        """处理批次事件"""
        if not batch:
            return

        start_time = datetime.now()

        try:
            self.logger.debug(f"开始处理批次: {len(batch)} 个事件")

            # 并行处理事件
            tasks = [self._process_single_event(event) for event in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 统计结果
            successful = 0
            failed = 0

            for result in results:
                if isinstance(result, ProcessingResult) and result.success:
                    successful += 1
                    self.stats['events_processed'] += 1
                    self.stats['processing_time_total'] += result.processing_time
                else:
                    failed += 1
                    self.stats['events_failed'] += 1

            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['batches_processed'] += 1

            self.logger.info(f"批次处理完成: {successful}/{len(batch)} 成功, "
                           f"耗时: {processing_time:.2f}s")

        except Exception as e:
            self.logger.error(f"批次处理异常: {e}")
            self.stats['events_failed'] += len(batch)

    async def _process_single_event(self, event: StreamEvent) -> ProcessingResult:
        """处理单个事件"""
        start_time = datetime.now()

        try:
            # 根据事件类型进行处理
            if event.event_type == "market_data":
                result = await self._process_market_data_event(event)
            elif event.event_type == "trade_data":
                result = await self._process_trade_data_event(event)
            elif event.event_type == "news_data":
                result = await self._process_news_data_event(event)
            else:
                # 默认处理
                result = await self._process_generic_event(event)

            processing_time = (datetime.now() - start_time).total_seconds()

            return ProcessingResult(
                event=event,
                processed_data=result,
                processing_time=processing_time,
                success=True,
                transformed_records=len(result) if isinstance(result, list) else 1
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"事件处理失败 {event.event_id}: {e}")

            return ProcessingResult(
                event=event,
                processed_data={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    async def _process_market_data_event(self, event: StreamEvent) -> Dict[str, Any]:
        """处理市场数据事件"""
        # 实时计算技术指标
        data = event.data

        # 计算简单移动平均
        if 'price_history' in data:
            prices = data['price_history']
            if len(prices) >= 5:
                data['sma_5'] = sum(prices[-5:]) / 5
            if len(prices) >= 10:
                data['sma_10'] = sum(prices[-10:]) / 10

        # 计算价格变动率
        if len(prices) >= 2:
            data['price_change_pct'] = (prices[-1] - prices[-2]) / prices[-2] * 100

        # 实时风险指标
        if 'volume_history' in data and len(data['volume_history']) >= 5:
            volumes = data['volume_history']
            avg_volume = sum(volumes[-5:]) / 5
            data['volume_sma_5'] = avg_volume

        return data

    async def _process_trade_data_event(self, event: StreamEvent) -> Dict[str, Any]:
        """处理交易数据事件"""
        data = event.data

        # 实时交易统计
        trades = data.get('trades', [])

        if trades:
            # 计算成交量统计
            volumes = [t.get('volume', 0) for t in trades]
            prices = [t.get('price', 0) for t in trades if t.get('price', 0) > 0]

            data['total_volume'] = sum(volumes)
            data['trade_count'] = len(trades)

            if prices:
                data['avg_price'] = sum(prices) / len(prices)
                data['price_volatility'] = self._calculate_volatility(prices)

            # 大单检测
            large_trades = [t for t in trades if t.get('volume', 0) > 10000]  # 假设1万股为大单
            data['large_trade_count'] = len(large_trades)

        return data

    async def _process_news_data_event(self, event: StreamEvent) -> Dict[str, Any]:
        """处理新闻数据事件"""
        data = event.data

        # 实时情感分析（简化实现）
        title = data.get('title', '')
        content = data.get('content', '')

        # 简单关键词分析
        positive_keywords = ['上涨', '增长', '利好', '突破', '创新']
        negative_keywords = ['下跌', '亏损', '利空', '风险', '危机']

        positive_score = sum(1 for word in positive_keywords if word in title + content)
        negative_score = sum(1 for word in negative_keywords if word in title + content)

        data['sentiment_score'] = positive_score - negative_score
        data['sentiment_label'] = 'positive' if positive_score > negative_score else 'negative' if negative_score > positive_score else 'neutral'

        # 影响范围分析
        mentioned_symbols = self._extract_mentioned_symbols(content)
        data['mentioned_symbols'] = mentioned_symbols
        data['impact_scope'] = len(mentioned_symbols)

        return data

    async def _process_generic_event(self, event: StreamEvent) -> Dict[str, Any]:
        """处理通用事件"""
        # 数据清理和标准化
        data = event.data

        # 标准化时间戳
        if 'timestamp' in data:
            if isinstance(data['timestamp'], str):
                try:
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                except ValueError:
                    data['timestamp'] = datetime.now()

        # 标准化数值字段
        numeric_fields = ['price', 'volume', 'amount', 'change_pct']
        for field in numeric_fields:
            if field in data and data[field] is not None:
                try:
                    data[field] = float(data[field])
                except (ValueError, TypeError):
                    data[field] = None

        # 添加处理元数据
        data['_processed_at'] = datetime.now()
        data['_event_id'] = event.event_id

        return data

    def _calculate_volatility(self, prices: List[float]) -> float:
        """计算波动率"""
        if len(prices) < 2:
            return 0.0

        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)

        if not returns:
            return 0.0

        # 计算标准差作为波动率度量
        try:
            import statistics
            return statistics.stdev(returns)
        except:
            return abs(max(returns) - min(returns)) if returns else 0.0

    def _extract_mentioned_symbols(self, content: str) -> List[str]:
        """提取提到的股票代码（简化实现）"""
        # 这里应该使用更复杂的NLP方法
        # 暂时使用简单的模式匹配
        import re

        # 匹配6位数字的股票代码
        symbols = re.findall(r'\b\d{6}\b', content)

        # 去重并过滤明显不是股票代码的
        valid_symbols = []
        seen = set()

        for symbol in symbols:
            if symbol not in seen and len(symbol) == 6:
                # 简单的有效性检查（可以根据实际需求扩展）
                if symbol.startswith(('0', '3', '6')):  # A股代码特征
                    valid_symbols.append(symbol)
                    seen.add(symbol)

        return valid_symbols

    async def _buffer_management_loop(self):
        """缓冲区管理循环"""
        while self.running:
            try:
                await asyncio.sleep(self.config.buffer_timeout_seconds)

                # 检查是否需要刷新缓冲区
                if self.config.enable_buffering:
                    await self._check_buffer_flush()

            except Exception as e:
                self.logger.error(f"缓冲区管理异常: {e}")

    async def _check_buffer_flush(self):
        """检查缓冲区刷新"""
        async with self.buffer_lock:
            now = datetime.now()

            # 条件1: 缓冲区大小达到阈值
            size_threshold = len(self.buffer) >= self.config.batch_size

            # 条件2: 时间间隔达到阈值
            time_threshold = (now - self.last_flush_time).total_seconds() >= self.config.buffer_timeout_seconds

            if size_threshold or time_threshold:
                if self.buffer:
                    await self._flush_buffer()
                    self.stats['buffer_flushes'] += 1

    async def _flush_buffer(self):
        """刷新缓冲区"""
        if not self.buffer:
            return

        try:
            self.logger.debug(f"刷新缓冲区: {len(self.buffer)} 个事件")

            # 将缓冲区事件提交到处理队列
            for event in self.buffer:
                try:
                    await self.processing_queue.put(event)
                except asyncio.QueueFull:
                    self.logger.warning("处理队列已满，丢弃缓冲事件")

            self.buffer.clear()
            self.last_flush_time = datetime.now()

        except Exception as e:
            self.logger.error(f"缓冲区刷新失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()

        # 计算实时指标
        total_events = stats['events_received']
        if total_events > 0:
            stats['success_rate'] = stats['events_processed'] / total_events
            stats['failure_rate'] = stats['events_failed'] / total_events

        if stats['events_processed'] > 0:
            stats['avg_processing_time'] = stats['processing_time_total'] / stats['events_processed']

        stats['queue_size'] = self.event_queue.qsize()
        stats['buffer_size'] = len(self.buffer)
        stats['is_running'] = self.running

        return stats


class DataStreamConnector:
    """数据流连接器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 连接管理
        self.connections: Dict[str, Any] = {}
        self.stream_processors: Dict[str, StreamProcessor] = {}

        # 会话管理
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect_stream(self, stream_config: StreamConfig) -> AsyncGenerator[StreamEvent, None]:
        """
        连接数据流

        Args:
            stream_config: 流配置

        Yields:
            流事件
        """
        stream_name = stream_config.stream_name

        try:
            self.logger.info(f"连接数据流: {stream_name}")

            # 创建流处理器
            processor = StreamProcessor(stream_config)
            self.stream_processors[stream_name] = processor
            await processor.start()

            # 根据数据源类型连接流
            if stream_config.data_source.startswith('ws://') or stream_config.data_source.startswith('wss://'):
                # WebSocket连接
                async for event in self._connect_websocket_stream(stream_config):
                    yield event

            elif stream_config.data_source.startswith('http://') or stream_config.data_source.startswith('https://'):
                # HTTP流连接
                async for event in self._connect_http_stream(stream_config):
                    yield event

            else:
                # 文件或本地流
                async for event in self._connect_file_stream(stream_config):
                    yield event

        except Exception as e:
            self.logger.error(f"数据流连接失败 {stream_name}: {e}")
            raise
        finally:
            # 清理资源
            if stream_name in self.stream_processors:
                await self.stream_processors[stream_name].stop()
                del self.stream_processors[stream_name]

    async def _connect_websocket_stream(self, config: StreamConfig) -> AsyncGenerator[StreamEvent, None]:
        """连接WebSocket流"""
        try:
            async with websockets.connect(config.data_source) as websocket:
                self.logger.info(f"WebSocket连接成功: {config.stream_name}")

                sequence_number = 0

                while True:
                    try:
                        # 接收消息
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=config.timeout_seconds
                        )

                        # 解析消息
                        data = json.loads(message)

                        # 创建事件
                        event = StreamEvent(
                            event_id=f"{config.stream_name}_{sequence_number}",
                            stream_name=config.stream_name,
                            event_type=data.get('type', 'unknown'),
                            data=data.get('data', {}),
                            source=config.data_source,
                            sequence_number=sequence_number,
                            metadata=data.get('metadata', {})
                        )

                        sequence_number += 1
                        yield event

                    except asyncio.TimeoutError:
                        self.logger.warning(f"WebSocket接收超时: {config.stream_name}")
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.warning(f"WebSocket连接断开: {config.stream_name}")
                        break

        except Exception as e:
            self.logger.error(f"WebSocket流连接异常: {e}")
            raise

    async def _connect_http_stream(self, config: StreamConfig) -> AsyncGenerator[StreamEvent, None]:
        """连接HTTP流"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        sequence_number = 0

        try:
            while True:
                try:
                    async with self.session.get(config.data_source) as response:
                        if response.status == 200:
                            # 读取流数据
                            async for line in response.content:
                                line = line.decode('utf-8').strip()
                                if line:
                                    try:
                                        data = json.loads(line)

                                        event = StreamEvent(
                                            event_id=f"{config.stream_name}_{sequence_number}",
                                            stream_name=config.stream_name,
                                            event_type=data.get('type', 'http_data'),
                                            data=data,
                                            source=config.data_source,
                                            sequence_number=sequence_number
                                        )

                                        sequence_number += 1
                                        yield event

                                    except json.JSONDecodeError:
                                        continue

                        else:
                            self.logger.warning(f"HTTP流请求失败: {response.status}")
                            await asyncio.sleep(5)  # 等待重试

                except Exception as e:
                    self.logger.error(f"HTTP流读取异常: {e}")
                    await asyncio.sleep(5)

                await asyncio.sleep(config.update_interval_seconds)

        except Exception as e:
            self.logger.error(f"HTTP流连接异常: {e}")
            raise

    async def _connect_file_stream(self, config: StreamConfig) -> AsyncGenerator[StreamEvent, None]:
        """连接文件流（用于测试）"""
        import time

        sequence_number = 0

        try:
            while True:
                # 生成模拟数据
                mock_data = {
                    "symbol": "000001.SZ",
                    "price": 100.0 + (sequence_number % 10),
                    "volume": 10000 + (sequence_number % 5000),
                    "timestamp": datetime.now().isoformat()
                }

                event = StreamEvent(
                    event_id=f"{config.stream_name}_{sequence_number}",
                    stream_name=config.stream_name,
                    event_type="mock_data",
                    data=mock_data,
                    source="file_stream",
                    sequence_number=sequence_number
                )

                sequence_number += 1
                yield event

                await asyncio.sleep(config.update_interval_seconds)

        except Exception as e:
            self.logger.error(f"文件流连接异常: {e}")
            raise

    async def submit_event_to_processor(self, stream_name: str, event: StreamEvent):
        """向处理器提交事件"""
        if stream_name in self.stream_processors:
            await self.stream_processors[stream_name].submit_event(event)
        else:
            self.logger.warning(f"流处理器不存在: {stream_name}")

    def get_processor_stats(self, stream_name: str) -> Optional[Dict[str, Any]]:
        """获取处理器统计"""
        if stream_name in self.stream_processors:
            return self.stream_processors[stream_name].get_stats()
        return None

    def get_all_processor_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有处理器统计"""
        stats = {}
        for name, processor in self.stream_processors.items():
            stats[name] = processor.get_stats()
        return stats

    async def close(self):
        """关闭连接器"""
        # 停止所有处理器
        for processor in self.stream_processors.values():
            await processor.stop()

        # 关闭会话
        if self.session:
            await self.session.close()

        self.logger.info("数据流连接器已关闭")


class RealtimeDataProcessor:
    """实时数据处理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 初始化组件
        self.stream_connector = DataStreamConnector(config.get('connector_config', {}))
        self.redis_cache = RedisCache(config.get('redis_config', {}))
        self.monitor = DataCollectionMonitor(config.get('monitor_config', {}))

        # 流管理
        self.active_streams: Dict[str, asyncio.Task] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}

        # 统计信息
        self.stats = {
            'streams_active': 0,
            'events_processed_total': 0,
            'processing_errors': 0,
            'uptime_seconds': 0
        }

        self.start_time = datetime.now()

    async def start_stream_processing(self, stream_configs: List[StreamConfig]):
        """
        启动流处理

        Args:
            stream_configs: 流配置列表
        """
        self.logger.info(f"启动实时数据处理: {len(stream_configs)} 个流")

        for stream_config in stream_configs:
            if stream_config.stream_name not in self.active_streams:
                # 启动流处理任务
                task = asyncio.create_task(self._process_stream(stream_config))
                self.active_streams[stream_config.stream_name] = task
                self.stats['streams_active'] += 1

        self.logger.info("实时数据处理启动完成")

    async def stop_stream_processing(self):
        """停止流处理"""
        self.logger.info("正在停止实时数据处理")

        # 取消所有流任务
        for stream_name, task in self.active_streams.items():
            if not task.done():
                task.cancel()

        # 等待任务完成
        await asyncio.gather(*list(self.active_streams.values()), return_exceptions=True)

        # 关闭连接器
        await self.stream_connector.close()

        self.stats['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()

        self.logger.info("实时数据处理已停止")

    async def _process_stream(self, stream_config: StreamConfig):
        """处理单个流"""
        stream_name = stream_config.stream_name

        try:
            self.logger.info(f"开始处理流: {stream_name}")

            async for event in self.stream_connector.connect_stream(stream_config):
                # 处理事件
                await self._handle_stream_event(event)

                # 更新统计
                self.stats['events_processed_total'] += 1

        except asyncio.CancelledError:
            self.logger.info(f"流处理被取消: {stream_name}")
        except Exception as e:
            self.logger.error(f"流处理异常 {stream_name}: {e}")
            self.stats['processing_errors'] += 1

        finally:
            if stream_name in self.active_streams:
                del self.active_streams[stream_name]
                self.stats['streams_active'] -= 1

    async def _handle_stream_event(self, event: StreamEvent):
        """处理流事件"""
        try:
            # 缓存实时数据
            await self._cache_realtime_data(event)

            # 触发事件处理器
            await self._trigger_event_handlers(event)

            # 记录监控信息
            await self.monitor.record_realtime_event(
                event.stream_name,
                event.event_type,
                event.timestamp
            )

        except Exception as e:
            self.logger.error(f"事件处理失败 {event.event_id}: {e}")

    async def _cache_realtime_data(self, event: StreamEvent):
        """缓存实时数据"""
        try:
            # 构建缓存键
            if event.event_type == "market_data":
                symbol = event.data.get('symbol', 'unknown')
                cache_key = f"realtime:market:{symbol}"

                # 缓存最新市场数据
                await self.redis_cache.set_json(cache_key, {
                    'data': event.data,
                    'timestamp': event.timestamp.isoformat(),
                    'source': event.source
                }, expire_seconds=300)  # 5分钟过期

            elif event.event_type == "trade_data":
                # 缓存交易数据摘要
                cache_key = "realtime:trades:latest"
                await self.redis_cache.set_json(cache_key, {
                    'summary': event.data,
                    'timestamp': event.timestamp.isoformat()
                }, expire_seconds=60)  # 1分钟过期

        except Exception as e:
            self.logger.error(f"实时数据缓存失败: {e}")

    async def _trigger_event_handlers(self, event: StreamEvent):
        """触发事件处理器"""
        handlers = self.event_handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                self.logger.error(f"事件处理器异常: {e}")

    def register_event_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        self.event_handlers[event_type].append(handler)
        self.logger.info(f"注册事件处理器: {event_type}")

    def unregister_event_handler(self, event_type: str, handler: Callable):
        """注销事件处理器"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
                self.logger.info(f"注销事件处理器: {event_type}")
            except ValueError:
                pass

    def get_stream_stats(self, stream_name: str) -> Optional[Dict[str, Any]]:
        """获取流统计"""
        return self.stream_connector.get_processor_stats(stream_name)

    def get_all_stream_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有流统计"""
        return self.stream_connector.get_all_processor_stats()

    def get_processor_stats(self) -> Dict[str, Any]:
        """获取处理器统计"""
        stats = self.stats.copy()
        stats['active_streams'] = list(self.active_streams.keys())
        stats['registered_handlers'] = list(self.event_handlers.keys())
        stats['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()

        return stats

    async def query_realtime_data(self, symbol: str, data_type: str = "market") -> Optional[Dict[str, Any]]:
        """
        查询实时数据

        Args:
            symbol: 标的代码
            data_type: 数据类型

        Returns:
            实时数据
        """
        try:
            cache_key = f"realtime:{data_type}:{symbol}"
            cached_data = await self.redis_cache.get_json(cache_key)

            if cached_data:
                # 检查数据是否过期（5分钟内）
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if (datetime.now() - cache_time) < timedelta(minutes=5):
                    return cached_data

            return None

        except Exception as e:
            self.logger.error(f"查询实时数据失败 {symbol}: {e}")
            return None

    async def broadcast_event(self, event: StreamEvent):
        """广播事件到所有相关的流处理器"""
        # 查找相关的流处理器
        for stream_name, processor in self.stream_connector.stream_processors.items():
            if stream_name in event.stream_name or event.source in stream_name:
                await self.stream_connector.submit_event_to_processor(stream_name, event)

    async def create_derived_stream(self, source_stream: str, derived_config: StreamConfig,
                                  transformation_func: Callable) -> str:
        """
        创建衍生流

        Args:
            source_stream: 源流名称
            derived_config: 衍生流配置
            transformation_func: 转换函数

        Returns:
            衍生流名称
        """
        derived_name = f"{source_stream}_derived_{derived_config.stream_name}"

        # 注册事件处理器来创建衍生数据
        async def derived_event_handler(event: StreamEvent):
            try:
                # 应用转换函数
                derived_data = await transformation_func(event.data)

                # 创建衍生事件
                derived_event = StreamEvent(
                    event_id=f"{event.event_id}_derived",
                    stream_name=derived_name,
                    event_type=f"{event.event_type}_derived",
                    data=derived_data,
                    timestamp=datetime.now(),
                    source=f"derived_from_{event.source}",
                    metadata={
                        'original_event': event.event_id,
                        'transformation': transformation_func.__name__
                    }
                )

                # 广播衍生事件
                await self.broadcast_event(derived_event)

            except Exception as e:
                self.logger.error(f"衍生事件处理失败: {e}")

        # 注册处理器
        self.register_event_handler("market_data", derived_event_handler)

        # 启动衍生流处理
        await self.start_stream_processing([derived_config])

        self.logger.info(f"创建衍生流: {derived_name}")
        return derived_name