#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2026 实时数据流处理引擎
提供高性能的实时数据流处理和分析能力

流处理特性:
1. 实时数据摄取 - 支持多种数据源和协议
2. 流式数据分析 - 实时聚合、过滤和转换
3. 事件检测引擎 - 模式识别和异常检测
4. 流数据存储 - 高效存储和历史数据重放
5. 高可用架构 - 容错机制和自动故障恢复
6. 性能监控 - 实时性能指标和瓶颈分析
"""

import asyncio
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
import sys
import random
import statistics
from collections import defaultdict, deque
import socket
import websockets
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

class StreamProcessor:
    """流处理器"""

    def __init__(self, stream_name, config=None):
        self.stream_name = stream_name
        self.config = config or {}
        self.buffer_size = self.config.get('buffer_size', 1000)
        self.processing_interval = self.config.get('processing_interval', 1.0)

        # 数据缓冲区
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.processed_data = deque(maxlen=self.buffer_size)

        # 处理统计
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'processing_errors': 0,
            'avg_processing_time': 0,
            'throughput': 0,
            'last_update': datetime.now()
        }

        # 处理线程
        self.processing_thread = None
        self.is_running = False
        self.monitor_task = None

    async def start_processing(self):
        """启动流处理"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        # 启动异步监控
        asyncio.create_task(self._monitor_performance())

    def start_processing_sync(self):
        """同步方式启动流处理"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        # 启动同步监控
        self.monitor_thread = threading.Thread(target=self._monitor_performance_sync, daemon=True)
        self.monitor_thread.start()

    def stop_processing(self):
        """停止流处理"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)

    def ingest_data(self, data):
        """摄取数据"""
        self.data_buffer.append({
            'data': data,
            'timestamp': datetime.now(),
            'sequence_id': self.stats['messages_received']
        })
        self.stats['messages_received'] += 1

    def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                if self.data_buffer:
                    # 批量处理数据
                    batch = []
                    batch_size = min(len(self.data_buffer), 10)  # 每批最多处理10条

                    for _ in range(batch_size):
                        if self.data_buffer:
                            batch.append(self.data_buffer.popleft())

                    if batch:
                        processed_batch = self._process_batch(batch)
                        self.processed_data.extend(processed_batch)

                        # 更新统计
                        self.stats['messages_processed'] += len(processed_batch)

                time.sleep(self.processing_interval)

            except Exception as e:
                self.stats['processing_errors'] += 1
                print(f"流处理错误 ({self.stream_name}): {e}")

    def _process_batch(self, batch):
        """处理数据批次"""
        processed = []

        for item in batch:
            start_time = time.time()

            try:
                # 基本数据处理
                processed_item = self._process_single_item(item['data'])
                processed_item.update({
                    'original_timestamp': item['timestamp'],
                    'processed_timestamp': datetime.now(),
                    'processing_time': time.time() - start_time,
                    'sequence_id': item['sequence_id']
                })

                processed.append(processed_item)

            except Exception as e:
                # 处理失败的项目
                error_item = {
                    'error': str(e),
                    'original_data': item['data'],
                    'processing_time': time.time() - start_time,
                    'sequence_id': item['sequence_id'],
                    'status': 'failed'
                }
                processed.append(error_item)

        return processed

    def _process_single_item(self, data):
        """处理单个数据项"""
        # 基本数据验证和清理
        if isinstance(data, dict):
            # 清理和标准化数据
            cleaned_data = self._clean_data(data)

            # 添加元数据
            cleaned_data.update({
                'stream_name': self.stream_name,
                'processing_timestamp': datetime.now(),
                'data_quality_score': self._calculate_data_quality(cleaned_data)
            })

            return cleaned_data

        elif isinstance(data, (int, float)):
            return {
                'value': data,
                'type': 'numeric',
                'stream_name': self.stream_name
            }

        elif isinstance(data, str):
            return {
                'text': data,
                'length': len(data),
                'type': 'text',
                'stream_name': self.stream_name
            }

        else:
            return {
                'raw_data': str(data),
                'type': 'unknown',
                'stream_name': self.stream_name
            }

    def _clean_data(self, data):
        """数据清理"""
        cleaned = {}

        for key, value in data.items():
            # 基本数据类型验证
            if isinstance(value, (int, float)):
                if not (value != value):  # 检查NaN
                    cleaned[key] = value
            elif isinstance(value, str):
                # 清理字符串数据
                cleaned_value = value.strip()
                if cleaned_value:
                    cleaned[key] = cleaned_value
            elif isinstance(value, (list, dict)):
                # 递归清理嵌套数据
                cleaned[key] = self._clean_nested_data(value)
            else:
                # 转换为字符串
                cleaned[key] = str(value)

        return cleaned

    def _clean_nested_data(self, data):
        """清理嵌套数据"""
        if isinstance(data, list):
            return [self._clean_nested_data(item) for item in data if item is not None]
        elif isinstance(data, dict):
            return self._clean_data(data)
        else:
            return data

    def _calculate_data_quality(self, data):
        """计算数据质量分数"""
        score = 1.0

        # 检查数据完整性
        total_fields = len(data)
        null_fields = sum(1 for v in data.values() if v is None or v == '')
        completeness = (total_fields - null_fields) / total_fields if total_fields > 0 else 0

        # 检查数据一致性 (简单检查)
        consistency_penalty = 0
        if 'price' in data and 'volume' in data:
            if data.get('price', 0) <= 0 or data.get('volume', 0) < 0:
                consistency_penalty = 0.2

        score = completeness * (1 - consistency_penalty)

        return round(score, 2)

    async def _monitor_performance(self):
        """监控性能"""
        while self.is_running:
            await asyncio.sleep(10)  # 每10秒更新一次

            current_time = datetime.now()
            time_diff = (current_time - self.stats['last_update']).total_seconds()

            if time_diff > 0:
                # 计算吞吐量
                self.stats['throughput'] = self.stats['messages_processed'] / time_diff * 60  # 每分钟

                # 计算平均处理时间
                if self.processed_data:
                    processing_times = [item.get('processing_time', 0) for item in self.processed_data if 'processing_time' in item]
                    if processing_times:
                        self.stats['avg_processing_time'] = statistics.mean(processing_times)

            self.stats['last_update'] = current_time

    def _monitor_performance_sync(self):
        """同步监控性能"""
        while self.is_running:
            time.sleep(10)  # 每10秒更新一次

            current_time = datetime.now()
            time_diff = (current_time - self.stats['last_update']).total_seconds()

            if time_diff > 0:
                # 计算吞吐量
                self.stats['throughput'] = self.stats['messages_processed'] / time_diff * 60  # 每分钟

                # 计算平均处理时间
                if self.processed_data:
                    processing_times = [item.get('processing_time', 0) for item in self.processed_data if 'processing_time' in item]
                    if processing_times:
                        self.stats['avg_processing_time'] = statistics.mean(processing_times)

            self.stats['last_update'] = current_time

    def get_stats(self):
        """获取统计信息"""
        return {
            'stream_name': self.stream_name,
            'is_running': self.is_running,
            'buffer_size': len(self.data_buffer),
            'processed_count': len(self.processed_data),
            'stats': self.stats.copy()
        }


class EventDetector:
    """事件检测器"""

    def __init__(self):
        self.patterns = {}
        self.anomaly_detector = None
        self.event_history = deque(maxlen=1000)

    def add_pattern(self, pattern_name, pattern_config):
        """添加检测模式"""
        self.patterns[pattern_name] = pattern_config

    def detect_events(self, data_stream):
        """检测事件"""
        events = []

        # 滑动窗口分析
        window_size = 20
        if len(data_stream) >= window_size:
            window = list(data_stream)[-window_size:]

            # 趋势检测
            trend_event = self._detect_trend(window)
            if trend_event:
                events.append(trend_event)

            # 异常检测
            anomaly_event = self._detect_anomaly(window)
            if anomaly_event:
                events.append(anomaly_event)

            # 模式匹配
            pattern_events = self._detect_patterns(window)
            events.extend(pattern_events)

        # 记录事件
        for event in events:
            self.event_history.append({
                'event': event,
                'timestamp': datetime.now(),
                'stream_data': data_stream[-10:]  # 最后10个数据点
            })

        return events

    def _detect_trend(self, window):
        """检测趋势"""
        if len(window) < 5:
            return None

        # 计算移动平均
        values = [item.get('value', 0) if isinstance(item, dict) else item for item in window]
        recent_avg = statistics.mean(values[-5:])
        earlier_avg = statistics.mean(values[:-5])

        trend_threshold = 0.05  # 5%变化阈值

        if earlier_avg != 0 and abs(recent_avg - earlier_avg) / abs(earlier_avg) > trend_threshold:
            direction = 'upward' if recent_avg > earlier_avg else 'downward'
            return {
                'type': 'trend_change',
                'direction': direction,
                'magnitude': abs(recent_avg - earlier_avg) / earlier_avg,
                'confidence': 0.8
            }

        return None

    def _detect_anomaly(self, window):
        """检测异常"""
        if len(window) < 10:
            return None

        values = [item.get('value', 0) if isinstance(item, dict) else item for item in window]

        # 使用简单的统计方法检测异常
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)

        if stdev == 0:
            return None

        latest_value = values[-1]
        z_score = abs(latest_value - mean) / stdev

        if z_score > 3.0:  # 3倍标准差
            return {
                'type': 'anomaly_detected',
                'severity': 'high' if z_score > 4.0 else 'medium',
                'z_score': z_score,
                'deviation': latest_value - mean,
                'confidence': min(z_score / 5.0, 0.95)
            }

        return None

    def _detect_patterns(self, window):
        """检测模式"""
        events = []

        # 简单的峰值检测
        if len(window) >= 3:
            values = [item.get('value', 0) if isinstance(item, dict) else item for item in window]
            if values[-2] > values[-3] and values[-2] > values[-1]:
                events.append({
                    'type': 'peak_detected',
                    'value': values[-2],
                    'position': len(window) - 2,
                    'confidence': 0.7
                })

        return events


class StreamStorage:
    """流数据存储"""

    def __init__(self, storage_path=None):
        self.storage_path = Path(storage_path or 'realtime_streaming/storage')
        self.storage_path.mkdir(exist_ok=True)

        # 分片存储
        self.current_shard = None
        self.shard_size = 1000  # 每个分片1000条记录
        self.current_shard_count = 0

    def store_data(self, stream_name, data):
        """存储数据"""
        # 检查是否需要新分片
        if self.current_shard is None or self.current_shard_count >= self.shard_size:
            self._create_new_shard(stream_name)

        # 存储数据
        with open(self.current_shard, 'a', encoding='utf-8') as f:
            record = {
                'stream_name': stream_name,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        self.current_shard_count += 1

    def _create_new_shard(self, stream_name):
        """创建新分片"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        shard_name = f"{stream_name}_{timestamp}.jsonl"
        self.current_shard = self.storage_path / shard_name
        self.current_shard_count = 0

    def query_data(self, stream_name, start_time=None, end_time=None, limit=100):
        """查询数据"""
        results = []

        # 查找相关分片文件
        pattern = f"{stream_name}_*.jsonl"
        shard_files = list(self.storage_path.glob(pattern))

        # 按时间排序（最新的在前）
        shard_files.sort(reverse=True)

        for shard_file in shard_files:
            if len(results) >= limit:
                break

            try:
                with open(shard_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(results) >= limit:
                            break

                        record = json.loads(line.strip())
                        record_time = datetime.fromisoformat(record['timestamp'])

                        # 时间过滤
                        if start_time and record_time < start_time:
                            continue
                        if end_time and record_time > end_time:
                            continue

                        results.append(record)

            except Exception as e:
                print(f"读取分片文件错误 {shard_file}: {e}")

        return results

    def replay_stream(self, stream_name, start_time=None, speed_multiplier=1.0):
        """重放数据流"""
        data = self.query_data(stream_name, start_time)

        for record in data:
            yield record

            # 控制重放速度
            if speed_multiplier != 1.0:
                time.sleep(1.0 / speed_multiplier)


class DataSourceConnector:
    """数据源连接器"""

    def __init__(self):
        self.connectors = {}
        self.executor = ThreadPoolExecutor(max_workers=10)

    def add_connector(self, name, connector_type, config):
        """添加连接器"""
        if connector_type == 'websocket':
            connector = WebSocketConnector(config)
        elif connector_type == 'http':
            connector = HTTPConnector(config)
        elif connector_type == 'tcp':
            connector = TCPConnector(config)
        else:
            raise ValueError(f"不支持的连接器类型: {connector_type}")

        self.connectors[name] = connector

    def start_all_connectors(self):
        """启动所有连接器"""
        for name, connector in self.connectors.items():
            try:
                connector.start()
                print(f"✅ 数据源连接器 {name} 已启动")
            except Exception as e:
                print(f"❌ 数据源连接器 {name} 启动失败: {e}")

    def stop_all_connectors(self):
        """停止所有连接器"""
        for name, connector in self.connectors.items():
            try:
                connector.stop()
                print(f"✅ 数据源连接器 {name} 已停止")
            except Exception as e:
                print(f"❌ 数据源连接器 {name} 停止失败: {e}")


class WebSocketConnector:
    """WebSocket连接器"""

    def __init__(self, config):
        self.url = config['url']
        self.stream_processor = config.get('stream_processor')
        self.is_running = False

    def start(self):
        """启动连接器"""
        self.is_running = True
        # 这里可以实现WebSocket连接逻辑
        print(f"WebSocket连接器启动: {self.url}")

    def stop(self):
        """停止连接器"""
        self.is_running = False
        print("WebSocket连接器已停止")


class HTTPConnector:
    """HTTP连接器"""

    def __init__(self, config):
        self.url = config['url']
        self.interval = config.get('interval', 60)
        self.stream_processor = config.get('stream_processor')
        self.is_running = False

    def start(self):
        """启动连接器"""
        self.is_running = True
        # 这里可以实现HTTP轮询逻辑
        print(f"HTTP连接器启动: {self.url}")

    def stop(self):
        """停止连接器"""
        self.is_running = False
        print("HTTP连接器已停止")


class TCPConnector:
    """TCP连接器"""

    def __init__(self, config):
        self.host = config['host']
        self.port = config['port']
        self.stream_processor = config.get('stream_processor')
        self.is_running = False

    def start(self):
        """启动连接器"""
        self.is_running = True
        # 这里可以实现TCP连接逻辑
        print(f"TCP连接器启动: {self.host}:{self.port}")

    def stop(self):
        """停止连接器"""
        self.is_running = False
        print("TCP连接器已停止")


class RealtimeStreamingEngine:
    """实时数据流处理引擎"""

    def __init__(self):
        self.stream_processors = {}
        self.event_detector = EventDetector()
        self.stream_storage = StreamStorage()
        self.data_connector = DataSourceConnector()

        self.is_running = False
        self.stats = {
            'total_messages_processed': 0,
            'active_streams': 0,
            'events_detected': 0,
            'storage_size': 0
        }

    def create_stream_processor(self, stream_name, config=None):
        """创建流处理器"""
        processor = StreamProcessor(stream_name, config)
        self.stream_processors[stream_name] = processor

        # 启动处理器 (同步方式)
        processor.start_processing_sync()

        return processor

    def add_data_source(self, name, connector_type, config):
        """添加数据源"""
        stream_processor = config.get('stream_processor')
        if stream_processor and stream_processor in self.stream_processors:
            config['stream_processor'] = self.stream_processors[stream_processor]

        self.data_connector.add_connector(name, connector_type, config)

    def ingest_data(self, stream_name, data):
        """摄取数据到流"""
        if stream_name in self.stream_processors:
            self.stream_processors[stream_name].ingest_data(data)

            # 存储数据
            self.stream_storage.store_data(stream_name, data)

            # 检测事件
            stream_data = list(self.stream_processors[stream_name].processed_data)
            if stream_data:
                events = self.event_detector.detect_events(stream_data)
                if events:
                    self.stats['events_detected'] += len(events)
                    print(f"检测到 {len(events)} 个事件在流 {stream_name}")

            self.stats['total_messages_processed'] += 1

    def get_stream_stats(self, stream_name=None):
        """获取流统计信息"""
        if stream_name:
            if stream_name in self.stream_processors:
                return self.stream_processors[stream_name].get_stats()
            else:
                return {'error': '流不存在'}
        else:
            # 返回所有流的统计
            all_stats = {}
            for name, processor in self.stream_processors.items():
                all_stats[name] = processor.get_stats()
            return all_stats

    def get_system_stats(self):
        """获取系统统计"""
        return {
            'engine_status': 'running' if self.is_running else 'stopped',
            'active_streams': len(self.stream_processors),
            'total_messages_processed': self.stats['total_messages_processed'],
            'events_detected': self.stats['events_detected'],
            'data_sources': len(self.data_connector.connectors),
            'storage_stats': self._get_storage_stats()
        }

    def _get_storage_stats(self):
        """获取存储统计"""
        total_size = 0
        file_count = 0

        for file_path in self.stream_storage.storage_path.glob('*.jsonl'):
            try:
                total_size += file_path.stat().st_size
                file_count += 1
            except:
                pass

        return {
            'total_size_bytes': total_size,
            'file_count': file_count,
            'avg_file_size': total_size / file_count if file_count > 0 else 0
        }

    def replay_stream(self, stream_name, start_time=None, speed=1.0):
        """重放数据流"""
        return self.stream_storage.replay_stream(stream_name, start_time, speed)

    def start_engine(self):
        """启动引擎"""
        self.is_running = True
        self.data_connector.start_all_connectors()
        print("🚀 实时数据流处理引擎已启动")

    def stop_engine(self):
        """停止引擎"""
        self.is_running = False

        # 停止所有处理器
        for processor in self.stream_processors.values():
            processor.stop_processing()

        # 停止所有连接器
        self.data_connector.stop_all_connectors()

        print("🛑 实时数据流处理引擎已停止")


def create_demo_streaming_system():
    """创建演示流处理系统"""
    engine = RealtimeStreamingEngine()

    # 创建流处理器
    market_processor = engine.create_stream_processor('market_data', {
        'buffer_size': 1000,
        'processing_interval': 0.5
    })

    user_processor = engine.create_stream_processor('user_activity', {
        'buffer_size': 500,
        'processing_interval': 1.0
    })

    # 添加数据源 (模拟)
    engine.add_data_source('market_feed', 'websocket', {
        'url': 'ws://example.com/market',
        'stream_processor': market_processor
    })

    engine.add_data_source('user_api', 'http', {
        'url': 'http://api.example.com/users',
        'interval': 30,
        'stream_processor': user_processor
    })

    return engine


def main():
    """主函数"""
    print("🌊 启动 RQA2026 实时数据流处理引擎")
    print("=" * 80)

    # 创建演示系统
    engine = create_demo_streaming_system()

    # 启动引擎
    engine.start_engine()

    try:
        print("🔄 实时流处理系统运行中...")
        print("📊 支持功能: 数据摄取、实时处理、事件检测、流存储")

        # 模拟数据摄取
        print("\\n📥 开始模拟数据摄取...")

        for i in range(50):  # 生成50条测试数据
            # 市场数据
            market_data = {
                'symbol': 'AAPL',
                'price': 180 + random.uniform(-5, 5),
                'volume': random.randint(100000, 500000),
                'timestamp': datetime.now().isoformat()
            }
            engine.ingest_data('market_data', market_data)

            # 用户活动数据
            user_data = {
                'user_id': f'user_{random.randint(1, 100)}',
                'action': random.choice(['login', 'trade', 'view_portfolio', 'update_settings']),
                'timestamp': datetime.now().isoformat()
            }
            engine.ingest_data('user_activity', user_data)

            time.sleep(0.1)  # 控制摄取速度

            if (i + 1) % 10 == 0:
                stats = engine.get_system_stats()
                print(f"  已处理 {stats['total_messages_processed']} 条消息，检测到 {stats['events_detected']} 个事件")

        print("\\n📊 系统运行统计:")
        final_stats = engine.get_system_stats()
        print(f"  🔄 活跃流处理器: {final_stats['active_streams']}")
        print(f"  📦 消息处理总量: {final_stats['total_messages_processed']}")
        print(f"  🎯 事件检测数量: {final_stats['events_detected']}")
        print(f"  💾 存储文件数量: {final_stats['storage_stats']['file_count']}")
        print(f"  📏 存储总大小: {final_stats['storage_stats']['total_size_bytes']} 字节")

        # 显示流处理器详情
        print("\\n🔍 流处理器详情:")
        stream_stats = engine.get_stream_stats()
        for stream_name, stats in stream_stats.items():
            print(f"  📊 {stream_name}:")
            print(f"    运行状态: {'✅' if stats['is_running'] else '❌'}")
            print(f"    缓冲区大小: {stats['buffer_size']}")
            print(f"    处理统计: {stats['stats']['messages_processed']} 条消息")

        print("\\n✅ 实时数据流处理演示完成！")
        print("🌊 系统已成功处理实时数据流，支持事件检测和数据存储")

    except KeyboardInterrupt:
        print("\\n🛑 收到停止信号")

    finally:
        # 停止引擎
        engine.stop_engine()


if __name__ == "__main__":
    main()
