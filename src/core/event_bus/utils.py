"""
事件总线工具类
包含重试管理器和性能监控器
"""

from typing import Dict, Any
from collections import defaultdict, deque
import time
import threading
import logging
from queue import Queue

from .models import Event

logger = logging.getLogger(__name__)


class EventRetryManager:

    """事件重试管理器 - 优化版"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, dead_letter_callback=None):

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_queue = Queue()
        self._running = False
        self._retry_thread = None
        self.dead_letter_callback = dead_letter_callback  # 死信队列回调函数

    def start(self):
        """启动重试管理器"""
        if not self._running:
            self._running = True
            self._retry_thread = threading.Thread(target=self._retry_worker, daemon=True)
            self._retry_thread.start()

    def stop(self):
        """停止重试管理器"""
        self._running = False
        if self._retry_thread:
            self._retry_thread.join(timeout=5)

    def add_retry_event(self, event: Event, error: Exception = None):
        """添加重试事件"""
        if event.retry_count < event.max_retries:
            event.retry_count += 1
            self.retry_queue.put((event, error))

    def _retry_worker(self):
        """重试工作线程"""
        from queue import Empty
        
        while self._running:
            try:
                event, error = self.retry_queue.get(timeout=1)
                time.sleep(self.retry_delay)

                # 检查是否达到最大重试次数
                if event.retry_count >= event.max_retries:
                    logger.warning(f"事件重试失败，已达到最大重试次数: {event.event_id}")
                    if self.dead_letter_callback:
                        try:
                            self.dead_letter_callback(event, error)
                        except Exception as e:
                            logger.error(f"调用死信队列回调失败: {e}")
                else:
                    # 重新发布事件进行重试
                    logger.info(f"重试事件: {event.event_id}, 重试次数: {event.retry_count}")
                    # 注意：这里需要EventBus提供重新发布的方法
                    # 暂时只记录日志，实际重试由EventBus处理

            except Empty:
                # 队列为空是正常情况（timeout），不需要记录错误
                continue
            except Exception as e:
                logger.error(f"重试工作线程异常: {e}", exc_info=True)
                continue


class EventPerformanceMonitor:

    """事件性能监控 - 优化版"""

    def __init__(self):

        self.stats = defaultdict(lambda: {
            'total_count': 0,
            'success_count': 0,
            'failure_count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'last_updated': time.time()
        })
        self.recent_events = deque(maxlen=1000)  # 限制最近事件数量
        self._lock = threading.Lock()

    def record_event_start(self, event_type: str):
        """记录事件开始"""
        with self._lock:
            self.stats[event_type]['last_updated'] = time.time()

    def record_event_end(self, event_type: str, success: bool, processing_time: float):
        """记录事件结束"""
        with self._lock:
            stats = self.stats[event_type]
            stats['total_count'] += 1
            stats['total_time'] += processing_time

            if success:
                stats['success_count'] += 1
            else:
                stats['failure_count'] += 1

            # 更新平均时间
            stats['avg_time'] = stats['total_time'] / stats['total_count']

            # 更新最小和最大时间
            if processing_time < stats['min_time']:
                stats['min_time'] = processing_time
            if processing_time > stats['max_time']:
                stats['max_time'] = processing_time

            # 记录最近事件
            self.recent_events.append({
                'event_type': event_type,
                'success': success,
                'processing_time': processing_time,
                'timestamp': time.time()
            })

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self._lock:
            total_events = sum(stats['total_count'] for stats in self.stats.values())
            total_success = sum(stats['success_count'] for stats in self.stats.values())
            total_time = sum(stats['total_time'] for stats in self.stats.values())

            return {
                'total_processed_events': total_events,
                'avg_processing_time': total_time / total_events if total_events > 0 else 0,
                'success_rate': total_success / total_events if total_events > 0 else 0,
                'event_type_metrics': dict(self.stats)
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self._lock:
            return dict(self.stats)

    def get_recent_events(self, minutes: int = 5) -> Dict[str, int]:
        """获取最近事件统计"""
        with self._lock:
            cutoff_time = time.time() - (minutes * 60)
            recent_counts = defaultdict(int)

            for event in self.recent_events:
                if event['timestamp'] >= cutoff_time:
                    recent_counts[event['event_type']] += 1

            return dict(recent_counts)

    def clear_stats(self):
        """清除统计"""
        with self._lock:
            self.stats.clear()
            self.recent_events.clear()
