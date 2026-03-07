#!/usr/bin/env python3
"""
RQA2025 基础设施层消息路由器

负责消息的路由、分发和管理。
这是从ComponentBus中拆分出来的消息路由组件。
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .component_bus import Message, MessageType

logger = logging.getLogger(__name__)


class MessageRouter:
    """
    消息路由器

    负责消息的路由、分发和管理，支持同步和异步消息处理。
    """

    def __init__(self, max_workers: int = 4, queue_size: int = 1000):
        """
        初始化消息路由器

        Args:
            max_workers: 最大工作线程数
            queue_size: 消息队列大小
        """
        self.max_workers = max_workers
        self.message_queue = queue.Queue(maxsize=queue_size)
        self.response_handlers: Dict[str, Callable] = {}
        self._lock = threading.RLock()

        # 异步处理
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="MessageRouter")
        self.processing_thread = None
        self.stop_event = threading.Event()

        # 统计信息
        self.stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'messages_queued': 0,
            'async_operations': 0,
            'start_time': datetime.now()
        }

        # 启动异步处理
        self._start_async_processing()

        logger.info(f"消息路由器初始化完成，最大工作线程: {max_workers}")

    def route_message(self, message: Message, callback: Optional[Callable] = None) -> Optional[Any]:
        """
        路由消息

        Args:
            message: 消息对象
            callback: 异步回调函数

        Returns:
            Optional[Any]: 同步响应结果
        """
        try:
            self.stats['messages_processed'] += 1

            # 记录消息路由
            logger.debug(f"路由消息: {message.type} -> {message.target}")

            # 处理同步消息
            if message.type in [MessageType.COMMAND, MessageType.QUERY]:
                return self._handle_sync_message(message)

            # 处理异步消息
            elif message.type in [MessageType.EVENT, MessageType.NOTIFICATION]:
                return self._handle_async_message(message, callback)

            else:
                logger.warning(f"未知消息类型: {message.type}")
                return None

        except Exception as e:
            self.stats['messages_failed'] += 1
            logger.error(f"消息路由失败: {e}")
            return None

    def register_response_handler(self, message_type: str, handler: Callable):
        """
        注册响应处理器

        Args:
            message_type: 消息类型
            handler: 处理函数
        """
        with self._lock:
            self.response_handlers[message_type] = handler
            logger.debug(f"注册响应处理器: {message_type}")

    def unregister_response_handler(self, message_type: str):
        """
        注销响应处理器

        Args:
            message_type: 消息类型
        """
        with self._lock:
            self.response_handlers.pop(message_type, None)
            logger.debug(f"注销响应处理器: {message_type}")

    def broadcast_message(self, message: Message, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        广播消息到多个目标

        Args:
            message: 消息对象
            targets: 目标列表，如果为None则广播到所有订阅者

        Returns:
            Dict[str, Any]: 广播结果
        """
        results = {}
        futures = []

        # 提交异步任务
        if targets:
            for target in targets:
                msg_copy = Message(
                    type=message.type,
                    target=target,
                    source=message.source,
                    data=message.data.copy() if message.data else {},
                    correlation_id=message.correlation_id
                )
                future = self.executor.submit(self.route_message, msg_copy)
                futures.append((target, future))
        else:
            # 全局广播 - 这里需要依赖订阅管理器
            logger.warning("全局广播需要订阅管理器支持")

        # 收集结果
        for target, future in futures:
            try:
                result = future.result(timeout=30)
                results[target] = {'success': True, 'result': result}
            except Exception as e:
                results[target] = {'success': False, 'error': str(e)}

        return {
            'total_targets': len(targets) if targets else 0,
            'successful': sum(1 for r in results.values() if r['success']),
            'failed': sum(1 for r in results.values() if not r['success']),
            'results': results
        }

    def get_message_stats(self) -> Dict[str, Any]:
        """
        获取消息统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        current_time = datetime.now()
        uptime = (current_time - self.stats['start_time']).total_seconds()

        return {
            'messages_processed': self.stats['messages_processed'],
            'messages_failed': self.stats['messages_failed'],
            'messages_queued': self.stats['messages_queued'],
            'async_operations': self.stats['async_operations'],
            'success_rate': (self.stats['messages_processed'] - self.stats['messages_failed']) / max(self.stats['messages_processed'], 1),
            'queue_size': self.message_queue.qsize(),
            'uptime_seconds': uptime,
            'throughput_per_second': self.stats['messages_processed'] / max(uptime, 1)
        }

    def _handle_sync_message(self, message: Message) -> Optional[Any]:
        """
        处理同步消息

        Args:
            message: 消息对象

        Returns:
            Optional[Any]: 处理结果
        """
        # 查找响应处理器
        handler = self.response_handlers.get(str(message.type))
        if handler:
            try:
                return handler(message)
            except Exception as e:
                logger.error(f"同步消息处理失败: {e}")
                return None
        else:
            logger.warning(f"未找到同步消息处理器: {message.type}")
            return None

    def _handle_async_message(self, message: Message, callback: Optional[Callable] = None):
        """
        处理异步消息

        Args:
            message: 消息对象
            callback: 回调函数
        """
        try:
            # 放入队列
            self.message_queue.put((message, callback), timeout=1)
            self.stats['messages_queued'] += 1
            self.stats['async_operations'] += 1

        except queue.Full:
            logger.error(f"消息队列已满，丢弃消息: {message.type}")
            self.stats['messages_failed'] += 1

    def _start_async_processing(self):
        """启动异步消息处理"""
        if self.processing_thread and self.processing_thread.is_alive():
            return

        self.processing_thread = threading.Thread(
            target=self._process_messages_async,
            name="MessageRouter-AsyncProcessor",
            daemon=True
        )
        self.processing_thread.start()
        logger.info("异步消息处理线程已启动")

    def _process_messages_async(self):
        """异步处理消息队列"""
        logger.info("开始异步消息处理")

        while not self.stop_event.is_set():
            try:
                # 获取消息（带超时）
                try:
                    message, callback = self.message_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # 处理消息
                self._handle_async_message_internal(message, callback)

                # 标记任务完成
                self.message_queue.task_done()

            except Exception as e:
                logger.error(f"异步消息处理异常: {e}")
                time.sleep(0.1)  # 短暂延迟避免忙等

        logger.info("异步消息处理线程已停止")

    def _handle_async_message_internal(self, message: Message, callback: Optional[Callable]):
        """
        内部异步消息处理

        Args:
            message: 消息对象
            callback: 回调函数
        """
        try:
            # 这里可以根据消息类型进行不同处理
            # 目前主要处理事件和通知类型消息

            if message.type == MessageType.EVENT:
                # 事件处理 - 通常不需要返回值
                logger.debug(f"处理事件消息: {message.target}")
                # 可以在这里添加事件处理逻辑

            elif message.type == MessageType.NOTIFICATION:
                # 通知处理
                logger.debug(f"处理通知消息: {message.target}")
                # 可以在这里添加通知处理逻辑

            # 调用回调
            if callback:
                try:
                    callback(message, None)  # result暂时为None
                except Exception as e:
                    logger.error(f"异步回调执行失败: {e}")

        except Exception as e:
            logger.error(f"异步消息处理失败: {e}")
            self.stats['messages_failed'] += 1

            # 调用错误回调
            if callback:
                try:
                    callback(message, e)
                except Exception as callback_error:
                    logger.error(f"错误回调执行失败: {callback_error}")

    def collect_results(self, futures: List, timeout: float = 30.0) -> Dict[str, Any]:
        """
        收集异步操作结果

        Args:
            futures: Future对象列表
            timeout: 超时时间

        Returns:
            Dict[str, Any]: 收集结果
        """
        results = []
        completed = 0
        failed = 0

        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
                completed += 1
            except Exception as e:
                logger.error(f"异步操作失败: {e}")
                results.append({'error': str(e)})
                failed += 1

        return {
            'total': len(futures),
            'completed': completed,
            'failed': failed,
            'results': results
        }

    def shutdown(self, timeout: float = 5.0):
        """
        关闭消息路由器

        Args:
            timeout: 关闭超时时间
        """
        logger.info("正在关闭消息路由器...")

        # 设置停止标志
        self.stop_event.set()

        # 等待异步处理线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=timeout)

        # 关闭线程池
        self.executor.shutdown(wait=True, timeout=timeout)

        logger.info("消息路由器已关闭")

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取消息路由器的健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            stats = self.get_message_stats()

            issues = []

            # 检查队列积压
            if stats['queue_size'] > self.message_queue.maxsize * 0.8:
                issues.append(f"消息队列积压严重: {stats['queue_size']}")

            # 检查失败率
            if stats['success_rate'] < 0.95:
                issues.append(".1%")

            # 检查线程状态
            if not self.processing_thread or not self.processing_thread.is_alive():
                issues.append("异步处理线程未运行")

            return {
                'status': 'healthy' if not issues else 'warning',
                'stats': stats,
                'issues': issues,
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取健康状态失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


# 全局消息路由器实例
global_message_router = MessageRouter()
