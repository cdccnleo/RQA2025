#!/usr/bin/env python3
"""
RQA2025 基础设施层组件通信总线

提供组件间的解耦通信机制，支持事件驱动架构和异步通信。
"""

from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from datetime import datetime
import asyncio
import threading
import queue
import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class MessageType(Enum):
    """消息类型"""
    EVENT = "event"          # 事件消息
    COMMAND = "command"      # 命令消息
    QUERY = "query"          # 查询消息
    RESPONSE = "response"    # 响应消息
    NOTIFICATION = "notification"  # 通知消息


@dataclass
class Message:
    """组件间消息"""
    message_id: str
    message_type: MessageType
    topic: str
    sender: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    headers: Dict[str, Any] = field(default_factory=dict)
    ttl: Optional[int] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.ttl is None:
            return False
        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.ttl

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'topic': self.topic,
            'sender': self.sender,
            'payload': self.payload,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'headers': self.headers,
            'ttl': self.ttl
        }


@dataclass
class Subscription:
    """订阅信息"""
    subscriber_id: str
    component_name: str
    topic_pattern: str
    handler: Callable[[Message], None]
    priority: int = 0
    active: bool = True


class ComponentBus:
    """
    组件通信总线

    提供组件间的解耦通信，支持：
    - 发布订阅模式
    - 异步消息处理
    - 消息过滤和路由
    - 消息持久化和重试
    """

    def __init__(self, enable_async: bool = True, max_queue_size: int = 1000):
        """
        初始化组件总线

        Args:
            enable_async: 是否启用异步处理
            max_queue_size: 消息队列最大大小
        """
        self.enable_async = enable_async
        self.max_queue_size = max_queue_size

        # 订阅管理
        self.subscriptions: Dict[str, List[Subscription]] = {}
        self._subscription_lock = threading.RLock()

        # 消息队列
        self.message_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._message_sequence = itertools.count()

        # 异步处理
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.running = False

        # 统计信息
        self.messages_processed = 0
        self.messages_dropped = 0
        self.processing_errors = 0

        # 消息处理器映射
        self.message_handlers: Dict[str, Callable] = {}

        if enable_async:
            self._start_async_processing()

    def subscribe(self, component_name: str, topic_pattern: str,
                  handler: Callable[[Message], None], priority: int = 0) -> str:
        """
        订阅消息

        Args:
            component_name: 组件名称
            topic_pattern: 主题模式（支持通配符）
            handler: 消息处理函数
            priority: 处理优先级

        Returns:
            str: 订阅ID
        """
        subscriber_id = f"{component_name}_{topic_pattern}_{id(handler)}"

        subscription = Subscription(
            subscriber_id=subscriber_id,
            component_name=component_name,
            topic_pattern=topic_pattern,
            handler=handler,
            priority=priority,
            active=True
        )

        with self._subscription_lock:
            if topic_pattern not in self.subscriptions:
                self.subscriptions[topic_pattern] = []
            self.subscriptions[topic_pattern].append(subscription)

        logger.info(f"组件 {component_name} 订阅了主题 {topic_pattern}")
        return subscriber_id

    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        取消订阅

        Args:
            subscriber_id: 订阅ID

        Returns:
            bool: 是否成功取消
        """
        with self._subscription_lock:
            for topic_pattern, subscriptions in self.subscriptions.items():
                for i, subscription in enumerate(subscriptions):
                    if subscription.subscriber_id == subscriber_id:
                        subscriptions.pop(i)
                        logger.info(f"取消订阅: {subscriber_id}")
                        return True
        return False

    def publish(self, message: Message) -> bool:
        """
        发布消息

        Args:
            message: 要发布的消息

        Returns:
            bool: 是否成功发布
        """
        try:
            if message.is_expired():
                logger.warning(f"消息已过期: {message.message_id}")
                return False

            # 计算优先级权重（负数因为PriorityQueue是小顶堆）
            priority_weight = -message.priority.value

            sequence = next(self._message_sequence)
            self.message_queue.put((priority_weight, sequence, message), block=False)
            logger.debug(f"消息已发布: {message.topic} from {message.sender}")

            return True

        except queue.Full:
            self.messages_dropped += 1
            logger.warning(f"消息队列已满，丢弃消息: {message.message_id}")
            return False
        except Exception as e:
            logger.error(f"发布消息失败: {e}")
            return False

    def send_command(self, target_component: str, command: str,
                    payload: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
        """
        发送命令并等待响应

        Args:
            target_component: 目标组件
            command: 命令名称
            payload: 命令参数
            timeout: 超时时间（秒）

        Returns:
            Optional[Dict[str, Any]]: 命令响应
        """
        import uuid
        import time

        correlation_id = str(uuid.uuid4())

        command_message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COMMAND,
            topic=f"command.{target_component}.{command}",
            sender="ComponentBus",
            payload=payload,
            correlation_id=correlation_id,
            ttl=timeout
        )

        # 创建响应等待器
        response_event = threading.Event()
        response_data = {}

        def response_handler(response_message: Message):
            if response_message.correlation_id == correlation_id:
                response_data.update(response_message.payload)
                response_event.set()

        # 订阅响应
        response_topic = f"response.{correlation_id}"
        self.subscribe("CommandClient", response_topic, response_handler)

        try:
            # 发送命令
            if self.publish(command_message):
                # 如果是同步模式，立即处理消息
                if not self.enable_async:
                    try:
                        while not self.message_queue.empty():
                            priority_weight, sequence, queued_message = self.message_queue.get(block=False)
                            self._handle_message(queued_message)
                            self.message_queue.task_done()
                    except queue.Empty:
                        pass

                # 等待响应
                if response_event.wait(timeout=timeout):
                    return response_data
                else:
                    logger.warning(f"命令响应超时: {command}")
                    return None
            else:
                logger.error(f"发送命令失败: {command}")
                return None

        finally:
            # 清理订阅
            self.unsubscribe(f"CommandClient_{response_topic}_{id(response_handler)}")

    def query(self, query_topic: str, query_payload: Dict[str, Any],
             timeout: int = 10) -> List[Dict[str, Any]]:
        """
        发送查询并收集响应

        Args:
            query_topic: 查询主题
            query_payload: 查询参数
            timeout: 超时时间（秒）

        Returns:
            List[Dict[str, Any]]: 查询结果列表
        """
        import uuid
        import time

        correlation_id = str(uuid.uuid4())
        results = []
        results_lock = threading.Lock()

        def collect_results(message: Message):
            if message.correlation_id == correlation_id:
                with results_lock:
                    results.append(message.payload)

        # 订阅查询响应
        self.subscribe("QueryClient", f"query.{correlation_id}.response", collect_results)

        try:
            # 发送查询
            query_message = Message(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.QUERY,
                topic=query_topic,
                sender="ComponentBus",
                payload=query_payload,
                correlation_id=correlation_id,
                ttl=timeout
            )

            if self.publish(query_message):
                if not self.enable_async:
                    self._process_messages_async(process_once=True)
                # 等待一段时间收集响应
                time.sleep(min(timeout, 2.0))  # 等待最多2秒或指定超时时间
                if not self.enable_async:
                    self._process_messages_async(process_once=True)
                return results
            else:
                logger.error(f"发送查询失败: {query_topic}")
                return []

        finally:
            # 清理订阅
            self.unsubscribe(f"QueryClient_query.{correlation_id}.response_{id(collect_results)}")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取总线统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._subscription_lock:
            total_subscriptions = sum(len(subs) for subs in self.subscriptions.values())

        return {
            'queue_size': self.message_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'total_subscriptions': total_subscriptions,
            'messages_processed': self.messages_processed,
            'messages_dropped': self.messages_dropped,
            'processing_errors': self.processing_errors,
            'active_topics': list(self.subscriptions.keys()),
            'running': self.running,
            'async_enabled': self.enable_async
        }

    def _start_async_processing(self):
        """启动异步消息处理"""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(
            target=self._process_messages_async,
            name="ComponentBus-Processor",
            daemon=True
        )
        self.processing_thread.start()

        logger.info("组件总线异步处理已启动")

    def _process_messages_async(self, process_once: bool = False):
        """异步处理消息
        
        Args:
            process_once: 如果为True，只处理队列中当前的消息后退出（用于测试）
                         如果为False，持续循环处理消息直到running=False
        """
        try:
            # 如果process_once=True或enable_async=False，只处理当前队列
            if process_once or not self.enable_async:
                # 处理队列中的所有当前消息
                processed = 0
                max_iterations = self.message_queue.qsize() + 10  # 防止无限循环
                
                while processed < max_iterations:
                    try:
                        # 非阻塞获取消息
                        priority_weight, sequence, message = self.message_queue.get_nowait()
                        
                        # 处理消息
                        self._handle_message(message)
                        
                        # 标记任务完成
                        self.message_queue.task_done()
                        processed += 1
                        
                    except queue.Empty:
                        # 队列为空，退出
                        break
                    except Exception as e:
                        self.processing_errors += 1
                        logger.error(f"处理消息时发生错误: {e}")
                        break
                
                return  # 处理完毕，退出
            
            # 持续循环处理（用于后台线程）
            while self.running:
                try:
                    # 获取消息（带超时）
                    priority_weight, sequence, message = self.message_queue.get(timeout=1.0)

                    # 处理消息
                    self._handle_message(message)

                    # 标记任务完成
                    self.message_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    self.processing_errors += 1
                    logger.error(f"处理消息时发生错误: {e}")

        except Exception as e:
            logger.error(f"消息处理循环异常退出: {e}")
        finally:
            logger.info("组件总线异步处理已停止")

    def _handle_message(self, message: Message):
        """
        处理单个消息

        Args:
            message: 要处理的消息
        """
        try:
            # 查找匹配的订阅者
            matching_subscriptions = self._find_matching_subscriptions(message.topic)

            if not matching_subscriptions:
                logger.debug(f"没有找到消息 {message.topic} 的订阅者")
                return

            # 按优先级排序订阅者
            matching_subscriptions.sort(key=lambda s: s.priority, reverse=True)

            # 调用所有匹配的处理器
            for subscription in matching_subscriptions:
                if subscription.active:
                    try:
                        subscription.handler(message)
                    except Exception as e:
                        logger.error(f"消息处理器异常 ({subscription.component_name}): {e}")

            self.messages_processed += 1

        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            self.processing_errors += 1

    def _find_matching_subscriptions(self, topic: str) -> List[Subscription]:
        """
        查找匹配的订阅

        Args:
            topic: 消息主题

        Returns:
            List[Subscription]: 匹配的订阅列表
        """
        matching = []

        with self._subscription_lock:
            for topic_pattern, subscriptions in self.subscriptions.items():
                if self._topic_matches(topic, topic_pattern):
                    matching.extend(subscriptions)

        return matching

    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """
        检查主题是否匹配模式

        Args:
            topic: 消息主题
            pattern: 主题模式（支持通配符）

        Returns:
            bool: 是否匹配
        """
        # 简单的通配符匹配
        if pattern == topic:
            return True

        # 支持 * 通配符
        if '*' in pattern:
            import fnmatch
            return fnmatch.fnmatch(topic, pattern)

        return False

    def shutdown(self):
        """关闭组件总线"""
        logger.info("正在关闭组件总线...")

        self.running = False

        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        # 清空队列
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except:
                break

        logger.info("组件总线已关闭")


# 全局组件总线实例
global_component_bus = ComponentBus()


# 便捷函数
def publish_event(topic: str, payload: Dict[str, Any], sender: str = "unknown") -> bool:
    """
    发布事件消息

    Args:
        topic: 事件主题
        payload: 事件数据
        sender: 发送者

    Returns:
        bool: 是否成功发布
    """
    import uuid

    message = Message(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.EVENT,
        topic=topic,
        sender=sender,
        payload=payload
    )

    return global_component_bus.publish(message)


def send_notification(topic: str, payload: Dict[str, Any], sender: str = "unknown") -> bool:
    """
    发送通知消息

    Args:
        topic: 通知主题
        payload: 通知数据
        sender: 发送者

    Returns:
        bool: 是否成功发送
    """
    import uuid

    message = Message(
        message_id=str(uuid.uuid4()),
        message_type=MessageType.NOTIFICATION,
        topic=topic,
        sender=sender,
        payload=payload,
        priority=MessagePriority.HIGH
    )

    return global_component_bus.publish(message)
