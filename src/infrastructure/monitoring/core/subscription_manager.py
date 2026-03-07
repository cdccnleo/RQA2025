#!/usr/bin/env python3
"""
RQA2025 基础设施层订阅管理器

负责消息订阅关系的注册、管理和通知。
这是从ComponentBus中拆分出来的订阅管理组件。
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Set
import threading
from datetime import datetime
from collections import defaultdict

from .component_bus import Message, MessageType

logger = logging.getLogger(__name__)


class SubscriptionManager:
    """
    订阅管理器

    负责管理组件之间的消息订阅关系，支持动态订阅和取消订阅。
    """

    def __init__(self):
        """初始化订阅管理器"""
        self.subscriptions: Dict[str, Set[Callable]] = defaultdict(set)
        self.topic_subscriptions: Dict[str, Set[Callable]] = defaultdict(set)
        self.pattern_subscriptions: List[tuple] = []  # (pattern, handler)
        self._lock = threading.RLock()

        # 统计信息
        self.stats = {
            'total_subscriptions': 0,
            'active_subscribers': 0,
            'messages_delivered': 0,
            'delivery_failures': 0,
            'start_time': datetime.now()
        }

        logger.info("订阅管理器初始化完成")

    def subscribe(self, message_type: str, handler: Callable,
                  pattern: Optional[str] = None) -> bool:
        """
        订阅消息

        Args:
            message_type: 消息类型
            handler: 处理函数
            pattern: 消息模式匹配（可选）

        Returns:
            bool: 是否订阅成功
        """
        with self._lock:
            try:
                if pattern:
                    # 模式订阅
                    self.pattern_subscriptions.append((pattern, handler))
                    logger.debug(f"添加模式订阅: {message_type} -> {pattern}")
                else:
                    # 精确订阅
                    self.subscriptions[message_type].add(handler)
                    logger.debug(f"添加精确订阅: {message_type}")

                self.stats['total_subscriptions'] += 1
                self._update_active_subscribers()

                return True

            except Exception as e:
                logger.error(f"订阅失败: {e}")
                return False

    def unsubscribe(self, message_type: str, handler: Callable,
                   pattern: Optional[str] = None) -> bool:
        """
        取消订阅

        Args:
            message_type: 消息类型
            handler: 处理函数
            pattern: 消息模式匹配（可选）

        Returns:
            bool: 是否取消成功
        """
        with self._lock:
            try:
                if pattern:
                    # 移除模式订阅
                    self.pattern_subscriptions = [
                        (p, h) for p, h in self.pattern_subscriptions
                        if not (p == pattern and h == handler)
                    ]
                    logger.debug(f"移除模式订阅: {message_type} -> {pattern}")
                else:
                    # 移除精确订阅
                    self.subscriptions[message_type].discard(handler)
                    logger.debug(f"移除精确订阅: {message_type}")

                self.stats['total_subscriptions'] -= 1
                self._update_active_subscribers()

                return True

            except Exception as e:
                logger.error(f"取消订阅失败: {e}")
                return False

    def subscribe_topic(self, topic: str, handler: Callable) -> bool:
        """
        订阅主题

        Args:
            topic: 主题名称
            handler: 处理函数

        Returns:
            bool: 是否订阅成功
        """
        with self._lock:
            try:
                self.topic_subscriptions[topic].add(handler)
                self.stats['total_subscriptions'] += 1
                self._update_active_subscribers()

                logger.debug(f"订阅主题: {topic}")
                return True

            except Exception as e:
                logger.error(f"主题订阅失败: {e}")
                return False

    def unsubscribe_topic(self, topic: str, handler: Callable) -> bool:
        """
        取消主题订阅

        Args:
            topic: 主题名称
            handler: 处理函数

        Returns:
            bool: 是否取消成功
        """
        with self._lock:
            try:
                self.topic_subscriptions[topic].discard(handler)
                self.stats['total_subscriptions'] -= 1
                self._update_active_subscribers()

                logger.debug(f"取消主题订阅: {topic}")
                return True

            except Exception as e:
                logger.error(f"主题取消订阅失败: {e}")
                return False

    def publish(self, message: Message) -> Dict[str, Any]:
        """
        发布消息给订阅者

        Args:
            message: 消息对象

        Returns:
            Dict[str, Any]: 发布结果
        """
        with self._lock:
            results = {
                'total_subscribers': 0,
                'delivered': 0,
                'failed': 0,
                'errors': []
            }

            try:
                # 收集所有相关的处理器
                handlers = set()

                # 1. 精确类型匹配
                handlers.update(self.subscriptions.get(str(message.type), set()))

                # 2. 主题匹配
                if hasattr(message, 'topic') and message.topic:
                    handlers.update(self.topic_subscriptions.get(message.topic, set()))

                # 3. 模式匹配
                for pattern, handler in self.pattern_subscriptions:
                    if self._matches_pattern(str(message.type), pattern):
                        handlers.add(handler)

                results['total_subscribers'] = len(handlers)

                # 分发消息
                for handler in handlers:
                    try:
                        handler(message)
                        results['delivered'] += 1
                        self.stats['messages_delivered'] += 1

                    except Exception as e:
                        results['failed'] += 1
                        results['errors'].append(str(e))
                        self.stats['delivery_failures'] += 1
                        logger.error(f"消息分发失败: {e}")

                logger.debug(f"消息发布完成: {message.type}, 订阅者: {results['total_subscribers']}, 成功: {results['delivered']}")

            except Exception as e:
                logger.error(f"消息发布异常: {e}")
                results['errors'].append(str(e))

            return results

    def get_subscription_info(self, message_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取订阅信息

        Args:
            message_type: 消息类型，如果为None则返回所有

        Returns:
            Dict[str, Any]: 订阅信息
        """
        with self._lock:
            if message_type:
                # 特定类型的订阅信息
                exact_subs = len(self.subscriptions.get(message_type, set()))
                topic_subs = sum(1 for handlers in self.topic_subscriptions.values()
                               for h in handlers if h in self.subscriptions.get(message_type, set()))
                pattern_subs = sum(1 for p, h in self.pattern_subscriptions
                                 if self._matches_pattern(message_type, p))

                return {
                    'message_type': message_type,
                    'exact_subscriptions': exact_subs,
                    'topic_subscriptions': topic_subs,
                    'pattern_subscriptions': pattern_subs,
                    'total': exact_subs + topic_subs + pattern_subs
                }
            else:
                # 总体订阅信息
                all_types = set(self.subscriptions.keys()) | set(self.topic_subscriptions.keys())

                return {
                    'total_message_types': len(all_types),
                    'total_subscriptions': self.stats['total_subscriptions'],
                    'active_subscribers': self.stats['active_subscribers'],
                    'subscription_types': {
                        'exact': len(self.subscriptions),
                        'topic': len(self.topic_subscriptions),
                        'pattern': len(self.pattern_subscriptions)
                    }
                }

    def list_subscribers(self, message_type: str) -> List[str]:
        """
        列出指定消息类型的订阅者

        Args:
            message_type: 消息类型

        Returns:
            List[str]: 订阅者列表（函数名或描述）
        """
        with self._lock:
            subscribers = []

            # 精确订阅
            for handler in self.subscriptions.get(message_type, set()):
                subscribers.append(f"exact:{handler.__name__}")

            # 主题订阅（如果相关）
            for topic, handlers in self.topic_subscriptions.items():
                for handler in handlers:
                    subscribers.append(f"topic:{topic}:{handler.__name__}")

            # 模式订阅
            for pattern, handler in self.pattern_subscriptions:
                if self._matches_pattern(message_type, pattern):
                    subscribers.append(f"pattern:{pattern}:{handler.__name__}")

            return subscribers

    def clear_subscriptions(self, message_type: Optional[str] = None):
        """
        清除订阅

        Args:
            message_type: 消息类型，如果为None则清除所有
        """
        with self._lock:
            if message_type:
                # 清除特定类型的订阅
                self.subscriptions.pop(message_type, None)

                # 清除相关的模式订阅
                self.pattern_subscriptions = [
                    (p, h) for p, h in self.pattern_subscriptions
                    if not self._matches_pattern(message_type, p)
                ]

                logger.info(f"清除订阅: {message_type}")
            else:
                # 清除所有订阅
                self.subscriptions.clear()
                self.topic_subscriptions.clear()
                self.pattern_subscriptions.clear()
                self.stats['total_subscriptions'] = 0
                self._update_active_subscribers()

                logger.info("清除所有订阅")

    def get_subscription_stats(self) -> Dict[str, Any]:
        """
        获取订阅统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        current_time = datetime.now()
        uptime = (current_time - self.stats['start_time']).total_seconds()

        info = self.get_subscription_info()

        return {
            'total_subscriptions': self.stats['total_subscriptions'],
            'active_subscribers': self.stats['active_subscribers'],
            'messages_delivered': self.stats['messages_delivered'],
            'delivery_failures': self.stats['delivery_failures'],
            'delivery_success_rate': (self.stats['messages_delivered'] /
                                    max(self.stats['messages_delivered'] + self.stats['delivery_failures'], 1)),
            'subscription_types': info['subscription_types'],
            'uptime_seconds': uptime,
            'messages_per_second': self.stats['messages_delivered'] / max(uptime, 1)
        }

    def _update_active_subscribers(self):
        """更新活跃订阅者数量"""
        total = 0
        for handlers in self.subscriptions.values():
            total += len(handlers)
        for handlers in self.topic_subscriptions.values():
            total += len(handlers)
        total += len(self.pattern_subscriptions)

        self.stats['active_subscribers'] = total

    def _matches_pattern(self, message_type: str, pattern: str) -> bool:
        """
        检查消息类型是否匹配模式

        Args:
            message_type: 消息类型
            pattern: 匹配模式

        Returns:
            bool: 是否匹配
        """
        # 简单通配符匹配
        if '*' in pattern:
            import fnmatch
            return fnmatch.fnmatch(message_type, pattern)
        else:
            return message_type == pattern

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取订阅管理器的健康状态

        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            stats = self.get_subscription_stats()

            issues = []

            # 检查订阅数量异常
            if stats['total_subscriptions'] > 1000:
                issues.append(f"订阅数量过多: {stats['total_subscriptions']}")

            # 检查失败率
            if stats['delivery_success_rate'] < 0.95:
                issues.append(".1%")

            # 检查活跃订阅者
            if stats['active_subscribers'] == 0:
                issues.append("没有活跃的订阅者")

            # 检查内存使用（简单的启发式）
            total_handlers = sum(len(handlers) for handlers in self.subscriptions.values()) + \
                           sum(len(handlers) for handlers in self.topic_subscriptions.values()) + \
                           len(self.pattern_subscriptions)

            if total_handlers > 10000:  # 假设的合理上限
                issues.append(f"处理器数量过多: {total_handlers}")

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


# 全局订阅管理器实例
global_subscription_manager = SubscriptionManager()
