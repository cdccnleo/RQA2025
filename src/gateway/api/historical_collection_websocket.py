#!/usr/bin/env python3
"""
历史数据采集WebSocket处理器

提供实时监控数据的WebSocket通信支持
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from src.core.monitoring.historical_data_monitor import get_historical_data_monitor
from src.core.orchestration.historical_data_scheduler import get_historical_data_scheduler

logger = logging.getLogger(__name__)


class HistoricalCollectionWebSocketManager:
    """
    历史数据采集WebSocket管理器

    管理WebSocket连接，处理订阅和广播实时数据更新
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, Dict[str, Any]] = {}
        self.monitor = get_historical_data_monitor()
        self.scheduler = get_historical_data_scheduler()
        self.broadcast_task: Optional[asyncio.Task] = None
        self.is_broadcasting = False

    async def connect(self, websocket: WebSocket):
        """建立WebSocket连接"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = {
            'topics': [],
            'connected_at': asyncio.get_event_loop().time()
        }

        logger.info(f"WebSocket客户端连接: {len(self.active_connections)} 个活跃连接")

        # 启动广播任务（如果还没有启动）
        if not self.is_broadcasting:
            self.is_broadcasting = True
            self.broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if websocket in self.subscriptions:
            del self.subscriptions[websocket]

        logger.info(f"WebSocket客户端断开: {len(self.active_connections)} 个活跃连接")

    async def handle_message(self, websocket: WebSocket, message: str):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)

            if data.get('type') == 'subscribe':
                await self._handle_subscribe(websocket, data)
            elif data.get('type') == 'unsubscribe':
                await self._handle_unsubscribe(websocket, data)
            elif data.get('type') == 'ping':
                await self._handle_ping(websocket)
            else:
                logger.warning(f"未知消息类型: {data.get('type')}")

        except json.JSONDecodeError:
            logger.error(f"无效的JSON消息: {message}")
        except Exception as e:
            logger.error(f"处理WebSocket消息失败: {e}")

    async def _handle_subscribe(self, websocket: WebSocket, data: Dict[str, Any]):
        """处理订阅请求"""
        topics = data.get('topics', [])
        if websocket in self.subscriptions:
            self.subscriptions[websocket]['topics'] = topics

        logger.info(f"客户端订阅主题: {topics}")

        # 发送确认消息
        await websocket.send_json({
            'type': 'subscribed',
            'topics': topics,
            'timestamp': asyncio.get_event_loop().time()
        })

    async def _handle_unsubscribe(self, websocket: WebSocket, data: Dict[str, Any]):
        """处理取消订阅请求"""
        topics = data.get('topics', [])
        if websocket in self.subscriptions:
            current_topics = self.subscriptions[websocket]['topics']
            for topic in topics:
                if topic in current_topics:
                    current_topics.remove(topic)

        logger.info(f"客户端取消订阅主题: {topics}")

    async def _handle_ping(self, websocket: WebSocket):
        """处理ping消息"""
        await websocket.send_json({
            'type': 'pong',
            'timestamp': asyncio.get_event_loop().time()
        })

    async def _broadcast_loop(self):
        """广播循环"""
        logger.info("WebSocket广播循环启动")

        while self.is_broadcasting and self.active_connections:
            try:
                # 收集要广播的数据
                updates = await self._collect_updates()

                # 广播给所有订阅的客户端
                for websocket in self.active_connections:
                    try:
                        subscription = self.subscriptions.get(websocket, {})
                        topics = subscription.get('topics', [])

                        # 根据订阅的主题过滤数据
                        filtered_updates = self._filter_updates_by_topics(updates, topics)

                        for update in filtered_updates:
                            await websocket.send_json(update)

                    except Exception as e:
                        logger.error(f"广播到客户端失败: {e}")
                        # 移除故障连接
                        if websocket in self.active_connections:
                            await self.disconnect(websocket)

                # 等待下一轮广播
                await asyncio.sleep(2)  # 每2秒广播一次

            except Exception as e:
                logger.error(f"广播循环异常: {e}")
                await asyncio.sleep(5)

        logger.info("WebSocket广播循环结束")
        self.is_broadcasting = False

    async def _collect_updates(self) -> List[Dict[str, Any]]:
        """收集需要广播的更新"""
        updates = []

        try:
            # 监控状态更新
            monitoring_data = self.monitor.get_monitoring_data()
            updates.append({
                'type': 'status_update',
                'topic': 'historical_collection_status',
                'data': monitoring_data,
                'timestamp': asyncio.get_event_loop().time()
            })

            # 调度器状态更新
            scheduler_status = self.scheduler.get_scheduler_status()
            updates.append({
                'type': 'status_update',
                'topic': 'scheduler_status',
                'data': scheduler_status,
                'timestamp': asyncio.get_event_loop().time()
            })

            # 检查是否有新的任务完成或告警
            # 这里可以根据需要添加更多实时更新类型

        except Exception as e:
            logger.error(f"收集更新数据失败: {e}")

        return updates

    def _filter_updates_by_topics(self, updates: List[Dict[str, Any]], topics: List[str]) -> List[Dict[str, Any]]:
        """根据订阅主题过滤更新"""
        if not topics or 'all' in topics:
            return updates

        filtered = []
        for update in updates:
            topic = update.get('topic')
            if topic in topics:
                filtered.append(update)

        return filtered

    async def broadcast_task_update(self, task_id: str, progress: float,
                                  records_collected: int = 0, status: str = 'running'):
        """广播任务更新"""
        update = {
            'type': 'task_progress',
            'task_id': task_id,
            'progress': progress,
            'records_collected': records_collected,
            'status': status,
            'timestamp': asyncio.get_event_loop().time()
        }

        await self._broadcast_to_all(update)

    async def broadcast_task_completed(self, task_id: str, success: bool, records_collected: int = 0):
        """广播任务完成"""
        update = {
            'type': 'task_completed',
            'task_id': task_id,
            'success': success,
            'records_collected': records_collected,
            'timestamp': asyncio.get_event_loop().time()
        }

        await self._broadcast_to_all(update)

    async def broadcast_alert(self, alert: Dict[str, Any]):
        """广播告警"""
        update = {
            'type': 'alert',
            'alert': alert,
            'timestamp': asyncio.get_event_loop().time()
        }

        await self._broadcast_to_all(update)

    async def _broadcast_to_all(self, message: Dict[str, Any]):
        """广播消息给所有连接的客户端"""
        disconnected = []

        for websocket in self.active_connections:
            try:
                subscription = self.subscriptions.get(websocket, {})
                topics = subscription.get('topics', [])

                # 检查是否订阅了相关主题
                if self._should_receive_message(message, topics):
                    await websocket.send_json(message)

            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                disconnected.append(websocket)

        # 清理断开的连接
        for websocket in disconnected:
            if websocket in self.active_connections:
                await self.disconnect(websocket)

    def _should_receive_message(self, message: Dict[str, Any], topics: List[str]) -> bool:
        """检查客户端是否应该接收此消息"""
        if not topics or 'all' in topics:
            return True

        message_type = message.get('type')
        message_topic = message.get('topic')

        # 根据消息类型判断
        if message_type == 'task_progress' and 'task_progress' in topics:
            return True
        elif message_type == 'task_completed' and 'task_completed' in topics:
            return True
        elif message_type == 'alert' and 'alerts' in topics:
            return True
        elif message_type == 'status_update' and message_topic in topics:
            return True

        return False

    def get_connection_count(self) -> int:
        """获取活跃连接数"""
        return len(self.active_connections)

    def get_subscription_stats(self) -> Dict[str, Any]:
        """获取订阅统计"""
        topic_counts = {}
        for subscription in self.subscriptions.values():
            for topic in subscription.get('topics', []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

        return {
            'total_connections': len(self.active_connections),
            'topic_subscriptions': topic_counts
        }


# 全局WebSocket管理器实例
_websocket_manager: Optional[HistoricalCollectionWebSocketManager] = None


def get_historical_collection_websocket_manager() -> HistoricalCollectionWebSocketManager:
    """获取WebSocket管理器实例"""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = HistoricalCollectionWebSocketManager()
    return _websocket_manager


# WebSocket端点函数
async def handle_historical_collection_websocket(websocket: WebSocket):
    """
    处理历史数据采集WebSocket连接

    Args:
        websocket: WebSocket连接对象
    """
    manager = get_historical_collection_websocket_manager()

    await manager.connect(websocket)

    try:
        while True:
            message = await websocket.receive_text()
            await manager.handle_message(websocket, message)

    except WebSocketDisconnect:
        logger.info("WebSocket客户端主动断开")
    except Exception as e:
        logger.error(f"WebSocket处理异常: {e}")
    finally:
        await manager.disconnect(websocket)