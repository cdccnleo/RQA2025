#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket发布器

功能：
- WebSocket连接管理
- 实时数据推送
- 订阅管理
- 广播和单播支持

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
from dataclasses import asdict
import threading

try:
    from flask_socketio import SocketIO, emit
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False
    logger = logging.getLogger(__name__)
    logger.warning("flask-socketio 未安装，WebSocket功能将不可用")

logger = logging.getLogger(__name__)


class WebSocketPublisher:
    """
    WebSocket发布器
    
    职责：
    1. WebSocket连接管理
    2. 实时数据推送
    3. 订阅管理
    4. 广播和单播支持
    """
    
    def __init__(self, socketio: Optional[Any] = None):
        """
        初始化WebSocket发布器
        
        Args:
            socketio: Flask-SocketIO实例
        """
        self.socketio = socketio
        
        # 客户端连接管理
        self._clients: Dict[str, Dict[str, Any]] = {}  # sid -> client_info
        self._symbol_subscribers: Dict[str, Set[str]] = {}  # symbol -> set(sid)
        
        # 消息队列
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._is_running = False
        self._publish_task: Optional[asyncio.Task] = None
        
        # 统计信息
        self._stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_dropped': 0
        }
        
        self._lock = threading.RLock()
        
        if HAS_SOCKETIO:
            logger.info("WebSocket发布器初始化完成")
        else:
            logger.warning("WebSocket发布器初始化完成（功能受限）")
    
    def start(self):
        """启动发布器"""
        if self._is_running:
            return
        
        self._is_running = True
        
        # 启动发布任务
        if asyncio.get_event_loop().is_running():
            self._publish_task = asyncio.create_task(self._publish_loop())
        
        logger.info("WebSocket发布器已启动")
    
    def stop(self):
        """停止发布器"""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # 取消发布任务
        if self._publish_task:
            self._publish_task.cancel()
        
        logger.info("WebSocket发布器已停止")
    
    def on_connect(self, sid: str, environ: dict):
        """
        客户端连接回调
        
        Args:
            sid: 会话ID
            environ: 环境信息
        """
        with self._lock:
            self._clients[sid] = {
                'sid': sid,
                'connected_at': datetime.now(),
                'subscribed_symbols': set(),
                'environ': environ
            }
            self._stats['total_connections'] += 1
            self._stats['active_connections'] += 1
        
        logger.info(f"客户端连接: {sid}")
    
    def on_disconnect(self, sid: str):
        """
        客户端断开回调
        
        Args:
            sid: 会话ID
        """
        with self._lock:
            if sid in self._clients:
                client = self._clients[sid]
                
                # 从所有股票订阅中移除
                for symbol in client['subscribed_symbols']:
                    if symbol in self._symbol_subscribers:
                        self._symbol_subscribers[symbol].discard(sid)
                
                del self._clients[sid]
                self._stats['active_connections'] -= 1
        
        logger.info(f"客户端断开: {sid}")
    
    def subscribe_symbol(self, sid: str, symbol: str):
        """
        订阅股票
        
        Args:
            sid: 会话ID
            symbol: 股票代码
        """
        with self._lock:
            if sid not in self._clients:
                logger.warning(f"客户端不存在: {sid}")
                return
            
            # 添加到客户端的订阅列表
            self._clients[sid]['subscribed_symbols'].add(symbol)
            
            # 添加到股票的订阅者列表
            if symbol not in self._symbol_subscribers:
                self._symbol_subscribers[symbol] = set()
            self._symbol_subscribers[symbol].add(sid)
        
        logger.info(f"客户端 {sid} 订阅股票: {symbol}")
    
    def unsubscribe_symbol(self, sid: str, symbol: str):
        """
        取消订阅股票
        
        Args:
            sid: 会话ID
            symbol: 股票代码
        """
        with self._lock:
            if sid in self._clients:
                self._clients[sid]['subscribed_symbols'].discard(symbol)
            
            if symbol in self._symbol_subscribers:
                self._symbol_subscribers[symbol].discard(sid)
        
        logger.info(f"客户端 {sid} 取消订阅股票: {symbol}")
    
    def publish_to_symbol(self, symbol: str, data: Any):
        """
        向订阅了特定股票的所有客户端推送数据
        
        Args:
            symbol: 股票代码
            data: 数据
        """
        if not HAS_SOCKETIO or not self.socketio:
            return
        
        try:
            # 序列化数据
            if hasattr(data, 'to_dict'):
                message = data.to_dict()
            elif hasattr(data, '__dict__'):
                message = data.__dict__
            else:
                message = data
            
            # 添加时间戳
            if isinstance(message, dict):
                message['server_timestamp'] = datetime.now().isoformat()
            
            # 推送到消息队列
            asyncio.create_task(self._message_queue.put({
                'type': 'symbol',
                'symbol': symbol,
                'data': message
            }))
            
        except Exception as e:
            logger.error(f"发布数据失败 {symbol}: {e}")
    
    def broadcast(self, data: Any, event: str = 'message'):
        """
        广播数据到所有客户端
        
        Args:
            data: 数据
            event: 事件名称
        """
        if not HAS_SOCKETIO or not self.socketio:
            return
        
        try:
            # 序列化数据
            if hasattr(data, 'to_dict'):
                message = data.to_dict()
            elif hasattr(data, '__dict__'):
                message = data.__dict__
            else:
                message = data
            
            # 添加时间戳
            if isinstance(message, dict):
                message['server_timestamp'] = datetime.now().isoformat()
            
            # 推送到消息队列
            asyncio.create_task(self._message_queue.put({
                'type': 'broadcast',
                'event': event,
                'data': message
            }))
            
        except Exception as e:
            logger.error(f"广播数据失败: {e}")
    
    def send_to_client(self, sid: str, data: Any, event: str = 'message'):
        """
        向特定客户端发送数据
        
        Args:
            sid: 会话ID
            data: 数据
            event: 事件名称
        """
        if not HAS_SOCKETIO or not self.socketio:
            return
        
        try:
            # 序列化数据
            if hasattr(data, 'to_dict'):
                message = data.to_dict()
            elif hasattr(data, '__dict__'):
                message = data.__dict__
            else:
                message = data
            
            # 推送到消息队列
            asyncio.create_task(self._message_queue.put({
                'type': 'unicast',
                'sid': sid,
                'event': event,
                'data': message
            }))
            
        except Exception as e:
            logger.error(f"发送数据失败 {sid}: {e}")
    
    async def _publish_loop(self):
        """发布循环"""
        logger.info("启动发布循环")
        
        while self._is_running:
            try:
                # 从队列获取消息
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                # 处理消息
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"发布循环错误: {e}")
    
    async def _process_message(self, message: dict):
        """
        处理消息
        
        Args:
            message: 消息字典
        """
        try:
            msg_type = message.get('type')
            
            if msg_type == 'symbol':
                # 向特定股票的订阅者推送
                symbol = message.get('symbol')
                data = message.get('data')
                
                with self._lock:
                    subscribers = self._symbol_subscribers.get(symbol, set()).copy()
                
                for sid in subscribers:
                    if sid in self._clients:
                        self._emit_to_client(sid, 'market_data', data)
                
                self._stats['messages_sent'] += len(subscribers)
                
            elif msg_type == 'broadcast':
                # 广播
                event = message.get('event', 'message')
                data = message.get('data')
                
                self._emit_broadcast(event, data)
                
                with self._lock:
                    self._stats['messages_sent'] += self._stats['active_connections']
                
            elif msg_type == 'unicast':
                # 单播
                sid = message.get('sid')
                event = message.get('event', 'message')
                data = message.get('data')
                
                self._emit_to_client(sid, event, data)
                self._stats['messages_sent'] += 1
                
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            self._stats['messages_dropped'] += 1
    
    def _emit_to_client(self, sid: str, event: str, data: dict):
        """
        向客户端发送事件
        
        Args:
            sid: 会话ID
            event: 事件名称
            data: 数据
        """
        if not HAS_SOCKETIO or not self.socketio:
            return
        
        try:
            # 使用socketio.emit发送到特定客户端
            self.socketio.emit(event, data, room=sid)
        except Exception as e:
            logger.error(f"发送事件失败 {sid}: {e}")
    
    def _emit_broadcast(self, event: str, data: dict):
        """
        广播事件
        
        Args:
            event: 事件名称
            data: 数据
        """
        if not HAS_SOCKETIO or not self.socketio:
            return
        
        try:
            # 使用socketio.emit广播
            self.socketio.emit(event, data, broadcast=True)
        except Exception as e:
            logger.error(f"广播事件失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                **self._stats,
                'symbol_subscriptions': len(self._symbol_subscribers),
                'avg_subscriptions_per_symbol': (
                    sum(len(s) for s in self._symbol_subscribers.values()) /
                    len(self._symbol_subscribers)
                    if self._symbol_subscribers else 0
                )
            }
    
    def get_client_info(self, sid: str) -> Optional[Dict[str, Any]]:
        """
        获取客户端信息
        
        Args:
            sid: 会话ID
            
        Returns:
            客户端信息
        """
        with self._lock:
            client = self._clients.get(sid)
            if client:
                return {
                    'sid': client['sid'],
                    'connected_at': client['connected_at'].isoformat(),
                    'subscribed_symbols': list(client['subscribed_symbols'])
                }
            return None
    
    def get_all_clients(self) -> List[Dict[str, Any]]:
        """
        获取所有客户端信息
        
        Returns:
            客户端信息列表
        """
        with self._lock:
            return [
                {
                    'sid': client['sid'],
                    'connected_at': client['connected_at'].isoformat(),
                    'subscribed_symbols': list(client['subscribed_symbols'])
                }
                for client in self._clients.values()
            ]


# 单例实例
_publisher: Optional[WebSocketPublisher] = None


def get_websocket_publisher(socketio: Optional[Any] = None) -> WebSocketPublisher:
    """
    获取WebSocket发布器单例
    
    Args:
        socketio: Flask-SocketIO实例
        
    Returns:
        WebSocketPublisher实例
    """
    global _publisher
    if _publisher is None:
        _publisher = WebSocketPublisher(socketio=socketio)
    return _publisher
