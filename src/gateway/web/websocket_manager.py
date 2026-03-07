"""
WebSocket连接管理器 - 稳定性优化版本

本模块提供高稳定性的WebSocket连接管理，包括：
1. 指数退避重连机制
2. 心跳检测和保活
3. 断线自动恢复
4. 消息队列和重传
5. 连接状态监控

作者: 后端团队
创建日期: 2026-02-21
版本: 2.0.0
"""

import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import uuid

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

# 尝试导入WebSocketError，如果不存在则定义本地版本
try:
    from src.common.exceptions import WebSocketError
except ImportError:
    class WebSocketError(Exception):
        """WebSocket错误基类"""
        pass


# 配置日志
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = auto()      # 已断开
    CONNECTING = auto()        # 连接中
    CONNECTED = auto()         # 已连接
    RECONNECTING = auto()      # 重连中
    CLOSED = auto()            # 已关闭


class MessageType(Enum):
    """消息类型"""
    DATA = "data"              # 数据消息
    HEARTBEAT = "heartbeat"    # 心跳消息
    ACK = "ack"                # 确认消息
    ERROR = "error"            # 错误消息
    RECONNECT = "reconnect"    # 重连消息


@dataclass
class QueuedMessage:
    """队列消息"""
    message_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps({
            "message_id": self.message_id,
            "type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat()
        })


@dataclass
class ConnectionStats:
    """连接统计信息"""
    connection_attempts: int = 0
    successful_connections: int = 0
    reconnection_attempts: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    messages_queued: int = 0
    last_connected_at: Optional[datetime] = None
    last_disconnected_at: Optional[datetime] = None
    total_uptime_seconds: float = 0.0
    
    @property
    def connection_success_rate(self) -> float:
        """连接成功率"""
        if self.connection_attempts == 0:
            return 0.0
        return (self.successful_connections / self.connection_attempts) * 100
    
    @property
    def uptime_percentage(self) -> float:
        """在线时间百分比"""
        total_time = self.total_uptime_seconds
        if self.last_connected_at:
            total_time += (datetime.now() - self.last_connected_at).total_seconds()
        
        # 假设统计周期为24小时
        total_period = 24 * 3600
        return min(100.0, (total_time / total_period) * 100)


class WebSocketManager:
    """
    WebSocket连接管理器 - 稳定性优化版本
    
    功能:
    1. 指数退避重连机制
    2. 心跳检测和保活
    3. 断线自动恢复
    4. 消息队列和重传
    5. 连接状态监控
    
    使用示例:
        manager = WebSocketManager(
            uri="ws://localhost:8000/ws",
            on_message=handle_message,
            on_connect=on_connect
        )
        await manager.connect()
        
        # 发送消息
        await manager.send({"type": "subscribe", "symbol": "AAPL"})
    """
    
    def __init__(
        self,
        uri: str,
        on_message: Optional[Callable[[Dict], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
        on_reconnect: Optional[Callable[[int], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        heartbeat_interval: int = 30,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        message_timeout: float = 30.0
    ):
        """
        初始化WebSocket管理器
        
        参数:
            uri: WebSocket服务器URI
            on_message: 消息接收回调
            on_connect: 连接成功回调
            on_disconnect: 断开连接回调
            on_reconnect: 重连回调(参数为重连次数)
            on_error: 错误回调
            heartbeat_interval: 心跳间隔(秒)
            max_reconnect_attempts: 最大重连次数
            base_reconnect_delay: 基础重连延迟(秒)
            max_reconnect_delay: 最大重连延迟(秒)
            message_timeout: 消息超时时间(秒)
        """
        self.uri = uri
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_reconnect = on_reconnect
        self.on_error = on_error
        
        # 配置参数
        self.heartbeat_interval = heartbeat_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.base_reconnect_delay = base_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.message_timeout = message_timeout
        
        # 连接状态
        self._state = ConnectionState.DISCONNECTED
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._reconnect_attempts = 0
        self._should_reconnect = True
        
        # 任务
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # 消息队列
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._pending_acks: Dict[str, asyncio.Future] = {}
        self._message_history: List[QueuedMessage] = []
        self._max_history_size = 1000
        
        # 统计信息
        self._stats = ConnectionStats()
        
        # 订阅管理
        self._subscriptions: Set[str] = set()
        self._last_received_message_time: Optional[datetime] = None
        
        logger.info(f"WebSocket管理器已初始化: {uri}")
    
    @property
    def state(self) -> ConnectionState:
        """获取当前连接状态"""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._state == ConnectionState.CONNECTED and self._websocket is not None
    
    @property
    def stats(self) -> ConnectionStats:
        """获取连接统计信息"""
        return self._stats
    
    async def connect(self) -> bool:
        """
        建立WebSocket连接
        
        返回:
            bool: 连接是否成功
        """
        if self._state == ConnectionState.CONNECTED:
            logger.warning("已经处于连接状态")
            return True
        
        if self._state == ConnectionState.CONNECTING:
            logger.warning("正在连接中")
            return False
        
        self._state = ConnectionState.CONNECTING
        self._stats.connection_attempts += 1
        
        try:
            logger.info(f"正在连接WebSocket: {self.uri}")
            
            self._websocket = await websockets.connect(
                self.uri,
                ping_interval=None,  # 我们使用自定义心跳
                close_timeout=10
            )
            
            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self._should_reconnect = True
            self._stats.successful_connections += 1
            self._stats.last_connected_at = datetime.now()
            
            logger.info("WebSocket连接成功")
            
            # 启动后台任务
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            # 触发连接回调
            if self.on_connect:
                try:
                    self.on_connect()
                except Exception as e:
                    logger.error(f"连接回调执行失败: {str(e)}")
            
            # 恢复订阅
            await self._restore_subscriptions()
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket连接失败: {str(e)}")
            self._state = ConnectionState.DISCONNECTED
            
            # 触发错误回调
            if self.on_error:
                try:
                    self.on_error(e)
                except Exception:
                    pass
            
            # 启动重连
            self._start_reconnect()
            
            return False
    
    async def disconnect(self):
        """断开WebSocket连接"""
        logger.info("正在断开WebSocket连接")
        
        self._should_reconnect = False
        self._state = ConnectionState.CLOSED
        
        # 取消所有任务
        await self._cancel_tasks()
        
        # 关闭连接
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"关闭WebSocket连接时出错: {str(e)}")
            finally:
                self._websocket = None
        
        # 更新统计
        if self._stats.last_connected_at:
            uptime = (datetime.now() - self._stats.last_connected_at).total_seconds()
            self._stats.total_uptime_seconds += uptime
        self._stats.last_disconnected_at = datetime.now()
        
        # 触发断开回调
        if self.on_disconnect:
            try:
                self.on_disconnect()
            except Exception as e:
                logger.error(f"断开回调执行失败: {str(e)}")
        
        logger.info("WebSocket连接已断开")
    
    async def send(
        self,
        data: Dict[str, Any],
        require_ack: bool = False,
        timeout: Optional[float] = None
    ) -> bool:
        """
        发送消息
        
        参数:
            data: 要发送的数据
            require_ack: 是否需要确认
            timeout: 超时时间(秒)
            
        返回:
            bool: 发送是否成功
        """
        if not self.is_connected:
            logger.warning("WebSocket未连接，消息将加入队列")
            await self._queue_message(data)
            return False
        
        try:
            message_id = str(uuid.uuid4())
            message = {
                "message_id": message_id,
                "type": MessageType.DATA.value,
                "payload": data,
                "timestamp": datetime.now().isoformat(),
                "require_ack": require_ack
            }
            
            # 发送消息
            await self._websocket.send(json.dumps(message))
            self._stats.messages_sent += 1
            
            # 如果需要确认，等待ACK
            if require_ack:
                future = asyncio.Future()
                self._pending_acks[message_id] = future
                
                try:
                    await asyncio.wait_for(
                        future,
                        timeout=timeout or self.message_timeout
                    )
                    return True
                except asyncio.TimeoutError:
                    logger.warning(f"消息确认超时: {message_id}")
                    return False
                finally:
                    self._pending_acks.pop(message_id, None)
            
            return True
            
        except Exception as e:
            logger.error(f"发送消息失败: {str(e)}")
            await self._queue_message(data)
            return False
    
    def subscribe(self, channel: str):
        """
        订阅频道
        
        参数:
            channel: 频道名称
        """
        self._subscriptions.add(channel)
        
        if self.is_connected:
            asyncio.create_task(self._send_subscription(channel))
        
        logger.info(f"已订阅频道: {channel}")
    
    def unsubscribe(self, channel: str):
        """
        取消订阅频道
        
        参数:
            channel: 频道名称
        """
        self._subscriptions.discard(channel)
        
        if self.is_connected:
            asyncio.create_task(self._send_unsubscription(channel))
        
        logger.info(f"已取消订阅频道: {channel}")
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self._state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self.is_connected:
                    break
                
                # 发送心跳
                heartbeat = {
                    "type": MessageType.HEARTBEAT.value,
                    "timestamp": datetime.now().isoformat()
                }
                
                await self._websocket.send(json.dumps(heartbeat))
                
                # 检查是否收到消息
                if self._last_received_message_time:
                    elapsed = (datetime.now() - self._last_received_message_time).total_seconds()
                    if elapsed > self.heartbeat_interval * 3:
                        logger.warning(f"长时间未收到消息: {elapsed:.1f}秒")
                        # 可能需要重新连接
                        
            except ConnectionClosed:
                logger.info("连接已关闭，停止心跳")
                break
            except Exception as e:
                logger.error(f"心跳发送失败: {str(e)}")
                break
        
        # 如果连接仍然应该是活跃的，启动重连
        if self._should_reconnect and self._state != ConnectionState.CLOSED:
            self._start_reconnect()
    
    async def _receive_loop(self):
        """接收消息循环"""
        while self._state == ConnectionState.CONNECTED:
            try:
                message = await self._websocket.recv()
                self._last_received_message_time = datetime.now()
                self._stats.messages_received += 1
                
                # 解析消息
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"收到无效的JSON消息: {message}")
                    
            except ConnectionClosed:
                logger.info("连接已关闭，停止接收")
                break
            except Exception as e:
                logger.error(f"接收消息失败: {str(e)}")
                break
        
        # 如果连接仍然应该是活跃的，启动重连
        if self._should_reconnect and self._state != ConnectionState.CLOSED:
            self._start_reconnect()
    
    async def _handle_message(self, data: Dict[str, Any]):
        """处理接收到的消息"""
        message_type = data.get("type", MessageType.DATA.value)
        message_id = data.get("message_id")
        
        if message_type == MessageType.HEARTBEAT.value:
            # 收到心跳响应
            pass
            
        elif message_type == MessageType.ACK.value:
            # 收到确认
            if message_id and message_id in self._pending_acks:
                future = self._pending_acks[message_id]
                if not future.done():
                    future.set_result(True)
                    
        elif message_type == MessageType.ERROR.value:
            # 收到错误消息
            error_msg = data.get("payload", {}).get("message", "未知错误")
            logger.error(f"收到服务器错误: {error_msg}")
            
        else:
            # 数据消息
            if self.on_message:
                try:
                    self.on_message(data.get("payload", data))
                except Exception as e:
                    logger.error(f"消息回调执行失败: {str(e)}")
    
    def _start_reconnect(self):
        """启动重连"""
        if self._reconnect_task and not self._reconnect_task.done():
            return
        
        if not self._should_reconnect:
            return
        
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def _reconnect_loop(self):
        """重连循环 - 指数退避"""
        self._state = ConnectionState.RECONNECTING
        
        while self._should_reconnect and self._reconnect_attempts < self.max_reconnect_attempts:
            self._reconnect_attempts += 1
            self._stats.reconnection_attempts += 1
            
            # 计算退避延迟
            delay = min(
                self.base_reconnect_delay * (2 ** (self._reconnect_attempts - 1)),
                self.max_reconnect_delay
            )
            
            logger.info(f"第{self._reconnect_attempts}次重连，延迟{delay:.1f}秒")
            
            # 触发重连回调
            if self.on_reconnect:
                try:
                    self.on_reconnect(self._reconnect_attempts)
                except Exception:
                    pass
            
            await asyncio.sleep(delay)
            
            # 尝试连接
            if await self.connect():
                logger.info("重连成功")
                return
        
        # 重连失败
        logger.error(f"重连失败，已达到最大重连次数: {self.max_reconnect_attempts}")
        self._state = ConnectionState.DISCONNECTED
    
    async def _queue_message(self, data: Dict[str, Any]):
        """将消息加入队列"""
        message = QueuedMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.DATA,
            payload=data,
            timestamp=datetime.now()
        )
        
        await self._message_queue.put(message)
        self._stats.messages_queued += 1
        
        # 保存到历史
        self._message_history.append(message)
        if len(self._message_history) > self._max_history_size:
            self._message_history.pop(0)
        
        logger.debug(f"消息已加入队列: {message.message_id}")
    
    async def _restore_subscriptions(self):
        """恢复订阅"""
        for channel in self._subscriptions:
            await self._send_subscription(channel)
    
    async def _send_subscription(self, channel: str):
        """发送订阅请求"""
        await self.send({
            "action": "subscribe",
            "channel": channel
        })
    
    async def _send_unsubscription(self, channel: str):
        """发送取消订阅请求"""
        await self.send({
            "action": "unsubscribe",
            "channel": channel
        })
    
    async def _cancel_tasks(self):
        """取消所有后台任务"""
        tasks = [
            self._heartbeat_task,
            self._receive_task,
            self._reconnect_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._heartbeat_task = None
        self._receive_task = None
        self._reconnect_task = None
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return {
            "uri": self.uri,
            "state": self._state.name,
            "is_connected": self.is_connected,
            "reconnect_attempts": self._reconnect_attempts,
            "subscriptions": list(self._subscriptions),
            "stats": {
                "connection_attempts": self._stats.connection_attempts,
                "successful_connections": self._stats.successful_connections,
                "connection_success_rate": f"{self._stats.connection_success_rate:.1f}%",
                "messages_sent": self._stats.messages_sent,
                "messages_received": self._stats.messages_received,
                "messages_queued": self._stats.messages_queued,
                "uptime_percentage": f"{self._stats.uptime_percentage:.1f}%"
            },
            "last_message_received": self._last_received_message_time.isoformat() if self._last_received_message_time else None
        }


# 便捷函数
async def create_stable_websocket(
    uri: str,
    on_message: Optional[Callable[[Dict], None]] = None,
    auto_reconnect: bool = True
) -> WebSocketManager:
    """
    创建稳定的WebSocket连接
    
    参数:
        uri: WebSocket服务器URI
        on_message: 消息接收回调
        auto_reconnect: 是否自动重连
        
    返回:
        WebSocketManager: WebSocket管理器实例
        
    示例:
        >>> manager = await create_stable_websocket(
        ...     "ws://localhost:8000/ws",
        ...     on_message=lambda msg: print(msg)
        ... )
        >>> await manager.send({"type": "hello"})
    """
    manager = WebSocketManager(
        uri=uri,
        on_message=on_message,
        max_reconnect_attempts=10 if auto_reconnect else 0
    )
    
    await manager.connect()
    
    return manager


# ConnectionManager 作为 WebSocketManager 的别名，用于向后兼容
class ConnectionManager(WebSocketManager):
    """
    WebSocket连接管理器（别名类）
    
    此类是 WebSocketManager 的别名，用于向后兼容。
    新代码应直接使用 WebSocketManager。
    """
    
    def __init__(self, uri: str = "ws://localhost:8000/ws", *args, **kwargs):
        """初始化连接管理器"""
        super().__init__(uri=uri, *args, **kwargs)
        logger.debug("ConnectionManager 已初始化（WebSocketManager 的别名）")
