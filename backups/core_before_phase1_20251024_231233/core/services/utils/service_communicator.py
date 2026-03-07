#!/usr/bin/env python3
"""
服务通信器

实现微服务之间的通信，支持REST API、消息队列、异步通信等
"""

import logging
import json
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import uuid

logger = logging.getLogger(__name__)

# 尝试导入消息队列库
try:
    import aio_pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False
    logger.warning("aio - pika不可用，RabbitMQ支持将被禁用")

try:
    import kafka
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka - python不可用，Kafka支持将被禁用")


class CommunicationType(Enum):

    """通信类型"""
    REST = "rest"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    MESSAGE_QUEUE = "message_queue"
    EVENT_STREAM = "event_stream"


class MessageType(Enum):

    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    COMMAND = "command"
    NOTIFICATION = "notification"


class CommunicationProtocol(Enum):

    """通信协议"""
    HTTP = "http"
    HTTPS = "https"
    WS = "ws"
    WSS = "wss"
    AMQP = "amqp"
    KAFKA = "kafka"


@dataclass
class ServiceEndpoint:

    """服务端点"""
    service_name: str
    url: str
    protocol: CommunicationProtocol = CommunicationProtocol.HTTP
    timeout: int = 30
    retries: int = 3
    circuit_breaker_enabled: bool = True


@dataclass
class Message:

    """消息"""
    id: str
    type: MessageType
    sender: str
    recipient: Optional[str] = None
    payload: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: Optional[int] = None

    @property
    def is_expired(self) -> bool:
        """是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.timestamp).seconds > self.ttl


@dataclass
class CommunicationResult:

    """通信结果"""
    success: bool
    response: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    processing_time: float = 0.0
    correlation_id: Optional[str] = None


class CircuitBreaker:

    """熔断器"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):

        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self.lock = threading.Lock()

    def call(self, func: Callable) -> Any:
        """执行带熔断器的函数"""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = func()
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e

    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout

    def _on_success(self):
        """成功回调"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0

    def _on_failure(self):
        """失败回调"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class ConnectionPool:

    """连接池"""

    def __init__(self, max_connections: int = 10, max_keepalive: int = 300):

        self.max_connections = max_connections
        self.max_keepalive = max_keepalive
        self.connections: Dict[str, List[aiohttp.ClientSession]] = {}
        self.lock = threading.Lock()

    def get_connection(self, service_name: str, url: str) -> aiohttp.ClientSession:
        """获取连接"""
        with self.lock:
            if service_name not in self.connections:
                self.connections[service_name] = []

            available_connections = [
                conn for conn in self.connections[service_name]
                if not conn.closed
            ]
            self.connections[service_name] = available_connections

            if available_connections:
                return available_connections[0]

            # 创建新连接
            connector = aiohttp.TCPConnector(limit=self.max_connections)
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.max_keepalive)
            )

            self.connections[service_name].append(session)
            return session

    def cleanup(self):
        """清理连接池"""
        with self.lock:
            for service_name, connections in self.connections.items():
                for connection in connections:
                    if not connection.closed:
                        asyncio.create_task(connection.close())


class RestCommunicator:

    """REST通信器"""

    def __init__(self, service_discovery, connection_pool: ConnectionPool):

        self.service_discovery = service_discovery
        self.connection_pool = connection_pool
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    async def send_request(self, service_name: str, method: str, path: str,
                           data: Optional[Any] = None, headers: Optional[Dict[str, str]] = None,
                           timeout: Optional[int] = None) -> CommunicationResult:
        """发送REST请求"""
        start_time = time.time()
        correlation_id = str(uuid.uuid4())

        try:
            # 服务发现
            service_instance = self.service_discovery.discover_service(service_name)
            if not service_instance:
                return CommunicationResult(
                    success=False,
                    error=f"Service not found: {service_name}",
                    correlation_id=correlation_id
                )

            # 构建完整URL
            url = f"{service_instance.url}{path}"

            # 获取或创建熔断器
            if service_name not in self.circuit_breakers:
                self.circuit_breakers[service_name] = CircuitBreaker()

            circuit_breaker = self.circuit_breakers[service_name]

            # 使用熔断器发送请求

            def make_request():

                return asyncio.run(self._make_http_request(
                    url, method, data, headers, timeout or service_instance.ttl
                ))

            if circuit_breaker.state == "OPEN":
                return CommunicationResult(
                    success=False,
                    error="Circuit breaker is OPEN",
                    correlation_id=correlation_id
                )

            response = await self._make_http_request(url, method, data, headers, timeout or service_instance.ttl)

            processing_time = time.time() - start_time

            if response.status == 200:
                circuit_breaker._on_success()
                return CommunicationResult(
                    success=True,
                    response=response,
                    status_code=response.status,
                    processing_time=processing_time,
                    correlation_id=correlation_id
                )
            else:
                circuit_breaker._on_failure()
                return CommunicationResult(
                    success=False,
                    error=f"HTTP {response.status}",
                    status_code=response.status,
                    processing_time=processing_time,
                    correlation_id=correlation_id
                )

        except Exception as e:
            processing_time = time.time() - start_time
            if service_name in self.circuit_breakers:
                self.circuit_breakers[service_name]._on_failure()

            return CommunicationResult(
                success=False,
                error=str(e),
                processing_time=processing_time,
                correlation_id=correlation_id
            )

    async def _make_http_request(self, url: str, method: str, data: Optional[Any],
                                 headers: Optional[Dict[str, str]], timeout: int) -> aiohttp.ClientResponse:
        """发送HTTP请求"""
        # 准备请求数据
        json_data = None
        if data is not None:
            if isinstance(data, dict):
                json_data = data
            else:
                json_data = json.loads(data) if isinstance(data, str) else data

        # 准备请求头
        request_headers = headers or {}
        if 'Content - Type' not in request_headers:
            request_headers['Content - Type'] = 'application / json'

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.request(
                method=method.upper(),
                url=url,
                json=json_data,
                headers=request_headers
            ) as response:
                return response


class MessageQueueCommunicator:

    """消息队列通信器"""

    def __init__(self, config: Dict[str, Any]):

        self.config = config
        self.rabbitmq_connection = None
        self.kafka_producer = None
        self.kafka_consumer = None

        self.message_handlers: Dict[str, Callable] = {}
        self.event_loop = None

    async def initialize_rabbitmq(self):
        """初始化RabbitMQ连接"""
        if not RABBITMQ_AVAILABLE:
            raise ImportError("RabbitMQ支持不可用")

        try:
            connection_url = self.config.get('rabbitmq_url', 'amqp://guest:guest@localhost:5672/')
            self.rabbitmq_connection = await aio_pika.connect_robust(connection_url)
            logger.info("RabbitMQ连接已建立")

        except Exception as e:
            logger.error(f"RabbitMQ连接失败: {e}")
            raise

    async def initialize_kafka(self):
        """初始化Kafka连接"""
        if not KAFKA_AVAILABLE:
            raise ImportError("Kafka支持不可用")

        try:
            kafka_config = {
                'bootstrap_servers': self.config.get('kafka_servers', ['localhost:9092']),
                'client_id': self.config.get('kafka_client_id', 'service_communicator')
            }

            self.kafka_producer = kafka.KafkaProducer(**kafka_config)
            logger.info("Kafka生产者已初始化")

        except Exception as e:
            logger.error(f"Kafka初始化失败: {e}")
            raise

    async def send_message(self, queue_name: str, message: Message,
                           message_type: str = "direct") -> bool:
        """发送消息"""
        try:
            if self.rabbitmq_connection:
                return await self._send_rabbitmq_message(queue_name, message, message_type)
            elif self.kafka_producer:
                return await self._send_kafka_message(queue_name, message)
            else:
                logger.error("没有可用的消息队列后端")
                return False

        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False

    async def _send_rabbitmq_message(self, queue_name: str, message: Message,
                                     message_type: str) -> bool:
        """发送RabbitMQ消息"""
        try:
            channel = await self.rabbitmq_connection.channel()

            # 声明队列
            await channel.declare_queue(queue_name, durable=True)

            # 序列化消息
            message_body = {
                'id': message.id,
                'type': message.type.value,
                'sender': message.sender,
                'recipient': message.recipient,
                'payload': message.payload,
                'headers': message.headers,
                'correlation_id': message.correlation_id,
                'timestamp': message.timestamp.isoformat(),
                'ttl': message.ttl
            }

            # 发送消息
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(message_body).encode(),
                    message_id=message.id,
                    correlation_id=message.correlation_id,
                    headers=message.headers
                ),
                routing_key=queue_name
            )

            logger.debug(f"RabbitMQ消息已发送到队列: {queue_name}")
            return True

        except Exception as e:
            logger.error(f"RabbitMQ消息发送失败: {e}")
            return False

    async def _send_kafka_message(self, topic: str, message: Message) -> bool:
        """发送Kafka消息"""
        try:
            message_body = {
                'id': message.id,
                'type': message.type.value,
                'sender': message.sender,
                'recipient': message.recipient,
                'payload': message.payload,
                'headers': message.headers,
                'correlation_id': message.correlation_id,
                'timestamp': message.timestamp.isoformat(),
                'ttl': message.ttl
            }

            future = self.kafka_producer.send(
                topic,
                json.dumps(message_body).encode(),
                key=message.id.encode()
            )

            # 等待发送完成
            record_metadata = future.get(timeout=10)
            logger.debug(f"Kafka消息已发送到主题: {topic}, 分区: {record_metadata.partition}")
            return True

        except Exception as e:
            logger.error(f"Kafka消息发送失败: {e}")
            return False

    def register_message_handler(self, queue_name: str, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[queue_name] = handler
        logger.info(f"消息处理器已注册: {queue_name}")

    async def start_consuming(self):
        """开始消费消息"""
        if self.rabbitmq_connection:
            await self._start_rabbitmq_consuming()
        elif self.kafka_consumer:
            await self._start_kafka_consuming()

    async def _start_rabbitmq_consuming(self):
        """开始RabbitMQ消息消费"""
        try:
            channel = await self.rabbitmq_connection.channel()

            for queue_name, handler in self.message_handlers.items():
                # 声明队列
                queue = await channel.declare_queue(queue_name, durable=True)

                # 设置消息处理器
                async def message_handler(message: aio_pika.IncomingMessage):
                    async with message.process():
                        try:
                            message_body = json.loads(message.body.decode())
                            message_obj = self._deserialize_message(message_body)

                            if not message_obj.is_expired:
                                await handler(message_obj)
                            else:
                                logger.warning(f"消息已过期: {message_obj.id}")

                        except Exception as e:
                            logger.error(f"消息处理失败: {e}")

                # 开始消费
                await queue.consume(message_handler)
                logger.info(f"RabbitMQ消息消费已启动: {queue_name}")

            # 保持连接
            await asyncio.Future()  # 运行直到被取消

        except Exception as e:
            logger.error(f"RabbitMQ消息消费启动失败: {e}")

    async def _start_kafka_consuming(self):
        """开始Kafka消息消费"""
        try:
            for topic, handler in self.message_handlers.items():
                # 在线程池中运行Kafka消费者

                def kafka_consumer_worker():

                    try:
                        consumer = kafka.KafkaConsumer(
                            topic,
                            bootstrap_servers=self.config.get('kafka_servers', ['localhost:9092']),
                            group_id=self.config.get('kafka_group_id', 'service_communicator'),
                            auto_offset_reset='latest'
                        )

                        for message in consumer:
                            try:
                                message_body = json.loads(message.value.decode())
                                message_obj = self._deserialize_message(message_body)

                                if not message_obj.is_expired:
                                    # 在事件循环中调用处理器
                                    if self.event_loop:
                                        asyncio.run_coroutine_threadsafe(
                                            handler(message_obj),
                                            self.event_loop
                                        )
                                else:
                                    logger.warning(f"消息已过期: {message_obj.id}")

                            except Exception as e:
                                logger.error(f"Kafka消息处理失败: {e}")

                    except Exception as e:
                        logger.error(f"Kafka消费者异常: {e}")

                # 启动消费者线程
                consumer_thread = threading.Thread(target=kafka_consumer_worker, daemon=True)
                consumer_thread.start()
                logger.info(f"Kafka消息消费已启动: {topic}")

        except Exception as e:
            logger.error(f"Kafka消息消费启动失败: {e}")

    def _deserialize_message(self, data: Dict[str, Any]) -> Message:
        """反序列化消息"""
        return Message(
            id=data['id'],
            type=MessageType(data['type']),
            sender=data['sender'],
            recipient=data.get('recipient'),
            payload=data.get('payload'),
            headers=data.get('headers', {}),
            correlation_id=data.get('correlation_id'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            ttl=data.get('ttl')
        )


class ServiceCommunicator:

    """服务通信器"""

    def __init__(self, service_discovery, config: Optional[Dict[str, Any]] = None):

        self.service_discovery = service_discovery
        self.config = config or {}

        # 通信组件
        self.connection_pool = ConnectionPool(
            max_connections=self.config.get('max_connections', 10),
            max_keepalive=self.config.get('max_keepalive', 300)
        )

        self.rest_communicator = RestCommunicator(service_discovery, self.connection_pool)
        self.message_queue_communicator = MessageQueueCommunicator(self.config)

        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'messages_sent': 0,
            'messages_received': 0
        }

        logger.info("服务通信器初始化完成")

    async def initialize(self):
        """初始化通信器"""
        try:
            # 初始化消息队列
            if self.config.get('enable_rabbitmq', False):
                await self.message_queue_communicator.initialize_rabbitmq()
            elif self.config.get('enable_kafka', False):
                await self.message_queue_communicator.initialize_kafka()

            logger.info("服务通信器初始化完成")

        except Exception as e:
            logger.error(f"服务通信器初始化失败: {e}")

    async def send_rest_request(self, service_name: str, method: str, path: str,
                                data: Optional[Any] = None, headers: Optional[Dict[str, str]] = None,
                                timeout: Optional[int] = None) -> CommunicationResult:
        """发送REST请求"""
        self.stats['total_requests'] += 1

        try:
            result = await self.rest_communicator.send_request(
                service_name, method, path, data, headers, timeout
            )

            if result.success:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1

            self._update_avg_response_time(result.processing_time)
            return result

        except Exception as e:
            self.stats['failed_requests'] += 1
            return CommunicationResult(
                success=False,
                error=str(e)
            )

    async def send_message(self, recipient: str, message_type: MessageType,
                           payload: Any, headers: Optional[Dict[str, Any]] = None) -> bool:
        """发送消息"""
        try:
            message = Message(
                id=str(uuid.uuid4()),
                type=message_type,
                sender=self.config.get('service_name', 'unknown'),
                recipient=recipient,
                payload=payload,
                headers=headers or {}
            )

            # 使用消息队列发送
            success = await self.message_queue_communicator.send_message(
                recipient, message
            )

            if success:
                self.stats['messages_sent'] += 1

            return success

        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False

    def register_message_handler(self, queue_name: str, handler: Callable):
        """注册消息处理器"""
        self.message_queue_communicator.register_message_handler(queue_name, handler)

    async def start_message_consuming(self):
        """开始消息消费"""
        await self.message_queue_communicator.start_consuming()

    def publish_event(self, event_type: str, event_data: Any,


                      recipients: Optional[List[str]] = None):
        """发布事件"""
        async def publish():
            if recipients:
                for recipient in recipients:
                    await self.send_message(
                        recipient,
                        MessageType.EVENT,
                        {'event_type': event_type, 'data': event_data}
                    )
            else:
                # 广播事件
                await self.send_message(
                    'event_bus',
                    MessageType.EVENT,
                    {'event_type': event_type, 'data': event_data}
                )

        # 在后台运行
        asyncio.create_task(publish())

    def subscribe_event(self, event_type: str, handler: Callable):
        """订阅事件"""

        def event_handler(message: Message):

            if (message.type == MessageType.EVENT
                    and message.payload.get('event_type') == event_type):
                return handler(message.payload.get('data'))

        self.register_message_handler(f"event_{event_type}", event_handler)

    async def health_check(self, service_name: str) -> Dict[str, Any]:
        """服务健康检查"""
        try:
            result = await self.send_rest_request(service_name, 'GET', '/health')
            return {
                'service': service_name,
                'healthy': result.success,
                'response_time': result.processing_time,
                'status_code': result.status_code,
                'error': result.error
            }

        except Exception as e:
            return {
                'service': service_name,
                'healthy': False,
                'error': str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

    def _update_avg_response_time(self, processing_time: float):
        """更新平均响应时间"""
        total_requests = self.stats['successful_requests'] + self.stats['failed_requests']
        if total_requests > 0:
            current_avg = self.stats['avg_response_time']
            self.stats['avg_response_time'] = (
                (current_avg * (total_requests - 1)) + processing_time
            ) / total_requests

    async def cleanup(self):
        """清理资源"""
        try:
            self.connection_pool.cleanup()

            if hasattr(self.message_queue_communicator, 'rabbitmq_connection'):
                if self.message_queue_communicator.rabbitmq_connection:
                    await self.message_queue_communicator.rabbitmq_connection.close()

            logger.info("服务通信器资源已清理")

        except Exception as e:
            logger.error(f"清理资源失败: {e}")
