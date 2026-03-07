"""
RQA2025 微服务通信模块

提供微服务间的通信机制，支持REST API、消息队列、gRPC等多种通信方式
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Protocol
from dataclasses import dataclass, field
from datetime import datetime
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoint:

    """服务端点配置"""
    name: str
    host: str
    port: int
    protocol: str = "http"
    health_check_path: str = "/health"
    timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0

    @property
    def base_url(self) -> str:
        """获取基础URL"""
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        """获取健康检查URL"""
        return f"{self.base_url}{self.health_check_path}"


@dataclass
class Message:

    """消息结构"""
    id: str
    type: str
    source: str
    destination: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'type': self.type,
            'source': self.source,
            'destination': self.destination,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'headers': self.headers
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        return cls(
            id=data['id'],
            type=data['type'],
            source=data['source'],
            destination=data['destination'],
            payload=data['payload'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            correlation_id=data.get('correlation_id'),
            headers=data.get('headers', {})
        )


class ServiceRegistry:

    """服务注册中心"""

    def __init__(self):

        self.services: Dict[str, ServiceEndpoint] = {}
        self.service_health: Dict[str, bool] = {}
        self.service_instances: Dict[str, List[ServiceEndpoint]] = {}  # 支持多实例
        self.service_load: Dict[str, int] = {}  # 服务负载统计
        self.service_response_times: Dict[str, List[float]] = {}  # 响应时间历史
        self.lock = threading.RLock()
        self.health_check_interval = 30  # 健康检查间隔(秒)
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False
        self.max_response_time_history = 100  # 最大响应时间历史记录数

    def register_service(self, endpoint: ServiceEndpoint):
        """注册服务"""
        with self.lock:
            self.services[endpoint.name] = endpoint
            self.service_health[endpoint.name] = False
            logger.info(f"服务已注册: {endpoint.name} -> {endpoint.base_url}")

    def unregister_service(self, service_name: str):
        """注销服务"""
        with self.lock:
            if service_name in self.services:
                del self.services[service_name]
                del self.service_health[service_name]
                logger.info(f"服务已注销: {service_name}")

    def get_service(self, service_name: str) -> Optional[ServiceEndpoint]:
        """获取服务端点"""
        with self.lock:
            return self.services.get(service_name)

    def list_services(self) -> List[str]:
        """列出所有服务"""
        with self.lock:
            return list(self.services.keys())

    def is_service_healthy(self, service_name: str) -> bool:
        """检查服务是否健康"""
        with self.lock:
            return self.service_health.get(service_name, False)

    def start_health_checks(self):
        """启动健康检查"""
        if self._running:
            return

        self._running = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
        logger.info("服务健康检查已启动")

    def stop_health_checks(self):
        """停止健康检查"""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
        logger.info("服务健康检查已停止")

    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"健康检查出错: {e}")

            time.sleep(self.health_check_interval)

    def _perform_health_checks(self):
        """执行健康检查"""
        with self.lock:
            services_to_check = list(self.services.items())

        for service_name, endpoint in services_to_check:
            try:
                response = requests.get(
                    endpoint.health_url,
                    timeout=endpoint.timeout
                )
                is_healthy = response.status_code == 200

                with self.lock:
                    old_health = self.service_health.get(service_name, False)
                    self.service_health[service_name] = is_healthy

                    if old_health != is_healthy:
                        status = "健康" if is_healthy else "不健康"
                        logger.info(f"服务状态变更: {service_name} -> {status}")

            except Exception as e:
                with self.lock:
                    old_health = self.service_health.get(service_name, False)
                    self.service_health[service_name] = False

                    if old_health:
                        logger.warning(f"服务健康检查失败: {service_name} - {e}")


class RESTCommunicator:

    """REST API通信器"""

    def __init__(self, registry: ServiceRegistry):

        self.registry = registry
        self.session: Optional[aiohttp.ClientSession] = None
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def initialize(self):
        """初始化异步会话"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """关闭异步会话"""
        if self.session:
            await self.session.close()
            self.session = None

    async def call_service(
        self,
        service_name: str,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        调用服务API

        Args:
            service_name: 服务名称
            method: HTTP方法
            path: API路径
            data: 请求数据
            headers: 请求头
            timeout: 超时时间

        Returns:
            响应结果
        """
        await self.initialize()

        service = self.registry.get_service(service_name)
        if not service:
            raise ValueError(f"服务未找到: {service_name}")

        if not self.registry.is_service_healthy(service_name):
            raise ConnectionError(f"服务不可用: {service_name}")

        url = f"{service.base_url}{path}"

        # 准备请求
        request_data = None
        if data:
            if method.upper() in ['POST', 'PUT', 'PATCH']:
                request_data = json.dumps(data)
                if not headers:
                    headers = {}
                headers['Content - Type'] = 'application / json'
            else:
                # GET请求的查询参数
                import urllib.parse
                query_string = urllib.parse.urlencode(data)
                url = f"{url}?{query_string}"

        try:
            async with self.session.request(
                method.upper(),
                url,
                data=request_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout or service.timeout)
            ) as response:
                result = {
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'url': str(response.url)
                }

                if response.status == 200:
                    try:
                        result['data'] = await response.json()
                    except BaseException:
                        result['data'] = await response.text()
                else:
                    result['error'] = await response.text()

                return result

        except asyncio.TimeoutError:
            raise TimeoutError(f"服务调用超时: {service_name}")
        except Exception as e:
            raise ConnectionError(f"服务调用失败: {service_name} - {e}")

    def call_service_sync(


        self,
        service_name: str,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """同步调用服务API"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.call_service(service_name, method, path, data, headers, timeout)
            )
        finally:
            loop.close()


class MessageQueueCommunicator:

    """消息队列通信器"""

    def __init__(self, registry: ServiceRegistry):

        self.registry = registry
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def register_handler(self, message_type: str, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
        logger.info(f"消息处理器已注册: {message_type}")

    def unregister_handler(self, message_type: str):
        """注销消息处理器"""
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]
            logger.info(f"消息处理器已注销: {message_type}")

    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        try:
            # 这里可以集成实际的消息队列，如RabbitMQ、Kafka等
            # 目前使用简化的实现

            destination_service = self.registry.get_service(message.destination)
            if not destination_service:
                logger.error(f"目标服务不存在: {message.destination}")
                return False

            if not self.registry.is_service_healthy(message.destination):
                logger.error(f"目标服务不可用: {message.destination}")
                return False

            # 将消息放入队列，等待处理
            await self.message_queue.put(message)
            logger.info(f"消息已发送: {message.type} -> {message.destination}")
            return True

        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False

    async def start_message_processing(self):
        """启动消息处理"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_messages())
        logger.info("消息处理已启动")

    async def stop_message_processing(self):
        """停止消息处理"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("消息处理已停止")

    async def _process_messages(self):
        """处理消息队列"""
        while self._running:
            try:
                message = await self.message_queue.get()

                # 查找消息处理器
                handler = self.message_handlers.get(message.type)
                if handler:
                    try:
                        await handler(message)
                        logger.info(f"消息处理成功: {message.type}")
                    except Exception as e:
                        logger.error(f"消息处理失败: {message.type} - {e}")
                else:
                    logger.warning(f"未找到消息处理器: {message.type}")

                self.message_queue.task_done()

            except Exception as e:
                logger.error(f"消息处理循环出错: {e}")
                await asyncio.sleep(1)


class ServiceCommunicator:

    """微服务通信器主类"""

    def __init__(self):

        self.registry = ServiceRegistry()
        self.rest_communicator = RESTCommunicator(self.registry)
        self.mq_communicator = MessageQueueCommunicator(self.registry)
        self._initialized = False

    async def initialize(self):
        """初始化通信器"""
        if self._initialized:
            return

        await self.rest_communicator.initialize()
        await self.mq_communicator.start_message_processing()
        self.registry.start_health_checks()

        self._initialized = True
        logger.info("微服务通信器已初始化")

    async def close(self):
        """关闭通信器"""
        if not self._initialized:
            return

        await self.rest_communicator.close()
        await self.mq_communicator.stop_message_processing()
        self.registry.stop_health_checks()

        self._initialized = False
        logger.info("微服务通信器已关闭")

    def register_service(self, endpoint: ServiceEndpoint):
        """注册服务"""
        self.registry.register_service(endpoint)

    def unregister_service(self, service_name: str):
        """注销服务"""
        self.registry.unregister_service(service_name)

    def get_service(self, service_name: str) -> Optional[ServiceEndpoint]:
        """获取服务"""
        return self.registry.get_service(service_name)

    def list_services(self) -> List[str]:
        """列出服务"""
        return self.registry.list_services()

    def is_service_healthy(self, service_name: str) -> bool:
        """检查服务健康状态"""
        return self.registry.is_service_healthy(service_name)

    async def call_service(
        self,
        service_name: str,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """调用服务API"""
        return await self.rest_communicator.call_service(
            service_name, method, path, data, headers, timeout
        )

    def call_service_sync(


        self,
        service_name: str,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """同步调用服务API"""
        return self.rest_communicator.call_service_sync(
            service_name, method, path, data, headers, timeout
        )

    def register_message_handler(self, message_type: str, handler: Callable):
        """注册消息处理器"""
        self.mq_communicator.register_handler(message_type, handler)

    async def send_message(self, message: Message) -> bool:
        """发送消息"""
        return await self.mq_communicator.send_message(message)

    def create_message(


        self,
        msg_type: str,
        destination: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Message:
        """创建消息"""
        import uuid
        return Message(
            id=str(uuid.uuid4()),
            type=msg_type,
            source="current_service",  # 应该从配置中获取
            destination=destination,
            payload=payload,
            correlation_id=correlation_id,
            headers=headers or {}
        )

    def get_status(self) -> Dict[str, Any]:
        """获取通信器状态"""
        return {
            'initialized': self._initialized,
            'services': {
                name: {
                    'endpoint': f"{endpoint.base_url}",
                    'healthy': self.registry.is_service_healthy(name)
                }
                for name, endpoint in self.registry.services.items()
            },
            'message_handlers': list(self.mq_communicator.message_handlers.keys())
        }


# 全局通信器实例
_communicator_instance: Optional[ServiceCommunicator] = None
_communicator_lock = threading.Lock()


def get_service_communicator() -> ServiceCommunicator:
    """获取服务通信器实例（单例模式）"""
    global _communicator_instance

    if _communicator_instance is None:
        with _communicator_lock:
            if _communicator_instance is None:
                _communicator_instance = ServiceCommunicator()

    return _communicator_instance


async def initialize_communicator():
    """初始化全局通信器"""
    communicator = get_service_communicator()
    await communicator.initialize()


async def close_communicator():
    """关闭全局通信器"""
    communicator = get_service_communicator()
    await communicator.close()


# 便捷函数

def register_service(name: str, host: str, port: int, **kwargs):
    """注册服务便捷函数"""
    endpoint = ServiceEndpoint(name=name, host=host, port=port, **kwargs)
    get_service_communicator().register_service(endpoint)


def call_service_sync(service_name: str, method: str, path: str, **kwargs) -> Dict[str, Any]:
    """同步调用服务便捷函数"""
    return get_service_communicator().call_service_sync(service_name, method, path, **kwargs)


# ============================================================================
# 云原生服务优化器 (新增)
# ============================================================================


# 云原生服务优化相关协议
class ServiceCommunicationOptimizer(Protocol):
    """服务通信优化器协议"""
    async def optimize_communication(self, service_name: str, request: Dict) -> Dict: ...


class ContainerResourceOptimizer(Protocol):
    """容器资源优化器协议"""
    async def optimize_resources(self, service_name: str): ...


class ServiceHealthMonitor(Protocol):
    """服务健康监控器协议"""
    async def monitor_health(self, service_name: str): ...


class AdaptiveTimeoutCalculator(Protocol):
    """自适应超时计算器协议"""
    async def calculate_timeout(self, service_name: str, request: Dict) -> float: ...


class PerformanceStatsRecorder(Protocol):
    """性能统计记录器协议"""
    async def record_stats(self, service_name: str, response_time: float, success: bool): ...


@dataclass
class CloudNativeConfig:
    """云原生服务优化配置"""
    enable_service_mesh: bool = True
    enable_container_optimization: bool = True
    enable_health_monitoring: bool = True
    enable_performance_tracking: bool = True
    adaptive_timeout_enabled: bool = True
    monitoring_interval: int = 30


class ServiceCommunicationOptimizerImpl:
    """服务通信优化器实现 - 职责：优化服务间通信"""

    def __init__(self, config: CloudNativeConfig, service_mesh, connection_pool, timeout_calculator: AdaptiveTimeoutCalculator, stats_recorder: PerformanceStatsRecorder):
        self.config = config
        self.service_mesh = service_mesh
        self.connection_pool = connection_pool
        self.timeout_calculator = timeout_calculator
        self.stats_recorder = stats_recorder

    async def optimize_communication(self, service_name: str, request: Dict) -> Dict:
        """优化服务间通信"""
        start_time = time.time()

        try:
            # 1. 智能路由选择
            optimal_instance = await self.service_mesh.select_optimal_instance(service_name)
            if not optimal_instance:
                raise ValueError(f"没有可用的服务实例: {service_name}")

            # 2. 连接池复用优化
            connection = await self.connection_pool.get_connection(optimal_instance)

            # 3. 自适应超时设置
            timeout = await self.timeout_calculator.calculate_timeout(service_name, request)

            # 4. 发送优化请求
            response = await self._send_optimized_request(connection, request, timeout)

            # 5. 记录性能统计
            response_time = time.time() - start_time
            await self.stats_recorder.record_stats(service_name, response_time, True)

            # 6. 连接回收
            await self.connection_pool.return_connection(connection)

            return response

        except Exception as e:
            # 记录失败统计
            response_time = time.time() - start_time
            await self.stats_recorder.record_stats(service_name, response_time, False)
            raise e

    async def _send_optimized_request(self, connection, request: Dict, timeout: float) -> Dict:
        """发送优化后的请求"""
        # 这里实现具体的请求发送逻辑
        # 可以使用连接池复用、压缩、缓存等优化技术
        pass


class ContainerResourceOptimizerImpl:
    """容器资源优化器实现 - 职责：优化容器资源配置"""

    def __init__(self, config: CloudNativeConfig, resource_manager):
        self.config = config
        self.resource_manager = resource_manager

    async def optimize_resources(self, service_name: str):
        """优化容器资源配置"""
        logger.info(f"开始优化容器资源: {service_name}")

        # 1. 监控当前资源使用
        current_usage = await self.resource_manager.get_current_usage(service_name)

        # 2. 预测未来资源需求
        predicted_usage = await self.resource_manager.predict_usage(service_name)

        # 3. 计算最优资源配置
        optimal_config = await self._calculate_optimal_resources(current_usage, predicted_usage)

        # 4. 应用资源配置
        await self.resource_manager.apply_resource_config(service_name, optimal_config)

        logger.info(f"容器资源优化完成: {service_name} -> {optimal_config}")

    async def _calculate_optimal_resources(self, current_usage: Dict, predicted_usage: Dict) -> Dict:
        """计算最优资源配置"""
        # 实现资源配置计算逻辑
        return {
            'cpu': max(current_usage.get('cpu', 0.5), predicted_usage.get('cpu', 0.5)),
            'memory': max(current_usage.get('memory', 512), predicted_usage.get('memory', 512))
        }


class ServiceHealthMonitorImpl:
    """服务健康监控器实现 - 职责：监控服务健康状态"""

    def __init__(self, config: CloudNativeConfig, health_monitor, stats_recorder: PerformanceStatsRecorder):
        self.config = config
        self.health_monitor = health_monitor
        self.stats_recorder = stats_recorder

    async def monitor_health(self, service_name: str):
        """监控服务健康状态"""
        while True:
            try:
                # 1. 执行健康检查
                health_status = await self.health_monitor.perform_health_check(service_name)

                # 2. 收集性能指标
                performance_metrics = await self.health_monitor.collect_metrics(service_name)

                # 3. 计算健康评分
                health_score = self._calculate_health_score(health_status, performance_metrics)

                # 4. 触发告警（如果需要）
                if health_score < 0.8:
                    await self._trigger_health_alert(service_name, health_score, performance_metrics)

                # 5. 记录健康统计
                await self._record_health_stats(service_name, health_score, performance_metrics)

            except Exception as e:
                logger.error(f"服务健康监控异常 {service_name}: {e}")

            await asyncio.sleep(self.config.monitoring_interval)

    def _calculate_health_score(self, health_status: Dict, performance_metrics: Dict) -> float:
        """计算健康评分"""
        # 实现健康评分计算逻辑
        base_score = 1.0 if health_status.get('healthy', False) else 0.0

        # 基于性能指标调整评分
        response_time = performance_metrics.get('avg_response_time', 1.0)
        error_rate = performance_metrics.get('error_rate', 0.0)

        # 响应时间惩罚
        if response_time > 5.0:
            base_score *= 0.8
        elif response_time > 2.0:
            base_score *= 0.9

        # 错误率惩罚
        base_score *= (1.0 - error_rate)

        return max(0.0, min(1.0, base_score))

    async def _trigger_health_alert(self, service_name: str, health_score: float, metrics: Dict):
        """触发健康告警"""
        logger.warning(f"服务健康告警: {service_name}, 评分: {health_score:.2f}")
        # 这里可以实现告警通知逻辑

    async def _record_health_stats(self, service_name: str, health_score: float, metrics: Dict):
        """记录健康统计"""
        # 实现健康统计记录逻辑
        pass


class AdaptiveTimeoutCalculatorImpl:
    """自适应超时计算器实现 - 职责：计算自适应超时时间"""

    def __init__(self, config: CloudNativeConfig, performance_stats: Dict):
        self.config = config
        self.performance_stats = performance_stats

    async def calculate_timeout(self, service_name: str, request: Dict) -> float:
        """计算自适应超时时间"""
        if not self.config.adaptive_timeout_enabled:
            return 10.0  # 默认超时

        # 基于历史响应时间计算超时
        if service_name in self.performance_stats:
            stats = self.performance_stats[service_name]
            avg_response_time = stats.get('avg_response_time', 1.0)
            # 设置为平均响应时间的3倍
            timeout = min(avg_response_time * 3, 30.0)  # 最大30秒
        else:
            timeout = 10.0  # 默认10秒

        # 基于请求复杂度调整
        request_complexity = self._estimate_request_complexity(request)
        timeout *= (1 + request_complexity * 0.5)  # 复杂度越高超时越长

        return timeout

    def _estimate_request_complexity(self, request: Dict) -> float:
        """估算请求复杂度"""
        complexity = 0.0

        # 基于数据大小估算复杂度
        data_size = len(str(request))
        if data_size > 10000:  # 10KB
            complexity += 0.3
        elif data_size > 1000:  # 1KB
            complexity += 0.1

        # 基于嵌套深度估算复杂度
        def calculate_depth(obj, current_depth=0):
            if isinstance(obj, dict):
                return max([calculate_depth(v, current_depth + 1) for v in obj.values()] + [current_depth])
            elif isinstance(obj, list):
                return max([calculate_depth(item, current_depth + 1) for item in obj] + [current_depth])
            else:
                return current_depth

        depth = calculate_depth(request)
        complexity += min(depth * 0.1, 0.4)  # 最大增加0.4

        return min(complexity, 1.0)


class PerformanceStatsRecorderImpl:
    """性能统计记录器实现 - 职责：记录性能统计"""

    def __init__(self, config: CloudNativeConfig, performance_stats: Dict):
        self.config = config
        self.performance_stats = performance_stats

    async def record_stats(self, service_name: str, response_time: float, success: bool):
        """记录性能统计"""
        if service_name not in self.performance_stats:
            self.performance_stats[service_name] = {
                'total_calls': 0,
                'total_time': 0.0,
                'success_count': 0,
                'avg_response_time': 0.0,
                'error_rate': 0.0
            }

        stats = self.performance_stats[service_name]
        stats['total_calls'] += 1
        stats['total_time'] += response_time

        if success:
            stats['success_count'] += 1

        stats['avg_response_time'] = stats['total_time'] / stats['total_calls']
        stats['error_rate'] = 1.0 - (stats['success_count'] / stats['total_calls'])


class CloudNativeServiceOptimizer:

    """
    云原生服务优化器 - 重构版：组合模式

    提供服务网格优化、容器资源管理、健康监控等功能
    """

    def __init__(self, config: Optional[CloudNativeConfig] = None):
        # 初始化配置
        self.config = config or CloudNativeConfig()

        # 初始化底层组件
        self.service_mesh = ServiceMeshOptimizer()
        self.resource_manager = ContainerResourceManager()
        self.health_monitor = ServiceHealthMonitor()
        self.connection_pool = OptimizedConnectionPool()
        self.performance_stats = {}

        # 初始化专门的组件
        self.timeout_calculator = AdaptiveTimeoutCalculatorImpl(self.config, self.performance_stats)
        self.stats_recorder = PerformanceStatsRecorderImpl(self.config, self.performance_stats)

        self.comm_optimizer = ServiceCommunicationOptimizerImpl(
            self.config, self.service_mesh, self.connection_pool,
            self.timeout_calculator, self.stats_recorder
        )
        self.resource_optimizer = ContainerResourceOptimizerImpl(self.config, self.resource_manager)
        self.health_monitor_impl = ServiceHealthMonitorImpl(self.config, self.health_monitor, self.stats_recorder)

        logger.info("重构后的云原生服务优化器初始化完成")

    # 代理方法到专门的组件
    async def optimize_service_communication(self, service_name: str, request: Dict) -> Dict:
        """优化服务间通信 - 代理到通信优化器"""
        return await self.comm_optimizer.optimize_communication(service_name, request)

    async def optimize_container_resources(self, service_name: str):
        """优化容器资源配置 - 代理到资源优化器"""
        return await self.resource_optimizer.optimize_resources(service_name)

    async def monitor_service_health(self, service_name: str):
        """监控服务健康状态 - 代理到健康监控器"""
        return await self.health_monitor_impl.monitor_health(service_name)

    # 保持向后兼容性
    async def _calculate_adaptive_timeout(self, service_name: str, request: Dict) -> float:
        """计算自适应超时时间（向后兼容）"""
        return await self.timeout_calculator.calculate_timeout(service_name, request)

    def _estimate_request_complexity(self, request: Dict) -> float:
        """估算请求复杂度（向后兼容）"""
        return self.timeout_calculator._estimate_request_complexity(request)

    async def _send_optimized_request(self, connection, request: Dict, timeout: float) -> Dict:
        """发送优化后的请求（向后兼容）"""
        return await self.comm_optimizer._send_optimized_request(connection, request, timeout)

    async def _record_performance_stats(self, service_name: str, response_time: float, success: bool):
        """记录性能统计（向后兼容）"""
        return await self.stats_recorder.record_stats(service_name, response_time, success)
        cpu_limit = max(cpu_current * 1.2, cpu_predicted * 1.1)
if __name__ == "__main__":
    # 测试代码
    print("微服务通信模块测试")

    # 创建通信器
    communicator = get_service_communicator()

    # 注册测试服务
    test_service = ServiceEndpoint(
        name="test - service",
        host="localhost",
        port=5001,
        protocol="http"
    )
    communicator.register_service(test_service)

    print("服务已注册:", communicator.list_services())
    print("通信器状态:", communicator.get_status())

    # 测试云原生优化器
    optimizer = get_cloud_native_optimizer()
    stats = optimizer.get_optimization_stats()
    print("云原生优化器状态:", stats)

    print("微服务通信模块测试完成")
