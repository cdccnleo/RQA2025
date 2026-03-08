"""
Load Balancer Module
负载均衡器模块

This module provides load balancing capabilities for async operations
此模块为异步操作提供负载均衡能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, TypeVar
from datetime import datetime
from enum import Enum
import threading
import time
import secrets
import hashlib
import statistics

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LoadBalancingStrategy(Enum):

    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"          # Round - robin distribution
    RANDOM = "random"                    # Random distribution
    LEAST_CONNECTIONS = "least_connections"  # Least connections
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # Weighted round - robin
    IP_HASH = "ip_hash"                  # IP hash - based
    LEAST_RESPONSE_TIME = "least_response_time"  # Least response time


class BackendStatus(Enum):

    """Backend server status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"


class BackendServer:

    """
    Backend Server Class
    后端服务器类

    Represents a backend server in the load balancer
    表示负载均衡器中的后端服务器
    """

    def __init__(self,


                 server_id: str,
                 address: str,
                 port: int = 80,
                 weight: int = 1,
                 max_connections: int = 100):
        """
        Initialize backend server
        初始化后端服务器

        Args:
            server_id: Unique server identifier
                      唯一服务器标识符
            address: Server address (IP or hostname)
                   服务器地址（IP或主机名）
            port: Server port
                 服务器端口
            weight: Server weight for weighted algorithms
                  加权算法的服务器权重
            max_connections: Maximum concurrent connections
                           最大并发连接数
        """
        self.server_id = server_id
        self.address = address
        self.port = port
        self.weight = weight
        self.max_connections = max_connections

        # Runtime state
        self.status = BackendStatus.HEALTHY
        self.active_connections = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0.0
        self.last_health_check = datetime.now()
        self.response_times: List[float] = []

        # Thread safety
        self._lock = threading.Lock()

    def get_connection_string(self) -> str:
        """
        Get connection string for the server
        获取服务器的连接字符串

        Returns:
            str: Connection string (address:port)
                 连接字符串（地址:端口）
        """
        return f"{self.address}:{self.port}"

    def can_accept_connection(self) -> bool:
        """
        Check if server can accept new connections
        检查服务器是否可以接受新连接

        Returns:
            bool: True if can accept, False otherwise
                  如果可以接受则返回True，否则返回False
        """
        return (self.status == BackendStatus.HEALTHY
                and self.active_connections < self.max_connections)

    def record_request(self, response_time: float, success: bool = True) -> None:
        """
        Record a request completion
        记录请求完成

        Args:
            response_time: Time taken for the request (seconds)
                          请求所用时间（秒）
            success: Whether the request was successful
                    请求是否成功
        """
        with self._lock:
            self.total_requests += 1
            self.response_times.append(response_time)

            # Keep only last 100 response times
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]

            if success:
                self.successful_requests += 1
                # Update average response time
                self.average_response_time = statistics.mean(self.response_times)
            else:
                self.failed_requests += 1

    def get_load_factor(self) -> float:
        """
        Get load factor for this server (0.0 to 1.0)
        获取此服务器的负载因子（0.0到1.0）

        Returns:
            float: Load factor (higher = more loaded)
                   负载因子（越高=负载越大）
        """
        connection_load = self.active_connections / max(self.max_connections, 1)
        response_time_factor = min(self.average_response_time / 10.0,
                                   1.0) if self.average_response_time > 0 else 0.0

        return (connection_load + response_time_factor) / 2.0

    def get_health_score(self) -> float:
        """
        Get health score for this server (0.0 to 1.0)
        获取此服务器的健康评分（0.0到1.0）

        Returns:
            float: Health score (higher = healthier)
                   健康评分（越高=越健康）
        """
        if self.status != BackendStatus.HEALTHY:
            return 0.0

        success_rate = (self.successful_requests
                        / max(self.total_requests, 1))

        # Health score based on success rate and response time
        health_score = success_rate

        # Penalize slow response times
        if self.average_response_time > 5.0:
            health_score *= 0.8
        elif self.average_response_time > 2.0:
            health_score *= 0.9

        return health_score

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert server info to dictionary
        将服务器信息转换为字典

        Returns:
            dict: Server information
                  服务器信息
        """
        return {
            'server_id': self.server_id,
            'address': self.address,
            'port': self.port,
            'weight': self.weight,
            'status': self.status.value,
            'active_connections': self.active_connections,
            'total_requests': self.total_requests,
            'success_rate': (self.successful_requests / max(self.total_requests, 1)) * 100,
            'average_response_time': self.average_response_time,
            'load_factor': self.get_load_factor(),
            'health_score': self.get_health_score(),
            'last_health_check': self.last_health_check.isoformat()
        }


class LoadBalancer:

    """
    Load Balancer Class
    负载均衡器类

    Distributes requests across multiple backend servers
    在多个后端服务器之间分配请求
    """

    def __init__(self,


                 name: str = "default_load_balancer",
                 strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        """
        Initialize load balancer
        初始化负载均衡器

        Args:
            name: Name of the load balancer
                负载均衡器的名称
            strategy: Load balancing strategy to use
                     要使用的负载均衡策略
        """
        self.name = name
        self.strategy = strategy
        self.servers: Dict[str, BackendServer] = {}
        self.healthy_servers: List[BackendServer] = []

        # Strategy - specific state
        self.round_robin_index = 0
        self.ip_hash_cache: Dict[str, BackendServer] = {}

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Health checking
        self.health_check_interval = 30  # seconds
        self.health_check_thread: Optional[threading.Thread] = None
        self.is_running = False

        logger.info(f"Load balancer {name} initialized with strategy {strategy.value}")

    def add_server(self,


                   server_id: str,
                   address: str,
                   port: int = 80,
                   weight: int = 1,
                   max_connections: int = 100) -> None:
        """
        Add a backend server to the load balancer
        将后端服务器添加到负载均衡器

        Args:
            server_id: Unique server identifier
                      唯一服务器标识符
            address: Server address
                    服务器地址
            port: Server port
                   服务器端口
            weight: Server weight
                   服务器权重
            max_connections: Maximum connections
                           最大连接数
        """
        server = BackendServer(server_id, address, port, weight, max_connections)
        self.servers[server_id] = server

        if server.can_accept_connection():
            self.healthy_servers.append(server)

        logger.info(f"Added server {server_id} to load balancer {self.name}")

    def remove_server(self, server_id: str) -> bool:
        """
        Remove a backend server from the load balancer
        从负载均衡器中移除后端服务器

        Args:
            server_id: Server identifier
                      服务器标识符

        Returns:
            bool: True if removed successfully, False otherwise
                  移除成功返回True，否则返回False
        """
        if server_id in self.servers:
            server = self.servers[server_id]
            del self.servers[server_id]

            if server in self.healthy_servers:
                self.healthy_servers.remove(server)

            logger.info(f"Removed server {server_id} from load balancer {self.name}")
            return True

        return False

    def get_server(self,


                   client_ip: Optional[str] = None,
                   request_data: Optional[Any] = None) -> Optional[BackendServer]:
        """
        Get the next server based on load balancing strategy
        根据负载均衡策略获取下一个服务器

        Args:
            client_ip: Client IP address for IP hash strategy
                      客户端IP地址，用于IP哈希策略
            request_data: Request data for custom strategies
                        请求数据，用于自定义策略

        Returns:
            BackendServer: Selected server or None if no healthy servers
                          选定的服务器，如果没有健康服务器则返回None
        """
        if not self.healthy_servers:
            logger.warning(f"No healthy servers available in {self.name}")
            return None

        self.total_requests += 1

        try:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection()
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return self._random_selection()
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection()
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection()
            elif self.strategy == LoadBalancingStrategy.IP_HASH:
                return self._ip_hash_selection(client_ip or "127.0.0.1")
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_selection()
            else:
                return self._round_robin_selection()

        except Exception as e:
            logger.error(f"Error selecting server in {self.name}: {str(e)}")
            return secrets.choice(self.healthy_servers)

    def _round_robin_selection(self) -> BackendServer:
        """Round - robin server selection"""
        server = self.healthy_servers[self.round_robin_index]
        self.round_robin_index = (self.round_robin_index + 1) % len(self.healthy_servers)
        return server

    def _random_selection(self) -> BackendServer:
        """Random server selection"""
        return secrets.choice(self.healthy_servers)

    def _least_connections_selection(self) -> BackendServer:
        """Least connections server selection"""
        return min(self.healthy_servers, key=lambda s: s.active_connections)

    def _weighted_round_robin_selection(self) -> BackendServer:
        """Weighted round - robin server selection"""
        total_weight = sum(server.weight for server in self.healthy_servers)
        if total_weight == 0:
            return self._round_robin_selection()

        # Simple weighted selection
        rand_value = secrets.randint(1, total_weight)
        current_weight = 0

        for server in self.healthy_servers:
            current_weight += server.weight
            if rand_value <= current_weight:
                return server

        return self.healthy_servers[0]  # Fallback

    def _ip_hash_selection(self, client_ip: str) -> BackendServer:
        """IP hash - based server selection"""
        if client_ip in self.ip_hash_cache:
            return self.ip_hash_cache[client_ip]

        # Create hash from IP
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        server_index = hash_value % len(self.healthy_servers)
        server = self.healthy_servers[server_index]

        # Cache the result
        self.ip_hash_cache[client_ip] = server

        return server

    def _least_response_time_selection(self) -> BackendServer:
        """Least response time server selection"""
        return min(self.healthy_servers, key=lambda s: s.average_response_time or float('inf'))

    def record_request_result(self,


                              server_id: str,
                              response_time: float,
                              success: bool = True) -> None:
        """
        Record the result of a request to a server
        记录对服务器请求的结果

        Args:
            server_id: Server identifier
                      服务器标识符
            response_time: Response time in seconds
                          响应时间（秒）
            success: Whether the request was successful
                    请求是否成功
        """
        if server_id in self.servers:
            server = self.servers[server_id]
            server.record_request(response_time, success)

            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

    def start_health_monitoring(self) -> bool:
        """
        Start health monitoring for all servers
        开始所有服务器的健康监控

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning(f"{self.name} health monitoring already running")
            return False

        try:
            self.is_running = True
            self.health_check_thread = threading.Thread(
                target=self._health_monitoring_loop, daemon=True)
            self.health_check_thread.start()
            logger.info(f"Health monitoring started for {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {str(e)}")
            self.is_running = False
            return False

    def stop_health_monitoring(self) -> bool:
        """
        Stop health monitoring
        停止健康监控

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning(f"{self.name} health monitoring not running")
            return False

        try:
            self.is_running = False
            if self.health_check_thread and self.health_check_thread.is_alive():
                self.health_check_thread.join(timeout=5.0)
            logger.info(f"Health monitoring stopped for {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop health monitoring: {str(e)}")
            return False

    def _health_monitoring_loop(self) -> None:
        """
        Health monitoring loop
        健康监控循环
        """
        logger.info(f"Health monitoring loop started for {self.name}")

        while self.is_running:
            try:
                self._perform_health_checks()
                time.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health monitoring loop error in {self.name}: {str(e)}")
                time.sleep(self.health_check_interval)

        logger.info(f"Health monitoring loop stopped for {self.name}")

    def _perform_health_checks(self) -> None:
        """
        Perform health checks on all servers
        对所有服务器执行健康检查
        """
        for server in self.servers.values():
            try:
                # Simple health check (ping)
                is_healthy = self._check_server_health(server)

                # Update server status
                old_status = server.status
                server.status = BackendStatus.HEALTHY if is_healthy else BackendStatus.UNHEALTHY
                server.last_health_check = datetime.now()

                # Update healthy servers list
                if is_healthy and server not in self.healthy_servers:
                    self.healthy_servers.append(server)
                elif not is_healthy and server in self.healthy_servers:
                    self.healthy_servers.remove(server)

                # Log status changes
                if old_status != server.status:
                    logger.info(
                        f"Server {server.server_id} status changed: {old_status.value} -> {server.status.value}")

            except Exception as e:
                logger.error(f"Health check failed for server {server.server_id}: {str(e)}")
                server.status = BackendStatus.UNHEALTHY

    def _check_server_health(self, server: BackendServer) -> bool:
        """
        Check health of a server
        检查服务器的健康状态

        Args:
            server: Server to check
                   要检查的服务器

        Returns:
            bool: True if healthy, False otherwise
                  如果健康则返回True，否则返回False
        """
        # Placeholder health check - in real implementation, this would
        # perform actual connectivity and service checks
        try:
            # Simulate health check
            time.sleep(0.01)  # Simulate network latency
            return True  # Assume healthy for demo
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get load balancer status
        获取负载均衡器状态

        Returns:
            dict: Status information
                  状态信息
        """
        return {
            'name': self.name,
            'strategy': self.strategy.value,
            'total_servers': len(self.servers),
            'healthy_servers': len(self.healthy_servers),
            'total_requests': self.total_requests,
            'success_rate': (self.successful_requests / max(self.total_requests, 1)) * 100,
            'servers': {sid: server.to_dict() for sid, server in self.servers.items()},
            'is_running': self.is_running
        }


# Global load balancer instance
# 全局负载均衡器实例
load_balancer = LoadBalancer("global_load_balancer")

__all__ = [
    'LoadBalancingStrategy',
    'BackendStatus',
    'BackendServer',
    'LoadBalancer',
    'load_balancer'
]
