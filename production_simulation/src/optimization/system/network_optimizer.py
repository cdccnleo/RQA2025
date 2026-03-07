"""
Network Optimization Module
网络优化模块

This module provides network optimization capabilities for quantitative trading systems
此模块为量化交易系统提供网络优化能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import threading
import time
import socket
import asyncio
import aiohttp
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import psutil

logger = logging.getLogger(__name__)


class NetworkOptimizer:

    """
    Network Optimizer Class
    网络优化器类

    Provides network communication optimization including connection pooling,
    DNS optimization, and data transfer optimization
    提供网络通信优化，包括连接池、DNS优化和数据传输优化
    """

    def __init__(self,


                 max_connections: int = 100,
                 max_keepalive: int = 10,
                 timeout: float = 30.0):
        """
        Initialize network optimizer
        初始化网络优化器

        Args:
            max_connections: Maximum number of connections
                           最大连接数
            max_keepalive: Maximum keep - alive connections
                          最大保持连接数
            timeout: Default timeout for network operations
                    网络操作的默认超时时间
        """
        self.max_connections = max_connections
        self.max_keepalive = max_keepalive
        self.timeout = timeout

        # HTTP session management
        self.http_session = None
        self.async_session = None

        # DNS cache
        self.dns_cache = {}
        self.dns_cache_lock = threading.Lock()
        self.dns_cache_ttl = 300  # 5 minutes

        # Connection pooling
        self.connection_pools = {}
        self.connection_pool_lock = threading.Lock()

        # Network statistics
        self.network_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'average_response_time': 0.0,
            'dns_lookups': 0,
            'connection_pool_hits': 0,
            'connection_pool_misses': 0
        }

        # Optimization settings
        self.enable_connection_pooling = True
        self.enable_dns_caching = True
        self.enable_compression = True
        self.retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504]
        )

        logger.info("Network optimizer initialized")

    def create_http_session(self) -> requests.Session:
        """
        Create optimized HTTP session
        创建优化的HTTP会话

        Returns:
            requests.Session: Optimized HTTP session
                            优化的HTTP会话
        """
        if self.http_session is None:
            self.http_session = requests.Session()

            # Configure retry strategy
            adapter = HTTPAdapter(
                max_retries=self.retry_strategy,
                pool_connections=self.max_connections,
                pool_maxsize=self.max_connections,
                pool_block=False
            )

            # Mount adapter for both HTTP and HTTPS
            self.http_session.mount('http://', adapter)
            self.http_session.mount('https://', adapter)

            # Set default timeout
            self.http_session.timeout = self.timeout

        return self.http_session

    async def create_async_session(self) -> aiohttp.ClientSession:
        """
        Create optimized async HTTP session
        创建优化的异步HTTP会话

        Returns:
            aiohttp.ClientSession: Optimized async session
                                 优化的异步会话
        """
        if self.async_session is None:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_keepalive,
                ttl_dns_cache=self.dns_cache_ttl,
                use_dns_cache=self.enable_dns_caching
            )

            timeout = aiohttp.ClientTimeout(total=self.timeout)

            self.async_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                trust_env=True
            )

        return self.async_session

    def make_http_request(self,


                          url: str,
                          method: str = 'GET',
                          **kwargs) -> Dict[str, Any]:
        """
        Make optimized HTTP request
        发出优化的HTTP请求

        Args:
            url: Request URL
                请求URL
            method: HTTP method
                   HTTP方法
            **kwargs: Additional request parameters
                     其他请求参数

        Returns:
            dict: Request result
                  请求结果
        """
        start_time = time.time()

        try:
            session = self.create_http_session()

            # Set default parameters
            request_kwargs = {
                'timeout': self.timeout,
                'allow_redirects': True
            }
            request_kwargs.update(kwargs)

            # Make request
            response = session.request(method, url, **request_kwargs)

            response_time = time.time() - start_time

            # Update statistics
            self._update_request_stats(True, response_time, len(response.content))

            result = {
                'success': True,
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'content': response.content,
                'text': response.text,
                'url': response.url,
                'response_time': response_time,
                'encoding': response.encoding
            }

            logger.debug(f"HTTP request to {url} completed in {response_time:.3f}s")
            return result

        except Exception as e:
            response_time = time.time() - start_time
            self._update_request_stats(False, response_time, 0)

            logger.error(f"HTTP request to {url} failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'response_time': response_time,
                'url': url
            }

    async def make_async_request(self,
                                 url: str,
                                 method: str = 'GET',
                                 **kwargs) -> Dict[str, Any]:
        """
        Make optimized async HTTP request
        发出优化的异步HTTP请求

        Args:
            url: Request URL
                请求URL
            method: HTTP method
                   HTTP方法
            **kwargs: Additional request parameters
                     其他请求参数

        Returns:
            dict: Request result
                  请求结果
        """
        start_time = time.time()

        try:
            session = await self.create_async_session()

            # Make request
            async with session.request(method, url, **kwargs) as response:
                content = await response.read()
                response_time = time.time() - start_time

                # Update statistics
                self._update_request_stats(True, response_time, len(content))

                result = {
                    'success': True,
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'content': content,
                    'text': content.decode('utf - 8', errors='ignore'),
                    'url': str(response.url),
                    'response_time': response_time,
                    'encoding': response.charset
                }

                logger.debug(f"Async HTTP request to {url} completed in {response_time:.3f}s")
                return result

        except Exception as e:
            response_time = time.time() - start_time
            self._update_request_stats(False, response_time, 0)

            logger.error(f"Async HTTP request to {url} failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'response_time': response_time,
                'url': url
            }

    def resolve_dns(self, hostname: str) -> Optional[str]:
        """
        Resolve DNS with caching
        使用缓存解析DNS

        Args:
            hostname: Hostname to resolve
                     要解析的主机名

        Returns:
            str: Resolved IP address or None
                 解析的IP地址或None
        """
        if not self.enable_dns_caching:
            return self._resolve_dns_direct(hostname)

        # Check cache first
        with self.dns_cache_lock:
            cached_result = self.dns_cache.get(hostname)
        if cached_result:
            cache_time, ip_address = cached_result
        if datetime.now() - cache_time < timedelta(seconds=self.dns_cache_ttl):
            return ip_address

        # Resolve and cache
        ip_address = self._resolve_dns_direct(hostname)
        if ip_address:
            with self.dns_cache_lock:
                self.dns_cache[hostname] = (datetime.now(), ip_address)

        self.network_stats['dns_lookups'] += 1
        return ip_address

    def _resolve_dns_direct(self, hostname: str) -> Optional[str]:
        """
        Direct DNS resolution without caching
        不使用缓存的直接DNS解析

        Args:
            hostname: Hostname to resolve
                     要解析的主机名

        Returns:
            str: Resolved IP address or None
                 解析的IP地址或None
        """
        try:
            # Try IPv4 first
            result = socket.getaddrinfo(hostname, None, socket.AF_INET)
            if result:
                return result[0][4][0]

            # Try IPv6
            result = socket.getaddrinfo(hostname, None, socket.AF_INET6)
            if result:
                return result[0][4][0]

        except Exception as e:
            logger.error(f"DNS resolution failed for {hostname}: {str(e)}")

        return None

    def get_connection_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status
        获取连接池状态

        Returns:
            dict: Connection pool information
                  连接池信息
        """
        with self.connection_pool_lock:
            return {
                'total_pools': len(self.connection_pools),
                'pools': self.connection_pools.copy(),
                'max_connections': self.max_connections,
                'max_keepalive': self.max_keepalive
            }

    def optimize_connection_pool(self, host: str, port: int = 80) -> None:
        """
        Optimize connection pool for specific host
        为特定主机优化连接池

        Args:
            host: Target host
                目标主机
            port: Target port
                目标端口
        """
        pool_key = f"{host}:{port}"

        with self.connection_pool_lock:
            if pool_key not in self.connection_pools:
                self.connection_pools[pool_key] = {
                    'connections': 0,
                    'max_connections': min(self.max_connections, 10),  # Per host limit
                    'created_at': datetime.now(),
                    'last_used': datetime.now()
                }

            self.network_stats['connection_pool_hits'] += 1

    def _update_request_stats(self,


                              success: bool,
                              response_time: float,
                              bytes_transferred: int) -> None:
        """
        Update network request statistics
        更新网络请求统计信息

        Args:
            success: Whether request was successful
                    请求是否成功
            response_time: Time taken for request
                          请求所用时间
            bytes_transferred: Bytes transferred
                             传输的字节数
        """
        self.network_stats['total_requests'] += 1

        if success:
            self.network_stats['successful_requests'] += 1
            self.network_stats['total_bytes_received'] += bytes_transferred
        else:
            self.network_stats['failed_requests'] += 1

        # Update average response time
        total_requests = self.network_stats['total_requests']
        current_avg = self.network_stats['average_response_time']
        self.network_stats['average_response_time'] = (
            (current_avg * (total_requests - 1)) + response_time
        ) / total_requests

    def get_network_stats(self) -> Dict[str, Any]:
        """
        Get network statistics
        获取网络统计信息

        Returns:
            dict: Network statistics
                  网络统计信息
        """
        try:
            net_io = psutil.net_io_counters()
            network_stats = self.network_stats.copy()

            if net_io:
                network_stats.update({
                    'system_bytes_sent': net_io.bytes_sent,
                    'system_bytes_recv': net_io.bytes_recv,
                    'system_packets_sent': net_io.packets_sent,
                    'system_packets_recv': net_io.packets_recv,
                    'system_errin': net_io.errin,
                    'system_errout': net_io.errout
                })

            network_stats['success_rate'] = (
                network_stats['successful_requests']
                / max(network_stats['total_requests'], 1) * 100
            )

            return network_stats

        except Exception as e:
            logger.error(f"Failed to get network stats: {str(e)}")
            return self.network_stats.copy()

    def test_network_connectivity(self,


                                  host: str,
                                  port: int = 80,
                                  timeout: float = 5.0) -> Dict[str, Any]:
        """
        Test network connectivity to a host
        测试到主机的网络连接

        Args:
            host: Target host
                目标主机
            port: Target port
                目标端口
            timeout: Connection timeout
                    连接超时时间

        Returns:
            dict: Connectivity test result
                  连接测试结果
        """
        start_time = time.time()

        try:
            # Resolve DNS first
            ip_address = self.resolve_dns(host)
            if not ip_address:
                return {
                    'success': False,
                    'error': 'DNS resolution failed',
                    'host': host,
                    'response_time': time.time() - start_time
                }

            # Test TCP connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)

            result = sock.connect_ex((ip_address, port))
            sock.close()

            response_time = time.time() - start_time

            return {
                'success': result == 0,
                'host': host,
                'ip_address': ip_address,
                'port': port,
                'response_time': response_time,
                'error_code': result if result != 0 else None
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'host': host,
                'port': port,
                'response_time': time.time() - start_time
            }

    def optimize_data_transfer(self,


                               data: bytes,
                               compression: bool = True) -> bytes:
        """
        Optimize data for network transfer
        优化数据以进行网络传输

        Args:
            data: Data to optimize
                 要优化的数据
            compression: Whether to use compression
                        是否使用压缩

        Returns:
            bytes: Optimized data
                   优化的数据
        """
        if not compression or not self.enable_compression:
            return data

        try:
            import gzip
            compressed_data = gzip.compress(data)
            # Only use compression if it's actually smaller
            return compressed_data if len(compressed_data) < len(data) else data
        except Exception as e:
            logger.warning(f"Data compression failed: {str(e)}")
            return data

    def cleanup_resources(self) -> None:
        """
        Cleanup network resources
        清理网络资源

        Returns:
            None
        """
        try:
            if self.http_session:
                self.http_session.close()
                self.http_session = None

            if self.async_session:
                asyncio.run(self.async_session.close())
                self.async_session = None

            # Clear DNS cache
            with self.dns_cache_lock:
                self.dns_cache.clear()

            # Clear connection pools
            with self.connection_pool_lock:
                self.connection_pools.clear()

            logger.info("Network resources cleaned up")

        except Exception as e:
            logger.error(f"Failed to cleanup network resources: {str(e)}")

    def get_network_optimizer_status(self) -> Dict[str, Any]:
        """
        Get network optimizer status
        获取网络优化器状态

        Returns:
            dict: Optimizer status information
                  优化器状态信息
        """
        return {
            'max_connections': self.max_connections,
            'max_keepalive': self.max_keepalive,
            'timeout': self.timeout,
            'enable_connection_pooling': self.enable_connection_pooling,
            'enable_dns_caching': self.enable_dns_caching,
            'enable_compression': self.enable_compression,
            'dns_cache_size': len(self.dns_cache),
            'connection_pool_status': self.get_connection_pool_status(),
            'network_stats': self.get_network_stats(),
            'http_session_active': self.http_session is not None,
            'async_session_active': self.async_session is not None
        }


# Global network optimizer instance
# 全局网络优化器实例
network_optimizer = NetworkOptimizer()

__all__ = ['NetworkOptimizer', 'network_optimizer']
