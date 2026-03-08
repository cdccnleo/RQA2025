#!/usr/bin/env python3
"""
异步HTTP客户端
高性能异步HTTP请求，支持连接池和自动重试
"""

import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class HTTPConfig:
    """HTTP客户端配置"""
    timeout: float = 30.0
    max_connections: int = 100
    max_keepalive: int = 20
    retry_attempts: int = 3
    retry_delay: float = 1.0
    verify_ssl: bool = True


class AsyncHTTPClient:
    """
    异步HTTP客户端
    
    特性：
    1. 连接池复用
    2. 自动重试机制
    3. 请求超时控制
    4. 并发请求限制
    5. 响应缓存
    """
    
    def __init__(self, config: Optional[HTTPConfig] = None):
        self.config = config or HTTPConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        # 统计信息
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retried_requests': 0,
            'total_time': 0.0
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._init_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def _init_session(self):
        """初始化会话"""
        if self._session is None:
            # 配置连接池
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                limit_per_host=self.config.max_keepalive,
                enable_cleanup_closed=True,
                force_close=False,
            )
            
            # 配置超时
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            # 创建会话
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
            
            # 创建信号量限制并发
            self._semaphore = asyncio.Semaphore(self.config.max_connections)
            
            logger.info(f"HTTP client initialized: max_connections={self.config.max_connections}")
    
    async def close(self):
        """关闭会话"""
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("HTTP client closed")
    
    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """带重试的请求"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with self._semaphore:
                    start_time = time.time()
                    
                    async with self._session.request(
                        method=method,
                        url=url,
                        ssl=self.config.verify_ssl,
                        **kwargs
                    ) as response:
                        
                        elapsed = time.time() - start_time
                        self._stats['total_time'] += elapsed
                        
                        # 读取响应内容
                        content = await response.read()
                        
                        # 更新统计
                        self._stats['total_requests'] += 1
                        
                        if response.status < 400:
                            self._stats['successful_requests'] += 1
                        else:
                            self._stats['failed_requests'] += 1
                            response.raise_for_status()
                        
                        return response
                        
            except aiohttp.ClientError as e:
                last_exception = e
                self._stats['retried_requests'] += 1
                
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # 指数退避
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.config.retry_attempts} attempts: {e}")
                    raise
        
        raise last_exception
    
    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """GET请求"""
        await self._init_session()
        
        response = await self._request_with_retry(
            'GET',
            url,
            params=params,
            headers=headers,
            **kwargs
        )
        
        return await self._parse_response(response)
    
    async def post(
        self,
        url: str,
        data: Optional[Any] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """POST请求"""
        await self._init_session()
        
        response = await self._request_with_retry(
            'POST',
            url,
            data=data,
            json=json_data,
            headers=headers,
            **kwargs
        )
        
        return await self._parse_response(response)
    
    async def _parse_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """解析响应"""
        content_type = response.headers.get('Content-Type', '')
        
        if 'application/json' in content_type:
            return await response.json()
        else:
            text = await response.text()
            return {'status': response.status, 'text': text}
    
    async def batch_requests(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """
        批量并发请求
        
        Args:
            requests: 请求列表，每个请求是包含method, url等参数的字典
            max_concurrent: 最大并发数
            
        Returns:
            响应列表
        """
        await self._init_session()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _single_request(req: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    method = req.pop('method', 'GET')
                    url = req.pop('url')
                    
                    response = await self._request_with_retry(method, url, **req)
                    return await self._parse_response(response)
                except Exception as e:
                    logger.error(f"Batch request failed: {e}")
                    return {'error': str(e)}
        
        # 并发执行所有请求
        tasks = [_single_request(req.copy()) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            result if not isinstance(result, Exception) else {'error': str(result)}
            for result in results
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        
        if stats['total_requests'] > 0:
            stats['average_time'] = stats['total_time'] / stats['total_requests']
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
        else:
            stats['average_time'] = 0.0
            stats['success_rate'] = 0.0
        
        return stats


class HTTPClientManager:
    """HTTP客户端管理器"""
    
    _instance: Optional['HTTPClientManager'] = None
    _clients: Dict[str, AsyncHTTPClient] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def create_client(
        self,
        name: str,
        config: Optional[HTTPConfig] = None
    ) -> AsyncHTTPClient:
        """创建客户端"""
        client = AsyncHTTPClient(config)
        await client._init_session()
        self._clients[name] = client
        logger.info(f"Created HTTP client: {name}")
        return client
    
    def get_client(self, name: str = "default") -> AsyncHTTPClient:
        """获取客户端"""
        if name not in self._clients:
            raise KeyError(f"HTTP client '{name}' not found")
        return self._clients[name]
    
    async def close_all(self):
        """关闭所有客户端"""
        for name, client in self._clients.items():
            await client.close()
        self._clients.clear()
        logger.info("All HTTP clients closed")


# 全局HTTP客户端管理器
_http_manager: Optional[HTTPClientManager] = None


async def setup_http_client(config: Optional[HTTPConfig] = None):
    """设置HTTP客户端"""
    global _http_manager
    _http_manager = HTTPClientManager()
    await _http_manager.create_client("default", config)
    logger.info("HTTP client setup complete")


def get_http_client() -> AsyncHTTPClient:
    """获取默认HTTP客户端"""
    if _http_manager is None:
        raise RuntimeError("HTTP client not initialized")
    return _http_manager.get_client("default")


def cached_http_request(ttl: int = 300):
    """
    HTTP请求缓存装饰器
    
    用法:
        @cached_http_request(ttl=600)
        async def fetch_market_data(symbol: str):
            return await http_client.get(f"https://api.example.com/data/{symbol}")
    """
    from src.utils.performance.multi_level_cache import get_cache
    
    def decorator(func):
        cache = get_cache()
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"http:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # 尝试获取缓存
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"HTTP cache hit: {cache_key}")
                return cached_result
            
            # 执行请求
            result = await func(*args, **kwargs)
            
            # 缓存结果
            await cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# 示例用法
if __name__ == "__main__":
    async def test_http_client():
        print("=== 异步HTTP客户端测试 ===\n")
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        
        # 配置
        config = HTTPConfig(
            timeout=10.0,
            max_connections=20,
            retry_attempts=3
        )
        
        try:
            # 初始化HTTP客户端
            await setup_http_client(config)
            client = get_http_client()
            
            # 测试1: GET请求
            print("测试1: GET请求")
            result = await client.get("https://httpbin.org/get")
            print(f"响应状态: {result.get('status', 'N/A')}")
            
            # 测试2: POST请求
            print("\n测试2: POST请求")
            result = await client.post(
                "https://httpbin.org/post",
                json_data={"test": "data"}
            )
            print(f"响应状态: {result.get('status', 'N/A')}")
            
            # 测试3: 批量请求
            print("\n测试3: 批量请求")
            requests = [
                {"method": "GET", "url": f"https://httpbin.org/get?n={i}"}
                for i in range(5)
            ]
            results = await client.batch_requests(requests, max_concurrent=3)
            print(f"批量请求完成: {len(results)} 个")
            
            # 测试4: 统计信息
            print("\n测试4: 统计信息")
            stats = client.get_stats()
            print(f"总请求数: {stats['total_requests']}")
            print(f"成功请求: {stats['successful_requests']}")
            print(f"成功率: {stats['success_rate']:.2%}")
            print(f"平均响应时间: {stats['average_time']:.3f}s")
            
            print("\n=== 测试完成 ===")
            
        except Exception as e:
            print(f"测试失败: {e}")
        finally:
            if _http_manager:
                await _http_manager.close_all()
    
    # 运行测试
    asyncio.run(test_http_client())
