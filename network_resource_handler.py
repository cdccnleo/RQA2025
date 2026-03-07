#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络和资源异常处理工具
解决网络超时、连接失败、资源访问等问题
"""

import requests
import time
import socket
import threading
from typing import Dict, Optional, Any, Callable
from functools import wraps
import logging
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkResourceHandler:
    """网络资源处理器"""

    def __init__(self):
        self.session_pool = {}
        self.connection_pool = {}
        self.retry_strategy = self._create_retry_strategy()

    def _create_retry_strategy(self) -> Retry:
        """创建重试策略"""
        return Retry(
            total=3,  # 总重试次数
            backoff_factor=0.3,  # 退避因子
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]  # 需要重试的方法
        )

    def create_resilient_session(self, timeout: float = 30.0,
                                 max_retries: int = 3) -> requests.Session:
        """创建有弹性的HTTP会话"""
        session = requests.Session()

        # 设置超时
        session.timeout = timeout

        # 配置重试策略
        adapter = HTTPAdapter(max_retries=self.retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # 禁用SSL警告（生产环境应该保持启用）
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        return session

    def safe_http_request(self, method: str, url: str,
                          timeout: float = 30.0,
                          max_retries: int = 3,
                          **kwargs) -> Optional[requests.Response]:
        """安全的HTTP请求"""
        session = None
        try:
            session = self.create_resilient_session(timeout, max_retries)

            # 设置默认参数
            request_kwargs = {
                'timeout': timeout,
                'verify': False,  # 生产环境应该设置为True
                **kwargs
            }

            # 发送请求
            response = session.request(method.upper(), url, **request_kwargs)

            # 检查响应状态
            response.raise_for_status()

            return response

        except requests.exceptions.Timeout:
            logger.error(f"请求超时: {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"连接错误: {url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP错误 {e.response.status_code}: {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"请求异常: {url} - {str(e)}")
            return None
        finally:
            if session:
                session.close()

    def safe_json_request(self, method: str, url: str,
                          timeout: float = 30.0,
                          **kwargs) -> Optional[Dict]:
        """安全的JSON请求"""
        response = self.safe_http_request(method, url, timeout, **kwargs)

        if response is None:
            return None

        try:
            return response.json()
        except json.JSONDecodeError:
            logger.error(f"JSON解析失败: {url}")
            return None
        except Exception as e:
            logger.error(f"响应解析失败: {url} - {str(e)}")
            return None

    def safe_socket_connect(self, host: str, port: int,
                            timeout: float = 10.0) -> bool:
        """安全的套接字连接测试"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()

            return result == 0

        except socket.error:
            return False
        except Exception:
            return False

    def safe_database_connect(self, connection_func: Callable,
                              max_retries: int = 3,
                              retry_delay: float = 1.0) -> Any:
        """安全的数据库连接"""
        last_exception = None

        for attempt in range(max_retries):
            try:
                connection = connection_func()
                # 测试连接
                if hasattr(connection, 'ping'):
                    connection.ping()
                elif hasattr(connection, 'cursor'):
                    cursor = connection.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()

                return connection

            except Exception as e:
                last_exception = e
                logger.warning(f"数据库连接失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")

                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # 指数退避

        logger.error(f"数据库连接失败，已达到最大重试次数: {str(last_exception)}")
        return None


class ResourceAccessController:
    """资源访问控制器"""

    def __init__(self):
        self.resource_locks = {}
        self.access_limits = {}
        self.monitoring_data = {}

    def safe_file_operation(self, file_path: str, operation: str, *args, **kwargs) -> Any:
        """安全文件操作"""
        try:
            # 检查文件路径安全性
            if not self._is_safe_path(file_path):
                raise ValueError(f"不安全的文件路径: {file_path}")

            # 检查文件权限
            if not self._check_file_permissions(file_path, operation):
                raise PermissionError(f"文件权限不足: {file_path}")

            # 执行操作
            if operation == 'read':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif operation == 'write':
                with open(file_path, 'w', encoding='utf-8') as f:
                    return f.write(*args, **kwargs)
            elif operation == 'append':
                with open(file_path, 'a', encoding='utf-8') as f:
                    return f.write(*args, **kwargs)
            else:
                raise ValueError(f"不支持的操作: {operation}")

        except FileNotFoundError:
            logger.error(f"文件不存在: {file_path}")
            return None
        except PermissionError:
            logger.error(f"文件权限错误: {file_path}")
            return None
        except IOError as e:
            logger.error(f"文件IO错误: {file_path} - {str(e)}")
            return None
        except Exception as e:
            logger.error(f"文件操作异常: {file_path} - {str(e)}")
            return None

    def _is_safe_path(self, file_path: str) -> bool:
        """检查文件路径安全性"""
        # 检查路径遍历攻击
        if '..' in file_path or file_path.startswith('/'):
            return False

        # 检查可疑字符
        suspicious_chars = ['<', '>', '|', '&', ';', '$']
        if any(char in file_path for char in suspicious_chars):
            return False

        return True

    def _check_file_permissions(self, file_path: str, operation: str) -> bool:
        """检查文件权限"""
        import os

        if operation in ['read', 'write', 'append']:
            return os.access(file_path, os.R_OK if operation == 'read' else os.W_OK)

        return False

    def safe_resource_access(self, resource_name: str, access_func: Callable, *args, **kwargs) -> Any:
        """安全资源访问"""
        lock = self.resource_locks.get(resource_name)
        if lock is None:
            lock = threading.RLock()
            self.resource_locks[resource_name] = lock

        with lock:
            try:
                # 检查访问限制
                if not self._check_access_limits(resource_name):
                    raise RuntimeError(f"资源访问受限: {resource_name}")

                # 执行访问
                result = access_func(*args, **kwargs)

                # 更新监控数据
                self._update_monitoring_data(resource_name, 'success')

                return result

            except Exception as e:
                # 更新监控数据
                self._update_monitoring_data(resource_name, 'error', str(e))
                logger.error(f"资源访问失败: {resource_name} - {str(e)}")
                raise

    def _check_access_limits(self, resource_name: str) -> bool:
        """检查访问限制"""
        limits = self.access_limits.get(resource_name, {})
        if not limits:
            return True

        # 检查频率限制
        current_time = time.time()
        window = limits.get('window', 60)  # 默认60秒窗口
        max_requests = limits.get('max_requests', 100)  # 默认最大100次请求

        if resource_name not in self.monitoring_data:
            self.monitoring_data[resource_name] = {'requests': []}

        requests = self.monitoring_data[resource_name]['requests']

        # 清理过期请求
        requests[:] = [req for req in requests if current_time - req < window]

        # 检查是否超过限制
        if len(requests) >= max_requests:
            return False

        # 记录新请求
        requests.append(current_time)
        return True

    def _update_monitoring_data(self, resource_name: str, status: str, error: str = None):
        """更新监控数据"""
        if resource_name not in self.monitoring_data:
            self.monitoring_data[resource_name] = {
                'success_count': 0,
                'error_count': 0,
                'last_error': None,
                'requests': []
            }

        data = self.monitoring_data[resource_name]

        if status == 'success':
            data['success_count'] += 1
        elif status == 'error':
            data['error_count'] += 1
            data['last_error'] = error


class CircuitBreaker:
    """熔断器"""

    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Exception = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """执行带熔断器的函数调用"""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise RuntimeError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)

            if self.state == 'HALF_OPEN':
                self._reset()

            return result

        except self.expected_exception as e:
            self._record_failure()
            raise e

    def _record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning("Circuit breaker opened due to too many failures")

    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        if self.last_failure_time is None:
            return True

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _reset(self):
        """重置熔断器"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'
        logger.info("Circuit breaker reset to CLOSED state")

# 装饰器


def with_retry(max_retries: int = 3, delay: float = 1.0,
               backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (backoff ** attempt)
                        logger.warning(f"函数 {func.__name__} 执行失败，{sleep_time:.1f}秒后重试: {str(e)}")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"函数 {func.__name__} 执行失败，已达到最大重试次数: {str(e)}")

            raise last_exception
        return wrapper
    return decorator


def with_timeout(timeout: float = 30.0):
    """超时装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(f"函数 {func.__name__} 执行超时 ({timeout}秒)")

            # 设置信号处理器
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))

            try:
                return func(*args, **kwargs)
            finally:
                # 恢复信号处理器
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


def with_circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """熔断器装饰器"""
    breaker = CircuitBreaker(failure_threshold, recovery_timeout)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator

# 实用工具函数


def safe_url_join(base_url: str, path: str) -> str:
    """安全的URL拼接"""
    if not base_url or not path:
        return ""

    base_url = base_url.rstrip('/')
    path = path.lstrip('/')

    return f"{base_url}/{path}"


def validate_url(url: str) -> bool:
    """验证URL格式"""
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return url_pattern.match(url) is not None


if __name__ == "__main__":
    # 测试网络处理器
    network_handler = NetworkResourceHandler()

    print("测试网络连接:")
    is_connected = network_handler.safe_socket_connect("google.com", 80, timeout=5.0)
    print(f"连接 google.com:80 - {'成功' if is_connected else '失败'}")

    # 测试资源访问控制器
    resource_controller = ResourceAccessController()

    print("\n测试文件操作:")
    test_content = "测试内容"
    result = resource_controller.safe_file_operation("test_file.txt", "write", test_content)
    print(f"写文件结果: {result}")

    read_result = resource_controller.safe_file_operation("test_file.txt", "read")
    print(f"读文件结果: {read_result}")

    # 测试熔断器
    breaker = CircuitBreaker(failure_threshold=3)

    @breaker.call
    def test_function():
        raise ValueError("测试异常")

    print("\n测试熔断器:")
    for i in range(5):
        try:
            test_function()
        except ValueError:
            print(f"调用 {i+1}: 失败")
        except RuntimeError as e:
            print(f"调用 {i+1}: {str(e)}")

    print(f"熔断器状态: {breaker.state}")
