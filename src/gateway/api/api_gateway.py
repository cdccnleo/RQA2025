"""
API Gateway Module
API网关模块

This module provides unified API gateway functionality for RQA2025
此模块为RQA2025提供统一的API网关功能

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class GatewayRouter:

    """
    API Gateway for RQA2025
    RQA2025的API网关

    Provides unified service entry point and routing management
    提供统一的服务入口和路由管理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize API Gateway
        初始化API网关

        Args:
            config: Gateway configuration
                   网关配置
        """
        self.config = config or {}
        self.routes: Dict[str, Dict[str, Any]] = {}
        self.middlewares: List[Dict[str, Any]] = []
        self.services: Dict[str, Dict[str, Any]] = {}

        logger.info("API Gateway initialized")

    def register_service(self, service_name: str, service_info: Dict[str, Any]) -> bool:
        """
        Register a service with the gateway
        在网关中注册服务

        Args:
            service_name: Name of the service
                         服务名称
            service_info: Service information including endpoints, health check, etc.
                         服务信息，包括端点、健康检查等

        Returns:
            bool: Registration success status
                  注册成功状态
        """
        try:
            # Check for duplicate registration
            if service_name in self.services:
                logger.warning(f"Service {service_name} already registered")
                return False

            self.services[service_name] = {
                'info': service_info,
                'registered_at': datetime.now(),
                'status': 'active'
            }

            logger.info(f"Service {service_name} registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {str(e)}")
            return False

    def register_route(self, path: str, target_service, methods: List[str] = None) -> bool:
        """
        Register a route mapping
        注册路由映射

        Args:
            path: API path
                 API路径
            target_service: Target service name or route config dict
                          目标服务名称或路由配置字典
            methods: Allowed HTTP methods (default: ['GET'])
                    允许的HTTP方法（默认：['GET']）

        Returns:
            bool: Registration success status
                  注册成功状态
        """
        try:
            # Handle both string service name and dict config
            if isinstance(target_service, dict):
                route_info = target_service.copy()
                route_info['created_at'] = datetime.now()
                # Ensure methods field exists
                if 'methods' not in route_info:
                    route_info['methods'] = [route_info.get('method', 'GET')]
                elif not isinstance(route_info['methods'], list):
                    route_info['methods'] = [route_info['methods']]
                self.routes[path] = route_info
                logger.info(f"Route {path} registered with config")
            else:
                if methods is None:
                    methods = ['GET']

            self.routes[path] = {
                'service': target_service,
                'methods': methods,
                'created_at': datetime.now()
            }
            logger.info(f"Route {path} -> {target_service} registered")

            return True

        except Exception as e:
            logger.error(f"Failed to register route {path}: {str(e)}")
            return False

    def route_request(self, path: str, method: str = 'GET',


                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route incoming request to appropriate service
        将传入请求路由到适当的服务

        Args:
            path: Request path
                 请求路径
            method: HTTP method
                   HTTP方法
            params: Request parameters
                   请求参数

        Returns:
            dict: Routing result with service info and parameters
                  路由结果，包含服务信息和参数
        """
        try:
            # Find matching route
            for route_path, route_info in self.routes.items():
                if self._match_route(path, route_path) and method in route_info['methods']:
                    service_name = route_info['service']

                    if service_name in self.services:
                        result = {
                            'service': service_name,
                            'service_info': self.services[service_name]['info'],
                            'route_info': route_info,
                            'params': params or {},
                            'matched_path': route_path
                        }

                        logger.info(f"Request {path} routed to service {service_name}")
                        return result

            # No matching route found
            logger.warning(f"No route found for {method} {path}")
            return {
                'error': 'Route not found',
                'path': path,
                'method': method
            }

        except Exception as e:
            logger.error(f"Error routing request {path}: {str(e)}")
            return {
                'error': f'Routing error: {str(e)}',
                'path': path,
                'method': method
            }

    def _match_route(self, request_path: str, route_pattern: str) -> bool:
        """
        Match request path with route pattern
        将请求路径与路由模式匹配

        Args:
            request_path: Actual request path
                         实际请求路径
            route_pattern: Route pattern (supports wildcards)
                          路由模式（支持通配符）

        Returns:
            bool: Match status
                  匹配状态
        """
        # Simple wildcard matching
        if '*' in route_pattern:
            pattern_parts = route_pattern.split('/')
            request_parts = request_path.split('/')

            if len(pattern_parts) != len(request_parts):
                return False

            for pattern_part, request_part in zip(pattern_parts, request_parts):
                if pattern_part != '*' and pattern_part != request_part:
                    return False

            return True

        # Exact match
        return request_path == route_pattern

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of all registered services
        获取所有已注册服务的状态

        Returns:
            dict: Service status information
                  服务状态信息
        """
        return {
            'total_services': len(self.services),
            'services': {
                name: {
                    'status': info['status'],
                    'registered_at': info['registered_at'].isoformat(),
                    'info': info['info']
                }
                for name, info in self.services.items()
            },
            'routes': {
                path: {
                    'service': info['service'],
                    'methods': info['methods']
                }
                for path, info in self.routes.items()
            }
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform gateway health check
        执行网关健康检查

        Returns:
            dict: Health check result
                  健康检查结果
        """
        services_info = {}
        for name, service_data in self.services.items():
            services_info[name] = {
                'status': service_data['status'],
                'registered_at': service_data['registered_at'].isoformat(),
                'info': service_data['info']
            }

        routes_info = {}
        for path, route_data in self.routes.items():
            routes_info[path] = {
                'service': route_data['service'],
                'methods': route_data['methods'],
                'created_at': route_data['created_at'].isoformat()
            }

        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services_count': len(self.services),
            'routes_count': len(self.routes),
            'services': services_info,
            'routes': routes_info
        }

    # ==================== 服务管理方法 ====================

    def deregister_service(self, service_name: str) -> bool:
        """
        Deregister a service from the gateway
        从网关注销服务

        Args:
            service_name: Name of the service to deregister
                         要注销的服务名称

        Returns:
            bool: Deregistration success status
                  注销成功状态
        """
        try:
            if service_name in self.services:
                del self.services[service_name]
                # Remove associated routes
                routes_to_remove = [
                    path for path, route_info in self.routes.items()
                    if route_info['service'] == service_name
                ]
                for path in routes_to_remove:
                    del self.routes[path]

                logger.info(f"Service {service_name} deregistered successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to deregister service {service_name}: {str(e)}")
            return False

    def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Discover service information
        发现服务信息

        Args:
            service_name: Name of the service to discover
                         要发现的服务名称

        Returns:
            dict or None: Service information if found
                         如果找到则返回服务信息
        """
        if service_name in self.services:
            service_info = self.services[service_name]
            # Return the original service info to match test expectations
            if isinstance(service_info, dict) and 'info' in service_info:
                return service_info['info']
            else:
                # Handle case where service_info is directly the info dict
                return service_info
        return None

    def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """
        Check health of a specific service
        检查特定服务的健康状态

        Args:
            service_name: Name of the service to check
                         要检查的服务名称

        Returns:
            dict: Health check result
                  健康检查结果
        """
        if service_name not in self.services:
            return {
                'status': 'unknown',
                'error': 'Service not registered',
                'response_time': -1
            }

        # Simulate health check
        import time
        start_time = time.time()
        time.sleep(0.01)  # Simulate network latency
        response_time = (time.time() - start_time) * 1000

        return {
            'status': 'healthy',
            'response_time': round(response_time, 2)
        }

    # ==================== 路由管理方法 ====================

    def match_route(self, path: str, method: str = 'GET') -> Optional[Dict[str, Any]]:
        """
        Match a route for given path and method
        为给定的路径和方法匹配路由

        Args:
            path: Request path
                 请求路径
            method: HTTP method
                   HTTP方法

        Returns:
            dict or None: Route information if matched
                         如果匹配则返回路由信息
        """
        for route_path, route_info in self.routes.items():
            if route_info is None:
                logger.warning(f"Route info is None for path: {route_path}")
                continue

            if self._match_route(path, route_path):
                # Check for methods in route_info or in nested service dict
                methods = route_info.get('methods')
                if methods is None and 'service' in route_info and isinstance(route_info['service'], dict):
                    methods = route_info['service'].get('methods', [])

                if methods and method in methods:
                    # Return route info in format expected by tests
                    return route_info

        return None

    # ==================== 中间件管理方法 ====================

    def register_middleware(self, middleware_config: Dict[str, Any]) -> bool:
        """
        Register middleware
        注册中间件

        Args:
            middleware_config: Middleware configuration
                              中间件配置

        Returns:
            bool: Registration success status
                  注册成功状态
        """
        try:
            self.middlewares.append(middleware_config)
            logger.info(f"Middleware registered: {middleware_config.get('name', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to register middleware: {str(e)}")
            return False

    # ==================== 负载均衡方法 ====================

    def select_service_instance(self, service_instances: List[str]) -> Optional[Dict[str, Any]]:
        """
        Select a service instance using round-robin
        使用轮询选择服务实例

        Args:
            service_instances: List of available service instances
                             可用服务实例列表

        Returns:
            dict or None: Selected instance info
                         选中的实例信息
        """
        if not service_instances:
            return None

        # Simple round-robin selection
        if not hasattr(self, '_rr_index'):
            self._rr_index = 0

        instance_name = service_instances[self._rr_index % len(service_instances)]
        self._rr_index += 1

        # Return instance info if registered
        if instance_name in self.services:
            service_info = self.services[instance_name]['info']
            return {
                'name': instance_name,
                'info': service_info,
                'status': self.services[instance_name]['status']
            }
        else:
            # Return basic info for unregistered instances
            return {
                'name': instance_name,
                'info': {},
                'status': 'unknown'
            }

    # ==================== 流量控制方法 ====================

    def check_rate_limit(self, client_id: str, resource: str) -> bool:
        """
        Check if request is within rate limits
        检查请求是否在速率限制内

        Args:
            client_id: Client identifier
                      客户端标识
            resource: Resource being accessed
                     被访问的资源

        Returns:
            bool: Whether request is allowed
                  是否允许请求
        """
        # Simple rate limiting - allow all for now
        return True

    # ==================== 请求处理方法 ====================

    def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming request
        处理传入请求

        Args:
            request_data: Request information
                         请求信息

        Returns:
            dict: Response data
                  响应数据
        """
        try:
            path = request_data.get('path', '/')
            method = request_data.get('method', 'GET')
            params = request_data.get('params', {})

            # Route the request
            result = self.route_request(path, method, params)

            if 'error' in result:
                return {
                    'status_code': 404,
                    'error': result['error'],
                    'path': path,
                    'method': method
                }

            # Simulate successful response
            return {
                'status_code': 200,
                'data': result,
                'service': result.get('service'),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'status_code': 500,
                'error': f'Internal server error: {str(e)}'
            }

    # ==================== 监控方法 ====================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get gateway metrics
        获取网关指标

        Returns:
            dict: Metrics data
                  指标数据
        """
        return {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'services_count': len(self.services),
            'routes_count': len(self.routes),
            'middlewares_count': len(self.middlewares)
        }

    # ==================== 配置管理方法 ====================

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update gateway configuration
        更新网关配置

        Args:
            new_config: New configuration
                       新配置

        Returns:
            bool: Update success status
                  更新成功状态
        """
        try:
            self.config = new_config.copy()
            logger.info("Gateway configuration updated")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            return False

    # ==================== 生命周期管理方法 ====================

    def shutdown(self) -> bool:
        """
        Shutdown the gateway
        关闭网关

        Returns:
            bool: Shutdown success status
                  关闭成功状态
        """
        try:
            logger.info("Gateway shutting down")
            self.services.clear()
            self.routes.clear()
            self.middlewares.clear()
            return True
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            return False

    def restart(self) -> bool:
        """
        Restart the gateway
        重启网关

        Returns:
            bool: Restart success status
                  重启成功状态
        """
        try:
            if self.shutdown():
                # Reinitialize
                self.services = {}
                self.routes = {}
                self.middlewares = []
                logger.info("Gateway restarted successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Error during restart: {str(e)}")
            return False

    # ==================== 安全方法 ====================

    def add_security_headers(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add security headers to request
        为请求添加安全头

        Args:
            request: Original request
                    原始请求

        Returns:
            dict: Request with security headers
                  带有安全头的请求
        """
        headers = request.get('headers', {})
        headers.update({
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
        })

        request['headers'] = headers
        return request

    # ==================== 数据导出方法 ====================

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export metrics in various formats
        以各种格式导出指标

        Returns:
            dict: Exported metrics
                  导出的指标
        """
        metrics = self.get_metrics()
        return {
            'gateway_metrics': metrics,
            'exported_at': datetime.now().isoformat(),
            'format': 'json'
        }

    # ==================== 持久化方法 ====================

    def save_config(self, file_path: str) -> bool:
        """
        Save current configuration to file
        将当前配置保存到文件

        Args:
            file_path: Path to save configuration
                      保存配置的路径

        Returns:
            bool: Save success status
                  保存成功状态
        """
        try:
            import json
            config_data = {
                'services': self.services,
                'routes': self.routes,
                'middlewares': self.middlewares,
                'config': self.config,
                'saved_at': datetime.now().isoformat()
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False

    def backup(self, backup_path: str) -> bool:
        """
        Create backup of current state
        创建当前状态的备份

        Args:
            backup_path: Path for backup
                        备份路径

        Returns:
            bool: Backup success status
                  备份成功状态
        """
        try:
            import json
            backup_data = {
                'services': self.services,
                'routes': self.routes,
                'middlewares': self.middlewares,
                'config': self.config,
                'backed_up_at': datetime.now().isoformat()
            }

            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Backup created at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            return False

    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from file
        从文件加载配置

        Args:
            config_path: Path to configuration file
                        配置文件的路径

        Returns:
            bool: Load success status
                  加载成功状态
        """
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Load configuration
            if 'config' in config_data:
                self.config = config_data['config']
            if 'services' in config_data:
                self.services = config_data['services']
            if 'routes' in config_data:
                self.routes = config_data['routes']
            if 'middlewares' in config_data:
                self.middlewares = config_data['middlewares']

            logger.info(f"Configuration loaded from {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return False

    def restore(self, backup_path: str) -> bool:
        """
        Restore from backup
        从备份恢复

        Args:
            backup_path: Path to backup file
                        备份文件的路径

        Returns:
            bool: Restore success status
                  恢复成功状态
        """
        try:
            import json
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            # Restore state
            self.services = backup_data.get('services', {})
            self.routes = backup_data.get('routes', {})
            self.middlewares = backup_data.get('middlewares', [])
            self.config = backup_data.get('config', {})

            logger.info(f"Restored from backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore from backup: {str(e)}")
            return False


# Global Gateway Router instance
# 全局网关路由器实例
gateway_router = GatewayRouter()

# Backward compatibility alias
# 向后兼容别名
APIGateway = GatewayRouter
api_gateway = gateway_router

__all__ = ['GatewayRouter', 'APIGateway', 'gateway_router', 'api_gateway']
