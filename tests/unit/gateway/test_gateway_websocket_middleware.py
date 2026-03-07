"""
深度测试Gateway模块WebSocket和中间件功能
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import json

# 处理导入错误
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    from gateway.web.websocket_api import WebSocketAPI
    from gateway.web.route_components import RouteManager
    from gateway.web.server_components import ServerComponents
except ImportError as e:
    pytest.skip(f"网关模块导入失败: {e}", allow_module_level=True)


class TestWebSocketAPIDeep:
    """深度测试WebSocket API"""

    def setup_method(self):
        """测试前准备"""
        self.websocket_api = WebSocketAPI()

    @pytest.mark.asyncio
    async def test_websocket_connection_handling(self):
        """测试WebSocket连接处理"""
        # 模拟WebSocket连接
        mock_websocket = AsyncMock()
        mock_websocket.receive_text = AsyncMock(return_value='{"type": "subscribe", "channels": ["trades"]}')
        mock_websocket.send_text = AsyncMock()
        mock_websocket.close = AsyncMock()

        # 处理连接
        await self.websocket_api.handle_connection(mock_websocket)

        # 验证连接已建立并发送了欢迎消息
        mock_websocket.send_text.assert_called()
        welcome_message = mock_websocket.send_text.call_args[0][0]
        welcome_data = json.loads(welcome_message)

        assert welcome_data["type"] == "connection_established"
        assert "connection_id" in welcome_data

    @pytest.mark.asyncio
    async def test_websocket_message_routing(self):
        """测试WebSocket消息路由"""
        mock_websocket = AsyncMock()
        connection_id = "test_conn_123"

        # 模拟订阅消息
        subscribe_message = {
            "type": "subscribe",
            "channels": ["market_data", "trades"],
            "symbols": ["AAPL", "GOOGL"]
        }

        await self.websocket_api.handle_message(mock_websocket, json.dumps(subscribe_message), connection_id)

        # 验证订阅已处理
        subscriptions = self.websocket_api.get_subscriptions(connection_id)
        assert "market_data" in subscriptions
        assert "trades" in subscriptions

    @pytest.mark.asyncio
    async def test_websocket_broadcasting(self):
        """测试WebSocket广播功能"""
        # 创建多个模拟连接
        mock_websockets = [AsyncMock() for _ in range(5)]
        connection_ids = [f"conn_{i}" for i in range(5)]

        # 注册连接
        for ws, conn_id in zip(mock_websockets, connection_ids):
            self.websocket_api.register_connection(ws, conn_id)
            # 订阅市场数据
            await self.websocket_api.handle_message(
                ws,
                json.dumps({"type": "subscribe", "channels": ["market_data"]}),
                conn_id
            )

        # 广播市场数据
        market_data = {
            "type": "market_data",
            "symbol": "AAPL",
            "price": 150.25,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        }

        await self.websocket_api.broadcast_to_channel("market_data", market_data)

        # 验证所有连接都收到了消息
        for ws in mock_websockets:
            ws.send_text.assert_called()
            sent_message = ws.send_text.call_args[0][0]
            sent_data = json.loads(sent_message)
            assert sent_data["type"] == "market_data"
            assert sent_data["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_websocket_connection_limits(self):
        """测试WebSocket连接限制"""
        # 设置连接限制
        self.websocket_api.max_connections = 3

        # 创建超过限制的连接
        mock_websockets = [AsyncMock() for _ in range(5)]

        successful_connections = 0
        rejected_connections = 0

        for i, ws in enumerate(mock_websockets):
            try:
                await self.websocket_api.handle_connection(ws)
                successful_connections += 1
            except Exception as e:
                if "connection limit" in str(e).lower():
                    rejected_connections += 1

        # 验证连接限制生效
        assert successful_connections <= 3
        assert rejected_connections >= 2

    @pytest.mark.asyncio
    async def test_websocket_error_handling(self):
        """测试WebSocket错误处理"""
        mock_websocket = AsyncMock()
        mock_websocket.receive_text = AsyncMock(side_effect=Exception("Connection lost"))
        mock_websocket.send_text = AsyncMock()

        # 处理有问题的连接
        with pytest.raises(Exception):
            await self.websocket_api.handle_connection(mock_websocket)

        # 验证错误已被记录和处理
        # 这里可以检查日志或错误处理逻辑

    def test_websocket_connection_cleanup(self):
        """测试WebSocket连接清理"""
        mock_websocket = AsyncMock()
        connection_id = "cleanup_test_conn"

        # 注册连接
        self.websocket_api.register_connection(mock_websocket, connection_id)

        # 验证连接已注册
        assert connection_id in self.websocket_api.connections

        # 清理连接
        self.websocket_api.cleanup_connection(connection_id)

        # 验证连接已清理
        assert connection_id not in self.websocket_api.connections


class TestRouteManagerDeep:
    """深度测试路由管理器"""

    def setup_method(self):
        """测试前准备"""
        self.route_manager = RouteManager()

    def test_complex_route_registration(self):
        """测试复杂路由注册"""
        # 注册REST API路由
        routes = [
            {
                "path": "/api/v1/trades",
                "methods": ["GET", "POST"],
                "handler": "trade_handler",
                "middleware": ["auth", "rate_limit"],
                "tags": ["trading", "api"]
            },
            {
                "path": "/api/v1/market/{symbol}",
                "methods": ["GET"],
                "handler": "market_data_handler",
                "middleware": ["cache", "cors"],
                "parameters": {
                    "symbol": {"type": "string", "pattern": r"^[A-Z]+$"}
                }
            },
            {
                "path": "/api/v1/portfolio/{user_id}/positions",
                "methods": ["GET", "PUT"],
                "handler": "portfolio_handler",
                "middleware": ["auth", "audit"],
                "parameters": {
                    "user_id": {"type": "integer", "minimum": 1}
                }
            }
        ]

        for route in routes:
            self.route_manager.register_route(**route)

        # 验证路由已注册
        assert len(self.route_manager.routes) == 3

    def test_route_matching_with_parameters(self):
        """测试带参数的路由匹配"""
        # 注册参数化路由
        self.route_manager.register_route(
            path="/api/v1/market/{symbol}/quotes",
            methods=["GET"],
            handler="quote_handler"
        )

        # 测试路由匹配
        match_result = self.route_manager.match_route("/api/v1/market/AAPL/quotes", "GET")

        assert match_result["matched"] == True
        assert match_result["handler"] == "quote_handler"
        assert match_result["parameters"]["symbol"] == "AAPL"

    def test_route_middleware_execution(self):
        """测试路由中间件执行"""
        executed_middleware = []

        # 定义模拟中间件
        def auth_middleware(request, next_handler):
            executed_middleware.append("auth")
            return next_handler(request)

        def rate_limit_middleware(request, next_handler):
            executed_middleware.append("rate_limit")
            return next_handler(request)

        def audit_middleware(request, next_handler):
            executed_middleware.append("audit")
            return next_handler(request)

        # 注册带中间件的路由
        self.route_manager.register_route(
            path="/api/v1/secure/endpoint",
            methods=["POST"],
            handler="secure_handler",
            middleware=["auth", "rate_limit", "audit"]
        )

        # 注册中间件
        self.route_manager.register_middleware("auth", auth_middleware)
        self.route_manager.register_middleware("rate_limit", rate_limit_middleware)
        self.route_manager.register_middleware("audit", audit_middleware)

        # 执行路由（模拟）
        request = {"method": "POST", "path": "/api/v1/secure/endpoint"}
        result = self.route_manager.execute_route(request)

        # 验证中间件按正确顺序执行
        assert executed_middleware == ["auth", "rate_limit", "audit"]

    def test_route_performance_monitoring(self):
        """测试路由性能监控"""
        # 注册路由
        self.route_manager.register_route(
            path="/api/v1/performance/test",
            methods=["GET"],
            handler="perf_handler"
        )

        # 模拟多次请求
        request = {"method": "GET", "path": "/api/v1/performance/test"}

        import time
        start_time = time.time()

        for _ in range(100):
            self.route_manager.match_route(request["path"], request["method"])

        end_time = time.time()

        # 获取性能指标
        performance = self.route_manager.get_route_performance("/api/v1/performance/test", "GET")

        assert "average_response_time" in performance
        assert "total_requests" in performance
        assert performance["total_requests"] == 100

    def test_route_security_validation(self):
        """测试路由安全验证"""
        # 注册需要认证的路由
        self.route_manager.register_route(
            path="/api/v1/admin/users",
            methods=["GET", "POST"],
            handler="admin_handler",
            security=["jwt_auth", "role_admin"]
        )

        # 测试安全验证
        valid_request = {
            "method": "GET",
            "path": "/api/v1/admin/users",
            "headers": {"Authorization": "Bearer valid_jwt_token"},
            "user": {"role": "admin", "user_id": 123}
        }

        invalid_request = {
            "method": "GET",
            "path": "/api/v1/admin/users",
            "headers": {},
            "user": {"role": "user", "user_id": 456}
        }

        # 验证有效请求
        security_check = self.route_manager.validate_security(valid_request)
        assert security_check["authorized"] == True

        # 验证无效请求
        security_check = self.route_manager.validate_security(invalid_request)
        assert security_check["authorized"] == False


class TestServerComponentsDeep:
    """深度测试服务器组件"""

    def setup_method(self):
        """测试前准备"""
        self.server_components = ServerComponents()

    def test_server_initialization_with_config(self):
        """测试带配置的服务器初始化"""
        server_config = {
            "host": "0.0.0.0",
            "port": 8080,
            "ssl_enabled": True,
            "max_connections": 1000,
            "timeout": 30,
            "middleware": ["cors", "compression", "security"],
            "routes": [
                {"path": "/api/v1/health", "methods": ["GET"]},
                {"path": "/api/v1/trades", "methods": ["GET", "POST"]}
            ]
        }

        server = self.server_components.create_server(server_config)

        assert server.config["host"] == "0.0.0.0"
        assert server.config["port"] == 8080
        assert server.config["ssl_enabled"] == True
        assert len(server.config["middleware"]) == 3

    def test_server_request_processing_pipeline(self):
        """测试服务器请求处理流水线"""
        # 创建服务器
        server_config = {
            "middleware": ["logging", "auth", "rate_limit", "compression"]
        }
        server = self.server_components.create_server(server_config)

        # 模拟请求
        request = {
            "method": "GET",
            "path": "/api/v1/market/AAPL",
            "headers": {"Authorization": "Bearer token123"},
            "query_params": {"format": "json"},
            "body": None
        }

        # 处理请求
        response = server.process_request(request)

        assert "status_code" in response
        assert "headers" in response
        assert "body" in response
        assert response["status_code"] in [200, 401, 429]  # 成功、未授权或限流

    def test_server_load_balancing(self):
        """测试服务器负载均衡"""
        # 创建多服务器集群
        servers_config = [
            {"id": "server_1", "host": "10.0.0.1", "port": 8080, "weight": 10},
            {"id": "server_2", "host": "10.0.0.2", "port": 8080, "weight": 8},
            {"id": "server_3", "host": "10.0.0.3", "port": 8080, "weight": 6}
        ]

        load_balancer = self.server_components.create_load_balancer(servers_config)

        # 模拟大量请求
        request_distribution = {"server_1": 0, "server_2": 0, "server_3": 0}

        for _ in range(240):  # 总权重24的倍数，确保均衡分布
            server_id = load_balancer.select_server({"method": "GET", "path": "/api/test"})
            request_distribution[server_id] += 1

        # 验证负载均衡（按权重分配）
        assert request_distribution["server_1"] == 100  # 10/24 的请求
        assert request_distribution["server_2"] == 80   # 8/24 的请求
        assert request_distribution["server_3"] == 60   # 6/24 的请求

    def test_server_health_monitoring(self):
        """测试服务器健康监控"""
        server_config = {
            "health_check_enabled": True,
            "health_check_interval": 30,
            "health_check_timeout": 5
        }

        server = self.server_components.create_server(server_config)

        # 启动健康监控
        health_monitor = self.server_components.start_health_monitoring(server)

        # 等待一段时间让监控运行
        import time
        time.sleep(0.1)

        # 获取健康状态
        health_status = health_monitor.get_health_status()

        assert "overall_health" in health_status
        assert "component_health" in health_status
        assert health_status["overall_health"] in ["healthy", "degraded", "unhealthy"]

    def test_server_graceful_shutdown(self):
        """测试服务器优雅关闭"""
        server_config = {"shutdown_timeout": 10}
        server = self.server_components.create_server(server_config)

        # 启动服务器（模拟）
        server.start()

        # 发起优雅关闭
        shutdown_result = self.server_components.graceful_shutdown(server)

        assert shutdown_result["shutdown_completed"] == True
        assert "active_connections_closed" in shutdown_result
        assert "pending_requests_completed" in shutdown_result
        assert shutdown_result["shutdown_time_seconds"] < 15  # 应该在超时时间内完成

    def test_server_configuration_hot_reload(self):
        """测试服务器配置热重载"""
        initial_config = {
            "rate_limit": {"requests_per_minute": 100},
            "cors": {"allowed_origins": ["https://example.com"]}
        }

        server = self.server_components.create_server(initial_config)

        # 验证初始配置
        assert server.config["rate_limit"]["requests_per_minute"] == 100

        # 更新配置
        updated_config = {
            "rate_limit": {"requests_per_minute": 200},
            "cors": {"allowed_origins": ["https://example.com", "https://app.example.com"]}
        }

        # 执行热重载
        reload_result = self.server_components.hot_reload_config(server, updated_config)

        assert reload_result["reload_successful"] == True
        assert server.config["rate_limit"]["requests_per_minute"] == 200
        assert len(server.config["cors"]["allowed_origins"]) == 2

    def test_server_error_handling_and_recovery(self):
        """测试服务器错误处理和恢复"""
        server_config = {
            "error_recovery_enabled": True,
            "max_error_threshold": 5,
            "recovery_strategy": "circuit_breaker"
        }

        server = self.server_components.create_server(server_config)

        # 模拟连续错误
        for i in range(7):
            try:
                # 模拟会失败的请求
                server.process_request({"method": "GET", "path": "/api/failing/endpoint"})
            except Exception:
                server.record_error()

        # 检查错误处理
        error_status = server.get_error_status()

        assert error_status["error_count"] >= 5
        assert "circuit_breaker_status" in error_status
        assert error_status["circuit_breaker_status"] in ["open", "half_open", "closed"]

        # 测试恢复机制
        if error_status["circuit_breaker_status"] == "open":
            # 等待恢复尝试
            import time
            time.sleep(0.1)

            recovery_result = server.attempt_recovery()
            assert "recovery_attempted" in recovery_result
            assert "recovery_successful" in recovery_result
