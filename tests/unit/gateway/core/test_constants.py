#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网关层常量测试

测试目标：提升constants.py的覆盖率
"""

import sys
import importlib
from pathlib import Path
import pytest

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入网关常量模块
try:
    constants = importlib.import_module('src.gateway.core.constants')
    if constants is None:
        pytest.skip("网关常量模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("网关常量模块导入失败", allow_module_level=True)


class TestGatewayConstants:
    """测试网关层常量"""
    
    def test_http_status_codes(self):
        """测试HTTP状态码常量"""
        assert constants.HTTP_OK == 200
        assert constants.HTTP_CREATED == 201
        assert constants.HTTP_BAD_REQUEST == 400
        assert constants.HTTP_UNAUTHORIZED == 401
        assert constants.HTTP_FORBIDDEN == 403
        assert constants.HTTP_NOT_FOUND == 404
        assert constants.HTTP_METHOD_NOT_ALLOWED == 405
        assert constants.HTTP_TOO_MANY_REQUESTS == 429
        assert constants.HTTP_INTERNAL_SERVER_ERROR == 500
    
    def test_request_processing_parameters(self):
        """测试请求处理参数常量"""
        assert constants.DEFAULT_REQUEST_TIMEOUT == 30
        assert constants.MAX_REQUEST_SIZE_BYTES == 10485760
        assert constants.MAX_REQUESTS_PER_MINUTE == 1000
        assert constants.MAX_CONCURRENT_REQUESTS == 100
    
    def test_routing_parameters(self):
        """测试路由参数常量"""
        assert constants.DEFAULT_ROUTE_TIMEOUT == 10
        assert constants.MAX_ROUTE_DEPTH == 10
        assert constants.ROUTE_CACHE_SIZE == 1000
    
    def test_load_balancing_parameters(self):
        """测试负载均衡参数常量"""
        assert constants.DEFAULT_LOAD_BALANCER_TIMEOUT == 5
        assert constants.HEALTH_CHECK_INTERVAL == 30
        assert constants.MAX_FAILED_REQUESTS == 3
    
    def test_security_parameters(self):
        """测试安全参数常量"""
        assert constants.DEFAULT_AUTH_TIMEOUT == 5
        assert constants.TOKEN_EXPIRY_SECONDS == 3600
        assert constants.RATE_LIMIT_WINDOW_SECONDS == 60
    
    def test_caching_parameters(self):
        """测试缓存参数常量"""
        assert constants.RESPONSE_CACHE_SIZE == 10000
        assert constants.CACHE_TTL_SECONDS == 300
        assert constants.CACHE_CLEANUP_INTERVAL == 60
    
    def test_logging_parameters(self):
        """测试日志参数常量"""
        assert constants.LOG_RETENTION_DAYS == 30
        assert constants.MAX_LOG_FILE_SIZE_MB == 100
        assert constants.LOG_ROTATION_COUNT == 10
    
    def test_monitoring_parameters(self):
        """测试监控参数常量"""
        assert constants.METRICS_UPDATE_INTERVAL == 10
        assert constants.ALERT_THRESHOLD_HIGH == 0.9
        assert constants.ALERT_THRESHOLD_MEDIUM == 0.7
    
    def test_websocket_parameters(self):
        """测试WebSocket参数常量"""
        assert constants.WS_CONNECTION_TIMEOUT == 30
        assert constants.WS_MESSAGE_SIZE_LIMIT == 65536
        assert constants.WS_HEARTBEAT_INTERVAL == 30
    
    def test_api_version_parameters(self):
        """测试API版本参数常量"""
        assert constants.DEFAULT_API_VERSION == "v1"
        assert constants.API_VERSION_HEADER == "X-API-Version"
        assert constants.SUPPORTED_API_VERSIONS == ["v1", "v2"]
    
    def test_performance_parameters(self):
        """测试性能参数常量"""
        assert constants.RESPONSE_TIME_TARGET_MS == 200
        assert constants.THROUGHPUT_TARGET_RPS == 1000
        assert constants.ERROR_RATE_TARGET_PCT == 1.0
    
    def test_resource_limits(self):
        """测试资源限制常量"""
        assert constants.MEMORY_USAGE_THRESHOLD_PCT == 80
        assert constants.CPU_USAGE_THRESHOLD_PCT == 70
        assert constants.DISK_USAGE_THRESHOLD_PCT == 85

