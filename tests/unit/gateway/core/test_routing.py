#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网关路由模块测试

测试目标：提升routing.py的覆盖率
"""

import pytest

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

# 动态导入网关路由模块
try:
    routing_module = importlib.import_module('src.gateway.core.routing')
    RouteRule = getattr(routing_module, 'RouteRule', None)
    if RouteRule is None:
        pytest.skip("网关路由模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("网关路由模块导入失败", allow_module_level=True)


class TestRouteRule:
    """测试路由规则"""
    
    def test_route_rule_basic(self):
        """测试基本路由规则"""
        rule = RouteRule(path="/api/data", method="GET", target="service1")
        assert rule.path == "/api/data"
        assert rule.method == "GET"
        assert rule.target == "service1"
        assert rule.middleware == []
    
    def test_route_rule_with_middleware(self):
        """测试带中间件的路由规则"""
        middleware = ["auth", "logging"]
        rule = RouteRule(path="/api/data", method="POST", target="service1", middleware=middleware)
        assert rule.path == "/api/data"
        assert rule.method == "POST"
        assert rule.target == "service1"
        assert rule.middleware == middleware
    
    def test_route_rule_default_method(self):
        """测试默认方法"""
        rule = RouteRule(path="/api/data", target="service1")
        assert rule.method == "GET"
    
    def test_route_rule_post_init(self):
        """测试__post_init__方法"""
        rule = RouteRule(path="/api/data", method="GET", target="service1")
        # middleware应该被初始化为空列表
        assert isinstance(rule.middleware, list)
        assert len(rule.middleware) == 0

