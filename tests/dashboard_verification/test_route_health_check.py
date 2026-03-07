"""
路由健康检查测试
验证路由健康检查功能是否正常工作
"""

import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.gateway.web.api import app
from src.gateway.web.route_health_check import RouteHealthChecker


class TestRouteHealthCheck:
    """路由健康检查测试类"""
    
    def test_health_checker_initialization(self):
        """测试健康检查器初始化"""
        checker = RouteHealthChecker(app)
        assert checker.app is not None
        assert len(checker.expected_routes) > 0
        assert len(checker.expected_websocket_routes) > 0
    
    def test_check_routes(self):
        """测试路由检查功能"""
        checker = RouteHealthChecker(app)
        results = checker.check_routes()
        
        assert "total_routes" in results
        assert "checked_routes" in results
        assert "missing_routes" in results
        assert "health_status" in results
        assert results["health_status"] in ["healthy", "unhealthy"]
        
        print(f"\n路由检查结果: {results['health_status']}")
        print(f"总路由数: {results['total_routes']}")
    
    def test_route_health_status(self):
        """测试路由健康状态"""
        checker = RouteHealthChecker(app)
        results = checker.check_routes()
        
        # 所有预期路由应该都已注册
        assert results["health_status"] == "healthy", \
            f"路由健康检查失败: {results.get('errors', [])}"
    
    def test_validate_routes(self):
        """测试路由验证功能"""
        checker = RouteHealthChecker(app)
        is_healthy = checker.validate_routes()
        
        assert is_healthy, "路由验证失败，存在缺失的路由"
    
    def test_expected_routes_exist(self):
        """测试预期路由是否存在"""
        checker = RouteHealthChecker(app)
        results = checker.check_routes()
        
        # 检查每个模块的路由
        for category, category_results in results["checked_routes"].items():
            assert category_results["missing"] == [], \
                f"{category}模块缺少路由: {category_results['missing']}"
            assert category_results["found"] == category_results["expected"], \
                f"{category}模块路由数量不匹配"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

