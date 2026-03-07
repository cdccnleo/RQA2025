"""
订单路由路由测试

测试订单智能路由监控API的各项功能，包括：
- 权限控制验证
- 数据脱敏验证
- 告警通知机制
- 审计日志记录
- 详情查询功能
- 筛选功能
- WebSocket连接

量化交易系统合规要求：
- QTS-015: 权限控制
- QTS-016: 操作日志
- QTS-017: 数据脱敏
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# 被测试模块
from src.gateway.web.order_routing_routes import router, _mask_decision_data, _mask_id
from src.gateway.web.auth_middleware import Permission, Role, User
from src.gateway.web.audit_logger import AuditCategory


# ==================== Fixtures ====================

@pytest.fixture
def mock_user():
    """创建测试用户"""
    return User(
        user_id="test_trader",
        username="test_trader",
        roles=[Role.TRADER],
        is_active=True
    )


@pytest.fixture
def mock_admin_user():
    """创建管理员测试用户"""
    return User(
        user_id="admin",
        username="admin",
        roles=[Role.ADMIN],
        is_active=True
    )


@pytest.fixture
def sample_routing_decisions():
    """示例路由决策数据"""
    return [
        {
            "order_id": "ORD_123456789",
            "decision_id": "DEC_987654321",
            "routing_strategy": "smart_route",
            "target_route": "exchange_a",
            "cost": 0.001,
            "latency": 15,
            "status": "success",
            "timestamp": int(datetime.now().timestamp()),
            "strategy_id": "strategy_001"
        },
        {
            "order_id": "ORD_987654321",
            "decision_id": "DEC_123456789",
            "routing_strategy": "cost_optimal",
            "target_route": "exchange_b",
            "cost": 0.002,
            "latency": 25,
            "status": "failed",
            "timestamp": int((datetime.now() - timedelta(hours=1)).timestamp()),
            "failure_reason": "交易所连接超时",
            "strategy_id": "strategy_002"
        }
    ]


@pytest.fixture
def mock_auth_headers():
    """模拟认证请求头"""
    return {"Authorization": "Bearer test_token"}


# ==================== 权限控制测试 ====================

class TestPermissionControl:
    """权限控制测试类"""
    
    def test_trading_view_permission_exists(self):
        """测试 TRADING_VIEW 权限已定义"""
        assert hasattr(Permission, 'TRADING_VIEW')
        assert Permission.TRADING_VIEW.value == "trading:view"
    
    def test_trader_has_trading_view_permission(self, mock_user):
        """测试交易员角色拥有 TRADING_VIEW 权限"""
        assert mock_user.has_permission(Permission.TRADING_VIEW)
    
    def test_admin_has_all_trading_permissions(self, mock_admin_user):
        """测试管理员拥有所有交易权限"""
        assert mock_admin_user.has_permission(Permission.TRADING_VIEW)
        assert mock_admin_user.has_permission(Permission.TRADING_EXECUTE)
        assert mock_admin_user.has_permission(Permission.ORDER_MANAGE)


# ==================== 数据脱敏测试 ====================

class TestDataMasking:
    """数据脱敏测试类"""
    
    def test_mask_order_id(self, sample_routing_decisions):
        """测试订单ID脱敏"""
        masked = _mask_decision_data(sample_routing_decisions)
        
        # 验证订单ID已被脱敏
        assert masked[0]['order_id'] != sample_routing_decisions[0]['order_id']
        assert masked[0]['order_id'].startswith('ORD***')
        
    def test_mask_decision_id(self, sample_routing_decisions):
        """测试决策ID脱敏"""
        masked = _mask_decision_data(sample_routing_decisions)
        
        # 验证决策ID已被脱敏
        assert masked[0]['decision_id'] != sample_routing_decisions[0]['decision_id']
        assert masked[0]['decision_id'].startswith('DEC***')
    
    def test_other_fields_preserved(self, sample_routing_decisions):
        """测试其他字段保持原样"""
        masked = _mask_decision_data(sample_routing_decisions)
        
        # 验证非敏感字段未被修改
        assert masked[0]['routing_strategy'] == sample_routing_decisions[0]['routing_strategy']
        assert masked[0]['target_route'] == sample_routing_decisions[0]['target_route']
        assert masked[0]['cost'] == sample_routing_decisions[0]['cost']
        assert masked[0]['latency'] == sample_routing_decisions[0]['latency']
        assert masked[0]['status'] == sample_routing_decisions[0]['status']


# ==================== 服务层函数测试 ====================

class TestServiceFunctions:
    """服务层函数测试类"""
    
    @patch('src.gateway.web.order_routing_service.get_routing_decisions')
    def test_get_routing_decision_detail_found(self, mock_get_decisions, sample_routing_decisions):
        """测试获取存在的决策详情"""
        from src.gateway.web.order_routing_service import get_routing_decision_detail
        
        mock_get_decisions.return_value = sample_routing_decisions
        
        detail = get_routing_decision_detail("ORD_123456789")
        
        assert detail is not None
        assert detail['order_id'] == "ORD_123456789"
        assert 'query_time' in detail
        assert detail['detail_level'] == 'full'
    
    @patch('src.gateway.web.order_routing_service.get_routing_decisions')
    def test_get_routing_decision_detail_not_found(self, mock_get_decisions, sample_routing_decisions):
        """测试获取不存在的决策详情"""
        from src.gateway.web.order_routing_service import get_routing_decision_detail
        
        mock_get_decisions.return_value = sample_routing_decisions
        
        detail = get_routing_decision_detail("NON_EXISTENT_ID")
        
        assert detail is None
    
    @patch('src.gateway.web.order_routing_service.get_routing_decisions')
    def test_get_filtered_routing_decisions_by_status(self, mock_get_decisions, sample_routing_decisions):
        """测试按状态筛选路由决策"""
        from src.gateway.web.order_routing_service import get_filtered_routing_decisions
        
        mock_get_decisions.return_value = sample_routing_decisions
        
        filtered = get_filtered_routing_decisions(status="success")
        
        assert len(filtered) == 1
        assert filtered[0]['status'] == "success"
    
    @patch('src.gateway.web.order_routing_service.get_routing_decisions')
    def test_get_filtered_routing_decisions_by_strategy(self, mock_get_decisions, sample_routing_decisions):
        """测试按策略ID筛选路由决策"""
        from src.gateway.web.order_routing_service import get_filtered_routing_decisions
        
        mock_get_decisions.return_value = sample_routing_decisions
        
        filtered = get_filtered_routing_decisions(strategy_id="strategy_001")
        
        assert len(filtered) == 1
        assert filtered[0]['strategy_id'] == "strategy_001"
    
    @patch('src.gateway.web.order_routing_service.get_routing_decisions')
    def test_get_filtered_routing_decisions_limit(self, mock_get_decisions, sample_routing_decisions):
        """测试返回数量限制"""
        from src.gateway.web.order_routing_service import get_filtered_routing_decisions
        
        # 创建大量测试数据
        many_decisions = sample_routing_decisions * 100
        mock_get_decisions.return_value = many_decisions
        
        filtered = get_filtered_routing_decisions(limit=50)
        
        assert len(filtered) == 50


# ==================== 审计日志类别测试 ====================

class TestAuditCategory:
    """审计日志类别测试类"""
    
    def test_trading_category_exists(self):
        """测试 TRADING 审计类别已定义"""
        assert hasattr(AuditCategory, 'TRADING')
        assert AuditCategory.TRADING.value == "trading"
    
    def test_alert_category_exists(self):
        """测试 ALERT 审计类别已定义"""
        assert hasattr(AuditCategory, 'ALERT')
        assert AuditCategory.ALERT.value == "alert"


# ==================== API端点测试 ====================

class TestAPIEndpoints:
    """API端点测试类"""
    
    def test_routing_endpoints_defined(self):
        """测试路由端点已正确定义"""
        # 验证路由已包含在router中
        routes = [route.path for route in router.routes]
        
        # 检查主要端点是否存在
        assert "/trading/routing/decisions" in routes or any("decisions" in r for r in routes)
        assert "/trading/routing/stats" in routes or any("stats" in r for r in routes)
        assert "/trading/routing/performance" in routes or any("performance" in r for r in routes)


# ==================== WebSocket测试 ====================

class TestWebSocket:
    """WebSocket测试类"""
    
    def test_websocket_endpoint_exists(self):
        """测试WebSocket端点已定义"""
        # WebSocket端点已在 websocket_routes.py 中定义
        from src.gateway.web.websocket_routes import websocket_order_routing
        assert websocket_order_routing is not None


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试类"""
    
    def test_all_required_permissions_defined(self):
        """测试所有必需的权限已定义"""
        required_permissions = [
            Permission.TRADING_VIEW,
            Permission.TRADING_EXECUTE,
            Permission.ORDER_MANAGE,
            Permission.ALERT_VIEW,
            Permission.ALERT_ACKNOWLEDGE
        ]
        
        for perm in required_permissions:
            assert perm is not None
    
    def test_all_required_audit_categories_defined(self):
        """测试所有必需的审计类别已定义"""
        required_categories = [
            AuditCategory.TRADING,
            AuditCategory.ALERT,
            AuditCategory.USER_ACTION
        ]
        
        for cat in required_categories:
            assert cat is not None


# ==================== 性能测试 ====================

class TestPerformance:
    """性能测试类"""
    
    def test_mask_decision_data_performance(self):
        """测试数据脱敏性能"""
        import time
        
        # 创建大量测试数据
        large_decisions = [
            {
                "order_id": f"ORD_{i}",
                "decision_id": f"DEC_{i}",
                "routing_strategy": "smart_route",
                "target_route": "exchange_a",
                "cost": 0.001,
                "latency": 15,
                "status": "success",
                "timestamp": int(datetime.now().timestamp())
            }
            for i in range(1000)
        ]
        
        start_time = time.time()
        masked = _mask_decision_data(large_decisions)
        end_time = time.time()
        
        # 验证性能 - 1000条记录应在100ms内完成
        assert (end_time - start_time) < 0.1
        assert len(masked) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
