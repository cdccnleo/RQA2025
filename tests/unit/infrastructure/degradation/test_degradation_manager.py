import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.degradation_manager import DegradationManager, ServiceLevel, DegradationRule

@pytest.fixture
def mock_config_manager():
    """创建mock的ConfigManager"""
    mock_cm = MagicMock()
    mock_cm.get_config.return_value = {
        'services': [
            {
                'name': 'trading_engine',
                'priority': 8,
                'core': True,
                'max_level': 2
            },
            {
                'name': 'data_service',
                'priority': 5,
                'core': False,
                'max_level': 3
            }
        ],
        'rules': [
            {
                'condition': 'health.database.status == "DOWN"',
                'action': 'degrade:data_service',
                'level': 2,
                'cooldown': 60
            }
        ]
    }
    return mock_cm

@pytest.fixture
def mock_health_checker():
    """创建mock的HealthChecker"""
    mock_hc = MagicMock()
    mock_hc.get_status.return_value = {
        'database': Mock(status='UP'),
        'redis': Mock(status='UP'),
        'trading_engine': Mock(status='UP')
    }
    return mock_hc

@pytest.fixture
def mock_circuit_breaker():
    """创建mock的CircuitBreaker"""
    mock_cb = MagicMock()
    mock_cb.state = 'closed'
    return mock_cb

@pytest.fixture
def degradation_manager(mock_config_manager, mock_health_checker, mock_circuit_breaker):
    """创建降级管理器实例，使用mock的依赖"""
    manager = DegradationManager(
        config={},
        config_manager=mock_config_manager,
        health_checker=mock_health_checker,
        circuit_breaker=mock_circuit_breaker
    )
    return manager

def test_register_service(degradation_manager):
    """测试注册服务"""
    degradation_manager.register_service(
        name="test_service",
        priority=7,
        core=True,
        max_level=3
    )
    
    assert "test_service" in degradation_manager.services
    service = degradation_manager.services["test_service"]
    assert service.name == "test_service"
    assert service.priority == 7
    assert service.core is True
    assert service.max_level == 3
    assert service.current_level == 0

def test_add_rule(degradation_manager):
    """测试添加降级规则"""
    # 清空现有规则（因为配置加载时已经添加了规则）
    degradation_manager.rules.clear()
    
    degradation_manager.add_rule(
        condition="health.database.status == 'DOWN'",
        action="degrade:data_service",
        level=2,
        cooldown=60
    )
    
    assert len(degradation_manager.rules) == 1
    rule = degradation_manager.rules[0]
    assert rule.condition == "health.database.status == 'DOWN'"
    assert rule.action == "degrade:data_service"
    assert rule.level == 2
    assert rule.cooldown == 60

def test_degrade_service(degradation_manager):
    """测试降级服务"""
    # 先注册服务
    degradation_manager.register_service("test_service", 5, False, 3)
    
    # 降级服务
    result = degradation_manager.degrade_service("test_service", 2)
    assert result is True
    
    # 验证降级状态
    assert degradation_manager.get_service_level("test_service") == 2
    assert degradation_manager.is_degraded("test_service") is True

def test_restore_service(degradation_manager):
    """测试恢复服务"""
    # 先注册并降级服务
    degradation_manager.register_service("test_service", 5, False, 3)
    degradation_manager.degrade_service("test_service", 2)
    
    # 恢复服务
    result = degradation_manager.restore_service("test_service")
    assert result is True
    
    # 验证恢复状态
    assert degradation_manager.get_service_level("test_service") == 0
    assert degradation_manager.is_degraded("test_service") is False

def test_degrade_service_not_found(degradation_manager):
    """测试降级不存在的服务"""
    result = degradation_manager.degrade_service("unknown_service", 1)
    assert result is False

def test_degrade_service_exceed_max_level(degradation_manager):
    """测试降级级别超过最大限制"""
    degradation_manager.register_service("test_service", 5, False, 2)
    
    result = degradation_manager.degrade_service("test_service", 3)
    assert result is False

def test_get_service_level_not_found(degradation_manager):
    """测试获取不存在服务的级别"""
    level = degradation_manager.get_service_level("unknown_service")
    assert level is None

def test_start_stop_monitoring(degradation_manager):
    """测试启动和停止监控"""
    # 启动监控
    degradation_manager.start()
    assert degradation_manager.running is True
    
    # 停止监控
    degradation_manager.stop()
    assert degradation_manager.running is False

def test_get_status_report(degradation_manager):
    """测试获取状态报告"""
    # 注册服务
    degradation_manager.register_service("test_service", 5, False, 3)
    degradation_manager.degrade_service("test_service", 2)
    
    # 获取报告
    report = degradation_manager.get_status_report()
    
    assert "timestamp" in report
    assert "services" in report
    assert "degraded_services" in report
    assert "rules" in report
    
    # 验证服务信息（包括配置加载的服务）
    services = report["services"]
    assert len(services) >= 1  # 至少有我们添加的服务
    test_service = next((s for s in services if s["name"] == "test_service"), None)
    assert test_service is not None
    assert test_service["current_level"] == 2
    
    # 验证降级服务
    degraded = report["degraded_services"]
    test_degraded = next((s for s in degraded if s["name"] == "test_service"), None)
    assert test_degraded is not None

def test_force_degrade_all(degradation_manager):
    """测试强制降级所有服务"""
    # 注册多个服务
    degradation_manager.register_service("service1", 5, False, 3)
    degradation_manager.register_service("service2", 7, True, 2)
    
    # 强制降级
    degradation_manager.force_degrade_all(2)
    
    # 验证降级状态
    assert degradation_manager.get_service_level("service1") == 2
    assert degradation_manager.get_service_level("service2") == 2

def test_force_restore_all(degradation_manager):
    """测试强制恢复所有服务"""
    # 注册并降级服务
    degradation_manager.register_service("service1", 5, False, 3)
    degradation_manager.degrade_service("service1", 2)
    
    # 强制恢复
    degradation_manager.force_restore_all()
    
    # 验证恢复状态
    assert degradation_manager.get_service_level("service1") == 0
    assert degradation_manager.is_degraded("service1") is False

def test_load_config_from_manager(degradation_manager, mock_config_manager):
    """测试从配置管理器加载配置"""
    # 清空现有数据
    degradation_manager.services.clear()
    degradation_manager.rules.clear()
    
    # 重新加载配置
    degradation_manager._load_config()
    
    # 验证服务被注册
    assert "trading_engine" in degradation_manager.services
    assert "data_service" in degradation_manager.services
    
    # 验证规则被添加
    assert len(degradation_manager.rules) == 1
    rule = degradation_manager.rules[0]
    assert rule.condition == 'health.database.status == "DOWN"'
    assert rule.action == 'degrade:data_service' 