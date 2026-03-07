#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层初始化模块组件测试

测试目标：提升init_infrastructure.py的真实覆盖率
实际导入和使用src.infrastructure.init_infrastructure模块
"""

import pytest
import sys
from unittest.mock import MagicMock


# 在导入模块之前设置mock，避免导入错误
def setup_mocks():
    """设置所有需要的mock模块"""
    mock_modules = {
        'infrastructure': MagicMock(),
        'infrastructure.error': MagicMock(),
        'infrastructure.error.retry_handler': MagicMock(),
        'infrastructure.config': MagicMock(),
        'infrastructure.config.unified_manager': MagicMock(),
        'infrastructure.core': MagicMock(),
        'infrastructure.core.config': MagicMock(),
        'infrastructure.core.config.core': MagicMock(),
        'infrastructure.core.config.core.unified_manager': MagicMock(),
        'infrastructure.error.error_handler': MagicMock(),
        'infrastructure.logging': MagicMock(),
        'infrastructure.logging.log_manager': MagicMock(),
        'infrastructure.monitoring': MagicMock(),
        'infrastructure.monitoring.application_monitor': MagicMock(),
        'infrastructure.monitoring.system_monitor': MagicMock(),
        'infrastructure.resource': MagicMock(),
        'infrastructure.resource.gpu_manager': MagicMock(),
        'infrastructure.resource.resource_manager': MagicMock(),
    }
    
    for module_name, mock_module in mock_modules.items():
        if module_name not in sys.modules:
            sys.modules[module_name] = mock_module
    
    # 设置mock对象的属性
    sys.modules['infrastructure.error.retry_handler'].ResilienceManager = MagicMock
    sys.modules['infrastructure.error.retry_handler'].RetryConfig = MagicMock
    sys.modules['infrastructure.error.retry_handler'].CircuitBreakerConfig = MagicMock
    sys.modules['infrastructure.error.retry_handler'].RetryHandler = MagicMock
    sys.modules['infrastructure.config.unified_manager'].UnifiedConfigManager = MagicMock
    sys.modules['infrastructure.core.config.core.unified_manager'].UnifiedConfigManager = MagicMock
    sys.modules['infrastructure.error.error_handler'].ErrorHandler = MagicMock
    sys.modules['infrastructure.logging.log_manager'].LogManager = MagicMock
    sys.modules['infrastructure.monitoring.application_monitor'].ApplicationMonitor = MagicMock
    sys.modules['infrastructure.monitoring.system_monitor'].SystemMonitor = MagicMock
    sys.modules['infrastructure.resource.gpu_manager'].GPUManager = MagicMock
    sys.modules['infrastructure.resource.resource_manager'].ResourceManager = MagicMock

# 在模块级别设置mock
setup_mocks()


class TestConfigManager:
    """测试配置管理器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.init_infrastructure import ConfigManager
        
        manager = ConfigManager()
        
        assert manager is not None
    
    def test_get(self):
        """测试获取配置"""
        from src.infrastructure.init_infrastructure import ConfigManager
        
        manager = ConfigManager()
        result = manager.get("test_key", "default_value")
        
        assert result == "default_value"
    
    def test_get_with_default(self):
        """测试使用默认值获取配置"""
        from src.infrastructure.init_infrastructure import ConfigManager
        
        manager = ConfigManager()
        result = manager.get("nonexistent_key", "default")
        
        assert result == "default"


class TestErrorHandler:
    """测试错误处理器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.init_infrastructure import ErrorHandler
        
        handler = ErrorHandler()
        
        assert handler is not None


class TestRetryHandler:
    """测试重试处理器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.init_infrastructure import RetryHandler
        
        handler = RetryHandler()
        
        assert handler is not None


class TestResourceManager:
    """测试资源管理器类"""
    
    def test_init(self):
        """测试初始化"""
        # 直接测试类定义，避免触发Infrastructure初始化
        import sys
        if 'src.infrastructure.init_infrastructure' in sys.modules:
            del sys.modules['src.infrastructure.init_infrastructure']
        
        from src.infrastructure.init_infrastructure import ResourceManager
        
        manager = ResourceManager()
        
        assert manager is not None
    
    def test_init_with_kwargs(self):
        """测试使用关键字参数初始化"""
        # 直接测试类定义，避免触发Infrastructure初始化
        import sys
        if 'src.infrastructure.init_infrastructure' in sys.modules:
            del sys.modules['src.infrastructure.init_infrastructure']
        
        from src.infrastructure.init_infrastructure import ResourceManager
        
        manager = ResourceManager(max_workers=10, timeout=30)
        
        assert manager is not None
    
    def test_start_monitoring(self):
        """测试启动监控"""
        # 直接测试类定义，避免触发Infrastructure初始化
        import sys
        if 'src.infrastructure.init_infrastructure' in sys.modules:
            del sys.modules['src.infrastructure.init_infrastructure']
        
        from src.infrastructure.init_infrastructure import ResourceManager
        
        manager = ResourceManager()
        manager.start_monitoring()  # 应该不抛出异常
        
        assert True
    
    def test_stop_monitoring(self):
        """测试停止监控"""
        # 直接测试类定义，避免触发Infrastructure初始化
        import sys
        if 'src.infrastructure.init_infrastructure' in sys.modules:
            del sys.modules['src.infrastructure.init_infrastructure']
        
        from src.infrastructure.init_infrastructure import ResourceManager
        
        manager = ResourceManager()
        manager.stop_monitoring()  # 应该不抛出异常
        
        assert True


class TestInfrastructure:
    """测试基础设施入口类"""
    
    def test_get_infrastructure(self):
        """测试获取基础设施实例"""
        from src.infrastructure.init_infrastructure import get_infrastructure
        
        infra = get_infrastructure()
        
        assert infra is not None
    
    def test_init_infrastructure(self):
        """测试初始化基础设施"""
        from src.infrastructure.init_infrastructure import init_infrastructure
        
        infra = init_infrastructure()
        
        assert infra is not None
    
    def test_initialize_infrastructure(self):
        """测试初始化基础设施（兼容性函数）"""
        from src.infrastructure.init_infrastructure import initialize_infrastructure
        
        infra = initialize_infrastructure()
        
        assert infra is not None

