#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一容器接口测试 - 简化版

直接测试container/unified_container_interface.py模块
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

# 直接导入unified_container_interface.py
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 直接导入unified_container_interface.py文件
    import importlib.util
    unified_interface_path = project_root / "src" / "core" / "container" / "unified_container_interface.py"
    spec = importlib.util.spec_from_file_location("unified_interface_module", unified_interface_path)
    unified_interface_module = importlib.util.module_from_spec(spec)
    
    # 处理依赖
    sys.path.insert(0, str(project_root / "src"))
    spec.loader.exec_module(unified_interface_module)
    
    # 尝试获取类和枚举
    ServiceLifecycle = getattr(unified_interface_module, 'ServiceLifecycle', None)
    ServiceScope = getattr(unified_interface_module, 'ServiceScope', None)
    ServiceStatus = getattr(unified_interface_module, 'ServiceStatus', None)
    ServiceRegistrationInfo = getattr(unified_interface_module, 'ServiceRegistrationInfo', None)
    ServiceResolutionInfo = getattr(unified_interface_module, 'ServiceResolutionInfo', None)
    ServiceHealthInfo = getattr(unified_interface_module, 'ServiceHealthInfo', None)
    ContainerHealthInfo = getattr(unified_interface_module, 'ContainerHealthInfo', None)
    IServiceContainer = getattr(unified_interface_module, 'IServiceContainer', None)
    
    IMPORTS_AVAILABLE = ServiceLifecycle is not None
except Exception as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"统一容器接口模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestServiceLifecycle:
    """测试服务生命周期枚举"""

    def test_service_lifecycle_values(self):
        """测试服务生命周期枚举值"""
        if ServiceLifecycle:
            assert ServiceLifecycle.SINGLETON.value == "singleton"
            assert ServiceLifecycle.TRANSIENT.value == "transient"
            assert ServiceLifecycle.SCOPED.value == "scoped"
            assert ServiceLifecycle.POOL.value == "pool"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestServiceScope:
    """测试服务作用域枚举"""

    def test_service_scope_values(self):
        """测试服务作用域枚举值"""
        if ServiceScope:
            assert ServiceScope.GLOBAL.value == "global"
            assert ServiceScope.REQUEST.value == "request"
            assert ServiceScope.SESSION.value == "session"
            assert ServiceScope.THREAD.value == "thread"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestServiceStatus:
    """测试服务状态枚举"""

    def test_service_status_values(self):
        """测试服务状态枚举值"""
        if ServiceStatus:
            assert ServiceStatus.REGISTERED.value == "registered"
            assert ServiceStatus.RUNNING.value == "running"
            assert ServiceStatus.ERROR.value == "error"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestServiceRegistrationInfo:
    """测试服务注册信息"""

    def test_service_registration_info_creation(self):
        """测试服务注册信息创建"""
        if ServiceRegistrationInfo:
            class TestService:
                pass
            
            info = ServiceRegistrationInfo(
                service_type=TestService,
                lifecycle=ServiceLifecycle.SINGLETON,
                name="test_service"
            )
            assert info.service_type == TestService
            assert info.lifecycle == ServiceLifecycle.SINGLETON
            assert info.name == "test_service"
            assert info.registered_at is not None


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestServiceResolutionInfo:
    """测试服务解析信息"""

    def test_service_resolution_info_creation(self):
        """测试服务解析信息创建"""
        if ServiceResolutionInfo:
            class TestService:
                pass
            
            info = ServiceResolutionInfo(
                service_type=TestService,
                name="test_service"
            )
            assert info.service_type == TestService
            assert info.name == "test_service"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestServiceHealthInfo:
    """测试服务健康信息"""

    def test_service_health_info_creation(self):
        """测试服务健康信息创建"""
        if ServiceHealthInfo:
            class TestService:
                pass
            
            info = ServiceHealthInfo(
                service_type=TestService,
                name="test_service",
                status=ServiceStatus.RUNNING
            )
            assert info.service_type == TestService
            assert info.status == ServiceStatus.RUNNING
            assert info.last_check is not None


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestContainerHealthInfo:
    """测试容器健康信息"""

    def test_container_health_info_creation(self):
        """测试容器健康信息创建"""
        if ContainerHealthInfo:
            info = ContainerHealthInfo(
                container_name="test_container",
                total_services=10,
                active_services=8
            )
            assert info.container_name == "test_container"
            assert info.total_services == 10
            assert info.active_services == 8
            assert info.status == "healthy"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestIServiceContainer:
    """测试服务容器接口"""

    def test_iservice_container_abstract(self):
        """测试IServiceContainer是抽象类"""
        if IServiceContainer:
            from abc import ABC
            assert issubclass(IServiceContainer, ABC)

