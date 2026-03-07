#!/usr/bin/env python3
"""
统一conftest.py - 基于统一测试框架

整合所有pytest配置和fixtures，使用统一测试框架管理测试环境。
"""

import sys
import os
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock
import asyncio

# 导入统一测试框架
import sys
from pathlib import Path

# 添加tests目录到路径
_tests_dir = Path(__file__).parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from framework.unified_test_framework import get_unified_framework

# 获取框架实例
framework = get_unified_framework()


# ==================== 全局配置 ====================

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """pytest配置钩子"""
    # 注册自定义标记
    markers = [
        # 测试类型标记
        "unit: 单元测试",
        "integration: 集成测试",
        "e2e: 端到端测试",
        "performance: 性能测试",

        # 层级标记
        "infrastructure: 基础设施层测试",
        "core: 核心服务层测试",
        "data: 数据管理层测试",
        "features: 特征分析层测试",
        "ml: 机器学习层测试",
        "optimization: 优化层测试",
        "strategy: 策略服务层测试",
        "trading: 交易层测试",
        "risk: 风险控制层测试",
        "monitoring: 监控层测试",
        "gateway: 网关层测试",

        # 功能标记
        "config: 配置系统测试",
        "cache: 缓存系统测试",
        "logging: 日志系统测试",
        "database: 数据库测试",
        "async: 异步测试",
        "mock: 使用Mock的测试"
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """修改测试收集项"""
    for item in items:
        # 根据测试路径自动添加层级标记
        if 'infrastructure' in str(item.fspath):
            item.add_marker(pytest.mark.infrastructure)
        elif 'core' in str(item.fspath):
            item.add_marker(pytest.mark.core)
        elif 'data' in str(item.fspath):
            item.add_marker(pytest.mark.data)
        elif 'features' in str(item.fspath):
            item.add_marker(pytest.mark.features)
        elif 'ml' in str(item.fspath):
            item.add_marker(pytest.mark.ml)
        elif 'optimization' in str(item.fspath):
            item.add_marker(pytest.mark.optimization)
        elif 'strategy' in str(item.fspath):
            item.add_marker(pytest.mark.strategy)
        elif 'trading' in str(item.fspath):
            item.add_marker(pytest.mark.trading)
        elif 'risk' in str(item.fspath):
            item.add_marker(pytest.mark.risk)
        elif 'monitoring' in str(item.fspath):
            item.add_marker(pytest.mark.monitoring)
        elif 'gateway' in str(item.fspath):
            item.add_marker(pytest.mark.gateway)


# ==================== 全局Fixtures ====================

@pytest.fixture(scope="session", autouse=True)
def setup_global_test_environment():
    """设置全局测试环境"""
    # 确保路径正确设置（由框架处理）
    yield

    # 清理工作


@pytest.fixture(scope="session")
def event_loop():
    """全局事件循环"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_framework():
    """测试框架fixture"""
    return framework


# ==================== 层级特定Fixtures ====================

@pytest.fixture(scope="session")
def infrastructure_setup():
    """基础设施层环境设置"""
    framework.setup_layer_environment('infrastructure')
    yield
    # 清理工作


@pytest.fixture(scope="session")
def core_setup():
    """核心服务层环境设置"""
    framework.setup_layer_environment('core')
    yield


@pytest.fixture(scope="session")
def data_setup():
    """数据管理层环境设置"""
    framework.setup_layer_environment('data')
    yield


@pytest.fixture(scope="session")
def ml_setup():
    """机器学习层环境设置"""
    framework.setup_layer_environment('ml')
    yield


@pytest.fixture(scope="session")
def trading_setup():
    """交易层环境设置"""
    framework.setup_layer_environment('trading')
    yield


# ==================== 通用Mock Fixtures ====================

@pytest.fixture
def mock_cache_manager():
    """Cache Manager Mock"""
    return framework.create_mock_component('cache_manager')


@pytest.fixture
def mock_event_bus():
    """Event Bus Mock"""
    return framework.create_mock_component('event_bus')


@pytest.fixture
def mock_data_manager():
    """Data Manager Mock"""
    return framework.create_mock_component('data_manager')


@pytest.fixture
def mock_logger():
    """Logger Mock"""
    mock_logger = Mock()
    mock_logger.info = Mock()
    mock_logger.error = Mock()
    mock_logger.warning = Mock()
    mock_logger.debug = Mock()
    return mock_logger


@pytest.fixture
def mock_container():
    """Dependency Container Mock"""
    return framework.create_mock_component('container')


# ==================== 工具Fixtures ====================

@pytest.fixture
def temp_dir(tmp_path):
    """临时目录fixture"""
    return tmp_path


@pytest.fixture
def sample_data():
    """示例数据fixture"""
    return {
        'stocks': ['AAPL', 'GOOGL', 'MSFT'],
        'prices': [150.0, 2800.0, 300.0],
        'volumes': [1000000, 500000, 2000000]
    }


@pytest.fixture
def mock_async_context():
    """异步上下文Mock"""
    async def async_mock(*args, **kwargs):
        return Mock()

    mock = Mock()
    mock.__aenter__ = async_mock
    mock.__aexit__ = async_mock
    return mock


# ==================== 性能测试Fixtures ====================

@pytest.fixture
def performance_timer():
    """性能计时器fixture"""
    import time

    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0

        def reset(self):
            self.start_time = None
            self.end_time = None

    return PerformanceTimer()


# ==================== 配置Fixtures ====================

@pytest.fixture
def test_config():
    """测试配置fixture"""
    return {
        'debug': True,
        'log_level': 'INFO',
        'timeout': 30,
        'max_retries': 3,
        'cache_enabled': False
    }


@pytest.fixture
def mock_config():
    """Mock配置对象"""
    config = Mock()
    config.get = Mock(return_value="test_value")
    config.set = Mock(return_value=True)
    config.save = Mock(return_value=True)
    return config


# ==================== 错误处理Fixtures ====================

@pytest.fixture
def error_handler():
    """错误处理器fixture"""
    def handle_error(error_type, error_msg):
        print(f"[{error_type}] {error_msg}")
        return True

    return handle_error


@pytest.fixture
def mock_exception():
    """Mock异常fixture"""
    class MockTestException(Exception):
        def __init__(self, message="Test exception"):
            super().__init__(message)
            self.error_code = 500

    return MockTestException


# ==================== 层级特定自动应用Fixtures ====================

@pytest.fixture(autouse=True)
def auto_setup_by_layer(request):
    """根据测试层级自动设置环境"""
    # 检查测试标记
    if hasattr(request, 'keywords'):
        if 'infrastructure' in request.keywords:
            framework.setup_layer_environment('infrastructure')
        elif 'core' in request.keywords:
            framework.setup_layer_environment('core')
        elif 'data' in request.keywords:
            framework.setup_layer_environment('data')
        elif 'ml' in request.keywords:
            framework.setup_layer_environment('ml')
        elif 'trading' in request.keywords:
            framework.setup_layer_environment('trading')


# ==================== 清理Fixtures ====================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """测试后清理"""
    yield
    # 清理工作（如果需要）


# ==================== 报告生成 ====================

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """生成自定义测试报告"""
    outcome = yield
    report = outcome.get_result()

    # 添加自定义信息到报告
    if hasattr(item, 'keywords'):
        layer_markers = [k for k in item.keywords if k in [
            'infrastructure', 'core', 'data', 'features', 'ml',
            'optimization', 'strategy', 'trading', 'risk', 'monitoring', 'gateway'
        ]]
        if layer_markers:
            report.layer = layer_markers[0]
        else:
            report.layer = 'unknown'

    return report


# ==================== 插件配置 ====================

# 启用pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


if __name__ == "__main__":
    print("统一conftest.py配置测试")
    print(f"框架实例: {framework}")
    print(f"项目根目录: {framework.project_root}")
    print("conftest.py配置验证完成")
