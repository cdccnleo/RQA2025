#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一测试框架配置 - RQA2025量化交易系统

整合所有pytest配置，标准化测试环境和Mock管理
创建时间: 2025-12-04
"""

import asyncio
import sys
import os
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock, patch
import logging

# 配置日志
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)

# 统一路径配置
_project_root = Path(__file__).resolve().parent.parent
_src_path = _project_root / "src"

# 确保路径正确设置
for path in [_project_root, _src_path]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# 导入统一Mock管理器
try:
    from tests.mock_manager import UnifiedMockManager
    mock_manager = UnifiedMockManager()
except ImportError:
    mock_manager = None


# ==================== pytest配置 ====================

def pytest_configure(config):
    """pytest配置钩子 - 注册统一标记"""

    # 测试类型标记
    config.addinivalue_line("markers", "unit: 单元测试 - 测试单个函数或方法")
    config.addinivalue_line("markers", "integration: 集成测试 - 测试模块间协作")
    config.addinivalue_line("markers", "e2e: 端到端测试 - 完整业务流程测试")
    config.addinivalue_line("markers", "performance: 性能测试 - 性能基准和压力场景")

    # 层级标记
    config.addinivalue_line("markers", "infrastructure: 基础设施层测试")
    config.addinivalue_line("markers", "core: 核心服务层测试")
    config.addinivalue_line("markers", "data: 数据管理层测试")
    config.addinivalue_line("markers", "features: 特征分析层测试")
    config.addinivalue_line("markers", "strategy: 策略服务层测试")
    config.addinivalue_line("markers", "trading: 交易层测试")
    config.addinivalue_line("markers", "risk: 风险控制层测试")
    config.addinivalue_line("markers", "monitoring: 监控层测试")
    config.addinivalue_line("markers", "ml: 机器学习层测试")

    # 特殊条件标记
    config.addinivalue_line("markers", "boundary: 边界条件测试 - 极端输入和边界情况")
    config.addinivalue_line("markers", "concurrent: 并发测试 - 多线程与竞争场景")
    config.addinivalue_line("markers", "error: 错误处理测试 - 异常恢复与降级")
    config.addinivalue_line("markers", "slow: 慢速测试 - 执行时间较长")
    config.addinivalue_line("markers", "flaky: 不稳定测试 - 仍需修复，默认跳过")
    config.addinivalue_line("markers", "smoke: 冒烟测试 - 快速回归验证")

    # 风险标记
    config.addinivalue_line("markers", "deadlock_risk: 可能存在死锁风险")
    config.addinivalue_line("markers", "infinite_loop_risk: 可能存在无限循环风险")

    # 业务功能标记
    config.addinivalue_line("markers", "cache: 缓存系统测试")
    config.addinivalue_line("markers", "config: 配置系统测试")
    config.addinivalue_line("markers", "logging: 日志系统测试")
    config.addinivalue_line("markers", "security: 安全相关测试")
    config.addinivalue_line("markers", "protocol: 协议接口测试")


def pytest_collection_modifyitems(config, items):
    """修改测试收集结果 - 智能标记和统计"""

    # 统计测试分布
    total_tests = len(items)
    print(f"\n📊 测试收集完成: 发现 {total_tests} 个测试")

    # 按层级统计
    layer_stats = {}
    for item in items:
        test_path = str(item.fspath)
        if 'unit/' in test_path:
            parts = test_path.split('unit/')
            if len(parts) > 1:
                layer = parts[1].split('/')[0]
                layer_stats[layer] = layer_stats.get(layer, 0) + 1

    if layer_stats:
        print("🏗️ 层级分布:")
        for layer, count in sorted(layer_stats.items()):
            print(f"  {layer}: {count} 个测试")

    # 智能标记测试
    for item in items:
        # 根据路径自动添加层级标记
        test_path = str(item.fspath)
        if 'unit/infrastructure' in test_path:
            item.add_marker(pytest.mark.infrastructure)
        elif 'unit/core' in test_path:
            item.add_marker(pytest.mark.core)
        elif 'unit/data' in test_path:
            item.add_marker(pytest.mark.data)
        elif 'unit/features' in test_path:
            item.add_marker(pytest.mark.features)
        elif 'unit/strategy' in test_path:
            item.add_marker(pytest.mark.strategy)
        elif 'unit/trading' in test_path:
            item.add_marker(pytest.mark.trading)
        elif 'unit/risk' in test_path:
            item.add_marker(pytest.mark.risk)
        elif 'unit/monitoring' in test_path:
            item.add_marker(pytest.mark.monitoring)


# ==================== 标准fixtures ====================

@pytest.fixture(scope="session")
def event_loop():
    """全局事件循环fixture"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def project_root():
    """项目根目录fixture"""
    return _project_root


@pytest.fixture(scope="session")
def src_path():
    """源码目录fixture"""
    return _src_path


# ==================== Mock fixtures ====================

@pytest.fixture(scope="session")
def mock_manager():
    """统一Mock管理器fixture"""
    return mock_manager


@pytest.fixture
def standard_mock(mock_manager):
    """标准Mock构建器fixture"""
    if mock_manager:
        return mock_manager.create_standard_mock()
    return Mock()


@pytest.fixture
def infrastructure_mock(mock_manager):
    """基础设施Mock fixture"""
    if mock_manager:
        return mock_manager.create_infrastructure_mock()
    return Mock()


@pytest.fixture
def data_mock(mock_manager):
    """数据层Mock fixture"""
    if mock_manager:
        return mock_manager.create_data_mock()
    return Mock()


@pytest.fixture
def strategy_mock(mock_manager):
    """策略层Mock fixture"""
    if mock_manager:
        return mock_manager.create_strategy_mock()
    return Mock()


@pytest.fixture
def risk_mock(mock_manager):
    """风险控制Mock fixture"""
    if mock_manager:
        return mock_manager.create_risk_mock()
    return Mock()


# ==================== 业务对象fixtures ====================

@pytest.fixture
def sample_portfolio():
    """示例投资组合fixture"""
    return {
        "AAPL": {"quantity": 100, "price": 150.0, "weight": 0.4},
        "GOOGL": {"quantity": 50, "price": 2800.0, "weight": 0.35},
        "MSFT": {"quantity": 80, "price": 300.0, "weight": 0.25}
    }


@pytest.fixture
def sample_market_data():
    """示例市场数据fixture"""
    return {
        "symbol": "AAPL",
        "price": 150.0,
        "volume": 1000000,
        "high": 152.0,
        "low": 148.0,
        "timestamp": "2023-12-04 14:30:00"
    }


@pytest.fixture
def sample_strategy_config():
    """示例策略配置fixture"""
    return {
        "strategy_id": "test_strategy",
        "name": "Test Strategy",
        "type": "momentum",
        "parameters": {
            "lookback_period": 20,
            "threshold": 0.02
        },
        "risk_limits": {
            "max_position": 100000,
            "stop_loss": 0.05
        }
    }


# ==================== 测试环境fixtures ====================

@pytest.fixture(autouse=True)
def setup_test_environment():
    """自动设置测试环境"""
    # 这里可以添加通用的测试环境设置
    yield
    # 清理逻辑可以放在这里


@pytest.fixture
def isolated_test_env(tmp_path):
    """隔离测试环境fixture"""
    # 创建临时测试目录
    test_dir = tmp_path / "test_env"
    test_dir.mkdir()

    # 设置环境变量
    original_env = dict(os.environ)
    os.environ["TEST_ENV"] = "true"
    os.environ["TEST_DATA_DIR"] = str(test_dir)

    yield test_dir

    # 恢复环境变量
    os.environ.clear()
    os.environ.update(original_env)


# ==================== 工具fixtures ====================

@pytest.fixture
def performance_timer():
    """性能计时器fixture"""
    import time
    start_time = time.time()

    class Timer:
        def elapsed(self):
            return time.time() - start_time

        def reset(self):
            nonlocal start_time
            start_time = time.time()

    return Timer()


@pytest.fixture
def memory_monitor():
    """内存监控fixture"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    class MemoryMonitor:
        def current_usage(self):
            return process.memory_info().rss

        def memory_delta(self):
            return self.current_usage() - initial_memory

    return MemoryMonitor()


# ==================== 配置fixtures ====================

@pytest.fixture
def test_config():
    """测试配置fixture"""
    return {
        "log_level": "WARNING",
        "test_timeout": 30,
        "parallel_workers": 4,
        "coverage_target": 80.0
    }


@pytest.fixture
def database_config():
    """数据库配置fixture（测试用）"""
    return {
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_pass"
    }


@pytest.fixture
def redis_config():
    """Redis配置fixture（测试用）"""
    return {
        "host": "localhost",
        "port": 6379,
        "db": 1,
        "password": None
    }


# ==================== 导入助手 ====================

def safe_import(module_name, fallback=None):
    """安全导入函数"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        if fallback:
            print(f"⚠️ 模块 {module_name} 导入失败，使用降级方案")
        return False


# 初始化时检查关键模块
@pytest.fixture(scope="session", autouse=True)
def check_dependencies():
    """检查依赖模块可用性"""
    critical_modules = [
        "pandas",
        "numpy",
        "pytest",
        "unittest.mock"
    ]

    missing_modules = []
    for module in critical_modules:
        if not safe_import(module):
            missing_modules.append(module)

    if missing_modules:
        pytest.skip(f"缺少关键依赖模块: {missing_modules}")


# ==================== 自定义断言 ====================

def assert_dict_contains_subset(subset, superset, msg=None):
    """断言字典包含子集"""
    for key, value in subset.items():
        assert key in superset, f"Key '{key}' not found in superset"
        assert superset[key] == value, f"Value mismatch for key '{key}': expected {value}, got {superset[key]}"


def assert_list_contains_elements(lst, elements, msg=None):
    """断言列表包含指定元素"""
    for element in elements:
        assert element in lst, f"Element '{element}' not found in list"


# 注册自定义断言到pytest
pytest.register_assert_rewrite(__file__)


# ==================== 报告增强 ====================

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """增强终端报告"""
    # 这里可以添加自定义的测试总结报告
    pass


def pytest_sessionfinish(session, exitstatus):
    """测试会话结束时的处理"""
    # 这里可以添加会话结束时的清理逻辑
    pass
