"""
全局pytest配置文件 - 简化版本

配置pytest-asyncio和其他全局fixtures
创建日期: 2025-12-03
目的: 提供简化的测试配置，避免复杂的导入钩子
"""

import asyncio
import sys
from pathlib import Path
import logging

# 配置日志 - 只显示警告和错误
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# 在导入pytest之前配置路径（最高优先级）
# 从conftest.py的位置（tests目录）向上查找项目根目录
_conftest_file = Path(__file__).resolve()
_project_root = _conftest_file.parent.parent
_project_root_str = str(_project_root)
_src_path_str = str(_project_root / "src")

# 确保项目根目录和src目录在sys.path的最前面
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)
if _src_path_str not in sys.path:
    sys.path.insert(0, _src_path_str)

# 额外确保data.adapters包可以被发现
_data_adapters_path = str(_project_root / "src" / "data" / "adapters")
if _data_adapters_path not in sys.path:
    sys.path.insert(0, _data_adapters_path)

import pytest
from unittest.mock import Mock

# 启用pytest-asyncio插件
pytest_plugins = ('pytest_asyncio',)


def pytest_configure(config):
    """
    pytest配置钩子 - 注册自定义markers并确保包路径
    """
    # 再次确保路径设置（防止pytest覆盖）
    _project_root = Path(__file__).resolve().parent.parent
    _project_root_str = str(_project_root)
    _src_path_str = str(_project_root / "src")

    if _project_root_str not in sys.path:
        sys.path.insert(0, _project_root_str)
    if _src_path_str not in sys.path:
        sys.path.insert(0, _src_path_str)

    # 强制重新导入相关模块
    import importlib
    try:
        if 'src.data.adapters' in sys.modules:
            importlib.reload(sys.modules['src.data.adapters'])
    except:
        pass

    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "boundary: marks tests as boundary condition tests")
    config.addinivalue_line("markers", "concurrent: marks tests as concurrent tests")
    config.addinivalue_line("markers", "error: marks tests as error handling tests")
    config.addinivalue_line("markers", "flaky: marks tests as potentially flaky")
    config.addinivalue_line("markers", "timeout: marks tests that use pytest-timeout")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")


def pytest_collection_modifyitems(config, items):
    """
    修改测试收集结果 - 智能跳过有问题的模块
    """
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


def pytest_runtest_setup(item):
    """
    测试运行前设置 - 处理导入失败的优雅降级
    """
    # 这里可以添加测试前的准备逻辑
    pass


def pytest_runtest_makereport(item, call):
    """
    生成测试报告 - 添加自定义报告逻辑
    """
    pass

# 基础fixtures
@pytest.fixture(scope="session")
def event_loop():
    """创建全局事件循环"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def import_manager():
    """导入管理器fixture"""
    from tests.import_manager import import_manager
    return import_manager

@pytest.fixture
def mock_builder():
    """提供标准Mock构建器"""
    from tests.fixtures.infrastructure_mocks import StandardMockBuilder
    return StandardMockBuilder()

# 基础测试fixtures
@pytest.fixture
def cache_mock(mock_builder):
    """缓存Mock fixture"""
    return mock_builder.create_cache_mock()

@pytest.fixture
def config_mock(mock_builder):
    """配置Mock fixture"""
    return mock_builder.create_config_mock()

@pytest.fixture
def logger_mock(mock_builder):
    """日志Mock fixture"""
    return mock_builder.create_logger_mock()
