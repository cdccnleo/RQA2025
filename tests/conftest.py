import pytest
from prometheus_client import REGISTRY

# 可选：添加标记描述

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unit: 单元测试标记"
    )
    config.addinivalue_line(
        "markers",
        "integration: 集成测试标记"
    )
    config.addinivalue_line(
        "markers",
        "performance: 性能测试标记"
    )
    config.addinivalue_line(
        "markers",
        "slow: 慢速测试标记"
    )
    config.addinivalue_line(
        "markers",
        "ashare: A股特定功能测试"
    )
    config.addinivalue_line(
        "markers",
        "trading_hours: 交易时段相关测试"
    )
    config.addinivalue_line(
        "markers",
        "windows: Windows平台特定测试"
    )
    config.addinivalue_line(
        "markers",
        "linux: Linux平台特定测试"
    )

@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
