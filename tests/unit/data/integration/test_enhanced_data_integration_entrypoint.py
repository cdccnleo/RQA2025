import importlib

import pytest


@pytest.fixture
def entry_module():
    import src.data.integration.enhanced_data_integration as module

    # 重新加载一次，确保其他测试对模块的猴补丁不会污染本用例
    return importlib.reload(module)


def test_create_enhanced_data_integration_uses_exported_class(monkeypatch, entry_module):
    created = {}

    class DummyIntegration:
        def __init__(self, config=None):
            created["config"] = config

    monkeypatch.setattr(entry_module, "EnhancedDataIntegration", DummyIntegration)

    config = {"custom": "config"}
    instance = entry_module.create_enhanced_data_integration(config)

    assert isinstance(instance, DummyIntegration)
    assert created["config"] == config


def test_entrypoint_reexports_expected_symbols(entry_module):
    expected_symbols = {
        "IntegrationConfig",
        "EnhancedDataIntegration",
        "TaskPriority",
        "LoadTask",
        "EnhancedParallelLoadingManager",
        "DynamicThreadPoolManager",
        "ConnectionPoolManager",
        "MemoryOptimizer",
        "FinancialDataOptimizer",
        "create_enhanced_loader",
        "create_enhanced_data_integration",
        "shutdown",
        "get_integration_stats",
    }

    exported = set(entry_module.__all__)
    assert expected_symbols <= exported
    for symbol in expected_symbols:
        assert hasattr(entry_module, symbol)

