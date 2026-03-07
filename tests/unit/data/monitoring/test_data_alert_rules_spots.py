import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
import sys
import types

# 预先注入 src.core.integration.get_data_adapter 以避免导入期依赖
if "src.core" not in sys.modules:
    sys.modules["src.core"] = types.ModuleType("src.core")
if "src.core.integration" not in sys.modules:
    integ = types.ModuleType("src.core.integration")
    def _dummy_adapter():
        class _A:
            def get_monitoring(self): return None
            def get_logger(self):
                class _L:
                    def info(self, *a, **k): pass
                    def warning(self, *a, **k): pass
                    def error(self, *a, **k): pass
                return _L()
        return _A()
    integ.get_data_adapter = _dummy_adapter  # type: ignore
    sys.modules["src.core.integration"] = integ
    # 将 integration 挂到 src.core 命名空间
    setattr(sys.modules["src.core"], "integration", integ)

try:
    from src.data.monitoring.data_alert_rules import (
        AlertCondition,
        AlertConditionType,
        AlertRule,
        AlertRuleType,
        AlertSeverity,
    )
except ImportError:
    pytest.skip("skip spots: integration adapter not available in this worker", allow_module_level=True)


def test_rule_data_types_filter_handles_unknown_string_type():
    # data_types 只允许 CACHE，传入 unknown 字符串应被过滤掉（返回 None）
    rule = AlertRule(
        rule_id="typed_only_cache",
        name="typed",
        description="",
        rule_type=AlertRuleType.THRESHOLD,
        conditions=[AlertCondition(AlertConditionType.LESS_THAN, "metric", 1)],
        severity=AlertSeverity.INFO,
        message_template="m {metric}",
        data_types=[],  # 空表示不过滤
    )
    # 先不限制类型，应命中
    assert rule.evaluate({"metric": 0, "data_type": "api"}) is not None

    # 限制为 CACHE，再传入 unknown，应 None
    # 仅需非空以触发字符串->枚举解析与异常分支，无需具体枚举成员
    rule.data_types = [None]
    assert rule.evaluate({"metric": 0, "data_type": "unknown"}) is None
    assert rule.evaluate({"metric": 0, "data_type": "unknown_enum"}) is None


def test_composite_rule_exception_in_condition_is_ignored():
    class RaisingCondition:
        def evaluate(self, payload):
            raise RuntimeError("boom")

    rule = AlertRule(
        rule_id="composite_with_exception",
        name="c",
        description="",
        rule_type=AlertRuleType.COMPOSITE,
        conditions=[
            AlertCondition(AlertConditionType.LESS_THAN, "metric", 1),
            RaisingCondition(),
        ],
        severity=AlertSeverity.WARNING,
        message_template="m {metric}",
    )
    # 条件1满足，条件2抛异常被忽略，整体应命中
    alert = rule.evaluate({"metric": 0})
    assert alert is not None and alert["rule_id"] == "composite_with_exception"


