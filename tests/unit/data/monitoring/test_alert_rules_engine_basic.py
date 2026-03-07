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


import json
import time
from datetime import datetime, timedelta
import sys
import types
from enum import Enum

# 提供 src.core.integration 测试桩，避免引擎导入失败
_stub = types.ModuleType("src.core.integration")

def get_data_adapter():
    class _Adapter:
        def get_monitoring(self):
            return None
        def get_logger(self):
            import logging
            return logging.getLogger(__name__)
    return _Adapter()

_stub.get_data_adapter = get_data_adapter
sys.modules["src.core.integration"] = _stub

# 提供 interfaces.standard_interfaces 测试桩，兼容两种命名空间
class DataSourceType(Enum):
    stock = "stock"
    index = "index"
    news = "news"

# src.data.interfaces.standard_interfaces
pkg_data_interfaces = types.ModuleType("src.data.interfaces")
sys.modules["src.data.interfaces"] = pkg_data_interfaces
std_mod_data = types.ModuleType("src.data.interfaces.standard_interfaces")
std_mod_data.DataSourceType = DataSourceType
sys.modules["src.data.interfaces.standard_interfaces"] = std_mod_data

# src.interfaces.standard_interfaces
pkg_interfaces = types.ModuleType("src.interfaces")
sys.modules["src.interfaces"] = pkg_interfaces
std_mod = types.ModuleType("src.interfaces.standard_interfaces")
std_mod.DataSourceType = DataSourceType
sys.modules["src.interfaces.standard_interfaces"] = std_mod

from src.data.monitoring.data_alert_rules import (
    DataAlertRulesEngine,
    AlertRule, AlertRuleType,
    AlertSeverity,
    AlertCondition, AlertConditionType,
    AlertSuppression
)


def test_threshold_rule_and_cooldown():
    eng = DataAlertRulesEngine()
    data = {"hit_rate": 0.5}
    alerts1 = eng.evaluate_rules(data)
    assert any(a["rule_id"] == "cache_hit_rate_low" for a in alerts1)
    alerts2 = eng.evaluate_rules(data)
    assert not any(a["rule_id"] == "cache_hit_rate_low" for a in alerts2)


def test_composite_rule_triggers_on_any_condition():
    eng = DataAlertRulesEngine()
    data = {"completeness": 0.90, "accuracy": 0.99, "timeliness": 0.95}
    alerts = eng.evaluate_rules(data)
    assert any(a["rule_id"] == "data_quality_degraded" for a in alerts)


def test_suppression_blocks_alerts_by_rule_id():
    eng = DataAlertRulesEngine()
    # 抑制条件匹配输入数据（非 alert 对象）；命中时抑制指定 rule_ids
    cond = AlertCondition(type=AlertConditionType.GREATER_THAN, field="hit_rate", value=0.0)
    sup = AlertSuppression(
        suppression_id="sup1",
        rule_ids=["cache_hit_rate_low"],
        condition=cond,
        duration_minutes=5,
        reason="temporary maintenance",
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(minutes=5),
    )
    assert eng.add_suppression(sup) is True
    data = {"hit_rate": 0.2}
    alerts = eng.evaluate_rules(data)
    assert not any(a["rule_id"] == "cache_hit_rate_low" for a in alerts)


def test_export_and_import_rules_roundtrip():
    eng = DataAlertRulesEngine()
    exported = eng.export_rules()
    # 新引擎：先清空默认规则再导入，避免重复冲突
    eng2 = DataAlertRulesEngine()
    eng2.rules.clear()
    eng2.rule_performance.clear()
    res = eng2.import_rules(exported)
    assert res.get("imported_rules", 0) >= 1
    assert res.get("success", True) is True


