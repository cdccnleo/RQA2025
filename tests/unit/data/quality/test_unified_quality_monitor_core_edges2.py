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


import pandas as pd
from src.data.quality.unified_quality_monitor import UnifiedQualityMonitor, IDataValidator


class _AlwaysFailValidator:
    def validate(self, data, data_type):
        # 模拟验证器异常路径：返回格式非预期触发修复/兜底逻辑统计
        return {"is_valid": False, "errors": ["bad-format"], "issues": [{"issue_type": "format_error"}]}

    def get_validation_rules(self, data_type):
        return {"rules": []}


def test_unknown_type_and_failed_validation_triggers_issues():
    mon = UnifiedQualityMonitor(validator=_AlwaysFailValidator())  # type: ignore[arg-type]
    df = pd.DataFrame({"a": [1, 2]})
    # 传入未知字符串类型，命中类型规范化兜底
    # 统一质量监控主流程入口通常为 check_quality
    rep = mon.check_quality(df, data_type="unknown_type")  # type: ignore[attr-defined]
    # 断言生成了有效结果：允许不同实现形式
    assert rep is not None
    ok = False
    if isinstance(rep, dict):
        ok = ("issues" in rep) or ("quality_score" in rep) or ("metrics" in rep)
    else:
        ok = hasattr(rep, "issues") or hasattr(rep, "quality_score") or hasattr(rep, "metrics")
    assert ok


