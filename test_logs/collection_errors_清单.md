# Collection Errors 完整清单

**生成时间**: 2025-11-02  
**目的**: 识别所有无法被pytest收集的测试文件

---

## ❌ 总计：17个Collection Errors

### Strategy层（2个）
1. ❌ `tests\unit\strategy\test_strategy_execution.py`
2. ❌ `tests\unit\strategy\test_strategy_signals.py`
- ✅ 其他: 920个测试正常收集

### Trading层（9个）
3. ❌ `tests\unit\trading\test_broker_adapter.py`
4. ❌ `tests\unit\trading\test_execution_engine_advanced.py`
5. ❌ `tests\unit\trading\test_execution_engine_core.py`
6. ❌ `tests\unit\trading\test_live_trading.py`
7. ❌ `tests\unit\trading\test_order_management_advanced.py`
8. ❌ `tests\unit\trading\test_order_manager_basic.py`
9. ❌ `tests\unit\trading\test_position_management_advanced.py`
10. ❌ `tests\unit\trading\test_smart_execution.py`
11. ❌ `tests\unit\trading\test_trading_engine_advanced.py`
- ✅ 其他: 514个测试正常收集

### Risk层（6个）
12. ❌ `tests\unit\risk\test_compliance_workflow.py`
13. ❌ `tests\unit\risk\test_real_time_monitor_coverage.py`
14. ❌ `tests\unit\risk\test_realtime_risk_monitor.py`
15. ❌ `tests\unit\risk\test_risk_assessment.py`
16. ❌ `tests\unit\risk\test_risk_manager.py`
17. ❌ `tests\unit\risk\test_risk_manager_coverage.py`
- ✅ 其他: 328个测试正常收集

---

## 🎯 修复计划

**优先级**: 按层级顺序修复
1. Strategy层（2个）→ 今天
2. Trading层（9个）→ 明天
3. Risk层（6个）→ 后天

**目标**: 3天内修复所有17个errors


