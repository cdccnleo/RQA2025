# Monitoring Core/Handlers 补测计划

## 目标范围
- src/infrastructure/monitoring/alert_system.py
- src/infrastructure/monitoring/handlers/component_monitor.py
- src/infrastructure/monitoring/application/__init__.py
- src/infrastructure/monitoring/application/application_monitor.py
- src/infrastructure/monitoring/application/logger_pool_monitor.py

## 现状
- 最新覆盖率（2025-11-11，本轮补测后）：
  - `alert_system.py`：100%（核心导出路径与降级分支全部覆盖）
  - `handlers/component_monitor.py`：99%（仅剩 `if __name__ == "__main__"` 守卫未覆盖）
  - `application/__init__.py`：100%（依赖可用与导入失败双路径均验证）
  - `application/application_monitor.py`：100%（性能采集、阈值告警、线程循环、异常分支均覆盖）
  - `application/logger_pool_monitor.py`：64%（回退路径与全局单例已覆盖，高阶组件分支待补）
- 现有测试已经补齐核心 handler 的直接单测，剩余低覆盖集中在 legacy monitor 组件实现文件（尤其是 logger_pool 分层组件），后续可在专项治理中分批补测。

## 测试补充思路
1. alert_system.py
   - 构造最小化的 AlertManager/Notification stub，验证 AlertSystem.send_alert、configure_alert_rules、get_alert_history 等接口。
   - 模拟异常路径，确保错误处理分支被覆盖。
2. handlers/component_monitor.py
   - 使用 fake component/metrics collector，验证 ComponentMonitor 的注册、健康检查、生命周期管理逻辑。
   - 补充异常/边界场景（组件未注册、重复注册、状态转换）。
3. application/__init__.py
   - 编写 smoke test 确保关键导出的类/函数可导入，捕捉动态依赖问题。

## 计划产出
- ✅ `tests/unit/infrastructure/monitoring/handlers/test_alert_system_core.py`
- ✅ `tests/unit/infrastructure/monitoring/handlers/test_component_monitor_core.py`
- ✅ `tests/unit/infrastructure/monitoring/application/test_application_init_smoke.py`
- ✅ `tests/unit/infrastructure/monitoring/application/test_application_monitor_core.py`
- 🔁 `tests/unit/infrastructure/monitoring/application/test_logger_pool_monitor_core.py`（已覆盖核心回退逻辑，后续计划新增组件化路径测试）

## 时间建议
- 阶段1：alert_system + component_monitor（对覆盖率提升最大）
- 阶段2：application/__init__.py（轻量，可与阶段1并行）
- 阶段3：application/application_monitor（线程、异常、告警路径已齐备，可进入维护阶段）
- 下一轮重点：`logger_pool_monitor` 及其组件化实现，需构建轻量 stub 命中 `COMPONENTS_AVAILABLE=True` 分支、Prometheus 导出、监控循环回调等路径。


