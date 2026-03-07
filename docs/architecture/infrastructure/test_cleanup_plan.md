# 基础设施层测试清理计划

## 清理目标

根据架构审查结果，删除不符合架构设计的测试用例，避免冗余代码干扰。

## 需要删除的测试文件

### 1. 测试不存在模块的测试文件

#### 1.1 统一接口管理器测试
- **文件**: `tests/unit/infrastructure/test_unified_interface_manager.py`
- **问题**: 测试了不存在的 `src.infrastructure.interfaces.unified_interface_manager` 模块
- **状态**: 该模块只存在于备份目录中，主代码库中不存在
- **建议**: 删除此测试文件

#### 1.2 覆盖率改进测试
- **文件**: `tests/unit/infrastructure/test_coverage_improvement.py`
- **问题**: 测试了大量不存在的模块，如：
  - `src.infrastructure.compliance`
  - `src.infrastructure.config.enhanced_config_manager`
  - `src.infrastructure.config.monitoring`
  - `src.infrastructure.config.performance`
  - `src.infrastructure.config.security`
  - `src.infrastructure.config.services`
  - `src.infrastructure.config.storage`
  - `src.infrastructure.config.validation`
  - `src.infrastructure.dashboard`
  - `src.infrastructure.disaster`
  - `src.infrastructure.network`
  - `src.infrastructure.resource`
  - `src.infrastructure.scheduler`
  - `src.infrastructure.storage.adapters`
- **状态**: 这些模块在当前架构中不存在或已被重构
- **建议**: 删除此测试文件

#### 1.3 配置相关测试
- **文件**: `tests/unit/infrastructure/test_unified_config_manager.py`
- **问题**: 测试了不存在的 `src.infrastructure.config.unified_manager` 和 `src.infrastructure.config.interfaces.unified_interface` 模块
- **状态**: 这些模块已被重构到 `core/config/` 目录
- **建议**: 删除此测试文件

#### 1.4 验证器测试
- **文件**: `tests/unit/infrastructure/test_validators.py`
- **问题**: 测试了不存在的 `src.infrastructure.config.services.validators` 模块
- **状态**: 该模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.5 模式验证器测试
- **文件**: `tests/unit/infrastructure/test_schema_validator.py`
- **问题**: 测试了不存在的 `src.infrastructure.config.validation.schema` 模块
- **状态**: 该模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.6 事件服务测试
- **文件**: `tests/unit/infrastructure/test_event_service.py`
- **问题**: 测试了不存在的 `src.infrastructure.config.services.event_service` 和 `src.infrastructure.config.error.exceptions` 模块
- **状态**: 这些模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.7 标准接口测试
- **文件**: `tests/unit/infrastructure/test_standard_interfaces.py`
- **问题**: 测试了不存在的 `src.infrastructure.interfaces.standard_interfaces` 模块
- **状态**: 该模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.8 存储异常测试
- **文件**: `tests/unit/infrastructure/test_storage_exceptions.py`
- **问题**: 测试了不存在的 `src.infrastructure.storage.exceptions` 模块
- **状态**: 该模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.9 网络重试策略测试
- **文件**: `tests/unit/infrastructure/test_retry_policy.py`
- **问题**: 测试了不存在的 `src.infrastructure.network.retry_policy` 和 `src.infrastructure.network.exceptions` 模块
- **状态**: 这些模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.10 环境加载器测试
- **文件**: `tests/unit/infrastructure/test_env_loader.py`
- **问题**: 测试了不存在的 `src.infrastructure.config.error.exceptions` 模块
- **状态**: 该模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.11 文件存储测试
- **文件**: `tests/unit/infrastructure/test_file_storage.py`
- **问题**: 测试了不存在的 `src.infrastructure.config.storage.file_storage` 模块
- **状态**: 该模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.12 数据库存储测试
- **文件**: `tests/unit/infrastructure/test_database_storage.py`
- **问题**: 测试了不存在的 `src.infrastructure.config.storage.database_storage` 模块
- **状态**: 该模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.13 工厂测试
- **文件**: `tests/unit/infrastructure/test_factory.py`
- **问题**: 测试了不存在的 `src.infrastructure.config.factory` 模块
- **状态**: 该模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.14 YAML加载器测试
- **文件**: `tests/unit/infrastructure/test_yaml_loader.py`
- **问题**: 测试了不存在的 `src.infrastructure.config.strategies.yaml_loader` 和 `src.infrastructure.config.error.exceptions` 模块
- **状态**: 这些模块在当前架构中不存在
- **建议**: 删除此测试文件

#### 1.15 JSON加载器测试
- **文件**: `tests/unit/infrastructure/test_json_loader.py`
- **问题**: 测试了不存在的 `src.infrastructure.config.strategies.json_loader` 和 `src.infrastructure.config.error.exceptions` 模块
- **状态**: 这些模块在当前架构中不存在
- **建议**: 删除此测试文件

### 2. 测试已重构模块的测试文件

#### 2.1 错误处理器测试
- **文件**: `tests/unit/infrastructure/test_error_handler.py`
- **问题**: 测试了 `src.infrastructure.error.error_handler` 模块，但该模块已被重构到 `core/error/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.2 系统监控测试
- **文件**: `tests/unit/infrastructure/test_system_monitor.py`
- **问题**: 测试了 `src.infrastructure.monitoring.system_monitor` 模块，但该模块已被重构到 `core/monitoring/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.3 应用监控测试
- **文件**: `tests/unit/infrastructure/test_application_monitor.py`
- **问题**: 测试了 `src.infrastructure.monitoring.application_monitor` 模块，但该模块已被重构到 `core/monitoring/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.4 告警管理器测试
- **文件**: `tests/unit/infrastructure/test_alert_manager.py`
- **问题**: 测试了 `src.infrastructure.monitoring.alert_manager` 模块，但该模块已被重构到 `core/monitoring/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.5 存储监控测试
- **文件**: `tests/unit/infrastructure/test_storage_monitor.py`
- **问题**: 测试了 `src.infrastructure.monitoring.storage_monitor` 模块，但该模块已被重构到 `core/monitoring/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.6 资源API测试
- **文件**: `tests/unit/infrastructure/test_resource_api.py`
- **问题**: 测试了 `src.infrastructure.monitoring.resource_api` 模块，但该模块已被重构到 `core/monitoring/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.7 行为监控测试
- **文件**: `tests/unit/infrastructure/test_behavior_monitor.py`
- **问题**: 测试了 `src.infrastructure.monitoring.behavior_monitor` 模块，但该模块已被重构到 `core/monitoring/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.8 回测监控测试
- **文件**: `tests/unit/infrastructure/test_backtest_monitor.py`
- **问题**: 测试了 `src.infrastructure.monitoring.backtest_monitor` 模块，但该模块已被重构到 `core/monitoring/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.9 资源管理器测试
- **文件**: `tests/unit/infrastructure/test_resource_manager.py`
- **问题**: 测试了 `src.infrastructure.resource.resource_manager` 模块，但该模块已被重构到 `core/resource_management/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.10 GPU管理器测试
- **文件**: `tests/unit/infrastructure/test_gpu_manager.py`
- **问题**: 测试了 `src.infrastructure.resource.gpu_manager` 模块，但该模块已被重构到 `core/resource_management/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.11 资源仪表板测试
- **文件**: `tests/unit/infrastructure/test_resource_dashboard.py`
- **问题**: 测试了 `src.infrastructure.dashboard.resource_dashboard` 模块，但该模块已被重构到 `extensions/dashboard/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

#### 2.12 应用工厂测试
- **文件**: `tests/unit/infrastructure/test_app_factory.py`
- **问题**: 测试了 `src.infrastructure.web.app_factory` 模块，但该模块已被重构到 `extensions/web/` 目录
- **状态**: 需要更新导入路径或删除
- **建议**: 更新导入路径或删除此测试文件

### 3. 测试根目录模块的测试文件

#### 3.1 自动恢复测试
- **文件**: `tests/unit/infrastructure/test_auto_recovery.py`
- **问题**: 测试了 `src.infrastructure.auto_recovery` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.2 熔断器测试
- **文件**: `tests/unit/infrastructure/test_circuit_breaker.py`
- **问题**: 测试了 `src.infrastructure.circuit_breaker` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.3 数据同步测试
- **文件**: `tests/unit/infrastructure/test_data_sync.py`
- **问题**: 测试了 `src.infrastructure.data_sync` 和 `src.infrastructure.notification` 模块，这些模块位于根目录
- **状态**: 需要检查这些模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.4 事件测试
- **文件**: `tests/unit/infrastructure/test_event.py`
- **问题**: 测试了 `src.infrastructure.event` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.5 锁测试
- **文件**: `tests/unit/infrastructure/test_lock.py`
- **问题**: 测试了 `src.infrastructure.lock` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.6 版本测试
- **文件**: `tests/unit/infrastructure/test_version.py`
- **问题**: 测试了 `src.infrastructure.version` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.7 可视化监控测试
- **文件**: `tests/unit/infrastructure/test_visual_monitor.py`
- **问题**: 测试了 `src.infrastructure.visual_monitor` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.8 服务启动器测试
- **文件**: `tests/unit/infrastructure/test_service_launcher.py`
- **问题**: 测试了 `src.infrastructure.service_launcher` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.9 初始化基础设施测试
- **文件**: `tests/unit/infrastructure/test_init_infrastructure.py`
- **问题**: 测试了 `src.infrastructure.init_infrastructure` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.10 降级管理器测试
- **文件**: `tests/unit/infrastructure/test_degradation_manager.py`
- **问题**: 测试了 `src.infrastructure.degradation_manager` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.11 灾难恢复测试
- **文件**: `tests/unit/infrastructure/test_disaster_recovery.py`
- **问题**: 测试了 `src.infrastructure.disaster_recovery` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.12 线程管理测试
- **文件**: `tests/unit/infrastructure/test_thread_management.py`
- **问题**: 测试了 `src.infrastructure.disaster_recovery` 和 `src.infrastructure.resource.resource_manager` 模块
- **状态**: 需要检查这些模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.13 最终部署检查测试
- **文件**: `tests/unit/infrastructure/test_final_deployment_check.py`
- **问题**: 测试了 `src.infrastructure.final_deployment_check` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.14 异步推理引擎测试
- **文件**: `tests/unit/infrastructure/test_async_inference_engine.py`
- **问题**: 测试了 `src.infrastructure.inference_engine_async` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

#### 3.15 异步推理引擎Top20测试
- **文件**: `tests/unit/infrastructure/test_async_inference_engine_top20.py`
- **问题**: 测试了 `src.infrastructure.inference_engine_async` 模块，该模块位于根目录
- **状态**: 需要检查该模块是否仍然存在
- **建议**: 检查模块存在性，如果不存在则删除测试文件

## 清理步骤

### 第一步：删除测试不存在模块的测试文件
1. 删除 `test_unified_interface_manager.py`
2. 删除 `test_coverage_improvement.py`
3. 删除 `test_unified_config_manager.py`
4. 删除 `test_validators.py`
5. 删除 `test_schema_validator.py`
6. 删除 `test_event_service.py`
7. 删除 `test_standard_interfaces.py`
8. 删除 `test_storage_exceptions.py`
9. 删除 `test_retry_policy.py`
10. 删除 `test_env_loader.py`
11. 删除 `test_file_storage.py`
12. 删除 `test_database_storage.py`
13. 删除 `test_factory.py`
14. 删除 `test_yaml_loader.py`
15. 删除 `test_json_loader.py`

### 第二步：检查根目录模块存在性
1. 检查 `src/infrastructure/auto_recovery.py` 是否存在
2. 检查 `src/infrastructure/circuit_breaker.py` 是否存在
3. 检查 `src/infrastructure/data_sync.py` 是否存在
4. 检查 `src/infrastructure/event.py` 是否存在
5. 检查 `src/infrastructure/lock.py` 是否存在
6. 检查 `src/infrastructure/version.py` 是否存在
7. 检查 `src/infrastructure/visual_monitor.py` 是否存在
8. 检查 `src/infrastructure/service_launcher.py` 是否存在
9. 检查 `src/infrastructure/init_infrastructure.py` 是否存在
10. 检查 `src/infrastructure/degradation_manager.py` 是否存在
11. 检查 `src/infrastructure/disaster_recovery.py` 是否存在
12. 检查 `src/infrastructure/final_deployment_check.py` 是否存在
13. 检查 `src/infrastructure/inference_engine_async.py` 是否存在

### 第三步：更新或删除测试已重构模块的测试文件
1. 更新或删除 `test_error_handler.py`
2. 更新或删除 `test_system_monitor.py`
3. 更新或删除 `test_application_monitor.py`
4. 更新或删除 `test_alert_manager.py`
5. 更新或删除 `test_storage_monitor.py`
6. 更新或删除 `test_resource_api.py`
7. 更新或删除 `test_behavior_monitor.py`
8. 更新或删除 `test_backtest_monitor.py`
9. 更新或删除 `test_resource_manager.py`
10. 更新或删除 `test_gpu_manager.py`
11. 更新或删除 `test_resource_dashboard.py`
12. 更新或删除 `test_app_factory.py`

## 预期效果

通过清理不符合架构设计的测试用例，预期达到以下效果：

1. **减少测试噪音**: 删除测试不存在模块的测试文件，避免测试失败干扰
2. **提高测试效率**: 减少不必要的测试执行时间
3. **保持架构一致性**: 确保测试用例与当前架构设计一致
4. **简化维护**: 减少需要维护的测试代码
5. **提高测试质量**: 专注于测试实际存在的功能

## 验证方法

清理完成后，运行以下命令验证：

```bash
python -m pytest tests/unit/infrastructure/ -v
```

预期结果：
- 测试通过率提高
- 测试失败率降低
- 测试跳过率合理
- 无导入错误
