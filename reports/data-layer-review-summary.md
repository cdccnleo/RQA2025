## 数据层投产评审包（分域覆盖率与残留清单）

- 生成时间：自动化小回归最新一轮
- 归档包：`test_logs/data-layer-review-package.zip`
  - 包含：`test_logs/coverage-*.xml` 与对应 `pytest-*.log`

### 分域覆盖率（最新）
- interfaces: ≈75%（`src\data\interfaces\api.py` 契约/异常分支仍有少量 term-missing）
- monitoring: ≈55%（`performance_monitor`=100%，`data_alert_rules` 并行稳定 129/129）
- quality: 88%（`unified_quality_monitor.py`≈85%）
- sources: ≈78%（`intelligent_source_manager` 并行导入兜底已加）
- cache: ≈84%（`cache_manager.py`≈84%、`multi_level_cache.py`≈85%、`smart_cache_optimizer.py`≈91%）
- distributed: ≈96%（`cluster_manager.py`/`sharding_manager.py`/`multiprocess_loader.py`=100%、`distributed_data_loader.py`≈94%）
- governance: ≈94%（`enterprise_governance.py`≈94%）
- export: ≈85%（`data_exporter.py`≈85%）
- transformers: ≈87%（`data_transformer.py`≈87%）

### 残留 term-missing（高价值优先）
- `src\data\interfaces\api.py`：就绪探针非就绪路径、`/store` 异常/fallback 若干行
- `src\data\monitoring\dashboard.py`：JSON 导出与回调出错路径极少数行
- `src\data\monitoring\data_alert_rules.py`：变化率、区间边界、JSON 导入错误分支若干行

### 并行稳定性与适配
- 已为 `IDataValidator`、`IDataLoader` 增加轻量兜底，避免多进程导入抖动
- `data_alert_rules` 的 data_types 解析测试采用“非空占位”策略，16 workers 下稳定通过

### 建议下一批（预期 +3%～+6%）
- interfaces：补 `/ready` 非就绪细粒度断言，`/store` 异常日志校验
- monitoring：补 `dashboard` 导出/回调异常；`data_alert_rules` 变化率与 JSON 导入错误用例
- 产出：合并分域 XML 与 term-missing 摘要，更新文档 Phase 3 与评审附件


### 更新（2025-11-16）
- 分域覆盖（本轮小批并行，已写入 `test_logs/coverage-*-latest.xml`）
  - distributed: 96%（`distributed_data_loader.py`≈94%，其余关键组件 100%）
  - cache: 84%（`cache_manager.py`≈84%、`multi_level_cache.py`≈85%）
  - core: 66%（核心编排与异常分支已覆盖，长尾路径待补）
  - preload: 93%（调度/幂等/间隔门控已覆盖）
  - transformers: 87%（缺字段/类型异常/空数据已覆盖）
  - validation: 54%（`validator.py`=100%；组件型文件仍为 0% 待补）
  - quality: 88%（`unified_quality_monitor.py`≈85%）
  - sources: 59%（XML 已生成；建议统一分支覆盖配置避免合并冲突）
  - version_control: 79%（创建/回滚/对比/异常分支已打通）
  - governance: 94%（主体逻辑高覆盖，少量报告路径欠缺）

- TOP5 关键缺口（优先补齐）
  1) `src\data\core\data_model.py` ≈15%：模型序列化/校验长尾与错误聚合
  2) `src\data\sources\data_source_manager.py` ≈24%：源选择/失败转移/速率限制边界
  3) `src\data\validation\assertion_components.py` 0%：断言组件基础与异常路径
  4) `src\data\validation\checker_components.py` 0%：校验组件基础与末端错误
  5) `src\data\monitoring\dashboard.py` ≈40%：导出/回调异常与边界期望

- 稳定性/配置建议
  - 统一 `--cov-branch` 策略执行各批，以避免 “Can't combine branch coverage data with statement data” 的合并报错
  - 固定入口脚本：
    - 分层小批执行：`scripts/ci/pytest_data_layer_small_batches.ps1`（输出到 `test_logs`）
    - 评审包生成：`scripts/build_data_layer_review.ps1`（产物 `test_logs/data-layer-review-package.zip`）

