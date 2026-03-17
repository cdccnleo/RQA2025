# feature_engineering_tasks 表结构改进实施计划

## 目标
完善 feature_engineering_tasks 表结构，增加溯源字段和业务标识字段，提升任务可追溯性和查询效率。

## 实施步骤

### 步骤 1：数据库表结构变更
- [ ] 添加新列到 feature_engineering_tasks 表
  - source_task_id VARCHAR(100) - 源任务ID
  - parent_task_id VARCHAR(100) - 父任务ID
  - data_source VARCHAR(50) - 数据源
  - symbol VARCHAR(20) - 股票代码
  - stock_name VARCHAR(100) - 股票名称
  - stock_code VARCHAR(20) - 带前缀的股票代码
  - indicators JSONB - 指标列表
  - start_date DATE - 数据开始日期
  - end_date DATE - 数据结束日期
  - worker_id VARCHAR(50) - 执行工作节点ID
  - retry_count INTEGER DEFAULT 0 - 重试次数
  - priority INTEGER DEFAULT 5 - 任务优先级
  - execution_time_ms INTEGER - 执行耗时
  - memory_usage_mb INTEGER - 内存使用

- [ ] 创建索引
  - idx_feature_tasks_source (source_task_id)
  - idx_feature_tasks_parent (parent_task_id)
  - idx_feature_tasks_data_source (data_source)
  - idx_feature_tasks_symbol (symbol)
  - idx_feature_tasks_stock_code (stock_code)
  - idx_feature_tasks_symbol_status (symbol, status)
  - idx_feature_tasks_datasource_status (data_source, status)
  - idx_feature_tasks_date_range (start_date, end_date)

### 步骤 2：修改持久化模块
- [ ] 修改 _save_to_postgresql 函数，支持新字段的保存
- [ ] 修改 INSERT/UPDATE SQL 语句，包含新字段
- [ ] 从 config JSON 中提取 symbol、data_source 等字段

### 步骤 3：修改任务创建逻辑
- [ ] 修改 save_feature_task 函数，接收更多参数
- [ ] 在任务创建时填充新字段
- [ ] 更新所有调用 save_feature_task 的代码

### 步骤 4：数据迁移
- [ ] 编写数据迁移脚本，从现有 config 字段提取数据到新列
- [ ] 执行迁移，填充 symbol、data_source 等字段
- [ ] 验证迁移结果

### 步骤 5：验证和测试
- [ ] 验证新任务能正确保存新字段
- [ ] 验证查询接口正常工作
- [ ] 验证索引生效

## 相关文件

- src/gateway/web/feature_task_persistence.py - 持久化模块
- src/gateway/web/postgresql_persistence.py - PostgreSQL 连接
- src/gateway/web/feature_engineering_routes.py - API 路由
- src/gateway/web/feature_task_executor.py - 任务执行器

## 回滚策略

1. 保留 config 字段作为备份
2. 新字段均为可空，不影响现有代码
3. 逐步迁移，可随时回滚
