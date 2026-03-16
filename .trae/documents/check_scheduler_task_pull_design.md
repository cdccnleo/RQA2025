# 检查统一调度器从PostgreSQL库中自动拉取特征提取任务设计与逻辑

## 目标
全面检查统一调度器从PostgreSQL库中自动拉取特征提取任务的设计与逻辑，确保其正确性和完整性。

## 检查范围

### 1. 调度器任务拉取机制
- [ ] 检查 `_db_task_pull_loop()` 方法的实现
- [ ] 检查 `_load_pending_tasks_from_db()` 方法的实现
- [ ] 检查任务拉取频率和异常处理
- [ ] 检查任务拉取日志记录

### 2. 任务提交逻辑
- [ ] 检查 `submit_task()` 方法的实现
- [ ] 检查任务ID的使用（原始数据库任务ID vs 调度器生成ID）
- [ ] 检查任务payload的构建
- [ ] 检查任务提交到工作队列的逻辑

### 3. 任务处理器逻辑
- [ ] 检查 `feature_extraction_handler()` 的实现
- [ ] 检查任务ID的获取方式
- [ ] 检查任务状态更新逻辑
- [ ] 检查 `_update_task_status()` 的实现

### 4. 任务状态更新
- [ ] 检查 `update_task_status()` 在 `feature_task_persistence.py` 中的实现
- [ ] 检查PostgreSQL状态更新逻辑
- [ ] 检查文件系统降级逻辑
- [ ] 检查状态更新日志

### 5. 数据库查询逻辑
- [ ] 检查 `list_feature_tasks()` 的实现
- [ ] 检查查询submitted状态任务的SQL
- [ ] 检查任务数量限制（limit=100）

### 6. 调度器主循环
- [ ] 检查 `_scheduler_loop()` 中任务拉取循环的启动
- [ ] 检查任务拉取循环的停止逻辑
- [ ] 检查调度器启动/停止日志

### 7. 应用生命周期
- [ ] 检查 `lifespan()` 函数中调度器的启动
- [ ] 检查调度器是否正确注册到FastAPI
- [ ] 检查调度器启动错误处理

## 检查步骤

### 步骤1: 代码审查
1. 读取 `unified_scheduler.py` 中的相关方法
2. 读取 `feature_extraction_handler.py` 中的处理器逻辑
3. 读取 `feature_task_persistence.py` 中的状态更新逻辑
4. 读取 `api.py` 中的调度器启动逻辑

### 步骤2: 日志分析
1. 检查容器日志中的调度器启动日志
2. 检查任务拉取日志
3. 检查任务执行日志
4. 检查状态更新日志

### 步骤3: 数据库验证
1. 查询submitted状态任务数量
2. 查询completed状态任务数量
3. 验证任务状态是否正确更新

### 步骤4: 问题识别与修复
1. 识别设计或逻辑问题
2. 提出修复方案
3. 实施修复
4. 验证修复效果

## 预期结果
- 调度器每30秒自动从PostgreSQL拉取submitted状态任务
- 任务使用原始数据库任务ID提交到调度器
- 任务执行完成后，状态正确更新为completed
- submitted任务数量逐渐减少，completed任务数量逐渐增加

## 相关文件
- `src/core/orchestration/scheduler/unified_scheduler.py`
- `src/core/orchestration/scheduler/handlers/feature_extraction_handler.py`
- `src/gateway/web/feature_task_persistence.py`
- `src/gateway/web/api.py`
