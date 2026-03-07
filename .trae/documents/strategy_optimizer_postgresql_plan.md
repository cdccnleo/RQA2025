# 策略优化结果PostgreSQL持久化支持实施计划

## 计划目标
参考特征工程和模型训练的持久化机制，为策略优化结果实现PostgreSQL持久化支持，即双写机制（文件+数据库）。

## 实施范围

### 1. 数据库表设计
- 创建优化结果表结构
- 设计索引策略
- 定义字段类型和约束

### 2. 双写机制实现
- 文件系统写入（保留现有功能）
- PostgreSQL数据库写入（新增）
- 写入失败回滚机制

### 3. 数据读取支持
- 优先从PostgreSQL读取
- PostgreSQL失败时回退到文件系统
- 数据一致性校验

### 4. 数据迁移（可选）
- 现有文件数据迁移到数据库
- 迁移脚本开发

## 实施步骤

### 第一阶段：数据库表设计（30分钟）
1. 设计optimization_results表结构
2. 创建SQL迁移脚本
3. 在PostgreSQL中创建表

### 第二阶段：双写机制开发（1小时）
1. 实现PostgreSQL写入函数
2. 修改save_optimization_result实现双写
3. 添加错误处理和回滚机制
4. 添加日志记录

### 第三阶段：数据读取支持（45分钟）
1. 实现PostgreSQL读取函数
2. 修改load_optimization_result支持双源读取
3. 修改list_optimization_results支持双源读取
4. 实现数据一致性校验

### 第四阶段：删除和更新支持（30分钟）
1. 修改delete_optimization_result支持双删除
2. 确保数据一致性

### 第五阶段：测试验证（45分钟）
1. 单元测试双写功能
2. 测试数据库故障回退
3. 测试数据一致性
4. 性能测试对比

### 第六阶段：文档更新（30分钟）
1. 更新API文档
2. 更新架构文档
3. 记录配置说明

## 技术方案

### 数据库表结构

```sql
CREATE TABLE optimization_results (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(255) UNIQUE NOT NULL,
    strategy_id VARCHAR(255) NOT NULL,
    strategy_name VARCHAR(500),
    method VARCHAR(100) NOT NULL,
    target VARCHAR(100) NOT NULL,
    results JSONB NOT NULL,
    completed_at TIMESTAMP,
    saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 索引
    CONSTRAINT idx_optimization_results_task_id UNIQUE (task_id),
    CONSTRAINT idx_optimization_results_strategy_id UNIQUE (strategy_id, task_id)
);

-- 创建索引
CREATE INDEX idx_optimization_results_strategy ON optimization_results(strategy_id);
CREATE INDEX idx_optimization_results_saved_at ON optimization_results(saved_at DESC);
```

### 双写机制流程

```
保存请求
    │
    ├─> 1. 写入文件系统
    │      └─> 成功/失败
    │
    ├─> 2. 写入PostgreSQL（异步）
    │      └─> 成功/失败（不影响主流程）
    │
    └─> 返回结果
```

### 读取机制流程

```
读取请求
    │
    ├─> 1. 尝试从PostgreSQL读取
    │      ├─> 成功：返回数据
    │      └─> 失败：继续下一步
    │
    └─> 2. 从文件系统读取
           └─> 返回数据
```

## 代码实现参考

### 特征工程参考实现
- 文件：`src/gateway/web/feature_task_persistence.py`
- 函数：`save_feature_task()` - 双写机制
- 函数：`_save_to_postgresql()` - PostgreSQL写入

### 模型训练参考实现
- 文件：`src/gateway/web/training_job_persistence.py`
- 函数：`save_training_job()` - 双写机制
- 函数：`_save_to_postgresql()` - PostgreSQL写入

## 配置要求

### 环境变量
```bash
# PostgreSQL连接配置（已存在）
DATABASE_URL=postgresql://user:password@localhost:5432/rqa2025

# 双写开关（可选）
ENABLE_POSTGRESQL_PERSISTENCE=true
```

## 预期产出
1. 数据库表创建脚本
2. 更新后的strategy_persistence.py
3. 双写机制实现
4. 数据读取支持
5. 测试用例
6. 更新文档

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 数据库写入失败 | 低 | 中 | 异步写入，不影响主流程 |
| 数据不一致 | 低 | 中 | 定期同步脚本 |
| 性能下降 | 低 | 低 | 异步写入，批量操作 |
