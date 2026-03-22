# PostgreSQL查询错误分析报告

**报告编号**: RQA-ERR-2026-0322-001  
**报告日期**: 2026-03-22  
**分析人员**: AI Assistant  
**错误类型**: 数据库字段名不匹配

---

## 一、错误概述

### 1.1 错误日志

```
2026-03-22 10:40:27,361 - src.gateway.web.feature_selection_task_persistence - WARNING - 
⚠️ PostgreSQL查询异常: column "source_task_id" does not exist 

LINE 3:                            source_task_id, selection_method,... 
                                   ^
```

### 1.2 错误影响

- **影响范围**: 特征选择任务查询功能
- **严重程度**: 中等（触发降级机制，系统仍可运行）
- **用户体验**: 数据从文件系统加载，可能不是最新数据

---

## 二、错误分析

### 2.1 根本原因

**数据库表结构与代码不一致**

| 项目 | 代码中使用 | 数据库实际字段 |
|------|-----------|---------------|
| 源任务ID字段 | `source_task_id` | `parent_task_id` |

**产生原因**:
1. 数据库表结构由其他模块创建，使用 `parent_task_id` 字段名
2. `feature_selection_task_persistence.py` 代码中使用 `source_task_id` 字段名
3. 两者命名不一致导致查询失败

### 2.2 错误发生位置

**文件**: `src/gateway/web/feature_selection_task_persistence.py`

**涉及函数**:
1. `list_selection_tasks()` - 第258行、268行
2. `get_selection_task_detail()` - 第471行、486行
3. `save_selection_task()` - 第96行、125行、141行
4. `create_selection_task()` - 第552行

### 2.3 错误触发条件

当以下SQL查询执行时触发错误:
```sql
SELECT task_id, task_type, status, progress, symbol,
       source_task_id, selection_method, n_features,
       created_at, updated_at
FROM feature_selection_tasks
```

由于 `source_task_id` 字段不存在，PostgreSQL抛出异常。

---

## 三、降级机制评估

### 3.1 降级机制触发情况

✅ **降级机制正确触发**

当PostgreSQL查询失败时，系统自动降级到文件系统存储:
```python
except Exception as e:
    logger.warning(f"⚠️ PostgreSQL查询异常: {e}，降级到文件系统")
    
# 从文件系统查询
if os.path.exists(FEATURE_SELECTION_TASKS_DIR):
    # 文件系统查询逻辑
```

### 3.2 降级机制执行状态

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 错误捕获 | ✅ | 正确捕获PostgreSQL异常 |
| 降级触发 | ✅ | 自动切换到文件系统 |
| 数据完整性 | ✅ | 文件系统数据完整 |
| 日志记录 | ✅ | 记录降级操作 |

### 3.3 文件系统数据状态

**数据位置**: `/app/data/feature_selection_tasks/`

**数据量**: 19个任务文件

**数据完整性**: ✅ 完整，包含所有必要字段

---

## 四、修复措施

### 4.1 修复方案

将代码中的 `source_task_id` 统一改为 `parent_task_id`，以匹配数据库实际表结构。

### 4.2 修复文件

**文件**: `src/gateway/web/feature_selection_task_persistence.py`

**修改内容**:

1. **建表语句** (第96行):
   ```python
   # 修改前
   source_task_id VARCHAR(100),
   
   # 修改后
   parent_task_id VARCHAR(100),
   ```

2. **INSERT语句** (第125行):
   ```python
   # 修改前
   source_task_id, selection_method, n_features, auto_execute,
   
   # 修改后
   parent_task_id, selection_method, n_features, auto_execute,
   ```

3. **INSERT参数** (第141行):
   ```python
   # 修改前
   task.get("source_task_id"),
   
   # 修改后
   task.get("parent_task_id") or task.get("source_task_id"),
   ```

4. **SELECT语句** (第258行、268行):
   ```python
   # 修改前
   SELECT task_id, task_type, status, progress, symbol,
          source_task_id, selection_method, n_features,
          created_at, updated_at
   
   # 修改后
   SELECT task_id, task_type, status, progress, symbol,
          parent_task_id, selection_method, n_features,
          created_at, updated_at
   ```

5. **结果映射** (第283行):
   ```python
   # 修改前
   "source_task_id": row[5],
   
   # 修改后
   "parent_task_id": row[5],
   ```

6. **create_selection_task函数** (第552行):
   ```python
   # 添加兼容性处理
   "parent_task_id": source_task_id,
   "source_task_id": source_task_id,
   ```

### 4.3 修复验证

**验证方法**:
```bash
# 1. 重新构建容器
docker-compose -f docker-compose.prod.yml up -d --build app

# 2. 测试API
curl -s "http://localhost:8000/api/v1/features/engineering/selection/tasks"
```

**验证结果**: ✅ 
- API正常返回19个特征选择任务
- 无PostgreSQL错误日志
- 降级机制未触发（数据库查询成功）

---

## 五、预防措施

### 5.1 数据库 schema 管理

1. **统一字段命名规范**
   - 建立字段命名标准文档
   - 使用 `parent_task_id` 而非 `source_task_id`

2. **版本控制**
   - 数据库迁移脚本纳入版本控制
   - 使用工具如 Alembic 管理 schema 变更

3. **代码审查**
   - 审查时检查字段名一致性
   - 使用 ORM 减少手写 SQL

### 5.2 测试策略

1. **单元测试**
   - 测试数据库操作函数
   - 验证字段名正确性

2. **集成测试**
   - 测试完整的数据流
   - 验证降级机制

3. **Schema 验证测试**
   - 定期检查代码与数据库 schema 一致性
   - 自动化测试防止回归

### 5.3 监控告警

1. **错误日志监控**
   - 监控 "column does not exist" 错误
   - 设置告警阈值

2. **降级机制监控**
   - 监控降级触发频率
   - 分析降级原因

---

## 六、总结

### 6.1 问题总结

| 项目 | 内容 |
|------|------|
| 错误类型 | 数据库字段名不匹配 |
| 根本原因 | 代码与数据库表结构不一致 |
| 影响范围 | 特征选择任务查询功能 |
| 严重程度 | 中等（降级机制保障可用性） |

### 6.2 修复总结

| 项目 | 状态 |
|------|------|
| 问题定位 | ✅ 完成 |
| 代码修复 | ✅ 完成 |
| 修复验证 | ✅ 通过 |
| 预防措施 | ✅ 已制定 |

### 6.3 经验教训

1. **Schema 一致性**: 代码与数据库 schema 必须保持一致
2. **降级机制**: 完善的降级机制保障了系统可用性
3. **测试覆盖**: 需要加强数据库 schema 一致性测试
4. **监控告警**: 需要建立 schema 不一致的监控机制

---

## 七、附录

### 7.1 数据库表结构

```sql
CREATE TABLE feature_selection_tasks (
    task_id VARCHAR(100) PRIMARY KEY,
    task_type VARCHAR(50) NOT NULL DEFAULT 'feature_selection',
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    symbol VARCHAR(20),
    parent_task_id VARCHAR(100),  -- 实际字段名
    selection_method VARCHAR(50),
    n_features INTEGER DEFAULT 10,
    auto_execute BOOLEAN DEFAULT TRUE,
    input_features JSONB,
    total_input_features INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### 7.2 相关文件

- `src/gateway/web/feature_selection_task_persistence.py`
- `src/gateway/web/postgresql_persistence.py`

---

**报告完成**

**签字**: AI Assistant  
**日期**: 2026-03-22
