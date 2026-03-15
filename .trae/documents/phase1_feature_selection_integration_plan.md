# Phase 1：统一调度器集成特征选择任务 - 实施计划

## 1. 项目概述

### 1.1 目标
在统一调度器中完整集成特征选择任务类型，补齐量化策略开发流程的关键环节，确保架构完整性和业务流程闭环。

### 1.2 背景
- 统一调度器`JobType`枚举已定义`FEATURE_SELECTION`（第172行）
- 但`get_task_types()`API未暴露该任务类型
- 无对应的任务处理器实现
- 特征选择历史表为空，流程未打通

### 1.3 预期成果
- ✅ 统一调度器支持特征选择任务调度
- ✅ 特征选择过程仪表盘显示历史数据
- ✅ 量化策略开发流程完整闭环

## 2. 实施步骤

### 阶段1：暴露特征选择任务类型API（P0）

**文件**: `src/gateway/web/scheduler_routes.py`

**修改内容**:
```python
# 在get_task_types()函数中，特征层任务列表添加：
{"value": "feature_selection", "label": "特征选择", "category": "特征层"}
```

**验证方法**:
```bash
curl http://localhost:8000/api/v1/scheduler/task-types
# 应返回包含feature_selection的任务类型列表
```

### 阶段2：实现特征选择任务处理器（P0）

**文件**: `src/core/orchestration/scheduler/handlers/feature_selection_handler.py`（新建）

**实现内容**:
```python
async def feature_selection_handler(task: Task) -> Dict[str, Any]:
    """
    特征选择任务处理器
    
    执行流程：
    1. 解析任务参数（symbols, method, top_k等）
    2. 获取特征数据
    3. 调用FeatureSelector执行选择
    4. 记录选择历史到feature_selection_history表
    5. 返回选择结果
    """
    pass
```

**核心逻辑**:
- 参数验证和解析
- 调用`src/features/utils/feature_selector.py`
- 使用`FeatureSelectorHistoryManager`记录历史
- 异常处理和日志记录

### 阶段3：注册任务处理器（P0）

**文件**: `src/core/orchestration/scheduler/unified_scheduler.py`

**修改内容**:
```python
def _register_default_handlers(self):
    """注册默认任务处理器"""
    # 现有处理器...
    
    # 新增特征选择处理器
    try:
        from .handlers.feature_selection_handler import feature_selection_handler
        self.register_task_handler("feature_selection", feature_selection_handler)
        logger.info("✅ 特征选择任务处理器已注册")
    except Exception as e:
        logger.warning(f"⚠️ 特征选择处理器注册失败: {e}")
```

### 阶段4：测试验证（P0）

**4.1 API测试**:
```bash
# 1. 获取任务类型列表
curl http://localhost:8000/api/v1/scheduler/task-types

# 2. 提交特征选择任务
curl -X POST http://localhost:8000/api/v1/scheduler/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "feature_selection",
    "name": "特征选择-测试任务",
    "params": {
      "symbols": ["000917", "300124"],
      "method": "importance",
      "top_k": 10
    }
  }'

# 3. 查询任务状态
curl http://localhost:8000/api/v1/scheduler/tasks/{task_id}

# 4. 验证历史记录
curl http://localhost:8000/api/v1/features/engineering/features?page=1
```

**4.2 仪表盘验证**:
- 访问特征工程监控页面
- 查看特征选择过程图表
- 验证历史记录显示

### 阶段5：文档更新（P1）

**更新内容**:
- 统一调度器架构文档：更新任务类型列表
- API文档：添加特征选择任务接口说明
- 部署文档：更新任务类型配置

## 3. 技术实现细节

### 3.1 任务参数设计

```python
class FeatureSelectionTaskParams(BaseModel):
    """特征选择任务参数"""
    symbols: List[str] = Field(..., description="股票代码列表")
    method: str = Field("importance", description="选择方法: importance/correlation/mutual_info/kbest")
    top_k: int = Field(10, description="选择前k个特征")
    min_quality: Optional[float] = Field(None, description="最小质量分数")
    target_column: Optional[str] = Field(None, description="目标变量列名")
```

### 3.2 任务执行流程

```
┌─────────────────────────────────────────────────────────┐
│                  特征选择任务执行流程                     │
├─────────────────────────────────────────────────────────┤
│ 1. 接收任务参数                                          │
│    └─ symbols, method, top_k, min_quality               │
│                                                          │
│ 2. 获取特征数据                                          │
│    └─ 调用get_features()获取特征列表                     │
│    └─ 按symbols过滤                                      │
│                                                          │
│ 3. 执行特征选择                                          │
│    └─ 初始化FeatureSelector                              │
│    └─ 调用select_features()                              │
│    └─ 获取选择结果                                       │
│                                                          │
│ 4. 记录历史                                              │
│    └─ 创建FeatureSelectionRecord                         │
│    └─ 保存到feature_selection_history表                  │
│                                                          │
│ 5. 返回结果                                              │
│    └─ selected_features, selection_ratio, metrics       │
└─────────────────────────────────────────────────────────┘
```

### 3.3 错误处理策略

| 错误类型 | 处理方式 | 重试策略 |
|---------|---------|---------|
| 参数验证失败 | 立即失败，返回错误信息 | 不重试 |
| 特征数据获取失败 | 失败，记录日志 | 指数退避，最多3次 |
| 特征选择执行失败 | 失败，记录详细错误 | 不重试 |
| 历史记录保存失败 | 警告，任务成功 | 不重试 |

## 4. 验收标准

- [ ] `get_task_types()` API返回包含`feature_selection`
- [ ] 可成功提交特征选择任务
- [ ] 任务执行完成且状态为completed
- [ ] feature_selection_history表有数据
- [ ] 特征选择过程仪表盘显示图表
- [ ] 单元测试覆盖率>80%
- [ ] 集成测试通过

## 5. 风险与应对

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|---------|
| FeatureSelector接口不兼容 | 中 | 高 | 提前验证接口，必要时封装适配器 |
| 历史记录表结构不匹配 | 低 | 中 | 检查表结构，必要时迁移 |
| 性能问题（大数据量） | 中 | 中 | 实现分批处理，添加超时控制 |
| 并发执行冲突 | 低 | 中 | 使用数据库锁或分布式锁 |

## 6. 实施时间表

| 阶段 | 预计时间 | 依赖 |
|------|---------|------|
| 阶段1：暴露API | 30分钟 | 无 |
| 阶段2：实现处理器 | 2小时 | 阶段1 |
| 阶段3：注册处理器 | 30分钟 | 阶段2 |
| 阶段4：测试验证 | 1小时 | 阶段3 |
| 阶段5：文档更新 | 30分钟 | 阶段4 |
| **总计** | **4.5小时** | - |

## 7. 后续Phase规划

### Phase 2：功能完善（下周）
- 添加特征选择结果评估
- 实现选择质量自动评估
- 完善历史记录查询API

### Phase 3：监控优化（后续）
- 添加特征选择性能指标
- 实现选择策略推荐
- 优化仪表盘交互体验
