# RQA2025 架构变更记录

## 变更概览

本文档记录RQA2025量化交易系统架构的重要变更历史。

---

## 2026-02-23: 数据采集自动调度功能

### 变更类型
- **类型**: 功能增强
- **影响范围**: 网关层、数据管理层、分布式协调器层
- **变更级别**: 高
- **向后兼容**: 是

### 变更背景
原有数据采集流程需要手动触发，已启用的数据源不会根据配置的采集频率自动执行采集，导致数据更新不及时。需要实现自动化的数据采集调度机制。

### 变更内容

#### 1. 新增核心组件
| 组件 | 位置 | 功能 | 说明 |
|------|------|------|------|
| `rate_limit_parser.py` | `src/gateway/web/` | 采集频率解析器 | 解析"1次/天"等格式为秒数间隔 |
| `data_collection_scheduler_manager.py` | `src/gateway/web/` | 采集调度管理器 | 定时检查并自动生成采集任务 |
| `collection_history_manager.py` | `src/gateway/web/` | 采集历史管理器 | 管理采集历史记录和统计 |

#### 2. 数据库表
```sql
CREATE TABLE data_collection_history (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(100) NOT NULL,
    collection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL,
    records_collected INTEGER,
    error_message TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_ms INTEGER,
    task_id VARCHAR(100),
    collection_type VARCHAR(50) DEFAULT 'scheduled'
);
```

#### 3. 新增API端点
| 方法 | 端点 | 功能 |
|------|------|------|
| POST | `/api/v1/data/scheduler/auto-collection/start` | 启动自动采集 |
| POST | `/api/v1/data/scheduler/auto-collection/stop` | 停止自动采集 |
| GET | `/api/v1/data/scheduler/auto-collection/status` | 获取自动采集状态 |
| GET | `/api/v1/data/collection/history` | 获取采集历史记录 |
| GET | `/api/v1/data/collection/history/{source_id}` | 获取指定数据源历史 |
| GET | `/api/v1/data/collection/stats` | 获取采集统计信息 |

#### 4. 修改的组件
- `data_collector_worker.py`: 采集成功后自动更新 `last_test` 字段
- `datasource_routes.py`: 添加自动采集和采集历史API路由
- `data-sources-config.html`: 添加自动采集状态显示和控制按钮

### 架构改进

#### 自动化能力
- **采集频率解析**: 支持多种格式（"X次/天"、"X次/小时"、"X次/分钟"）
- **定时任务调度**: 每分钟检查已启用的数据源
- **任务去重**: 避免重复提交相同采集任务
- **历史记录**: 完整的采集历史追踪和统计

#### 架构符合度
| 维度 | 优化前 | 优化后 |
|------|--------|--------|
| 采集自动化 | 手动触发 | 自动调度 |
| 时间更新 | 手动测试时更新 | 采集成功后自动更新 |
| 历史追踪 | 无 | 完整记录 |
| 可观测性 | 有限 | 实时监控 |

### 影响分析

#### 正面影响
1. **数据及时性**: 已启用数据源按配置频率自动采集
2. **运维效率**: 减少手动操作，降低人为错误
3. **可观测性**: 完整的采集历史和统计信息
4. **灵活性**: 支持手动和自动两种采集模式

#### 更新文档
- `docs/architecture/data_layer_architecture_design.md` (v2.2)
- `docs/architecture/gateway_layer_architecture_design.md` (v2.3)

---

## 2026-02-15: 统一工作节点注册表迁移

### 变更类型
- **类型**: 架构优化
- **影响范围**: 分布式协调器层、特征层、ML层
- **变更级别**: 中等
- **向后兼容**: 是（通过更新导入路径）

### 变更背景
统一工作节点注册表 (`UnifiedWorkerRegistry`) 原本位于特征层 (`src/features/distributed/`)，但其服务对象涵盖全系统各层级（特征层、ML层、推理层、数据层），架构位置不够准确，违反了单一职责原则。

### 变更内容

#### 1. 文件迁移
| 文件 | 原位置 | 新位置 | 说明 |
|------|--------|--------|------|
| `unified_worker_registry.py` | `src/features/distributed/` | `src/distributed/registry/` | 核心迁移 |
| `__init__.py` | - | `src/distributed/registry/__init__.py` | 新增模块初始化 |
| `cluster_manager.py` | - | `src/distributed/coordinator/` | 新增集群管理器 |

#### 2. 目录结构变化
```
变更前:
src/features/distributed/
├── unified_worker_registry.py    # 待迁移
├── task_scheduler.py
├── worker_manager.py
└── worker_executor.py

变更后:
src/distributed/
├── registry/                      # 新增目录
│   ├── __init__.py
│   └── unified_worker_registry.py
├── coordinator/
│   ├── cluster_manager.py        # 新增
│   └── ...
├── consistency/
└── discovery/

src/features/distributed/
├── task_scheduler.py             # 更新导入路径
├── worker_manager.py             # 更新导入路径
└── worker_executor.py
```

#### 3. 导入路径更新
```python
# 旧导入方式（已废弃）
from src.features.distributed.unified_worker_registry import (
    get_unified_worker_registry,
    WorkerType,
    WorkerStatus
)

# 新导入方式
from src.distributed.registry import (
    get_unified_worker_registry,
    WorkerType,
    WorkerStatus,
    WorkerNode
)
```

#### 4. 更新的文件列表
- `src/features/distributed/task_scheduler.py`
- `src/features/distributed/worker_manager.py`
- `src/ml/training/training_executor_manager.py`
- `src/ml/training/async_training_manager.py`
- `src/distributed/__init__.py`

### 架构改进

#### 职责分离
- **分布式协调器层**: 负责节点注册、集群管理、服务发现
- **特征层**: 专注于特征工程、特征处理
- **ML层**: 专注于模型训练、推理

#### 架构符合度
| 维度 | 优化前 | 优化后 |
|------|--------|--------|
| 层级归属 | 特征层（不准确） | 分布式协调器层（正确） |
| 职责清晰度 | 部分混合 | 清晰分离 |
| 全局服务 | 隐式依赖 | 显式依赖 |

### 影响分析

#### 正面影响
1. **架构清晰度提升**: 全局服务组件归位到正确的架构层级
2. **职责分离明确**: 各层职责更加清晰
3. **可维护性增强**: 统一的节点管理入口
4. **扩展性改善**: 支持更多类型的工作节点

#### 潜在风险
1. **导入路径变更**: 需要更新所有引用文件
2. **文档同步**: 需要同步更新架构设计文档

### 缓解措施
1. ✅ 已更新所有引用文件的导入路径
2. ✅ 已更新分布式协调器架构设计文档
3. ✅ 已更新特征层架构设计文档
4. ✅ 已更新核心服务层架构设计文档
5. ✅ 已创建架构变更记录文档

### 验证结果
- ✅ 容器构建成功
- ✅ 系统启动正常
- ✅ 训练执行器注册正常
- ✅ 特征任务调度正常
- ✅ 无离线节点警告

### 相关文档更新
- `docs/architecture/distributed_coordinator_architecture_design.md` (v2.2)
- `docs/architecture/feature_layer_architecture_design.md` (v3.1)
- `docs/architecture/core_service_layer_architecture_design.md` (v6.3)
- `docs/architecture/ARCHITECTURE_CHANGE_LOG.md` (本文件)

### 变更负责人
- **负责人**: RQA2025 Team
- **审核人**: 架构委员会
- **实施日期**: 2026-02-15

---

## 变更记录模板

### 变更类型标识
- **[ARCH]**: 架构调整
- **[REFACTOR]**: 代码重构
- **[FEATURE]**: 新功能
- **[FIX]**: 问题修复
- **[DOCS]**: 文档更新

### 变更级别
- **重大**: 影响系统核心架构，需要全面测试
- **中等**: 影响部分模块，需要局部测试
- **轻微**: 不影响功能，仅代码组织优化

---

## 附录

### A. 架构层级说明
RQA2025系统采用21层架构设计：
1. 基础设施层 (Infrastructure Layer)
2. 核心服务层 (Core Service Layer)
3. 数据管理层 (Data Management Layer)
4. 特征分析层 (Feature Layer)
5. 机器学习层 (ML Layer)
6. 策略服务层 (Strategy Layer)
7. 交易层 (Trading Layer)
8. 风险控制层 (Risk Control Layer)
9. 监控层 (Monitoring Layer)
10. 流处理层 (Streaming Layer)
11. 网关层 (Gateway Layer)
12. 优化层 (Optimization Layer)
13. 适配器层 (Adapter Layer)
14. 自动化层 (Automation Layer)
15. 弹性层 (Resilience Layer)
16. 测试层 (Testing Layer)
17. 工具层 (Utils Layer)
18. 分布式协调器层 (Distributed Coordinator Layer)
19. 异步处理器层 (Async Processor Layer)
20. 移动端层 (Mobile Layer)
21. 业务边界层 (Boundary Coordinator Layer)

### B. 参考资料
- [业务流程驱动架构设计](BUSINESS_PROCESS_DRIVEN_ARCHITECTURE.md)
- [架构总览](ARCHITECTURE_OVERVIEW.md)
- [分布式协调器架构设计](distributed_coordinator_architecture_design.md)
