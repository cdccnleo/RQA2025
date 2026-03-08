# 业务编排层架构迁移文档

## 迁移概述

**迁移日期**: 2026-03-08
**迁移版本**: v2.1.0
**迁移类型**: 架构重构

## 迁移原因

### 原架构问题
1. **循环依赖风险**: `src/infrastructure/orchestration` 大量依赖 `src/core` 模块
   - EventBus (`src/core/event_bus`)
   - BusinessProcessState (`src/core`)
   - foundation.base (`src/core/foundation`)

2. **复杂的相对导入**: 需要使用 `....core.xxx` 这样的深层相对导入

3. **架构层级混乱**:
   - **Core 层**: 应该包含核心业务逻辑和流程编排
   - **Infrastructure 层**: 应该提供底层技术支持（数据库、缓存、消息队列等）

## 迁移方案

### 目录结构变更

**迁移前**:
```
src/
├── core/
│   ├── event_bus/              # 事件总线
│   ├── business_process/       # 业务流程
│   └── ...
└── infrastructure/
    ├── orchestration/          # ❌ 业务流程编排（位置不当）
    │   ├── orchestrator_refactored.py
    │   ├── components/
    │   ├── configs/
    │   └── ...
    └── ...
```

**迁移后**:
```
src/
├── core/
│   ├── orchestration/          # ✅ 业务流程编排（核心业务层）
│   │   ├── orchestrator_refactored.py
│   │   ├── components/
│   │   ├── configs/
│   │   └── __init__.py
│   ├── event_bus/              # 事件总线
│   ├── business_process/       # 业务流程
│   └── ...
└── infrastructure/
    ├── orchestration/          # 保留基础设施相关的调度功能
    │   └── scheduler/          # 任务调度器
    ├── automation/             # 自动化引擎
    └── ...
```

### 导入路径变更

**迁移前**:
```python
# 从infrastructure导入（复杂路径）
from ..infrastructure.orchestration.orchestrator_refactored import BusinessProcessOrchestrator

# 内部导入（深层相对路径）
from ....core.event_bus.core import EventBus
from ....core.foundation.base import BaseComponent
```

**迁移后**:
```python
# 从core导入（简化路径）
from .orchestration.orchestrator_refactored import BusinessProcessOrchestrator

# 内部导入（同层调用）
from ..event_bus.core import EventBus
from ..foundation.base import BaseComponent
```

## 迁移内容

### 1. 创建新目录结构
```bash
mkdir -p src/core/orchestration/components
mkdir -p src/core/orchestration/configs
mkdir -p src/core/orchestration/business_process
mkdir -p src/core/orchestration/pool
```

### 2. 迁移文件
| 源文件 | 目标文件 | 说明 |
|--------|----------|------|
| `src/infrastructure/orchestration/orchestrator_refactored.py` | `src/core/orchestration/orchestrator_refactored.py` | 主编排器 |
| `src/infrastructure/orchestration/components/__init__.py` | `src/core/orchestration/components/__init__.py` | 组件导出 |
| `src/infrastructure/orchestration/configs/__init__.py` | `src/core/orchestration/configs/__init__.py` | 配置导出 |
| `src/infrastructure/orchestration/configs/orchestrator_configs.py` | `src/core/orchestration/configs/orchestrator_configs.py` | 配置类 |

### 3. 更新导入路径

#### orchestrator_refactored.py
```python
# 变更前
from src.core.constants import MAX_RETRIES, MAX_RECORDS
from .components import EventBus, ...
from .configs import OrchestratorConfig
from ...core import BusinessProcessState, EventType
from ...core.foundation.base import BaseComponent, ComponentStatus

# 变更后
from ..constants import MAX_RETRIES, MAX_RECORDS
from .components import EventBus, ...
from .configs import OrchestratorConfig
from .. import BusinessProcessState, EventType
from ..foundation.base import BaseComponent, ComponentStatus
```

#### components/__init__.py
```python
# 变更前
from ....core.event_bus.core import EventBus
from ....core.business_process.state_machine.state_machine import BusinessProcessStateMachine

# 变更后
from ...event_bus.core import EventBus
from ...business_process.state_machine.state_machine import BusinessProcessStateMachine
```

#### configs/orchestrator_configs.py
```python
# 变更前
from src.core.constants import MAX_RECORDS, ...

# 变更后
from ...constants import MAX_RECORDS, ...
```

### 4. 更新 src/core/__init__.py
```python
# 变更前
from ..infrastructure.orchestration.orchestrator_refactored import BusinessProcessOrchestrator

# 变更后
from .orchestration.orchestrator_refactored import BusinessProcessOrchestrator
```

### 5. 删除旧文件
```bash
rm src/infrastructure/orchestration/orchestrator_refactored.py
rm -rf src/infrastructure/orchestration/components
rm -rf src/infrastructure/orchestration/configs
```

## 架构优势

### 1. 消除循环依赖
- 编排器与 EventBus、BusinessProcess 同层调用
- 无需跨层导入

### 2. 简化导入路径
- 使用相对导入 `.event_bus` 而非 `....core.event_bus`
- 代码更清晰、维护更容易

### 3. 清晰的架构分层
- **Core 层**: 业务流程编排、事件驱动、状态管理
- **Infrastructure 层**: 底层技术实现、外部系统集成

### 4. 提高可维护性
- 核心业务逻辑集中在 core 层
- 职责单一，易于理解和修改

## 向后兼容性

### 导出接口保持不变
```python
# src/core/__init__.py 仍然导出 BusinessProcessOrchestrator
from src.core import BusinessProcessOrchestrator  # ✅ 仍然有效
```

### 类接口保持不变
- `BusinessProcessOrchestrator` 类接口完全兼容
- `OrchestratorConfig` 配置类完全兼容
- 所有方法签名保持不变

## 测试验证

### 构建验证
```bash
docker-compose -f docker-compose.prod.yml up -d --build app
```

### 功能验证
- ✅ 业务流程编排器初始化成功
- ✅ EventBus 集成正常
- ✅ 状态机工作正常
- ✅ 配置管理正常
- ✅ 流程监控正常

## Git 提交记录

```
commit: <待填充>
Author: AI Assistant
Date: 2026-03-08

refactor: 业务编排层架构迁移 - 从infrastructure迁移到core层

- 将业务流程编排器从 src/infrastructure/orchestration 迁移到 src/core/orchestration
- 简化导入路径，消除循环依赖
- 更新所有相对导入路径
- 保持100%向后兼容
- 更新架构设计文档

优势:
- 核心业务逻辑集中在core层
- 消除复杂的深层相对导入
- 清晰的架构分层
- 提高可维护性
```

## 相关文档

- [事件总线设计文档](./event_bus_design.md)
- [核心业务层架构](./core_layer_architecture.md)
- [基础设施层架构](./infrastructure_layer_architecture.md)

## 后续建议

1. **代码审查**: 检查其他模块是否有类似的架构问题
2. **文档更新**: 更新所有相关架构文档
3. **团队培训**: 向团队介绍新的架构分层
4. **监控观察**: 观察迁移后的系统稳定性

---

**迁移完成状态**: ✅ 已完成
**验证状态**: ✅ 已通过
**文档更新**: ✅ 已更新
