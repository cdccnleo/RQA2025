# 核心服务层快速参考

## 📂 目录结构速查

### 10个核心子目录

```
src/core/
├── foundation/              基础组件（BaseComponent、异常、接口、模式）
├── interfaces/             🆕 统一接口管理
├── event_bus/              事件总线v4.0
├── orchestration/          业务流程编排v2.0
├── integration/            统一集成层（适配器、降级服务）
├── container/              依赖注入容器
├── business_process/       业务流程管理
├── core_optimization/      核心层优化
├── core_services/          核心服务实现
└── utils/                  通用工具（仅2个文件）
```

---

## 🔍 快速查找指南

### 我的组件应该放在哪里？

| 组件类型 | 放置位置 | 示例 |
|---------|---------|------|
| 基础类、枚举、异常 | `foundation/` | BaseComponent, ComponentStatus |
| 标准接口定义 | `interfaces/` | CoreInterfaces, LayerInterfaces |
| 事件相关 | `event_bus/` | Event, EventHandler, EventBus |
| 流程编排 | `orchestration/` | Orchestrator, StateMachine |
| 适配器、集成 | `integration/` | DataAdapter, ServiceBridge |
| 依赖注入 | `container/` | DependencyContainer |
| 业务流程配置 | `business_process/` | ProcessConfig, ProcessMonitor |
| 核心层性能优化 | `core_optimization/` | OptimizationImplementer |
| 核心服务 | `core_services/` | BusinessService, ApiService |
| 通用工具函数 | `utils/` | AsyncProcessor, ServiceFactory |

---

## 📝 常用Import路径

### 基础组件

```python
# 基础类和枚举
from src.core.foundation.base import BaseComponent, ComponentStatus

# 异常
from src.core.foundation.exceptions.core_exceptions import EventBusException

# 接口
from src.core.interfaces.core_interfaces import IService
from src.core.interfaces.layer_interfaces import BaseLayerInterface
```

### 核心功能

```python
# 事件总线
from src.core.event_bus.core import EventBus
from src.core.event_bus.models import Event, EventHandler
from src.core.event_bus.types import EventType, EventPriority

# 业务流程编排
from src.core.orchestration.orchestrator_refactored import BusinessProcessOrchestrator
from src.core.orchestration.configs.process_config_loader import ProcessConfigLoader

# 依赖注入
from src.core.container.container import DependencyContainer
from src.core.container.service_container import ServiceContainer
```

### 集成和工具

```python
# 集成适配器
from src.core.integration.adapters.features_adapter import FeaturesLayerAdapter
from src.core.integration.core.business_adapters import UnifiedBusinessAdapterFactory

# 工具
from src.core.utils.async_processor_components import AsyncProcessor
from src.core.utils.service_factory import ServiceFactory
```

---

## ⚠️ 注意事项

### 命名规范

1. **核心层专用组件** 使用 `core_*` 前缀
   - ✅ `core_services/`
   - ✅ `core_optimization/`
   - ✅ `core_infrastructure/` (已删除)

2. **业务相关组件** 明确职责
   - ✅ `business_process/`（不是business）
   
3. **通用组件** 保持简洁
   - ✅ `foundation/`
   - ✅ `interfaces/`
   - ✅ `utils/`

### 常见错误

❌ **错误**: 
```python
from src.core.infrastructure.container import DependencyContainer
from src.core.business import ProcessConfig
from src.core.patterns import StandardComponent
```

✅ **正确**: 
```python
from src.core.container.container import DependencyContainer
from src.core.business_process.config import ProcessConfig
from src.core.foundation.patterns.standard_interface_template import StandardComponent
```

---

## 🎯 快速决策树

### 新增组件时如何选择目录？

```
开始
 │
 ├─ 是基础类/异常/接口？
 │  └─ Yes → foundation/
 │
 ├─ 是事件相关？
 │  └─ Yes → event_bus/
 │
 ├─ 是流程编排相关？
 │  └─ Yes → orchestration/
 │
 ├─ 是系统集成/适配器？
 │  └─ Yes → integration/
 │
 ├─ 是依赖注入相关？
 │  └─ Yes → container/
 │
 ├─ 是业务流程配置/监控？
 │  └─ Yes → business_process/
 │
 ├─ 是核心层性能优化？
 │  └─ Yes → core_optimization/
 │
 ├─ 是核心服务实现？
 │  └─ Yes → core_services/
 │
 ├─ 是通用工具函数？
 │  └─ Yes → utils/
 │
 └─ 都不是？
    └─ 重新思考职责定位或咨询架构团队
```

---

## 📞 获取帮助

### 文档索引

**快速入门**: 本文档  
**详细架构**: `CORE_FINAL_ARCHITECTURE.md`  
**重构过程**: `CORE_REFACTOR_COMPLETE_SUMMARY.md`  
**总体架构**: `ARCHITECTURE_OVERVIEW.md`

### 常见问题

**Q: 为什么有两个interfaces目录？**  
A: `core/interfaces/` 是新的统一目录，`foundation/interfaces/` 保留供现有代码向后兼容。

**Q: core_infrastructure去哪了？**  
A: 已删除。内容移至 orchestration/configs/ 和 infrastructure/security_core/

**Q: utils为什么只有2个文件？**  
A: 业务组件已移至业务层，仅保留通用工具（异步处理、服务工厂）。

**Q: 如何导入编排器？**  
A: `from src.core.orchestration.orchestrator_refactored import BusinessProcessOrchestrator`

---

## ✅ 检查清单

### 新增代码前

- [ ] 确定组件职责
- [ ] 选择正确目录
- [ ] 遵循命名规范
- [ ] 实现标准接口
- [ ] 编写单元测试
- [ ] 更新文档

### 代码审查时

- [ ] 职责是否单一？
- [ ] 位置是否合理？
- [ ] 命名是否明确？
- [ ] 有无重复代码？
- [ ] 测试是否完备？
- [ ] 文档是否更新？

---

**快速参考版本**: v1.0  
**架构版本**: v4.0 Final  
**最后更新**: 2025-01-XX

