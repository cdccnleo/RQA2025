## 🚨 违规依赖修复

**违规**: core → core
- 文件: src\core\optimizations\short_term_optimizations.py:556
- 导入: `from src.core import EventBus, EventType, EventPriority`
- **建议**: 考虑使用依赖倒置或接口抽象

**违规**: core → core
- 文件: src\core\optimizations\short_term_optimizations.py:593
- 导入: `from src.core import DependencyContainer, Lifecycle`
- **建议**: 考虑使用依赖倒置或接口抽象

**违规**: core → core
- 文件: src\core\optimizations\short_term_optimizations.py:630
- 导入: `from src.core import BusinessProcessOrchestrator`
- **建议**: 考虑使用依赖倒置或接口抽象

**违规**: core → core
- 文件: src\core\optimizations\short_term_optimizations.py:871
- 导入: `from src.core import EventBus, EventType, EventPriority`
- **建议**: 考虑使用依赖倒置或接口抽象

**违规**: core → core
- 文件: src\core\optimizations\short_term_optimizations.py:935
- 导入: `from src.core import DependencyContainer, Lifecycle`
- **建议**: 考虑使用依赖倒置或接口抽象

**违规**: core → core
- 文件: src\core\optimizations\short_term_optimizations.py:996
- 导入: `from src.core import BusinessProcessOrchestrator, BusinessProcessState`
- **建议**: 考虑使用依赖倒置或接口抽象

**违规**: core → core
- 文件: src\core\optimizations\short_term_optimizations.py:1051
- 导入: `from src.core import EventBus, EventType, EventPriority`
- **建议**: 考虑使用依赖倒置或接口抽象

**违规**: core → core
- 文件: src\core\optimizations\short_term_optimizations.py:1101
- 导入: `from src.core import DependencyContainer, Lifecycle`
- **建议**: 考虑使用依赖倒置或接口抽象

**违规**: core → core
- 文件: src\core\optimizations\short_term_optimizations.py:1138
- 导入: `from src.core import EventBus, DependencyContainer, BusinessProcessOrchestrator`
- **建议**: 考虑使用依赖倒置或接口抽象

**违规**: core → core
- 文件: src\core\optimizations\short_term_optimizations.py:1190
- 导入: `from src.core import EventBus, DependencyContainer, BusinessProcessOrchestrator`
- **建议**: 考虑使用依赖倒置或接口抽象

## 🔄 循环依赖解决

**循环**: core → core
- **建议**: 提取共同接口或使用事件驱动模式

**循环**: infrastructure → infrastructure
- **建议**: 提取共同接口或使用事件驱动模式

## ⚡ 依赖关系优化建议

### 1. 依赖倒置原则

- 上层模块不应依赖下层模块，应依赖抽象

- 具体实现应依赖抽象接口


### 2. 接口分离原则

- 为每个模块提供专门的接口

- 避免大型通用接口


### 3. 依赖注入

- 使用依赖注入替代直接依赖

- 通过构造函数或setter注入依赖

