# 核心模块文档

## 📋 模块概述

核心模块 (`src/core/`) 是系统的基础组件，提供核心功能和接口定义。基于业务流程驱动的架构，实现了事件总线、依赖注入容器、服务容器、业务流程编排器等核心服务。

## 🏗️ 模块结构

```
src/core/
├── __init__.py                    # 模块初始化和导出
├── base.py                       # 基础组件抽象和工具函数
├── exceptions.py                 # 统一异常处理机制
├── event_bus.py                  # 事件总线实现
├── container.py                  # 依赖注入容器
├── service_container.py          # 服务容器管理
├── business_process_orchestrator.py  # 业务流程编排器
├── architecture_layers.py        # 架构层实现
├── layer_interfaces.py           # 层接口定义
└── optimizations/                # 优化实现
    ├── optimization_implementer.py    # 优化任务管理
    ├── short_term_optimizations.py    # 短期优化
    ├── medium_term_optimizations.py   # 中期优化
    └── long_term_optimizations.py     # 长期优化
```

## 📚 文档索引

### 核心组件
- [核心层使用指南](CORE_LAYER_USAGE_GUIDE.md) - 详细使用指南和最佳实践
- [核心层优化完成报告](CORE_LAYER_OPTIMIZATION_COMPLETION_REPORT.md) - 核心层优化成果和后续建议
- [核心层性能测试报告](CORE_LAYER_PERFORMANCE_REPORT.md) - 核心层性能基准和优化效果
- [优化建议实现报告](OPTIMIZATION_IMPLEMENTATION_REPORT.md) - 所有优化阶段的详细实施报告
- [API文档](API_DOCUMENTATION.md) - 完整的API接口文档和使用示例

## 🚀 快速开始

### 基本导入

```python
from src.core import (
    # 基础组件
    BaseComponent,
    BaseService,
    ComponentStatus,
    ComponentHealth,
    ComponentInfo,
    generate_id,
    validate_config,
    retry_on_failure,

    # 核心服务
    EventBus,
    DependencyContainer,
    ServiceContainer,
    BusinessProcessOrchestrator,

    # 异常处理
    CoreException,
    EventBusException,
    ContainerException,
    OrchestratorException,
    ServiceException,
    ConfigurationException,
    ValidationException,
    StateTransitionException,
    HealthCheckException,

    # 架构层
    CoreServicesLayer,
    InfrastructureLayer,
    DataManagementLayer,
    FeatureProcessingLayer,
    ModelInferenceLayer,
    StrategyDecisionLayer,
    RiskComplianceLayer,
    TradingExecutionLayer,
    MonitoringFeedbackLayer
)
```

### 基本使用示例

```python
# 1. 事件总线使用
event_bus = EventBus()
event_bus.initialize()

def data_handler(event):
    print(f"收到数据事件: {event.data}")

event_bus.subscribe(EventType.DATA_READY, data_handler)
event_id = event_bus.publish(EventType.DATA_READY, {"symbol": "000001.SZ"})

# 2. 依赖注入容器使用
container = DependencyContainer()
container.register("data_service", DataService(), lifecycle=Lifecycle.SINGLETON)
data_service = container.get("data_service")

# 3. 业务流程编排器使用
orchestrator = BusinessProcessOrchestrator()
orchestrator.initialize()
process_id = orchestrator.start_trading_cycle(
    symbols=["000001.SZ"],
    strategy_config={"type": "momentum"}
)
```

## 🧪 测试

### 运行所有测试

```bash
# 运行所有核心测试
python -m pytest tests/unit/core/ -v

# 运行特定测试
python -m pytest tests/unit/core/test_event_bus.py -v
python -m pytest tests/unit/core/test_container.py -v
python -m pytest tests/unit/core/test_orchestrator.py -v
```

### 测试覆盖

- ✅ 事件总线测试 (8个测试)
- ✅ 依赖注入容器测试 (3个测试)
- ✅ 业务流程编排器测试 (19个测试)
- ✅ 服务容器测试 (6个测试)
- ✅ 增强版核心服务测试 (15个测试)
- ✅ 边界测试 (4个测试)
- ✅ 接口测试 (10个测试)
- ✅ 优化测试 (15个测试)

**总计**: 92个测试通过，1个跳过，2个警告

## 📊 性能指标

### 事件总线性能
- **吞吐量**: 72.24 ops/s (1000个事件)
- **内存使用**: 1.23MB
- **CPU使用**: 36.90%

### 依赖注入容器性能
- **吞吐量**: 199.68 ops/s (1000个服务)
- **内存使用**: 1.02MB
- **CPU使用**: 0.30%

## 🔧 配置

### 环境变量

```bash
# 事件总线配置
EVENT_BUS_MAX_WORKERS=10
EVENT_BUS_ENABLE_ASYNC=true
EVENT_BUS_ENABLE_PERSISTENCE=true

# 容器配置
CONTAINER_MAX_SERVICES=1000
CONTAINER_ENABLE_HEALTH_CHECK=true

# 编排器配置
ORCHESTRATOR_MAX_INSTANCES=100
ORCHESTRATOR_CONFIG_DIR=config/processes
```

### 配置文件

```yaml
# config/core.yaml
event_bus:
  max_workers: 10
  enable_async: true
  enable_persistence: true
  batch_size: 10
  max_queue_size: 10000

container:
  max_services: 1000
  enable_health_check: true
  auto_discovery: false

orchestrator:
  max_instances: 100
  config_dir: config/processes
  auto_cleanup: true
```

## 🐛 故障排除

### 常见问题

1. **事件总线未初始化**
   ```python
   # 确保在使用前调用initialize()
   event_bus = EventBus()
   event_bus.initialize()
   ```

2. **服务注册失败**
   ```python
   # 检查服务名称是否已存在
   if not container.has("service_name"):
       container.register("service_name", service)
   ```

3. **业务流程状态错误**
   ```python
   # 检查编排器是否已初始化
   if orchestrator.get_status() == ComponentStatus.INITIALIZED:
       process_id = orchestrator.start_trading_cycle(symbols, config)
   ```

### 日志级别

```python
import logging

# 设置日志级别
logging.getLogger("src.core").setLevel(logging.DEBUG)
```

## 📈 监控

### 健康检查

```python
# 检查组件健康状态
if event_bus.health_check():
    print("事件总线健康")
else:
    print("事件总线异常")

if container.health_check():
    print("容器健康")
else:
    print("容器异常")
```

### 性能监控

```python
# 获取性能统计
event_stats = event_bus.get_event_statistics()
container_stats = container.get_statistics()
orchestrator_stats = orchestrator.get_process_metrics()
```

## 🤝 贡献

### 开发指南

1. **代码风格**: 遵循PEP 8规范
2. **测试**: 新功能必须包含测试
3. **文档**: 更新相关文档
4. **类型注解**: 使用类型注解

### 提交规范

```
feat: 添加新功能
fix: 修复bug
docs: 更新文档
test: 添加测试
refactor: 重构代码
style: 代码格式调整
```

## 📄 许可证

本项目采用MIT许可证，详见 [LICENSE](../../LICENSE) 文件。

## 📞 支持

如有问题或建议，请通过以下方式联系：

- 📧 邮箱: support@rqa2025.com
- 🐛 问题反馈: [GitHub Issues](https://github.com/rqa2025/issues)
- 📖 文档: [在线文档](https://docs.rqa2025.com)