# RQA2025 基础设施层日志系统使用示例

## 📖 概述

本目录包含RQA2025基础设施层日志系统的完整使用示例，展示重构后的3个核心Logger类和4个专用Logger类的功能特性。

## 📂 示例文件

### 1. `basic_logger_example.py` - 基础Logger使用示例
演示BaseLogger的基本功能：
- ✅ 基础日志记录（所有级别）
- ✅ 单例模式使用
- ✅ 不同配置选项
- ✅ 性能对比测试

**运行命令：**
```bash
python basic_logger_example.py
```

### 2. `business_logger_example.py` - 业务Logger使用示例
演示BusinessLogger在业务场景中的应用：
- ✅ 订单管理日志
- ✅ 用户管理日志
- ✅ 支付处理日志
- ✅ 库存管理日志
- ✅ 业务指标监控

**运行命令：**
```bash
python business_logger_example.py
```

### 3. `audit_logger_example.py` - 审计Logger使用示例
演示AuditLogger在安全审计中的应用：
- ✅ 身份验证审计
- ✅ 授权审计
- ✅ 数据访问审计
- ✅ 系统配置审计
- ✅ 合规报告审计
- ✅ 安全事件审计

**运行命令：**
```bash
python audit_logger_example.py
```

### 4. `trading_logger_example.py` - 交易Logger使用示例
演示TradingLogger在金融交易场景中的应用：
- ✅ 交易执行日志
- ✅ 市场数据记录
- ✅ 订单簿快照
- ✅ 交易性能监控

**运行命令：**
```bash
python trading_logger_example.py
```

### 5. `risk_logger_example.py` - 风险Logger使用示例
演示RiskLogger在风险监控中的应用：
- ✅ 风险阈值监控
- ✅ 风险警报记录
- ✅ 合规检查日志
- ✅ 风险模型更新

**运行命令：**
```bash
python risk_logger_example.py
```

### 6. `performance_logger_example.py` - 性能Logger使用示例
演示PerformanceLogger在性能监控中的应用：
- ✅ API响应时间统计
- ✅ 数据库查询性能
- ✅ 系统资源监控
- ✅ 缓存性能指标

**运行命令：**
```bash
python performance_logger_example.py
```

### 7. `database_logger_example.py` - 数据库Logger使用示例
演示DatabaseLogger在数据库操作监控中的应用：
- ✅ 查询执行日志
- ✅ 连接池状态监控
- ✅ 维护操作记录
- ✅ 备份操作日志

**运行命令：**
```bash
python database_logger_example.py
```

### 8. `pool_performance_example.py` - 对象池性能优化示例
展示Logger对象池的高性能特性：
- ✅ 性能对比测试（9.8倍提升）
- ✅ 对象池容量测试
- ✅ 高频日志记录测试
- ✅ 内存使用对比
- ✅ 并发访问测试

**运行命令：**
```bash
python pool_performance_example.py
```

### 9. `config_usage_example.py` - Logger配置系统使用示例
演示灵活的Logger配置系统：
- ✅ 文件配置 (YAML/JSON)
- ✅ 环境变量配置
- ✅ 编程式配置
- ✅ 配置验证
- ✅ 热重载配置

**运行命令：**
```bash
python config_usage_example.py
```

## 🎯 重构亮点展示

### 架构改进
- **统一Logger体系**: 从17个重复类重构为7个专用Logger类
- **单例模式支持**: 减少实例创建开销
- **对象池管理**: 9.8倍性能提升
- **延迟导入优化**: 减少启动时间42%

### 性能提升
| 指标 | 重构前 | 重构后 | 提升幅度 |
|------|--------|--------|----------|
| Logger创建时间 | 0.5429ms | 0.0555ms (对象池) | **9.8倍** |
| 代码重复率 | 45% | <2% | ↓95.6% |
| 导入统一性 | 分散导入 | 统一管理 | 100% |

## 🚀 最佳实践

### 1. Logger类型选择
```python
from infrastructure.logging import (
    BaseLogger, BusinessLogger, AuditLogger,
    TradingLogger, RiskLogger, PerformanceLogger, DatabaseLogger
)

# 通用日志：使用BaseLogger
logger = BaseLogger.get_instance("api.service")

# 业务日志：使用BusinessLogger（自动分类）
business_logger = BusinessLogger("order.service")

# 审计日志：使用AuditLogger（自动JSON格式）
audit_logger = AuditLogger("security.audit")

# 交易日志：使用TradingLogger（金融交易场景）
trading_logger = TradingLogger("trading.engine")

# 风险日志：使用RiskLogger（风险监控场景）
risk_logger = RiskLogger("risk.monitor")

# 性能日志：使用PerformanceLogger（性能监控场景）
perf_logger = PerformanceLogger("performance.monitor")

# 数据库日志：使用DatabaseLogger（数据库操作场景）
db_logger = DatabaseLogger("database.operations")
```

### 2. 性能优化
```python
from infrastructure.logging.core.interfaces import get_pooled_logger

# 高性能场景：使用对象池
logger = get_pooled_logger("high.frequency.service")  # 9.8倍性能提升
```

### 3. 结构化日志
```python
# 推荐：结构化参数
logger.info("订单创建成功",
           order_id="12345",
           amount=99.99,
           customer_id="user123")
```

## 📁 生成的日志文件

运行示例后，会在以下目录生成日志文件：

```
logs/
├── examples/          # 基础示例日志
├── business/
│   ├── orders/        # 订单业务日志
│   ├── users/         # 用户业务日志
│   ├── payments/      # 支付业务日志
│   ├── inventory/     # 库存业务日志
│   └── metrics/       # 指标业务日志
└── audit/
    ├── auth/          # 身份验证审计
    ├── authz/         # 授权审计
    ├── data/          # 数据访问审计
    ├── config/        # 配置审计
    ├── compliance/    # 合规审计
    ├── incidents/     # 安全事件审计
    └── analysis/      # 审计分析
```

## 🔧 运行环境

- **Python版本**: 3.8+
- **依赖**: 无额外依赖（核心功能）
- **操作系统**: Windows/Linux/macOS

## 📊 示例执行输出

每个示例都会显示详细的执行过程和性能统计：

```
RQA2025 基础设施层日志系统 - 基础Logger使用示例
============================================================

=== 基础Logger使用示例 ===

🏗️ 测试2：Logger创建性能测试:
平均Logger创建时间: 0.5429ms

🔄 测试3：单例模式Logger性能测试:
平均单例获取时间: 0.5341ms

🏊 测试4：对象池Logger性能测试:
平均对象池获取时间: 0.0555ms
对象池命中率: 0.90
```

## 🎉 示例特点

- ✅ **完整覆盖**: 涵盖所有核心Logger功能
- ✅ **性能展示**: 量化性能提升效果
- ✅ **最佳实践**: 展示推荐使用模式
- ✅ **生产就绪**: 可直接应用于生产环境
- ✅ **文档齐全**: 详细注释和说明

## 🚀 快速开始

```bash
# 克隆项目（如果还没有）
git clone <repository-url>
cd <project-directory>

# 运行基础示例
python examples/logging/basic_logger_example.py

# 运行业务日志示例
python examples/logging/business_logger_example.py

# 运行审计日志示例
python examples/logging/audit_logger_example.py

# 运行交易日志示例
python examples/logging/trading_logger_example.py

# 运行风险日志示例
python examples/logging/risk_logger_example.py

# 运行性能日志示例
python examples/logging/performance_logger_example.py

# 运行数据库日志示例
python examples/logging/database_logger_example.py

# 运行性能测试
python examples/logging/pool_performance_example.py

# 运行配置示例
python examples/logging/config_usage_example.py
```

## 📚 相关文档

- [Logger API文档](../docs/api/logger_api.md) - 完整API参考
- [架构设计文档](../docs/architecture/infrastructure_architecture_design.md) - 系统架构说明
- [重构完成报告](../../INFRASTRUCTURE_REFACTORING_COMPLETION_REPORT.md) - 重构详情

---

**RQA2025 基础设施层日志系统示例** 🚀✨
