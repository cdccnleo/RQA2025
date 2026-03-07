# RQA2025 团队培训指南

## 概述

本指南旨在帮助团队成员快速理解和应用新的代码组织规范和导入标准。

## 培训目标

1. **理解分层架构**: 掌握基础设施层和通用工具层的职责分工
2. **掌握导入规范**: 学会正确使用推荐和不推荐的导入方式
3. **避免代码重复**: 了解如何避免和修复代码重复问题
4. **使用工具链**: 掌握代码质量检查和监控工具的使用

## 核心概念

### 1. 分层架构

#### 基础设施层 (`src/infrastructure/utils/`)
- **定位**: 系统级功能、高级特性、复杂配置
- **特点**: 完整、稳定、支持复杂配置
- **使用场景**: 
  - 系统级组件开发
  - 高级功能需求
  - 性能监控和分析
  - 复杂的日志配置

#### 通用工具层 (`src/utils/`)
- **定位**: 简化API、业务场景、重定向实现
- **特点**: 专注于特定业务场景，重定向到基础设施层
- **使用场景**:
  - 日常开发
  - 标准功能需求
  - 业务逻辑实现

### 2. 导入规范

#### ✅ 推荐导入方式
```python
# 标准功能 - 使用通用层接口
from src.utils.logger import get_logger
from src.utils.date_utils import convert_timezone, get_business_date
from src.utils.math_utils import calculate_returns
from src.utils.data_utils import normalize_data
```

#### ⚠️ 特殊情况导入
```python
# 仅在需要高级功能时使用基础设施层
from src.infrastructure.utils.logger import LoggerFactory, configure_logging
from src.infrastructure.utils.date_utils import DateUtils
```

#### ❌ 不推荐的导入方式
```python
# 避免直接导入基础设施层标准功能
from src.infrastructure.utils.logger import get_logger  # 不推荐
from src.infrastructure.utils.date_utils import convert_timezone  # 不推荐
```

## 实践练习

### 练习1: 正确的导入选择

**场景**: 开发一个新的数据处理模块

```python
# 任务: 选择合适的导入方式

# 1. 需要基本的日志功能
# 选择: A) from src.utils.logger import get_logger
#        B) from src.infrastructure.utils.logger import get_logger

# 2. 需要时区转换功能
# 选择: A) from src.utils.date_utils import convert_timezone
#        B) from src.infrastructure.utils.date_utils import convert_timezone

# 3. 需要高级日志配置
# 选择: A) from src.utils.logger import LoggerFactory
#        B) from src.infrastructure.utils.logger import LoggerFactory
```

**答案**:
1. A - 基本日志功能使用通用层
2. A - 时区转换使用通用层
3. B - 高级配置使用基础设施层

### 练习2: 代码重构

**原始代码**:
```python
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.utils.date_utils import convert_timezone

def process_data(data):
    logger = get_logger(__name__)
    logger.info("开始处理数据")
    
    # 时区转换
    converted_time = convert_timezone(data['timestamp'], 'UTC', 'Asia/Shanghai')
    
    return processed_data
```

**重构后**:
```python
from src.utils.logger import get_logger
from src.utils.date_utils import convert_timezone

def process_data(data):
    logger = get_logger(__name__)
    logger.info("开始处理数据")
    
    # 时区转换
    converted_time = convert_timezone(data['timestamp'], 'UTC', 'Asia/Shanghai')
    
    return processed_data
```

### 练习3: 架构设计

**场景**: 设计一个新的工具模块

**需求分析**:
1. 提供基本的数学计算功能
2. 支持缓存机制
3. 需要性能监控

**设计方案**:
```python
# 通用层: src/utils/math_utils.py
from src.infrastructure.utils.cache_utils import model_cache
from src.infrastructure.utils.monitoring import performance_monitor

@model_cache()
@performance_monitor
def calculate_returns(prices):
    """计算收益率 - 通用层接口"""
    # 重定向到基础设施层实现
    from src.infrastructure.utils.math_utils import _calculate_returns_impl
    return _calculate_returns_impl(prices)

# 基础设施层: src/infrastructure/utils/math_utils.py
def _calculate_returns_impl(prices):
    """计算收益率 - 基础设施层实现"""
    # 完整的实现逻辑
    pass
```

## 工具使用

### 1. 代码质量检查

```bash
# 检查导入一致性
python scripts/development/check_import_consistency.py

# 检查代码重复
python scripts/development/migrate_imports.py --dry-run

# 运行验证测试
python scripts/development/verify_migration.py

# 完整质量监控
python scripts/development/code_quality_monitor.py
```

### 2. 代码审查清单

#### 提交前检查
- [ ] 是否使用了推荐的导入方式
- [ ] 是否避免了不推荐的导入
- [ ] 是否在必要时使用了高级功能导入
- [ ] 是否避免了循环导入

#### 功能检查
- [ ] 是否避免了重复实现
- [ ] 是否使用了统一的API接口
- [ ] 是否保持了向后兼容性
- [ ] 是否遵循了分层原则

### 3. 常见问题解决

#### Q1: 什么时候使用基础设施层？
**A1**: 仅在需要以下高级功能时：
- 复杂的日志配置
- 高级的日期时间处理
- 性能监控和分析
- 系统级工具功能

#### Q2: 如何处理现有的不推荐导入？
**A2**: 
1. 使用批量迁移脚本
2. 逐步替换为推荐导入
3. 保持向后兼容性
4. 更新相关文档

#### Q3: 如何确保导入一致性？
**A3**:
1. 使用自动化检查工具
2. 建立代码审查流程
3. 培训团队成员
4. 定期检查和更新

## 最佳实践

### 1. 开发流程

#### 新功能开发
1. **需求分析**: 确定功能属于哪个层次
2. **接口设计**: 设计统一的API接口
3. **实现开发**: 在合适的层次实现功能
4. **测试验证**: 编写相关测试用例
5. **文档更新**: 更新相关文档

#### 代码重构
1. **问题识别**: 使用工具识别问题
2. **影响分析**: 评估修改的影响范围
3. **逐步迁移**: 分步骤进行迁移
4. **测试验证**: 确保功能正常
5. **文档更新**: 更新相关文档

### 2. 团队协作

#### 代码审查
- 使用统一的审查清单
- 重点关注导入规范
- 检查架构分层合规性
- 验证代码质量

#### 知识分享
- 定期分享最佳实践
- 讨论架构设计决策
- 分享工具使用经验
- 更新培训材料

### 3. 持续改进

#### 定期评估
- 每月运行质量监控
- 分析问题趋势
- 优化工具和流程
- 更新规范和指南

#### 反馈机制
- 收集团队反馈
- 改进培训内容
- 优化工具功能
- 完善文档体系

## 考核标准

### 初级开发者
- [ ] 理解分层架构概念
- [ ] 掌握推荐导入方式
- [ ] 能够使用基本工具
- [ ] 遵循代码审查流程

### 中级开发者
- [ ] 能够设计合理的架构
- [ ] 能够重构现有代码
- [ ] 能够指导初级开发者
- [ ] 能够改进工具和流程

### 高级开发者
- [ ] 能够制定架构规范
- [ ] 能够设计工具链
- [ ] 能够培训团队成员
- [ ] 能够推动持续改进

## 资源链接

### 文档
- [代码重复定义分析报告](code_duplication_analysis.md)
- [统一导入规范](import_standards.md)
- [代码审查指南](code_review_guidelines.md)

### 工具
- [批量迁移脚本](../scripts/development/migrate_imports.py)
- [导入一致性检查](../scripts/development/check_import_consistency.py)
- [验证测试脚本](../scripts/development/verify_migration.py)
- [质量监控脚本](../scripts/development/code_quality_monitor.py)

### 示例
- [测试用例](../tests/unit/utils/test_import_migration.py)
- [最佳实践示例](../examples/)

## 版本历史

- **v1.0** (2025-01-19): 初始版本
- **v1.1** (2025-01-19): 添加实践练习
- **v1.2** (2025-01-19): 完善工具使用说明 