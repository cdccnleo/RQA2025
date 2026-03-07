# RQA2025 代码审查指南

## 概述

为确保代码质量和一致性，建立统一的代码审查流程和标准。

## 审查检查点

### 1. 导入规范检查

#### ✅ 推荐导入方式
```python
# 标准功能
from src.utils.logger import get_logger
from src.utils.date_utils import convert_timezone, get_business_date
from src.utils.math_utils import calculate_returns
from src.utils.data_utils import normalize_data
```

#### ⚠️ 特殊情况导入
```python
# 仅在需要高级功能时使用
from src.infrastructure.utils.logger import LoggerFactory, configure_logging
from src.infrastructure.utils.date_utils import DateUtils
```

#### ❌ 不推荐的导入方式
```python
# 避免直接导入基础设施层标准功能
from src.infrastructure.utils.logger import get_logger  # 不推荐
from src.infrastructure.utils.date_utils import convert_timezone  # 不推荐
```

### 2. 代码重复检查

#### 检查项目
- [ ] 是否存在重复的函数定义
- [ ] 是否存在重复的类定义
- [ ] 是否存在重复的常量定义
- [ ] 是否存在重复的导入语句

#### 重复代码处理原则
1. **优先使用通用层接口**
2. **重定向到基础设施层实现**
3. **保留业务专用功能**
4. **避免循环依赖**

### 3. 架构分层检查

#### 基础设施层 (`src/infrastructure/`)
- **职责**: 系统级功能、高级特性、复杂配置
- **使用场景**: 系统组件、高级功能需求
- **检查点**:
  - [ ] 是否提供了完整的基础功能
  - [ ] 是否支持复杂配置
  - [ ] 是否适合系统级使用

#### 通用工具层 (`src/utils/`)
- **职责**: 简化API、业务场景、重定向实现
- **使用场景**: 日常开发、标准功能需求
- **检查点**:
  - [ ] 是否提供了简化的接口
  - [ ] 是否重定向到基础设施层
  - [ ] 是否专注于特定业务场景

### 4. 代码质量检查

#### 功能检查
- [ ] 函数功能是否单一明确
- [ ] 是否避免了循环依赖
- [ ] 是否保持了向后兼容性
- [ ] 是否使用了统一的API接口

#### 性能检查
- [ ] 导入是否高效
- [ ] 是否避免了不必要的依赖
- [ ] 是否合理使用了缓存机制
- [ ] 是否避免了内存泄漏

#### 安全检查
- [ ] 是否处理了异常情况
- [ ] 是否验证了输入参数
- [ ] 是否保护了敏感信息
- [ ] 是否遵循了安全最佳实践

### 5. 文档检查

#### 代码文档
- [ ] 函数是否有清晰的文档字符串
- [ ] 参数和返回值是否明确说明
- [ ] 是否有使用示例
- [ ] 是否说明了异常情况

#### 架构文档
- [ ] 是否更新了相关文档
- [ ] 是否说明了设计决策
- [ ] 是否记录了重要的变更
- [ ] 是否提供了迁移指南

## 审查流程

### 1. 提交前检查
```bash
# 运行代码质量检查
python scripts/development/check_import_consistency.py

# 运行验证测试
python scripts/development/verify_migration.py

# 运行相关单元测试
python run_tests.py --env rqa --test-file <test_file>
```

### 2. 审查清单

#### 导入检查
- [ ] 是否使用了推荐的导入方式
- [ ] 是否避免了不推荐的导入
- [ ] 是否在必要时使用了高级功能导入
- [ ] 是否避免了循环导入

#### 功能检查
- [ ] 是否避免了重复实现
- [ ] 是否使用了统一的API接口
- [ ] 是否保持了向后兼容性
- [ ] 是否遵循了分层原则

#### 质量检查
- [ ] 代码是否清晰易读
- [ ] 是否处理了异常情况
- [ ] 是否验证了输入参数
- [ ] 是否遵循了编码规范

### 3. 审查反馈

#### 反馈格式
```
## 审查结果

### ✅ 通过的项目
- 导入规范符合要求
- 代码质量良好
- 文档完整

### ⚠️ 需要改进的项目
- [具体问题描述]
- [建议的解决方案]

### ❌ 需要修复的项目
- [严重问题描述]
- [必须修复的原因]
```

## 自动化工具

### 1. 导入一致性检查
```bash
python scripts/development/check_import_consistency.py --output reports/import_check.md
```

### 2. 代码重复检测
```bash
python scripts/development/migrate_imports.py --dry-run --verbose
```

### 3. 验证测试
```bash
python scripts/development/verify_migration.py
```

## 最佳实践

### 1. 导入规范
- 优先使用通用层接口
- 仅在需要高级功能时使用基础设施层
- 避免直接导入基础设施层标准功能

### 2. 代码组织
- 保持函数功能单一
- 避免循环依赖
- 使用统一的API接口

### 3. 文档维护
- 及时更新相关文档
- 记录重要的设计决策
- 提供清晰的使用示例

### 4. 测试验证
- 编写相关的单元测试
- 验证向后兼容性
- 检查导入一致性

## 常见问题

### Q1: 什么时候使用基础设施层？
A1: 仅在需要以下高级功能时：
- 复杂的日志配置
- 高级的日期时间处理
- 性能监控和分析
- 系统级工具功能

### Q2: 如何处理现有的不推荐导入？
A2: 
1. 使用批量迁移脚本
2. 逐步替换为推荐导入
3. 保持向后兼容性
4. 更新相关文档

### Q3: 如何确保导入一致性？
A3:
1. 使用自动化检查工具
2. 建立代码审查流程
3. 培训团队成员
4. 定期检查和更新

## 版本历史

- **v1.0** (2025-01-19): 初始版本
- **v1.1** (2025-01-19): 添加自动化工具说明
- **v1.2** (2025-01-19): 完善审查流程 