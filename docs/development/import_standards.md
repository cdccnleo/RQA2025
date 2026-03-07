# RQA2025 统一导入规范

## 概述

为避免代码重复定义和导入混乱，制定统一的导入规范，确保项目架构清晰、维护简单。

## 导入分层原则

### 1. 通用工具层 (`src/utils/`)
**推荐使用** - 提供简化的API接口，重定向到基础设施层实现

```python
# ✅ 推荐导入方式
from src.utils.logger import get_logger
from src.utils.date_utils import convert_timezone, get_business_date
from src.utils.math_utils import calculate_returns
from src.utils.data_utils import normalize_data
```

### 2. 基础设施层 (`src/infrastructure/utils/`)
**特殊情况使用** - 提供完整、稳定的基础功能

```python
# ⚠️ 仅在需要高级功能时使用
from src.infrastructure.utils.logger import LoggerFactory, configure_logging
from src.infrastructure.utils.date_utils import DateUtils
from src.infrastructure.utils.performance import PerformanceMonitor
```

## 具体导入规范

### Logger 模块

#### 标准用法
```python
# ✅ 推荐
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Application started")
```

#### 高级用法
```python
# ⚠️ 仅在需要高级功能时使用
from src.infrastructure.utils.logger import LoggerFactory, configure_logging

# 配置全局日志
configure_logging({'level': 'DEBUG'})

# 创建专用日志记录器
logger = LoggerFactory.create_logger("trading", level="INFO", add_file=True)
```

### Date Utils 模块

#### 标准用法
```python
# ✅ 推荐
from src.utils.date_utils import (
    convert_timezone,
    get_business_date,
    is_trading_day,
    next_trading_day
)

# 时区转换
dt = convert_timezone(datetime.now(), "UTC", "Asia/Shanghai")

# 获取交易日
trading_date = get_business_date()
```

#### 高级用法
```python
# ⚠️ 仅在需要高级功能时使用
from src.infrastructure.utils.date_utils import DateUtils, parse_date_range

# 日期范围处理
start_dt, end_dt = parse_date_range("2024-01-01", "2024-12-31", "Asia/Shanghai")

# 使用工具类
formatted_date = DateUtils.format_date(datetime.now(), "%Y-%m-%d")
```

### Math Utils 模块

```python
# ✅ 推荐
from src.utils.math_utils import calculate_returns, sharpe_ratio, annualized_volatility

returns = calculate_returns(prices)
sharpe = sharpe_ratio(returns)
vol = annualized_volatility(returns)
```

### Data Utils 模块

```python
# ✅ 推荐
from src.utils.data_utils import normalize_data, denormalize_data

normalized = normalize_data(data)
original = denormalize_data(normalized)
```

## 迁移指南

### 从基础设施层迁移到通用层

#### 旧代码
```python
# ❌ 旧方式
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.utils.date_utils import convert_timezone
```

#### 新代码
```python
# ✅ 新方式
from src.utils.logger import get_logger
from src.utils.date_utils import convert_timezone
```

### 批量迁移脚本

```python
# 迁移脚本示例
import re

def migrate_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换导入
    content = re.sub(
        r'from src\.infrastructure\.utils\.logger import get_logger',
        'from src.utils.logger import get_logger',
        content
    )
    
    content = re.sub(
        r'from src\.infrastructure\.utils\.date_utils import convert_timezone',
        'from src.utils.date_utils import convert_timezone',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
```

## 代码审查检查点

### 1. 导入检查
- [ ] 是否使用了推荐的通用层导入
- [ ] 是否避免了直接导入基础设施层
- [ ] 是否在必要时才使用基础设施层高级功能

### 2. 功能检查
- [ ] 是否避免了重复实现相同功能
- [ ] 是否使用了统一的API接口
- [ ] 是否保持了向后兼容性

### 3. 性能检查
- [ ] 导入是否高效
- [ ] 是否避免了循环导入
- [ ] 是否合理使用了缓存机制

## 常见问题

### Q1: 什么时候使用基础设施层？
A1: 仅在需要以下高级功能时：
- 复杂的日志配置
- 高级的日期时间处理
- 性能监控和分析
- 系统级工具功能

### Q2: 如何确保导入一致性？
A2: 
1. 使用推荐的导入方式
2. 定期运行代码审查
3. 使用自动化工具检查
4. 建立团队导入规范

### Q3: 如何处理现有代码？
A3:
1. 逐步迁移到新规范
2. 保持向后兼容
3. 更新文档和示例
4. 培训团队成员

## 版本历史

- **v1.0** (2025-01-19): 初始版本
- **v1.1** (2025-01-19): 添加迁移指南
- **v1.2** (2025-01-19): 完善代码审查检查点 