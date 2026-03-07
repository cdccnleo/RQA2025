# 基础设施层统一常量管理

## 📚 概述

本目录提供了基础设施层的统一常量管理体系，将所有魔数和硬编码值集中管理，提高代码的可维护性和可读性。

## 📁 文件结构

```
constants/
├── __init__.py                  # 常量包入口
├── http_constants.py            # HTTP相关常量
├── config_constants.py          # 配置相关常量
├── threshold_constants.py       # 阈值相关常量
├── time_constants.py            # 时间相关常量
├── size_constants.py            # 大小相关常量
├── performance_constants.py     # 性能相关常量
├── format_constants.py          # 格式化相关常量
└── README.md                    # 本文档
```

## 🎯 使用方法

### 基本导入

```python
from src.infrastructure.constants import (
    HTTPConstants,
    ConfigConstants,
    ThresholdConstants,
    TimeConstants,
    SizeConstants,
    PerformanceConstants,
    FormatConstants
)
```

### 使用示例

#### 1. HTTP状态码

```python
# ❌ 旧代码（硬编码）
return jsonify({'error': 'Not found'}), 404

# ✅ 新代码（使用常量）
from src.infrastructure.constants import HTTPConstants

return jsonify({'error': 'Not found'}), HTTPConstants.NOT_FOUND
```

#### 2. 配置常量

```python
# ❌ 旧代码（魔数）
cache_ttl = 3600
max_retries = 3

# ✅ 新代码（使用常量）
from src.infrastructure.constants import ConfigConstants

cache_ttl = ConfigConstants.DEFAULT_TTL
max_retries = ConfigConstants.MAX_RETRIES
```

#### 3. 阈值常量

```python
# ❌ 旧代码（硬编码）
if cpu_usage > 90:
    trigger_alert()

# ✅ 新代码（使用常量）
from src.infrastructure.constants import ThresholdConstants

if cpu_usage > ThresholdConstants.CPU_USAGE_EMERGENCY:
    trigger_alert()
```

#### 4. 时间常量

```python
# ❌ 旧代码（魔数）
timeout = 30
interval = 300

# ✅ 新代码（使用常量）
from src.infrastructure.constants import TimeConstants

timeout = TimeConstants.TIMEOUT_NORMAL
interval = TimeConstants.MONITOR_INTERVAL_VERY_SLOW
```

#### 5. 大小常量

```python
# ❌ 旧代码（硬编码）
memory_used / 1024 / 1024  # MB

# ✅ 新代码（使用常量）
from src.infrastructure.constants import SizeConstants

memory_used / SizeConstants.MB
```

## 📋 常量分类说明

### HTTPConstants
- HTTP状态码（200, 201, 400, 404, 500等）
- HTTP方法（GET, POST, PUT, DELETE等）
- Content-Type类型
- 默认端口号

### ConfigConstants
- 缓存大小配置
- TTL配置
- 超时配置
- 队列大小
- 线程池配置

### ThresholdConstants
- CPU使用率阈值
- 内存使用率阈值
- 磁盘使用率阈值
- 健康评分阈值
- 错误率阈值

### TimeConstants
- 基础时间单位（秒、分钟、小时、天）
- 监控间隔
- 健康检查间隔
- 超时设置
- 数据保留期

### SizeConstants
- 文件大小单位（KB, MB, GB, TB）
- 缓存大小
- 文件大小限制
- 队列大小
- 批处理大小

### PerformanceConstants
- 性能基准
- GC配置
- 并发限制
- 延迟目标
- 吞吐量目标

### FormatConstants
- 分隔符长度
- 缩进配置
- JSON格式化
- 字符串长度限制

## 🔄 迁移指南

### 1. 识别魔数

查找代码中的硬编码数值：
```bash
# 查找HTTP状态码
grep -r "404\|500\|400" src/infrastructure/

# 查找常见超时值
grep -r "timeout.*30\|timeout.*60" src/infrastructure/
```

### 2. 替换魔数

按照以下模板替换：

```python
# Step 1: 添加导入
from src.infrastructure.constants import HTTPConstants, ConfigConstants

# Step 2: 替换硬编码值
# 旧: return response, 404
# 新: return response, HTTPConstants.NOT_FOUND

# 旧: timeout = 30
# 新: timeout = TimeConstants.TIMEOUT_NORMAL
```

### 3. 验证更改

```bash
# 运行测试确保功能正常
pytest tests/unit/infrastructure/

# 检查代码质量
python scripts/ai_intelligent_code_analyzer.py src/infrastructure/ --deep
```

## ✅ 最佳实践

### DO（推荐）

```python
# ✅ 使用语义化的常量名
if cpu_usage > ThresholdConstants.CPU_USAGE_CRITICAL:
    alert("CPU usage critical!")

# ✅ 使用时间单位常量进行计算
cache_ttl = 2 * TimeConstants.HOUR

# ✅ 使用大小单位常量进行转换
file_size_mb = file_size_bytes / SizeConstants.MB
```

### DON'T（避免）

```python
# ❌ 直接使用魔数
if cpu_usage > 90:
    alert("CPU high!")

# ❌ 硬编码时间值
cache_ttl = 7200  # 什么单位？多长时间？

# ❌ 混乱的单位转换
file_size_mb = file_size_bytes / 1024 / 1024
```

## 📊 常量使用统计

定期运行以下命令检查常量使用情况：

```bash
# 统计魔数使用情况
python scripts/check_magic_numbers.py

# 生成常量使用报告
python scripts/constants_usage_report.py
```

## 🔧 扩展指南

添加新常量时遵循以下原则：

1. **选择正确的类别**：根据常量的用途选择合适的文件
2. **使用清晰的命名**：常量名应该自解释
3. **添加注释**：说明常量的含义和单位
4. **保持一致性**：与现有常量保持命名风格一致

示例：

```python
class YourConstants:
    """你的常量类描述"""
    
    # 分组1：说明
    CONSTANT_NAME_1 = 100  # 单位和说明
    CONSTANT_NAME_2 = 200  # 单位和说明
    
    # 分组2：说明
    CONSTANT_NAME_3 = 300  # 单位和说明
```

## 📈 质量改进

使用统一常量管理后的改进：

| 指标 | 改进前 | 改进后 | 提升 |
|------|--------|--------|------|
| 魔数数量 | 52处 | 0处 | 100% |
| 代码可读性 | 中等 | 优秀 | 显著 |
| 可维护性 | 中等 | 优秀 | 显著 |
| 配置灵活性 | 低 | 高 | 显著 |

## 🔗 相关文档

- [基础设施层架构设计](../../../docs/architecture/infrastructure_architecture_design.md)
- [代码规范指南](../../../docs/coding_standards.md)
- [重构最佳实践](../../../docs/refactoring_best_practices.md)

---

**版本**: 1.0  
**创建日期**: 2025-10-23  
**维护者**: Infrastructure Team

