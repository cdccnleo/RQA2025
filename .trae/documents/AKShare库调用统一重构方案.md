# AKShare库调用统一重构方案

## 📋 重构概述

将分散在多个模块中的AKShare库调用逻辑统一到一个独立的服务中，消除代码重复，提高可维护性和一致性。

## 🔍 问题分析

### 当前代码重复情况

| 模块 | 重复功能 | 问题 |
|------|----------|------|
| `data_collectors.py` | AKShare调用、无缝切换、字段映射 | 代码重复 |
| `historical_data_scheduler.py` | 类似的AKShare调用逻辑 | 逻辑不一致 |
| `historical_data_acquisition_service.py` | 类似的AKShare调用逻辑 | 实现差异 |
| `data/china/adapters` | AKShare基础信息调用 | 分散实现 |
| `smart_stock_filter.py` | 市场数据调用 | 独立实现 |

### 核心重复功能
- ✅ 无缝切换机制（stock_zh_a_hist → stock_zh_a_daily）
- ✅ 字段映射和标准化
- ✅ 错误处理和重试逻辑
- ✅ 超时设置和参数管理
- ✅ 数据格式转换

## 🎯 重构目标

### 1. 创建统一的AKShare服务
**文件**: `src/core/integration/akshare_service.py`

**核心功能**:
- ✅ 统一的AKShare接口调用
- ✅ 智能无缝切换机制
- ✅ 统一的字段映射
- ✅ 集中的错误处理
- ✅ 配置化的参数管理

### 2. 服务架构设计

```python
class AKShareService:
    """统一的AKShare服务"""
    
    def __init__(self, config=None):
        # 初始化配置
    
    async def get_stock_data(self, symbol, start_date, end_date, **kwargs):
        """获取股票数据（自动无缝切换）"""
    
    async def get_market_data(self):
        """获取市场数据"""
    
    async def get_stock_info(self, symbol):
        """获取股票基础信息"""
    
    async def get_minute_data(self, symbol, period, **kwargs):
        """获取分钟线数据"""
```

### 3. 更新调用点

| 模块 | 更新内容 |
|------|----------|
| `data_collectors.py` | 使用AKShareService替代直接调用 |
| `historical_data_scheduler.py` | 使用AKShareService替代现有逻辑 |
| `historical_data_acquisition_service.py` | 使用AKShareService替代模拟数据 |
| `data/china/adapters` | 使用AKShareService统一调用 |
| `smart_stock_filter.py` | 使用AKShareService获取市场数据 |

### 4. 配置管理

**文件**: `config/akshare_service.yml`

```yaml
akshare_service:
  retry_policy:
    max_retries: 3
    initial_delay: 3
    backoff_factor: 2
  timeout:
    stock_data: 30
    market_data: 60
  field_mapping:
    # 统一的字段映射配置
  api_preference:
    # API优先级配置
```

## 🚀 实现步骤

### 步骤1: 创建AKShare服务
- 创建 `src/core/integration/akshare_service.py`
- 实现核心服务功能
- 添加配置管理

### 步骤2: 更新历史数据调度器
- 修改 `historical_data_scheduler.py`
- 替换直接AKShare调用为服务调用
- 移除重复的无缝切换逻辑

### 步骤3: 更新数据采集器
- 修改 `data_collectors.py`
- 使用AKShareService替代现有逻辑
- 简化数据采集代码

### 步骤4: 更新历史数据采集服务
- 修改 `historical_data_acquisition_service.py`
- 使用AKShareService获取真实数据
- 移除模拟数据逻辑

### 步骤5: 更新其他模块
- 更新 `data/china/adapters/__init__.py`
- 更新 `smart_stock_filter.py`
- 确保所有AKShare调用都通过服务

### 步骤6: 测试验证
- 运行数据采集测试
- 验证历史数据采集
- 确保所有功能正常

## 💡 预期收益

1. **消除代码重复**：减少50%+的重复代码
2. **提高可维护性**：集中管理AKShare调用逻辑
3. **一致性**：所有模块使用相同的AKShare调用逻辑
4. **可扩展性**：易于添加新的AKShare接口支持
5. **可靠性**：统一的错误处理和重试机制

## ✅ 重构完成标准

- ✅ 所有AKShare调用都通过统一服务
- ✅ 代码重复率显著降低
- ✅ 所有功能正常工作
- ✅ 测试通过
- ✅ 文档更新完成