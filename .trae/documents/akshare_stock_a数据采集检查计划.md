# akshare_stock_a 数据采集检查计划

## 问题描述
用户报告数据源配置 `akshare_stock_a` 采用自选股池，但已采集的数据似乎使用了硬编码的 `000001`（平安银行股票代码），而不是使用自选股池中的股票代码。

## 目标
1. 检查 `akshare_stock_a` 数据源的配置，确认自选股池设置
2. 检查数据采集代码，确认是否正确使用自选股池
3. 检查已采集的数据，确认是否存在硬编码 `000001` 的问题
4. 修复问题，确保数据采集使用正确的自选股池

## 检查范围

### 1. 数据源配置检查
- 检查 `akshare_stock_a` 的配置文件
- 确认 `stock_pool` 或 `symbols` 配置项
- 确认数据源是否启用了自选股池模式

### 2. 数据采集代码检查
- 检查 `akshare_stock_a` 对应的数据采集器
- 检查股票代码列表获取逻辑
- 检查是否存在硬编码 `000001` 的情况
- 检查自选股池是否正确加载

### 3. 已采集数据检查
- 检查数据库中已采集的数据
- 确认数据中的股票代码分布
- 确认是否存在只有 `000001` 数据的情况

## 实现方案

### 步骤1：检查数据源配置
**检查文件**:
- `config/data_sources.json` 或相关配置文件
- 数据源配置中的 `stock_pool`、`symbols`、`universe` 等字段

**预期发现**:
- 自选股池配置
- 股票代码列表

### 步骤2：检查数据采集代码
**检查文件**:
- `src/data/collectors/akshare_collector.py` 或类似文件
- 数据采集器的 `fetch_data()` 或 `collect()` 方法
- 股票代码列表获取逻辑

**关键检查点**:
- 是否存在 `symbols = ["000001"]` 或类似的硬编码
- 是否正确从配置中读取股票代码列表
- 自选股池是否正确加载

### 步骤3：检查已采集数据
**检查数据库**:
```sql
-- 检查已采集数据的股票代码分布
SELECT symbol, COUNT(*) as count 
FROM stock_daily_data 
WHERE source_id = 'akshare_stock_a' 
GROUP BY symbol 
ORDER BY count DESC;
```

**预期发现**:
- 如果只有 `000001` 数据，说明存在硬编码问题
- 如果有多个股票代码，说明配置可能正确

### 步骤4：修复问题
根据检查结果，修复以下可能的问题：

#### 情况A：硬编码股票代码
**修复**: 修改数据采集器，从配置中读取股票代码列表
```python
# 错误示例（硬编码）
symbols = ["000001"]

# 正确示例（从配置读取）
symbols = config.get("stock_pool", []) or config.get("symbols", [])
```

#### 情况B：自选股池未正确加载
**修复**: 确保自选股池服务正确初始化并加载
```python
# 从自选股池服务获取
from src.data.services.stock_pool_service import get_stock_pool_service
stock_pool_service = get_stock_pool_service()
symbols = stock_pool_service.get_symbols()
```

#### 情况C：配置错误
**修复**: 修正数据源配置
```json
{
  "id": "akshare_stock_a",
  "stock_pool": ["000001", "000002", "600000"],
  "use_stock_pool": true
}
```

## 任务列表

### 检查任务
- [ ] 检查 `akshare_stock_a` 数据源配置
- [ ] 检查数据采集代码中的股票代码获取逻辑
- [ ] 检查数据库中已采集数据的股票代码分布
- [ ] 确认问题原因（硬编码/配置错误/加载失败）

### 修复任务
- [ ] 修复数据采集代码（如果是硬编码问题）
- [ ] 修复自选股池加载逻辑（如果是加载问题）
- [ ] 修正数据源配置（如果是配置问题）
- [ ] 测试修复后的数据采集

### 验证任务
- [ ] 验证数据采集使用正确的股票代码
- [ ] 验证数据库中数据的股票代码分布正确
- [ ] 验证自选股池修改后能正确反映到采集中

## 相关文件

### 配置文件
- `config/data_sources.json`
- `config/production_config.yml`

### 代码文件
- `src/data/collectors/akshare_collector.py`
- `src/data/collectors/base_collector.py`
- `src/data/services/stock_pool_service.py`

### 数据库表
- `stock_daily_data`
- `data_source_configs`

## 时间估计
- 检查阶段：30分钟
- 修复阶段：30-60分钟（根据问题复杂度）
- 验证阶段：15分钟
- **总计：约1.5-2小时**
