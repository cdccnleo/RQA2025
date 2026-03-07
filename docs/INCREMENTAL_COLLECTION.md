# 增量采集功能实现文档

## 📋 概述

增量采集功能通过智能检测数据库中已存在的数据，只采集缺失的日期范围，避免重复采集，提高效率，同时确保数据完整性。

## 🎯 核心特性

### 1. 自动检测已存在数据

- ✅ 查询数据库中已存在的日期
- ✅ 按股票代码分别处理
- ✅ 只考虑请求日期范围内的数据

### 2. 智能计算缺失范围

- ✅ 自动计算缺失的日期范围
- ✅ 支持多个不连续的缺失范围
- ✅ 确保边界情况正确处理

### 3. 数据完整性保证

- ✅ 不会遗漏任何数据
- ✅ 处理所有边界情况
- ✅ 数据库去重作为最后保障

## 🔍 实现逻辑

### 步骤1: 查询已存在数据

```python
def query_existing_dates(
    source_id: str,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime
) -> Dict[str, set]:
    """查询数据库中已存在的数据日期"""
    # 对每个股票代码查询已存在的日期
    # 返回: {symbol: {date1, date2, ...}}
```

**查询逻辑**：
- 对每个股票代码单独查询
- 只查询请求日期范围内的数据
- 返回日期集合，便于快速查找

### 步骤2: 计算缺失日期范围

```python
def calculate_missing_date_ranges(
    start_date: datetime,
    end_date: datetime,
    existing_dates: set
) -> List[tuple]:
    """计算缺失的日期范围"""
    # 1. 如果没有任何已存在数据，返回整个范围
    # 2. 遍历已存在的日期，找出缺失的区间
    # 3. 返回缺失日期范围的列表
```

**计算逻辑**：

```
请求范围: 2024-01-01 至 2024-12-31
已存在日期: {2024-03-01, 2024-03-02, ..., 2024-06-30, 2024-09-01, ..., 2024-09-30}

缺失范围:
1. 2024-01-01 至 2024-02-29 (3月之前)
2. 2024-07-01 至 2024-08-31 (6月之后，9月之前)
3. 2024-10-01 至 2024-12-31 (9月之后)
```

**边界情况处理**：
- ✅ 开始日期之前：包含在缺失范围
- ✅ 结束日期之后：包含在缺失范围
- ✅ 连续日期中间：正确分割缺失范围
- ✅ 完全覆盖：返回空列表（跳过采集）

### 步骤3: 增量采集

```python
async def collect_from_akshare_adapter(...):
    # 1. 检查是否启用增量采集
    enable_incremental = request_data.get("incremental", True)
    
    # 2. 查询已存在数据
    if enable_incremental:
        existing_dates_map = query_existing_dates(...)
    
    # 3. 对每个股票计算缺失范围
    for symbol in symbols:
        existing_dates = existing_dates_map.get(symbol, set())
        missing_ranges = calculate_missing_date_ranges(...)
        
        # 4. 只采集缺失的日期范围
        for range_start, range_end in missing_ranges:
            df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=range_start.strftime("%Y%m%d"),
                end_date=range_end.strftime("%Y%m%d"),
                adjust="qfq"
            )
```

## 📊 使用示例

### 示例1: 首次采集（全量）

```python
request_data = {
    "symbols": ["000001"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "incremental": False,  # 禁用增量采集
    "persist": True
}
```

**结果**：
- 采集整个日期范围的数据
- 数据库中没有任何已存在数据
- 采集所有交易日数据

### 示例2: 增量采集（相同范围）

```python
request_data = {
    "symbols": ["000001"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "incremental": True,  # 启用增量采集
    "persist": True
}
```

**结果**：
- 查询数据库发现数据已完整
- 跳过采集（返回0条记录）
- 节省API调用

### 示例3: 增量采集（扩展范围）

```python
request_data = {
    "symbols": ["000001"],
    "start_date": "2023-01-01",  # 扩展开始日期
    "end_date": "2024-12-31",
    "incremental": True,  # 启用增量采集
    "persist": True
}
```

**结果**：
- 查询数据库发现已有 2024-01-01 至 2024-12-31 的数据
- 只采集 2023-01-01 至 2023-12-31 的数据
- 确保数据完整性

### 示例4: 增量采集（中间缺失）

```python
# 假设数据库中已有:
# - 2024-01-01 至 2024-03-31
# - 2024-07-01 至 2024-12-31
# 缺失: 2024-04-01 至 2024-06-30

request_data = {
    "symbols": ["000001"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "incremental": True,
    "persist": True
}
```

**结果**：
- 计算缺失范围: [(2024-04-01, 2024-06-30)]
- 只采集缺失的3个月数据
- 确保数据完整性

## 🔒 数据完整性保证

### 1. 边界检查

```python
# 开始日期之前
if current_start < existing_date:
    missing_ranges.append((current_start, existing_date - 1天))

# 结束日期之后
if current_start <= end:
    missing_ranges.append((current_start, end))
```

### 2. 数据库去重保障

即使增量采集逻辑出现问题，数据库的UNIQUE约束也会确保：
- 不会插入重复数据
- 已存在数据会被更新（ON CONFLICT DO UPDATE）
- 数据完整性得到保障

### 3. 错误处理

```python
try:
    existing_dates_map = query_existing_dates(...)
except Exception as e:
    logger.warning(f"查询已存在日期失败，将进行全量采集: {e}")
    # 降级为全量采集，确保不遗漏数据
    existing_dates_map = {}
```

## ⚙️ 配置选项

### 数据源配置

```json
{
  "id": "akshare_stock",
  "config": {
    "enable_incremental": true,  // 默认启用增量采集
    "default_symbols": ["000001", "000002"],
    "default_days": 30
  }
}
```

### API请求参数

```python
{
    "incremental": true,  // 是否启用增量采集（默认true）
    "symbols": ["000001"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
}
```

## 📈 性能优势

### 传统方式（全量采集）

```
请求: 采集 2024-01-01 至 2024-12-31
数据库已有: 2024-01-01 至 2024-06-30

操作:
1. 调用API采集整个范围 (250个交易日)
2. 数据库去重 (跳过125条，插入125条)

问题: 浪费125次API调用
```

### 增量采集方式

```
请求: 采集 2024-01-01 至 2024-12-31
数据库已有: 2024-01-01 至 2024-06-30

操作:
1. 查询数据库已存在日期 (125个)
2. 计算缺失范围: 2024-07-01 至 2024-12-31
3. 调用API只采集缺失范围 (125个交易日)
4. 数据库插入 (125条新记录)

优势: 节省125次API调用，提高效率
```

## 🧪 测试验证

运行测试脚本验证增量采集功能：

```bash
python scripts/test_incremental_collection.py
```

**测试场景**：
1. ✅ 首次采集（全量）
2. ✅ 增量采集（相同范围，应跳过）
3. ✅ 增量采集（扩展范围，应只采集新增部分）
4. ✅ 数据完整性验证

## 🔍 关键代码位置

### 1. 查询已存在日期

**文件**: `src/gateway/web/postgresql_persistence.py`

```python
def query_existing_dates(...) -> Dict[str, set]:
    # 第164-224行
```

### 2. 计算缺失日期范围

**文件**: `src/gateway/web/postgresql_persistence.py`

```python
def calculate_missing_date_ranges(...) -> List[tuple]:
    # 第227-277行
```

### 3. 增量采集逻辑

**文件**: `src/gateway/web/api.py`

```python
async def collect_from_akshare_adapter(...):
    # 第2319-2560行
    # 增量采集逻辑在第2414-2435行
```

## ✅ 总结

增量采集功能实现了：

1. ✅ **智能检测**: 自动查询已存在数据
2. ✅ **精确计算**: 准确计算缺失日期范围
3. ✅ **高效采集**: 只采集缺失部分，节省API调用
4. ✅ **数据完整**: 确保不遗漏任何数据
5. ✅ **容错机制**: 查询失败时降级为全量采集
6. ✅ **数据库保障**: UNIQUE约束作为最后保障

通过增量采集，系统可以：
- 减少API调用次数
- 提高采集效率
- 降低数据源压力
- 确保数据完整性

