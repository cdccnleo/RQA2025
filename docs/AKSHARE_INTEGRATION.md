# AKShare A股数据采集集成文档

## 📋 概述

本文档说明如何在RQA2025量化交易系统中集成AKShare接口获取A股数据。

## 🎯 实现内容

### 1. API层数据采集支持

在 `src/gateway/web/api.py` 的 `collect_data_via_data_layer()` 函数中添加了股票数据类型的支持：

```python
elif source_type.lower() in ["股票数据", "stock", "akshare", "a股", "astock"]:
    data = await collect_from_akshare_adapter(source_config, request_data)
```

### 2. AKShare适配器实现

实现了 `collect_from_akshare_adapter()` 函数，支持：

- ✅ 指定股票代码列表采集
- ✅ 自定义日期范围
- ✅ 自动获取A股股票列表（未指定时）
- ✅ 默认日期范围（最近30天）
- ✅ 支持主要接口和备用接口自动切换
- ✅ 中英文列名兼容处理
- ✅ 完整的数据字段映射

### 3. AStockAdapter增强

在 `src/data/china/adapters/__init__.py` 中实现了：

- ✅ `get_stock_basic()` - 获取股票基础信息
- ✅ `get_daily_quotes()` - 获取日线行情数据
- ✅ 支持单只股票和全部股票列表
- ✅ 自动接口切换和错误处理

### 4. 数据源配置

在 `data/data_sources_config.json` 中添加了AKShare数据源配置：

```json
{
  "id": "akshare_stock",
  "name": "AKShare A股数据",
  "type": "股票数据",
  "url": "https://akshare.akfamily.xyz",
  "rate_limit": "100次/分钟",
  "enabled": true,
  "config": {
    "default_symbols": ["000001", "000002", "600000", "600036", "000858"],
    "default_days": 30,
    "adjust_type": "qfq"
  }
}
```

## 📊 数据字段映射

AKShare返回的中文列名映射到标准字段：

| AKShare列名 | 标准字段名 | 说明 |
|------------|-----------|------|
| 日期 | date | 交易日期 |
| 股票代码 | symbol | 股票代码 |
| 开盘 | open | 开盘价 |
| 收盘 | close | 收盘价 |
| 最高 | high | 最高价 |
| 最低 | low | 最低价 |
| 成交量 | volume | 成交量 |
| 成交额 | amount | 成交额 |
| 涨跌幅 | pct_change | 涨跌幅(%) |
| 涨跌额 | change | 涨跌额 |
| 换手率 | turnover_rate | 换手率(%) |
| 振幅 | amplitude | 振幅(%) |

## 🚀 使用方法

### API调用示例

```bash
# POST /api/v1/data/sources/akshare_stock/collect
curl -X POST http://localhost:8000/api/v1/data/sources/akshare_stock/collect \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["000001", "600000"],
    "start_date": "2024-12-01",
    "end_date": "2024-12-31",
    "data_type": "daily"
  }'
```

### Python代码示例

```python
import requests

url = "http://localhost:8000/api/v1/data/sources/akshare_stock/collect"
data = {
    "symbols": ["000001", "600000"],
    "start_date": "2024-12-01",
    "end_date": "2024-12-31",
    "data_type": "daily"
}

response = requests.post(url, json=data)
result = response.json()

print(f"采集记录数: {len(result['data'])}")
for record in result['data'][:5]:
    print(f"{record['symbol']} {record['date']}: 收盘价 {record['close']}")
```

### 使用适配器类

```python
from src.data.china.adapters import AStockAdapter

# 创建适配器实例
adapter = AStockAdapter()
adapter.connect()

# 获取股票基础信息
stock_list = adapter.get_stock_basic()
print(f"A股股票总数: {len(stock_list)}")

# 获取单只股票信息
stock_info = adapter.get_stock_basic("000001")

# 获取日线数据
quotes = adapter.get_daily_quotes("000001", "2024-12-01", "2024-12-31")
print(f"数据条数: {len(quotes)}")
```

## 📋 请求参数说明

### collect API参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|-------|------|------|--------|------|
| symbols | List[str] | 否 | 自动获取前10只 | 股票代码列表 |
| start_date | str | 否 | 30天前 | 开始日期 (YYYY-MM-DD) |
| end_date | str | 否 | 今天 | 结束日期 (YYYY-MM-DD) |
| data_type | str | 否 | "daily" | 数据类型 ("daily" 或 "hist") |

## 📊 响应数据格式

```json
{
  "success": true,
  "source_id": "akshare_stock",
  "data": [
    {
      "symbol": "000001",
      "date": "2024-12-02",
      "open": 10.79,
      "high": 10.89,
      "low": 10.77,
      "close": 10.79,
      "volume": 975434.0,
      "amount": 10523456.78,
      "pct_change": 0.09,
      "change": 0.01,
      "turnover_rate": 0.50,
      "amplitude": 0.83,
      "timestamp": 1767159704.39,
      "data_source": "akshare",
      "source_id": "akshare_stock"
    }
  ],
  "metadata": {
    "source_id": "akshare_stock",
    "source_type": "股票数据",
    "collection_timestamp": 1767159704.39,
    "data_points": 44
  },
  "collection_time": 0.94,
  "quality_score": 100.0
}
```

## ⚙️ 配置说明

### 数据源配置项

- `default_symbols`: 默认股票代码列表（未指定时使用）
- `default_days`: 默认采集天数（未指定日期范围时）
- `adjust_type`: 复权类型 ("qfq"前复权, "hfq"后复权, "none"不复权)

## 🔧 技术实现细节

### 接口自动切换

系统会优先使用 `ak.stock_zh_a_daily()`，失败时自动切换到 `ak.stock_zh_a_hist()`。

### 数据质量保证

- ✅ 量化交易系统合规：只使用真实数据，无模拟数据
- ✅ 数据验证：检查数据完整性和有效性
- ✅ 错误处理：完善的异常处理和日志记录
- ✅ 性能优化：支持批量采集和并发处理

### 依赖要求

```bash
pip install akshare pandas
```

## 🎯 测试验证

运行测试脚本验证集成：

```bash
python test_akshare_integration.py
```

测试结果：
- ✅ 服务状态检查
- ✅ 数据源配置验证
- ✅ 数据采集功能测试
- ✅ 默认参数测试

## 📈 性能指标

- **采集速度**: ~0.9秒/2只股票/30天数据
- **数据质量**: 100% (完整字段，无缺失)
- **接口可用性**: 高（主备接口自动切换）

## 🔍 故障排除

### 常见问题

1. **akshare未安装**
   ```
   错误: akshare未安装，无法采集A股数据
   解决: pip install akshare
   ```

2. **数据为空**
   ```
   检查: 股票代码是否正确，日期范围是否有效
   解决: 使用 ak.stock_info_a_code_name() 获取有效股票列表
   ```

3. **接口调用失败**
   ```
   系统会自动切换到备用接口
   检查日志获取详细错误信息
   ```

## 🚀 后续扩展

可以进一步扩展的功能：

1. **更多数据类型支持**
   - 分钟线数据
   - 周线/月线数据
   - 财务数据
   - 资金流向数据

2. **性能优化**
   - 数据缓存机制
   - 并发采集优化
   - 增量更新支持

3. **数据质量增强**
   - 数据清洗和标准化
   - 异常值检测
   - 数据完整性验证

## 📝 更新日志

- **2025-12-31**: 完成AKShare A股数据采集集成
  - 实现API层数据采集支持
  - 实现AKShare适配器函数
  - 增强AStockAdapter类
  - 添加数据源配置
  - 支持中英文列名兼容
  - 完成测试验证

## 📚 参考资源

- [AKShare官方文档](https://akshare.akfamily.xyz/)
- [AKShare GitHub](https://github.com/akfamily/akshare)
- RQA2025数据层架构文档

