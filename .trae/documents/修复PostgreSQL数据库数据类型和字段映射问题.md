## 📋 问题分析

**数据库状态检查结果：**
- `data_type` 为 'volume'：366条记录，`change`和`amplitude`字段全为空
- `data_type` 为 'fundamental'：2条记录，`change`和`amplitude`字段全为空
- `data_type` 为 'price'：366条记录，`change`和`amplitude`字段全为空
- `data_type` 为 'daily'：50条记录，`change`和`amplitude`字段全部有值

**问题根源：**
1. **配置不一致**：
   - **标准数据采集器**（standard_data_collector.py 第52行）：使用 `["daily", "weekly", "monthly"]`
   - **历史数据调度器**（historical_data_scheduler.py 第157行）：使用 `['price', 'volume', 'fundamental']`

2. **字段映射不完整**：BaoStock返回的字段名与代码期望不一致
   - BaoStock返回 `pctChg` 而不是 `涨跌幅`
   - BaoStock返回 `turn` 而不是 `换手率`
   - 可能没有直接的 `涨跌额` 和 `振幅` 字段

## 🔧 解决方案

### 步骤1：统一数据类型配置
**文件：** `src/core/orchestration/historical_data_scheduler.py`

**修改内容：**
- 将第157行的默认数据类型配置改为：
  ```python
  self.collection_data_types: List[str] = ['daily']  # 标准数据类型
  ```
- 确保所有数据采集都使用标准数据类型

### 步骤2：增强字段映射逻辑
**文件：** `src/core/orchestration/historical_data_scheduler.py`

**修改内容：**
- 在字段映射部分（858-939行）添加BaoStock特定字段名的映射
- 添加change和amplitude字段的计算逻辑：
  ```python
  # 计算涨跌额 (close - open)
  if 'close' in record and 'open' in record:
      record['change'] = record['close'] - record['open']
  
  # 计算振幅 ((high - low) / open * 100)
  if 'high' in record and 'low' in record and 'open' in record and record['open'] > 0:
      record['amplitude'] = (record['high'] - record['low']) / record['open'] * 100
  ```

### 步骤3：清理数据库中的非标准数据
**SQL命令：**
```sql
-- 删除非标准数据类型的记录
DELETE FROM akshare_stock_data WHERE data_type NOT IN ('1分钟线', '5分钟线', '日线', '周线', '月线', 'daily');

-- 验证清理结果
SELECT data_type, COUNT(*) FROM akshare_stock_data GROUP BY data_type;
```

### 步骤4：验证修复结果
**验证步骤：**
1. 触发历史数据立即采集
2. 检查App运行日志，确认数据采集成功
3. 查询数据库，验证新采集的数据类型为标准值
4. 检查change和amplitude字段是否有值

## 📊 预期修复效果

修复后的数据状态：
- ✅ 所有记录的data_type为标准类型（如daily、weekly、monthly）
- ✅ 所有记录的change字段（涨跌额）有值
- ✅ 所有记录的amplitude字段（振幅）有值
- ✅ 历史数据调度器与标准数据采集器使用一致的数据类型配置

## 🔒 安全考虑

- 数据库清理操作会删除不符合标准的记录，请确保在执行前进行数据备份
- 修改后的代码需要进行充分测试，确保不影响其他功能
- 字段映射逻辑需要兼容多种数据源的返回格式