# 修复fundamental数据存储问题

## 问题分析
在历史数据采集代码中，fundamental数据被错误地存储到了akshare_stock_data表中，而不是专门的akshare_fundamental_data表中。

### 根本原因
1. **数据采集逻辑**：在`historical_data_scheduler.py`的`_collect_data`方法中，所有类型的数据（包括fundamental）都被添加到同一个`collected_data`列表中
2. **数据类型判断**：在`_save_collected_data`方法中，调用`_determine_data_type`方法判断数据类型
3. **判断逻辑缺陷**：`_determine_data_type`方法根据字段判断数据类型，由于fundamental数据包含`date`字段，被错误判断为"stock"类型
4. **持久化函数选择**：由于被判断为"stock"类型，调用了`persist_akshare_data_to_postgresql`函数，将数据存储到akshare_stock_data表中

## 解决方案
修改`_save_collected_data`方法，使其能够根据数据中的`data_type`字段正确区分fundamental数据，并调用相应的持久化函数。

### 具体修改步骤
1. **修改`_save_collected_data`方法**：
   - 检查采集的数据列表中是否包含`data_type='fundamental'`的记录
   - 如果包含，将fundamental数据分离出来
   - 对fundamental数据调用`persist_akshare_fundamental_data_to_postgresql`函数
   - 对其他数据继续使用现有的持久化逻辑

2. **修改点**：
   - 文件：`src/core/orchestration/historical_data_scheduler.py`
   - 方法：`_save_collected_data`

### 预期结果
- fundamental数据将正确存储到`akshare_fundamental_data`表中
- 其他类型的数据继续存储到相应的表中
- 保持现有数据采集和存储逻辑的完整性