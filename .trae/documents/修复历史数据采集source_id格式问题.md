## 问题分析
根据日志分析，历史数据采集任务已成功执行，但数据持久化时使用了非标准的source_id格式，导致采集的数据不符合要求。

### 关键问题
- 历史数据采集调度器在 `src/core/orchestration/historical_data_scheduler.py` 中使用了 `f"historical_collection_{symbol}"` 作为 source_id
- 标准要求 source_id 应为 `akshare_stock_a`，与日常增量数据采集保持一致

## 解决方案

### 1. 修改历史数据采集调度器
- **文件**: `src/core/orchestration/historical_data_scheduler.py`
- **修改点**: 第 1061 行左右的 `_save_collected_data` 方法
- **变更**: 将 `source_id` 从 `f"historical_collection_{symbol}"` 修改为 `"akshare_stock_a"`

### 2. 验证修改
- 重启容器应用
- 启动历史数据采集调度器
- 检查新采集数据的 source_id 格式
- 验证数据字段映射是否正确

### 3. 清理不符合要求的数据
- 清理所有使用非标准 source_id 格式的数据
- 确保数据库中只保留符合标准的数据

## 预期结果
- 历史数据采集使用与日常增量数据采集相同的 source_id 格式
- 所有采集数据符合标准要求
- 系统能够正确处理和持久化历史数据