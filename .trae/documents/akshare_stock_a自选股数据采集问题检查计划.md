# akshare_stock_a 自选股数据采集问题检查计划

## 问题描述
当前数据源配置 `akshare_stock_a` 和 `baostock_a股数据` 均使用了自选股，数据采集应当按照数据源配置自选股进行数据采集。当前并未配置 `000001`，但数据采集后 PostgreSQL 库 `akshare_stock_data` 表多了 `000001` 数据。

## 目标
全面检查数据采集流程，找出为什么 `000001` 数据会出现在数据库中，即使它不在自选股配置中。

## 检查范围

### 1. 数据源配置详细检查
- 检查 `akshare_stock_a` 的完整配置
- 检查 `baostock_a股数据` 的完整配置
- 确认自选股池的具体配置内容
- 检查配置文件中是否有隐藏的 `000001` 配置

### 2. 数据采集流程全面检查
- 检查所有可能的数据采集入口
- 检查测试连接功能是否会写入数据
- 检查手动采集功能是否会写入数据
- 检查自动采集任务是否会写入数据
- 检查是否有定时任务或后台进程在采集数据

### 3. 数据写入路径检查
- 检查所有写入 `akshare_stock_data` 表的代码
- 检查是否有硬编码的 `000001` 写入逻辑
- 检查测试数据或示例数据是否会写入生产库

### 4. 历史数据分析
- 检查 `000001` 数据的写入时间
- 检查 `000001` 数据的来源标识
- 检查 `000001` 数据是否与特定操作相关

## 实现方案

### 步骤1：详细检查数据源配置
**检查文件**:
- `data/data_sources_config.json`
- `data/production/data_sources_config.json`
- 检查是否有 `custom_stocks`、`default_symbols`、`stock_pool` 等配置

**检查命令**:
```bash
# 查看完整的akshare_stock_a配置
cat data/data_sources_config.json | grep -A 50 '"id": "akshare_stock_a"'

# 查看是否有000001相关的配置
grep -r "000001" data/
```

### 步骤2：检查所有数据写入路径
**检查文件**:
- `src/data/collectors/akshare_collector.py`
- `src/data/collectors/enhanced_akshare_collector.py`
- `src/gateway/web/data_collectors.py`
- 所有包含 `INSERT INTO akshare_stock_data` 或 `save_to_database` 的代码

**关键检查点**:
- 是否存在硬编码的 `symbol = "000001"`
- 测试连接功能是否会写入数据
- 示例代码或演示代码是否会写入数据

### 步骤3：检查测试连接功能
**检查文件**:
- `src/gateway/web/datasource_routes.py` 中的 `test_data_source` 函数
- 检查测试连接时是否会采集并保存数据样本

**关键代码**:
```python
# 在test_data_source函数中，检查以下逻辑：
# 1. 测试连接时是否会调用collect_data_via_data_layer
# 2. 是否会将测试数据保存到数据库
# 3. 使用的symbol是什么
```

### 步骤4：检查数据采集触发点
**检查内容**:
- 前端页面的"测试连接"按钮
- 前端页面的"立即采集"按钮
- 定时任务或调度器
- 系统启动时的初始化代码

### 步骤5：数据库历史数据分析
**检查命令**:
```sql
-- 查看000001数据的详细信息
SELECT * FROM akshare_stock_data 
WHERE symbol = '000001' 
ORDER BY date DESC;

-- 查看000001数据的最早和最晚日期
SELECT MIN(date) as earliest, MAX(date) as latest, COUNT(*) as count 
FROM akshare_stock_data 
WHERE symbol = '000001';

-- 查看000001数据的source_id分布
SELECT source_id, COUNT(*) as count 
FROM akshare_stock_data 
WHERE symbol = '000001' 
GROUP BY source_id;

-- 查看所有数据的source_id分布
SELECT source_id, symbol, COUNT(*) as count 
FROM akshare_stock_data 
GROUP BY source_id, symbol 
ORDER BY source_id, count DESC;
```

## 可能的问题原因

### 原因A：测试连接时写入数据
**场景**: 用户在测试 `akshare_stock_a` 连接时，系统自动采集了 `000001` 作为测试数据并保存到数据库。

**验证方法**: 检查 `test_data_source` 函数中使用的测试股票代码。

### 原因B：默认测试股票代码
**场景**: 系统在测试连接时默认使用 `000001` 作为测试股票，即使用户配置了自选股。

**验证方法**: 检查测试连接代码中是否有 `symbol = "000001"` 的硬编码。

### 原因C：示例数据或演示数据
**场景**: 系统初始化或演示时写入了 `000001` 的示例数据。

**验证方法**: 检查初始化脚本或演示代码。

### 原因D：其他数据源写入
**场景**: 其他数据源（如 `baostock_a股数据`）采集了 `000001` 数据，但写入了 `akshare_stock_data` 表。

**验证方法**: 检查所有数据源的数据保存逻辑。

## 任务列表

### 检查任务
- [ ] 详细检查 `akshare_stock_a` 和 `baostock_a股数据` 的完整配置
- [ ] 检查所有写入 `akshare_stock_data` 表的代码路径
- [ ] 检查测试连接功能是否会写入数据以及使用的股票代码
- [ ] 检查是否有定时任务或后台进程在采集数据
- [ ] 分析数据库中 `000001` 数据的来源和时间

### 修复任务
- [ ] 根据检查结果，修复硬编码或错误的数据采集逻辑
- [ ] 确保测试连接不会写入不需要的数据
- [ ] 确保只采集自选股池中的股票数据

### 验证任务
- [ ] 验证修复后不再出现非自选股数据
- [ ] 验证数据采集只使用自选股池中的股票
- [ ] 验证测试连接功能正常工作且不写入多余数据

## 相关文件

### 配置文件
- `data/data_sources_config.json`
- `data/production/data_sources_config.json`

### 代码文件
- `src/gateway/web/datasource_routes.py` (测试连接功能)
- `src/gateway/web/data_collectors.py` (数据采集逻辑)
- `src/data/collectors/akshare_collector.py`
- `src/data/collectors/enhanced_akshare_collector.py`

### 数据库表
- `akshare_stock_data`

## 时间估计
- 检查阶段：45分钟
- 修复阶段：30-60分钟（根据问题复杂度）
- 验证阶段：15分钟
- **总计：约1.5-2小时**
