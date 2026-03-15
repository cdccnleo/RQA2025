# 特征选择过程实现及数据持久化全面检查计划

## 目标
全面系统化检查特征选择过程的实现，以及 feature_selection_history 数据持久化机制，确保：
1. 特征选择任务按股票数据正确执行
2. 任务ID格式与特征提取任务一致
3. 数据持久化与股票代码关联
4. 历史记录数据完整准确

## 检查范围

### 1. 特征选择任务处理器实现检查

#### 1.1 任务ID生成机制
- [ ] 检查任务ID格式是否符合规范
- [ ] 对比特征提取任务ID格式
- [ ] 验证任务ID是否包含股票代码信息

#### 1.2 按股票代码执行逻辑
- [ ] 检查是否按股票代码循环处理
- [ ] 验证每个股票独立执行特征选择
- [ ] 检查股票代码是否正确传递到选择算法

#### 1.3 参数解析与传递
- [ ] 检查payload参数解析
- [ ] 验证symbols参数处理
- [ ] 检查method、top_k等参数传递

### 2. 特征选择算法实现检查

#### 2.1 特征选择器实现
- [ ] 检查FeatureSelector类实现
- [ ] 验证各种选择方法（importance/correlation/mutual_info/kbest）
- [ ] 检查特征质量评估逻辑

#### 2.2 数据过滤与处理
- [ ] 检查按股票代码过滤特征数据
- [ ] 验证质量阈值过滤
- [ ] 检查特征选择结果生成

### 3. 数据持久化机制检查

#### 3.1 feature_selection_history表结构
- [ ] 检查表字段定义
- [ ] 验证symbol字段是否存在
- [ ] 检查外键关联关系

#### 3.2 历史记录保存逻辑
- [ ] 检查保存时机（任务完成时）
- [ ] 验证保存的数据完整性
- [ ] 检查symbol字段是否正确填充

#### 3.3 历史记录查询逻辑
- [ ] 检查get_selection_history实现
- [ ] 验证按股票代码查询功能
- [ ] 检查时间范围过滤逻辑

### 4. 数据关联性检查

#### 4.1 任务与历史记录关联
- [ ] 检查task_id关联
- [ ] 验证selection_id生成
- [ ] 检查股票代码在记录中的体现

#### 4.2 特征数据与选择结果关联
- [ ] 检查input_features记录
- [ ] 验证selected_features记录
- [ ] 检查特征与股票的对应关系

### 5. 当前数据问题分析

#### 5.1 现有数据检查
- [ ] 分析现有feature_selection_history数据
- [ ] 检查symbol字段是否为空
- [ ] 验证数据与股票代码的关联性

#### 5.2 问题定位
- [ ] 定位symbol字段缺失原因
- [ ] 分析数据持久化流程中的问题
- [ ] 确定修复方案

## 检查步骤

### 阶段1：代码审查
1. **审查特征选择处理器**
   - 文件：`src/core/orchestration/scheduler/handlers/feature_selection_handler.py`
   - 重点：任务ID生成、按股票执行、参数传递

2. **审查特征选择器实现**
   - 文件：`src/features/selection/feature_selector.py`
   - 重点：选择算法、数据过滤、结果生成

3. **审查历史记录管理器**
   - 文件：`src/features/selection/feature_selector_history.py`
   - 重点：保存逻辑、表结构、查询逻辑

### 阶段2：数据库检查
1. **检查表结构**
   ```sql
   \d feature_selection_history
   ```

2. **检查现有数据**
   ```sql
   SELECT selection_id, task_id, selection_method, symbol, timestamp 
   FROM feature_selection_history 
   ORDER BY timestamp DESC;
   ```

3. **验证symbol字段**
   ```sql
   SELECT COUNT(*) as total,
          COUNT(symbol) as with_symbol,
          COUNT(*) - COUNT(symbol) as without_symbol
   FROM feature_selection_history;
   ```

### 阶段3：API测试
1. **测试特征选择任务提交**
   - 提交多股票特征选择任务
   - 验证任务执行

2. **测试历史记录查询**
   - 查询选择历史API
   - 验证返回数据完整性

3. **测试按股票查询**
   - 按股票代码查询选择历史
   - 验证关联性

### 阶段4：问题修复
1. **修复表结构**
   - 添加symbol字段（如缺失）
   - 添加索引优化查询

2. **修复保存逻辑**
   - 确保symbol字段正确填充
   - 完善数据关联

3. **修复查询逻辑**
   - 支持按股票查询
   - 优化数据返回

## 预期结果

### 数据完整性
- [ ] feature_selection_history表包含symbol字段
- [ ] 每条记录都有对应的股票代码
- [ ] 任务ID格式统一（包含股票代码）

### 功能正确性
- [ ] 特征选择按股票独立执行
- [ ] 历史记录正确保存
- [ ] 支持按股票查询选择历史

### 关联性
- [ ] 选择历史与股票代码强关联
- [ ] 选择结果与特征数据对应
- [ ] 任务与历史记录一一对应

## 修复方案

### 方案1：表结构修复
```sql
-- 添加symbol字段
ALTER TABLE feature_selection_history 
ADD COLUMN IF NOT EXISTS symbol VARCHAR(20);

-- 添加索引
CREATE INDEX IF NOT EXISTS idx_feature_selection_symbol 
ON feature_selection_history(symbol);

-- 更新现有数据（如可能）
UPDATE feature_selection_history 
SET symbol = SUBSTRING(task_id FROM 'task_(\w+)_') 
WHERE symbol IS NULL;
```

### 方案2：保存逻辑修复
```python
# 在保存历史记录时确保symbol字段
record = {
    "selection_id": selection_id,
    "task_id": task_id,
    "symbol": symbol,  # 确保包含股票代码
    "selection_method": method,
    ...
}
```

### 方案3：查询逻辑优化
```python
# 支持按股票查询
def get_selection_history(symbol=None, ...):
    query = "SELECT * FROM feature_selection_history WHERE 1=1"
    if symbol:
        query += " AND symbol = %s"
    ...
```

## 输出物

1. **检查报告**
   - 代码审查结果
   - 数据库检查结果
   - API测试结果

2. **问题清单**
   - 发现的所有问题
   - 优先级排序

3. **修复记录**
   - 修复内容
   - 验证结果

4. **改进建议**
   - 架构优化建议
   - 性能优化建议
