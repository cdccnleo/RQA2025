# 检查一键执行流程数据优先级计划

## 目标
检查一键执行完整流程中所有涉及数据加载和持久化的函数，确保它们优先从PostgreSQL数据库操作，并在失败时降级到文件系统。

## 检查范围

### 1. 回测步骤数据流
- 回测参数加载（策略配置）
- 回测结果保存
- 回测历史记录

### 2. 优化步骤数据流
- 优化结果加载/保存
- 优化进度记录
- 优化历史记录

### 3. 应用步骤数据流
- 策略配置加载
- 策略参数更新
- 策略版本记录

### 4. 工作流状态管理
- 工作流状态加载/保存
- 工作流历史记录

## 检查标准

每个数据操作函数应遵循以下模式：
```python
def load_data():
    # 1. 优先从PostgreSQL加载
    try:
        result = load_from_postgresql()
        if result:
            return result
    except Exception as e:
        logger.warning(f"从PostgreSQL加载失败: {e}")
    
    # 2. 降级到文件系统
    try:
        result = load_from_filesystem()
        if result:
            return result
    except Exception as e:
        logger.error(f"从文件系统加载失败: {e}")
    
    return None

def save_data(data):
    # 1. 优先保存到PostgreSQL
    pg_success = False
    try:
        save_to_postgresql(data)
        pg_success = True
    except Exception as e:
        logger.warning(f"保存到PostgreSQL失败: {e}")
    
    # 2. 同时保存到文件系统（双写机制）
    try:
        save_to_filesystem(data)
    except Exception as e:
        logger.error(f"保存到文件系统失败: {e}")
        if not pg_success:
            raise  # 两者都失败才抛出异常
```

## 实施步骤

### 第一阶段：识别所有数据操作函数（30分钟）
1. 搜索所有加载函数（load_*）
2. 搜索所有保存函数（save_*）
3. 搜索所有持久化函数（persist_*）
4. 列出涉及的数据实体

### 第二阶段：检查每个函数的实现（45分钟）
1. 检查回测相关函数
2. 检查优化相关函数
3. 检查策略相关函数
4. 检查工作流相关函数

### 第三阶段：修复不符合标准的函数（45分钟）
1. 修改加载函数优先级
2. 修改保存函数双写机制
3. 添加异常处理和日志
4. 确保降级机制正确

### 第四阶段：验证修复（30分钟）
1. 测试完整一键执行流程
2. 验证PostgreSQL优先加载
3. 验证文件系统降级
4. 测试双写机制

## 预期产出
1. 数据操作函数清单
2. 修复后的代码
3. 测试验证报告
