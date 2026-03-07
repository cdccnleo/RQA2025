# 检查 strategy-lifecycle 页面部署模型策略数据持久化计划

## 目标
验证部署模型策略功能的数据持久化是否正确，确保数据能够正确保存到数据库和文件系统。

## 检查范围

### 1. 策略构思数据持久化
- [ ] 检查策略构思是否正确保存到 PostgreSQL 数据库
- [ ] 检查策略构思是否正确保存到文件系统
- [ ] 检查策略构思数据结构完整性

### 2. 生命周期数据持久化
- [ ] 检查生命周期记录是否正确创建
- [ ] 检查生命周期状态是否正确设置
- [ ] 检查生命周期事件是否正确记录

### 3. 模型关联数据持久化
- [ ] 检查模型ID是否正确关联到策略
- [ ] 检查模型参数是否正确保存
- [ ] 检查策略类型是否正确设置为 model_based

### 4. 数据一致性检查
- [ ] 检查数据库和文件系统数据是否一致
- [ ] 检查策略列表是否正确显示新部署的策略
- [ ] 检查生命周期流程是否正确显示

## 具体检查项

### 数据库表检查

#### strategy_conceptions 表
```sql
-- 检查点：
-- 1. 策略ID是否正确生成
-- 2. 策略名称是否正确保存
-- 3. 策略类型是否为 model_based
-- 4. 模型ID是否在 parameters 字段中
-- 5. 创建时间是否正确
```

#### strategy_lifecycle 表
```sql
-- 检查点：
-- 1. 生命周期记录是否正确创建
-- 2. 当前状态是否为 development
-- 3. 策略ID是否正确关联
-- 4. 创建时间是否正确
```

### 文件系统检查

#### 策略文件检查
```
/app/data/
└── strategy_conceptions/
    └── {strategy_id}.json

-- 检查点：
-- 1. 文件是否正确创建
-- 2. JSON 数据是否完整
-- 3. 模型参数是否正确保存
```

#### 生命周期文件检查
```
/app/data/
└── lifecycle/
    └── {strategy_id}.json

-- 检查点：
-- 1. 生命周期文件是否正确创建
-- 2. 初始状态是否为 development
-- 3. 事件记录是否正确
```

## 验证步骤

### 步骤 1: 部署测试
1. 在页面上选择模型
2. 填写策略名称
3. 点击部署按钮
4. 记录返回的策略ID

### 步骤 2: 数据库验证
1. 查询 strategy_conceptions 表
2. 查询 strategy_lifecycle 表
3. 验证数据完整性

### 步骤 3: 文件系统验证
1. 检查策略文件是否存在
2. 检查生命周期文件是否存在
3. 验证 JSON 数据完整性

### 步骤 4: 前端验证
1. 刷新策略列表
2. 检查新策略是否显示
3. 检查生命周期流程是否正确

## 预期结果

### 正常持久化
1. **数据库**: strategy_conceptions 表中新增一条记录
2. **数据库**: strategy_lifecycle 表中新增一条记录
3. **文件系统**: /app/data/strategy_conceptions/{strategy_id}.json 文件创建
4. **文件系统**: /app/data/lifecycle/{strategy_id}.json 文件创建
5. **前端**: 策略列表显示新部署的策略
6. **前端**: 生命周期流程显示为 "已创建" 状态

### 数据结构

#### 策略构思数据
```json
{
  "id": "model_strategy_1234567890",
  "name": "测试策略",
  "type": "model_based",
  "description": "基于模型 model_xxx 的策略",
  "target_market": "stock",
  "risk_level": "medium",
  "nodes": [],
  "connections": [],
  "parameters": {
    "model_id": "model_xxx",
    "prediction_threshold": 0.5,
    "confidence_threshold": 0.7,
    "position_sizing": "equal",
    "max_position_size": 0.1
  },
  "version": 1,
  "created_at": 1234567890,
  "updated_at": 1234567890
}
```

#### 生命周期数据
```json
{
  "strategy_id": "model_strategy_1234567890",
  "strategy_name": "测试策略",
  "current_stage": "development",
  "created_at": "2026-02-19T...",
  "updated_at": "2026-02-19T...",
  "events": [...]
}
```

## 问题排查

### 常见问题
1. **数据库保存失败**: 检查数据库连接和表结构
2. **文件保存失败**: 检查文件系统权限和路径
3. **数据不一致**: 检查数据库和文件系统的同步逻辑
4. **前端不显示**: 检查 API 返回的数据格式

### 修复方案
根据检查结果，可能需要修复：
1. 数据库插入逻辑
2. 文件写入逻辑
3. 数据序列化逻辑
4. 错误处理逻辑
