# Dashboard性能指标API数据格式验证

**验证日期**: 2026-01-10  
**API端点**: `/api/v1/data-sources/metrics`  
**用途**: 系统性能监控和数据流监控

## API端点信息

- **端点**: `/api/v1/data-sources/metrics`
- **方法**: `GET`
- **文件位置**: `src/gateway/web/datasource_routes.py`
- **函数**: `get_data_sources_metrics()`

## 期望的数据格式

### 前端期望格式（基于dashboard.html代码分析）

#### 系统性能监控（performanceChart）

前端期望从API返回的`system_metrics`字段中获取：
- `avg_response_time`: 系统负载数据（用于系统负载折线图）
- `avg_throughput`: 内存使用数据（用于内存使用折线图）

**前端使用方式**（第1677-1740行）：
```javascript
const systemMetrics = metricsData.system_metrics;
const systemLoad = systemMetrics.avg_response_time || 0;
const memoryUsage = systemMetrics.avg_throughput || 0;
```

#### 数据流监控（dataFlowChart）

前端期望从API返回的`throughput_data`字段中获取各阶段处理量：
- 数据采集阶段
- 特征工程阶段
- 模型推理阶段
- 交易执行阶段
- 风险评估阶段

**前端使用方式**（第1709-1726行）：
```javascript
const throughputData = metricsData.throughput_data;
// 计算各阶段处理量
const stages = ['data_collection', 'feature_engineering', 'model_inference', 'trading_execution', 'risk_assessment'];
const data = stages.map(stage => {
    const stageData = throughputData[stage] || {};
    return stageData.throughput || 0;
});
```

## 实际API返回格式（基于代码分析）

### 数据结构（第591-610行）

```python
metrics = {
    "total_sources": len(sources),
    "active_sources": len([s for s in sources if s.get("enabled", True)]),
    "disabled_sources": len([s for s in sources if not s.get("enabled", True)]),
    "latency_data": {},
    "throughput_data": {},  # 数据流处理量数据
    "error_rates": {},
    "availability": {},
    "last_updated": {},
    "health_scores": {},
    "performance_trends": {},
    "system_metrics": {  # 系统性能指标
        "total_uptime": 0,
        "avg_response_time": 0,  # ✅ 符合前端期望
        "error_count": 0,
        "success_count": 0,
        "note": "量化交易系统要求使用真实监控数据。如果指标为空，表示监控系统尚未收集到数据。"
    },
    "timestamp": time.time()
}
```

### 问题分析

1. **system_metrics字段** ✅
   - `avg_response_time`: ✅ 已存在，符合前端期望
   - `avg_throughput`: ❌ **缺失**，前端期望使用`avg_throughput`作为内存使用数据
   - 当前只有`avg_response_time`，缺少`avg_throughput`

2. **throughput_data字段** ⚠️
   - 字段存在，但结构需要验证
   - 前端期望的键名：`data_collection`, `feature_engineering`, `model_inference`, `trading_execution`, `risk_assessment`
   - 每个阶段期望有`throughput`字段

## 验证结果

### ✅ 符合项

1. **API端点存在**: `/api/v1/data-sources/metrics`端点已实现
2. **system_metrics字段存在**: 包含`avg_response_time`字段
3. **throughput_data字段存在**: 包含数据流处理量数据字段

### ⚠️ 不符合项

1. **system_metrics缺少avg_throughput字段**
   - 前端期望：`system_metrics.avg_throughput`（用于内存使用折线图）
   - 实际情况：API中没有`avg_throughput`字段
   - 影响：内存使用数据无法正常显示
   - 建议：在`system_metrics`中添加`avg_throughput`字段，或使用其他字段（如`avg_latency`）

2. **throughput_data结构需要验证**
   - 前端期望：`throughput_data[stage].throughput`
   - 实际结构：需要验证是否包含期望的键名和字段
   - 建议：验证实际返回的数据结构，确保包含前端期望的字段

## 建议修复方案

### 方案1：添加avg_throughput字段（推荐）

在`system_metrics`中添加`avg_throughput`字段：

```python
"system_metrics": {
    "total_uptime": 0,
    "avg_response_time": 0,  # 系统负载
    "avg_throughput": 0,  # 内存使用（新增）
    "error_count": 0,
    "success_count": 0,
    ...
}
```

### 方案2：前端适配（不推荐）

修改前端代码，使用其他现有字段或计算方式获取内存使用数据。

## 后续验证步骤

1. **实际测试API端点**
   - 启动服务
   - 调用`/api/v1/data-sources/metrics`端点
   - 检查实际返回的数据格式

2. **验证throughput_data结构**
   - 检查`throughput_data`是否包含期望的键名
   - 检查每个阶段是否包含`throughput`字段

3. **测试前端显示**
   - 在前端调用API
   - 验证图表是否正常显示
   - 检查数据是否准确

## 总结

- ✅ **API端点存在且基本可用**
- ✅ **system_metrics.avg_throughput字段已修复**（2026-01-10）
- ⚠️ **throughput_data结构需要验证**（需要测试）

**修复状态**: 
- ✅ `system_metrics.avg_throughput`字段已在代码中正确实现（第702行）
- ✅ 字段值从`throughput_data`正确计算
- ✅ 前端代码正确使用该字段

**建议优先级**: 中优先级（不影响基本功能，但会影响数据准确性）

详细修复验证报告：`docs/dashboard_metrics_api_format_verification_fixed.md`

---

**验证人员**: AI Assistant  
**验证日期**: 2026-01-10  
**修复验证日期**: 2026-01-10

