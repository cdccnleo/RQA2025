# 首次数据采集补全指南

## 📋 概述

首次数据采集时，由于没有历史数据记录，系统需要通过特殊的补全策略来建立完整的数据基础。本指南详细说明首次采集的补全流程和最佳实践。

---

## 🎯 首次采集补全机制

### 1️⃣ **自动检测首次采集**

系统通过以下方式识别首次采集：

```python
# 在data_complement_scheduler.py中
def _should_trigger_complement(self, schedule: ComplementSchedule) -> bool:
    # 如果从未补全过，立即触发
    if schedule.last_complement_date is None:
        return True  # 首次采集立即触发补全
```

**触发条件**：
- `schedule.last_complement_date is None` - 从未执行过补全任务
- 新注册的数据源会自动创建补全调度配置

### 2️⃣ **首次补全策略配置**

系统为不同类型数据源预设首次补全策略：

```json
{
  "core_stocks": {
    "mode": "MONTHLY",
    "priority": "CRITICAL",
    "schedule_interval_days": 1,
    "complement_window_days": 30,
    "min_gap_days": 25,
    "description": "核心股票：首次补全最近30天数据"
  },
  "major_indices": {
    "mode": "WEEKLY",
    "priority": "HIGH",
    "schedule_interval_days": 1,
    "complement_window_days": 7,
    "min_gap_days": 5,
    "description": "主要指数：首次补全最近7天数据"
  },
  "all_stocks": {
    "mode": "QUARTERLY",
    "priority": "MEDIUM",
    "schedule_interval_days": 7,
    "complement_window_days": 90,
    "min_gap_days": 80,
    "description": "全市场股票：首次补全最近90天数据"
  },
  "macro_data": {
    "mode": "SEMI_ANNUAL",
    "priority": "LOW",
    "schedule_interval_days": 30,
    "complement_window_days": 180,
    "min_gap_days": 150,
    "description": "宏观数据：首次补全最近180天数据"
  }
}
```

### 3️⃣ **首次补全时间窗口计算**

**核心逻辑**（在`_create_complement_task`方法中）：

```python
def _create_complement_task(self, schedule: ComplementSchedule) -> ComplementTask:
    current_time = datetime.now()

    # 首次补全时间范围计算
    if schedule.last_complement_date is None:  # 首次补全
        # 补全最近的窗口期数据
        start_date = current_time - timedelta(days=schedule.complement_window_days)
        end_date = current_time
    else:
        # 非首次：从上次补全时间开始
        start_date = schedule.last_complement_date
        end_date = current_time
```

**首次补全示例**：
- 核心股票：补全最近30天的完整交易数据
- 主要指数：补全最近7天的指数数据
- 全市场股票：补全最近90天的市场数据
- 宏观数据：补全最近180天的宏观指标

---

## 🔄 首次采集补全执行流程

### **步骤1：数据源注册与配置**

```python
# 注册新的数据源补全配置
from src.core.orchestration.data_complement_scheduler import get_data_complement_scheduler

scheduler = get_data_complement_scheduler()

# 为新数据源注册补全配置
scheduler.register_complement_schedule(
    source_id='000001',           # 数据源ID
    data_type='stock',            # 数据类型
    mode=ComplementMode.MONTHLY,  # 补全模式
    priority=ComplementPriority.CRITICAL  # 补全优先级
)
```

### **步骤2：自动触发补全检查**

系统每日自动检查所有注册的数据源：

```python
# 每日补全检查（在调度器中自动执行）
for source_id in registered_sources:
    is_needed, task = scheduler.check_complement_needed(source_id)
    if is_needed and task:
        # 提交补全任务到优先级队列
        priority_manager.enqueue_task(task)
```

### **步骤3：补全任务分批执行**

首次补全任务会被分解为多个小批次：

```python
# 批次分解示例（30天补全任务）
# 核心股票：批次大小7天，分4个批次并行处理
batches = [
    {"batch_1": "2024-01-01 ~ 2024-01-07", "records": 1250},
    {"batch_2": "2024-01-08 ~ 2024-01-14", "records": 1300},
    {"batch_3": "2024-01-15 ~ 2024-01-21", "records": 1180},
    {"batch_4": "2024-01-22 ~ 2024-01-28", "records": 1220}
]
```

### **步骤4：并行补全执行**

```python
# 补全任务并行执行
async def execute_complement_batch(batch: ComplementBatch):
    # 1. 确定补全时间范围
    start_date = batch.start_date
    end_date = batch.end_date

    # 2. 从数据源获取缺失数据
    missing_data = await fetch_missing_data(
        source_id=batch.source_id,
        start_date=start_date,
        end_date=end_date
    )

    # 3. 数据验证和清洗
    validated_data = validate_and_clean(missing_data)

    # 4. 批量插入数据库
    await batch_insert_to_database(validated_data)

    # 5. 更新补全进度
    await update_progress(batch.batch_id, completed=True)
```

---

## 📊 首次补全数据量估算

### **记录数估算逻辑**

```python
def _estimate_complement_records(self, schedule: ComplementSchedule,
                               start_date: datetime, end_date: datetime) -> int:
    # 计算时间范围（天数）
    days_range = (end_date - start_date).days

    # 根据数据类型估算日均记录数
    daily_estimates = {
        'stock': 1,      # 股票：每日1条记录
        'index': 1,      # 指数：每日1条记录
        'macro': 0.1,    # 宏观：平均每10天1条记录
        'news': 10,      # 新闻：每日10条记录
        'bond': 0.5,     # 债券：每2天1条记录
        'futures': 2,    # 期货：每日2条记录
        'forex': 5,      # 外汇：每日5条记录
        'crypto': 10     # 加密货币：每日10条记录
    }

    daily_rate = daily_estimates.get(schedule.data_type, 1)
    estimated_records = int(days_range * daily_rate)

    return estimated_records
```

### **典型首次补全数据量**

| 数据类型 | 补全周期 | 估算记录数 | 实际处理时间 |
|----------|----------|------------|--------------|
| **核心股票** | 30天 | ~30条 | 5-10分钟 |
| **主要指数** | 7天 | ~7条 | 2-5分钟 |
| **全市场股票** | 90天 | ~90条 | 15-30分钟 |
| **宏观数据** | 180天 | ~18条 | 3-8分钟 |
| **新闻数据** | 30天 | ~300条 | 10-20分钟 |
| **债券数据** | 90天 | ~45条 | 8-15分钟 |
| **期货数据** | 30天 | ~60条 | 10-15分钟 |

---

## ⚡ 首次补全优化策略

### **1. 优先级排序执行**

```python
# 补全任务优先级排序
priority_order = [
    "core_stocks",      # 1. 核心股票 - CRITICAL
    "major_indices",    # 2. 主要指数 - HIGH
    "futures_data",     # 3. 期货数据 - HIGH
    "all_stocks",       # 4. 全市场股票 - MEDIUM
    "bond_data",        # 5. 债券数据 - MEDIUM
    "forex_data",       # 6. 外汇数据 - MEDIUM
    "crypto_data",      # 7. 加密货币 - MEDIUM
    "news_data",        # 8. 新闻数据 - LOW
    "macro_data"        # 9. 宏观数据 - LOW
]
```

### **2. 分阶段补全策略**

```python
# 分阶段补全执行计划
complement_phases = {
    "phase_1_critical": {  # 第一阶段：关键数据
        "duration": "1-2天",
        "targets": ["core_stocks", "major_indices"],
        "parallel_limit": 2,
        "description": "优先补全交易决策所需的核心数据"
    },
    "phase_2_important": {  # 第二阶段：重要数据
        "duration": "3-5天",
        "targets": ["futures_data", "bond_data"],
        "parallel_limit": 3,
        "description": "补全重要的衍生品和固定收益数据"
    },
    "phase_3_standard": {  # 第三阶段：标准数据
        "duration": "1-2周",
        "targets": ["all_stocks", "forex_data", "crypto_data"],
        "parallel_limit": 5,
        "description": "补全全市场和国际市场数据"
    },
    "phase_4_auxiliary": {  # 第四阶段：辅助数据
        "duration": "2-4周",
        "targets": ["news_data", "macro_data"],
        "parallel_limit": 2,
        "description": "补全新闻和宏观分析数据"
    }
}
```

### **3. 资源分配优化**

```python
# 根据数据源重要性分配系统资源
resource_allocation = {
    "CRITICAL": {
        "max_concurrent_batches": 3,
        "cpu_priority": "high",
        "memory_limit": "2GB",
        "network_priority": "high"
    },
    "HIGH": {
        "max_concurrent_batches": 2,
        "cpu_priority": "high",
        "memory_limit": "1GB",
        "network_priority": "normal"
    },
    "MEDIUM": {
        "max_concurrent_batches": 1,
        "cpu_priority": "normal",
        "memory_limit": "512MB",
        "network_priority": "normal"
    },
    "LOW": {
        "max_concurrent_batches": 1,
        "cpu_priority": "low",
        "memory_limit": "256MB",
        "network_priority": "low"
    }
}
```

---

## 🛠️ 首次补全故障排除

### **常见问题及解决方案**

#### **问题1：补全任务未触发**
```python
# 检查补全调度配置
schedule = scheduler.schedules.get(source_id)
if schedule is None:
    # 解决方案：重新注册补全配置
    scheduler.register_complement_schedule(source_id, data_type, mode, priority)
```

#### **问题2：补全进度缓慢**
```python
# 检查系统负载和批次大小
current_load = get_system_load()
if current_load > 0.8:
    # 解决方案：减少并发批次数，增大批次间隔
    processor.max_concurrent_batches = max(1, processor.max_concurrent_batches - 1)
```

#### **问题3：数据质量问题**
```python
# 检查补全数据的质量
quality_report = validator.validate_batch(complement_data, data_type)
if quality_report['quality_score'] < 0.8:
    # 解决方案：启用数据清洗和重新验证
    cleaned_data = clean_invalid_records(complement_data)
    await revalidate_and_store(cleaned_data)
```

#### **问题4：数据库连接超时**
```python
# 检查数据库连接和批量插入配置
try:
    # 减小批次大小
    batch_size = max(10, batch_size // 2)
    # 增加重试次数
    max_retries = 5
    # 启用连接池
    connection_pool = get_connection_pool()
except Exception as e:
    logger.error(f"数据库操作失败: {e}")
    # 回退到单条插入模式
    await fallback_single_insert(data)
```

---

## 📈 首次补全效果验证

### **补全完成度检查**

```python
def verify_complement_completion(source_id: str) -> Dict[str, Any]:
    """验证补全完成度"""
    # 检查数据连续性
    data_gaps = check_data_continuity(source_id)

    # 检查数据质量
    quality_metrics = assess_data_quality(source_id)

    # 计算补全覆盖率
    coverage_rate = calculate_coverage_rate(source_id)

    return {
        "data_gaps": len(data_gaps),
        "quality_score": quality_metrics['overall_score'],
        "coverage_rate": coverage_rate,
        "is_complete": len(data_gaps) == 0 and coverage_rate >= 0.95
    }
```

### **补全效果指标**

| 指标 | 目标值 | 验证方法 |
|------|--------|----------|
| **数据连续性** | 0个缺口 | 检查时间序列连续性 |
| **覆盖率** | ≥95% | (实际记录/期望记录) |
| **质量评分** | ≥0.85 | 数据完整性+准确性+一致性 |
| **补全耗时** | 按计划完成 | 实际完成时间vs计划时间 |

---

## 🎯 最佳实践建议

### **1. 准备工作**
- ✅ 提前规划补全优先级和时间安排
- ✅ 确保系统资源充足（CPU、内存、网络）
- ✅ 备份现有数据，避免补全过程出现问题
- ✅ 配置监控告警，及时发现异常情况

### **2. 执行策略**
- ✅ 按照优先级顺序执行，先核心数据后辅助数据
- ✅ 监控系统负载，避免影响现有业务
- ✅ 分批次执行，避免一次性压力过大
- ✅ 定期检查补全进度和数据质量

### **3. 质量保障**
- ✅ 对补全数据进行完整性验证
- ✅ 检查数据一致性和准确性
- ✅ 对比补全前后数据统计，确认数据合理性
- ✅ 建立补全数据的使用验证机制

### **4. 运维监控**
- ✅ 设置补全任务的监控指标
- ✅ 配置异常情况的告警机制
- ✅ 记录补全过程的详细日志
- ✅ 建立补全任务的回滚机制

---

**总结**：首次数据采集补全是一个系统性的过程，通过智能调度、优先级管理、分批处理等机制，确保在不影响现有业务的前提下，有序高效地建立完整的数据基础。补全完成后，系统将进入正常的增量采集模式，确保数据的持续更新和完整性。