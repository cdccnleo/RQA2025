# PostgreSQLDataLoader 迭代优化开发计划

## 概述

基于数据管理层架构设计文档和实际代码分析，制定 PostgreSQLDataLoader 的迭代优化计划，分阶段实施以提升功能完整性和架构一致性。

## 当前状态

### 已实现功能
- ✅ 数据库连接池管理
- ✅ 环境变量配置读取
- ✅ SQL 查询执行
- ✅ 股票日线数据加载
- ✅ 多股票批量加载
- ✅ 连接验证
- ✅ 单例模式管理

### 缺失功能
- ❌ 自动重试机制
- ❌ 缓存集成
- ❌ 质量检查
- ❌ 性能监控
- ❌ 限流控制

## 迭代计划

### 第一阶段：基础功能增强（立即实施）

#### 1.1 添加自动重试机制
**目标**: 实现指数退避重试策略
**文件**: `src/data/loader/postgresql_loader.py`
**改动**:
- 添加 `load_with_retry()` 方法
- 支持可配置的重试次数和退避策略
- 记录每次重试的日志

**代码示例**:
```python
def load_with_retry(self, query: str, params: Optional[Dict] = None, 
                    max_retries: int = 3, backoff_factor: float = 2.0) -> LoadResult:
    """带重试的加载方法"""
    for attempt in range(max_retries):
        result = self.load(query, params)
        if result.success:
            return result
        
        if attempt < max_retries - 1:
            wait_time = backoff_factor ** attempt
            logger.warning(f"查询失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{max_retries})")
            time.sleep(wait_time)
    
    return result
```

#### 1.2 集成缓存系统
**目标**: 集成多级缓存（内存 + Redis）
**文件**: `src/data/loader/postgresql_loader.py`
**改动**:
- 添加 `CacheManager` 集成
- 实现 `load_with_cache()` 方法
- 支持缓存键生成和 TTL 配置

**代码示例**:
```python
from ..cache.cache_manager import CacheManager

def load_with_cache(self, query: str, params: Optional[Dict] = None, 
                   cache_ttl: int = 300) -> LoadResult:
    """带缓存的加载方法"""
    cache_key = self._generate_cache_key(query, params)
    
    # 检查缓存
    if self.cache_manager:
        cached_data = self.cache_manager.get(cache_key)
        if cached_data is not None:
            logger.debug(f"缓存命中: {cache_key}")
            return LoadResult(
                data=cached_data,
                success=True,
                message="从缓存加载",
                row_count=len(cached_data) if hasattr(cached_data, '__len__') else 0,
                load_time_ms=0
            )
    
    # 加载数据
    result = self.load(query, params)
    if result.success and self.cache_manager:
        self.cache_manager.set(cache_key, result.data, ttl=cache_ttl)
    
    return result
```

**预计时间**: 2-3 小时
**优先级**: 🔴 高

---

### 第二阶段：质量与监控集成（后续迭代）

#### 2.1 集成数据质量检查
**目标**: 集成统一质量监控器
**文件**: `src/data/loader/postgresql_loader.py`
**改动**:
- 添加 `UnifiedQualityMonitor` 集成
- 实现 `load_with_quality_check()` 方法
- 支持质量阈值配置

**代码示例**:
```python
from ..quality.unified_quality_monitor import UnifiedQualityMonitor, QualityConfig

def load_with_quality_check(self, query: str, params: Optional[Dict] = None,
                           quality_threshold: float = 0.8) -> LoadResult:
    """带质量检查的加载方法"""
    result = self.load(query, params)
    
    if result.success and result.data is not None and self.quality_monitor:
        quality_result = self.quality_monitor.check_quality(
            result.data, DataSourceType.STOCK
        )
        
        metrics = quality_result.get('metrics', {})
        overall_score = getattr(metrics, 'overall_score', 0)
        
        if overall_score < quality_threshold:
            logger.warning(f"数据质量得分较低: {overall_score:.2f}")
            result.metadata['quality_warning'] = True
            result.metadata['quality_score'] = overall_score
    
    return result
```

#### 2.2 集成性能监控
**目标**: 集成性能监控器
**文件**: `src/data/loader/postgresql_loader.py`
**改动**:
- 添加 `PerformanceMonitor` 集成
- 实现操作跟踪和指标收集
- 支持性能报告生成

**代码示例**:
```python
from ..monitoring.performance_monitor import PerformanceMonitor

def load_with_monitoring(self, query: str, params: Optional[Dict] = None) -> LoadResult:
    """带性能监控的加载方法"""
    if not self.performance_monitor:
        return self.load(query, params)
    
    with self.performance_monitor.track_operation("postgresql_load"):
        result = self.load(query, params)
        
        # 记录指标
        if result.success:
            self.performance_monitor.record_metric(
                "postgresql_load_success", 1
            )
            self.performance_monitor.record_metric(
                "postgresql_load_time", result.load_time_ms
            )
        else:
            self.performance_monitor.record_metric(
                "postgresql_load_failure", 1
            )
        
        return result
```

**预计时间**: 3-4 小时
**优先级**: 🟡 中

---

### 第三阶段：架构统一（长期规划）

#### 3.1 统一接口实现
**目标**: 实现架构设计中定义的 `DataLoader` 接口
**文件**: `src/data/loader/postgresql_loader.py`
**改动**:
- 修改 `LoadResult` 为 `DataResponse`
- 统一配置类为 `LoaderConfig`
- 实现标准接口方法

**代码示例**:
```python
from ..core.base_loader import DataLoader
from src.infrastructure.interfaces.standard_interfaces import DataRequest, DataResponse

class PostgreSQLDataLoader(DataLoader):
    """符合架构设计的 PostgreSQL 数据加载器"""
    
    def load_data(self, request: DataRequest, **kwargs) -> DataResponse:
        """标准数据加载接口"""
        # 转换 DataRequest 为内部查询
        query = self._build_query(request)
        params = self._build_params(request)
        
        result = self.load_with_retry(query, params)
        
        return DataResponse(
            request=request,
            data=result.data,
            success=result.success,
            error_message=result.message if not result.success else None
        )
```

#### 3.2 数据湖集成
**目标**: 与 `DataLakeManager` 集成
**文件**: `src/data/loader/postgresql_loader.py`
**改动**:
- 添加 `DataLakeManager` 集成
- 实现数据湖存储和读取
- 支持数据版本控制

**代码示例**:
```python
from ..lake.data_lake_manager import DataLakeManager

def load_to_data_lake(self, query: str, dataset_name: str,
                     params: Optional[Dict] = None) -> str:
    """加载数据到数据湖"""
    result = self.load(query, params)
    
    if result.success and self.data_lake_manager:
        file_path = self.data_lake_manager.store_data(
            data=result.data,
            dataset_name=dataset_name,
            metadata={
                'source': 'postgresql',
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
        )
        return file_path
    
    return None
```

**预计时间**: 4-6 小时
**优先级**: 🟢 低

---

## 实施时间表

| 阶段 | 任务 | 预计时间 | 优先级 | 依赖 |
|------|------|----------|--------|------|
| 第一阶段 | 自动重试机制 | 1-2 小时 | 🔴 高 | 无 |
| 第一阶段 | 缓存集成 | 1-2 小时 | 🔴 高 | 无 |
| 第二阶段 | 质量检查 | 2 小时 | 🟡 中 | 第一阶段 |
| 第二阶段 | 性能监控 | 1-2 小时 | 🟡 中 | 第一阶段 |
| 第三阶段 | 统一接口 | 2-3 小时 | 🟢 低 | 第二阶段 |
| 第三阶段 | 数据湖集成 | 2-3 小时 | 🟢 低 | 第二阶段 |

**总计**: 约 10-15 小时

---

## 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 缓存集成导致内存溢出 | 中 | 高 | 设置缓存大小限制，启用 LRU 淘汰 |
| 重试机制导致响应延迟 | 低 | 中 | 设置最大重试次数，使用指数退避 |
| 质量检查影响性能 | 中 | 中 | 异步执行质量检查，可配置开关 |
| 接口变更导致兼容性问题 | 高 | 高 | 保持向后兼容，逐步迁移 |

---

## 验收标准

### 第一阶段验收
- [ ] 自动重试机制正常工作（3次重试，指数退避）
- [ ] 缓存命中率 > 80%
- [ ] 所有现有测试通过

### 第二阶段验收
- [ ] 质量检查正确识别数据问题
- [ ] 性能监控指标正确收集
- [ ] 告警机制正常工作

### 第三阶段验收
- [ ] 符合架构设计接口规范
- [ ] 数据湖集成正常工作
- [ ] 性能提升 > 20%

---

## 下一步行动

1. **立即开始**: 第一阶段开发（自动重试 + 缓存集成）
2. **测试验证**: 每个阶段完成后进行测试
3. **文档更新**: 更新相关文档和注释
4. **代码审查**: 每个阶段完成后进行代码审查
