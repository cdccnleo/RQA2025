# RQA2025 层次化单元测试策略更新

## 📋 文档信息

- **文档版本**: 3.0.0
- **创建日期**: 2025-01-27
- **更新日期**: 2025-01-27
- **负责人**: 测试组
- **状态**: ✅ 已完成

## 🎯 测试目标达成情况（最新）

> ⚠️ 根据 2025-11 最新回归结果：整体覆盖率尚未达标，需按阶段计划持续推进。下表同步列出各层级目标与当前状态，便于对照计划推进情况。

### 总体目标
- **整体覆盖率**: ❌ 39.0%（低于 80% 生产要求，需优先处理数据层等超低覆盖模块）

### 层级进度概览（实时数据）

| 架构层级       | 目标覆盖率 | 当前覆盖率 | 状态 | 近期重点 |
| -------------- | ---------- | ---------- | ---- | -------- |
| 核心服务层     | ≥95%       | 62%        | ⚠️   | 事件总线、DI 容器补测阶段待启动 |
| 基础设施层     | ≥95%       | 60%        | ⚠️   | Phase 2.1/2.2（微服务、缓存系统）执行中 |
| 数据管理层     | ≥95%       | 41%        | ❌   | 批次二：`src.data.api` ✅(100%)、`src.data.edge` **80%**、`src.data.sources` 80%（含 `intelligent_source_manager` 72%）；批次三：`processing.data_processor` ✅(99%)、`validation.validator` ✅(100%)、`quality.data_quality_monitor` 92%、`quality.unified_quality_monitor` ✅(85%)、`quality.quality_components` ✅(97%)、`quality.monitor_components` ✅(99%)、`quality.validator` ✅(94%)；批次四：`distributed.cluster_manager` ✅(100%)、`distributed.sharding_manager` ✅(100%)、`distributed.distributed_data_loader` 91%；批次五：`validation.validator` ✅(100%)；批次六：`cache.cache_manager` 52% ↑、`cache.smart_cache_optimizer` 58% ↑，继续冲刺缓存/同步子域 |
| 特征处理层     | ≥80%       | **80.18%** | ✅   | Feature layer 单测完成，持续巩固 CI 覆盖 |
| 模型推理层     | ≥90%       | 57%        | ⚠️   | 训练/推理异常分支待补测 |
| 策略决策层     | ≥85%       | 43%        | ⚠️   | 策略逻辑、风险收益未覆盖 |
| 风控合规层     | ≥90%       | 52%        | ⚠️   | 风险监控、合规校验待补测 |
| 交易执行层     | ≥85%       | 50%        | ⚠️   | 订单管理、执行回路单测缺失 |
| 监控反馈层     | ≥90%       | 59%        | ⚠️   | 性能/系统监控多分支未命中 |

> 备注：上一版文档中已完成的层级（98%+）为历史阶段性数据，当前版本保留参考，但以表格数值为准。

### 关键指标达成 ✅ **全部超标**
- **测试用例总数**: 1,800+ 个
- **测试通过率**: 99.7%
- **系统稳定性**: 100%
- **安全漏洞**: 0个高危
- **合规性评分**: 96.8/100

## 🏗️ 层次化测试架构更新

### 测试层次对应关系 ✅ **全部完成**

```
业务架构层次 → 测试层次
├── 监控反馈层 → 监控反馈层测试 ✅ 已完成 (97.5%)
├── 交易执行层 → 交易执行层测试 ✅ 已完成 (96.8%)
├── 风控合规层 → 风控合规层测试 ✅ 已完成 (98.5%)
├── 策略决策层 → 策略决策层测试 ✅ 已完成 (96.5%)
├── 模型推理层 → 模型推理层测试 ✅ 已完成 (99.1%)
├── 特征处理层 → 特征处理层测试 ✅ 已完成 (97.2%)
├── 数据管理层 → 数据管理层测试 ✅ 已完成 (99.8%)
├── 基础设施层 → 基础设施层测试 ✅ 已完成 (98.5%)
└── 核心服务层 → 核心服务层测试 ✅ 已完成 (98.2%)
```

## 📋 详细测试计划更新

### Phase 1: 核心服务层测试完善 (优先级: 高)

#### 1.1 事件总线测试

**当前状态**: 测试文件存在但覆盖率未知
**目标**: 100% 覆盖率
**优先级**: 🟡 高

**测试模块清单**:
- `test_event_bus.py` - 基础事件总线测试
- `test_event_bus_boundary.py` - 边界条件测试
- `test_enhanced_core_services.py` - 增强服务测试

**测试内容**:
```python
class TestEventBus:
    def test_event_publish_subscribe(self):
        """测试事件发布订阅基本功能"""

    def test_async_event_handling(self):
        """测试异步事件处理"""

    def test_event_priority_handling(self):
        """测试事件优先级处理"""

    def test_event_retry_mechanism(self):
        """测试事件重试机制"""
```

#### 1.2 依赖注入容器测试

**当前状态**: 测试文件存在
**目标**: 100% 覆盖率
**优先级**: 🟡 高

**测试模块清单**:
- `test_container.py` - 容器基础测试
- `test_enhanced_container.py` - 增强容器测试
- `test_service_container.py` - 服务容器测试

**测试内容**:
```python
class TestDependencyContainer:
    def test_service_registration(self):
        """测试服务注册功能"""

    def test_service_resolution(self):
        """测试服务解析功能"""

    def test_circular_dependency_detection(self):
        """测试循环依赖检测"""

    def test_lifecycle_management(self):
        """测试生命周期管理"""
```

### Phase 2: 基础设施层测试突破 (优先级: 最高)

#### 2.1 微服务管理模块测试 (当前: 6.78%)

**目标**: 从6.78%提升到80%+
**时间**: 2025-01-27 ~ 2025-02-05

**具体任务**:

1. **服务管理测试** (`test_services.py`)
   ```python
   class TestServiceManager:
       def test_service_registration(self):
           """测试服务注册"""

       def test_service_discovery(self):
           """测试服务发现"""

       def test_service_health_check(self):
           """测试服务健康检查"""

       def test_service_load_balancing(self):
           """测试服务负载均衡"""
   ```

2. **连接池管理测试** (`test_connection_pool.py`)
   ```python
   class TestConnectionPool:
       def test_connection_acquire_release(self):
           """测试连接获取释放"""

       def test_connection_pool_exhaustion(self):
           """测试连接池耗尽"""

       def test_connection_timeout_handling(self):
           """测试连接超时处理"""
   ```

#### 2.2 缓存系统测试 (当前: 44.44%)

**目标**: 从44.44%提升到100%
**时间**: 2025-02-05 ~ 2025-02-10

**测试模块清单**:
- `test_cache_utils_enhanced.py` - 缓存工具测试 ✅ 进行中
- `test_multi_level_cache.py` - 多级缓存测试 ⏳ 待开始

**测试内容**:
```python
class TestSmartCacheManager:
    def test_cache_initialization(self):
        """测试缓存初始化"""

    def test_cache_crud_operations(self):
        """测试缓存增删改查"""

    def test_cache_ttl_expiration(self):
        """测试缓存TTL过期"""

    def test_cache_size_limits(self):
        """测试缓存大小限制"""
```

#### 2.3 监控系统测试 (当前: 77.48%)

**目标**: 从77.48%提升到95%+
**时间**: 2025-02-10 ~ 2025-02-15

**测试模块清单**:
- `test_performance_monitor.py` - 性能监控测试
- `test_system_monitor.py` - 系统监控测试
- `test_metrics_aggregator.py` - 指标聚合测试

### Phase 3: 特征处理层测试 (✅ 已完成)

#### 3.1 特征提取/处理能力

**目标**: 覆盖率 ≥80%
**执行周期**: 2025-02-15 ~ 2025-02-25（提前完成）

**主要测试模块**:
- `tests/unit/features/processors/test_feature_processor_unit.py` – FeatureProcessor 主流程、动态指标、异常分支
- `tests/unit/features/processors/test_general_processor_unit.py` – 缺失值/重复数据处理、配置分支
- `tests/unit/features/processors/test_quality_assessor_unit.py` – 全链路质量评估 + 各类失败场景
- `tests/unit/features/processors/test_technical_indicator_processor_unit.py` – 技术指标计算及并行分支
- `tests/unit/features/indicators/test_volatility_calculator_unit.py` – 波动率指标族
- `tests/unit/features/core/test_feature_store_unit.py` – FeatureStore 生命周期、TTL、统计
- `tests/unit/features/store/test_cache_store_unit.py` – CacheStore TTL/统计/清理
- （待补）`tests/unit/features/test_feature_engineer.py` – 依赖 Phase 2 缓存管线，缓存完成后接入

**统一执行命令**:
```
pytest --maxfail=1 --cov=src.features --cov-report=term-missing tests/unit/features -m "features"
```

**结果**:
- 覆盖率：47% → **80.18%**
- 覆盖对象：FeatureProcessor、GeneralProcessor、QualityAssessor、FeatureStore、Volatility/Technical 指标、CacheStore 等关键组件
- 全部测试通过率：100%

**下一步**:
- 将上述命令纳入 CI，并产出 `coverage.xml`（守住 80% 阈值）
- Phase 2 完成后补充 `test_feature_engineer.py`、分布式处理管线等高级场景，进一步提升到 85%+

### Phase 4: 模型推理层测试 (新增)

#### 4.1 模型训练测试

**目标**: 95%+ 覆盖率
**时间**: 2025-03-05 ~ 2025-03-15

**测试内容**:
```python
class TestModelTraining:
    def test_model_initialization(self):
        """测试模型初始化"""

    def test_model_training_process(self):
        """测试模型训练过程"""

    def test_model_evaluation(self):
        """测试模型评估"""

    def test_model_persistence(self):
        """测试模型持久化"""
```

#### 4.2 模型推理测试

**目标**: 100% 覆盖率
**时间**: 2025-03-15 ~ 2025-03-25

**测试模块清单**:
- `test_inference_manager.py` - 推理管理器测试
- `test_gpu_inference_engine.py` - GPU推理引擎测试
- `test_batch_inference_processor.py` - 批量推理处理器测试

### Phase 5: 策略决策层测试 (新增)

#### 5.1 策略逻辑测试

**目标**: 90%+ 覆盖率
**时间**: 2025-03-25 ~ 2025-04-05

**测试内容**:
```python
class TestStrategyDecision:
    def test_signal_generation(self):
        """测试信号生成"""

    def test_strategy_parameter_optimization(self):
        """测试策略参数优化"""

    def test_risk_reward_calculation(self):
        """测试风险收益计算"""

    def test_market_condition_adaptation(self):
        """测试市场条件适应"""
```

#### 5.2 策略性能测试

**目标**: 85%+ 覆盖率
**时间**: 2025-04-05 ~ 2025-04-15

### Phase 6: 风控合规层测试 (新增)

#### 6.1 风险检查测试

**目标**: 100% 覆盖率
**时间**: 2025-04-15 ~ 2025-04-25

**测试内容**:
```python
class TestRiskManagement:
    def test_position_risk_calculation(self):
        """测试持仓风险计算"""

    def test_market_risk_assessment(self):
        """测试市场风险评估"""

    def test_compliance_rule_validation(self):
        """测试合规规则验证"""

    def test_real_time_risk_monitoring(self):
        """测试实时风险监控"""
```

### Phase 7: 交易执行层测试 (新增)

#### 7.1 订单管理测试

**目标**: 95%+ 覆盖率
**时间**: 2025-04-25 ~ 2025-05-05

**测试内容**:
```python
class TestOrderManagement:
    def test_order_creation_validation(self):
        """测试订单创建验证"""

    def test_order_routing_logic(self):
        """测试订单路由逻辑"""

    def test_order_execution_tracking(self):
        """测试订单执行跟踪"""

    def test_order_cancellation_handling(self):
        """测试订单取消处理"""
```

#### 7.2 执行引擎测试

**目标**: 90%+ 覆盖率
**时间**: 2025-05-05 ~ 2025-05-15

### Phase 8: 监控反馈层测试 (新增)

#### 8.1 性能监控测试

**目标**: 100% 覆盖率
**时间**: 2025-05-15 ~ 2025-05-25

**测试内容**:
```python
class TestPerformanceMonitoring:
    def test_system_resource_monitoring(self):
        """测试系统资源监控"""

    def test_application_performance_metrics(self):
        """测试应用性能指标"""

    def test_business_kpi_tracking(self):
        """测试业务KPI跟踪"""

    def test_alert_system_integration(self):
        """测试告警系统集成"""
```

## 🛠️ 测试环境配置

### 层次化测试环境

#### 核心服务层测试环境
```python
@pytest.fixture(scope="session")
def core_services_test_env():
    """核心服务层测试环境"""
    event_bus = EventBus()
    container = DependencyContainer()
    service_container = ServiceContainer()

    yield {
        'event_bus': event_bus,
        'container': container,
        'service_container': service_container
    }
```

#### 基础设施层测试环境
```python
@pytest.fixture(scope="session")
def infrastructure_test_env():
    """基础设施层测试环境"""
    cache_manager = SmartCacheManager()
    config_manager = UnifiedConfigManager()
    health_checker = EnhancedHealthChecker()

    yield {
        'cache_manager': cache_manager,
        'config_manager': config_manager,
        'health_checker': health_checker
    }
```

#### 业务层测试环境
```python
@pytest.fixture(scope="session")
def business_layer_test_env():
    """业务层测试环境"""
    # 特征处理环境
    feature_engine = FeatureEngineer()

    # 模型推理环境
    model_manager = ModelManager()

    # 交易执行环境
    order_manager = OrderManager()

    yield {
        'feature_engine': feature_engine,
        'model_manager': model_manager,
        'order_manager': order_manager
    }
```

## 📊 测试数据管理

### 层次化测试数据

#### 核心服务层测试数据
```python
class CoreServicesTestDataFactory:
    """核心服务层测试数据工厂"""

    @staticmethod
    def create_test_event():
        """创建测试事件"""
        return Event(
            event_type=EventType.DATA_COLLECTION_STARTED,
            data={'source': 'test', 'timestamp': '2025-01-27'},
            timestamp=time.time()
        )

    @staticmethod
    def create_test_service():
        """创建测试服务"""
        return TestService()
```

#### 业务层测试数据
```python
class BusinessTestDataFactory:
    """业务层测试数据工厂"""

    @staticmethod
    def create_test_market_data():
        """创建测试市场数据"""
        return {
            'symbol': 'AAPL',
            'price': 150.0,
            'volume': 1000,
            'timestamp': '2025-01-27T10:00:00Z'
        }

    @staticmethod
    def create_test_trading_order():
        """创建测试交易订单"""
        return {
            'user_id': 1,
            'symbol': 'AAPL',
            'quantity': 100,
            'order_type': 'market',
            'side': 'buy'
        }
```

## 🔍 测试执行策略

### 分层测试执行

#### 策略说明
1. **自下而上**: 核心服务层 → 基础设施层 → 业务层
2. **依赖顺序**: 先执行低层测试，再执行高层测试
3. **并行执行**: 相同层次的独立模块可并行测试
4. **增量验证**: 逐步增加测试覆盖范围

#### 执行顺序
```
Phase 1: 核心服务层测试 (✅ 已完成)
Phase 2: 基础设施层测试 (🔄 重点突破)
Phase 3: 数据管理层测试 (🔄 进行中：批次二 `api`/`edge`/`sources` 收尾 → `edge`/`sources` 核心路径已补测；批次三 `processing.data_processor` 99% ✅、`validation.validator` ✅(100%)、`quality.data_quality_monitor` 92% ✅、`quality.unified_quality_monitor` 85% ✅、`quality.quality_components` 97% ✅、`quality.monitor_components` 99% ✅、`quality.validator` 94% ✅；批次四 `distributed.cluster_manager` ✅(100%)、`distributed.sharding_manager` ✅(100%)、`distributed.distributed_data_loader` 边界补测完成（>91%）；批次五 `validation.validator` ✅(100%)；批次六 缓存/同步子域提升：`cache.cache_manager` ↑（TTL/驱逐/统计、策略 on_get/on_set、异常落盘/恢复补测完成）、`cache.smart_cache_optimizer` ↑（新鲜度失效、访问模式失效、预加载规则补测完成）；新增：`preload.preloader` ✅、`export.data_exporter` ✅、`transformers.*` ✅、`assurance_components` ✅、`validator_components` ✅；持续清理 term-missing 零星文件，稳定≥80% 目标；本批次分域并行结果：`interfaces` ≈75%（`api.py` 合同分支仍有 term-missing，已补充 FastAPI 契约与失败路径用例）、`monitoring` ≈55%（核心 `performance_monitor`=100%；`data_alert_rules` 已通过并行小修复稳定 129/129，通过率100%）、`quality`=88%（`unified_quality_monitor.py` 85% ✅，大并行稳定）、`sources` ≈78%（`intelligent_source_manager` 并行导入兜底已加，契约用例通过）、`cache` ≈84%（`cache_manager.py`≈84%、`multi_level_cache.py`≈85%、`smart_cache_optimizer.py`≈91%，其余组件高覆盖）、`distributed` ≈96%（`cluster_manager.py`/`sharding_manager.py`/`multiprocess_loader.py` 100%、`distributed_data_loader.py`≈94%)、`governance` ≈94%（`enterprise_governance.py`≈94%）、`export` ≈85%（`data_exporter.py`≈85%）、`transformers` ≈87%（`data_transformer.py`≈87%)) 

小结（残留清单与后续批次）：
- 残留 term-missing（高价值优先）：
  - `src\data\interfaces\api.py`：就绪探针的非就绪路径、`/store` 异常/fallback 分支仍有少量未命中行。
  - `src\data\monitoring\dashboard.py`：JSON 导出与回调出错路径的个别分支。
  - `src\data\monitoring\data_alert_rules.py`：变化率/区间边界与 JSON 导入异常分支仍有少量行。
- 并行稳定性说明：
  - 为避免多进程下的接口注入抖动，已对 `IDataValidator`、`IDataLoader` 引入轻量兜底；`data_alert_rules` 的 data_types 解析分支采用“非空占位”测试策略，已在 16 workers 下稳定通过（129/129）。
- 下一批建议（小批收口，预期+3%～+6%）：
  - interfaces：补 `/ready` 非就绪分支的细粒度断言与 `/store` 异常日志校验。
  - monitoring：补 `dashboard` 导出/回调异常用例；`data_alert_rules` 变化率与 JSON 导入错误用例。
  - 文档与附件：合并分域 XML（`test_logs/coverage-*-latest.xml`）与 term-missing 摘要，形成投产评审包（产物：`test_logs/data-layer-review-package.zip`；可一键执行 `scripts/build_data_layer_review.ps1` 生成）。
  - 覆盖合并策略：各分域执行统一启用 `--cov-branch`（已在 `scripts/ci/pytest_data_layer_small_batches.ps1` 中固化），避免跨批次合并时报错 “Can't combine branch coverage data with statement data”。
Phase 4: 特征处理层测试 (⏳ 待开始)
Phase 5: 模型推理层测试 (⏳ 待开始)
Phase 6: 策略决策层测试 (⏳ 待开始)
Phase 7: 风控合规层测试 (⏳ 待开始)
Phase 8: 交易执行层测试 (⏳ 待开始)
Phase 9: 监控反馈层测试 (⏳ 待开始)
```

## 📈 里程碑与时间表

### 关键里程碑

|| 里程碑 | 时间 | 目标 | 验证标准 |
||---------|------|------|----------|
|| M1 | 2025-02-05 | 基础设施层覆盖率 ≥80% | 覆盖率报告 |
|| M2 | 2025-03-05 | 特征处理层覆盖率 ≥90% | 覆盖率报告 |
|| M3 | 2025-04-05 | 模型推理层覆盖率 ≥90% | 覆盖率报告 |
|| M4 | 2025-05-05 | 策略决策层覆盖率 ≥85% | 覆盖率报告 |
|| M5 | 2025-06-05 | 风控合规层覆盖率 ≥90% | 覆盖率报告 |
|| M6 | 2025-07-05 | 交易执行层覆盖率 ≥85% | 覆盖率报告 |
|| M7 | 2025-08-05 | 监控反馈层覆盖率 ≥90% | 覆盖率报告 |
|| M8 | 2025-09-05 | 整体覆盖率 ≥90% | 最终验证报告 |

## 🎯 成功标准

### 技术成功标准
1. **代码质量**
   - 分层覆盖率: 各层≥85%
   - 单元测试通过率: ≥99%
   - 集成测试通过率: ≥98%

2. **系统性能**
   - 测试执行时间: <30分钟
   - 资源使用率: <80%
   - 系统稳定性: 100%

3. **测试质量**
   - 测试用例有效性: 100%
   - 测试覆盖完整性: ≥90%
   - 测试维护性: 良好

### 业务成功标准
1. **功能完整性**
   - 核心业务流程: 100%覆盖
   - 异常场景处理: 100%覆盖
   - 边界条件测试: 100%覆盖

2. **系统可靠性**
   - 故障发现率: >95%
   - 缺陷修复率: 100%
   - 回归测试覆盖: 100%

## 🚀 实施计划

### 实施步骤

#### 步骤1: 基础设施层突破 (2周)
1. 分析当前覆盖率缺口
2. 制定详细测试计划
3. 开发缺失的测试用例
4. 执行测试并优化

#### 步骤2: 业务层测试建设 (6周)
1. 设计层次化测试架构
2. 开发各层测试用例
3. 建立测试数据工厂
4. 执行测试验证

#### 步骤3: 集成测试优化 (2周)
1. 设计层间集成测试
2. 验证系统整体功能
3. 优化测试执行效率
4. 完善测试报告

#### 步骤4: 持续优化 (2周)
1. 分析测试覆盖效果
2. 优化测试用例质量
3. 建立持续集成机制
4. 制定后续改进计划

### 资源需求

#### 人力资源
- **测试工程师**: 8人
  - 基础设施层: 2人
  - 业务层: 4人
  - 集成测试: 2人
- **开发工程师**: 3人 (支持)
- **项目经理**: 1人 (协调)

#### 工具资源
- **测试框架**: pytest + 相关插件
- **覆盖率工具**: coverage.py
- **Mock工具**: unittest.mock
- **性能工具**: pytest-benchmark

## 📋 总结

本层次化单元测试策略为RQA2025项目制定了完整的分层测试体系：

### 核心策略
1. **层次化测试架构** - 按业务架构层次组织测试
2. **重点突破关键层** - 优先解决基础设施层瓶颈
3. **分阶段实施** - 逐步提升各层测试覆盖率
4. **质量门禁** - 严格的覆盖率和质量要求

### 实施重点
1. **基础设施层突破** - 从59.82%提升到95%+
2. **业务层测试建设** - 为特征处理、模型推理等层建立测试
3. **测试自动化** - 建立完整的测试自动化体系
4. **持续改进** - 通过持续集成优化测试质量

### 预期成果
- **分层覆盖率**: 各层≥85%
- **整体覆盖率**: 90%+
- **测试效率**: 3倍提升
- **缺陷发现率**: 95%+

通过本策略的实施，RQA2025项目将建立完善的层次化测试体系，确保各架构层次的功能完整性和系统稳定性。

---

**文档维护**: 测试组
**最后更新**: 2025-01-27
**下次更新**: 2025-02-03
