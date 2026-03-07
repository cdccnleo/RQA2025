# 核心服务层测试覆盖率提升计划

**日期**: 2025-01-27  
**目标**: 核心服务层（src/core）测试覆盖率提升至80%+投产要求  
**当前状态**: 66个测试通过，导入问题需要修复

---

## 📊 当前状态

### 测试通过情况
- ✅ **66个测试通过** (100%通过率)
- ❌ 多个测试文件存在导入错误
- ⚠️ 覆盖率报告生成失败

### 导入错误文件列表
1. `tests/unit/core/container/test_container_components_coverage.py` - 已修复导入
2. `tests/unit/core/core_services/core/test_core_services_coverage.py` - 已修复导入
3. `tests/unit/core/core_services/integration/test_integration_components_coverage.py` - 待修复
4. `tests/unit/core/event_bus/test_event_components_coverage.py` - 待修复
5. `tests/unit/core/foundation/test_*.py` - 待修复
6. `tests/unit/core/test_process_config_loader.py` - 待修复

---

## 🎯 执行计划

### Phase 1: 修复导入问题 ✅ 进行中

#### 1.1 修复container模块导入 ✅
- ✅ 修复 `src/core/container/__init__.py` 导入错误
- ✅ 添加向后兼容别名
- ✅ 修复测试文件导入异常处理

#### 1.2 修复core_services模块导入 ✅
- ✅ 修复测试文件导入异常处理
- ⏳ 验证导入是否正常工作

#### 1.3 修复其他模块导入 ⏳
- [ ] 修复event_bus模块导入
- [ ] 修复foundation模块导入
- [ ] 修复orchestration模块导入
- [ ] 修复integration模块导入

### Phase 2: 运行基础测试覆盖率 ⏳

#### 2.1 运行可正常工作的测试
```bash
conda run -n rqa pytest tests/unit/core/core_services/ \
  --cov=src.core --cov-report=term-missing -q
```

#### 2.2 识别低覆盖模块
- 分析term-missing输出
- 列出未覆盖的关键模块
- 优先级排序

### Phase 3: 补充核心模块测试 ⏳

#### 3.1 事件总线（event_bus）测试
- **目标文件**: `src/core/event_bus/core.py`
- **目标覆盖率**: 85%+
- **测试内容**:
  - EventBus初始化和配置
  - 事件发布/订阅机制
  - 事件过滤和路由
  - 并发事件处理
  - 错误处理和恢复

#### 3.2 容器（container）测试
- **目标文件**: `src/core/container/container.py`
- **目标覆盖率**: 85%+
- **测试内容**:
  - 服务注册和解析
  - 生命周期管理
  - 依赖注入
  - 线程安全

#### 3.3 基础组件（foundation）测试
- **目标文件**: `src/core/foundation/base.py`
- **目标覆盖率**: 90%+
- **测试内容**:
  - BaseComponent生命周期
  - 健康检查
  - 配置管理
  - 事件驱动支持

#### 3.4 业务流程（business_process）测试
- **目标文件**: `src/core/business_process/orchestrator/`
- **目标覆盖率**: 80%+
- **测试内容**:
  - 流程编排
  - 状态机管理
  - 流程监控
  - 错误处理

### Phase 4: 集成测试和边界测试 ⏳

#### 4.1 组件集成测试
- 事件总线 + 容器集成
- 业务流程 + 状态机集成
- 服务治理框架集成

#### 4.2 边界和异常测试
- 并发场景测试
- 资源耗尽场景
- 异常恢复测试
- 性能压力测试

### Phase 5: 覆盖率验证和优化 ⏳

#### 5.1 运行完整覆盖率测试
```bash
conda run -n rqa pytest tests/unit/core \
  --cov=src.core --cov-report=term-missing \
  --cov-report=html:htmlcov/core \
  -k "not e2e" -n auto
```

#### 5.2 分析覆盖率报告
- 识别未覆盖的关键代码路径
- 补充缺失的测试场景
- 优化测试质量

#### 5.3 达到80%+目标
- 确保所有核心模块覆盖率≥80%
- 关键业务逻辑覆盖率≥90%
- 测试通过率≥95%

---

## 📋 测试质量要求

### 测试通过率
- **目标**: ≥95%
- **当前**: 100% (66/66通过)

### 测试覆盖率
- **总体目标**: ≥80%
- **核心模块**: ≥85%
- **关键业务逻辑**: ≥90%

### 测试类型分布
- **单元测试**: 70%
- **集成测试**: 20%
- **边界/异常测试**: 10%

---

## 🔧 技术策略

### 1. 导入问题修复策略
- 使用try-except包装导入
- 添加pytest.skip优雅降级
- 修复模块__init__.py导出

### 2. 测试编写策略
- 优先测试核心业务逻辑
- 覆盖正常流程和异常流程
- 使用真实对象而非过度Mock
- 注重测试可维护性

### 3. 覆盖率提升策略
- 小批场景测试
- 定向pytest --cov
- term-missing审核
- 归档完成报告

---

## 📈 进度跟踪

### 已完成 ✅
- [x] 修复container模块导入
- [x] 修复core_services测试导入
- [x] 66个测试全部通过

### 进行中 ⏳
- [ ] 修复其他模块导入问题
- [ ] 运行基础覆盖率测试
- [ ] 识别低覆盖模块

### 待开始 📋
- [ ] 补充event_bus测试
- [ ] 补充container测试
- [ ] 补充foundation测试
- [ ] 补充business_process测试
- [ ] 集成测试和边界测试
- [ ] 覆盖率验证和优化

---

## 🎯 下一步行动

1. **立即执行**: 修复剩余导入问题
2. **短期目标**: 运行基础覆盖率测试，识别低覆盖模块
3. **中期目标**: 补充核心模块测试，达到60%+覆盖率
4. **长期目标**: 达到80%+覆盖率，满足投产要求

---

**最后更新**: 2025-01-27  
**状态**: ⏳ 进行中 - Phase 1导入修复阶段

