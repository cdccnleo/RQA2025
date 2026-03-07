# RQA2025 测试执行指南

## 📋 分层测试执行策略

根据系统架构和测试稳定性要求，采用分层测试执行策略：

### 🏗️ 基础设施层 (Infrastructure Layer)
- **执行模式**: 单线程执行
- **原因**: 避免模块导入冲突和并发竞争
- **覆盖率要求**: ≥95%
- **配置**: `pytest.infrastructure.ini`

### 🏭 其他层级 (Core/Data/ML/Feature/Strategy/Trading/Risk等)
- **执行模式**: 并行执行
- **原因**: 提升测试执行效率
- **覆盖率要求**: ≥70%
- **配置**: `pytest.ini`

## 🚀 执行方式

### 方式1: 分层执行脚本 (推荐)

```bash
# 执行所有层级的分层测试
python scripts/run_infrastructure_tests.py
```

此脚本会：
1. 使用单线程模式执行基础设施层测试
2. 使用并行模式执行其他层级测试
3. 自动生成分层覆盖率报告

### 方式2: 手动分层执行

#### 基础设施层测试 (单线程)
```bash
# 使用专用配置文件执行基础设施层测试
pytest --config-file=pytest.infrastructure.ini tests/unit/infrastructure/
```

#### 其他层级测试 (并行)
```bash
# 使用默认配置文件执行其他层级测试
pytest -n=auto --dist=loadscope tests/unit/core/ tests/unit/data/ tests/unit/ml/ ...
```

### 方式3: 按需执行特定层级

#### 核心服务层
```bash
pytest -n=auto tests/unit/core/
```

#### 数据管理层
```bash
pytest -n=auto tests/unit/data/
```

#### 机器学习层
```bash
pytest -n=auto tests/unit/ml/
```

#### 策略层
```bash
pytest -n=auto tests/unit/strategy/
```

#### 交易层
```bash
pytest -n=auto tests/unit/trading/
```

#### 风险控制层
```bash
pytest -n=auto tests/unit/risk/
```

#### 特征层
```bash
pytest -n=auto tests/unit/feature/
```

## 📊 覆盖率报告

### 基础设施层覆盖率报告
- 位置: `test_logs/infrastructure_coverage/`
- 格式: HTML + XML
- 要求: ≥95%

### 整体覆盖率报告
- 位置: `test_logs/coverage_reports/`
- 格式: HTML + XML
- 要求: ≥70%

## ⚙️ 配置说明

### pytest.infrastructure.ini (基础设施层专用)
```ini
[pytest]
# 单线程执行配置
addopts =
    --cov=src.infrastructure
    --cov-report=term-missing
    --cov-fail-under=95
    # 禁用并行: 无 -n 参数
```

### pytest.ini (其他层级默认)
```ini
[pytest]
# 并行执行配置
addopts =
    --cov=src
    -n=auto
    --dist=loadscope
    --cov-fail-under=70
```

## 🔧 故障排除

### 基础设施层测试失败
1. 检查模块导入路径
2. 确认单线程执行环境
3. 查看详细错误日志

### 并行测试冲突
1. 减少并行进程数: `pytest -n=2`
2. 检查共享资源竞争
3. 隔离测试环境

### 覆盖率报告缺失
1. 确认pytest-cov已安装
2. 检查覆盖率配置文件 `.coveragerc`
3. 验证源码路径配置

## 📈 性能优化建议

### 基础设施层
- 单线程执行确保稳定性
- 关注模块初始化顺序
- 避免并发资源竞争

### 其他层级
- 并行执行提升效率
- 合理设置进程数
- 优化测试依赖关系

## 🎯 最佳实践

1. **基础设施层优先**: 先确保基础设施层测试通过
2. **分层验证**: 按依赖关系顺序执行测试
3. **覆盖率监控**: 持续监控各层覆盖率指标
4. **环境隔离**: 不同层级使用独立测试环境
5. **CI/CD集成**: 在CI流水线中集成分层测试策略

