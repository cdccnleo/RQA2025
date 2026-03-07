# 测试覆盖率规范与最佳实践

## 概述

本文档定义了RQA2025项目的测试覆盖率统计规范，确保测试效率和质量。

## pytest-cov 工作原理

### coverage.py 加载机制

pytest-cov 底层调用 coverage.py，具有以下特点：

1. **一次性加载**: 启动时一次性将 `--cov` 指定的目录（或模块）全部加载到监控范围
2. **全局监控**: 无论在哪目录执行 pytest，都会统计这些目录的所有行
3. **非选择性统计**: 不是"只统计当前目录下被测试文件触发的行"

### 性能影响

- 全局覆盖率统计会显著降低测试执行速度
- 增加内存占用和CPU使用率
- 影响CI/CD流水线效率

## 测试覆盖率规范

### 1. 分层测试覆盖率统计

#### 数据层测试
```bash
# 正确：只统计数据层覆盖率
pytest tests/unit/data/ --cov=src/data --cov-report=html --cov-report=term-missing

# 错误：统计全局覆盖率
pytest tests/unit/data/ --cov=src --cov-report=html
```

#### 交易层测试
```bash
# 正确：只统计交易层覆盖率
pytest tests/unit/trading/ --cov=src/trading --cov-report=html --cov-report=term-missing

# 错误：统计全局覆盖率
pytest tests/unit/trading/ --cov=src --cov-report=html
```

#### 特征层测试
```bash
# 正确：只统计特征层覆盖率
pytest tests/unit/features/ --cov=src/features --cov-report=html --cov-report=term-missing

# 错误：统计全局覆盖率
pytest tests/unit/features/ --cov=src --cov-report=html
```

### 2. 模块级测试覆盖率统计

#### 特定模块测试
```bash
# 正确：只统计特定模块覆盖率
pytest tests/unit/features/test_feature_manager.py --cov=src/features/feature_manager.py --cov-report=term-missing

# 错误：统计整个特征层覆盖率
pytest tests/unit/features/test_feature_manager.py --cov=src/features --cov-report=term-missing
```

#### 集成测试覆盖率
```bash
# 正确：统计集成测试相关模块
pytest tests/integration/ --cov=src/trading --cov=src/backtest --cov-report=html --cov-report=term-missing

# 错误：统计所有模块
pytest tests/integration/ --cov=src --cov-report=html
```

### 3. 覆盖率报告规范

#### 报告格式
- **HTML报告**: 用于详细分析和可视化
- **终端报告**: 用于快速查看覆盖率
- **XML报告**: 用于CI/CD集成

```bash
# 标准覆盖率报告命令
pytest [test_path] --cov=[target_module] --cov-report=html --cov-report=term-missing --cov-report=xml
```

#### 报告命名规范
```bash
# 分层报告命名
pytest tests/unit/data/ --cov=src/data --cov-report=html:coverage_reports/data_layer_coverage.html

# 模块报告命名
pytest tests/unit/features/ --cov=src/features --cov-report=html:coverage_reports/features_layer_coverage.html
```

## 最佳实践

### 1. 测试执行策略

#### 开发阶段
```bash
# 快速测试，不统计覆盖率
pytest tests/unit/features/ -v

# 需要覆盖率时，只统计相关模块
pytest tests/unit/features/ --cov=src/features --cov-report=term-missing
```

#### CI/CD阶段
```bash
# 完整测试，分层统计覆盖率
pytest tests/unit/data/ --cov=src/data --cov-report=html:coverage_reports/data_layer.html --cov-report=term-missing
pytest tests/unit/trading/ --cov=src/trading --cov-report=html:coverage_reports/trading_layer.html --cov-report=term-missing
pytest tests/unit/features/ --cov=src/features --cov-report=html:coverage_reports/features_layer.html --cov-report=term-missing
```

### 2. 覆盖率阈值设置

#### 分层覆盖率目标
- **数据层**: ≥80%
- **交易层**: ≥75%
- **特征层**: ≥70%
- **基础设施层**: ≥85%
- **集成测试**: ≥60%

#### 覆盖率检查
```bash
# 检查覆盖率是否达标
pytest tests/unit/features/ --cov=src/features --cov-fail-under=70 --cov-report=term-missing
```

### 3. 性能优化

#### 并行测试
```bash
# 使用多进程加速测试
pytest tests/unit/features/ --cov=src/features -n auto --cov-report=term-missing
```

#### 缓存优化
```bash
# 使用pytest缓存
pytest tests/unit/features/ --cov=src/features --cache-clear --cov-report=term-missing
```

## 常见错误与解决方案

### 1. 全局覆盖率统计错误

**错误示例**:
```bash
pytest tests/unit/features/ --cov=src --cov-report=html
```

**正确做法**:
```bash
pytest tests/unit/features/ --cov=src/features --cov-report=html
```

### 2. 覆盖率报告路径错误

**错误示例**:
```bash
pytest tests/unit/features/ --cov=src/features --cov-report=html:reports/coverage.html
```

**正确做法**:
```bash
pytest tests/unit/features/ --cov=src/features --cov-report=html:coverage_reports/features_layer.html
```

### 3. 测试数据准备

**错误示例**:
```python
# 测试数据不完整
test_data = pd.DataFrame({'price': [100, 101, 102]})
```

**正确做法**:
```python
# 完整的测试数据
test_data = pd.DataFrame({
    'price': [100, 101, 102],
    'volume': [1000, 1100, 1200],
    'timestamp': pd.date_range('2023-01-01', periods=3, freq='D')
})
```

## 覆盖率分析工具

### 1. 覆盖率报告分析

#### HTML报告分析
- 查看未覆盖行
- 分析分支覆盖率
- 识别测试盲点

#### 命令行分析
```bash
# 查看覆盖率摘要
pytest tests/unit/features/ --cov=src/features --cov-report=term-missing

# 查看详细覆盖率
pytest tests/unit/features/ --cov=src/features --cov-report=term-missing --cov-report=html
```

### 2. 覆盖率趋势分析

#### 历史对比
```bash
# 生成历史覆盖率报告
pytest tests/unit/features/ --cov=src/features --cov-report=html:coverage_reports/features_layer_$(date +%Y%m%d).html
```

#### 覆盖率监控
```bash
# 设置覆盖率阈值
pytest tests/unit/features/ --cov=src/features --cov-fail-under=70 --cov-report=term-missing
```

## 总结

1. **精确指定覆盖率目标**: 只统计相关模块，避免全局统计
2. **分层测试策略**: 按功能模块分别统计覆盖率
3. **性能优化**: 使用并行测试和缓存机制
4. **质量保证**: 设置合理的覆盖率阈值
5. **持续监控**: 定期生成覆盖率报告并分析趋势

遵循这些规范可以显著提高测试效率，确保覆盖率统计的准确性和实用性。 