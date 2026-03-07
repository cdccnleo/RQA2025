# RQA2025 测试框架使用指南
<!-- 
版本更新记录：
2024-03-28 v1.0.0 - 初始版本
-->

## 1. 快速开始

### 1.1 本地测试
```bash
# 运行全部单元测试
pytest tests/unit/ -v

# 运行特定模块测试
pytest tests/unit/infrastructure/config/ -m "not slow"

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 1.2 CI集成
```yaml
# .github/workflows/test.yml 示例
- name: Run unit tests
  run: |
    pytest tests/unit/ \
      --cov=src \
      --cov-report=xml \
      -n auto
```

## 2. 核心功能

### 2.1 分层测试执行
```mermaid
graph TD
    A[单元测试] -->|验证组件| B[集成测试]
    B -->|验证交互| C[性能测试]
    C -->|基准数据| D[监控仪表盘]
```

### 2.2 测试标记系统
```ini
# pytest.ini 配置示例
[pytest]
markers =
    unit: 单元测试
    integration: 集成测试
    performance: 性能测试
    slow: 长时间运行测试
```

### 2.3 测试数据管理
```python
# 使用fixture示例
@pytest.fixture
def sample_config():
    return {"db": {"host": "test.db"}}

def test_config_load(sample_config):
    loader = ConfigLoader()
    assert loader.validate(sample_config)
```

## 3. 高级用法

### 3.1 性能基准测试
```python
# 基准测试示例
def test_cache_performance(benchmark):
    cache = ThreadSafeTTLCache()
    benchmark(cache.bulk_set, {"key": "value"})
```

### 3.2 异常测试
```python
# 测试异常场景
def test_invalid_config():
    with pytest.raises(ConfigError):
        load_config("invalid.json")
```

### 3.3 CI集成进阶
```yaml
# 并行测试配置
jobs:
  unit-tests:
    strategy:
      matrix:
        python: ["3.8", "3.9", "3.10"]
      fail-fast: true
```

## 4. 最佳实践

### 4.1 测试结构规范
```
tests/
├── unit/               # 单元测试
│   └── infrastructure/ # 基础设施测试
├── integration/        # 集成测试  
└── performance/        # 性能测试
```

### 4.2 测试编写原则
1. **原子性**：每个测试只验证一个功能点
2. **独立性**：测试之间不依赖执行顺序
3. **可重复**：每次运行结果一致
4. **文档化**：测试名称即文档

### 4.3 调试技巧
```bash
# 失败时进入PDB调试
pytest --pdb

# 输出详细日志
pytest -v --log-level=DEBUG

# 只运行上次失败的测试
pytest --last-failed
```

## 5. 与监控系统集成

### 5.1 测试指标暴露
```python
# 测试指标收集示例
def test_error_handling_metrics():
    handler = ErrorHandler()
    handler.process(sample_error)
    assert handler.metrics["processed"] == 1
```

### 5.2 Prometheus监控
```yaml
# prometheus配置示例
scrape_configs:
  - job_name: 'test_metrics'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['localhost:8000']
```

## 6. 故障排查

| 问题现象 | 排查步骤 | 解决方案 |
|---------|----------|----------|
| 测试超时 | 1. 检查资源使用<br>2. 分析测试依赖 | 增加超时阈值或优化测试 |
| 覆盖率下降 | 1. 对比差异版本<br>2. 检查新增代码 | 补充缺失测试用例 |
| 性能回退 | 1. 对比历史基准<br>2. 分析资源变化 | 优化热点代码路径 |

## 7. 更新与维护

```mermaid
graph LR
    A[代码变更] --> B{影响测试?}
    B -->|是| C[更新测试]
    B -->|否| D[验证通过]
    C --> E[版本记录]
```

## 无人值守自动化测试推荐方案（run_tests.py）

本项目推荐使用 `run_tests.py` 脚本实现无人值守、自动切换conda环境的批量测试，适用于本地开发、CI/CD流水线、定时任务等场景。

### 脚本位置
- 根目录下：`run_tests.py`

### 脚本功能
- 自动调用 `conda run` 切换到指定conda环境
- 支持传入任意pytest测试路径和参数
- 支持所有pytest原生参数
- 适用于Windows/Linux/Mac

### 使用方法

1. **基本用法**

```bash
python run_tests.py --env rqa --path tests/unit/features/ --pytest-args -v --tb=short --capture=no --maxfail=5
```
- `--env rqa`：指定conda环境名
- `--path tests/unit/features/`：指定pytest测试路径
- `--pytest-args ...`：后续所有参数均传递给pytest

2. **批量测试不同层**

```bash
python run_tests.py --env rqa --path tests/unit/data/ --pytest-args -v --tb=short
python run_tests.py --env rqa --path tests/unit/trading/ --pytest-args -v --tb=short
```

3. **集成到CI/CD或定时任务**
- 只需在CI脚本或计划任务中调用上述命令即可，无需人工干预。

### 参数说明
| 参数           | 说明                       |
|----------------|----------------------------|
| --env          | conda环境名                |
| --path         | pytest测试路径             |
| --pytest-args  | pytest原生参数（可选多个） |

### 典型输出
- pytest所有输出、报错、失败详情均会实时显示在终端或日志中。

### 注意事项
- 脚本无需手动激活conda环境，适合无人值守自动化。
- 如需生成HTML报告、失败重试等功能，可在`--pytest-args`中添加相关参数。

---

## 后续建议与自动化推进
1. 持续修复测试用例中的具体报错（如mock断言、fixture未定义等），提升通过率。
2. 可用循环或CI流水线批量调用该脚本，实现多层/多环境自动化。
3. 如需自动生成测试报告、失败重试、邮件通知等功能，可进一步扩展脚本。
4. 推荐将该脚本作为团队/CI的标准测试入口，保证一致性和可维护性。
