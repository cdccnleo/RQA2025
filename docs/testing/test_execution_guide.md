# 测试执行指南

## 本地开发环境

### 单元测试
```bash
pytest tests/unit/ -v
```

### 集成测试
```bash
pytest tests/integration/ -v
```

### 端到端测试
```bash
pytest tests/e2e/ -v
```

### 性能测试
```bash
pytest tests/performance/ -v --durations=0
```

### 带覆盖率测试
```bash
pytest --cov=src --cov-report=html
```

## CI/CD环境

### Full Test Suite
```bash
pytest -n auto --cov=src --cov-report=xml
```

### Smoke Tests
```bash
pytest tests/unit/ -k 'smoke' -x
```

### Critical Path
```bash
pytest tests/e2e/ -k 'critical' -x
```

## 调试技巧

### Single Test
```bash
pytest tests/unit/test_example.py::TestExample::test_method -v -s
```

### Failed Tests
```bash
pytest --lf
```

### New Tests
```bash
pytest --nf
```

### Slow Tests
```bash
pytest --durations=10
```

