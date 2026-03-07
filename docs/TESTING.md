# RQA2025 测试文档

## 概述

RQA2025 采用全面的测试策略，确保系统质量和稳定性。测试覆盖从单元测试到端到端测试的完整测试金字塔。

## 测试策略

### 测试金字塔

```
     ┌─────────────┐ 少量
     │ E2E测试     │
     │ (端到端)    │
     └─────────────┘
           │
     ┌─────────────┐ 中等
     │ 集成测试    │
     └─────────────┘
           │
     ┌─────────────┐ 大量
     │ 单元测试    │
     └─────────────┘
```

### 测试类型

#### 1. 单元测试 (Unit Tests)

**目标**: 验证单个函数、方法或类的正确性

**覆盖范围**:
- 策略算法实现
- 数据处理函数
- 业务逻辑组件
- 工具函数

**示例**:
```python
# tests/unit/test_momentum_strategy.py
import pytest
from src.strategy.strategies.momentum_strategy import MomentumStrategy

class TestMomentumStrategy:
    def test_signal_generation_buy_signal(self):
        """测试买入信号生成"""
        config = StrategyConfig(...)
        strategy = MomentumStrategy(config)

        market_data = {
            "000001.SZ": [
                {"close": 100.0},
                {"close": 102.0},
                {"close": 105.0}  # 上涨趋势
            ]
        }

        signals = strategy.generate_signals(market_data)

        assert len(signals) == 1
        assert signals[0].action == "BUY"
        assert signals[0].confidence > 0.5
```

#### 2. 集成测试 (Integration Tests)

**目标**: 验证组件间的交互和数据流

**覆盖范围**:
- 服务间通信
- 数据库操作
- 外部API调用
- 缓存机制

**示例**:
```python
# tests/integration/test_strategy_backtest_integration.py
import pytest
from src.strategy.core.strategy_service import UnifiedStrategyService
from src.strategy.backtest.backtest_service import BacktestService

class TestStrategyBacktestIntegration:
    @pytest.mark.asyncio
    async def test_create_and_backtest_strategy(self):
        """测试创建策略并执行回测的完整流程"""
        # 创建策略服务
        strategy_service = UnifiedStrategyService()

        # 创建回测服务
        backtest_service = BacktestService(
            strategy_service=strategy_service
        )

        # 创建策略
        strategy_config = StrategyConfig(...)
        await strategy_service.create_strategy(strategy_config)

        # 执行回测
        backtest_config = BacktestConfig(...)
        result = await backtest_service.run_backtest(backtest_config)

        # 验证结果
        assert result.status == BacktestStatus.COMPLETED
        assert "total_return" in result.metrics
        assert result.metrics["total_return"] is not None
```

#### 3. 端到端测试 (E2E Tests)

**目标**: 验证完整用户工作流

**覆盖范围**:
- 用户注册和登录
- 策略创建和配置
- 回测执行和分析
- 参数优化
- 结果可视化

**示例**:
```python
# tests/e2e/test_complete_workflow.py
import pytest
from tests.e2e.test_strategy_workflow import TestStrategyWorkflow

class TestCompleteWorkflow:
    @pytest.mark.asyncio
    async def test_user_to_result_workflow(self):
        """测试从用户注册到获得结果的完整工作流"""
        workflow = TestStrategyWorkflow()

        # 1. 用户注册和登录
        await workflow._test_user_registration_and_login()

        # 2. 策略创建
        strategy_id = await workflow._test_strategy_creation()

        # 3. 策略执行
        await workflow._test_strategy_execution(strategy_id)

        # 4. 回测分析
        await workflow._test_backtest_analysis(strategy_id)

        # 5. 参数优化
        await workflow._test_parameter_optimization(strategy_id)

        # 6. 系统监控
        await workflow._test_system_monitoring()

        # 7. Web API验证
        await workflow._test_web_api_integration(strategy_id)
```

#### 4. 性能测试 (Performance Tests)

**目标**: 验证系统性能指标

**测试指标**:
- 响应时间 (<100ms P95)
- 并发处理能力 (>1000 RPS)
- 内存使用 (<500MB)
- CPU使用率 (<80%)

**示例**:
```python
# tests/performance/benchmark_tests.py
import pytest
import time
from src.strategy.core.strategy_service import UnifiedStrategyService

class TestPerformanceBenchmarks:
    @pytest.mark.asyncio
    async def test_strategy_execution_performance(self):
        """测试策略执行性能"""
        strategy_service = UnifiedStrategyService()

        # 创建测试策略
        config = StrategyConfig(...)
        await strategy_service.create_strategy(config)

        # 准备测试数据
        market_data = generate_test_data(1000)

        # 执行性能测试
        start_time = time.time()
        result = await strategy_service.execute_strategy(config.strategy_id, market_data)
        execution_time = (time.time() - start_time) * 1000

        # 验证性能要求
        assert execution_time < 1000, f"执行时间过长: {execution_time}ms"
        assert result is not None
```

#### 5. 生产环境验证 (Production Validation)

**目标**: 验证生产环境部署的正确性

**验证内容**:
- 服务健康检查
- 配置正确性
- 数据库连接
- 网络连通性
- 安全设置

**示例**:
```python
# tests/production/production_validation.py
import pytest
import requests

class TestProductionValidation:
    def test_service_health(self):
        """测试服务健康状态"""
        response = requests.get("http://localhost:8000/health")
        assert response.status_code == 200

        health_data = response.json()
        assert health_data["status"] == "healthy"

    def test_api_endpoints(self):
        """测试API端点可用性"""
        endpoints = [
            "/api/strategies",
            "/api/backtests",
            "/api/monitoring/metrics"
        ]

        for endpoint in endpoints:
            response = requests.get(f"http://localhost:8000{endpoint}")
            # 对于受保护的端点，可能返回401，但服务应该是可用的
            assert response.status_code in [200, 401, 403]
```

## 测试框架和工具

### Pytest配置

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --disable-warnings
    --tb=short
    -ra
markers =
    slow: 标记耗时较长的测试
    integration: 集成测试标记
    e2e: 端到端测试标记
    performance: 性能测试标记
```

### 测试数据管理

```python
# tests/conftest.py
import pytest
from src.core.container import DependencyContainer

@pytest.fixture(scope="session")
def container():
    """测试容器fixture"""
    container = DependencyContainer()
    # 配置测试服务
    return container

@pytest.fixture
def sample_strategy_config():
    """示例策略配置fixture"""
    return {
        "strategy_id": "test_strategy",
        "strategy_name": "测试策略",
        "strategy_type": "momentum",
        "parameters": {
            "lookback_period": 20,
            "momentum_threshold": 0.05
        }
    }

@pytest.fixture
def sample_market_data():
    """示例市场数据fixture"""
    return {
        "000001.SZ": [
            {"timestamp": "2023-01-01", "close": 100.0, "volume": 1000000},
            {"timestamp": "2023-01-02", "close": 102.0, "volume": 1200000},
            {"timestamp": "2023-01-03", "close": 105.0, "volume": 1500000}
        ]
    }
```

### Mock和Stub

```python
# tests/unit/test_strategy_service.py
import pytest
from unittest.mock import Mock, patch
from src.strategy.core.strategy_service import UnifiedStrategyService

class TestUnifiedStrategyService:
    @patch('src.strategy.core.strategy_service.StrategyFactory')
    def test_create_strategy_success(self, mock_factory):
        """测试策略创建成功的情况"""
        # 创建mock对象
        mock_strategy = Mock()
        mock_factory.return_value.create_strategy_instance.return_value = mock_strategy

        service = UnifiedStrategyService()
        config = Mock()

        result = service.create_strategy(config)

        assert result is True
        mock_factory.return_value.create_strategy_instance.assert_called_once_with(config)

    @patch('src.strategy.core.strategy_service.StrategyFactory')
    def test_create_strategy_failure(self, mock_factory):
        """测试策略创建失败的情况"""
        mock_factory.return_value.create_strategy_instance.side_effect = Exception("创建失败")

        service = UnifiedStrategyService()
        config = Mock()

        result = service.create_strategy(config)

        assert result is False
```

## 测试执行

### 本地测试执行

```bash
# 运行所有测试
python scripts/run_tests.py

# 运行特定类型的测试
python scripts/run_tests.py --types unit integration

# 运行带覆盖率的测试
python scripts/run_tests.py --coverage

# 运行并行测试
python scripts/run_tests.py --parallel

# 详细输出
python scripts/run_tests.py --verbose
```

### CI/CD集成

```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: |
        python scripts/run_tests.py --coverage

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./htmlcov/coverage.xml
```

### 测试报告

#### HTML报告

```bash
# 生成HTML测试报告
pytest --html=test_report.html --self-contained-html

# 查看报告
open test_report.html
```

#### 覆盖率报告

```bash
# 生成覆盖率报告
pytest --cov=src --cov-report=html

# 查看HTML覆盖率报告
open htmlcov/index.html

# 终端覆盖率报告
pytest --cov=src --cov-report=term-missing
```

#### JUnit报告

```bash
# 生成JUnit XML报告
pytest --junitxml=test_results.xml

# 用于CI/CD系统集成
```

## 质量门禁

### 代码覆盖率

```python
# setup.cfg
[coverage:run]
source = src
omit =
    */tests/*
    */venv/*
    */__pycache__/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:

[coverage:html]
directory = htmlcov

[coverage:xml]
output = coverage.xml
```

### 质量检查

```yaml
# .pre-commit-config.yaml
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.4.0
  hooks:
  - id: trailing-whitespace
  - id: end-of-file-fixer
  - id: check-yaml
  - id: check-added-large-files

- repo: https://github.com/psf/black
  rev: 22.3.0
  hooks:
  - id: black
    language_version: python3

- repo: https://github.com/pycqa/isort
  rev: 5.10.1
  hooks:
  - id: isort
    args: ["--profile", "black"]

- repo: https://github.com/pycqa/flake8
  rev: 4.0.1
  hooks:
  - id: flake8
    args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v0.991
  hooks:
  - id: mypy
    additional_dependencies: [types-all]
```

### 性能基准

```python
# tests/performance/benchmarks.py
PERFORMANCE_THRESHOLDS = {
    "api_response_time_p95": 100,  # ms
    "strategy_execution_time": 1000,  # ms
    "memory_usage": 500 * 1024 * 1024,  # 500MB
    "cpu_usage_peak": 80,  # %
    "concurrent_users": 1000  # 用户数
}

def check_performance_thresholds(results):
    """检查性能阈值"""
    failures = []

    for metric, threshold in PERFORMANCE_THRESHOLDS.items():
        if metric in results:
            value = results[metric]
            if value > threshold:
                failures.append(f"{metric}: {value} > {threshold}")

    return failures
```

## 测试数据管理

### 测试数据生成

```python
# tests/data/test_data_generator.py
import random
import numpy as np
from datetime import datetime, timedelta

class TestDataGenerator:
    @staticmethod
    def generate_stock_data(symbol: str, days: int = 252,
                          start_price: float = 100.0) -> list:
        """生成股票测试数据"""
        dates = []
        prices = []
        volumes = []

        current_date = datetime.now() - timedelta(days=days)
        current_price = start_price

        for i in range(days):
            # 生成日期
            dates.append(current_date)

            # 生成价格 (随机游走)
            daily_return = random.gauss(0.0001, 0.02)  # 正态分布
            current_price *= (1 + daily_return)
            prices.append(round(current_price, 2))

            # 生成成交量
            base_volume = 1000000
            volume_noise = random.uniform(0.5, 1.5)
            volumes.append(int(base_volume * volume_noise))

            current_date += timedelta(days=1)

        return [
            {
                "date": dates[i].strftime("%Y-%m-%d"),
                "open": round(prices[i] * random.uniform(0.98, 1.02), 2),
                "high": round(prices[i] * random.uniform(1.00, 1.03), 2),
                "low": round(prices[i] * random.uniform(0.97, 1.00), 2),
                "close": prices[i],
                "volume": volumes[i]
            }
            for i in range(days)
        ]

    @staticmethod
    def generate_strategy_config(strategy_type: str = "momentum") -> dict:
        """生成策略配置"""
        base_configs = {
            "momentum": {
                "strategy_type": "momentum",
                "parameters": {
                    "lookback_period": random.randint(10, 50),
                    "momentum_threshold": random.uniform(0.01, 0.1),
                    "volume_threshold": random.uniform(1.0, 2.0)
                }
            },
            "mean_reversion": {
                "strategy_type": "mean_reversion",
                "parameters": {
                    "lookback_period": random.randint(10, 50),
                    "std_threshold": random.uniform(1.5, 3.0),
                    "profit_target": random.uniform(0.03, 0.08),
                    "stop_loss": random.uniform(-0.05, -0.01)
                }
            }
        }

        config = base_configs.get(strategy_type, base_configs["momentum"])
        config.update({
            "strategy_id": f"test_{strategy_type}_{random.randint(1000, 9999)}",
            "strategy_name": f"测试{strategy_type}策略",
            "risk_limits": {
                "max_position": random.randint(500, 2000),
                "risk_per_trade": random.uniform(0.01, 0.05)
            }
        })

        return config

    @staticmethod
    def generate_backtest_config(strategy_id: str) -> dict:
        """生成回测配置"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=random.randint(365, 730))

        return {
            "backtest_id": f"bt_{strategy_id}_{random.randint(1000, 9999)}",
            "strategy_id": strategy_id,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "initial_capital": random.randint(50000, 200000),
            "commission": random.uniform(0.0001, 0.0005),
            "slippage": random.uniform(0.0001, 0.001)
        }
```

### 测试数据清理

```python
# tests/data/test_data_cleanup.py
import os
import shutil
from pathlib import Path

class TestDataCleanup:
    @staticmethod
    def cleanup_test_databases():
        """清理测试数据库"""
        test_db_files = [
            "test.db",
            "test.sqlite",
            "test_database.db"
        ]

        for db_file in test_db_files:
            if os.path.exists(db_file):
                os.remove(db_file)

    @staticmethod
    def cleanup_test_files():
        """清理测试文件"""
        test_dirs = [
            "test_output",
            "test_results",
            "temp_test_data"
        ]

        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

    @staticmethod
    def cleanup_log_files():
        """清理测试日志文件"""
        log_patterns = [
            "test_*.log",
            "*_test.log",
            "pytest.log"
        ]

        for pattern in log_patterns:
            for log_file in Path(".").glob(pattern):
                log_file.unlink()

    @classmethod
    def cleanup_all(cls):
        """清理所有测试数据"""
        cls.cleanup_test_databases()
        cls.cleanup_test_files()
        cls.cleanup_log_files()
```

## 持续集成

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run linting
      run: |
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

    - name: Run type checking
      run: |
        mypy src/ --ignore-missing-imports

    - name: Run tests
      run: |
        python scripts/run_tests.py --coverage --parallel

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./htmlcov/coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Run security scan
      uses: github/super-linter/slim@v5
      env:
        DEFAULT_BRANCH: main
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        VALIDATE_PYTHON_BLACK: true
        VALIDATE_PYTHON_FLAKE8: true
        VALIDATE_PYTHON_ISORT: true

  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run performance tests
      run: |
        python scripts/run_tests.py --types performance

    - name: Check performance thresholds
      run: |
        # 检查性能阈值逻辑
        echo "性能测试完成"
```

---

*版本: v1.0.0*
*最后更新: {datetime.now().strftime('%Y-%m-%d')}*
