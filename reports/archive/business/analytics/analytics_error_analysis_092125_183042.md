# RQA2025 错误分析报告

**生成时间**: 2025-07-19 09:21:25

## 错误统计

- **总错误数**: 313

## 各层错误详情

### INFRASTRUCTURE 层
- **错误数**: 31
- **主要错误**:
  1. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\infrastructure\cache\te...
  2. ImportError: cannot import name 'ThreadSafeCache' from 'src.infrastructure.cache.thread_safe_cache' ...
  3. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\infrastructure\config\s...

### DATA 层
- **错误数**: 33
- **主要错误**:
  1. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\data\loader\test_base_l...
  2. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\data\loader\test_news_l...
  3. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\data\loader\test_sentim...

### FEATURES 层
- **错误数**: 105
- **主要错误**:
  1. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\features\optimizer\test...
  2. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\features\orderbook\test...
  3. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\features\orderbook\test...

### MODELS 层
- **错误数**: 3
- **主要错误**:
  1. ImportError while loading conftest 'C:\PythonProject\RQA2025\tests\unit\models\conftest.py'.
tests\u...
  2. ModuleNotFoundError: No module named 'scipy.sparse'...
  3. ERROR conda.cli.main_run:execute(127): `conda run python -m pytest tests/unit/models/ -v --tb=short`...

### TRADING 层
- **错误数**: 117
- **主要错误**:
  1. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\trading\execution\test_...
  2. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\trading\order\test_chin...
  3. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\trading\order\test_orde...

### BACKTEST 层
- **错误数**: 24
- **主要错误**:
  1. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\backtest\evaluation\tes...
  2. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\backtest\test_analyzer....
  3. ImportError while importing test module 'C:\PythonProject\RQA2025\tests\unit\backtest\test_backtest_...

## 修复建议

1. 修复导入: DataTransformer
2. 安装缺失模块: pip install scipy.optimize
3. 修复导入: Logger
4. 修复导入: ConfigNotFoundError
5. 修复导入: ThreadSafeCache
6. 检查语法错误，可能需要修复代码格式
7. 安装缺失模块: pip install scipy.sparse
8. 修复导入: ConnectionFailedException
9. 修复导入: ConnectionError
10. 修复导入: MonitoringService
