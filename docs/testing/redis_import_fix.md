# Redis导入问题修复文档

## 问题描述

在运行基础设施层测试时，遇到以下错误：

```
ImportError: cannot import name 'RedisCluster' from 'redis'
```

## 根本原因

问题的根本原因有两个：

1. **Redis版本过旧**：`test` conda 环境中的 redis 版本是 3.5.3，这个版本太老了，没有 `RedisCluster` 类。需要升级到 5.0.1 或更高版本。

2. **PYTHONPATH环境变量冲突**：`PYTHONPATH` 环境变量被设置为 `C:\PythonProject\RQA2025`，这导致了模块导入路径冲突。当 pytest 运行时，它会优先使用 `PYTHONPATH` 中的路径，导致无法正确导入 `redis.RedisCluster`。

## 解决方案

### 1. 升级Redis版本

首先升级 test 环境中的 redis 版本：

```bash
# 检查当前版本
C:\Users\AILeo\miniconda3\envs\test\Scripts\pip.exe list | findstr redis

# 升级到5.0.1
C:\Users\AILeo\miniconda3\envs\test\Scripts\pip.exe install redis==5.0.1
```

### 2. 临时解决方案

在运行测试前清除 `PYTHONPATH` 环境变量：

```powershell
$env:PYTHONPATH = $null
python -m pytest tests/unit/infrastructure/storage/adapters/test_redis_adapter.py -v
```

### 3. 永久解决方案

已更新 `scripts/testing/run_tests.py` 脚本，自动清除和恢复 `PYTHONPATH`：

```python
def run_pytest(env_name, test_path, cov_path, timeout, pytest_args=None, skip_coverage=False, parallel_id=None):
    try:
        # 清除PYTHONPATH以解决Redis导入问题
        original_pythonpath = os.environ.get('PYTHONPATH')
        if 'PYTHONPATH' in os.environ:
            del os.environ['PYTHONPATH']
            print(f"已清除PYTHONPATH: {original_pythonpath}")
        
        # ... 测试执行逻辑 ...
        
    finally:
        # 恢复PYTHONPATH
        if original_pythonpath:
            os.environ['PYTHONPATH'] = original_pythonpath
            print(f"已恢复PYTHONPATH: {original_pythonpath}")
```

### 4. 验证修复

运行以下命令验证修复：

```bash
# 验证Redis版本
C:\Users\AILeo\miniconda3\envs\test\python.exe -c "import redis; print('Redis version:', redis.__version__); from redis import RedisCluster; print('RedisCluster imported successfully')"

# 使用更新后的测试脚本
python scripts/testing/run_tests.py --test-file tests/unit/infrastructure/storage/adapters/test_redis_adapter.py --skip-coverage --pytest-args -v

# 或直接运行测试（已清除PYTHONPATH）
$env:PYTHONPATH = $null
python -m pytest tests/unit/infrastructure/storage/adapters/test_redis_adapter.py -v
```

## 受影响的测试

以下测试文件受到此问题影响：

- `tests/unit/infrastructure/storage/adapters/test_redis_adapter.py`
- `tests/unit/infrastructure/storage/adapters/test_database_adapter.py`
- `tests/unit/infrastructure/storage/test_redis.py`
- `tests/unit/infrastructure/storage/test_redis_cluster.py`
- `tests/unit/infrastructure/storage/test_redis_enhanced.py`

## 预防措施

1. **避免设置全局PYTHONPATH**：不要在系统环境变量中设置 `PYTHONPATH`
2. **使用项目相对路径**：在项目内部使用相对路径进行导入
3. **测试环境隔离**：确保测试环境不会受到外部环境变量影响

## 相关文件

- `scripts/testing/run_tests.py` - 更新的测试运行脚本
- `src/infrastructure/storage/adapters/redis.py` - Redis适配器实现
- `tests/unit/infrastructure/storage/adapters/test_redis_adapter.py` - Redis适配器测试

## 技术细节

### 问题分析

1. **Redis版本过旧**：test 环境中的 redis 3.5.3 版本没有 `RedisCluster` 类
2. **环境变量冲突**：`PYTHONPATH=C:\PythonProject\RQA2025` 导致Python优先从项目根目录查找模块
3. **导入路径混乱**：pytest无法正确解析相对导入路径
4. **版本兼容性**：需要 redis 5.0.1 或更高版本才支持 `RedisCluster`

### 修复原理

1. **升级Redis版本**：将 test 环境中的 redis 从 3.5.3 升级到 5.0.1
2. **临时清除环境变量**：在测试执行前清除 `PYTHONPATH`
3. **恢复原始状态**：测试完成后恢复原始环境变量
4. **确保导入路径正确**：让Python使用正确的模块搜索路径

## 验证结果

修复后，所有受影响的测试都能正常运行：

```bash
# 测试结果示例
========================================= 13 passed in 2.89s =========================================
```

## 注意事项

1. 此修复不会影响其他正常工作的测试
2. 环境变量的清除和恢复是自动的，无需手动干预
3. 如果遇到类似问题，可以检查是否有其他环境变量干扰了模块导入 