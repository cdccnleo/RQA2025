# 外部依赖模拟指南

## 概述

本文档说明如何在测试中模拟外部依赖，避免测试时连接真实的Redis、数据库和网络服务。

## 问题背景

测试中常见问题：
- `ConnectionRefusedError`: Redis连接被拒绝
- `TimeoutError`: 数据库连接超时
- `ConnectionError`: 网络请求失败

## 解决方案

### 1. 全局模拟配置

`tests/conftest.py` 中已配置全局模拟：

```python
@pytest.fixture(scope="session", autouse=True)
def mock_redis_globally():
    """全局模拟Redis连接"""
    with patch('redis.Redis') as mock_redis_cls:
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        yield {'redis': mock_redis}
```

### 2. Redis模拟工具

使用 `tests/utils/redis_mock.py`：

```python
from tests.utils.redis_mock import create_redis_mock

mock_redis = create_redis_mock()
mock_redis.set('test_key', '{"key": "value"}')
```

### 3. 测试环境配置

使用 `tests/config/test_environment.py`：

```python
from tests.config.test_environment import TestEnvironmentConfig

with TestEnvironmentConfig() as test_config:
    # 所有外部依赖都会被自动模拟
    redis_adapter = RedisAdapter()
```

## 使用方法

### 自动模拟（推荐）

测试运行脚本自动设置环境变量：

```bash
python scripts/testing/run_tests.py tests/unit/infrastructure/storage/
```

环境变量：
- `TESTING=true`
- `MOCK_EXTERNAL_DEPENDENCIES=true`

### 手动模拟

```python
import pytest
from unittest.mock import patch
from tests.utils.redis_mock import create_redis_mock

class TestRedisAdapter:
    @pytest.fixture
    def mock_redis(self):
        with patch('redis.Redis', return_value=create_redis_mock()):
            yield
```

## 最佳实践

1. **测试隔离**: 每个测试独立，不依赖外部状态
2. **模拟数据管理**: 使用 `RedisTestHelper` 管理测试数据
3. **错误场景测试**: 测试连接失败和异常情况
4. **性能测试**: 使用模拟避免网络延迟

## 故障排除

### 模拟未生效

检查环境变量：

```python
import os
print(f"TESTING: {os.environ.get('TESTING')}")
```

### 导入错误

确保模拟在导入前设置：

```python
from unittest.mock import patch
with patch('redis.Redis'):
    from src.infrastructure.storage.adapters.redis import RedisAdapter
```

## 总结

通过使用模拟工具可以：
1. 提高测试稳定性
2. 加快测试速度
3. 确保测试隔离
4. 简化CI/CD配置 