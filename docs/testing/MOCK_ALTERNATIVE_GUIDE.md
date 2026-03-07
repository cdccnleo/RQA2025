# RQA2025 Mock替代方案指南

## 概述

避免使用 `pytest-mock` 插件，改用Python标准库 `unittest.mock` 和项目测试钩子标准。

## 替代方案

### 方案1：使用Python标准库unittest.mock

```python
from unittest.mock import Mock, patch, MagicMock
import pytest

class TestDataLoader:
    def test_load_data_success(self):
        """使用标准库mock替代pytest-mock"""
        with patch('src.data.loader.stock_loader.akshare.stock_zh_a_hist') as mock_api:
            mock_api.return_value = pd.DataFrame({
                'date': ['2023-01-01', '2023-01-02'],
                'close': [100, 101]
            })
            
            loader = StockLoader()
            result = loader.load_data('000001')
            
            assert result is not None
            mock_api.assert_called_once()
```

### 方案2：使用测试钩子模式

```python
class TestConfigManager:
    def test_config_loading(self):
        """使用测试钩子模式"""
        mock_config = {
            'database': {'host': 'localhost'},
            'redis': {'port': 6379}
        }
        
        # 使用测试钩子注入mock对象
        config_manager = ConfigManager(
            config=mock_config,
            security_service=Mock()  # 测试钩子参数
        )
        
        assert config_manager.get('database.host') == 'localhost'
```

### 方案3：使用Dummy对象

```python
class DummyDataLoader:
    """Dummy数据加载器"""
    def __init__(self, return_data=None):
        self.return_data = return_data or pd.DataFrame()
        self.load_called = False
    
    def load(self, symbol):
        self.load_called = True
        return self.return_data

class TestWithDummyObjects:
    def test_with_dummy_loader(self):
        dummy_data = pd.DataFrame({'date': ['2023-01-01'], 'close': [100]})
        dummy_loader = DummyDataLoader(dummy_data)
        
        processor = DataProcessor()
        result = processor.process_data(
            data={'symbol': '000001'},
            data_loader=dummy_loader
        )
        
        assert dummy_loader.load_called
```

## 迁移步骤

### 1. 识别现有使用
```bash
grep -r "def test.*mocker" tests/
grep -r "mocker\." tests/
```

### 2. 逐步替换
- 优先使用测试钩子模式
- 使用标准库mock
- 创建Dummy对象
- 使用上下文管理器

### 3. 更新依赖
```bash
# 从requirements.txt中移除pytest-mock
# 确保使用Python标准库unittest.mock
```

### 4. 更新CI配置
```yaml
# 在CI配置中移除pytest-mock安装
pip install pytest pytest-cov pytest-xdist  # 移除pytest-mock
```

## 最佳实践

### Mock策略优先级
1. **测试钩子模式**：优先使用，最符合项目规范
2. **标准库mock**：适用于简单的外部依赖
3. **Dummy对象**：适用于复杂的业务逻辑
4. **上下文管理器**：适用于临时mock需求

### 类型安全
```python
from typing import Optional
from unittest.mock import Mock

def test_with_type_safety(self):
    mock_service: Optional[Mock] = Mock()
    mock_service.get_data.return_value = {'test': 'data'}
    
    processor = DataProcessor()
    result = processor.process(data={'input': 'test'}, service=mock_service)
    assert result is not None
```

## 优势

### 使用标准库mock的优势
- ✅ **无额外依赖**：使用Python标准库
- ✅ **更好的类型支持**：与IDE集成更好
- ✅ **更清晰的语法**：patch装饰器更直观
- ✅ **更好的性能**：减少插件开销

### 测试钩子模式的优势
- ✅ **符合项目规范**：遵循现有的测试钩子标准
- ✅ **更好的可测试性**：明确的依赖注入
- ✅ **向后兼容**：不影响生产代码
- ✅ **类型安全**：使用Optional类型注解

## 总结

通过采用以上替代方案，可以完全避免使用 `pytest-mock` 插件，同时保持测试的有效性和可维护性。建议优先使用测试钩子模式，这最符合项目的现有规范和架构设计。 