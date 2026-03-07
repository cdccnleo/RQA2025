# 避免使用 pytest-mock 插件指南

## 概述

基于项目现有的mock规范和测试钩子标准，本文档提供完整的方案来避免使用 `pytest-mock` 插件，改用Python标准库 `unittest.mock`。

## 当前状况分析

### 发现的问题
- 项目中有 **11个文件** 使用了 `pytest-mock` 插件
- 主要集中在数据层测试（7个文件）
- 涉及外部依赖如 `akshare`、`pandas`、`os` 等

### 使用统计
- **总计**: 67个mocker参数使用
- **patch调用**: 20处
- **mocker调用**: 18处

## 替代方案

### 方案1：使用Python标准库unittest.mock（推荐）

#### 基本用法
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

#### 装饰器用法
```python
@patch('src.data.loader.stock_loader.akshare.stock_zh_a_hist')
def test_load_data_with_decorator(mock_api):
    """使用装饰器进行mock"""
    mock_api.return_value = pd.DataFrame({'test': [1, 2, 3]})
    
    loader = StockLoader()
    result = loader.load_data('000001')
    
    assert result is not None
    mock_api.assert_called_once()
```

### 方案2：使用测试钩子模式（最符合项目规范）

#### 构造函数注入
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

#### 方法级注入
```python
def test_process_with_mock_dependencies(self):
    """测试带mock依赖的处理方法"""
    mock_data_loader = Mock()
    mock_data_loader.load.return_value = pd.DataFrame({'test': [1, 2, 3]})
    
    processor = DataProcessor()
    result = processor.process_data(
        data={'symbol': '000001'},
        data_loader=mock_data_loader  # 方法级测试钩子
    )
    
    assert result is not None
    mock_data_loader.load.assert_called_once()
```

### 方案3：使用Dummy对象

#### 创建Dummy类
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

### 步骤1：识别现有使用
```bash
# 使用迁移脚本分析
python scripts/testing/migrate_pytest_mock.py --report
```

### 步骤2：逐步替换
1. **优先使用测试钩子模式**：对于有测试钩子的模块
2. **使用标准库mock**：对于简单的外部依赖
3. **创建Dummy对象**：对于复杂的业务逻辑

### 步骤3：更新依赖
```bash
# 从requirements.txt中移除pytest-mock
# 确保使用Python标准库unittest.mock
```

### 步骤4：更新CI配置
```yaml
# 在CI配置中移除pytest-mock安装
pip install pytest pytest-cov pytest-xdist  # 移除pytest-mock
```

## 具体迁移示例

### 原代码（使用pytest-mock）
```python
def test_disk_cache_set_get(mocker):
    mocker.patch('src.data.cache.disk_cache.os.path.exists', return_value=True)
    cache = DiskCache('tmp')
    cache.set('key', 'value')
    result = cache.get('key')
    assert result == 'value'
```

### 迁移后（使用标准库mock）
```python
from unittest.mock import Mock, patch, MagicMock

def test_disk_cache_set_get():
    """使用标准库mock替代pytest-mock"""
    with patch('src.data.cache.disk_cache.os.path.exists', return_value=True):
        cache = DiskCache('tmp')
        cache.set('key', 'value')
        result = cache.get('key')
        assert result == 'value'
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

### 文档化Mock
```python
def test_with_documented_mock(self):
    """文档化的mock测试"""
    # Mock外部API调用
    with patch('src.data.loader.stock_loader.akshare.stock_zh_a_hist') as mock_api:
        # 设置mock行为
        mock_api.return_value = pd.DataFrame({
            'date': ['2023-01-01'],
            'close': [100]
        })
        
        # 执行测试
        loader = StockLoader()
        result = loader.load_data('000001')
        
        # 验证结果和mock调用
        assert result is not None
        mock_api.assert_called_once_with('000001')
```

## 优势对比

### 使用标准库mock的优势
- ✅ **无额外依赖**：使用Python标准库
- ✅ **更好的类型支持**：与IDE集成更好
- ✅ **更清晰的语法**：patch装饰器更直观
- ✅ **更好的性能**：减少插件开销
- ✅ **更好的兼容性**：与所有Python版本兼容

### 测试钩子模式的优势
- ✅ **符合项目规范**：遵循现有的测试钩子标准
- ✅ **更好的可测试性**：明确的依赖注入
- ✅ **向后兼容**：不影响生产代码
- ✅ **类型安全**：使用Optional类型注解
- ✅ **更好的维护性**：清晰的依赖关系

## 迁移工具

### 自动化迁移脚本
```bash
# 生成迁移报告
python scripts/testing/migrate_pytest_mock.py --report

# 迁移特定文件
python scripts/testing/migrate_pytest_mock.py --migrate-file tests/unit/data/test_cache_disk_cache.py

# 查看迁移建议（不实际修改）
python scripts/testing/migrate_pytest_mock.py --migrate-file tests/unit/data/test_cache_disk_cache.py --dry-run
```

### 手动迁移检查清单
- [ ] 移除 `mocker` 参数
- [ ] 添加 `from unittest.mock import Mock, patch, MagicMock`
- [ ] 将 `mocker.patch()` 替换为 `patch()`
- [ ] 使用 `with` 语句或装饰器
- [ ] 添加 `assert_called_once()` 验证
- [ ] 更新测试文档

## 总结

通过采用以上替代方案，可以完全避免使用 `pytest-mock` 插件，同时保持测试的有效性和可维护性。建议优先使用测试钩子模式，这最符合项目的现有规范和架构设计。

### 迁移优先级
1. **高优先级**：数据层测试文件（7个文件）
2. **中优先级**：基础设施层测试文件（2个文件）
3. **低优先级**：特征层和模型层测试文件（2个文件）

### 预期收益
- 减少项目依赖
- 提高测试性能
- 更好的类型支持
- 符合项目规范
- 提高代码可维护性 