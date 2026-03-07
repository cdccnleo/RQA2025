# Pytest路径配置问题总结

## 问题描述

核心服务层和工具层的测试文件无法导入`src`模块，错误信息：
- `ModuleNotFoundError: No module named 'src.core.container'`
- `ModuleNotFoundError: No module named 'src.utils'`

## 已尝试的修复方案

### 1. 修改pytest.ini配置
- ✅ 添加了 `pythonpath = . src`
- ⚠️ 效果：pytest的rootdir是`tests`目录，相对路径可能不正确
- ⚠️ 尝试了多行格式，但pytest可能不支持

### 2. 改进全局conftest.py
- ✅ 在模块级别添加了路径配置（在导入pytest之前）
- ✅ 在`pytest_configure`钩子中添加了路径配置（使用`tryfirst=True`）
- ⚠️ 效果：路径配置仍然没有生效

### 3. 创建局部conftest.py
- ✅ 为`tests/unit/core`创建了`conftest.py`
- ✅ 为`tests/unit/utils`创建了`conftest.py`
- ⚠️ 效果：路径配置仍然没有生效

### 4. 使用延迟导入
- ✅ 在测试函数内部使用`importlib`动态导入
- ⚠️ 效果：导入仍然失败

## 发现的问题

1. **pytest工作目录问题**：
   - pytest的rootdir是`C:\PythonProject\RQA2025\tests`
   - `pythonpath = . src`中的`.`指向`tests`目录，而不是项目根目录
   - 但`src`应该能正确解析为`tests/src`或项目根目录的`src`

2. **其他测试文件成功导入**：
   - `tests/unit/data/cache/test_enhanced_cache_manager.py`可以直接导入`from src.data.cache import ...`
   - 说明pytest.ini的pythonpath配置对某些测试是有效的
   - 可能这些测试文件在导入时路径已经配置好了

3. **导入时机问题**：
   - pytest在导入测试文件时，路径配置可能还没有执行
   - `pytest_configure`钩子可能在测试文件导入之后才执行

## 下一步建议

### 方案A：使用PYTHONPATH环境变量
```bash
$env:PYTHONPATH="C:\PythonProject\RQA2025;C:\PythonProject\RQA2025\src"
python -m pytest tests/unit/utils/test_logger.py
```

### 方案B：修改pytest.ini使用绝对路径
- 需要动态计算项目根目录
- 或者使用环境变量

### 方案C：使用pytest插件
- 创建一个pytest插件来配置路径
- 在`pytest_load_initial_conftests`钩子中配置路径

### 方案D：修改测试文件使用相对导入
- 但这可能影响其他测试文件

## 当前状态

- ⏳ **进行中**：已尝试多种方案，但路径配置问题仍然存在
- ⚠️ **待解决**：需要找到为什么其他测试文件可以成功导入，而工具层和核心服务层不行
- 📝 **建议**：可能需要检查这些模块的特殊导入依赖，或者使用环境变量方式

## 相关文件

- `pytest.ini` - 已修改
- `tests/conftest.py` - 已改进
- `tests/unit/core/conftest.py` - 已创建
- `tests/unit/utils/conftest.py` - 已创建
- `tests/unit/core/container/test_container_components_coverage.py` - 已修改
- `tests/unit/utils/test_logger.py` - 已修改

---
**更新时间**: 2025-01-28
**状态**: 进行中

