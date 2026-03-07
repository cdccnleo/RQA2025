# 工具层测试修复进展

## 问题描述

工具层测试文件创建后，所有测试因导入错误无法执行。主要错误：
- `ModuleNotFoundError: No module named 'src.utils'`

## 已尝试的修复方案

### 1. 在测试文件中添加路径配置
- ✅ 在模块级别添加了路径配置
- ⚠️ 效果：路径配置没有生效，pytest在导入测试文件时路径还未配置

### 2. 创建tests/unit/utils/conftest.py
- ✅ 添加了`pytest_configure`钩子配置路径
- ⚠️ 效果：路径配置仍然没有生效

### 3. 使用延迟导入
- ✅ 在测试函数内部使用`importlib`动态导入
- ⚠️ 效果：导入仍然失败

## 发现的问题

1. **路径配置时机问题**：
   - pytest在导入测试文件时，路径配置可能还没有执行
   - 这与核心服务层的导入问题类似

2. **pytest工作目录问题**：
   - pytest的rootdir是`C:\PythonProject\RQA2025\tests`
   - 相对路径`.`指向`tests`目录，而不是项目根目录

## 下一步建议

### 方案A：使用pytest.ini配置pythonpath
- 修改`pytest.ini`中的`pythonpath`使用绝对路径
- 或者使用环境变量`PYTHONPATH`

### 方案B：修复全局conftest.py的路径配置
- 确保`tests/conftest.py`中的路径配置在导入src模块之前执行
- 已经修复了循环导入问题，但路径配置可能还需要改进

### 方案C：使用相对导入
- 修改测试文件使用相对导入而不是绝对导入
- 但这可能影响其他测试文件

## 当前状态

- ⏳ **进行中**：已创建测试文件和conftest.py
- ⚠️ **待解决**：pytest环境中的路径配置问题
- 📝 **建议**：与核心服务层导入问题一起解决，可能需要统一修复pytest的路径配置机制

## 相关文件

- `tests/unit/utils/test_logger.py` - 已修改
- `tests/unit/utils/test_backtest_utils.py` - 已修改
- `tests/unit/utils/test_doc_manager.py` - 已修改
- `tests/unit/utils/conftest.py` - 已创建

---
**更新时间**: 2025-01-28
**状态**: 进行中

