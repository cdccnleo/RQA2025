# 核心服务层导入问题修复进展

## 问题描述

核心服务层测试覆盖率0%，所有测试因导入错误无法执行。主要错误：
- `No module named 'src.core.container'`
- `No module named 'src.core.core_services'`
- `No module named 'src.core.foundation'`

## 已尝试的修复方案

### 1. 修改pytest.ini配置
- ✅ 添加了 `pythonpath = . src`
- ⚠️ 效果：pytest的rootdir是`tests`目录，相对路径可能不正确

### 2. 创建tests/unit/core/conftest.py
- ✅ 添加了`pytest_configure`钩子配置路径
- ⚠️ 效果：路径配置可能执行时机不对

### 3. 修改测试文件使用延迟导入
- ✅ 修改了`test_container_components_coverage.py`：
  - 移除了模块级别的导入和`pytest.skip`
  - 添加了`_import_container_modules()`函数
  - 添加了`container_modules` fixture（function scope）
  - 所有测试方法都使用fixture获取导入的模块
- ⚠️ 效果：导入仍然失败，错误信息：`No module named 'src.core.container'`

### 4. 改进路径配置函数
- ✅ 修改了`_ensure_paths()`函数：
  - 使用绝对路径
  - 确保路径在sys.path的最前面
  - 添加了路径验证
- ⚠️ 效果：导入仍然失败

## 发现的问题

1. **pytest工作目录问题**：
   - pytest的rootdir是`C:\PythonProject\RQA2025\tests`
   - 相对路径`.`指向`tests`目录，而不是项目根目录

2. **全局conftest.py循环导入**：
   - `tests/conftest.py:46` 有循环导入：`from conftest import _ensure_stub_dependencies`
   - 这可能影响pytest的导入机制

3. **命令行导入成功**：
   - 在命令行中直接导入`src.core.container.factory_components`是成功的
   - 说明模块本身没有问题，问题在于pytest环境

## 下一步建议

### 方案A：修复全局conftest.py循环导入
1. 检查`tests/conftest.py`中的循环导入问题
2. 修复`_ensure_stub_dependencies`的导入方式

### 方案B：使用PYTHONPATH环境变量
1. 在pytest运行前设置`PYTHONPATH`环境变量
2. 或者修改pytest.ini中的pythonpath使用绝对路径

### 方案C：修改pytest的rootdir
1. 修改pytest.ini中的`testpaths`配置
2. 或者使用`--rootdir`参数指定项目根目录

### 方案D：使用相对导入
1. 修改测试文件使用相对导入而不是绝对导入
2. 但这可能影响其他测试文件

## 当前状态

- ✅ **已完成**：修复了全局conftest.py的循环导入问题
- ✅ **已完成**：修复了BaseConfig的fallback实现
- ✅ **已完成**：改进了路径配置函数，支持从tests目录向上查找项目根目录
- ⏳ **进行中**：测试可以正常收集，但导入仍然失败（`No module named 'src.core.container'`）
- ⚠️ **待解决**：pytest环境中的路径配置问题，需要进一步调查

## 相关文件

- `tests/unit/core/container/test_container_components_coverage.py` - 已修改
- `tests/unit/core/conftest.py` - 已创建
- `pytest.ini` - 已添加pythonpath配置
- `tests/conftest.py` - 需要修复循环导入

---
**更新时间**: 2025-01-28
**状态**: 进行中

