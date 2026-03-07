# infrastructure_index

**文件路径**: `tools\infrastructure_index.py`

## 模块描述

基础设施层 - 配置管理组件

infrastructure_index 模块

配置管理相关的文件
提供配置管理相关的功能实现。

## 函数

### get_interface_statistics

### get_interfaces_by_category

**参数**:

- `category: str`

**返回值**: `list`

### get_interfaces_by_priority

**参数**:

- `priority: str`

**返回值**: `list`

### get_interfaces_by_dependency

**参数**:

- `interface_name: str`

**返回值**: `list`

### get_interface_status

**参数**:

- `interface_name: str`

**返回值**: `dict`

### get_interface_test_status

**参数**:

- `interface_name: str`

**返回值**: `dict`

### get_interface_documentation_status

**参数**:

- `interface_name: str`

**返回值**: `dict`

