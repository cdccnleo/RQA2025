# 数据版本管理器

数据版本管理器（DataVersionManager）是一个用于管理数据版本的工具，它提供了创建、获取、比较、回滚和删除版本等功能。

## 功能特点

- **版本创建**：创建数据的新版本，并记录版本信息
- **版本获取**：获取指定版本的数据
- **版本列表**：列出所有版本，支持按标签、创建者和分支筛选
- **版本比较**：比较两个版本之间的差异，包括元数据差异和数据差异
- **版本回滚**：回滚到指定版本，创建一个新的版本
- **版本删除**：删除指定版本
- **血缘关系**：跟踪版本之间的血缘关系
- **分支管理**：支持多分支版本管理

## 使用示例

### 初始化版本管理器

```python
from src.data.version_control.version_manager import DataVersionManager

# 创建版本管理器
version_manager = DataVersionManager("./data/versions")
```

### 创建版本

```python
import pandas as pd
from src.data.data_manager import DataModel

# 创建数据模型
data = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})
metadata = {
    'source': 'sample_data',
    'created_at': '2023-01-01'
}
model = DataModel(data, metadata)

# 创建版本
version = version_manager.create_version(
    model,
    description="初始版本",
    tags=["sample", "v1"],
    creator="system"
)
print(f"创建版本: {version}")
```

### 获取版本

```python
# 获取指定版本
model = version_manager.get_version(version)
print(model.data)

# 获取当前版本
current_model = version_manager.get_version()
print(current_model.data)
```

### 列出版本

```python
# 列出所有版本
versions = version_manager.list_versions()
for v in versions:
    print(f"{v['version_id']}: {v['description']}")

# 按标签筛选
versions = version_manager.list_versions(tags=["v1"])
for v in versions:
    print(f"{v['version_id']}: {v['description']}")

# 按创建者筛选
versions = version_manager.list_versions(creator="system")
for v in versions:
    print(f"{v['version_id']}: {v['description']}")

# 限制返回数量
versions = version_manager.list_versions(limit=5)
for v in versions:
    print(f"{v['version_id']}: {v['description']}")
```

### 比较版本

```python
# 比较两个版本
diff = version_manager.compare_versions(version1, version2)

# 查看元数据差异
print("元数据差异:")
print(f"添加: {diff['metadata_diff']['added']}")
print(f"删除: {diff['metadata_diff']['removed']}")
print(f"修改: {diff['metadata_diff']['changed']}")

# 查看数据差异
print("数据差异:")
print(f"行数变化: {diff['data_diff']['shape_diff']['rows']}")
print(f"列数变化: {diff['data_diff']['shape_diff']['columns']}")
print(f"添加列: {diff['data_diff']['columns_diff']['added']}")
print(f"删除列: {diff['data_diff']['columns_diff']['removed']}")
print(f"值变化: {diff['data_diff']['value_diff']}")
```

### 回滚版本

```python
# 回滚到指定版本
new_version = version_manager.rollback(version1)
print(f"回滚创建了新版本: {new_version}")

# 获取回滚后的版本
rollback_model = version_manager.get_version(new_version)
print(rollback_model.data)
```

### 删除版本

```python
# 删除指定版本
version_manager.delete_version(version)
print("版本已删除")
```

### 获取版本血缘关系

```python
# 获取版本的血缘关系
lineage = version_manager.get_lineage(version)
print(f"版本: {lineage['version_id']}")
print("祖先版本:")
for ancestor in lineage['ancestors']:
    print(f"- {ancestor['version_id']}: {ancestor['description']}")
```

## 完整示例

查看 `examples/version_manager_example.py` 文件，了解完整的使用示例。

## 测试

运行测试用例：

```bash
python -m unittest src.data.version_control.test_version_manager
```

## 注意事项

- 版本管理器使用 parquet 文件格式存储数据，确保安装了相关依赖（pandas, pyarrow）
- 当前版本不能被删除，需要先切换到其他版本
- 版本回滚会创建一个新的版本，而不是直接修改当前版本
- 版本删除会更新血缘关系，确保子版本的血缘关系不会丢失
