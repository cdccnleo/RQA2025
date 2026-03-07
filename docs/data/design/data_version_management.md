# 数据版本管理系统

## 概述

数据版本管理系统是 RQA2025 项目的核心组件之一，用于管理数据的版本历史、血缘关系和变更追踪。该系统由 `DataVersionManager` 类实现，并集成到 `DataManager` 中，为量化交易模型提供数据版本控制和血缘追踪能力。

## 系统架构

数据版本管理系统主要包含以下组件：

1. **DataVersionManager**：版本管理器，负责创建、获取、比较、回滚和删除版本等核心功能
2. **DataManager**：数据管理器，集成版本管理功能，提供统一的数据管理接口
3. **DataModel**：数据模型，封装数据和元数据

### 文件结构

```
src/
├── data/
│   ├── version_control/
│   │   ├── version_manager.py     # 版本管理器实现
│   │   ├── test_version_manager.py # 版本管理器测试
│   │   ├── examples/              # 示例代码
│   │   │   └── version_manager_example.py
│   │   └── README.md              # 版本管理器使用说明
│   ├── data_manager.py            # 数据管理器实现
│   └── test_data_manager.py       # 数据管理器测试
└── examples/
    └── data_version_example.py    # 数据版本管理示例
```

## 核心功能

### 版本创建与管理

- **创建版本**：每次数据加载或处理后，自动创建新版本
- **版本元数据**：记录版本的描述、标签、创建者、时间戳等信息
- **版本存储**：使用 parquet 格式高效存储版本数据

### 版本比较与回滚

- **版本比较**：比较两个版本之间的元数据和数据差异
- **版本回滚**：回滚到指定版本，创建新版本而不是直接修改

### 血缘关系追踪

- **血缘记录**：记录版本之间的父子关系
- **血缘查询**：查询版本的祖先版本

### 版本查询与筛选

- **列出版本**：列出所有版本，支持按标签、创建者和分支筛选
- **获取版本**：获取指定版本的数据和元数据

## 使用方法

### 初始化数据管理器

```python
from src.data.data_manager import DataManager

# 创建配置
config = {
    'version_dir': './data/versions',
    'stock_config': {...},
    'index_config': {...},
    'financial_config': {...},
    'news_config': {...}
}

# 创建数据管理器
data_manager = DataManager(config)
```

### 加载数据并创建版本

```python
# 加载股票数据
stock_model = data_manager.load_data(
    data_type='stock',
    start_date='2023-01-01',
    end_date='2023-01-10',
    symbols=['000001.SZ', '000002.SZ']
)
stock_version = data_manager.version_manager.current_version
```

### 列出和筛选版本

```python
# 列出所有版本
versions = data_manager.list_versions()

# 按标签筛选
stock_versions = data_manager.list_versions(tags=['stock'])

# 按创建者筛选
user_versions = data_manager.list_versions(creator='user1')

# 限制返回数量
recent_versions = data_manager.list_versions(limit=5)
```

### 比较版本

```python
# 比较两个版本
diff = data_manager.compare_versions(version1, version2)

# 查看元数据差异
print(f"添加的元数据: {diff['metadata_diff']['added']}")
print(f"删除的元数据: {diff['metadata_diff']['removed']}")
print(f"修改的元数据: {diff['metadata_diff']['changed']}")

# 查看数据差异
print(f"行数变化: {diff['data_diff']['shape_diff']['rows']}")
print(f"列数变化: {diff['data_diff']['shape_diff']['columns']}")
print(f"添加的列: {diff['data_diff']['columns_diff']['added']}")
print(f"删除的列: {diff['data_diff']['columns_diff']['removed']}")
print(f"值变化: {diff['data_diff']['value_diff']}")
```

### 回滚版本

```python
# 回滚到指定版本
new_version = data_manager.rollback(target_version)

# 获取回滚后的数据
rollback_model = data_manager.get_version(new_version)
```

### 查询血缘关系

```python
# 获取版本的血缘关系
lineage = data_manager.get_lineage(version)

# 查看祖先版本
for ancestor in lineage['ancestors']:
    print(f"{ancestor['version_id']}: {ancestor['description']}")
```

### 合并数据

```python
# 合并多个数据源
merged_model = data_manager.merge_data(
    data_types=['stock', 'index', 'financial'],
    start_date='2023-01-01',
    end_date='2023-01-10',
    symbols=['000001.SZ', '000002.SZ']
)
```

## 最佳实践

1. **版本命名**：使用有意义的描述和标签，便于后续查找
2. **定期清理**：删除不再需要的旧版本，节省存储空间
3. **血缘追踪**：利用血缘关系追踪数据处理流程
4. **版本比较**：在模型训练前比较数据版本，确保数据一致性
5. **回滚策略**：出现问题时，回滚到已知良好的版本

## 技术细节

### 存储格式

- **数据存储**：使用 parquet 格式存储版本数据，提供高效的压缩和读写性能
- **元数据存储**：使用 JSON 格式存储版本元数据、历史记录和血缘关系

### 并发控制

- 使用线程锁保证在多线程环境下的数据一致性
- 版本创建、删除和回滚操作都在锁的保护下执行

### 异常处理

- 所有操作都有完善的异常处理机制
- 操作失败时会回滚已执行的部分操作，确保数据一致性

## 测试覆盖

系统包含全面的单元测试，覆盖以下方面：

1. **参数化测试**：使用 `@pytest.mark.parametrize` 覆盖多输入组合
2. **异常断言**：使用 `pytest.raises` 验证异常处理逻辑
3. **Fixtures 复用**：减少重复代码
4. **Mock 外部依赖**：模拟文件系统和数据加载器
5. **正则表达式匹配**：使用正则表达式匹配断言，避免描述不一致

## 示例

查看以下示例文件，了解更多使用方法：

1. `src/data/version_control/examples/version_manager_example.py`：版本管理器基本使用示例
2. `src/examples/data_version_example.py`：数据管理器集成版本管理的完整示例

## 未来扩展

1. **分布式版本存储**：支持将版本数据存储在分布式文件系统中
2. **版本差异可视化**：提供版本差异的可视化展示
3. **自动版本清理**：根据策略自动清理过期版本
4. **版本标签管理**：增强版本标签的管理功能
5. **版本权限控制**：增加版本的访问权限控制
