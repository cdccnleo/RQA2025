# QueryResult使用指南

## 📋 概述

项目中存在两个不同的`QueryResult`类，分别服务于不同的架构层次。本指南帮助开发者正确选择和使用。

## 🎯 快速选择

### 我应该使用哪个QueryResult？

```python
# 问自己：
# 1. 我在写数据库适配器吗？ → 使用 database_interfaces.QueryResult
# 2. 我在实现跨存储查询吗？ → 使用 unified_query.QueryResult
# 3. 我需要pandas DataFrame吗？ → 使用 unified_query.QueryResult
# 4. 我需要轻量级无依赖吗？ → 使用 database_interfaces.QueryResult
```

## 📊 两个QueryResult对比

| 特性 | database_interfaces.QueryResult | unified_query.QueryResult |
|------|--------------------------------|---------------------------|
| **用途** | 数据库适配器层 | 统一查询接口层 |
| **数据类型** | `List[Dict[str, Any]]` | `pd.DataFrame` |
| **依赖** | 无外部依赖 | 依赖pandas |
| **必需参数** | success, data, row_count, execution_time | query_id, success |
| **计数字段** | row_count | record_count |
| **追踪字段** | 无 | query_id, data_source |
| **使用层次** | 底层（靠近数据库） | 高层（靠近业务） |

## 💡 推荐导入方式

### 方式1：使用别名（推荐）

```python
# ✅ 推荐：清晰明确，避免混淆
from src.infrastructure.utils.interfaces.database_interfaces import (
    QueryResult as DBQueryResult
)
from src.infrastructure.utils.components.unified_query import (
    QueryResult as UnifiedQueryResult
)

# 使用
db_result: DBQueryResult = adapter.execute_query("SELECT 1")
unified_result: UnifiedQueryResult = interface.query_data(request)
```

### 方式2：使用完整路径

```python
# ✅ 也可以：最清晰但较长
import src.infrastructure.utils.interfaces.database_interfaces as db_interfaces
import src.infrastructure.utils.components.unified_query as unified_query

# 使用
result1: db_interfaces.QueryResult = ...
result2: unified_query.QueryResult = ...
```

### 方式3：仅在单一场景下使用

```python
# ✅ 可以：如果文件只用一种QueryResult
# 在数据库适配器文件中
from src.infrastructure.utils.interfaces.database_interfaces import QueryResult

# 或在统一查询文件中
from src.infrastructure.utils.components.unified_query import QueryResult
```

### ❌ 不推荐的方式

```python
# ❌ 不推荐：混淆，后导入的会覆盖前面的
from src.infrastructure.utils.interfaces.database_interfaces import QueryResult
from src.infrastructure.utils.components.unified_query import QueryResult  # 覆盖了上面的！

# ❌ 不推荐：不清楚是哪个QueryResult
from xxx import QueryResult  # 哪个？
```

## 📝 使用示例

### 示例1：数据库适配器

```python
"""PostgreSQL适配器示例"""
from typing import Dict, Any, List
from src.infrastructure.utils.interfaces.database_interfaces import (
    QueryResult as DBQueryResult  # 使用别名
)

class PostgreSQLAdapter:
    def execute_query(self, sql: str) -> DBQueryResult:
        """执行SQL查询"""
        try:
            # 执行查询
            cursor = self.connection.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            # 转换为字典列表
            data = [dict(row) for row in rows]
            
            # 返回数据库查询结果
            return DBQueryResult(
                success=True,
                data=data,
                row_count=len(data),
                execution_time=0.5
            )
        except Exception as e:
            return DBQueryResult(
                success=False,
                data=[],
                row_count=0,
                execution_time=0.0,
                error_message=str(e)
            )
```

### 示例2：统一查询接口

```python
"""统一查询接口示例"""
import pandas as pd
from src.infrastructure.utils.components.unified_query import (
    QueryResult as UnifiedQueryResult,  # 使用别名
    QueryRequest
)

class UnifiedQueryInterface:
    def query_data(self, request: QueryRequest) -> UnifiedQueryResult:
        """执行统一查询"""
        try:
            # 从不同数据源查询
            data = self._fetch_from_sources(request)
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 返回统一查询结果
            return UnifiedQueryResult(
                query_id=request.query_id,
                success=True,
                data=df,
                data_source="influxdb",
                record_count=len(df),
                execution_time=1.0
            )
        except Exception as e:
            return UnifiedQueryResult(
                query_id=request.query_id,
                success=False,
                error_message=str(e)
            )
```

### 示例3：结果转换

```python
"""在两种QueryResult之间转换"""
from src.infrastructure.utils.converters import (
    QueryResultConverter,
    convert_db_to_unified  # 便捷函数
)

# 场景：适配器返回DB结果，需要转换为统一结果
def query_with_conversion(adapter, query_id: str):
    # 1. 从适配器获取数据库结果
    db_result = adapter.execute_query("SELECT * FROM users")
    
    # 2. 转换为统一查询结果
    unified_result = QueryResultConverter.db_to_unified(
        db_result,
        query_id=query_id,
        data_source="postgresql"
    )
    
    # 或使用便捷函数
    # unified_result = convert_db_to_unified(db_result, query_id, "postgresql")
    
    # 3. 返回统一结果供业务层使用
    return unified_result
```

## 🔄 数据流示例

```
┌─────────────────┐
│   业务层         │  使用 UnifiedQueryResult
└────────┬────────┘
         │ 
         ↓
┌─────────────────┐
│ 统一查询接口     │  UnifiedQueryResult ←→ DBQueryResult
│ (转换层)         │  使用 QueryResultConverter
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ 数据库适配器     │  使用 DBQueryResult
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   数据库         │
└─────────────────┘
```

## ⚠️ 常见错误和解决方案

### 错误1：导入了错误的QueryResult

```python
# ❌ 错误：在适配器中使用了统一查询结果
from src.infrastructure.utils.components.unified_query import QueryResult

class PostgreSQLAdapter:
    def execute_query(self, sql: str) -> QueryResult:
        # 这需要query_id参数，但适配器层不应该关心这个
        return QueryResult(query_id=???, ...)  # 错误！
```

```python
# ✅ 正确：使用数据库查询结果
from src.infrastructure.utils.interfaces.database_interfaces import QueryResult

class PostgreSQLAdapter:
    def execute_query(self, sql: str) -> QueryResult:
        return QueryResult(
            success=True,
            data=[...],
            row_count=10,
            execution_time=0.5
        )
```

### 错误2：类型注解不明确

```python
# ❌ 错误：不知道是哪个QueryResult
def process(result: QueryResult):  # 哪个QueryResult？
    pass
```

```python
# ✅ 正确：使用完整路径或别名
from src.infrastructure.utils.interfaces.database_interfaces import (
    QueryResult as DBQueryResult
)

def process(result: DBQueryResult):  # 清楚！
    pass
```

### 错误3：混用两种结果

```python
# ❌ 错误：期望DataFrame但得到List[Dict]
from src.infrastructure.utils.interfaces.database_interfaces import QueryResult

result: QueryResult = adapter.execute_query("SELECT 1")
df = result.data.groupby("id").sum()  # 错误！data是List[Dict]不是DataFrame
```

```python
# ✅ 正确：使用转换器
from src.infrastructure.utils.converters import convert_db_to_unified

db_result = adapter.execute_query("SELECT 1")
unified_result = convert_db_to_unified(db_result, query_id="q1")
df = unified_result.data.groupby("id").sum()  # 正确！现在是DataFrame
```

## 📚 测试指南

### 测试数据库适配器

```python
# tests/test_adapter.py
from src.infrastructure.utils.interfaces.database_interfaces import (
    QueryResult as DBQueryResult
)

def test_adapter_query():
    adapter = PostgreSQLAdapter()
    result: DBQueryResult = adapter.execute_query("SELECT 1")
    
    assert isinstance(result, DBQueryResult)
    assert result.success
    assert isinstance(result.data, list)
    assert result.row_count >= 0
```

### 测试统一查询接口

```python
# tests/test_unified_query.py
import pandas as pd
from src.infrastructure.utils.components.unified_query import (
    QueryResult as UnifiedQueryResult
)

def test_unified_query():
    interface = UnifiedQueryInterface()
    result: UnifiedQueryResult = interface.query_data(request)
    
    assert isinstance(result, UnifiedQueryResult)
    assert result.query_id
    assert isinstance(result.data, (pd.DataFrame, type(None)))
    assert result.record_count >= 0
```

## 🎓 代码审查检查清单

在代码审查时，检查以下内容：

- [ ] 导入了正确的QueryResult类
- [ ] 使用了别名或完整路径（避免混淆）
- [ ] 函数签名中有明确的类型注解
- [ ] 在适配器层使用DBQueryResult
- [ ] 在统一查询层使用UnifiedQueryResult
- [ ] 需要转换时使用了QueryResultConverter
- [ ] 测试代码导入了正确的类

## 📖 相关文档

- [QueryResult架构分析](./QUERYRESULT_ARCHITECTURE_ANALYSIS.md)
- [数据库适配器接口](../../src/infrastructure/utils/interfaces/database_interfaces.py)
- [统一查询接口](../../src/infrastructure/utils/components/unified_query.py)
- [QueryResult转换器](../../src/infrastructure/utils/converters/query_result_converter.py)

## 🔗 快速参考

```python
# 数据库适配器层
from src.infrastructure.utils.interfaces.database_interfaces import QueryResult as DBQueryResult

# 统一查询接口层
from src.infrastructure.utils.components.unified_query import QueryResult as UnifiedQueryResult

# 转换工具
from src.infrastructure.utils.converters import QueryResultConverter

# 转换：底层 → 高层
unified = QueryResultConverter.db_to_unified(db_result, query_id="abc")

# 转换：高层 → 底层
db = QueryResultConverter.unified_to_db(unified_result)
```

---

*最后更新: 2025-10-25*  
*维护者: 架构团队*  
*版本: 1.0*

