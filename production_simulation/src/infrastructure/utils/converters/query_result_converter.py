"""
QueryResult转换器

提供两种QueryResult类之间的标准转换方法。

使用场景：
    当需要在数据库适配器层和统一查询接口层之间转换数据时使用。
    
架构说明：
    database_interfaces.QueryResult (底层) ←→ unified_query.QueryResult (高层)
    
导入说明：
    本模块已使用别名避免命名冲突：
    - DBQueryResult: database_interfaces.QueryResult
    - UnifiedQueryResult: unified_query.QueryResult
"""

import pandas as pd
from typing import Optional
from datetime import datetime

# 使用别名导入避免命名冲突
from src.infrastructure.utils.interfaces.database_interfaces import (
    QueryResult as DBQueryResult
)
from src.infrastructure.utils.components.unified_query import (
    QueryResult as UnifiedQueryResult
)


class QueryResultConverter:
    """
    QueryResult转换器
    
    提供database_interfaces.QueryResult和unified_query.QueryResult之间的双向转换。
    
    使用示例：
        >>> # 底层 → 高层
        >>> db_result = DBQueryResult(...)
        >>> unified = QueryResultConverter.db_to_unified(db_result, query_id="abc")
        
        >>> # 高层 → 底层
        >>> unified_result = UnifiedQueryResult(...)
        >>> db = QueryResultConverter.unified_to_db(unified_result)
    """
    
    @staticmethod
    def db_to_unified(
        db_result: DBQueryResult,
        query_id: str,
        data_source: Optional[str] = None
    ) -> UnifiedQueryResult:
        """
        将数据库查询结果转换为统一查询结果
        
        转换规则：
            - success → success (保持不变)
            - data (List[Dict]) → data (pd.DataFrame)
            - row_count → record_count
            - execution_time → execution_time (保持不变)
            - error_message → error_message (保持不变)
            - 新增: query_id (必需参数)
            - 新增: data_source (可选参数)
        
        Args:
            db_result: 数据库查询结果（底层）
            query_id: 查询唯一标识符（必需）
            data_source: 数据来源标识（可选，如"postgresql", "redis"等）
            
        Returns:
            统一查询结果（高层）
            
        Example:
            >>> db_result = DBQueryResult(
            ...     success=True,
            ...     data=[{"id": 1, "name": "test"}],
            ...     row_count=1,
            ...     execution_time=0.5
            ... )
            >>> unified = QueryResultConverter.db_to_unified(
            ...     db_result, 
            ...     query_id="query-001",
            ...     data_source="postgresql"
            ... )
            >>> print(unified.query_id)  # "query-001"
            >>> print(type(unified.data))  # <class 'pandas.DataFrame'>
        """
        # 转换数据格式：List[Dict] → pd.DataFrame
        df_data = None
        if db_result.data is not None and len(db_result.data) > 0:
            try:
                df_data = pd.DataFrame(db_result.data)
            except Exception as e:
                # 如果转换失败，记录错误但不中断
                print(f"警告: 数据转换为DataFrame失败: {e}")
                df_data = None
        
        return UnifiedQueryResult(
            query_id=query_id,
            success=db_result.success,
            data=df_data,
            error_message=db_result.error_message,
            execution_time=db_result.execution_time,
            data_source=data_source or "unknown",
            record_count=db_result.row_count
        )
    
    @staticmethod
    def unified_to_db(
        unified_result: UnifiedQueryResult
    ) -> DBQueryResult:
        """
        将统一查询结果转换为数据库查询结果
        
        转换规则：
            - success → success (保持不变)
            - data (pd.DataFrame) → data (List[Dict])
            - record_count → row_count
            - execution_time → execution_time (保持不变)
            - error_message → error_message (保持不变)
            - 丢弃: query_id, data_source (底层不需要)
        
        Args:
            unified_result: 统一查询结果（高层）
            
        Returns:
            数据库查询结果（底层）
            
        Example:
            >>> unified = UnifiedQueryResult(
            ...     query_id="query-001",
            ...     success=True,
            ...     data=pd.DataFrame([{"id": 1}]),
            ...     record_count=1,
            ...     execution_time=0.5
            ... )
            >>> db_result = QueryResultConverter.unified_to_db(unified)
            >>> print(type(db_result.data))  # <class 'list'>
            >>> print(db_result.data)  # [{"id": 1}]
        """
        # 转换数据格式：pd.DataFrame → List[Dict]
        list_data = []
        if unified_result.data is not None:
            try:
                list_data = unified_result.data.to_dict('records')
            except Exception as e:
                # 如果转换失败，记录错误但不中断
                print(f"警告: DataFrame转换为List[Dict]失败: {e}")
                list_data = []
        
        return DBQueryResult(
            success=unified_result.success,
            data=list_data,
            row_count=unified_result.record_count,
            execution_time=unified_result.execution_time,
            error_message=unified_result.error_message
        )
    
    @staticmethod
    def validate_db_result(db_result: DBQueryResult) -> bool:
        """
        验证数据库查询结果的有效性
        
        Args:
            db_result: 数据库查询结果
            
        Returns:
            是否有效
        """
        if not isinstance(db_result, DBQueryResult):
            return False
        
        # 基本字段检查
        if not isinstance(db_result.success, bool):
            return False
        
        if not isinstance(db_result.row_count, int) or db_result.row_count < 0:
            return False
        
        if not isinstance(db_result.execution_time, (int, float)) or db_result.execution_time < 0:
            return False
        
        return True
    
    @staticmethod
    def validate_unified_result(unified_result: UnifiedQueryResult) -> bool:
        """
        验证统一查询结果的有效性
        
        Args:
            unified_result: 统一查询结果
            
        Returns:
            是否有效
        """
        if not isinstance(unified_result, UnifiedQueryResult):
            return False
        
        # 基本字段检查
        if not unified_result.query_id or not isinstance(unified_result.query_id, str):
            return False
        
        if not isinstance(unified_result.success, bool):
            return False
        
        if not isinstance(unified_result.record_count, int) or unified_result.record_count < 0:
            return False
        
        if not isinstance(unified_result.execution_time, (int, float)) or unified_result.execution_time < 0:
            return False
        
        return True


# 便捷函数：提供更简短的调用方式
def convert_db_to_unified(db_result: DBQueryResult, query_id: str, 
                          data_source: Optional[str] = None) -> UnifiedQueryResult:
    """便捷函数：数据库结果 → 统一结果"""
    return QueryResultConverter.db_to_unified(db_result, query_id, data_source)


def convert_unified_to_db(unified_result: UnifiedQueryResult) -> DBQueryResult:
    """便捷函数：统一结果 → 数据库结果"""
    return QueryResultConverter.unified_to_db(unified_result)

