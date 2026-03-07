"""
PostgreSQL查询执行器组件

负责PostgreSQL数据库的查询执行、参数化查询和结果处理。
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    from src.infrastructure.utils.interfaces.database_interfaces import QueryResult
except ImportError:
    from dataclasses import dataclass
    
    @dataclass
    class QueryResult:
        success: bool
        data: Any
        error: Optional[str]
        execution_time: float


class PostgreSQLQueryExecutor:
    """PostgreSQL查询执行器"""
    
    def __init__(self, client=None):
        """
        初始化查询执行器
        
        Args:
            client: psycopg2连接客户端
        """
        self.client = client
    
    def set_client(self, client) -> None:
        """设置数据库客户端"""
        self.client = client
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        执行查询
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果
        """
        if not self.client:
            return QueryResult(
                success=False,
                data=[],
                row_count=0,
                error_message="数据库未连接",
                execution_time=0.0
            )
        
        start_time = time.time()
        
        try:
            with self.client.cursor() as cursor:
                # 执行查询
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # 获取结果
                results = cursor.fetchall()
                
                # 获取列名
                column_names = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # 转换为字典列表
                data = []
                for row in results:
                    data.append(dict(zip(column_names, row)))
                
                execution_time = time.time() - start_time
                
                return QueryResult(
                    success=True,
                    data=data,
                    error_message=None,
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"查询执行失败: {e}")
            
            return QueryResult(
                success=False,
                data=[],
                row_count=0,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def execute_query_simple(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        执行简单查询
        
        Args:
            query_params: 查询参数字典
            
        Returns:
            查询结果列表
        """
        query = query_params.get("query", "")
        params = query_params.get("params")
        
        result = self.execute_query(query, params)
        
        if result.success:
            return result.data
        else:
            return []
    
    def validate_query(self, query: str) -> bool:
        """
        验证SQL查询的安全性
        
        Args:
            query: SQL查询语句
            
        Returns:
            是否安全
        """
        # 简单的SQL注入检测
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', '--', ';']
        query_upper = query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                logger.warning(f"检测到危险SQL关键字: {keyword}")
                return False
        
        return True


# 为了避免循环导入，这里重新导入time
import time

