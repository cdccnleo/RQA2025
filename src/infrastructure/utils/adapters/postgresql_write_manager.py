"""
PostgreSQL写入操作管理器组件

负责PostgreSQL数据库的写入、更新、删除操作。
"""

import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

try:
    from src.infrastructure.utils.interfaces.database_interfaces import WriteResult
except ImportError:
    from dataclasses import dataclass
    
    @dataclass
    class WriteResult:
        success: bool
        affected_rows: int
        execution_time: float
        error_message: Optional[str] = None
        error: Optional[str] = None

        def __post_init__(self):
            if self.error is None and self.error_message is not None:
                self.error = self.error_message
            elif self.error is not None and self.error_message is None:
                self.error_message = self.error


class PostgreSQLWriteManager:
    """PostgreSQL写入操作管理器"""
    
    def __init__(self, client=None):
        """
        初始化写入管理器
        
        Args:
            client: psycopg2连接客户端
        """
        self.client = client

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def set_client(self, client) -> None:
        """设置数据库客户端"""
        self.client = client
    
    def execute_write(self, data: Dict[str, Any]) -> WriteResult:
        """
        执行写入操作
        
        Args:
            data: 写入数据，包含operation, table, values等
            
        Returns:
            写入结果
        """
        if not self.client:
            return self._build_result(False, 0, "数据库未连接", 0.0)
        
        operation = data.get("operation", "insert")
        
        if operation == "insert":
            return self._execute_insert(data)
        elif operation == "update":
            return self._execute_update(data)
        elif operation == "delete":
            return self._execute_delete(data)
        else:
            return self._build_result(False, 0, f"不支持的操作类型: {operation}", 0.0)
    
    def batch_write(self, data_list: List[Dict[str, Any]]) -> WriteResult:
        """
        批量写入
        
        Args:
            data_list: 写入数据列表
            
        Returns:
            批量写入结果
        """
        total_affected = 0
        start_time = time.time()
        
        try:
            if not self.client:
                return self._build_result(False, 0, "数据库未连接", 0.0)

            with self.client.cursor() as cursor:
                for data in data_list:
                    result = self._execute_write_with_cursor(data, cursor)
                    if result.success:
                        total_affected += result.affected_rows
                    else:
                        # 如果某个写入失败，回滚整个批次
                        self.client.rollback()
                        return self._build_result(
                            False,
                            0,
                            result.error,
                            time.time() - start_time,
                        )
                
                # 提交批量操作
                self.client.commit()
                
                return self._build_result(
                    True,
                    total_affected,
                    None,
                    time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"批量写入失败: {e}")
            try:
                self.client.rollback()
            except:
                pass
            
            return self._build_result(
                False,
                0,
                str(e),
                time.time() - start_time,
            )
    
    def _execute_insert(self, data: Dict[str, Any]) -> WriteResult:
        """执行插入操作"""
        start_time = time.time()
        
        try:
            with self.client.cursor() as cursor:
                table = data.get("table", "")
                values = data.get("values", {})
                
                # 构建INSERT语句
                columns = ", ".join(values.keys())
                placeholders = ", ".join([f"%({k})s" for k in values.keys()])
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                
                cursor.execute(query, values)
                self.client.commit()
                
                return self._build_result(
                    True,
                    cursor.rowcount,
                    None,
                    time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"插入操作失败: {e}")
            try:
                self.client.rollback()
            except:
                pass
            
            return self._build_result(
                False,
                0,
                str(e),
                time.time() - start_time,
            )
    
    def _execute_update(self, data: Dict[str, Any]) -> WriteResult:
        """执行更新操作"""
        start_time = time.time()
        
        try:
            with self.client.cursor() as cursor:
                table = data.get("table", "")
                values = data.get("values", {})
                conditions = data.get("where")
                if conditions is None:
                    conditions = data.get("conditions", {})
                
                # 构建UPDATE语句
                set_clause = ", ".join([f"{k} = %({k})s" for k in values.keys()])
                where_clause = " AND ".join([f"{k} = %({k})s" for k in conditions.keys()])
                query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
                
                # 合并参数
                all_params = {**values, **conditions}
                
                cursor.execute(query, all_params)
                self.client.commit()
                
                return self._build_result(
                    True,
                    cursor.rowcount,
                    None,
                    time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"更新操作失败: {e}")
            try:
                self.client.rollback()
            except:
                pass
            
            return self._build_result(
                False,
                0,
                str(e),
                time.time() - start_time,
            )
    
    def _execute_delete(self, data: Dict[str, Any]) -> WriteResult:
        """执行删除操作"""
        start_time = time.time()
        
        try:
            with self.client.cursor() as cursor:
                table = data.get("table", "")
                conditions = data.get("where")
                if conditions is None:
                    conditions = data.get("conditions", {})
                
                # 构建DELETE语句
                where_clause = " AND ".join([f"{k} = %({k})s" for k in conditions.keys()])
                query = f"DELETE FROM {table} WHERE {where_clause}"
                
                cursor.execute(query, conditions)
                self.client.commit()
                
                return self._build_result(
                    True,
                    cursor.rowcount,
                    None,
                    time.time() - start_time,
                )
                
        except Exception as e:
            logger.error(f"删除操作失败: {e}")
            try:
                self.client.rollback()
            except:
                pass
            
            return self._build_result(
                False,
                0,
                str(e),
                time.time() - start_time,
            )
    
    def _execute_write_with_cursor(self, data: Dict[str, Any], cursor) -> WriteResult:
        """使用指定游标执行写入操作（用于批量操作）"""
        operation = data.get("operation", "insert")
        start_time = time.time()
        
        try:
            if operation == "insert":
                table = data.get("table", "")
                values = data.get("values", {})
                columns = ", ".join(values.keys())
                placeholders = ", ".join([f"%({k})s" for k in values.keys()])
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                cursor.execute(query, values)
            elif operation == "update":
                table = data.get("table", "")
                values = data.get("values", {})
                conditions = data.get("where")
                if conditions is None:
                    conditions = data.get("conditions", {})
                set_clause = ", ".join([f"{k} = %({k})s" for k in values.keys()])
                where_clause = " AND ".join([f"{k} = %({k})s" for k in conditions.keys()])
                query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
                all_params = {**values, **conditions}
                cursor.execute(query, all_params)
            elif operation == "delete":
                table = data.get("table", "")
                conditions = data.get("where")
                if conditions is None:
                    conditions = data.get("conditions", {})
                where_clause = " AND ".join([f"{k} = %({k})s" for k in conditions.keys()])
                query = f"DELETE FROM {table} WHERE {where_clause}"
                cursor.execute(query, conditions)
            
            return self._build_result(
                True,
                cursor.rowcount,
                None,
                time.time() - start_time,
            )
            
        except Exception as e:
            return self._build_result(
                False,
                0,
                str(e),
                time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _build_result(success: bool, affected: int, error: Optional[str], execution_time: float) -> WriteResult:
        return WriteResult(
            success=success,
            affected_rows=affected,
            execution_time=execution_time,
            error=error,
            error_message=error,
        )

