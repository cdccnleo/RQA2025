"""
Database Integration Module
数据库集成模块

This module provides database integration capabilities for quantitative trading systems
此模块为量化交易系统提供数据库集成能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)


class DatabaseType(Enum):

    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    SQLSERVER = "sqlserver"
    ORACLE = "oracle"


class QueryType(Enum):

    """Query operation types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    BULK_INSERT = "bulk_insert"


@dataclass
class DatabaseConnection:

    """
    Database connection configuration
    数据库连接配置
    """
    connection_id: str
    database_type: str
    host: str
    port: int
    database: str
    username: str
    password: str
    connection_pool_size: int = 5
    max_overflow: int = 10
    connection_timeout: int = 30
    read_only: bool = False

    def get_connection_string(self) -> str:
        """Get database connection string"""
        if self.database_type == DatabaseType.POSTGRESQL.value:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.database_type == DatabaseType.MYSQL.value:
            return f"mysql + ymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.database_type == DatabaseType.SQLITE.value:
            return f"sqlite:///{self.database}"
        elif self.database_type == DatabaseType.SQLSERVER.value:
            return f"mssql + yodbc://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif self.database_type == DatabaseType.ORACLE.value:
            return f"oracle + x_oracle://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")


@dataclass
class DatabaseQuery:

    """
    Database query data class
    数据库查询数据类
    """
    query_id: str
    sql: str
    parameters: Optional[Dict[str, Any]] = None
    query_type: str = QueryType.SELECT.value
    timeout: int = 30
    retry_count: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class QueryResult:

    """
    Query result data class
    查询结果数据类
    """
    query_id: str
    success: bool
    execution_time: float
    row_count: int = 0
    data: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ConnectionPoolManager:

    """
    Connection Pool Manager Class
    连接池管理器类

    Manages database connection pools
    管理数据库连接池
    """

    def __init__(self):
        """
        Initialize connection pool manager
        初始化连接池管理器
        """
        self.pools: Dict[str, sa.engine.Engine] = {}
        self.sessions: Dict[str, sessionmaker] = {}

    def create_pool(self, connection: DatabaseConnection) -> None:
        """
        Create database connection pool
        创建数据库连接池

        Args:
            connection: Database connection configuration
                       数据库连接配置
        """
        engine = create_engine(
            connection.get_connection_string(),
            poolclass=QueuePool,
            pool_size=connection.connection_pool_size,
            max_overflow=connection.max_overflow,
            pool_timeout=connection.connection_timeout,
            pool_pre_ping=True  # Enable connection health checks
        )

        self.pools[connection.connection_id] = engine
        self.sessions[connection.connection_id] = sessionmaker(bind=engine)

        logger.info(f"Created connection pool for {connection.connection_id}")

    def get_session(self, connection_id: str) -> Session:
        """
        Get database session
        获取数据库会话

        Args:
            connection_id: Connection identifier
                          连接标识符

        Returns:
            Session: Database session
                    数据库会话
        """
        if connection_id not in self.sessions:
            raise ValueError(f"Connection pool {connection_id} not found")

        return self.sessions[connection_id]()

    def close_pool(self, connection_id: str) -> None:
        """
        Close database connection pool
        关闭数据库连接池

        Args:
            connection_id: Connection identifier
                          连接标识符
        """
        if connection_id in self.pools:
            self.pools[connection_id].dispose()
            del self.pools[connection_id]
            del self.sessions[connection_id]
            logger.info(f"Closed connection pool for {connection_id}")


class QueryExecutor:

    """
    Query Executor Class
    查询执行器类

    Executes database queries with error handling and retry logic
    使用错误处理和重试逻辑执行数据库查询
    """

    def __init__(self, pool_manager: ConnectionPoolManager):
        """
        Initialize query executor
        初始化查询执行器

        Args:
            pool_manager: Connection pool manager
                         连接池管理器
        """
        self.pool_manager = pool_manager
        self.query_history: List[Dict[str, Any]] = []

    def execute_query(self, connection_id: str, query: DatabaseQuery) -> QueryResult:
        """
        Execute database query
        执行数据库查询

        Args:
            connection_id: Connection identifier
                          连接标识符
            query: Database query
                  数据库查询

        Returns:
            QueryResult: Query execution result
                        查询执行结果
        """
        start_time = datetime.now()

        # Record query attempt
        query_record = {
            'query_id': query.query_id,
            'connection_id': connection_id,
            'query_type': query.query_type,
            'start_time': start_time,
            'attempts': 0,
            'success': False
        }

        last_exception = None

        # Execute with retry logic
        for attempt in range(query.retry_count + 1):
            query_record['attempts'] = attempt + 1

            try:
                session = self.pool_manager.get_session(connection_id)

                # Execute query based on type
                if query.query_type == QueryType.SELECT.value:
                    result = self._execute_select(session, query)
                elif query.query_type in [QueryType.INSERT.value, QueryType.UPDATE.value, QueryType.DELETE.value]:
                    result = self._execute_modify(session, query)
                elif query.query_type == QueryType.BULK_INSERT.value:
                    result = self._execute_bulk_insert(session, query)
                else:
                    raise ValueError(f"Unsupported query type: {query.query_type}")

                session.close()

                execution_time = (datetime.now() - start_time).total_seconds()

                query_record.update({
                    'success': True,
                    'execution_time': execution_time,
                    'end_time': datetime.now()
                })

                self.query_history.append(query_record)

                return QueryResult(
                    query_id=query.query_id,
                    success=True,
                    execution_time=execution_time,
                    row_count=result.get('row_count', 0),
                    data=result.get('data'),
                    metadata=result.get('metadata', {})
                )

            except Exception as e:
                session.close()
                last_exception = e

                if attempt < query.retry_count:
                    # Exponential backoff
                    import time
                    time.sleep(2 ** attempt)
                    continue

        # All retries failed
        execution_time = (datetime.now() - start_time).total_seconds()

        query_record.update({
            'success': False,
            'execution_time': execution_time,
            'error': str(last_exception),
            'end_time': datetime.now()
        })

        self.query_history.append(query_record)

        return QueryResult(
            query_id=query.query_id,
            success=False,
            execution_time=execution_time,
            error_message=str(last_exception)
        )

    def _execute_select(self, session: Session, query: DatabaseQuery) -> Dict[str, Any]:
        """
        Execute SELECT query
        执行SELECT查询

        Args:
            session: Database session
                   数据库会话
            query: Database query
                  数据库查询

        Returns:
            dict: Query result
                  查询结果
        """
        result = session.execute(text(query.sql), query.parameters or {})

        # Convert to DataFrame
        data = pd.DataFrame(result.fetchall(), columns=result.keys())
        row_count = len(data)

        return {
            'row_count': row_count,
            'data': data,
            'metadata': {
                'columns': list(data.columns),
                'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()}
            }
        }

    def _execute_modify(self, session: Session, query: DatabaseQuery) -> Dict[str, Any]:
        """
        Execute INSERT / UPDATE / DELETE query
        执行INSERT / UPDATE / DELETE查询

        Args:
            session: Database session
                   数据库会话
            query: Database query
                  数据库查询

        Returns:
            dict: Query result
                  查询结果
        """
        result = session.execute(text(query.sql), query.parameters or {})
        session.commit()

        row_count = result.rowcount

        return {
            'row_count': row_count,
            'metadata': {
                'operation': query.query_type,
                'affected_rows': row_count
            }
        }

    def _execute_bulk_insert(self, session: Session, query: DatabaseQuery) -> Dict[str, Any]:
        """
        Execute bulk insert operation
        执行批量插入操作

        Args:
            session: Database session
                   数据库会话
            query: Database query
                  数据库查询

        Returns:
            dict: Bulk insert result
                  批量插入结果
        """
        # For bulk insert, query.sql should be table name
        # and parameters should contain the data
        data = query.parameters.get('data', [])

        if not data:
            return {'row_count': 0, 'message': 'No data provided for bulk insert'}

        # Convert data to DataFrame for bulk insert
        df = pd.DataFrame(data)
        df.to_sql(query.sql, session.bind, if_exists='append', index=False)

        return {
            'row_count': len(data),
            'metadata': {
                'operation': 'bulk_insert',
                'table': query.sql,
                'inserted_rows': len(data)
            }
        }


class DatabaseIntegrationManager:

    """
    Database Integration Manager Class
    数据库集成管理器类

    Manages database connections and operations
    管理数据库连接和操作
    """

    def __init__(self, manager_name: str = "default_database_integration_manager"):
        """
        Initialize database integration manager
        初始化数据库集成管理器

        Args:
            manager_name: Name of the manager
                        管理器名称
        """
        self.manager_name = manager_name
        self.connections: Dict[str, DatabaseConnection] = {}
        self.pool_manager = ConnectionPoolManager()
        self.query_executor = QueryExecutor(self.pool_manager)

        # Statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_query_time': 0.0,
            'active_connections': 0
        }

        logger.info(f"Database integration manager {manager_name} initialized")

    def add_connection(self, connection: DatabaseConnection) -> None:
        """
        Add database connection
        添加数据库连接

        Args:
            connection: Database connection configuration
                       数据库连接配置
        """
        self.connections[connection.connection_id] = connection
        self.pool_manager.create_pool(connection)
        self.stats['active_connections'] = len(self.connections)

        logger.info(f"Added database connection: {connection.connection_id}")

    def remove_connection(self, connection_id: str) -> bool:
        """
        Remove database connection
        移除数据库连接

        Args:
            connection_id: Connection identifier
                          连接标识符

        Returns:
            bool: True if removed successfully
                  移除成功返回True
        """
        if connection_id in self.connections:
            self.pool_manager.close_pool(connection_id)
            del self.connections[connection_id]
            self.stats['active_connections'] = len(self.connections)
            logger.info(f"Removed database connection: {connection_id}")
            return True
        return False

    def execute_query(self,


                      connection_id: str,
                      sql: str,
                      parameters: Optional[Dict[str, Any]] = None,
                      query_type: QueryType = QueryType.SELECT,
                      timeout: int = 30) -> QueryResult:
        """
        Execute database query
        执行数据库查询

        Args:
            connection_id: Connection identifier
                          连接标识符
            sql: SQL query string
                SQL查询字符串
            parameters: Query parameters
                       查询参数
            query_type: Type of query
                       查询类型
            timeout: Query timeout in seconds
                    查询超时时间（秒）

        Returns:
            QueryResult: Query execution result
                        查询执行结果
        """
        if connection_id not in self.connections:
            return QueryResult(
                query_id=f"error_{datetime.now().strftime('%Y % m % d_ % H % M % S')}",
                success=False,
                execution_time=0.0,
                error_message=f"Connection {connection_id} not found"
            )

        query_id = f"query_{connection_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S_ % f')}"

        query = DatabaseQuery(
            query_id=query_id,
            sql=sql,
            parameters=parameters,
            query_type=query_type.value,
            timeout=timeout
        )

        result = self.query_executor.execute_query(connection_id, query)

        # Update statistics
        self._update_stats(result)

        return result

    def bulk_insert(self,


                    connection_id: str,
                    table_name: str,
                    data: List[Dict[str, Any]]) -> QueryResult:
        """
        Perform bulk insert operation
        执行批量插入操作

        Args:
            connection_id: Connection identifier
                          连接标识符
            table_name: Target table name
                       目标表名
            data: Data to insert
                 要插入的数据

        Returns:
            QueryResult: Bulk insert result
                        批量插入结果
        """
        query_id = f"bulk_{connection_id}_{datetime.now().strftime('%Y % m % d_ % H % M % S_ % f')}"

        query = DatabaseQuery(
            query_id=query_id,
            sql=table_name,
            parameters={'data': data},
            query_type=QueryType.BULK_INSERT.value
        )

        result = self.query_executor.execute_query(connection_id, query)

        # Update statistics
        self._update_stats(result)

        return result

    def get_connection_status(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get database connection status
        获取数据库连接状态

        Args:
            connection_id: Connection identifier
                          连接标识符

        Returns:
            dict: Connection status or None
                  连接状态或None
        """
        if connection_id not in self.connections:
            return None

        connection = self.connections[connection_id]

        # Test connection
        try:
            session = self.pool_manager.get_session(connection_id)
            session.execute(text("SELECT 1"))
            session.close()
            status = "connected"
        except Exception as e:
            status = f"error: {str(e)}"

        return {
            'connection_id': connection_id,
            'database_type': connection.database_type,
            'host': connection.host,
            'database': connection.database,
            'status': status,
            'read_only': connection.read_only
        }

    def list_connections(self) -> List[Dict[str, Any]]:
        """
        List all database connections
        列出所有数据库连接

        Returns:
            list: List of connection information
                  连接信息列表
        """
        connections = []

        for connection_id, connection in self.connections.items():
            status_info = self.get_connection_status(connection_id)
            if status_info:
                connections.append(status_info)

        return connections

    def get_table_schema(self, connection_id: str, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get table schema information
        获取表架构信息

        Args:
            connection_id: Connection identifier
                          连接标识符
            table_name: Table name
                       表名

        Returns:
            dict: Table schema or None
                  表架构或None
        """
        if connection_id not in self.connections:
            return None

        try:
            # Get column information
            query_result = self.execute_query(
                connection_id,
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
                """,
                {'table_name': table_name}
            )

            if not query_result.success or query_result.data is None:
                return None

            columns = []
            for _, row in query_result.data.iterrows():
                columns.append({
                    'name': row['column_name'],
                    'type': row['data_type'],
                    'nullable': row['is_nullable'] == 'YES',
                    'default': row['column_default']
                })

            return {
                'table_name': table_name,
                'columns': columns,
                'column_count': len(columns)
            }

        except Exception as e:
            logger.error(f"Failed to get table schema: {str(e)}")
            return None

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database integration statistics
        获取数据库集成统计信息

        Returns:
            dict: Database statistics
                  数据库统计信息
        """
        return {
            'manager_name': self.manager_name,
            'total_connections': len(self.connections),
            'active_connections': sum(1 for c in self.connections.values()
                                      if self.get_connection_status(c.connection_id)
                                      and self.get_connection_status(c.connection_id)['status'] == 'connected'),
            'stats': self.stats
        }

    def _update_stats(self, result: QueryResult) -> None:
        """
        Update integration statistics
        更新集成统计信息

        Args:
            result: Query result
                   查询结果
        """
        self.stats['total_queries'] += 1

        if result.success:
            self.stats['successful_queries'] += 1
        else:
            self.stats['failed_queries'] += 1

        # Update average query time
        total_queries = self.stats['total_queries']
        current_avg = self.stats['average_query_time']
        self.stats['average_query_time'] = (
            (current_avg * (total_queries - 1)) + result.execution_time
        ) / total_queries


class DataReplicationManager:

    """
    Data Replication Manager Class
    数据复制管理器类

    Manages data replication between databases
    管理数据库间的数据复制
    """

    def __init__(self, db_manager: DatabaseIntegrationManager):
        """
        Initialize data replication manager
        初始化数据复制管理器

        Args:
            db_manager: Database integration manager
                       数据库集成管理器
        """
        self.db_manager = db_manager
        self.replication_configs: Dict[str, Dict[str, Any]] = {}

    def configure_replication(self,


                              config_id: str,
                              source_connection: str,
                              target_connection: str,
                              tables: List[str],
                              replication_type: str = "full") -> None:
        """
        Configure data replication
        配置数据复制

        Args:
            config_id: Replication configuration ID
                      复制配置ID
            source_connection: Source connection ID
                             源连接ID
            target_connection: Target connection ID
                             目标连接ID
            tables: Tables to replicate
                   要复制的表
            replication_type: Type of replication ('full', 'incremental')
                             复制类型
        """
        self.replication_configs[config_id] = {
            'source_connection': source_connection,
            'target_connection': target_connection,
            'tables': tables,
            'replication_type': replication_type,
            'last_sync': None,
            'status': 'configured'
        }

        logger.info(f"Configured replication: {config_id}")

    def execute_replication(self, config_id: str) -> Dict[str, Any]:
        """
        Execute data replication
        执行数据复制

        Args:
            config_id: Replication configuration ID
                      复制配置ID

        Returns:
            dict: Replication result
                  复制结果
        """
        if config_id not in self.replication_configs:
            return {'success': False, 'error': f'Replication config {config_id} not found'}

        config = self.replication_configs[config_id]

        try:
            total_records = 0
            success_count = 0

            for table in config['tables']:
                # Get data from source
                source_result = self.db_manager.execute_query(
                    config['source_connection'],
                    f"SELECT * FROM {table}"
                )

                if source_result.success and source_result.data is not None:
                    # Insert into target
                    target_result = self.db_manager.bulk_insert(
                        config['target_connection'],
                        table,
                        source_result.data.to_dict('records')
                    )

                    if target_result.success:
                        success_count += 1
                        total_records += target_result.row_count

            success = success_count == len(config['tables'])

            if success:
                config['last_sync'] = datetime.now()
                config['status'] = 'success'

            return {
                'success': success,
                'tables_processed': len(config['tables']),
                'successful_tables': success_count,
                'total_records': total_records
            }

        except Exception as e:
            config['status'] = 'error'
            return {'success': False, 'error': str(e)}


# Global database integration manager instance
# 全局数据库集成管理器实例
database_integration_manager = DatabaseIntegrationManager()

# Global data replication manager instance
# 全局数据复制管理器实例
data_replication_manager = DataReplicationManager(database_integration_manager)

__all__ = [
    'DatabaseType',
    'QueryType',
    'DatabaseConnection',
    'DatabaseQuery',
    'QueryResult',
    'ConnectionPoolManager',
    'QueryExecutor',
    'DatabaseIntegrationManager',
    'DataReplicationManager',
    'database_integration_manager',
    'data_replication_manager'
]
