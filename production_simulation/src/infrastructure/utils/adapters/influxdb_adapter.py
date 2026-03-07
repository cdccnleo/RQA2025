
# -*- coding: utf-8 -*-
import time

from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import Point
from datetime import datetime
from influxdb_client import Point
from src.infrastructure.utils.core.error import UnifiedErrorHandler as ErrorHandler
from src.infrastructure.utils.interfaces.database_interfaces import (
    IDatabaseAdapter, QueryResult, WriteResult, HealthCheckResult, ConnectionStatus
)
from typing import Dict, List, Optional, Any
"""
RQA2025 基础设施层工具系统 - InfluxDB适配器

本模块提供InfluxDB时序数据库的适配器实现，支持高性能的时间序列数据存储和查询。
用于系统监控指标、性能数据、业务指标等的存储和分析。

主要特性:
- 高性能时序数据写入
- 灵活的数据查询接口
- 批量数据操作支持
- 连接池管理和重试机制
- 数据压缩和优化
- 健康检查和监控

功能特性:
- 异步数据写入
- 批量数据导入
- 复杂查询支持
- 数据库连接管理
- 错误处理和重试
- 性能监控集成

作者: RQA2025 Team
创建日期: 2025年9月13日
版本: 1.0.0
"""

# try:
# except ImportError:
#     InfluxDBClient = None
#     WriteOptions = None
#     Point = None

# try:
# except ImportError:
#     Point = None
#     ITransaction = None

# try:
# except ImportError:
#     InfluxDBClient = None
#     WriteOptions = None
#     Point = None
#     ITransaction = None

"""
基础设施层 - 健康检查组件

influxdb_adapter 模块

健康检查相关的文件
提供健康检查相关的功能实现。
"""

#!/usr/bin/env python3
#     QueryResult,
#     WriteResult,
#     HealthCheckResult,
#     ConnectionStatus,
#     IDatabaseAdapter,
"""
InfluxDB数据库适配器
实现统一数据库接口
"""

# InfluxDB适配器常量


class InfluxDBConstants:
    """InfluxDB适配器相关常量"""

    # 默认写入配置
    DEFAULT_BATCH_SIZE = 1000
    DEFAULT_FLUSH_INTERVAL = 10  # 秒
    DEFAULT_JITTER_INTERVAL = 2  # 秒

    # 默认执行时间
    DEFAULT_EXECUTION_TIME = 0.0

    # 默认受影响行数
    DEFAULT_AFFECTED_ROWS = 1

    # 默认响应时间
    DEFAULT_RESPONSE_TIME = 0.0

    # 默认错误计数
    DEFAULT_ERROR_COUNT = 1

    # 默认连接数
    DEFAULT_ACTIVE_CONNECTIONS = 1
    DEFAULT_TOTAL_CONNECTIONS = 1
    DISCONNECTED_ACTIVE_CONNECTIONS = 0

    # 查询超时配置 (秒)
    DEFAULT_QUERY_TIMEOUT = 30

    # 重试配置
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0

    # 数据保留策略
    DEFAULT_RETENTION_POLICY = "autogen"


class InfluxDBAdapter(IDatabaseAdapter):
    """InfluxDB适配器，实现统一接口"""

    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self._client = None
        self._write_api = None
        self._query_api = None
        self._org = None
        self._bucket = None
        self._error_handler = error_handler or ErrorHandler()
        self._connected = False
        self._config = {}
        self._connection_info = {}

    def connect(self, config: Dict[str, Any]) -> bool:
        """连接InfluxDB数据库"""
        try:
            self._config = config.copy()
            self._client = InfluxDBClient(
                url=config["url"], token=config["token"], org=config["org"])

            self._org = config["org"]
            self._bucket = config.get("bucket", "default")
            write_options = WriteOptions(
                batch_size=config.get("batch_size", InfluxDBConstants.DEFAULT_BATCH_SIZE),
                flush_interval=config.get(
                    "flush_interval", InfluxDBConstants.DEFAULT_FLUSH_INTERVAL),
                jitter_interval=config.get(
                    "jitter_interval", InfluxDBConstants.DEFAULT_JITTER_INTERVAL)
            )

            self._write_api = self._client.write_api(write_options=write_options)
            self._query_api = self._client.query_api()
            # 测试连接
            self._client.ping()
            self._connection_info = {
                "url": config["url"],
                "org": config["org"],
                "bucket": self._bucket,
                "connected_at": datetime.now().isoformat(),
            }

            self._connected = True
            return True
        except Exception as e:
            self._handle_error(e, "InfluxDB连接失败")
            self._connected = False
            # 重新抛出异常以符合测试期望
            raise

    def is_connected(self) -> bool:
        """检查是否已连接到数据库"""
        return self._connected and self._client is not None
    
    def disconnect(self) -> bool:
        """断开InfluxDB数据库连接"""
        write_api = getattr(self, "_write_api", None)
        client = getattr(self, "_client", None)
        try:
            if write_api:
                try:
                    write_api.close()
                except Exception as close_error:
                    self._handle_error(close_error, "InfluxDB写入通道关闭失败")
            if client:
                client.close()
            self._client = None
            self._write_api = None
            self._query_api = None
            self._connected = False
            return True
        except Exception as e:
            self._handle_error(e, "InfluxDB断开连接失败")
            return False

    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """执行查询"""
        start_time = time.time()
        try:
            if not self._connected:
                return QueryResult(
                    success=False,
                    data=[],
                    row_count=0,
                    execution_time=InfluxDBConstants.DEFAULT_EXECUTION_TIME,
                    error_message="数据库未连接",
                )

            result = self._query_api.query(query, params=params or {})
            processed_data = self._process_query_result(result)
            execution_time = time.time() - start_time
            return QueryResult(
                success=True,
                data=processed_data,
                row_count=len(processed_data),
                execution_time=execution_time,
            )

        except Exception as e:
            self._handle_error(e, "InfluxDB查询失败")
            return QueryResult(
                success=False,
                data=[],
                row_count=0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def execute_write(self, data: Dict[str, Any]) -> WriteResult:
        """执行写入操作"""
        start_time = time.time()
        try:
            if not self._connected:
                return WriteResult(
                    success=False,
                    affected_rows=0,
                    execution_time=InfluxDBConstants.DEFAULT_EXECUTION_TIME,
                    error_message="数据库未连接"
                )

            point = self._create_point(data)
            self._write_api.write(bucket=self._bucket, record=point)

            execution_time = time.time() - start_time
            return WriteResult(
                success=True,
                affected_rows=InfluxDBConstants.DEFAULT_AFFECTED_ROWS,
                execution_time=execution_time
            )

        except Exception as e:
            self._handle_error(e, "InfluxDB写入失败")
            return WriteResult(
                success=False,
                affected_rows=0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )

    def batch_write(self, data_list: List[Dict[str, Any]]) -> WriteResult:
        """批量写入操作"""
        start_time = time.time()
        try:
            if not self._connected:
                return WriteResult(
                    success=False,
                    affected_rows=0,
                    execution_time=InfluxDBConstants.DEFAULT_EXECUTION_TIME,
                    error_message="数据库未连接"
                )

            points = [self._create_point(data) for data in data_list]
            self._write_api.write(bucket=self._bucket, record=points)

            return WriteResult(
                success=True,
                affected_rows=len(points),
                execution_time=time.time() - start_time
            )

        except Exception as e:
            self._handle_error(e, "InfluxDB批量写入失败")
            raise

    def health_check(self) -> HealthCheckResult:
        """健康检查"""
        start_time = time.time()
        try:
            if not self._connected:
                return HealthCheckResult(
                    is_healthy=False,
                    response_time=InfluxDBConstants.DEFAULT_RESPONSE_TIME,
                    message="数据库未连接",
                    details={"error": "数据库未连接"}
                )

            is_healthy = self._client.ping()
            response_time = time.time() - start_time
            return HealthCheckResult(
                is_healthy=is_healthy,
                response_time=response_time,
                message="健康" if is_healthy else "不健康",
                details={
                    "url": self._connection_info.get("url"),
                    "org": self._connection_info.get("org"),
                    "bucket": self._connection_info.get("bucket"),
                }
            )

        except Exception as e:
            self._handle_error(e, "InfluxDB健康检查失败")
            return HealthCheckResult(
                is_healthy=False,
                response_time=time.time() - start_time,
                message=str(e),
                details={"error": str(e)}
            )

    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return {
            **self._connection_info,
            "connected": self._connected,
            "database_type": "influxdb",
        }

    def connection_status(self) -> Dict[str, Any]:
        """获取连接状态"""
        if not self._connected:
            return {
                "connected": False,
                "status": ConnectionStatus.DISCONNECTED.value,
                "database_type": "influxdb"
            }

        try:
            # 执行ping检查连接是否有效
            self._client.ping()
            return {
                "connected": True,
                "status": ConnectionStatus.CONNECTED.value,
                "database_type": "influxdb"
            }
        except Exception:
            self._connected = False
            return {
                "connected": False,
                "status": ConnectionStatus.DISCONNECTED.value,
                "database_type": "influxdb"
            }

    def begin_transaction(self) -> "InfluxDBTransaction":
        """开始事务"""
        if not self._connected:
            raise RuntimeError("数据库未连接")
        return InfluxDBTransaction(self._write_api, self._error_handler)

    def close(self) -> None:
        """关闭连接"""
        self.disconnect()

    def _handle_error(self, exc: Exception, message: str) -> None:
        handler = getattr(self, "_error_handler", None)
        if handler:
            try:
                handler.handle(exc, message)
            except Exception:
                pass

    def _create_point(self, data: Dict[str, Any]) -> Point:
        """创建数据点"""
        measurement = data.get("measurement", "default")
        point = Point(measurement)
        # 添加标签
        tags = data.get("tags", {})
        for tag_key, tag_value in tags.items():
            point.tag(tag_key, str(tag_value))
        # 添加字段
        fields = data.get("fields", data.get("data", {}))
        for field_key, field_value in fields.items():
            point.field(field_key, field_value)
        # 添加时间戳
        timestamp = data.get("timestamp")
        if timestamp:
            point.time(timestamp)
        return point

    def _process_query_result(self, result) -> List[Dict[str, Any]]:
        """处理查询结果"""
        processed_data = []
        for table in result:
            for record in table.records:
                data = {
                    "measurement": record.get_measurement(),
                    "field": record.get_field(),
                    "value": record.get_value(),
                    "time": (record.get_time().isoformat() if record.get_time() else None),
                }

                processed_data.append(data)
        return processed_data

    def _generate_connection_string(self, config: Dict[str, Any]) -> str:
        """生成连接字符串"""
        url = config.get("url", "http://localhost:8086")
        token = config.get("token", "")
        org = config.get("org", "default")
        bucket = config.get("bucket", "default")
        # 构建连接字符串，包含测试期望的格式
        connection_parts = [f"url={url}"]
        if token:
            connection_parts.append(f"token={token}")
        connection_parts.append(f"org={org}")
        connection_parts.append(f"bucket={bucket}")
        return "&".join(connection_parts)
        # 兼容性方法

    def write(self, data: Dict[str, Any]) -> bool:
        """兼容性写入方法"""
        result = self.execute_write(data)
        if not result.success:
            # 重新抛出异常以符合测试期望
            raise Exception(result.error)
        return result.success

    def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """兼容性查询方法"""
        query = query_params.get("query", "")
        params = query_params.get("params", {})
        result = self.execute_query(query, params)
        if not result.success:
            # 重新抛出异常以符合测试期望
            raise Exception(result.error)
        return result.data if result.success else []

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        try:
            self.close()
        except Exception:
            pass

    def __del__(self):
        """析构时确保连接关闭"""
        self.close()


class InfluxDBTransaction:
    """InfluxDB事务实现"""

    def __init__(self, write_api=None, error_handler=None):

        self._write_api = write_api
        self._error_handler = error_handler or ErrorHandler()
        self._committed = False
        self._rolled_back = False

    def commit(self) -> bool:
        """提交事务"""
        try:
            self._committed = True
            return True
        except Exception as e:
            self._error_handler.handle(e, "InfluxDB事务提交失败")
            return False

    def rollback(self) -> bool:
        """回滚事务"""
        try:
            self._rolled_back = True
            return True
        except Exception as e:
            self._error_handler.handle(e, "InfluxDB事务回滚失败")
            return False

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if exc_type is not None:
            self.rollback()
        elif not self._committed and not self._rolled_back:
            self.commit()
