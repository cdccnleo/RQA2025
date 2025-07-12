"""InfluxDB数据库适配器实现"""
from typing import Dict, List, Optional
from influxdb_client import InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from . import DatabaseAdapter
from .influxdb_error_handler import InfluxDBErrorHandler
from .. import config
from ..error.error_handler import ErrorHandler


class InfluxDBAdapter(DatabaseAdapter):
    """InfluxDB适配器实现"""

    def __init__(self):
        self._client = None
        self._write_api = None
        self._query_api = None
        self._error_handler = InfluxDBErrorHandler(ErrorHandler())

    def connect(self, config: Dict):
        """连接InfluxDB数据库"""
        self._client = InfluxDBClient(
            url=config['url'],
            token=config['token'],
            org=config['org']
        )
        # 配置批量写入选项
        write_options = WriteOptions(
            batch_size=config.get('batch_size', 1000),
            flush_interval=config.get('flush_interval', 10),
            jitter_interval=config.get('jitter_interval', 2)
        )
        self._write_api = self._client.write_api(write_options=write_options)
        self._query_api = self._client.query_api()

    def write(self, measurement: str, data: Dict, tags: Optional[Dict] = None):
        """写入时间序列数据"""
        from influxdb_client import Point
        point = Point(measurement)

        # 添加标签
        if tags:
            for tag_key, tag_value in tags.items():
                point.tag(tag_key, str(tag_value))

        # 添加字段
        for field_key, field_value in data.items():
            point.field(field_key, field_value)

        self._write_api.write(
            bucket=config.get('bucket', 'default'),
            record=point
        )

    def batch_write(self, points: List):
        """批量写入数据点"""
        self._write_api.write(
            bucket=config.get('bucket', 'default'),
            record=points
        )

    def query(self, query: str) -> List[Dict]:
        """执行Flux查询"""
        # 使用实例方法装饰器
        @self._error_handler.retry_on_exception
        def _query():
            result = self._query_api.query(query)
            return [
                {
                    'measurement': record.get_measurement(),
                    'time': record.get_time(),
                    'values': record.values
                }
                for record in result
            ]
        
        try:
            return _query()
        except Exception as e:
            self._error_handler.handle_query_error("query", e)
            raise

    def close(self):
        """关闭连接"""
        if self._write_api:
            self._write_api.close()
        if self._client:
            self._client.close()

    def __del__(self):
        """析构时确保连接关闭"""
        self.close()
