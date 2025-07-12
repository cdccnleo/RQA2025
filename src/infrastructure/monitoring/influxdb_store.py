from typing import Dict, List, Optional
import logging
from datetime import datetime
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

logger = logging.getLogger(__name__)

class InfluxDBStore:
    """InfluxDB监控数据存储"""

    def __init__(
        self,
        url: str,
        token: str,
        org: str,
        bucket: str,
        batch_size: int = 1000
    ):
        """
        初始化InfluxDB存储

        Args:
            url: InfluxDB服务器URL
            token: 认证token
            org: 组织名称
            bucket: 存储bucket
            batch_size: 批量写入大小
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.batch_size = batch_size

        # 初始化客户端
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

        # 批量写入缓冲区
        self.batch_buffer: List[Dict] = []

    def write_metric(self, measurement: str, fields: Dict, tags: Optional[Dict] = None, timestamp: Optional[datetime] = None):
        """
        写入监控指标

        Args:
            measurement: 指标名称
            fields: 指标字段
            tags: 指标标签
            timestamp: 时间戳
        """
        point = {
            "measurement": measurement,
            "tags": tags or {},
            "fields": fields,
            "time": timestamp or datetime.utcnow()
        }
        self.batch_buffer.append(point)

        # 达到批量大小时触发写入
        if len(self.batch_buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        """将缓冲区数据写入InfluxDB"""
        if not self.batch_buffer:
            return

        try:
            self.write_api.write(
                bucket=self.bucket,
                record=self.batch_buffer,
                write_precision="ms"
            )
            logger.debug(f"Successfully wrote {len(self.batch_buffer)} metrics to InfluxDB")
            self.batch_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to write metrics to InfluxDB: {e}")
            raise

    def query(self, query: str):
        """
        查询监控数据

        Args:
            query: Flux查询语句

        Returns:
            List: 查询结果
        """
        try:
            query_api = self.client.query_api()
            result = query_api.query(query)
            return result
        except Exception as e:
            logger.error(f"Failed to query InfluxDB: {e}")
            raise

    def close(self):
        """关闭连接"""
        self.flush()
        self.client.close()
        logger.info("InfluxDB connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
