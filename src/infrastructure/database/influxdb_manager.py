"""InfluxDB高级管理功能"""
from typing import Dict, List
from .influxdb_adapter import InfluxDBAdapter
from .influxdb_error_handler import InfluxDBErrorHandler
from influxdb_client.domain.bucket_retention_rules import BucketRetentionRules


class InfluxDBManager(InfluxDBAdapter):
    """InfluxDB高级管理功能"""

    def __init__(self):
        super().__init__()
        self._buckets_api = None
        self._retention_api = None

    def connect(self, config: Dict):
        """扩展连接方法"""
        super().connect(config)
        self._buckets_api = self._client.buckets_api()
        self._retention_api = self._client.buckets_api()

    def create_retention_policy(self, bucket: str, duration: str,
                              shard_duration: str = None) -> bool:

        try:
            bucket = self._buckets_api.find_bucket_by_name(bucket)
            if not bucket:
                raise ValueError(f"Bucket {bucket} not found")

            rule = BucketRetentionRules(every_seconds=self._parse_duration(duration))
            if shard_duration:
                rule.shard_group_duration_seconds = self._parse_duration(shard_duration)

            bucket.retention_rules = [rule]
            self._buckets_api.update_bucket(bucket)
            return True
        except Exception as e:
            print(f"Failed to create retention policy: {str(e)}")
            return False

    def create_continuous_query(self, name: str, source_bucket: str,
                              target_bucket: str, query: str,
                              interval: str) -> bool:
        """创建连续查询"""
        from influxdb_client.domain.continuous_query import ContinuousQuery
        
        try:
            cq = ContinuousQuery(
                name=name,
                query=query,
                every=self._parse_duration(interval),
                source_bucket_id=self._get_bucket_id(source_bucket),
                destination_bucket_id=self._get_bucket_id(target_bucket)
            )
            self._client.create_query_api().create_continuous_query(cq)
            return True
        except Exception as e:
            print(f"Failed to create continuous query: {str(e)}")
            return False
            
    def _parse_duration(self, duration: str) -> int:
        """将时间字符串转换为秒数"""
        units = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800
        }
        value = int(duration[:-1])
        unit = duration[-1]
        return value * units.get(unit, 1)
        
    def _get_bucket_id(self, name: str) -> str:
        """获取bucket ID"""
        bucket = self._buckets_api.find_bucket_by_name(name)
        if not bucket:
            raise ValueError(f"Bucket {name} not found")
        return bucket.id
        
    def optimize_for_high_frequency(self, bucket: str):
        """优化高频数据存储"""
        try:
            # 创建短期保留策略
            self.create_retention_policy(
                bucket=bucket,
                duration="7d",  # 保留7天原始数据
                shard_duration="1d"  # 每日分片
            )
            
            # 创建降采样连续查询
            self.create_continuous_query(
                name=f"downsample_{bucket}_1h",
                source_bucket=bucket,
                target_bucket=f"{bucket}_1h",
                query=f'''
                    from(bucket: "{bucket}")
                      |> range(start: -1h)
                      |> aggregateWindow(every: 1h, fn: mean)
                ''',
                interval="1h"
            )
            
            # 创建长期保留策略
            self.create_retention_policy(
                bucket=f"{bucket}_1h",
                duration="365d"  # 保留1年降采样数据
            )
        except Exception as e:
            self._error_handler.handle_management_error("optimize_for_high_frequency", e)
            raise
