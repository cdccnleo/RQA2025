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
        self._error_handler = InfluxDBErrorHandler(None)  # 临时创建，实际使用时需要传入真实的ErrorHandler

    def connect(self, config: Dict):
        """扩展连接方法"""
        super().connect(config)
        self._buckets_api = self._client.buckets_api()
        self._retention_api = self._client.buckets_api()

    def create_retention_policy(self, bucket: str, duration: str,
                              shard_duration: str = None) -> bool:
        """创建保留策略"""
        try:
            bucket_obj = self._buckets_api.find_bucket_by_name(bucket)
            if not bucket_obj:
                raise ValueError(f"Bucket {bucket} not found")

            rule = BucketRetentionRules(every_seconds=self._parse_duration(duration))
            if shard_duration:
                rule.shard_group_duration_seconds = self._parse_duration(shard_duration)

            bucket_obj.retention_rules = [rule]
            self._buckets_api.update_bucket(bucket_obj)
            return True
        except Exception as e:
            print(f"Failed to create retention policy: {str(e)}")
            return False

    def create_continuous_query(self, name: str, query: str,
                              destination_bucket: str = None, org: str = None) -> bool:
        """创建连续查询"""
        from influxdb_client.domain.continuous_query import ContinuousQuery
        
        try:
            # 解析查询以获取源bucket和目标bucket
            source_bucket = self._extract_source_bucket(query)
            target_bucket = destination_bucket or f"{source_bucket}_downsampled"
            
            cq = ContinuousQuery(
                name=name,
                query=query.strip(),
                every=self._parse_duration("1h"),  # 默认1小时间隔
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
        if not duration:
            raise ValueError("Duration cannot be empty")
            
        units = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800
        }
        
        if len(duration) < 2:
            raise ValueError("Invalid duration format")
            
        try:
            value = int(duration[:-1])
            unit = duration[-1]
            return value * units.get(unit, 1)
        except (ValueError, IndexError):
            raise ValueError(f"Invalid duration format: {duration}")
        
    def _get_bucket_id(self, name: str) -> str:
        """获取bucket ID"""
        bucket = self._buckets_api.find_bucket_by_name(name)
        if not bucket:
            raise ValueError(f"Bucket '{name}' not found")
        return bucket.id
        
    def _extract_source_bucket(self, query: str) -> str:
        """从查询中提取源bucket名称"""
        # 简单的解析逻辑，实际应用中可能需要更复杂的解析
        import re
        match = re.search(r'from\(bucket:\s*"([^"]+)"', query)
        if match:
            return match.group(1)
        else:
            # 默认返回一个bucket名称
            return "default_bucket"
        
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
                query=f'''
                    from(bucket: "{bucket}")
                      |> range(start: -1h)
                      |> aggregateWindow(every: 1h, fn: mean)
                ''',
                destination_bucket=f"{bucket}_1h"
            )
            
            # 创建长期保留策略
            self.create_retention_policy(
                bucket=f"{bucket}_1h",
                duration="365d"  # 保留1年降采样数据
            )
        except Exception as e:
            if hasattr(self, '_error_handler'):
                self._error_handler.handle_management_error("optimize_for_high_frequency", e)
            raise
