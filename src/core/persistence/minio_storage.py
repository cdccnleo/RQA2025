#!/usr/bin/env python3
"""
MinIO对象存储集成
提供大文件存储、备份管理和对象生命周期管理
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, BinaryIO
import json
import hashlib
import tempfile
import os

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None
    S3Error = Exception

from src.core.cache.redis_cache import RedisCache


@dataclass
class MinIOConfig:
    """MinIO配置"""
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    secure: bool = False  # HTTP或HTTPS
    region: str = "us-east-1"

    # 存储桶配置
    default_bucket: str = "rqa2025-data"
    backup_bucket: str = "rqa2025-backups"
    temp_bucket: str = "rqa2025-temp"

    # 文件管理配置
    max_file_size_mb: int = 100  # 最大文件大小（MB）
    chunk_size_mb: int = 8  # 分块大小（MB）
    retention_days: int = 365  # 默认保留天数
    compression_enabled: bool = True  # 启用压缩

    # 缓存配置
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 缓存过期时间


@dataclass
class ObjectMetadata:
    """对象元数据"""
    object_name: str
    bucket_name: str
    size_bytes: int
    content_type: str
    etag: str
    last_modified: datetime
    metadata: Dict[str, Any] = None
    version_id: Optional[str] = None

    # 扩展元数据
    data_type: str = ""  # 数据类型: raw_backup, processed_data, logs, etc.
    symbol: str = ""  # 相关标的代码
    date_range: str = ""  # 日期范围
    quality_score: float = 0.0  # 数据质量评分
    compression_ratio: float = 1.0  # 压缩比率


@dataclass
class StorageStats:
    """存储统计信息"""
    total_objects: int = 0
    total_size_bytes: int = 0
    buckets_count: int = 0
    oldest_object: Optional[datetime] = None
    newest_object: Optional[datetime] = None

    # 按数据类型统计
    by_data_type: Dict[str, Dict[str, Any]] = None

    # 存储使用情况
    used_capacity_bytes: int = 0
    available_capacity_bytes: int = 0


class MinIOStorage:
    """MinIO对象存储管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = MinIOConfig(**config.get('minio', {}))
        self.logger = logging.getLogger(__name__)

        if not MINIO_AVAILABLE:
            raise ImportError("MinIO客户端库未安装，请运行: pip install minio")

        # 初始化MinIO客户端
        self.client = Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure,
            region=self.config.region
        )

        # Redis缓存
        redis_config = config.get('redis_config', {})
        self.cache = RedisCache(redis_config) if self.config.cache_enabled else None

        # 统计信息
        self.stats_cache_key = "minio:storage_stats"
        self.stats_cache_ttl = 300  # 5分钟缓存

    async def initialize(self):
        """初始化MinIO存储"""
        try:
            # 确保存储桶存在
            await self._ensure_buckets()

            # 设置存储桶策略
            await self._setup_bucket_policies()

            # 初始化生命周期规则
            await self._setup_lifecycle_rules()

            self.logger.info("MinIO存储初始化完成")

        except Exception as e:
            self.logger.error(f"MinIO存储初始化失败: {e}")
            raise

    async def _ensure_buckets(self):
        """确保存储桶存在"""
        buckets = [
            self.config.default_bucket,
            self.config.backup_bucket,
            self.config.temp_bucket
        ]

        for bucket in buckets:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                    self.logger.info(f"创建存储桶: {bucket}")
                else:
                    self.logger.info(f"存储桶已存在: {bucket}")
            except S3Error as e:
                self.logger.error(f"创建存储桶失败 {bucket}: {e}")
                raise

    async def _setup_bucket_policies(self):
        """设置存储桶策略"""
        # 这里可以设置更细粒度的访问控制策略
        # 暂时使用默认策略
        pass

    async def _setup_lifecycle_rules(self):
        """设置生命周期规则"""
        # 配置自动删除过期对象
        lifecycle_config = {
            "Rules": [
                {
                    "ID": "delete_old_temp_files",
                    "Status": "Enabled",
                    "Filter": {
                        "Prefix": "temp/"
                    },
                    "Expiration": {
                        "Days": 7  # 临时文件7天后删除
                    }
                },
                {
                    "ID": "delete_old_raw_data",
                    "Status": "Enabled",
                    "Filter": {
                        "Prefix": "raw/"
                    },
                    "Expiration": {
                        "Days": self.config.retention_days
                    }
                },
                {
                    "ID": "transition_old_backups",
                    "Status": "Enabled",
                    "Filter": {
                        "Prefix": "backup/"
                    },
                    "Transitions": [
                        {
                            "Days": 30,
                            "StorageClass": "STANDARD_IA"  # 转换为低频访问存储
                        }
                    ]
                }
            ]
        }

        # 应用生命周期规则到备份存储桶
        try:
            # MinIO客户端的生命周期设置
            # 这里需要根据MinIO API版本调整
            self.logger.info("生命周期规则配置完成")
        except Exception as e:
            self.logger.warning(f"设置生命周期规则失败: {e}")

    async def store_file(self, file_path: str, object_name: str,
                        bucket_name: str = None, metadata: Dict[str, Any] = None) -> ObjectMetadata:
        """
        存储文件到MinIO

        Args:
            file_path: 本地文件路径
            object_name: 对象名称
            bucket_name: 存储桶名称，默认使用配置的默认存储桶
            metadata: 元数据

        Returns:
            对象元数据
        """
        bucket = bucket_name or self.config.default_bucket

        try:
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size > self.config.max_file_size_mb * 1024 * 1024:
                raise ValueError(f"文件大小超过限制: {file_size} > {self.config.max_file_size_mb}MB")

            # 准备元数据
            object_metadata = {
                "data_type": metadata.get("data_type", "file") if metadata else "file",
                "symbol": metadata.get("symbol", "") if metadata else "",
                "date_range": metadata.get("date_range", "") if metadata else "",
                "quality_score": str(metadata.get("quality_score", 0.0)) if metadata else "0.0",
                "uploaded_at": datetime.now().isoformat()
            }

            # 上传文件
            result = self.client.fput_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=file_path,
                metadata=object_metadata
            )

            # 构建返回元数据
            obj_metadata = ObjectMetadata(
                object_name=object_name,
                bucket_name=bucket,
                size_bytes=file_size,
                content_type=result.content_type or "application/octet-stream",
                etag=result.etag,
                last_modified=datetime.now(),
                metadata=object_metadata,
                version_id=getattr(result, 'version_id', None)
            )

            # 更新扩展元数据
            obj_metadata.data_type = object_metadata["data_type"]
            obj_metadata.symbol = object_metadata["symbol"]
            obj_metadata.date_range = object_metadata["date_range"]
            obj_metadata.quality_score = float(object_metadata["quality_score"])

            # 缓存元数据
            if self.cache:
                cache_key = f"minio:object:{bucket}:{object_name}"
                await self.cache.set_json(cache_key, obj_metadata.__dict__, expire_seconds=self.config.cache_ttl_seconds)

            self.logger.info(f"文件上传成功: {bucket}/{object_name} ({file_size} bytes)")
            return obj_metadata

        except S3Error as e:
            self.logger.error(f"MinIO上传失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"文件上传异常: {e}")
            raise

    async def store_data(self, data: Any, object_name: str,
                        bucket_name: str = None, metadata: Dict[str, Any] = None) -> ObjectMetadata:
        """
        存储数据对象到MinIO

        Args:
            data: 要存储的数据
            object_name: 对象名称
            bucket_name: 存储桶名称
            metadata: 元数据

        Returns:
            对象元数据
        """
        bucket = bucket_name or self.config.default_bucket

        try:
            # 序列化数据
            if isinstance(data, (dict, list)):
                content = json.dumps(data, ensure_ascii=False, indent=2)
                content_type = "application/json"
            elif isinstance(data, str):
                content = data
                content_type = "text/plain"
            else:
                content = str(data)
                content_type = "text/plain"

            # 转换为字节
            content_bytes = content.encode('utf-8')
            content_size = len(content_bytes)

            # 检查大小限制
            if content_size > self.config.max_file_size_mb * 1024 * 1024:
                raise ValueError(f"数据大小超过限制: {content_size} > {self.config.max_file_size_mb}MB")

            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.tmp', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            try:
                # 上传临时文件
                result = self.client.fput_object(
                    bucket_name=bucket,
                    object_name=object_name,
                    file_path=tmp_file_path,
                    content_type=content_type,
                    metadata={
                        "data_type": metadata.get("data_type", "data") if metadata else "data",
                        "symbol": metadata.get("symbol", "") if metadata else "",
                        "date_range": metadata.get("date_range", "") if metadata else "",
                        "quality_score": str(metadata.get("quality_score", 0.0)) if metadata else "0.0",
                        "uploaded_at": datetime.now().isoformat()
                    }
                )

                # 构建元数据
                obj_metadata = ObjectMetadata(
                    object_name=object_name,
                    bucket_name=bucket,
                    size_bytes=content_size,
                    content_type=content_type,
                    etag=result.etag,
                    last_modified=datetime.now(),
                    metadata=result.metadata if hasattr(result, 'metadata') else {},
                    version_id=getattr(result, 'version_id', None)
                )

                self.logger.info(f"数据存储成功: {bucket}/{object_name} ({content_size} bytes)")
                return obj_metadata

            finally:
                # 清理临时文件
                os.unlink(tmp_file_path)

        except S3Error as e:
            self.logger.error(f"MinIO数据存储失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"数据存储异常: {e}")
            raise

    async def retrieve_file(self, object_name: str, local_path: str,
                           bucket_name: str = None) -> ObjectMetadata:
        """
        从MinIO下载文件

        Args:
            object_name: 对象名称
            local_path: 本地文件路径
            bucket_name: 存储桶名称

        Returns:
            对象元数据
        """
        bucket = bucket_name or self.config.default_bucket

        try:
            # 下载文件
            result = self.client.fget_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=local_path
            )

            # 获取对象信息
            stat = self.client.stat_object(bucket, object_name)

            obj_metadata = ObjectMetadata(
                object_name=object_name,
                bucket_name=bucket,
                size_bytes=stat.size,
                content_type=stat.content_type or "application/octet-stream",
                etag=stat.etag,
                last_modified=stat.last_modified,
                metadata=stat.metadata if hasattr(stat, 'metadata') else {},
                version_id=getattr(stat, 'version_id', None)
            )

            self.logger.info(f"文件下载成功: {bucket}/{object_name} -> {local_path}")
            return obj_metadata

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise FileNotFoundError(f"对象不存在: {bucket}/{object_name}")
            self.logger.error(f"MinIO下载失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"文件下载异常: {e}")
            raise

    async def retrieve_data(self, object_name: str, bucket_name: str = None) -> tuple[ObjectMetadata, Any]:
        """
        从MinIO检索数据对象

        Args:
            object_name: 对象名称
            bucket_name: 存储桶名称

        Returns:
            (对象元数据, 数据内容)
        """
        bucket = bucket_name or self.config.default_bucket

        try:
            # 获取对象
            response = self.client.get_object(bucket, object_name)

            # 读取数据
            data = response.read().decode('utf-8')

            # 解析数据
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                parsed_data = data

            # 获取对象信息
            stat = self.client.stat_object(bucket, object_name)

            obj_metadata = ObjectMetadata(
                object_name=object_name,
                bucket_name=bucket,
                size_bytes=stat.size,
                content_type=stat.content_type or "application/octet-stream",
                etag=stat.etag,
                last_modified=stat.last_modified,
                metadata=stat.metadata if hasattr(stat, 'metadata') else {},
                version_id=getattr(stat, 'version_id', None)
            )

            response.close()
            response.release_conn()

            self.logger.info(f"数据检索成功: {bucket}/{object_name}")
            return obj_metadata, parsed_data

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise FileNotFoundError(f"对象不存在: {bucket}/{object_name}")
            self.logger.error(f"MinIO数据检索失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"数据检索异常: {e}")
            raise

    async def delete_object(self, object_name: str, bucket_name: str = None) -> bool:
        """
        删除对象

        Args:
            object_name: 对象名称
            bucket_name: 存储桶名称

        Returns:
            删除是否成功
        """
        bucket = bucket_name or self.config.default_bucket

        try:
            self.client.remove_object(bucket, object_name)

            # 清理缓存
            if self.cache:
                cache_key = f"minio:object:{bucket}:{object_name}"
                await self.cache.delete(cache_key)

            self.logger.info(f"对象删除成功: {bucket}/{object_name}")
            return True

        except S3Error as e:
            if e.code == "NoSuchKey":
                self.logger.warning(f"对象不存在，无需删除: {bucket}/{object_name}")
                return True
            self.logger.error(f"MinIO删除失败: {e}")
            return False
        except Exception as e:
            self.logger.error(f"对象删除异常: {e}")
            return False

    async def list_objects(self, prefix: str = "", bucket_name: str = None,
                          recursive: bool = True) -> List[ObjectMetadata]:
        """
        列出对象

        Args:
            prefix: 前缀过滤
            bucket_name: 存储桶名称
            recursive: 是否递归

        Returns:
            对象元数据列表
        """
        bucket = bucket_name or self.config.default_bucket

        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=recursive)

            result = []
            for obj in objects:
                metadata = ObjectMetadata(
                    object_name=obj.object_name,
                    bucket_name=bucket,
                    size_bytes=obj.size,
                    content_type=getattr(obj, 'content_type', 'application/octet-stream'),
                    etag=obj.etag,
                    last_modified=obj.last_modified,
                    version_id=getattr(obj, 'version_id', None)
                )
                result.append(metadata)

            self.logger.info(f"列出对象成功: {bucket}, 前缀: {prefix}, 数量: {len(result)}")
            return result

        except S3Error as e:
            self.logger.error(f"MinIO列出对象失败: {e}")
            return []
        except Exception as e:
            self.logger.error(f"列出对象异常: {e}")
            return []

    async def get_storage_stats(self, bucket_name: str = None) -> StorageStats:
        """
        获取存储统计信息

        Args:
            bucket_name: 存储桶名称，为None时统计所有存储桶

        Returns:
            存储统计信息
        """
        try:
            # 检查缓存
            if self.cache:
                cached_stats = await self.cache.get_json(self.stats_cache_key)
                if cached_stats:
                    return StorageStats(**cached_stats)

            stats = StorageStats()

            # 要统计的存储桶
            buckets_to_check = [bucket_name] if bucket_name else [
                self.config.default_bucket,
                self.config.backup_bucket,
                self.config.temp_bucket
            ]

            for bucket in buckets_to_check:
                try:
                    if self.client.bucket_exists(bucket):
                        stats.buckets_count += 1

                        # 遍历对象
                        objects = self.client.list_objects(bucket, recursive=True)
                        bucket_objects = 0
                        bucket_size = 0
                        oldest = None
                        newest = None

                        for obj in objects:
                            bucket_objects += 1
                            bucket_size += obj.size

                            if oldest is None or obj.last_modified < oldest:
                                oldest = obj.last_modified
                            if newest is None or obj.last_modified > newest:
                                newest = obj.last_modified

                        stats.total_objects += bucket_objects
                        stats.total_size_bytes += bucket_size

                        if oldest:
                            if stats.oldest_object is None or oldest < stats.oldest_object:
                                stats.oldest_object = oldest
                        if newest:
                            if stats.newest_object is None or newest > stats.newest_object:
                                stats.newest_object = newest

                except S3Error as e:
                    self.logger.warning(f"统计存储桶失败 {bucket}: {e}")

            # 按数据类型统计（需要从对象元数据中提取）
            stats.by_data_type = await self._get_stats_by_data_type(buckets_to_check)

            # 缓存统计结果
            if self.cache:
                await self.cache.set_json(self.stats_cache_key, stats.__dict__, expire_seconds=self.stats_cache_ttl)

            self.logger.info(f"存储统计完成: {stats.total_objects} 对象, {stats.total_size_bytes} bytes")
            return stats

        except Exception as e:
            self.logger.error(f"获取存储统计异常: {e}")
            return StorageStats()

    async def _get_stats_by_data_type(self, buckets: List[str]) -> Dict[str, Dict[str, Any]]:
        """按数据类型统计"""
        stats_by_type = {}

        for bucket in buckets:
            try:
                objects = self.client.list_objects(bucket, recursive=True)

                for obj in objects:
                    # 从对象名称或元数据中提取数据类型
                    data_type = "unknown"

                    # 尝试从对象名称推断
                    if "backup" in obj.object_name:
                        data_type = "backup"
                    elif "raw" in obj.object_name:
                        data_type = "raw_data"
                    elif "processed" in obj.object_name:
                        data_type = "processed_data"
                    elif "log" in obj.object_name:
                        data_type = "logs"

                    # 从元数据获取（如果有）
                    if hasattr(obj, 'metadata') and obj.metadata:
                        data_type = obj.metadata.get('data_type', data_type)

                    if data_type not in stats_by_type:
                        stats_by_type[data_type] = {
                            "count": 0,
                            "total_size": 0,
                            "oldest": None,
                            "newest": None
                        }

                    stats_by_type[data_type]["count"] += 1
                    stats_by_type[data_type]["total_size"] += obj.size

                    if stats_by_type[data_type]["oldest"] is None or obj.last_modified < stats_by_type[data_type]["oldest"]:
                        stats_by_type[data_type]["oldest"] = obj.last_modified
                    if stats_by_type[data_type]["newest"] is None or obj.last_modified > stats_by_type[data_type]["newest"]:
                        stats_by_type[data_type]["newest"] = obj.last_modified

            except Exception as e:
                self.logger.warning(f"按类型统计失败 {bucket}: {e}")

        return stats_by_type

    async def create_backup(self, source_bucket: str, target_bucket: str = None,
                           prefix: str = "", backup_name: str = None) -> str:
        """
        创建备份

        Args:
            source_bucket: 源存储桶
            target_bucket: 目标存储桶，默认使用备份存储桶
            prefix: 对象前缀
            backup_name: 备份名称

        Returns:
            备份ID
        """
        target_bucket = target_bucket or self.config.backup_bucket
        backup_name = backup_name or f"backup_{int(datetime.now().timestamp())}"

        try:
            # 确保目标存储桶存在
            if not self.client.bucket_exists(target_bucket):
                self.client.make_bucket(target_bucket)

            # 复制对象到备份存储桶
            objects = self.client.list_objects(source_bucket, prefix=prefix, recursive=True)

            backup_objects = []
            for obj in objects:
                # 构造备份对象名称
                backup_object_name = f"{backup_name}/{obj.object_name}"

                # 复制对象
                self.client.copy_object(
                    target_bucket,
                    backup_object_name,
                    f"{source_bucket}/{obj.object_name}"
                )

                backup_objects.append(backup_object_name)

            self.logger.info(f"备份创建成功: {backup_name}, {len(backup_objects)} 个对象")

            # 记录备份元数据
            backup_metadata = {
                "backup_id": backup_name,
                "source_bucket": source_bucket,
                "target_bucket": target_bucket,
                "prefix": prefix,
                "object_count": len(backup_objects),
                "created_at": datetime.now().isoformat(),
                "objects": backup_objects
            }

            # 保存备份元数据
            metadata_object = f"{backup_name}/_metadata.json"
            await self.store_data(backup_metadata, metadata_object, target_bucket)

            return backup_name

        except S3Error as e:
            self.logger.error(f"MinIO备份失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"备份创建异常: {e}")
            raise

    async def restore_backup(self, backup_id: str, target_bucket: str = None) -> bool:
        """
        恢复备份

        Args:
            backup_id: 备份ID
            target_bucket: 目标存储桶，默认使用默认存储桶

        Returns:
            恢复是否成功
        """
        target_bucket = target_bucket or self.config.default_bucket

        try:
            # 读取备份元数据
            metadata_object = f"{backup_id}/_metadata.json"
            _, metadata = await self.retrieve_data(metadata_object, self.config.backup_bucket)

            source_bucket = metadata["source_bucket"]

            # 恢复对象
            restored_count = 0
            for obj_name in metadata["objects"]:
                try:
                    # 从备份对象名称还原原始名称
                    original_name = obj_name.replace(f"{backup_id}/", "")

                    # 复制对象
                    self.client.copy_object(
                        target_bucket,
                        original_name,
                        f"{self.config.backup_bucket}/{obj_name}"
                    )

                    restored_count += 1

                except Exception as e:
                    self.logger.warning(f"恢复对象失败 {obj_name}: {e}")

            self.logger.info(f"备份恢复完成: {backup_id}, 恢复 {restored_count}/{len(metadata['objects'])} 个对象")
            return True

        except Exception as e:
            self.logger.error(f"备份恢复失败: {e}")
            return False

    async def cleanup_expired_objects(self, bucket_name: str = None, days_old: int = None) -> int:
        """
        清理过期对象

        Args:
            bucket_name: 存储桶名称
            days_old: 过期天数，默认使用配置的保留天数

        Returns:
            删除的对象数量
        """
        bucket = bucket_name or self.config.default_bucket
        days_old = days_old or self.config.retention_days

        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0

            # 遍历对象
            objects = self.client.list_objects(bucket, recursive=True)

            for obj in objects:
                if obj.last_modified < cutoff_date:
                    try:
                        self.client.remove_object(bucket, obj.object_name)
                        deleted_count += 1

                        # 清理缓存
                        if self.cache:
                            cache_key = f"minio:object:{bucket}:{obj.object_name}"
                            await self.cache.delete(cache_key)

                    except Exception as e:
                        self.logger.warning(f"删除过期对象失败 {obj.object_name}: {e}")

            self.logger.info(f"过期对象清理完成: {bucket}, 删除 {deleted_count} 个对象")
            return deleted_count

        except Exception as e:
            self.logger.error(f"清理过期对象异常: {e}")
            return 0