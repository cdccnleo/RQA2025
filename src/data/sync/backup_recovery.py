#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据备份恢复模块
提供数据备份、恢复、验证和监控功能
"""

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging


    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

import os
import shutil
import json
import hashlib
import zipfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from src.infrastructure.logging import get_infrastructure_logger

logger = get_infrastructure_logger(__name__)


@dataclass

class BackupConfig:

    """备份配置"""
    backup_dir: str = "./backups"
    max_backups: int = 30
    compression: bool = True
    verify_backup: bool = True
    auto_cleanup: bool = True
    backup_interval: int = 3600  # 秒
    retention_days: int = 30


@dataclass

class BackupInfo:

    """备份信息"""
    backup_id: str
    timestamp: datetime
    size: int
    checksum: str
    data_types: List[str]
    status: str  # created, verified, failed
    metadata: Dict[str, Any]


class DataBackupRecovery:

    """数据备份恢复管理器"""


    def __init__(self, config: Optional[BackupConfig] = None):

        """
        初始化备份恢复管理器

        Args:
            config: 备份配置
        """
        self.config = config or BackupConfig()
        self.backup_dir = Path(self.config.backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.backups: Dict[str, BackupInfo] = {}
        self.lock = threading.RLock()
        self._load_backup_index()

        logger.info(f"数据备份恢复管理器初始化完成，备份目录: {self.backup_dir}")


    def _load_backup_index(self):

        """加载备份索引"""
        index_file = self.backup_dir / "backup_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for backup_data in data.get('backups', []):
                        backup_info = BackupInfo(
                            backup_id=backup_data['backup_id'],
                            timestamp=datetime.fromisoformat(backup_data['timestamp']),
                            size=backup_data['size'],
                            checksum=backup_data['checksum'],
                            data_types=backup_data['data_types'],
                            status=backup_data['status'],
                            metadata=backup_data['metadata']
                        )
                        self.backups[backup_info.backup_id] = backup_info
                logger.info(f"加载了 {len(self.backups)} 个备份记录")
            except Exception as e:
                logger.error(f"加载备份索引失败: {e}")


    def _save_backup_index(self):

        """保存备份索引"""
        index_file = self.backup_dir / "backup_index.json"
        try:
            with self.lock:
                data = {
                    'backups': [
                        {
                            'backup_id': info.backup_id,
                            'timestamp': info.timestamp.isoformat(),
                            'size': info.size,
                            'checksum': info.checksum,
                            'data_types': info.data_types,
                            'status': info.status,
                            'metadata': info.metadata
                        }
                        for info in self.backups.values()
                    ]
                }
                with open(index_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存备份索引失败: {e}")


    def create_backup(self, data_sources: Dict[str, Any], description: str = "") -> str:

        """
        创建数据备份

        Args:
            data_sources: 数据源字典 {data_type: data}
            description: 备份描述

        Returns:
            str: 备份ID
        """
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000}"
        backup_path = self.backup_dir / backup_id

        try:
            with self.lock:
                # 创建备份目录
                backup_path.mkdir(exist_ok=True)

                # 备份数据
                data_types = []
                total_size = 0

                for data_type, data in data_sources.items():
                    data_types.append(data_type)
                    data_path = backup_path / f"{data_type}.parquet"

                    if isinstance(data, pd.DataFrame):
                        data.to_parquet(data_path, index=True)
                    elif isinstance(data, dict):
                        # 将字典转换为DataFrame
                        df = pd.DataFrame(data)
                        df.to_parquet(data_path, index=True)
                    else:
                        # 其他类型数据序列化
                        import pickle
                        with open(backup_path / f"{data_type}.pkl", 'wb') as f:
                            pickle.dump(data, f)

                    total_size += data_path.stat().st_size

                # 创建元数据
                metadata = {
                    'description': description,
                    'data_types': data_types,
                    'created_by': 'system',
                    'version': '1.0.0'
                }

                with open(backup_path / "metadata.json", 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                # 计算校验和
                checksum = self._calculate_checksum(backup_path)

                # 创建备份信息
                backup_info = BackupInfo(
                    backup_id=backup_id,
                    timestamp=datetime.now(),
                    size=total_size,
                    checksum=checksum,
                    data_types=data_types,
                    status='created',
                    metadata=metadata
                )

                self.backups[backup_id] = backup_info

                # 验证备份
                if self.config.verify_backup:
                    if self._verify_backup(backup_id):
                        backup_info.status = 'verified'
                        logger.info(f"备份验证成功: {backup_id}")
                    else:
                        backup_info.status = 'failed'
                        logger.error(f"备份验证失败: {backup_id}")

                # 保存索引
                self._save_backup_index()

                # 压缩备份
                if self.config.compression:
                    self._compress_backup(backup_id)

                # 清理旧备份
                if self.config.auto_cleanup:
                    self._cleanup_old_backups()

                logger.info(f"备份创建成功: {backup_id}, 大小: {total_size} bytes")
                return backup_id

        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            # 清理失败的备份
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise


    def restore_backup(self, backup_id: str, target_dir: Optional[str] = None) -> Dict[str, Any]:

        """
        恢复数据备份

        Args:
            backup_id: 备份ID
            target_dir: 目标目录，如果为None则恢复到原位置

        Returns:
            Dict[str, Any]: 恢复的数据
        """
        if backup_id not in self.backups:
            raise ValueError(f"备份不存在: {backup_id}")

        backup_info = self.backups[backup_id]
        backup_path = self.backup_dir / backup_id

        try:
            with self.lock:
                # 检查是否有压缩文件，如果有则先解压
                zip_path = self.backup_dir / f"{backup_id}.zip"
                if zip_path.exists() and not backup_path.exists():
                    self._decompress_backup(backup_id)

                # 解压后再次检查备份路径是否存在
                if not backup_path.exists():
                    raise FileNotFoundError(f"备份文件不存在: {backup_id}")

                # 读取元数据
                metadata_file = backup_path / "metadata.json"
                if not metadata_file.exists():
                    raise FileNotFoundError("备份元数据文件不存在")

                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # 恢复数据
                restored_data = {}
                for data_type in backup_info.data_types:
                    data_path = backup_path / f"{data_type}.parquet"
                    pkl_path = backup_path / f"{data_type}.pkl"

                    if data_path.exists():
                        # 恢复DataFrame数据
                        restored_data[data_type] = pd.read_parquet(data_path)
                    elif pkl_path.exists():
                        # 恢复其他类型数据
                        import pickle
                        with open(pkl_path, 'rb') as f:
                            restored_data[data_type] = pickle.load(f)
                    else:
                        logger.warning(f"数据文件不存在: {data_type}")

                # 如果指定了目标目录，复制到目标目录
                if target_dir:
                    target_path = Path(target_dir)
                    target_path.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(backup_path, target_path / backup_id, dirs_exist_ok=True)

                logger.info(f"备份恢复成功: {backup_id}")
                return restored_data

        except Exception as e:
            logger.error(f"恢复备份失败: {e}")
            raise


    def list_backups(self, data_type: Optional[str] = None, status: Optional[str] = None) -> List[BackupInfo]:

        """
        列出备份

        Args:
            data_type: 数据类型过滤
            status: 状态过滤

        Returns:
            List[BackupInfo]: 备份信息列表
        """
        backups = list(self.backups.values())

        if data_type:
            backups = [b for b in backups if data_type in b.data_types]

        if status:
            backups = [b for b in backups if b.status == status]

        return sorted(backups, key=lambda x: x.timestamp, reverse=True)


    def delete_backup(self, backup_id: str) -> bool:

        """
        删除备份

        Args:
            backup_id: 备份ID

        Returns:
            bool: 是否删除成功
        """
        if backup_id not in self.backups:
            logger.warning(f"备份不存在: {backup_id}")
            return False

        try:
            with self.lock:
                backup_path = self.backup_dir / backup_id
                if backup_path.exists():
                    shutil.rmtree(backup_path)

                # 删除压缩文件
                zip_path = self.backup_dir / f"{backup_id}.zip"
                if zip_path.exists():
                    zip_path.unlink()

                # 从索引中删除
                del self.backups[backup_id]
                self._save_backup_index()

                logger.info(f"备份删除成功: {backup_id}")
                return True

        except Exception as e:
            logger.error(f"删除备份失败: {e}")
            return False


    def _calculate_checksum(self, backup_path: Path) -> str:

        """计算备份校验和"""
        checksum = hashlib.md5()

        for file_path in backup_path.rglob("*"):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    checksum.update(f.read())

        return checksum.hexdigest()


    def _verify_backup(self, backup_id: str) -> bool:

        """验证备份完整性"""
        backup_info = self.backups[backup_id]
        backup_path = self.backup_dir / backup_id

        if not backup_path.exists():
            return False

        # 计算当前校验和
        current_checksum = self._calculate_checksum(backup_path)

        # 比较校验和
        return current_checksum == backup_info.checksum


    def _compress_backup(self, backup_id: str):

        """压缩备份"""
        backup_path = self.backup_dir / backup_id
        zip_path = self.backup_dir / f"{backup_id}.zip"

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in backup_path.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(backup_path)
                        zipf.write(file_path, arcname)

            # 删除原目录
            shutil.rmtree(backup_path)
            logger.info(f"备份压缩完成: {backup_id}")

        except Exception as e:
            logger.error(f"备份压缩失败: {e}")


    def _decompress_backup(self, backup_id: str):

        """解压备份"""
        backup_path = self.backup_dir / backup_id
        zip_path = self.backup_dir / f"{backup_id}.zip"

        if not zip_path.exists():
            return

        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(backup_path)
            logger.info(f"备份解压完成: {backup_id}")

        except Exception as e:
            logger.error(f"备份解压失败: {e}")


    def _cleanup_old_backups(self):

        """清理旧备份"""
        if len(self.backups) <= self.config.max_backups:
            return

        # 按时间排序，删除最旧的备份
        sorted_backups = sorted(self.backups.items(), key=lambda x: x[1].timestamp)
        to_delete = sorted_backups[:-self.config.max_backups]

        for backup_id, _ in to_delete:
            self.delete_backup(backup_id)

        logger.info(f"清理了 {len(to_delete)} 个旧备份")


    def get_backup_stats(self) -> Dict[str, Any]:

        """获取备份统计信息"""
        total_backups = len(self.backups)
        total_size = sum(b.size for b in self.backups.values())
        status_counts = {}

        for backup in self.backups.values():
            status_counts[backup.status] = status_counts.get(backup.status, 0) + 1

        return {
            'total_backups': total_backups,
            'total_size': total_size,
            'status_counts': status_counts,
            'oldest_backup': min(b.timestamp for b in self.backups.values()) if self.backups else None,
            'newest_backup': max(b.timestamp for b in self.backups.values()) if self.backups else None
        }
