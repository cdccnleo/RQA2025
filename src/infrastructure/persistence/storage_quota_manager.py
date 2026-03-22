#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
存储配额管理模块

提供存储空间配额管理、监控和自动清理功能，防止磁盘空间耗尽。
"""

import os
import shutil
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from threading import Lock
import json

logger = logging.getLogger(__name__)


@dataclass
class StorageQuotaConfig:
    """存储配额配置"""
    # 配额限制（字节）
    max_storage_size: int = 10 * 1024 * 1024 * 1024  # 默认10GB
    max_file_count: int = 10000  # 最大文件数量
    
    # 清理策略
    auto_cleanup_enabled: bool = True
    cleanup_threshold_percent: float = 80.0  # 使用率超过80%触发清理
    cleanup_target_percent: float = 60.0  # 清理目标使用率
    
    # 文件保留策略
    min_file_age_days: int = 7  # 最少保留天数
    max_file_age_days: int = 90  # 最大保留天数
    
    # 监控配置
    monitoring_enabled: bool = True
    check_interval_minutes: int = 60  # 检查间隔


@dataclass
class StorageStats:
    """存储统计信息"""
    total_size: int = 0
    file_count: int = 0
    directory_count: int = 0
    usage_percent: float = 0.0
    oldest_file: Optional[datetime] = None
    newest_file: Optional[datetime] = None
    largest_file: Optional[Path] = None
    largest_file_size: int = 0


class StorageQuotaManager:
    """
    存储配额管理器
    
    功能：
    - 监控存储空间使用情况
    - 自动清理过期文件
    - 配额告警
    - 存储统计报告
    """
    
    def __init__(self, storage_path: str, config: Optional[StorageQuotaConfig] = None):
        """
        初始化存储配额管理器
        
        Args:
            storage_path: 存储路径
            config: 配额配置
        """
        self.storage_path = Path(storage_path)
        self.config = config or StorageQuotaConfig()
        self._lock = Lock()
        self._cleanup_history: List[Dict[str, Any]] = []
        
        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"存储配额管理器初始化完成: {storage_path}")
    
    def get_storage_stats(self) -> StorageStats:
        """
        获取存储统计信息
        
        Returns:
            StorageStats: 存储统计
        """
        stats = StorageStats()
        
        if not self.storage_path.exists():
            return stats
        
        try:
            total_size = 0
            file_count = 0
            dir_count = 0
            oldest_time = None
            newest_time = None
            largest_size = 0
            largest_file = None
            
            for item in self.storage_path.rglob('*'):
                if item.is_file():
                    file_count += 1
                    size = item.stat().st_size
                    total_size += size
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    
                    if oldest_time is None or mtime < oldest_time:
                        oldest_time = mtime
                    if newest_time is None or mtime > newest_time:
                        newest_time = mtime
                    
                    if size > largest_size:
                        largest_size = size
                        largest_file = item
                        
                elif item.is_dir():
                    dir_count += 1
            
            stats.total_size = total_size
            stats.file_count = file_count
            stats.directory_count = dir_count
            stats.usage_percent = (total_size / self.config.max_storage_size * 100) if self.config.max_storage_size > 0 else 0
            stats.oldest_file = oldest_time
            stats.newest_file = newest_time
            stats.largest_file = largest_file
            stats.largest_file_size = largest_size
            
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
        
        return stats
    
    def check_quota(self) -> Dict[str, Any]:
        """
        检查配额状态
        
        Returns:
            Dict: 配额检查结果
        """
        stats = self.get_storage_stats()
        
        result = {
            "status": "ok",
            "usage_percent": stats.usage_percent,
            "total_size_mb": stats.total_size / (1024 * 1024),
            "max_size_mb": self.config.max_storage_size / (1024 * 1024),
            "file_count": stats.file_count,
            "max_file_count": self.config.max_file_count,
            "needs_cleanup": False
        }
        
        # 检查空间配额
        if stats.usage_percent >= self.config.cleanup_threshold_percent:
            result["status"] = "warning"
            result["needs_cleanup"] = True
            result["message"] = f"存储使用率 {stats.usage_percent:.1f}% 超过阈值 {self.config.cleanup_threshold_percent}%"
        
        # 检查文件数量配额
        if stats.file_count >= self.config.max_file_count:
            result["status"] = "critical"
            result["needs_cleanup"] = True
            result["message"] = f"文件数量 {stats.file_count} 超过限制 {self.config.max_file_count}"
        
        return result
    
    def cleanup_old_files(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        清理过期文件
        
        Args:
            dry_run: 是否为试运行模式（不实际删除）
            
        Returns:
            Dict: 清理结果
        """
        with self._lock:
            result = {
                "dry_run": dry_run,
                "deleted_count": 0,
                "deleted_size": 0,
                "deleted_files": [],
                "errors": []
            }
            
            if not self.config.auto_cleanup_enabled:
                result["message"] = "自动清理已禁用"
                return result
            
            try:
                cutoff_time = datetime.now() - timedelta(days=self.config.min_file_age_days)
                max_age_time = datetime.now() - timedelta(days=self.config.max_file_age_days)
                
                files_to_delete = []
                
                # 收集需要删除的文件
                for item in self.storage_path.rglob('*'):
                    if not item.is_file():
                        continue
                    
                    try:
                        stat = item.stat()
                        mtime = datetime.fromtimestamp(stat.st_mtime)
                        size = stat.st_size
                        
                        # 删除超过最大保留时间的文件
                        if mtime < max_age_time:
                            files_to_delete.append({
                                "path": item,
                                "size": size,
                                "mtime": mtime,
                                "reason": "max_age"
                            })
                        # 如果空间紧张，删除超过最小保留时间的文件
                        elif mtime < cutoff_time:
                            stats = self.get_storage_stats()
                            if stats.usage_percent >= self.config.cleanup_threshold_percent:
                                files_to_delete.append({
                                    "path": item,
                                    "size": size,
                                    "mtime": mtime,
                                    "reason": "space_pressure"
                                })
                    except Exception as e:
                        result["errors"].append(f"检查文件 {item} 失败: {e}")
                
                # 按修改时间排序，先删除最旧的
                files_to_delete.sort(key=lambda x: x["mtime"])
                
                # 执行删除
                target_size = self.config.max_storage_size * (self.config.cleanup_target_percent / 100)
                current_size = self.get_storage_stats().total_size
                
                for file_info in files_to_delete:
                    if current_size <= target_size:
                        break
                    
                    try:
                        if not dry_run:
                            file_info["path"].unlink()
                        
                        result["deleted_count"] += 1
                        result["deleted_size"] += file_info["size"]
                        result["deleted_files"].append({
                            "path": str(file_info["path"]),
                            "size": file_info["size"],
                            "reason": file_info["reason"]
                        })
                        current_size -= file_info["size"]
                        
                    except Exception as e:
                        result["errors"].append(f"删除文件 {file_info['path']} 失败: {e}")
                
                # 记录清理历史
                cleanup_record = {
                    "timestamp": datetime.now().isoformat(),
                    "dry_run": dry_run,
                    "result": result
                }
                self._cleanup_history.append(cleanup_record)
                
                if len(self._cleanup_history) > 100:
                    self._cleanup_history = self._cleanup_history[-100:]
                
                if not dry_run and result["deleted_count"] > 0:
                    logger.info(f"清理完成: 删除 {result['deleted_count']} 个文件, "
                              f"释放 {result['deleted_size'] / (1024 * 1024):.2f} MB")
                
            except Exception as e:
                logger.error(f"清理过程失败: {e}")
                result["errors"].append(str(e))
            
            return result
    
    def get_cleanup_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取清理历史
        
        Args:
            limit: 返回记录数量
            
        Returns:
            List: 清理历史记录
        """
        return self._cleanup_history[-limit:]
    
    def get_quota_report(self) -> Dict[str, Any]:
        """
        获取配额报告
        
        Returns:
            Dict: 配额报告
        """
        stats = self.get_storage_stats()
        quota_check = self.check_quota()
        
        return {
            "storage_path": str(self.storage_path),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_storage_size_mb": self.config.max_storage_size / (1024 * 1024),
                "max_file_count": self.config.max_file_count,
                "cleanup_threshold_percent": self.config.cleanup_threshold_percent,
                "auto_cleanup_enabled": self.config.auto_cleanup_enabled
            },
            "current_usage": {
                "total_size_mb": stats.total_size / (1024 * 1024),
                "usage_percent": stats.usage_percent,
                "file_count": stats.file_count,
                "directory_count": stats.directory_count,
                "oldest_file": stats.oldest_file.isoformat() if stats.oldest_file else None,
                "newest_file": stats.newest_file.isoformat() if stats.newest_file else None
            },
            "quota_status": quota_check,
            "cleanup_history_summary": {
                "total_cleanups": len(self._cleanup_history),
                "last_cleanup": self._cleanup_history[-1]["timestamp"] if self._cleanup_history else None
            }
        }


# 全局配额管理器实例
_quota_managers: Dict[str, StorageQuotaManager] = {}


def get_storage_quota_manager(storage_path: str, config: Optional[StorageQuotaConfig] = None) -> StorageQuotaManager:
    """
    获取存储配额管理器实例（单例模式）
    
    Args:
        storage_path: 存储路径
        config: 配额配置
        
    Returns:
        StorageQuotaManager: 配额管理器实例
    """
    if storage_path not in _quota_managers:
        _quota_managers[storage_path] = StorageQuotaManager(storage_path, config)
    return _quota_managers[storage_path]
