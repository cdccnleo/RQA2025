#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库备份管理器

实现PostgreSQL数据库的备份、恢复和管理功能。
"""

import os
import gzip
import shutil
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from threading import Thread, Event
import json

from src.infrastructure.persistence.database_config import DatabaseConfigManager

logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """备份配置"""
    backup_dir: str = "backups"  # 备份目录
    retention_days: int = 30  # 保留天数
    compress: bool = True  # 是否压缩
    include_schema: bool = True  # 是否包含schema
    include_data: bool = True  # 是否包含数据
    exclude_tables: List[str] = field(default_factory=list)  # 排除的表
    schedule_enabled: bool = True  # 是否启用定时备份
    schedule_interval_hours: int = 24  # 定时备份间隔（小时）
    max_backup_count: int = 100  # 最大备份数量


@dataclass
class BackupResult:
    """备份结果"""
    success: bool
    backup_path: Optional[str]
    backup_size: int
    duration_seconds: float
    tables_backed_up: int
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class BackupManager:
    """
    数据库备份管理器
    
    功能：
    - 执行数据库备份（pg_dump）
    - 执行数据库恢复（pg_restore/psql）
    - 管理备份文件（清理过期备份）
    - 定时自动备份
    - 备份验证
    """
    
    def __init__(self, config: Optional[BackupConfig] = None):
        """
        初始化备份管理器
        
        Args:
            config: 备份配置
        """
        self.config = config or BackupConfig()
        self._db_config = DatabaseConfigManager.get_config()
        self._backup_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._running = False
        
        # 确保备份目录存在
        self._ensure_backup_dir()
    
    def _ensure_backup_dir(self) -> None:
        """确保备份目录存在"""
        backup_path = Path(self.config.backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (backup_path / "full").mkdir(exist_ok=True)
        (backup_path / "incremental").mkdir(exist_ok=True)
        (backup_path / "schema").mkdir(exist_ok=True)
    
    def create_backup(
        self, 
        backup_type: str = "full",
        description: str = ""
    ) -> BackupResult:
        """
        创建数据库备份
        
        Args:
            backup_type: 备份类型 ("full", "schema", "data")
            description: 备份描述
            
        Returns:
            备份结果
        """
        import time
        
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 构建备份文件名
            filename = f"backup_{backup_type}_{timestamp}"
            if description:
                filename += f"_{description}"
            filename += ".sql"
            
            if self.config.compress:
                filename += ".gz"
            
            backup_path = Path(self.config.backup_dir) / backup_type / filename
            
            # 构建pg_dump命令
            cmd = self._build_pg_dump_command(backup_type, str(backup_path))
            
            logger.info(f"🔄 开始创建备份: {backup_path}")
            
            # 执行备份
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            if result.returncode != 0:
                error_msg = f"pg_dump失败: {result.stderr}"
                logger.error(f"❌ {error_msg}")
                return BackupResult(
                    success=False,
                    backup_path=None,
                    backup_size=0,
                    duration_seconds=time.time() - start_time,
                    tables_backed_up=0,
                    error_message=error_msg
                )
            
            # 获取备份文件大小
            backup_size = backup_path.stat().st_size if backup_path.exists() else 0
            
            # 创建备份元数据
            metadata = {
                "timestamp": timestamp,
                "type": backup_type,
                "description": description,
                "size": backup_size,
                "compressed": self.config.compress,
                "database": self._db_config.database,
                "tables": self._get_table_list()
            }
            
            metadata_path = backup_path.with_suffix(backup_path.suffix + ".json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            duration = time.time() - start_time
            logger.info(f"✅ 备份完成: {backup_path} ({backup_size} bytes, {duration:.2f}s)")
            
            return BackupResult(
                success=True,
                backup_path=str(backup_path),
                backup_size=backup_size,
                duration_seconds=duration,
                tables_backed_up=len(metadata["tables"]),
                timestamp=datetime.now()
            )
            
        except subprocess.TimeoutExpired:
            error_msg = "备份超时（超过1小时）"
            logger.error(f"❌ {error_msg}")
            return BackupResult(
                success=False,
                backup_path=None,
                backup_size=0,
                duration_seconds=time.time() - start_time,
                tables_backed_up=0,
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"备份失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return BackupResult(
                success=False,
                backup_path=None,
                backup_size=0,
                duration_seconds=time.time() - start_time,
                tables_backed_up=0,
                error_message=error_msg
            )
    
    def _build_pg_dump_command(self, backup_type: str, output_path: str) -> List[str]:
        """
        构建pg_dump命令
        
        Args:
            backup_type: 备份类型
            output_path: 输出路径
            
        Returns:
            命令列表
        """
        cmd = [
            "pg_dump",
            "-h", self._db_config.host,
            "-p", str(self._db_config.port),
            "-U", self._db_config.user,
            "-d", self._db_config.database,
            "--verbose"
        ]
        
        # 备份类型选项
        if backup_type == "schema":
            cmd.extend(["--schema-only"])
        elif backup_type == "data":
            cmd.extend(["--data-only"])
        
        # 排除表
        for table in self.config.exclude_tables:
            cmd.extend(["--exclude-table", table])
        
        # 输出处理
        if self.config.compress:
            # 使用gzip压缩
            cmd.extend(["-f", "-"])  # 输出到stdout
            cmd = ["pg_dump"] + cmd[1:]  # 移除pg_dump，后面重新组合
            full_cmd = f"set PGPASSWORD={self._db_config.password} && pg_dump " + " ".join(cmd[1:]) + f" | gzip > {output_path}"
            return ["cmd", "/c", full_cmd]
        else:
            cmd.extend(["-f", output_path])
            return ["cmd", "/c", f"set PGPASSWORD={self._db_config.password} && " + " ".join(cmd)]
    
    def _get_table_list(self) -> List[str]:
        """获取数据库表列表"""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self._db_config.host,
                port=self._db_config.port,
                database=self._db_config.database,
                user=self._db_config.user,
                password=self._db_config.password
            )
            
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public'
                """)
                tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return tables
            
        except Exception as e:
            logger.error(f"获取表列表失败: {e}")
            return []
    
    def restore_backup(
        self, 
        backup_path: str,
        target_database: Optional[str] = None,
        drop_existing: bool = False
    ) -> bool:
        """
        恢复数据库备份
        
        Args:
            backup_path: 备份文件路径
            target_database: 目标数据库名称（可选）
            drop_existing: 是否删除现有数据库
            
        Returns:
            是否恢复成功
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                logger.error(f"❌ 备份文件不存在: {backup_path}")
                return False
            
            target_db = target_database or self._db_config.database
            
            logger.info(f"🔄 开始恢复备份: {backup_path} -> {target_db}")
            
            # 如果需要，删除现有数据库
            if drop_existing:
                self._drop_database(target_db)
                self._create_database(target_db)
            
            # 构建恢复命令
            if backup_path.endswith('.gz'):
                # 解压并恢复
                cmd = [
                    "cmd", "/c",
                    f"gzip -dc {backup_path} | set PGPASSWORD={self._db_config.password} && psql -h {self._db_config.host} -p {self._db_config.port} -U {self._db_config.user} -d {target_db}"
                ]
            else:
                cmd = [
                    "cmd", "/c",
                    f"set PGPASSWORD={self._db_config.password} && psql -h {self._db_config.host} -p {self._db_config.port} -U {self._db_config.user} -d {target_db} -f {backup_path}"
                ]
            
            # 执行恢复
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2小时超时
            )
            
            if result.returncode != 0:
                logger.error(f"❌ 恢复失败: {result.stderr}")
                return False
            
            logger.info(f"✅ 恢复完成: {target_db}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ 恢复超时（超过2小时）")
            return False
        except Exception as e:
            logger.error(f"❌ 恢复失败: {e}")
            return False
    
    def _drop_database(self, database: str) -> bool:
        """删除数据库"""
        try:
            cmd = [
                "cmd", "/c",
                f"set PGPASSWORD={self._db_config.password} && dropdb -h {self._db_config.host} -p {self._db_config.port} -U {self._db_config.user} --if-exists {database}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"删除数据库失败: {e}")
            return False
    
    def _create_database(self, database: str) -> bool:
        """创建数据库"""
        try:
            cmd = [
                "cmd", "/c",
                f"set PGPASSWORD={self._db_config.password} && createdb -h {self._db_config.host} -p {self._db_config.port} -U {self._db_config.user} {database}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"创建数据库失败: {e}")
            return False
    
    def list_backups(self, backup_type: str = "all") -> List[Dict[str, Any]]:
        """
        列出所有备份
        
        Args:
            backup_type: 备份类型过滤 ("all", "full", "schema", "data")
            
        Returns:
            备份列表
        """
        backups = []
        backup_dir = Path(self.config.backup_dir)
        
        types_to_list = [backup_type] if backup_type != "all" else ["full", "schema", "data"]
        
        for btype in types_to_list:
            type_dir = backup_dir / btype
            if not type_dir.exists():
                continue
            
            for backup_file in type_dir.glob("*.sql*"):
                if backup_file.suffix == ".json":
                    continue
                
                # 读取元数据
                metadata = {}
                metadata_file = backup_file.with_suffix(backup_file.suffix + ".json")
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                stat = backup_file.stat()
                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "type": btype,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "metadata": metadata
                })
        
        # 按创建时间排序
        backups.sort(key=lambda x: x["created"], reverse=True)
        return backups
    
    def cleanup_old_backups(self) -> int:
        """
        清理过期备份
        
        Returns:
            清理的备份数量
        """
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        deleted_count = 0
        
        for backup_type in ["full", "schema", "data"]:
            type_dir = Path(self.config.backup_dir) / backup_type
            if not type_dir.exists():
                continue
            
            for backup_file in type_dir.glob("*.sql*"):
                if backup_file.suffix == ".json":
                    continue
                
                # 检查文件修改时间
                stat = backup_file.stat()
                file_mtime = datetime.fromtimestamp(stat.st_mtime)
                
                if file_mtime < cutoff_date:
                    try:
                        backup_file.unlink()
                        # 同时删除元数据文件
                        metadata_file = backup_file.with_suffix(backup_file.suffix + ".json")
                        if metadata_file.exists():
                            metadata_file.unlink()
                        deleted_count += 1
                        logger.info(f"🗑️ 删除过期备份: {backup_file.name}")
                    except Exception as e:
                        logger.error(f"删除备份失败 {backup_file}: {e}")
        
        logger.info(f"✅ 清理完成，共删除 {deleted_count} 个过期备份")
        return deleted_count
    
    def start_scheduled_backup(self) -> None:
        """启动定时备份"""
        if self._running:
            logger.warning("定时备份已在运行")
            return
        
        self._running = True
        self._stop_event.clear()
        
        def backup_worker():
            import time
            while not self._stop_event.is_set():
                try:
                    # 创建备份
                    result = self.create_backup("full", "scheduled")
                    if result.success:
                        logger.info(f"✅ 定时备份完成: {result.backup_path}")
                        # 清理过期备份
                        self.cleanup_old_backups()
                    else:
                        logger.error(f"❌ 定时备份失败: {result.error_message}")
                except Exception as e:
                    logger.error(f"定时备份异常: {e}")
                
                # 等待下一次备份
                self._stop_event.wait(self.config.schedule_interval_hours * 3600)
        
        self._backup_thread = Thread(target=backup_worker, name="BackupScheduler", daemon=True)
        self._backup_thread.start()
        logger.info(f"✅ 定时备份已启动，间隔: {self.config.schedule_interval_hours}小时")
    
    def stop_scheduled_backup(self) -> None:
        """停止定时备份"""
        if not self._running:
            return
        
        self._stop_event.set()
        self._running = False
        
        if self._backup_thread and self._backup_thread.is_alive():
            self._backup_thread.join(timeout=5)
        
        logger.info("✅ 定时备份已停止")
    
    def verify_backup(self, backup_path: str) -> bool:
        """
        验证备份文件完整性
        
        Args:
            backup_path: 备份文件路径
            
        Returns:
            是否有效
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                return False
            
            # 检查文件大小
            if backup_file.stat().st_size == 0:
                return False
            
            # 如果是压缩文件，检查是否能解压
            if backup_path.endswith('.gz'):
                import gzip
                try:
                    with gzip.open(backup_path, 'rb') as f:
                        # 尝试读取前1KB
                        f.read(1024)
                    return True
                except:
                    return False
            else:
                # 检查SQL文件格式
                with open(backup_path, 'r', encoding='utf-8', errors='ignore') as f:
                    first_line = f.readline()
                    return 'PostgreSQL' in first_line or 'pg_dump' in first_line
                    
        except Exception as e:
            logger.error(f"验证备份失败: {e}")
            return False
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """获取备份统计信息"""
        backups = self.list_backups()
        total_size = sum(b["size"] for b in backups)
        
        return {
            "total_backups": len(backups),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "by_type": {
                "full": len([b for b in backups if b["type"] == "full"]),
                "schema": len([b for b in backups if b["type"] == "schema"]),
                "data": len([b for b in backups if b["type"] == "data"])
            },
            "scheduled_backup_running": self._running,
            "retention_days": self.config.retention_days,
            "backup_dir": self.config.backup_dir
        }


# 全局备份管理器实例
_backup_manager: Optional[BackupManager] = None


def get_backup_manager(config: Optional[BackupConfig] = None) -> BackupManager:
    """
    获取备份管理器实例（单例模式）
    
    Args:
        config: 备份配置（首次调用时生效）
        
    Returns:
        备份管理器实例
    """
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager(config)
    return _backup_manager
