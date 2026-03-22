#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据备份模块

提供数据库备份和恢复功能，支持：
- PostgreSQL数据库备份
- 自动定时备份
- 增量备份
- 备份恢复
"""

from .backup_manager import BackupManager, BackupConfig, BackupResult

__all__ = [
    'BackupManager',
    'BackupConfig',
    'BackupResult'
]
