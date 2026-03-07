#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理策略服务层重复文件
Clean Duplicate Files in Strategy Service Layer

清理monitoring/和backtest/目录中的重复文件，保持代码整洁。
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DuplicateFileCleaner:
    """重复文件清理器"""

    def __init__(self, base_path: str = "src/strategy"):
        self.base_path = Path(base_path)
        self.monitoring_dir = self.base_path / "monitoring"
        self.backtest_dir = self.base_path / "backtest"
        self.backup_dir = self.base_path / "backup_duplicates"

    def calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def find_duplicate_files(self) -> Dict[str, List[Path]]:
        """查找重复文件"""
        duplicates = {}

        # 获取monitoring目录中的所有Python文件
        monitoring_files = list(self.monitoring_dir.glob("*.py"))
        monitoring_files.extend(self.monitoring_dir.glob("*/*.py"))

        for monitoring_file in monitoring_files:
            if monitoring_file.name.startswith('__'):
                continue

            # 检查backtest目录中是否有相同名称的文件
            backtest_file = self.backtest_dir / monitoring_file.name
            if backtest_file.exists():
                # 计算两个文件的哈希值
                monitoring_hash = self.calculate_file_hash(monitoring_file)
                backtest_hash = self.calculate_file_hash(backtest_file)

                if monitoring_hash == backtest_hash:
                    file_key = monitoring_file.name
                    if file_key not in duplicates:
                        duplicates[file_key] = []
                    duplicates[file_key].extend([monitoring_file, backtest_file])

        return duplicates

    def backup_duplicate_files(self, duplicates: Dict[str, List[Path]]) -> None:
        """备份重复文件"""
        self.backup_dir.mkdir(exist_ok=True)

        for file_name, file_paths in duplicates.items():
            backup_subdir = self.backup_dir / file_name.replace('.py', '')
            backup_subdir.mkdir(exist_ok=True)

            for i, file_path in enumerate(file_paths):
                backup_name = f"{file_name}.backup_{i}"
                backup_path = backup_subdir / backup_name
                shutil.copy2(file_path, backup_path)
                logger.info(f"Backed up {file_path} to {backup_path}")

    def remove_duplicate_files_from_monitoring(self, duplicates: Dict[str, List[Path]]) -> None:
        """从monitoring目录删除重复文件"""
        for file_name, file_paths in duplicates.items():
            # 删除monitoring目录中的重复文件
            for file_path in file_paths:
                if str(file_path).startswith(str(self.monitoring_dir)):
                    os.remove(file_path)
                    logger.info(f"Removed duplicate file: {file_path}")

    def clean_empty_directories(self) -> None:
        """清理空目录"""
        def remove_empty_dirs(path: Path):
            for subpath in path.rglob('*'):
                if subpath.is_dir() and not any(subpath.iterdir()):
                    subpath.rmdir()
                    logger.info(f"Removed empty directory: {subpath}")

        remove_empty_dirs(self.monitoring_dir)

    def create_monitoring_specific_files(self) -> None:
        """为monitoring目录创建特定文件"""
        monitoring_init = self.monitoring_dir / "__init__.py"
        if not monitoring_init.exists():
            monitoring_init.write_text('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略监控服务
Strategy Monitoring Service

提供策略运行监控、性能跟踪、告警管理等功能。
"""

from .monitoring_service import MonitoringService, AlertService
from .alert_service import AlertService

__all__ = [
    'MonitoringService',
    'AlertService'
]
''')

        # 创建monitoring_service.py的简化版本
        monitoring_service = self.monitoring_dir / "monitoring_service.py"
        if not monitoring_service.exists():
            monitoring_service.write_text('''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略监控服务实现
Strategy Monitoring Service Implementation

基于业务流程驱动架构，实现策略运行监控和告警功能。
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
from ..interfaces.monitoring_interfaces import IMonitoringService

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """监控配置"""
    monitoring_id: str
    strategy_id: str
    metrics_interval: int = 60  # 指标收集间隔（秒）

class MonitoringService(IMonitoringService):
    """
    监控服务实现
    Monitoring Service Implementation
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._metrics_cache: Dict[str, Any] = {}
        self._alerts: List[Dict[str, Any]] = []

    def collect_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """收集策略指标"""
        # 这里应该集成实际的指标收集逻辑
        return {
            "strategy_id": strategy_id,
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "execution_time": 0.0
        }

    def check_health(self, strategy_id: str) -> Dict[str, Any]:
        """检查策略健康状态"""
        return {
            "strategy_id": strategy_id,
            "status": "healthy",
            "last_check": datetime.now().isoformat()
        }

    def generate_report(self, strategy_id: str) -> Dict[str, Any]:
        """生成监控报告"""
        return {
            "strategy_id": strategy_id,
            "report_type": "monitoring",
            "generated_at": datetime.now().isoformat(),
            "metrics": self._metrics_cache.get(strategy_id, {})
        }
''')

    def run_cleanup(self) -> None:
        """执行清理过程"""
        logger.info("Starting duplicate file cleanup...")

        # 查找重复文件
        duplicates = self.find_duplicate_files()
        logger.info(f"Found {len(duplicates)} duplicate file groups")

        if not duplicates:
            logger.info("No duplicate files found")
            return

        # 显示找到的重复文件
        for file_name, file_paths in duplicates.items():
            logger.info(f"Duplicate: {file_name}")
            for file_path in file_paths:
                logger.info(f"  - {file_path}")

        # 备份重复文件
        logger.info("Backing up duplicate files...")
        self.backup_duplicate_files(duplicates)

        # 删除monitoring目录中的重复文件
        logger.info("Removing duplicate files from monitoring directory...")
        self.remove_duplicate_files_from_monitoring(duplicates)

        # 清理空目录
        logger.info("Cleaning empty directories...")
        self.clean_empty_directories()

        # 创建monitoring特定文件
        logger.info("Creating monitoring-specific files...")
        self.create_monitoring_specific_files()

        logger.info("Duplicate file cleanup completed!")


def main():
    """主函数"""
    cleaner = DuplicateFileCleaner()
    cleaner.run_cleanup()


if __name__ == "__main__":
    main()
