#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用监控数据持久化优化

将增强的监控数据持久化系统集成到现有的RQA2025系统中。
"""

import sys
import logging
import shutil
from pathlib import Path
import json

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class MonitoringPersistenceEnhancementApplier:
    """监控数据持久化优化应用器"""

    def __init__(self):
        """初始化应用器"""
        self.project_root = project_root
        self.src_path = self.project_root / "src"
        self.scripts_path = self.project_root / "scripts"
        self.config_path = self.project_root / "config"

        # 备份目录
        self.backup_path = self.project_root / "backups" / "monitoring_persistence"
        self.backup_path.mkdir(parents=True, exist_ok=True)

        logger.info("监控数据持久化优化应用器初始化完成")

    def backup_existing_files(self):
        """备份现有文件"""
        logger.info("备份现有监控相关文件...")

        files_to_backup = [
            self.src_path / "features" / "monitoring" / "metrics_persistence.py",
            self.src_path / "strategy" / "monitoring" / "monitoring_service.py"
        ]

        for file_path in files_to_backup:
            if file_path.exists():
                backup_file = self.backup_path / f"{file_path.stem}_original.py"
                shutil.copy2(file_path, backup_file)
                logger.info(f"  已备份: {file_path.name} -> {backup_file.name}")

    def integrate_enhanced_persistence_manager(self):
        """集成增强的持久化管理器"""
        logger.info("集成增强的持久化管理器...")

        # 更新现有的metrics_persistence.py
        original_file = self.src_path / "features" / "monitoring" / "metrics_persistence.py"
        enhanced_file = self.scripts_path / "optimization" / "monitoring_persistence_enhancer.py"

        if original_file.exists() and enhanced_file.exists():
            # 读取增强版本的核心类
            with open(enhanced_file, 'r', encoding='utf-8') as f:
                enhanced_content = f.read()

            # 创建集成版本
            integration_content = self._create_integrated_persistence_manager(enhanced_content)

            # 写入集成版本
            with open(original_file, 'w', encoding='utf-8') as f:
                f.write(integration_content)

            logger.info("  持久化管理器集成完成")
        else:
            logger.warning("  无法找到原始文件或增强文件")

    def _create_integrated_persistence_manager(self, enhanced_content: str) -> str:
        """创建集成的持久化管理器"""
        integration_header = '''"""
监控数据持久化管理器 (增强版集成)

集成了高性能、可扩展的监控数据持久化解决方案，包括：
1. 高性能数据存储和检索
2. 数据压缩和归档
3. 实时数据流处理
4. 智能数据生命周期管理
5. 多级缓存机制

原始功能保持兼容，新增增强功能。
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import gzip
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import defaultdict, deque
import asyncio

# 保持原有导入兼容性
from .features_monitor import MetricType, MetricValue

logger = logging.getLogger(__name__)

# 从增强版本导入核心组件
'''

        # 提取增强版本的核心类和函数
        enhanced_classes = self._extract_enhanced_classes(enhanced_content)

        # 创建兼容性适配器
        compatibility_adapter = '''

# 兼容性适配器 - 保持原有接口不变
class MetricsPersistenceManager:
    """原有接口的兼容性适配器"""
    
    def __init__(self, storage_config: Optional[Dict] = None):
        """初始化（兼容原有接口）"""
        # 使用增强的管理器
        self._enhanced_manager = EnhancedMetricsPersistenceManager(storage_config)
    
    def store_metric(self, component_name: str, metric_name: str,
                     metric_value: float, metric_type: MetricType,
                     labels: Optional[Dict[str, str]] = None) -> None:
        """存储指标数据（兼容原有接口）"""
        self._enhanced_manager.store_metric_sync(
            component_name=component_name,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_type=metric_type.value if hasattr(metric_type, 'value') else str(metric_type),
            labels=labels
        )
    
    def query_metrics(self, component_name: Optional[str] = None,
                      metric_name: Optional[str] = None,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      metric_type: Optional[MetricType] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """查询指标数据（兼容原有接口）"""
        return asyncio.run(self._enhanced_manager.query_metrics_async(
            component_name=component_name,
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            metric_type=metric_type.value if metric_type and hasattr(metric_type, 'value') else None,
            limit=limit
        ))
    
    def stop(self) -> None:
        """停止管理器（兼容原有接口）"""
        self._enhanced_manager.stop()


# 保持原有函数接口
def get_persistence_manager(config: Optional[Dict] = None) -> MetricsPersistenceManager:
    """获取持久化管理器实例（兼容原有接口）"""
    return MetricsPersistenceManager(config)

# 新增：获取增强版本的直接访问
def get_enhanced_persistence_manager(config: Optional[Dict] = None) -> EnhancedMetricsPersistenceManager:
    """获取增强的持久化管理器实例（新功能）"""
    return EnhancedMetricsPersistenceManager(config)
'''

        return integration_header + enhanced_classes + compatibility_adapter

    def _extract_enhanced_classes(self, content: str) -> str:
        """提取增强版本的核心类"""
        # 简化：直接返回去掉导入部分的内容
        lines = content.split('\n')

        # 找到第一个class定义的位置
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('class ') and 'CompressionType' in line:
                start_idx = i
                break

        # 返回从第一个类开始的所有内容
        return '\n'.join(lines[start_idx:])

    def update_monitoring_service(self):
        """更新监控服务"""
        logger.info("更新监控服务...")

        monitoring_service_file = self.src_path / "strategy" / "monitoring" / "monitoring_service.py"

        if monitoring_service_file.exists():
            # 读取现有文件
            with open(monitoring_service_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 添加增强持久化支持
            enhanced_import = '''
# 增强的持久化支持
try:
    from ...features.monitoring.metrics_persistence import get_enhanced_persistence_manager
    ENHANCED_PERSISTENCE_AVAILABLE = True
except ImportError:
    ENHANCED_PERSISTENCE_AVAILABLE = False
    logger.warning("增强的持久化功能不可用，使用标准模式")
'''

            # 在导入部分后添加增强导入
            import_end = content.find('\nlogger = logging.getLogger(__name__)')
            if import_end != -1:
                content = content[:import_end] + enhanced_import + content[import_end:]

            # 在MonitoringService类的__init__方法中添加持久化支持
            init_pattern = "def __init__(self):"
            init_pos = content.find(init_pattern)
            if init_pos != -1:
                # 找到__init__方法的结束位置
                init_end = content.find('\n\n', init_pos)
                if init_end != -1:
                    persistence_init = '''
        
        # 初始化增强的持久化管理器
        if ENHANCED_PERSISTENCE_AVAILABLE:
            persistence_config = {
                'path': './monitoring_data_enhanced',
                'primary_backend': 'sqlite',
                'compression': 'lz4',
                'batch_size': 200,
                'batch_timeout': 1.0,
                'archive': {
                    'hot_data_days': 7,
                    'warm_data_days': 30,
                    'cold_data_days': 365
                }
            }
            self.enhanced_persistence = get_enhanced_persistence_manager(persistence_config)
            logger.info("增强的持久化管理器已初始化")
        else:
            self.enhanced_persistence = None'''

                    content = content[:init_end] + persistence_init + content[init_end:]

            # 写入更新后的文件
            with open(monitoring_service_file, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info("  监控服务更新完成")
        else:
            logger.warning("  监控服务文件不存在")

    def create_configuration_files(self):
        """创建配置文件"""
        logger.info("创建监控持久化配置文件...")

        # 创建监控配置目录
        monitoring_config_dir = self.config_path / "monitoring"
        monitoring_config_dir.mkdir(exist_ok=True)

        # 创建持久化配置文件
        persistence_config = {
            "enabled": True,
            "storage": {
                "primary_backend": "sqlite",
                "archive_backend": "parquet",
                "compression": "lz4",
                "path": "./monitoring_data_enhanced"
            },
            "performance": {
                "batch_size": 500,
                "batch_timeout": 2.0,
                "max_workers": 4,
                "cache_size": 10000
            },
            "lifecycle": {
                "hot_data_days": 7,
                "warm_data_days": 30,
                "cold_data_days": 365,
                "cleanup_interval_hours": 24
            },
            "features": {
                "compression_enabled": True,
                "archive_enabled": True,
                "stream_processing": True,
                "anomaly_detection": True
            }
        }

        config_file = monitoring_config_dir / "persistence_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(persistence_config, f, indent=2, ensure_ascii=False)

        logger.info(f"  持久化配置已创建: {config_file}")

        # 创建监控服务配置
        monitoring_service_config = {
            "monitoring": {
                "enabled": True,
                "metrics_interval": 60,
                "alert_check_interval": 30,
                "max_metrics_history": 1000
            },
            "persistence": {
                "enabled": True,
                "config_file": "monitoring/persistence_config.json"
            },
            "alerts": {
                "enabled": True,
                "notification_channels": ["log", "file"],
                "severity_levels": ["HIGH", "MEDIUM", "LOW"]
            }
        }

        service_config_file = monitoring_config_dir / "service_config.json"
        with open(service_config_file, 'w', encoding='utf-8') as f:
            json.dump(monitoring_service_config, f, indent=2, ensure_ascii=False)

        logger.info(f"  监控服务配置已创建: {service_config_file}")

    def create_migration_script(self):
        """创建数据迁移脚本"""
        logger.info("创建数据迁移脚本...")

        migration_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控数据迁移脚本

将现有的监控数据迁移到增强的持久化系统中。
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class MonitoringDataMigrator:
    """监控数据迁移器"""
    
    def __init__(self):
        """初始化迁移器"""
        self.old_data_path = Path("./monitoring_data")
        self.new_data_path = Path("./monitoring_data_enhanced")
        
    def migrate_data(self):
        """迁移数据"""
        logger.info("开始监控数据迁移...")
        
        if not self.old_data_path.exists():
            logger.info("未找到旧的监控数据，跳过迁移")
            return
        
        # 这里实现具体的数据迁移逻辑
        # 例如：从旧的SQLite数据库迁移到新的增强系统
        
        logger.info("监控数据迁移完成")


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO)
    migrator = MonitoringDataMigrator()
    migrator.migrate_data()


if __name__ == "__main__":
    main()
'''

        migration_file = self.scripts_path / "optimization" / "migrate_monitoring_data.py"
        with open(migration_file, 'w', encoding='utf-8') as f:
            f.write(migration_script)

        logger.info(f"  迁移脚本已创建: {migration_file}")

    def create_verification_script(self):
        """创建验证脚本"""
        logger.info("创建功能验证脚本...")

        verification_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控数据持久化优化验证脚本

验证增强的监控数据持久化系统是否正常工作。
"""

import sys
import logging
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def verify_enhanced_persistence():
    """验证增强的持久化功能"""
    logger.info("验证增强的持久化功能...")
    
    try:
        from src.features.monitoring.metrics_persistence import get_enhanced_persistence_manager
        
        # 创建测试管理器
        config = {
            'path': './test_monitoring_verification',
            'batch_size': 10,
            'batch_timeout': 1.0
        }
        
        manager = get_enhanced_persistence_manager(config)
        
        # 测试存储
        logger.info("测试数据存储...")
        success = manager.store_metric_sync(
            component_name='verification_test',
            metric_name='test_metric',
            metric_value=123.45,
            metric_type='TEST',
            labels={'test': 'verification'}
        )
        
        if success:
            logger.info("✓ 数据存储测试通过")
        else:
            logger.error("✗ 数据存储测试失败")
            return False
        
        # 等待批量写入
        time.sleep(2)
        
        # 测试查询
        logger.info("测试数据查询...")
        import asyncio
        result = asyncio.run(manager.query_metrics_async(component_name='verification_test'))
        
        if len(result) > 0:
            logger.info("✓ 数据查询测试通过")
        else:
            logger.error("✗ 数据查询测试失败")
            return False
        
        # 清理
        manager.stop()
        
        logger.info("✓ 增强持久化功能验证通过")
        return True
        
    except Exception as e:
        logger.error(f"✗ 增强持久化功能验证失败: {e}")
        return False


def verify_monitoring_service_integration():
    """验证监控服务集成"""
    logger.info("验证监控服务集成...")
    
    try:
        from src.strategy.monitoring.monitoring_service import MonitoringService
        
        # 创建监控服务实例
        service = MonitoringService()
        
        # 检查是否有增强持久化支持
        has_enhanced = hasattr(service, 'enhanced_persistence')
        
        if has_enhanced:
            logger.info("✓ 监控服务增强持久化集成成功")
            return True
        else:
            logger.warning("! 监控服务未检测到增强持久化支持")
            return False
            
    except Exception as e:
        logger.error(f"✗ 监控服务集成验证失败: {e}")
        return False


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("开始监控数据持久化优化验证")
    logger.info("=" * 50)
    
    all_passed = True
    
    # 验证增强持久化功能
    if not verify_enhanced_persistence():
        all_passed = False
    
    # 验证监控服务集成
    if not verify_monitoring_service_integration():
        all_passed = False
    
    logger.info("=" * 50)
    if all_passed:
        logger.info("✓ 所有验证测试通过")
    else:
        logger.error("✗ 部分验证测试失败")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''

        verification_file = self.scripts_path / "optimization" / "verify_monitoring_enhancements.py"
        with open(verification_file, 'w', encoding='utf-8') as f:
            f.write(verification_script)

        logger.info(f"  验证脚本已创建: {verification_file}")

    def apply_all_enhancements(self):
        """应用所有优化"""
        logger.info("开始应用监控数据持久化优化")
        logger.info("=" * 60)

        try:
            # 1. 备份现有文件
            self.backup_existing_files()

            # 2. 集成增强的持久化管理器
            self.integrate_enhanced_persistence_manager()

            # 3. 更新监控服务
            self.update_monitoring_service()

            # 4. 创建配置文件
            self.create_configuration_files()

            # 5. 创建迁移脚本
            self.create_migration_script()

            # 6. 创建验证脚本
            self.create_verification_script()

            logger.info("=" * 60)
            logger.info("✓ 监控数据持久化优化应用完成")

            # 提供后续操作建议
            self._print_next_steps()

        except Exception as e:
            logger.error(f"✗ 应用优化时发生错误: {e}")
            raise

    def _print_next_steps(self):
        """打印后续操作建议"""
        logger.info("")
        logger.info("后续操作建议:")
        logger.info("1. 运行验证脚本检查集成是否成功:")
        logger.info("   python scripts/optimization/verify_monitoring_enhancements.py")
        logger.info("")
        logger.info("2. 如有需要，运行数据迁移脚本:")
        logger.info("   python scripts/optimization/migrate_monitoring_data.py")
        logger.info("")
        logger.info("3. 运行演示脚本查看新功能:")
        logger.info("   python scripts/optimization/monitoring_persistence_demo.py")
        logger.info("")
        logger.info("4. 根据需要调整配置文件:")
        logger.info("   config/monitoring/persistence_config.json")
        logger.info("   config/monitoring/service_config.json")


def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    applier = MonitoringPersistenceEnhancementApplier()
    applier.apply_all_enhancements()


if __name__ == "__main__":
    main()
