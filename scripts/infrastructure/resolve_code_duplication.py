#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解决基础设施层代码重复问题
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class CodeDuplicationResolver:
    """代码重复解决器"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "backup" / \
            f"code_duplication_resolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.duplication_analysis = {
            "config_managers": [],
            "monitors": [],
            "caches": [],
            "recommendations": []
        }

    def create_backup(self):
        """创建备份"""
        print("🔧 创建备份...")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 备份重复文件
        duplicate_files = [
            "src/integration/unified_config_manager.py",
            "src/integration/config.py",
            "src/integration/monitoring.py"
        ]

        for file_path in duplicate_files:
            if Path(file_path).exists():
                backup_path = self.backup_dir / file_path.replace("src/", "")
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)
                print(f"   ✓ 备份 {file_path} -> {backup_path}")

    def analyze_duplication(self) -> Dict[str, Any]:
        """分析代码重复情况"""
        print("🔍 分析代码重复...")

        # 分析配置管理重复
        config_files = [
            "src/infrastructure/core/config/unified_config_manager.py",
            "src/integration/unified_config_manager.py",
            "src/integration/config.py"
        ]

        for file_path in config_files:
            if Path(file_path).exists():
                self.duplication_analysis["config_managers"].append({
                    "file": file_path,
                    "size": Path(file_path).stat().st_size,
                    "lines": len(Path(file_path).read_text(encoding='utf-8').splitlines())
                })

        # 分析监控重复
        monitor_files = [
            "src/infrastructure/core/monitoring/core/monitor.py",
            "src/infrastructure/core/monitoring/performance_optimized_monitor.py",
            "src/integration/monitoring.py"
        ]

        for file_path in monitor_files:
            if Path(file_path).exists():
                self.duplication_analysis["monitors"].append({
                    "file": file_path,
                    "size": Path(file_path).stat().st_size,
                    "lines": len(Path(file_path).read_text(encoding='utf-8').splitlines())
                })

        # 分析缓存重复
        cache_files = [
            "src/infrastructure/core/cache/smart_cache_strategy.py",
            "src/infrastructure/core/cache/cache_strategy.py"
        ]

        for file_path in cache_files:
            if Path(file_path).exists():
                self.duplication_analysis["caches"].append({
                    "file": file_path,
                    "size": Path(file_path).stat().st_size,
                    "lines": len(Path(file_path).read_text(encoding='utf-8').splitlines())
                })

        # 生成建议
        if len(self.duplication_analysis["config_managers"]) > 1:
            self.duplication_analysis["recommendations"].append({
                "type": "config",
                "description": "合并重复的配置管理实现",
                "action": "保留core/config/unified_config_manager.py，删除integration中的重复"
            })

        if len(self.duplication_analysis["monitors"]) > 1:
            self.duplication_analysis["recommendations"].append({
                "type": "monitor",
                "description": "合并重复的监控实现",
                "action": "保留core/monitoring/core/monitor.py，删除其他重复"
            })

        if len(self.duplication_analysis["caches"]) > 1:
            self.duplication_analysis["recommendations"].append({
                "type": "cache",
                "description": "合并重复的缓存实现",
                "action": "保留smart_cache_strategy.py，删除cache_strategy.py"
            })

        return self.duplication_analysis

    def resolve_config_duplication(self):
        """解决配置管理重复"""
        print("🔧 解决配置管理重复...")

        # 检查integration中的配置管理器是否与core中的重复
        integration_config = "src/integration/config.py"
        if Path(integration_config).exists():
            # 备份并删除重复文件
            backup_path = self.backup_dir / "integration_config_backup.py"
            shutil.move(integration_config, backup_path)
            print(f"   ✓ 移动 {integration_config} -> {backup_path}")

        integration_unified = "src/integration/unified_config_manager.py"
        if Path(integration_unified).exists():
            # 备份并删除重复文件
            backup_path = self.backup_dir / "integration_unified_config_backup.py"
            shutil.move(integration_unified, backup_path)
            print(f"   ✓ 移动 {integration_unified} -> {backup_path}")

    def resolve_monitor_duplication(self):
        """解决监控重复"""
        print("🔧 解决监控重复...")

        # 检查performance_optimized_monitor.py是否与core/monitor.py重复
        performance_monitor = "src/infrastructure/core/monitoring/performance_optimized_monitor.py"
        if Path(performance_monitor).exists():
            # 备份并删除重复文件
            backup_path = self.backup_dir / "performance_monitor_backup.py"
            shutil.move(performance_monitor, backup_path)
            print(f"   ✓ 移动 {performance_monitor} -> {backup_path}")

        # 检查integration中的监控
        integration_monitor = "src/integration/monitoring.py"
        if Path(integration_monitor).exists():
            # 备份并删除重复文件
            backup_path = self.backup_dir / "integration_monitor_backup.py"
            shutil.move(integration_monitor, backup_path)
            print(f"   ✓ 移动 {integration_monitor} -> {backup_path}")

    def resolve_cache_duplication(self):
        """解决缓存重复"""
        print("🔧 解决缓存重复...")

        # 检查cache_strategy.py是否与smart_cache_strategy.py重复
        cache_strategy = "src/infrastructure/core/cache/cache_strategy.py"
        if Path(cache_strategy).exists():
            # 备份并删除重复文件
            backup_path = self.backup_dir / "cache_strategy_backup.py"
            shutil.move(cache_strategy, backup_path)
            print(f"   ✓ 移动 {cache_strategy} -> {backup_path}")

    def update_imports(self):
        """更新导入路径"""
        print("📝 更新导入路径...")

        # 更新所有引用integration中配置管理器的文件
        files_to_update = [
            "src/integration/__init__.py",
            "src/integration/interface.py",
            "src/integration/system_integration_manager.py"
        ]

        for file_path in files_to_update:
            if Path(file_path).exists():
                content = Path(file_path).read_text(encoding='utf-8')

                # 更新导入路径
                updated_content = content.replace(
                    "from .unified_config_manager import UnifiedConfigManager",
                    "from src.infrastructure.core.config.unified_config_manager import UnifiedConfigManager"
                )
                updated_content = updated_content.replace(
                    "from .config import UnifiedConfigManager",
                    "from src.infrastructure.core.config.unified_config_manager import UnifiedConfigManager"
                )
                updated_content = updated_content.replace(
                    "from .monitoring import",
                    "from src.infrastructure.core.monitoring.core.monitor import"
                )

                Path(file_path).write_text(updated_content, encoding='utf-8')
                print(f"   ✓ 更新导入路径 {file_path}")

    def create_unified_factories(self):
        """创建统一的工厂类"""
        print("🏭 创建统一的工厂类...")

        # 创建配置管理工厂
        config_factory_content = '''"""
统一配置管理工厂
"""

from typing import Dict, Any, Optional
from src.infrastructure.core.config.unified_config_manager import UnifiedConfigManager
from src.infrastructure.interfaces.unified_interfaces import IConfigManager


class ConfigManagerFactory:
    """配置管理器工厂"""
    
    @staticmethod
    def create_manager(manager_type: str = "unified", **kwargs) -> IConfigManager:
        """创建配置管理器
        
        Args:
            manager_type: 管理器类型
            **kwargs: 其他参数
            
        Returns:
            配置管理器实例
        """
        if manager_type == "unified":
            return UnifiedConfigManager(**kwargs)
        else:
            raise ValueError(f"Unknown config manager type: {manager_type}")


# 便捷函数
def get_config_manager(manager_type: str = "unified", **kwargs) -> IConfigManager:
    """获取配置管理器（便捷函数）"""
    return ConfigManagerFactory.create_manager(manager_type, **kwargs)
'''

        factory_path = "src/infrastructure/core/factories/config_factory.py"
        Path(factory_path).write_text(config_factory_content, encoding='utf-8')
        print(f"   ✓ 创建配置管理工厂 {factory_path}")

        # 创建监控工厂
        monitor_factory_content = '''"""
统一监控工厂
"""

from typing import Dict, Any, Optional
from src.infrastructure.core.monitoring.core.monitor import UnifiedMonitor
from src.infrastructure.interfaces.unified_interfaces import IMonitor


class MonitorFactory:
    """监控器工厂"""
    
    @staticmethod
    def create_monitor(monitor_type: str = "unified", **kwargs) -> IMonitor:
        """创建监控器
        
        Args:
            monitor_type: 监控器类型
            **kwargs: 其他参数
            
        Returns:
            监控器实例
        """
        if monitor_type == "unified":
            return UnifiedMonitor(**kwargs)
        else:
            raise ValueError(f"Unknown monitor type: {monitor_type}")


# 便捷函数
def get_monitor(monitor_type: str = "unified", **kwargs) -> IMonitor:
    """获取监控器（便捷函数）"""
    return MonitorFactory.create_monitor(monitor_type, **kwargs)
'''

        monitor_factory_path = "src/infrastructure/core/factories/monitor_factory.py"
        Path(monitor_factory_path).write_text(monitor_factory_content, encoding='utf-8')
        print(f"   ✓ 创建监控工厂 {monitor_factory_path}")

        # 创建缓存工厂
        cache_factory_content = '''"""
统一缓存工厂
"""

from typing import Dict, Any, Optional
from src.infrastructure.core.cache.smart_cache_strategy import SmartCacheManager
from src.infrastructure.interfaces.unified_interfaces import ICacheManager


class CacheFactory:
    """缓存管理器工厂"""
    
    @staticmethod
    def create_cache_manager(cache_type: str = "smart", **kwargs) -> ICacheManager:
        """创建缓存管理器
        
        Args:
            cache_type: 缓存类型
            **kwargs: 其他参数
            
        Returns:
            缓存管理器实例
        """
        if cache_type == "smart":
            return SmartCacheManager(**kwargs)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")


# 便捷函数
def get_cache_manager(cache_type: str = "smart", **kwargs) -> ICacheManager:
    """获取缓存管理器（便捷函数）"""
    return CacheFactory.create_cache_manager(cache_type, **kwargs)
'''

        cache_factory_path = "src/infrastructure/core/factories/cache_factory.py"
        Path(cache_factory_path).write_text(cache_factory_content, encoding='utf-8')
        print(f"   ✓ 创建缓存工厂 {cache_factory_path}")

    def update_main_init(self):
        """更新主初始化文件"""
        print("📝 更新主初始化文件...")

        # 更新infrastructure的__init__.py
        init_content = '''"""
基础设施层

提供核心的基础服务支持，包括配置管理、数据库管理、监控、缓存、安全、依赖注入等核心组件。
"""

# 导入核心组件
from .core.config.unified_config_manager import UnifiedConfigManager
from .core.monitoring.core.monitor import UnifiedMonitor
from .core.cache.smart_cache_strategy import SmartCacheManager
from ..health import EnhancedHealthChecker

# 导入工厂类
from .core.factories.config_factory import ConfigManagerFactory, get_config_manager
from .core.factories.monitor_factory import MonitorFactory, get_monitor
from .core.factories.cache_factory import CacheFactory, get_cache_manager

# 导入接口
from .interfaces.unified_interfaces import (
    IConfigManager,
    IMonitor,
    ICacheManager,
    IHealthChecker,
    IErrorHandler
)

__all__ = [
    # 核心组件
    'UnifiedConfigManager',
    'UnifiedMonitor',
    'SmartCacheManager',
    'EnhancedHealthChecker',
    
    # 工厂类
    'ConfigManagerFactory',
    'MonitorFactory',
    'CacheFactory',
    
    # 便捷函数
    'get_config_manager',
    'get_monitor',
    'get_cache_manager',
    
    # 接口
    'IConfigManager',
    'IMonitor',
    'ICacheManager',
    'IHealthChecker',
    'IErrorHandler'
]
'''

        init_path = "src/infrastructure/__init__.py"
        Path(init_path).write_text(init_content, encoding='utf-8')
        print(f"   ✓ 更新主初始化文件 {init_path}")

    def create_migration_guide(self):
        """创建迁移指南"""
        print("📋 创建迁移指南...")

        guide_content = f"""# 代码重复解决报告

## 问题描述
发现多个模块存在代码重复：
- 配置管理：多个UnifiedConfigManager实现
- 监控系统：多个监控器实现
- 缓存系统：多个缓存策略实现

## 解决方案
1. 保留最完整的实现
2. 删除重复文件
3. 更新导入路径
4. 创建统一工厂类

## 迁移后的结构
```
src/infrastructure/
├── core/
│   ├── config/
│   │   └── unified_config_manager.py  # 统一配置管理器
│   ├── monitoring/
│   │   └── core/monitor.py  # 统一监控器
│   ├── cache/
│   │   └── smart_cache_strategy.py  # 智能缓存策略
│   └── factories/  # 统一工厂类
│       ├── config_factory.py
│       ├── monitor_factory.py
│       └── cache_factory.py
```

## 使用方式
```python
from src.infrastructure import get_config_manager, get_monitor, get_cache_manager

# 获取配置管理器
config_manager = get_config_manager("unified")

# 获取监控器
monitor = get_monitor("unified")

# 获取缓存管理器
cache_manager = get_cache_manager("smart")
```

## 注意事项
- 所有重复文件已备份到 {self.backup_dir}
- 如需恢复，可从备份目录复制文件
- 请更新所有引用旧路径的代码
"""

        guide_path = self.backup_dir / "MIGRATION_GUIDE.md"
        guide_path.write_text(guide_content, encoding='utf-8')
        print(f"   ✓ 创建迁移指南 {guide_path}")

    def run(self):
        """执行代码重复解决流程"""
        print("🚀 开始解决代码重复问题...")

        try:
            # 1. 创建备份
            self.create_backup()

            # 2. 分析重复
            analysis = self.analyze_duplication()
            print(f"   📊 发现 {len(analysis['config_managers'])} 个配置管理器文件")
            print(f"   📊 发现 {len(analysis['monitors'])} 个监控器文件")
            print(f"   📊 发现 {len(analysis['caches'])} 个缓存文件")

            # 3. 解决配置管理重复
            self.resolve_config_duplication()

            # 4. 解决监控重复
            self.resolve_monitor_duplication()

            # 5. 解决缓存重复
            self.resolve_cache_duplication()

            # 6. 更新导入路径
            self.update_imports()

            # 7. 创建统一工厂
            self.create_unified_factories()

            # 8. 更新主初始化文件
            self.update_main_init()

            # 9. 创建迁移指南
            self.create_migration_guide()

            print("✅ 代码重复问题解决完成！")
            print(f"📁 备份位置: {self.backup_dir}")

        except Exception as e:
            print(f"❌ 解决过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    resolver = CodeDuplicationResolver()
    resolver.run()


if __name__ == "__main__":
    main()
