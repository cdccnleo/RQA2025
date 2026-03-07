#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复配置模块导入路径的脚本
将旧的导入路径更新为新的重构后的路径
"""

import os


def fix_imports_in_file(file_path: str) -> bool:
    """修复单个文件中的导入路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 修复导入路径映射
        import_fixes = {
            # 核心模块
            'from src.infrastructure.config.unified_manager import UnifiedConfigManager as ConfigManager':
                'from src.infrastructure.config.core.manager import ConfigManager',
            'from src.infrastructure.config.unified_manager import UnifiedConfigManager as Config':
                'from src.infrastructure.config.core.manager import ConfigManager',

            # 验证器
            'from src.infrastructure.config.validation import ConfigValidator':
                'from src.infrastructure.config.core.validator import ConfigValidator',
            'from src.infrastructure.config.config_validator import ConfigValidator':
                'from src.infrastructure.config.core.validator import ConfigValidator',

            # 提供者
            'from src.infrastructure.config.config_provider import ConfigProvider':
                'from src.infrastructure.config.core.provider import ConfigProvider',
            'from src.infrastructure.config.provider import ConfigProvider':
                'from src.infrastructure.config.core.provider import ConfigProvider',

            # 结果类
            'from src.infrastructure.config.config_result import ConfigResult':
                'from src.infrastructure.config.core.result import ConfigResult',
            'from src.infrastructure.config.result import ConfigResult':
                'from src.infrastructure.config.core.result import ConfigResult',

            # 异常类
            'from src.infrastructure.config.exceptions import ConfigLoadError':
                'from src.infrastructure.config.error.exceptions import ConfigLoadError',
            'from src.infrastructure.config.config_exceptions import ConfigLoadError':
                'from src.infrastructure.config.error.exceptions import ConfigLoadError',

            # 策略加载器
            'from src.infrastructure.config.strategies import JSONLoader':
                'from src.infrastructure.config.strategies.json_loader import JSONLoader',
            'from src.infrastructure.config.strategies import YAMLLoader':
                'from src.infrastructure.config.strategies.yaml_loader import YAMLLoader',
            'from src.infrastructure.config.strategies import EnvLoader':
                'from src.infrastructure.config.strategies.env_loader import EnvLoader',

            # 服务类
            'from src.infrastructure.config.services.version_service import VersionService':
                'from src.infrastructure.config.services.version_manager import ConfigVersionManager',
            'from src.infrastructure.config.version_service import VersionService':
                'from src.infrastructure.config.services.version_manager import ConfigVersionManager',

            # 路径管理
            'from src.infrastructure.config.paths import PathConfig, ConfigPaths, get_config_path':
                'from src.infrastructure.config.utils.paths import PathConfig, ConfigPaths, get_config_path',
            'from src.infrastructure.config.config_paths import PathConfig, ConfigPaths, get_config_path':
                'from src.infrastructure.config.utils.paths import PathConfig, ConfigPaths, get_config_path',

            # 迁移工具
            'from src.infrastructure.config.migration import ConfigMigration, MigrationManager':
                'from src.infrastructure.config.utils.migration import ConfigMigration, MigrationManager',
            'from src.infrastructure.config.config_migration import ConfigMigration, MigrationManager':
                'from src.infrastructure.config.utils.migration import ConfigMigration, MigrationManager',

            # 版本存储
            'from src.infrastructure.config.version_storage import FileVersionStorage':
                'from src.infrastructure.config.storage.file_storage import FileVersionStorage',
            'from src.infrastructure.config.config_version_storage import FileVersionStorage':
                'from src.infrastructure.config.storage.file_storage import FileVersionStorage',

            # 事件过滤器
            'from src.infrastructure.config.event_filters import IEventFilter, SensitiveDataFilter, EventTypeFilter':
                'from src.infrastructure.config.event.filters import IEventFilter, SensitiveDataFilter, EventTypeFilter',
            'from src.infrastructure.config.filters import IEventFilter, SensitiveDataFilter, EventTypeFilter':
                'from src.infrastructure.config.event.filters import IEventFilter, SensitiveDataFilter, EventTypeFilter',

            # 数据库配置
            'from src.infrastructure.config.database_config import DatabaseConfig':
                'from src.infrastructure.config.managers.database import DatabaseConfig',
            'from src.infrastructure.config.config_database import DatabaseConfig':
                'from src.infrastructure.config.managers.database import DatabaseConfig',

            # 部署管理器
            'from src.infrastructure.config.deployment_manager import DeploymentManager':
                'from src.infrastructure.config.managers.deployment import DeploymentManager',
            'from src.infrastructure.config.config_deployment import DeploymentManager':
                'from src.infrastructure.config.managers.deployment import DeploymentManager',

            # 性能配置
            'from src.infrastructure.config.performance_config import PerformanceConfig, HighFreqPerformanceConfig':
                'from src.infrastructure.config.managers.performance import PerformanceConfig, HighFreqPerformanceConfig',
            'from src.infrastructure.config.config_performance import PerformanceConfig, HighFreqPerformanceConfig':
                'from src.infrastructure.config.managers.performance import PerformanceConfig, HighFreqPerformanceConfig',

            # 安全服务
            'from src.infrastructure.config import SecurityService':
                'from src.infrastructure.config.services.security import SecurityService',
            'from src.infrastructure.config.security import SecurityService':
                'from src.infrastructure.config.services.security import SecurityService',

            # 缓存服务
            'from src.infrastructure.config.simple_cache import SimpleCache':
                'from src.infrastructure.config.services.cache_service import CacheService',
            'from src.infrastructure.config.cache_service import SimpleCache':
                'from src.infrastructure.config.services.cache_service import CacheService',

            # 验证示例
            'from src.infrastructure.config.validation import config_example':
                'from src.infrastructure.config.validation.config_example import config_example',
            'from src.infrastructure.config.config_example import config_example':
                'from src.infrastructure.config.validation.config_example import config_example',
        }

        # 应用修复
        for old_import, new_import in import_fixes.items():
            content = content.replace(old_import, new_import)

        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已修复: {file_path}")
            return True
        else:
            print(f"⏭️  无需修复: {file_path}")
            return False

    except Exception as e:
        print(f"❌ 修复失败: {file_path} - {str(e)}")
        return False


def find_test_files():
    """查找所有需要修复的测试文件"""
    test_dirs = [
        'tests/unit/infrastructure/config',
        'tests/unit/infrastructure',
        'tests/unit/performance'
    ]

    test_files = []
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.endswith('.py') and file.startswith('test_'):
                        test_files.append(os.path.join(root, file))

    return test_files


def main():
    """主函数"""
    print("🔧 开始修复配置模块导入路径...")

    # 查找所有测试文件
    test_files = find_test_files()
    print(f"📁 找到 {len(test_files)} 个测试文件")

    # 修复每个文件
    fixed_count = 0
    for test_file in test_files:
        if fix_imports_in_file(test_file):
            fixed_count += 1

    print(f"\n🎉 修复完成!")
    print(f"📊 统计:")
    print(f"   - 总文件数: {len(test_files)}")
    print(f"   - 修复文件数: {fixed_count}")
    print(f"   - 无需修复: {len(test_files) - fixed_count}")


if __name__ == "__main__":
    main()
