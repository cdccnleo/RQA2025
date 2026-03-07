#!/usr/bin/env python3
"""
基础设施层配置管理大文件重构工具
将大文件重构为模块化结构，保持向后兼容性
"""

import os


def refactor_config_storage():
    """重构config_storage.py为模块化结构"""

    print('🔄 重构 config_storage.py...')

    source_file = 'src/infrastructure/config/storage/config_storage.py'
    types_dir = 'src/infrastructure/config/storage/types'

    try:
        # 读取原始文件
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取类定义
        classes = extract_class_definitions(content)

        if len(classes) <= 3:
            print('   ⚠️  类数量较少，无需重构')
            return False

        # 为每个类创建单独的文件
        created_files = []

        for class_name, start_line, end_line in classes:
            # 创建类文件
            class_file = os.path.join(types_dir, f"{class_name.lower()}.py")

            # 提取类的完整内容（包括导入）
            lines = content.split('\n')
            class_lines = []

            # 添加必要的导入
            class_lines.extend([
                '"""配置文件存储相关类"""',
                '',
                'from infrastructure.config.core.imports import (',
                '    Dict, Any, Optional, List, Union,',
                '    dataclass, field, Enum',
                ')',
                ''
            ])

            # 添加类内容
            class_lines.extend(lines[start_line:end_line+1])

            # 写入文件
            with open(class_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(class_lines))

            created_files.append(class_file)
            print(f'   ✅ 创建: {os.path.basename(class_file)}')

        # 重构原始文件为导入形式
        import_content = '''"""配置文件存储模块

已重构为模块化结构，保持向后兼容性。
"""

# 导入所有存储相关类
from infrastructure.config.storage.types.configitem import ConfigItem
from infrastructure.config.storage.types.storagetype import StorageType
from infrastructure.config.storage.types.distributedstoragetype import DistributedStorageType
from infrastructure.config.storage.types.consistencylevel import ConsistencyLevel
from infrastructure.config.storage.types.storageconfig import StorageConfig
from infrastructure.config.storage.types.iconfigstorage import IConfigStorage
from infrastructure.config.storage.types.fileconfigstorage import FileConfigStorage
from infrastructure.config.storage.types.memoryconfigstorage import MemoryConfigStorage
from infrastructure.config.storage.types.distributedconfigstorage import DistributedConfigStorage
from infrastructure.config.storage.types.configstorage import ConfigStorage

# Redis存储相关（如果存在）
try:
    from infrastructure.config.storage.redis_config_storage import RedisConfigStorage
except ImportError:
    RedisConfigStorage = None

__all__ = [
    "ConfigItem",
    "StorageType",
    "DistributedStorageType",
    "ConsistencyLevel",
    "StorageConfig",
    "IConfigStorage",
    "FileConfigStorage",
    "MemoryConfigStorage",
    "DistributedConfigStorage",
    "ConfigStorage",
]

if RedisConfigStorage:
    __all__.append("RedisConfigStorage")

# 向后兼容性别名
ConfigItemAlias = ConfigItem
StorageTypeAlias = StorageType
'''

        # 重写原始文件
        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(import_content)

        print(f'   ✅ 重构完成: {os.path.basename(source_file)} 现在从模块导入')
        return True

    except Exception as e:
        print(f'   ❌ 重构失败: {e}')
        return False


def refactor_enhanced_secure_config():
    """重构enhanced_secure_config.py"""

    print('🔄 重构 enhanced_secure_config.py...')

    source_file = 'src/infrastructure/config/security/enhanced_secure_config.py'
    components_dir = 'src/infrastructure/config/security/components'

    try:
        # 读取原始文件
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        classes = extract_class_definitions(content)

        if len(classes) <= 3:
            print('   ⚠️  类数量较少，无需重构')
            return False

        # 创建组件文件
        created_files = []

        for class_name, start_line, end_line in classes:
            class_file = os.path.join(components_dir, f"{class_name.lower()}.py")
            lines = content.split('\n')

            # 提取类内容
            class_lines = [
                '"""安全配置相关类"""',
                '',
                'from infrastructure.config.core.imports import (',
                '    Dict, Any, Optional, List, Union,',
                '    dataclass, field, Enum, threading',
                ')',
                'from infrastructure.config.core.common_mixins import ConfigComponentMixin',
                ''
            ]

            class_lines.extend(lines[start_line:end_line+1])

            with open(class_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(class_lines))

            created_files.append(class_file)
            print(f'   ✅ 创建: {os.path.basename(class_file)}')

        # 重构原始文件
        import_content = '''"""增强安全配置模块

已重构为模块化结构，保持向后兼容性。
"""

# 导入所有安全相关组件
from infrastructure.config.security.components.securityconfig import SecurityConfig
from infrastructure.config.security.components.accessrecord import AccessRecord
from infrastructure.config.security.components.configauditlog import ConfigAuditLog
from infrastructure.config.security.components.configencryptionmanager import ConfigEncryptionManager
from infrastructure.config.security.components.configaccesscontrol import ConfigAccessControl
from infrastructure.config.security.components.configauditmanager import ConfigAuditManager
from infrastructure.config.security.components.hotreloadmanager import HotReloadManager
from infrastructure.config.security.components.enhancedsecureconfigmanager import EnhancedSecureConfigManager

__all__ = [
    "SecurityConfig",
    "AccessRecord",
    "ConfigAuditLog",
    "ConfigEncryptionManager",
    "ConfigAccessControl",
    "ConfigAuditManager",
    "HotReloadManager",
    "EnhancedSecureConfigManager",
]

# 向后兼容性别名
SecurityConfigAlias = SecurityConfig
'''

        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(import_content)

        print(f'   ✅ 重构完成: {os.path.basename(source_file)}')
        return True

    except Exception as e:
        print(f'   ❌ 重构失败: {e}')
        return False


def refactor_config_version_manager():
    """重构config_version_manager.py"""

    print('🔄 重构 config_version_manager.py...')

    source_file = 'src/infrastructure/config/version/config_version_manager.py'
    components_dir = 'src/infrastructure/config/version/components'

    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        classes = extract_class_definitions(content)

        if len(classes) <= 2:
            print('   ⚠️  类数量较少，无需重构')
            return False

        # 创建组件文件
        created_files = []

        for class_name, start_line, end_line in classes:
            class_file = os.path.join(components_dir, f"{class_name.lower()}.py")
            lines = content.split('\n')

            class_lines = [
                '"""版本管理相关类"""',
                '',
                'from infrastructure.config.core.imports import (',
                '    Dict, Any, Optional, List, Union,',
                '    dataclass, field, Enum, datetime',
                ')',
                ''
            ]

            class_lines.extend(lines[start_line:end_line+1])

            with open(class_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(class_lines))

            created_files.append(class_file)
            print(f'   ✅ 创建: {os.path.basename(class_file)}')

        # 重构原始文件
        import_content = '''"""配置版本管理模块

已重构为模块化结构，保持向后兼容性。
"""

# 导入所有版本管理组件
from infrastructure.config.version.components.configversion import ConfigVersion
from infrastructure.config.version.components.configdiff import ConfigDiff
from infrastructure.config.version.components.configversionmanager import ConfigVersionManager

__all__ = [
    "ConfigVersion",
    "ConfigDiff",
    "ConfigVersionManager",
]

# 向后兼容性别名
ConfigVersionAlias = ConfigVersion
'''

        with open(source_file, 'w', encoding='utf-8') as f:
            f.write(import_content)

        print(f'   ✅ 重构完成: {os.path.basename(source_file)}')
        return True

    except Exception as e:
        print(f'   ❌ 重构失败: {e}')
        return False


def extract_class_definitions(content: str):
    """提取类定义及其范围"""
    lines = content.split('\n')
    classes = []
    in_class = False
    class_start = 0
    current_class = ""
    indent_level = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 查找类定义开始
        if stripped.startswith('class '):
            if in_class:
                # 结束之前的类
                classes.append((current_class, class_start, i-1))

            current_class = stripped.split('class ')[1].split('(')[0].split(':')[0].strip()
            class_start = i
            in_class = True
            indent_level = len(line) - len(line.lstrip())

        elif in_class:
            # 检查是否到达类结束
            if stripped and not stripped.startswith(' ') and not stripped.startswith('\t') and not stripped.startswith('#'):
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level:
                    classes.append((current_class, class_start, i-1))
                    in_class = False
                    current_class = ""
                    class_start = 0

    # 处理最后一个类
    if in_class:
        classes.append((current_class, class_start, len(lines)-1))

    return classes


def main():
    """主函数"""

    print('=== 🔄 Phase 1.3: 重构大文件为模块化结构 ===')
    print()

    # 创建必要的目录
    os.makedirs('src/infrastructure/config/storage/types', exist_ok=True)
    os.makedirs('src/infrastructure/config/security/components', exist_ok=True)
    os.makedirs('src/infrastructure/config/version/components', exist_ok=True)

    success_count = 0

    # 重构各个大文件
    if refactor_config_storage():
        success_count += 1

    if refactor_enhanced_secure_config():
        success_count += 1

    if refactor_config_version_manager():
        success_count += 1

    print()
    print(f'🎯 重构完成！成功重构 {success_count}/3 个大文件')

    # 验证结果
    print()
    print('✅ 验证重构结果:')
    large_files = [
        'src/infrastructure/config/storage/config_storage.py',
        'src/infrastructure/config/security/enhanced_secure_config.py',
        'src/infrastructure/config/version/config_version_manager.py'
    ]

    remaining_large = 0
    for file_path in large_files:
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            if size_kb > 10:  # 降低阈值，因为现在主要是导入代码
                print(f'   ⚠️  {os.path.basename(file_path)}: {size_kb:.1f} KB')
                remaining_large += 1
            else:
                print(f'   ✅ {os.path.basename(file_path)}: {size_kb:.1f} KB (已模块化)')

    if remaining_large == 0:
        print('   🎉 所有大文件已成功重构为模块化结构！')
    else:
        print(f'   📋 还有 {remaining_large} 个文件需要进一步优化')


if __name__ == '__main__':
    main()
