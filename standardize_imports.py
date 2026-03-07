#!/usr/bin/env python3
"""
标准化所有ComponentFactory导入语句

统一使用绝对导入路径，避免相对导入问题。
"""

import os


def standardize_imports():
    """标准化导入语句"""

    # 需要修复的文件列表（通过grep查找包含ComponentFactory导入的文件）
    files_to_fix = []
    for root, dirs, files in os.walk('src/infrastructure'):
        for file in files:
            if file.endswith('.py') and file != 'base_components.py':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 查找各种ComponentFactory导入语句
                        if ('from .' in content and 'ComponentFactory' in content) or \
                           ('from ..' in content and 'ComponentFactory' in content) or \
                           ('from ...' in content and 'ComponentFactory' in content):
                            files_to_fix.append(file_path)
                except:
                    pass

    print(f"发现需要标准化的文件: {len(files_to_fix)}")

    fixed_count = 0
    for file_path in files_to_fix:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 替换各种相对导入为绝对导入
            replacements = [
                ('from .common.core.base_components import ComponentFactory',
                 'from infrastructure.utils.common.core.base_components import ComponentFactory'),
                ('from ..common.core.base_components import ComponentFactory',
                 'from infrastructure.utils.common.core.base_components import ComponentFactory'),
                ('from ...utils.common.core.base_components import ComponentFactory',
                 'from infrastructure.utils.common.core.base_components import ComponentFactory'),
                ('from .common.core.base_components import ComponentFactory, IComponentFactory',
                 'from infrastructure.utils.common.core.base_components import ComponentFactory, IComponentFactory'),
                ('from ..common.core.base_components import ComponentFactory, IComponentFactory',
                 'from infrastructure.utils.common.core.base_components import ComponentFactory, IComponentFactory'),
                ('from ...utils.common.core.base_components import ComponentFactory, IComponentFactory',
                 'from infrastructure.utils.common.core.base_components import ComponentFactory, IComponentFactory'),
            ]

            modified = False
            for old_import, new_import in replacements:
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    modified = True

            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ 标准化导入: {file_path}")
                fixed_count += 1

        except Exception as e:
            print(f"❌ 处理文件失败: {file_path} - {e}")

    print(f"\n✅ 导入标准化完成! 共处理 {fixed_count} 个文件")


if __name__ == '__main__':
    standardize_imports()
