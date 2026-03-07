#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

def should_use_absolute_import(current_file, target_module):
    """
    判断是否应该使用绝对导入

    对于深度嵌套的模块（嵌套层级 > 3）或跨大模块的导入，使用绝对导入
    对于浅层嵌套的模块，使用相对导入
    """
    # 计算当前文件的嵌套层级
    current_depth = len(current_file.replace('src/', '').split('/')) - 1

    # 如果当前文件深度 > 3，或者目标模块跨多个大模块，使用绝对导入
    if current_depth > 3:
        return True

    # 检查是否跨大模块 (如从infrastructure到core)
    current_module = current_file.replace('src/', '').split('/')[0]
    target_module_part = target_module.replace('src.', '').split('.')[0]

    if current_module != target_module_part:
        return True

    return False

def fix_imports_conservatively():
    """
    保守地修复导入问题：
    1. 深度嵌套的模块保留绝对导入
    2. 浅层模块使用相对导入
    3. 修复明显的错误导入
    """
    fixed_files = []

    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    original_content = content
                    lines = content.split('\n')
                    modified = False

                    for i, line in enumerate(lines):
                        original_line = line
                        stripped = line.strip()

                        # 处理 infrastructure.xxx 形式的导入，转换为相对导入（如果深度合适）
                        if stripped.startswith('from infrastructure.') or stripped.startswith('import infrastructure.'):
                            # 检查是否应该使用绝对导入
                            if not should_use_absolute_import(file_path, stripped.replace('from ', '').replace('import ', '').split('.')[0]):
                                # 计算相对路径
                                current_parts = file_path.replace('src/', '').split('/')
                                target_parts = stripped.replace('from infrastructure.', '').replace('import infrastructure.', '').split('.')

                                # 简单计算相对导入（只处理一级）
                                if 'infrastructure' in current_parts:
                                    infra_index = current_parts.index('infrastructure')
                                    if infra_index < len(current_parts) - 1:
                                        # 在infrastructure内部，使用相对导入
                                        relative_parts = ['..'] + target_parts
                                        relative_import = '.'.join(relative_parts)

                                        if stripped.startswith('from infrastructure.'):
                                            new_line = stripped.replace('from infrastructure.', f'from {relative_import}')
                                        else:
                                            new_line = stripped.replace('import infrastructure.', f'from {relative_import} import *')

                                        lines[i] = original_line.replace(stripped, new_line)
                                        modified = True

                    if modified:
                        new_content = '\n'.join(lines)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        fixed_files.append(file_path)

                except Exception as e:
                    print(f'Error processing {file_path}: {e}')

    return fixed_files

def main():
    print("🔧 开始保守修复导入问题...")

    fixed_files = fix_imports_conservatively()

    print(f"\n📊 修复结果:")
    print(f"  修复文件数: {len(fixed_files)}")

    if fixed_files:
        print("修复的文件:")
        for f in fixed_files[:5]:
            print(f"  ✅ {f}")
        if len(fixed_files) > 5:
            print(f"  ... 还有 {len(fixed_files) - 5} 个")

    print("\n🎉 保守导入修复完成！")

if __name__ == '__main__':
    main()
