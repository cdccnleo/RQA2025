#!/usr/bin/env python3
import os

config_dir = 'src/infrastructure/config'
total_files = 0
total_dirs = 0

print('各目录文件统计:')

for root, dirs, files in os.walk(config_dir):
    # 排除__pycache__目录
    dirs[:] = [d for d in dirs if d != '__pycache__']
    if dirs:  # 只统计当前级别的目录
        total_dirs += len(dirs)

    py_files = [f for f in files if f.endswith('.py')]
    if py_files:
        level = root.replace(config_dir, '').count(os.sep)
        indent = '  ' * level
        print(f'{indent}{os.path.basename(root)}/: {len(py_files)} 个文件')
        total_files += len(py_files)

print(f'\n总计:')
print(f'   • Python文件数: {total_files}')
print(f'   • 目录数: {total_dirs}')
print(f'   • 文档记录: 58个文件，15个目录')
print(f'   • 需要更新文档: {"YES" if total_files != 58 or total_dirs != 15 else "NO"}')
