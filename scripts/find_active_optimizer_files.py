#!/usr/bin/env python3
"""
查找项目中实际存在的optimizer_*.py模板文件
"""

import os
import re
from pathlib import Path


def find_active_optimizer_files():
    """查找实际存在的optimizer_*.py文件（排除备份目录）"""
    print("🔍 查找实际存在的optimizer_*.py文件...")
    print("="*60)

    # 排除目录
    exclude_dirs = {
        '__pycache__', '.git', 'node_modules', '.venv', 'venv',
        'backup', 'backups', 'temp', 'tmp', 'build', 'dist',
        'test', 'tests', 'testing'
    }

    # 排除包含备份关键词的路径
    exclude_patterns = ['_backup', '_optimization', 'backup_']

    optimizer_files = []

    for root, dirs, files in os.walk('.'):
        # 移除需要排除的目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        # 跳过包含备份关键词的路径
        if any(pattern in root.lower() for pattern in exclude_patterns):
            continue

        for file in files:
            if file.endswith('.py') and re.match(r'optimizer_\d+\.py$', file):
                file_path = Path(root) / file
                match = re.search(r'optimizer_(\d+)\.py$', file)
                if match:
                    optimizer_id = int(match.group(1))
                    size_kb = file_path.stat().st_size / 1024
                    optimizer_files.append({
                        'path': file_path,
                        'optimizer_id': optimizer_id,
                        'size_kb': size_kb,
                        'name': file_path.name,
                        'directory': file_path.parent
                    })

    print(f"   📁 发现 {len(optimizer_files)} 个实际存在的optimizer_*.py文件")

    if optimizer_files:
        # 按目录分组
        from collections import defaultdict
        dir_groups = defaultdict(list)
        for file_info in optimizer_files:
            dir_groups[file_info['directory']].append(file_info)

        print("   📂 分布在以下目录中:")
        for directory, files in dir_groups.items():
            print(f"      - {directory}: {len(files)}个文件")
            for file_info in files:
                print(f"        * {file_info['name']} ({file_info['size_kb']:.1f} KB)")
    else:
        print("   ✅ 没有发现需要处理的optimizer_*.py文件")

    return optimizer_files


if __name__ == "__main__":
    find_active_optimizer_files()
