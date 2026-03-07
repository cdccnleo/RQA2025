#!/usr/bin/env python3
"""
特征层导入路径修复脚本

修复特征层中错误的导入路径，将src.engine.*改为正确的路径
"""

import os
from pathlib import Path


def fix_import_paths():
    """修复特征层中的导入路径"""

    # 需要修复的文件路径
    features_dir = Path("src/features")

    # 导入路径映射
    import_mappings = {
        "from src.engine.logging.unified_logger import": "from src.infrastructure.logging.unified_logger import",
        "from src.engine.": "from src.infrastructure.",
        "src.engine.logging": "src.infrastructure.logging",
        "src.engine.": "src.infrastructure."
    }

    # 查找所有Python文件
    python_files = []
    for ext in ['*.py']:
        python_files.extend(features_dir.rglob(ext))

    fixed_files = []

    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            # 应用所有的导入路径映射
            for old_pattern, new_pattern in import_mappings.items():
                if old_pattern in content:
                    content = content.replace(old_pattern, new_pattern)
                    modified = True

            # 如果内容被修改，写回文件
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(str(file_path))
                print(f"Fixed: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("\nImport path fixing completed!")
    print(f"Fixed {len(fixed_files)} files:")
    for file in fixed_files:
        print(f"  - {file}")

    return fixed_files


if __name__ == "__main__":
    # 切换到项目根目录
    os.chdir(Path(__file__).parent.parent.parent)
    fix_import_paths()
