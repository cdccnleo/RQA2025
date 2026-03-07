#!/usr/bin/env python3
"""
修复Python文件编码声明问题
将有问题的编码声明从"# -*- coding: utf-8 -*-"修复为"# -*- coding: utf-8 -*-"
"""

import os
import glob
from pathlib import Path


def fix_encoding_declarations():
    """修复所有Python文件的编码声明"""
    project_root = Path(__file__).parent.parent

    # 查找所有Python文件
    pattern = os.path.join(str(project_root), "**", "*.py")
    python_files = glob.glob(pattern, recursive=True)

    fixed_count = 0
    for file_path in python_files:
        try:
            # 获取相对路径
            rel_path = os.path.relpath(file_path, str(project_root))

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 检查是否有问题的编码声明
            if "# -*- coding: utf-8 -*-" in content:
                # 修复编码声明
                new_content = content.replace("# -*- coding: utf-8 -*-", "# -*- coding: utf-8 -*-")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                fixed_count += 1
                print(f"✅ 修复文件: {rel_path}")

        except Exception as e:
            rel_path = os.path.relpath(file_path, str(project_root))
            print(f"❌ 处理文件失败: {rel_path} - {e}")

    print(f"\n🎯 编码修复完成，共修复了 {fixed_count} 个文件")


def fix_shebang_declarations():
    """修复shebang声明问题"""
    project_root = Path(__file__).parent.parent

    pattern = os.path.join(str(project_root), "**", "*.py")
    python_files = glob.glob(pattern, recursive=True)

    fixed_count = 0
    for file_path in python_files:
        try:
            rel_path = os.path.relpath(file_path, str(project_root))

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 检查是否有问题的shebang声明
            if "#!/usr/bin/env python3" in content:
                # 修复shebang声明
                new_content = content.replace("#!/usr/bin/env python3", "#!/usr/bin/env python3")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                fixed_count += 1
                print(f"✅ 修复shebang: {rel_path}")

        except Exception as e:
            rel_path = os.path.relpath(file_path, str(project_root))
            print(f"❌ 处理shebang失败: {rel_path} - {e}")

    print(f"\n🎯 shebang修复完成，共修复了 {fixed_count} 个文件")


if __name__ == "__main__":
    print("🔧 开始修复编码和shebang问题...")
    fix_encoding_declarations()
    fix_shebang_declarations()
    print("🎉 修复完成！")
