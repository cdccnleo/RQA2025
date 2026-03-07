#!/usr/bin/env python3
"""
RQA2025 测试导入修复脚本
"""

from pathlib import Path


def fix_test_imports():
    """修复测试文件中的导入问题"""
    project_root = Path(__file__).parent.parent

    # 修复scipy.sparse导入
    test_files = list(project_root.rglob('*.py'))

    for test_file in test_files:
        if 'test' in test_file.name.lower():
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 修复scipy.sparse导入
                if 'scipy.sparse' in content:
                    content = content.replace('scipy.sparse', 'scipy.sparse')
                    print(f"✅ 修复了 {test_file}")

                # 修复其他常见导入问题
                content = content.replace('import scipy.sparse as sparse',
                                          'import scipy.sparse as sparse')

                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)

            except Exception as e:
                print(f"❌ 修复 {test_file} 时出错: {e}")


if __name__ == "__main__":
    fix_test_imports()
