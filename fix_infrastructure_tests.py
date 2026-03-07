#!/usr/bin/env python3
"""
基础设施层测试文件修复脚本 - 简化版本

专门修复基础设施层测试文件的导入和语法问题
"""

import os
import re
from pathlib import Path

def fix_test_file(file_path):
    """
    修复单个测试文件
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 如果已经有正确的导入结构，跳过
        if 'project_root = Path(__file__).resolve()' in content and 'pytest.skip("基础设施模块导入失败"' in content:
            print(f"⏭️ 已修复: {file_path}")
            return False

        lines = content.split('\n')
        new_lines = []
        import_section_end = -1

        # 找到导入结束位置
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                continue
            elif line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.strip().startswith('"""'):
                import_section_end = i
                break

        if import_section_end == -1:
            import_section_end = len(lines)

        # 添加标准头部
        header = '''"""
基础设施层测试文件

自动修复导入问题
"""

import pytest
import sys
import importlib
from pathlib import Path

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

'''

        # 移除旧的导入问题
        cleaned_lines = []
        for line in lines:
            # 跳过损坏的导入语句
            if any(skip_pattern in line for skip_pattern in [
                'src_infrastructure_',
                'importlib.import_module',
                'project_root_str',
                'sys.path.insert'
            ]):
                continue
            cleaned_lines.append(line)

        # 重组文件
        result = header + '\n'.join(cleaned_lines[import_section_end:])

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(result)

        print(f"✅ 修复完成: {file_path}")
        return True

    except Exception as e:
        print(f"❌ 修复失败 {file_path}: {e}")
        return False

def main():
    """
    主函数 - 修复基础设施层测试文件
    """
    infrastructure_test_dir = Path('tests/unit/infrastructure')

    if not infrastructure_test_dir.exists():
        print(f"❌ 测试目录不存在: {infrastructure_test_dir}")
        return

    fixed_count = 0
    total_count = 0

    # 递归处理所有Python文件
    for py_file in infrastructure_test_dir.rglob('*.py'):
        if py_file.name.startswith('test_'):
            total_count += 1
            if fix_test_file(py_file):
                fixed_count += 1

    print(f"\n📊 修复完成统计:")
    print(f"   总文件数: {total_count}")
    print(f"   修复文件数: {fixed_count}")
    print(".1f")

if __name__ == '__main__':
    main()
