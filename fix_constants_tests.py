#!/usr/bin/env python3
"""
修复constants模块测试文件的导入问题
"""

import os
from pathlib import Path

def fix_constants_test_file(file_path):
    """
    修复单个constants测试文件
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否已经有正确的导入结构
        if 'import src.infrastructure.constants' in content and 'sys.path' in content:
            print(f"⏭️ 已修复: {file_path}")
            return False

        lines = content.split('\n')
        new_lines = []

        # 添加正确的头部
        header = '''"""
基础设施层常量测试文件
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

# 导入constants模块
try:
    import src.infrastructure.constants as constants_module
except ImportError:
    pytest.skip("基础设施模块导入失败", allow_module_level=True)

'''

        # 清理旧内容，移除重复的导入和路径设置
        cleaned_lines = []
        for line in lines:
            # 跳过旧的导入和注释
            if any(skip_pattern in line for skip_pattern in [
                'import pytest', 'import sys', 'import importlib', 'from pathlib import Path',
                '# 确保Python路径正确配置', 'project_root = Path', 'src_path_str =',
                'sys.path.insert', '# 动态导入模块', 'try:', 'except ImportError:',
                'pytest.skip(', '"""基础设施层测试文件"""'
            ]):
                continue
            cleaned_lines.append(line)

        # 重组文件
        new_content = header + '\n'.join(cleaned_lines)

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"✅ 修复完成: {file_path}")
        return True

    except Exception as e:
        print(f"❌ 修复失败 {file_path}: {e}")
        return False

def main():
    """
    主函数 - 修复constants模块所有测试文件
    """
    constants_test_dir = Path('tests/unit/infrastructure/constants')

    if not constants_test_dir.exists():
        print(f"❌ 测试目录不存在: {constants_test_dir}")
        return

    fixed_count = 0
    total_count = 0

    # 处理所有测试文件
    for py_file in constants_test_dir.glob('test_*.py'):
        total_count += 1
        if fix_constants_test_file(py_file):
            fixed_count += 1

    print(f"\n📊 修复完成统计:")
    print(f"   总文件数: {total_count}")
    print(f"   修复文件数: {fixed_count}")
    print(".1f")

if __name__ == '__main__':
    main()
