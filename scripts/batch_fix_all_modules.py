"""
批量修复所有基础设施模块的Mock配置

一次性修复所有剩余模块
"""

import os
import re
from pathlib import Path


def fix_mock_in_file(file_path, mock_type='cache'):
    """修复单个文件中的Mock配置"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ 读取失败: {file_path.name} - {e}")
        return False
    
    original_content = content
    
    # 添加导入
    if 'from tests.fixtures.infrastructure_mocks import' not in content:
        import_match = re.search(r'^(import |from )', content, re.MULTILINE)
        if import_match:
            insert_pos = import_match.start()
            import_line = "from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock\n"
            content = content[:insert_pos] + import_line + content[insert_pos:]
    
    # 通用替换模式
    replacements = [
        (r'mock\s*=\s*Mock\(\)', f'mock = StandardMockBuilder.create_{mock_type}_mock()'),
        (r'mock\s*=\s*MagicMock\(\)', f'mock = StandardMockBuilder.create_{mock_type}_mock()'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"⚠️ 写入失败: {file_path.name} - {e}")
            return False
    
    return False


def batch_fix_module(module_name, mock_type='cache'):
    """批量修复指定模块"""
    module_test_dir = Path(f'tests/unit/infrastructure/{module_name}')
    
    if not module_test_dir.exists():
        print(f"⚪ 目录不存在: {module_test_dir}")
        return []
    
    fixed_files = []
    total_files = 0
    
    for test_file in module_test_dir.rglob('test_*.py'):
        total_files += 1
        if fix_mock_in_file(test_file, mock_type):
            fixed_files.append(test_file.name)
    
    print(f"\n{module_name}模块:")
    print(f"  总文件: {total_files}")
    print(f"  修复数: {len(fixed_files)}")
    print(f"  修复率: {len(fixed_files)/total_files*100:.1f}%" if total_files > 0 else "  修复率: N/A")
    
    return fixed_files


if __name__ == '__main__':
    print("批量修复所有基础设施模块...")
    print("=" * 70)
    
    modules_to_fix = [
        ('api', 'config'),
        ('logging', 'logger'),
        ('monitoring', 'monitor'),
        ('resource', 'resource'),
        ('utils', 'cache'),
        ('distributed', 'distributed'),
        ('health', 'health'),
        ('error', 'cache'),
        ('versioning', 'config'),
    ]
    
    total_fixed = 0
    for module_name, mock_type in modules_to_fix:
        print(f"\n处理 {module_name} 模块...")
        fixed = batch_fix_module(module_name, mock_type)
        total_fixed += len(fixed)
    
    print("=" * 70)
    print(f"\n✅ 批量修复完成！总计修复: {total_fixed}个文件")


