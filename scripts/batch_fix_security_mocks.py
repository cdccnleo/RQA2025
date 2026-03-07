"""
批量修复Security模块Mock配置

修复tests/unit/infrastructure/security/目录下的所有Mock配置问题
目标: 修复约67个Security模块测试失败
"""

import os
import re
from pathlib import Path


def fix_security_mock_in_file(file_path):
    """修复单个文件中的Security Mock配置"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 添加导入
    if 'from tests.fixtures.infrastructure_mocks import' not in content:
        import_match = re.search(r'^(import |from )', content, re.MULTILINE)
        if import_match:
            insert_pos = import_match.start()
            import_line = "from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock\n"
            content = content[:insert_pos] + import_line + content[insert_pos:]
    
    # 替换模式
    replacements = [
        (r'mock\s*=\s*Mock\(\)', 'mock = StandardMockBuilder.create_security_mock()'),
        (r'mock\s*=\s*MagicMock\(\)', 'mock = StandardMockBuilder.create_security_mock()'),
        (r'mock_checker\s*=\s*Mock\(\)', 'mock_checker = StandardMockBuilder.create_security_mock()'),
        (r'mock_checker\s*=\s*MagicMock\(\)', 'mock_checker = StandardMockBuilder.create_security_mock()'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def batch_fix_security_directory():
    """批量修复security目录"""
    security_test_dir = Path('tests/unit/infrastructure/security')
    
    if not security_test_dir.exists():
        return []
    
    fixed_files = []
    for test_file in security_test_dir.rglob('test_*.py'):
        if fix_security_mock_in_file(test_file):
            fixed_files.append(test_file.name)
            print(f"✅ 修复: {test_file.name}")
    
    print(f"\n修复完成！总计: {len(fixed_files)}个文件")
    return fixed_files


if __name__ == '__main__':
    print("开始批量修复Security模块...")
    fixed = batch_fix_security_directory()


