"""
批量修复Cache模块Mock配置

修复tests/unit/infrastructure/cache/目录下的所有Mock配置问题
创建日期: 2025-01-31
目标: 修复约360个Cache模块测试失败
"""

import os
import re
from pathlib import Path


def fix_cache_mock_in_file(file_path):
    """修复单个文件中的Cache Mock配置"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 在文件开头添加导入（如果还没有）
    if 'from tests.fixtures.infrastructure_mocks import' not in content:
        # 找到第一个import语句的位置
        import_match = re.search(r'^(import |from )', content, re.MULTILINE)
        if import_match:
            insert_pos = import_match.start()
            import_line = "from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock\n"
            content = content[:insert_pos] + import_line + content[insert_pos:]
    
    # 替换模式列表
    replacements = [
        # 简单Mock()替换为标准Mock
        (r'mock\s*=\s*Mock\(\)', 'mock = StandardMockBuilder.create_cache_mock()'),
        (r'mock\s*=\s*MagicMock\(\)', 'mock = StandardMockBuilder.create_cache_mock()'),
        
        # 缓存管理器Mock
        (r'mock_cache_manager\s*=\s*Mock\(\)', 'mock_cache_manager = StandardMockBuilder.create_cache_mock()'),
        (r'mock_cache_manager\s*=\s*MagicMock\(\)', 'mock_cache_manager = StandardMockBuilder.create_cache_mock()'),
        
        # L1/L2/L3 Mock
        (r'mock_l1\s*=\s*Mock\(\)', 'mock_l1 = StandardMockBuilder.create_cache_mock()'),
        (r'mock_l2\s*=\s*Mock\(\)', 'mock_l2 = StandardMockBuilder.create_cache_mock()'),
        (r'mock_l3\s*=\s*Mock\(\)', 'mock_l3 = StandardMockBuilder.create_cache_mock()'),
        
        # 缓存组件Mock
        (r'mock_component\s*=\s*Mock\(\)', 'mock_component = StandardMockBuilder.create_cache_mock()'),
        (r'mock_tier\s*=\s*Mock\(\)', 'mock_tier = StandardMockBuilder.create_cache_mock()'),
    ]
    
    # 执行替换
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # 如果内容有变化，写回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def batch_fix_cache_directory():
    """批量修复cache目录下的所有测试文件"""
    
    cache_test_dir = Path('tests/unit/infrastructure/cache')
    
    if not cache_test_dir.exists():
        print(f"目录不存在: {cache_test_dir}")
        return
    
    fixed_files = []
    total_files = 0
    
    for test_file in cache_test_dir.rglob('test_*.py'):
        total_files += 1
        if fix_cache_mock_in_file(test_file):
            fixed_files.append(test_file.name)
            print(f"✅ 修复: {test_file.name}")
        else:
            print(f"⚪ 跳过: {test_file.name}（无需修复）")
    
    print(f"\n修复完成！")
    print(f"总文件数: {total_files}")
    print(f"修复文件数: {len(fixed_files)}")
    print(f"修复率: {len(fixed_files)/total_files*100:.1f}%")
    
    return fixed_files


if __name__ == '__main__':
    print("开始批量修复Cache模块Mock配置...")
    print("=" * 70)
    fixed = batch_fix_cache_directory()
    print("=" * 70)
    print(f"\n修复的文件列表:")
    for f in fixed:
        print(f"  - {f}")

