#!/usr/bin/env python3
"""
简单测试：直接测试一个具体文件
"""

import re

# 测试一个具体文件
test_file = "src/data/infrastructure_integration_manager.py"

print(f"测试文件: {test_file}")
print("=" * 50)

try:
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"文件长度: {len(content)} 字符")
    lines = content.split('\n')
    print(f"行数: {len(lines)}")

    # 查找基础设施导入语句
    import_lines = []
    for line in content.split('\n'):
        line = line.strip()
        if 'from src.infrastructure' in line:
            import_lines.append(line)

    print(f"\n发现基础设施导入语句 ({len(import_lines)}个):")
    for i, line in enumerate(import_lines):
        print(f"{i+1}. {line}")

    # 测试匹配
    print(f"\n测试匹配:")

    # 测试带缩进的匹配
    indented_patterns = [
        r'^\s*from src\.infrastructure\.cache\.unified_cache import UnifiedCacheManager',
        r'^\s*from src\.infrastructure\.config\.unified_manager import UnifiedConfigManager',
        r'^\s*from src\.infrastructure\.logging import UnifiedLogger'
    ]

    for pattern in indented_patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        if matches:
            print(f"✅ 匹配 '{pattern}': {len(matches)} 次")
            for match in matches:
                print(f"    匹配内容: '{match}'")
        else:
            print(f"❌ 未匹配 '{pattern}'")

    # 测试替换
    print(f"\n测试替换:")
    test_content = content  # 使用完整内容
    new_content = test_content

    replacement_map = [
        (r'^\s*from src\.infrastructure\.cache\.unified_cache import UnifiedCacheManager', 'from src.core.integration import get_data_adapter'),
        (r'^\s*from src\.infrastructure\.config\.unified_manager import UnifiedConfigManager', 'from src.core.integration import get_data_adapter'),
        (r'^\s*from src\.infrastructure\.logging import UnifiedLogger', 'from src.core.integration import get_data_adapter')
    ]

    replacements_made = 0
    for old_pattern, new_text in replacement_map:
        new_content, count = re.subn(old_pattern, new_text, new_content, flags=re.MULTILINE)
        if count > 0:
            replacements_made += count
            print(f"✅ 替换模式 '{old_pattern}' -> '{new_text}': {count} 次")

    print(f"\n总共替换次数: {replacements_made}")

    # 显示替换效果
    if replacements_made > 0:
        print(f"\n替换效果示例:")
        original_lines = content.split('\n')
        new_lines = new_content.split('\n')

        for i, (orig, new) in enumerate(zip(original_lines, new_lines)):
            if orig != new and 'from src.infrastructure' in orig:
                print(f"第{i+1}行:")
                print(f"  旧: {orig}")
                print(f"  新: {new}")
                print()

except Exception as e:
    print(f"错误: {e}")

print("\n测试完成")
