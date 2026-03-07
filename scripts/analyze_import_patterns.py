#!/usr/bin/env python3
"""
分析导入模式工具

详细分析测试文件中的导入路径模式
"""

import os
from pathlib import Path
from collections import defaultdict


def analyze_import_patterns():
    """分析导入模式"""

    tests_dir = Path('tests')
    import_patterns = defaultdict(int)
    import_examples = defaultdict(list)

    print("🔍 分析测试文件导入模式...")

    for root, dirs, files in os.walk(tests_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 查找完整的导入语句
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line.startswith(('from src.', 'import src.')):
                            # 记录完整的导入语句
                            import_patterns[line] += 1

                            # 记录前几个示例
                            if len(import_examples[line]) < 3:
                                import_examples[line].append(str(file_path.relative_to('.')))

                except Exception as e:
                    print(f"❌ 读取文件失败 {file_path}: {e}")

    # 打印分析结果
    print(f"\n📊 发现 {len(import_patterns)} 种不同的导入模式")

    print("\n🔍 最常见的导入模式:")
    sorted_patterns = sorted(import_patterns.items(), key=lambda x: x[1], reverse=True)

    for i, (pattern, count) in enumerate(sorted_patterns[:20], 1):
        print(f"{i:2d}. {pattern} ({count} 次)")
        if pattern in import_examples:
            for example in import_examples[pattern][:2]:
                print(f"      示例: {example}")

    print("\n📋 按模块分类的导入模式:")
    module_patterns = defaultdict(list)

    for pattern in import_patterns:
        if 'from src.' in pattern:
            parts = pattern.replace('from src.', '').split('.')
            if parts:
                module = parts[0]
                module_patterns[module].append(pattern)

    for module, patterns in sorted(module_patterns.items()):
        print(f"\n{module.upper()} 模块 ({len(patterns)} 个导入):")
        pattern_counts = [(p, import_patterns[p]) for p in patterns]
        pattern_counts.sort(key=lambda x: x[1], reverse=True)

        for pattern, count in pattern_counts[:5]:
            print(f"  - {pattern} ({count} 次)")

    return import_patterns, import_examples


if __name__ == "__main__":
    analyze_import_patterns()
