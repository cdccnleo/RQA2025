#!/usr/bin/env python3
"""
综合语法错误修复脚本
专门修复RQA2025项目中剩余的语法错误
"""

import os
import re


def fix_enhanced_health_checker():
    """专门修复enhanced_health_checker.py的语法错误"""
    file_path = "src/infrastructure/cache/enhanced_health_checker.py"

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    original_content = content

    # 修复字典初始化错误
    content = re.sub(
        r'return \{\}\s*\n(\s+)(\w+):',
        r'return {\n\1\2:',
        content
    )

    # 修复多行字典参数错误
    content = re.sub(
        r'(\w+)\s*=\s*\{\}\s*\n(\s+)(\'[^\']+\'|"[^"]+"|\w+):\s*([^,]+),\s*\n(\s+)(\'[^\\]+\'|"[^"]+"|\w+):\s*([^,]+),\s*\n(\s+)(\'[^\\]+\'|"[^"]+"|\w+):\s*([^,]+)',
        r'\1 = {\n\2\3: \4,\n\5\6: \7,\n\8\9: \10\n}',
        content
    )

    # 修复函数调用参数错误
    content = re.sub(
        r'threading\.Thread\(\)\s*\n(\s+)target=',
        r'threading.Thread(\n\1target=',
        content
    )

    # 修复HTML内容中的JavaScript错误
    content = re.sub(
        r'html\s*=\s*f"""\s*\n([^"]*?)""',
        r'html = f"""\1"""',
        content
    )

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 已修复: {file_path}")
        return True

    print(f"ℹ️ 无需修复: {file_path}")
    return False


def fix_remaining_files():
    """修复其他文件中的语法错误"""
    fix_patterns = [
        # 字符串字面量错误
        (r'\"\"\"\"', '\"\"\"'),

        # 字典初始化错误
        (r'(\w+)\s*=\s*\{\}\s*\n(\s+)(\w+):', r'\1 = {\n\2\3:'),

        # 函数定义缩进错误
        (r'    @abstractmethod\s*\n\s*\n(\w+)\s*\(', r'    @abstractmethod\n    def \1('),

        # 导入语句错误
        (r'from\s*\.\s*import\s*\(\)', r'from . import ()'),

        # 列表定义错误
        (r'__all__\s*=\s*\[\]\s*\n(\s+)(\w+)', r'__all__ = [\n\1\2'),

        # 函数参数错误
        (r'def\s+(\w+)\(([^)]*),?\)\s*\n(\s*)\)([^,]*),\s*\n(\s*)([^)]*)\):\s*\n',
         r'def \1(\2, \3\4, \5\6):\n'),
    ]

    files_to_fix = [
        "src/infrastructure/cache/enhanced_health_checker.py",
        "src/infrastructure/cache/websocket_api.py",
        "src/infrastructure/visual_monitor.py",
        "src/infrastructure/cache/dependency.py",
    ]

    total_fixed = 0

    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        original_content = content
        modified = False

        for pattern, replacement in fix_patterns:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                modified = True

        if modified and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已修复: {file_path}")
            total_fixed += 1
        else:
            print(f"ℹ️ 无需修复: {file_path}")

    return total_fixed


def validate_fixes():
    """验证修复结果"""
    print("\n🧪 验证修复结果...")

    test_files = [
        "src/infrastructure/cache/enhanced_health_checker.py",
        "src/infrastructure/cache/websocket_api.py",
        "src/infrastructure/visual_monitor.py",
        "src/infrastructure/cache/dependency.py",
    ]

    valid_files = 0
    total_files = len(test_files)

    for file_path in test_files:
        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✅ 语法正确: {file_path}")
            valid_files += 1
        except SyntaxError as e:
            print(f"❌ 语法错误: {file_path} - {e}")
        except Exception as e:
            print(f"⚠️ 其他错误: {file_path} - {e}")

    print(f"\n📊 验证结果: {valid_files}/{total_files} 个文件语法正确")
    return valid_files == total_files


def main():
    """主函数"""
    print("🔧 RQA2025综合语法错误修复工具")
    print("=" * 50)

    # 1. 修复enhanced_health_checker.py
    print("\n📝 步骤1: 修复enhanced_health_checker.py")
    fix_enhanced_health_checker()

    # 2. 修复其他文件
    print("\n📝 步骤2: 修复其他文件")
    fixed_count = fix_remaining_files()

    # 3. 验证修复结果
    print(f"\n📝 步骤3: 验证修复结果 (修复了 {fixed_count} 个文件)")

    if validate_fixes():
        print("\n🎉 所有语法错误已修复!")
        return True
    else:
        print("\n⚠️ 仍有语法错误需要手动修复")
        return False


if __name__ == "__main__":
    main()
