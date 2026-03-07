#!/usr/bin/env python3
"""
简化的语法错误修复脚本
专门修复RQA2025项目中的关键语法错误
"""

import os
import re


def fix_file_with_patterns(file_path, patterns):
    """使用正则表达式模式修复文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    original_content = content
    modified = False

    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        if new_content != content:
            content = new_content
            modified = True

    if modified and content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 已修复: {file_path}")
        return True

    print(f"ℹ️ 无需修复: {file_path}")
    return False


def fix_enhanced_health_checker():
    """修复enhanced_health_checker.py"""
    file_path = "src/infrastructure/cache/enhanced_health_checker.py"

    patterns = [
        # 修复字典初始化错误
        (r'return \{\}\s*\n(\s+)(\w+):', r'return {\n\1\2:'),

        # 修复多行字典错误
        (r'(\w+)\s*=\s*\{\}\s*\n(\s+)(\w+):\s*([^,\n]+),\s*\n(\s+)(\w+):\s*([^,\n]+),\s*\n(\s+)(\w+):\s*([^,\n]+)',
         r'\1 = {\n\2\3: \4,\n\5\6: \7,\n\8\9: \10\n}'),

        # 修复函数调用错误
        (r'threading\.Thread\(\)\s*\n(\s+)target=', r'threading.Thread(\n\1target='),

        # 修复字符串字面量错误
        (r'\"\"\"\"', '\"\"\"'),
    ]

    return fix_file_with_patterns(file_path, patterns)


def fix_websocket_api():
    """修复websocket_api.py"""
    file_path = "src/infrastructure/cache/websocket_api.py"

    patterns = [
        # 修复字符串字面量错误
        (r'\"\"\"\"', '\"\"\"'),

        # 修复字典初始化错误
        (r'return \{\}\s*\n(\s+)(\w+):', r'return {\n\1\2:'),

        # 修复HTML内容错误
        (r'html\s*=\s*f"""\s*\n([^"]*?)"""', r'html = f"""\1"""'),
    ]

    return fix_file_with_patterns(file_path, patterns)


def fix_visual_monitor():
    """修复visual_monitor.py"""
    file_path = "src/infrastructure/visual_monitor.py"

    patterns = [
        # 修复字符串字面量错误
        (r'\"\"\"\"', '\"\"\"'),

        # 修复字典错误
        (r'(\w+)\s*=\s*\{\}\s*\n(\s+)(\w+):', r'\1 = {\n\2\3:'),

        # 修复dataclass错误
        (r'(\w+)\s*=\s*dataclass\(\)\s*\n(\s+)(\w+):', r'\1 = dataclass(\n\2\3:'),
    ]

    return fix_file_with_patterns(file_path, patterns)


def fix_cache_dependency():
    """修复dependency.py"""
    file_path = "src/infrastructure/cache/dependency.py"

    patterns = [
        # 修复字符串字面量错误
        (r'\"\"\"\"', '\"\"\"'),

        # 修复方法定义错误
        (r'    def\s+(\w+)\([^)]*\):\s*\n(\s+)"""', r'    def \1():\n\2"""'),
    ]

    return fix_file_with_patterns(file_path, patterns)


def validate_files():
    """验证文件语法"""
    print("\n🧪 验证文件语法...")

    files_to_check = [
        "src/infrastructure/cache/enhanced_health_checker.py",
        "src/infrastructure/cache/websocket_api.py",
        "src/infrastructure/visual_monitor.py",
        "src/infrastructure/cache/dependency.py",
    ]

    valid_count = 0

    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✅ 语法正确: {file_path}")
            valid_count += 1
        except SyntaxError as e:
            print(f"❌ 语法错误: {file_path} - {e}")
        except Exception as e:
            print(f"⚠️ 其他错误: {file_path} - {e}")

    return valid_count


def main():
    """主函数"""
    print("🔧 RQA2025简化语法错误修复工具")
    print("=" * 50)

    # 修复各个文件
    print("\n📝 修复enhanced_health_checker.py...")
    fix_enhanced_health_checker()

    print("\n📝 修复websocket_api.py...")
    fix_websocket_api()

    print("\n📝 修复visual_monitor.py...")
    fix_visual_monitor()

    print("\n📝 修复dependency.py...")
    fix_cache_dependency()

    # 验证结果
    valid_count = validate_files()
    print(f"\n📊 验证结果: {valid_count}/4 个文件语法正确")

    if valid_count == 4:
        print("\n🎉 所有关键文件语法错误已修复!")
        return True
    else:
        print("\n⚠️ 仍有语法错误需要修复")
        return False


if __name__ == "__main__":
    main()
