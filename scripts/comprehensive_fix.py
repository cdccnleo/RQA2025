#!/usr/bin/env python3
"""
全面修复standard_interfaces.py中的语法错误
"""

import re


def comprehensive_fix():
    """全面修复standard_interfaces.py中的语法错误"""
    file_path = "src/infrastructure/interfaces/standard_interfaces.py"

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 修复类定义格式错误
    content = re.sub(
        r'class\s+(\w+):\s*"""([^"]*)"""\s*\n\}',
        r'class \1:\n    """\2"""',
        content,
        flags=re.MULTILINE
    )

    # 修复不匹配的右大括号
    content = re.sub(r'\n}\s*\n', r'\n\n', content)

    # 修复多余的右大括号
    lines = content.split('\n')
    fixed_lines = []
    brace_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped == '}':
            if brace_count > 0:
                brace_count -= 1
                continue  # 跳过这个多余的右大括号
        elif stripped.startswith('class ') or stripped.startswith('@dataclass'):
            brace_count += 1

        fixed_lines.append(line)

    content = '\n'.join(fixed_lines)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print('✅ 已全面修复 standard_interfaces.py 的语法错误')


def validate_syntax():
    """验证语法"""
    try:
        with open('src/infrastructure/interfaces/standard_interfaces.py', 'r', encoding='utf-8') as f:
            compile(f.read(), 'standard_interfaces.py', 'exec')
        print('✅ standard_interfaces.py 语法正确')
        return True
    except SyntaxError as e:
        print(f'❌ 语法错误: {e}')
        return False


def test_import():
    """测试导入"""
    print("\n🧪 测试导入...")
    try:
        from src.infrastructure.config import ConfigFactory
        print("✅ ConfigFactory导入成功")

        manager = ConfigFactory.create_config_manager()
        print("✅ ConfigFactory.create_config_manager() 成功")
        print(f"管理器类型: {type(manager)}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    comprehensive_fix()
    validate_syntax()
    test_import()
