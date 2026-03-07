#!/usr/bin/env python3
"""
修复测试类中的__init__构造函数问题
pytest不推荐测试类有__init__构造函数
"""
import os
import re
from pathlib import Path

def find_test_classes_with_init(test_dir):
    """查找所有有__init__构造函数的测试类"""
    classes_with_init = []

    for py_file in Path(test_dir).rglob('*.py'):
        try:
            content = py_file.read_text(encoding='utf-8')
            lines = content.split('\n')

            # 查找类定义和__init__方法
            in_class = False
            current_class = None
            class_line = None

            for i, line in enumerate(lines):
                # 查找类定义
                class_match = re.match(r'^\s*class\s+(\w+)\s*\(', line)
                if class_match:
                    current_class = class_match.group(1)
                    class_line = i
                    in_class = True
                    continue

                # 如果找到__init__方法
                if in_class and re.match(r'^\s*def\s+__init__\s*\(', line):
                    classes_with_init.append({
                        'file': str(py_file),
                        'class': current_class,
                        'line': class_line + 1,
                        'init_line': i + 1
                    })
                    break

        except Exception as e:
            print(f'Error reading {py_file}: {e}')

    return classes_with_init

def fix_test_class(content, class_info):
    """修复测试类"""
    lines = content.split('\n')

    # 找到类定义行
    class_line_idx = class_info['line'] - 1
    init_line_idx = class_info['init_line'] - 1

    # 检查类是否继承自测试基类以外的类
    class_line = lines[class_line_idx]
    if 'TestCase' in class_line or 'unittest.TestCase' in class_line:
        # 这是unittest.TestCase的子类，需要__init__
        return content

    # 检查__init__方法的内容
    init_start = init_line_idx
    init_end = init_start

    # 找到__init__方法的结束
    for i in range(init_start, len(lines)):
        line = lines[i]
        if line.strip().startswith('def ') and i > init_start:
            init_end = i - 1
            break
        elif i == len(lines) - 1:
            init_end = i
            break

    # 提取__init__方法的内容
    init_content = '\n'.join(lines[init_start:init_end+1])

    # 如果__init__只是调用super().__init__()，可以移除
    if 'super().__init__()' in init_content and len(init_content.split('\n')) <= 3:
        # 移除__init__方法
        new_lines = lines[:init_start] + lines[init_end+1:]
        return '\n'.join(new_lines)

    # 如果__init__有其他逻辑，需要重构为setup方法
    # 这里简化处理，只移除空的__init__
    return content

def main():
    test_dir = "tests/unit/infrastructure"
    classes_with_init = find_test_classes_with_init(test_dir)

    print(f"找到 {len(classes_with_init)} 个有__init__的测试类")

    for class_info in classes_with_init[:5]:  # 只处理前5个作为示例
        print(f"处理: {class_info['file']}:{class_info['class']}")

        try:
            with open(class_info['file'], 'r', encoding='utf-8') as f:
                content = f.read()

            fixed_content = fix_test_class(content, class_info)

            if fixed_content != content:
                with open(class_info['file'], 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"  已修复: {class_info['class']}")
            else:
                print(f"  跳过: {class_info['class']} (需要手动处理)")

        except Exception as e:
            print(f"  错误处理 {class_info['class']}: {e}")

if __name__ == "__main__":
    main()
