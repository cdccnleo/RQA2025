#!/usr/bin/env python3
"""
专门修复interfaces.py文件的缩进错误
"""


def fix_interfaces_file():
    with open('src/infrastructure/cache/interfaces.py', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 修复所有方法定义前的缩进错误
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 检查是否是@abstractmethod后跟空行然后是方法定义
        if line.strip() == '@abstractmethod' and i + 2 < len(lines):
            next_line = lines[i + 1]
            method_line = lines[i + 2]

            if next_line.strip() == '' and method_line.strip().startswith('def '):
                # 修复缩进
                fixed_lines.append('    @abstractmethod')
                fixed_lines.append('    def ' + method_line.strip()[4:])  # 去掉def和前面的空格
                i += 3
                continue

        fixed_lines.append(line)
        i += 1

    content = '\n'.join(fixed_lines)

    # 写回文件
    with open('src/infrastructure/cache/interfaces.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print('✅ 已修复 interfaces.py 的方法定义缩进错误')


if __name__ == "__main__":
    fix_interfaces_file()
