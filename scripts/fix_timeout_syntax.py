#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复超时装饰器语法错误
将错误位置的装饰器移动到正确位置
"""

import re
from pathlib import Path


def fix_timeout_syntax(root_path: str) -> int:
    """修复超时装饰器语法错误"""
    fixed_count = 0
    root_dir = Path(root_path)

    # 查找所有测试文件
    for py_file in root_dir.rglob('test_*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(py_file, 'r', encoding='latin-1') as f:
                    content = f.read()
            except UnicodeDecodeError:
                continue

        # 检查是否有错误的装饰器位置
        if '@pytest.mark.timeout(' in content:
            lines = content.split('\n')
            modified = False

            for i, line in enumerate(lines):
                # 查找错误的装饰器位置 (在类定义之后)
                if line.strip().startswith('class ') and 'Test' in line.strip():
                    # 检查下一行是否有装饰器
                    if i + 1 < len(lines) and '@pytest.mark.timeout(' in lines[i + 1]:
                        # 找到错误位置，修复它
                        # 移除错误的装饰器
                        del lines[i + 1]
                        # 在类定义前添加正确的装饰器
                        timeout_match = re.search(
                            r'@pytest\.mark\.timeout\((\d+)\)', lines[i + 1] if i + 1 < len(lines) else "")
                        if timeout_match:
                            timeout_value = timeout_match.group(1)
                            lines.insert(i, f"@pytest.mark.timeout({timeout_value})")
                            lines.insert(i + 1, "")  # 添加空行
                            modified = True
                            break

            if modified:
                # 写入修复后的内容
                try:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                    fixed_count += 1
                    print(f"Fixed: {py_file.name}")
                except Exception as e:
                    print(f"Failed to write {py_file}: {e}")

    return fixed_count


def main():
    """主函数"""
    print("开始修复超时装饰器语法错误...")

    fixed_count = fix_timeout_syntax("tests/unit/infrastructure")

    print(f"成功修复 {fixed_count} 个文件")

    if fixed_count > 0:
        print("\n建议重新运行测试以验证修复效果")


if __name__ == "__main__":
    main()
