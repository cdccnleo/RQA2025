#!/usr/bin/env python3
"""
修复缓存测试文件的路径配置问题

缓存目录中的测试文件路径计算错误，导致pytest无法正确收集测试。
"""

import os
import re
from pathlib import Path


def fix_test_file_paths():
    """修复测试文件的路径配置"""

    cache_test_dir = Path("tests/unit/infrastructure/cache")

    if not cache_test_dir.exists():
        print("❌ 找不到缓存测试目录")
        return

    fixed_count = 0

    # 遍历所有测试文件
    for test_file in cache_test_dir.glob("test_*.py"):
        if test_file.name == "test_cache_exceptions.py":
            continue  # 这个文件路径是正确的

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否包含错误的路径配置
            if 'parent.parent.parent.parent.parent' in content:
                print(f"🔧 修复文件: {test_file.name}")

                # 替换错误的路径配置
                old_path = 'project_root = Path(__file__).resolve().parent.parent.parent.parent.parent'
                new_path = 'project_root = Path(__file__).resolve().parent.parent.parent.parent'

                new_content = content.replace(old_path, new_path)

                # 写回文件
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                fixed_count += 1

        except Exception as e:
            print(f"❌ 处理文件 {test_file.name} 时出错: {e}")

    print(f"✅ 修复了 {fixed_count} 个测试文件的路径配置")


def test_fixed_files():
    """测试修复后的文件是否能被正确收集"""

    import subprocess
    import sys

    try:
        # 运行pytest收集测试
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            'tests/unit/infrastructure/cache/',
            '--collect-only', '--quiet'
        ], capture_output=True, text=True, timeout=30)

        # 分析输出
        lines = result.stdout.strip().split('\n')
        test_count = 0

        for line in lines:
            if 'tests collected' in line:
                # 提取测试数量
                import re
                match = re.search(r'(\d+) tests? collected', line)
                if match:
                    test_count = int(match.group(1))
                break

        print(f"📊 修复后可收集的测试数量: {test_count}")
        return test_count

    except Exception as e:
        print(f"❌ 测试收集失败: {e}")
        return 0


if __name__ == "__main__":
    print("🔧 开始修复缓存测试文件的路径配置...")
    fix_test_file_paths()

    print("\n🧪 测试修复效果...")
    test_count = test_fixed_files()

    if test_count > 13:
        print(f"✅ 修复成功！测试数量从13个增加到{test_count}个")
    else:
        print(f"⚠️ 修复效果有限，测试数量: {test_count}")
