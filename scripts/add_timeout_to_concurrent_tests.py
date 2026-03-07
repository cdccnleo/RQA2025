#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为所有涉及并发操作的测试文件添加超时设置
"""

import os
from pathlib import Path


def add_timeout_to_file(file_path, timeout_seconds=30):
    """为单个文件添加超时设置"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否已经有pytestmark设置
        if 'pytestmark =' in content:
            print(f"⚠️  {file_path}: 已有pytestmark设置，跳过")
            return False

        # 检查是否已经导入了pytest
        if 'import pytest' not in content:
            print(f"⚠️  {file_path}: 未导入pytest，跳过")
            return False

        # 找到import部分，在pytest import后添加超时设置
        lines = content.split('\n')
        insert_index = -1

        for i, line in enumerate(lines):
            if 'import pytest' in line:
                # 在import pytest后查找合适的插入位置
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == '' or lines[j].strip().startswith('#'):
                        continue
                    elif lines[j].strip().startswith('import ') or lines[j].strip().startswith('from '):
                        continue
                    else:
                        insert_index = j
                        break
                break

        if insert_index == -1:
            print(f"❌ {file_path}: 无法找到合适的插入位置")
            return False

        # 创建超时设置代码
        timeout_code = f"""
# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout({timeout_seconds}),  # {timeout_seconds}秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]
"""

        # 插入超时设置
        lines.insert(insert_index, timeout_code)

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(lines))

        print(f"✅ {file_path}: 添加了{timeout_seconds}秒超时设置")
        return True

    except Exception as e:
        print(f"❌ {file_path}: 处理出错 - {e}")
        return False


def find_concurrent_test_files():
    """查找所有涉及并发操作的测试文件"""
    test_root = Path("tests")

    if not test_root.exists():
        print("❌ 测试目录不存在")
        return []

    concurrent_files = []

    # 搜索包含并发相关关键词的文件
    concurrent_keywords = [
        'threading', 'concurrent', 'asyncio', 'async', 'await',
        'ThreadPoolExecutor', 'ProcessPoolExecutor', 'multiprocess',
        'parallel', 'distributed', 'queue', 'lock', 'semaphore'
    ]

    for py_file in test_root.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否包含并发关键词
            if any(keyword in content for keyword in concurrent_keywords):
                # 排除已经设置过的文件
                if 'pytestmark =' not in content:
                    concurrent_files.append(py_file)

        except Exception as e:
            print(f"⚠️  无法读取文件 {py_file}: {e}")

    return concurrent_files


def main():
    """主函数"""
    print("🔧 开始为并发测试文件添加超时设置")
    print("=" * 60)

    # 切换到项目根目录
    os.chdir(Path(__file__).parent.parent)

    # 查找并发测试文件
    concurrent_files = find_concurrent_test_files()

    print(f"找到 {len(concurrent_files)} 个并发测试文件需要设置超时")

    # 为每个文件添加超时设置
    success_count = 0
    for file_path in concurrent_files:
        timeout_seconds = 45 if 'ml' in str(file_path) or 'monitoring' in str(file_path) else 30
        if add_timeout_to_file(file_path, timeout_seconds):
            success_count += 1

    print(f"\n📊 处理结果:")
    print(f"总文件数: {len(concurrent_files)}")
    print(f"成功设置: {success_count}")
    print(f"失败/跳过: {len(concurrent_files) - success_count}")

    print("\n✅ 批量超时设置完成！")


if __name__ == "__main__":
    main()
