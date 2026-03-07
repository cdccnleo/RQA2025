#!/usr/bin/env python3
"""
批量修复测试导入错误脚本

修复数据层适配器模块的导入问题
"""

import os
import re
import glob
from pathlib import Path


def fix_adapter_imports():
    """修复数据适配器导入错误"""

    # 查找所有有问题的测试文件
    test_files = [
        'tests/unit/data/adapters/test_adapter_registry_edges2.py',
        'tests/unit/data/adapters/test_adapter_registry_real.py',
        'tests/unit/data/adapters/test_adapter_registry_unit.py',
        'tests/unit/data/adapters/test_base_adapter_real.py',
        'tests/unit/data/adapters/test_base_adapter_unit.py',
        'tests/unit/data/adapters/test_base_edges2.py',
        'tests/unit/data/adapters/test_china_base_adapter.py',
        'tests/unit/data/adapters/test_client_components_edges2.py',
        'tests/unit/data/adapters/test_connector_components_edges2.py',
        'tests/unit/data/adapters/test_db_client_edges2.py',
        'tests/unit/data/adapters/test_market_data_adapter_edges2.py',
        'tests/unit/data/adapters/test_provider_components_unit.py',
        'tests/unit/data/adapters/test_source_components_edges2.py',
        'tests/unit/data/china/test_adapters_impl.py',
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"修复文件: {file_path}")
            fix_single_file(file_path)


def fix_single_file(file_path):
    """修复单个文件"""

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复导入语句
    # 将 from src.data.adapters.xxx 改为直接导入
    patterns = [
        (r'from src\.data\.adapters\.(\w+) import',
         r'from data.adapters.\1 import'),
        (r'import src\.data\.adapters\.(\w+)',
         r'import data.adapters.\1'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    # 添加路径设置
    if 'import sys' not in content and 'sys.path.insert' not in content:
        # 在文件开头添加路径设置
        header_lines = []
        for line in content.split('\n'):
            header_lines.append(line)
            if line.startswith('import') or line.startswith('from'):
                break

        # 插入路径设置
        insert_pos = 0
        for i, line in enumerate(header_lines):
            if line.startswith('import') or line.startswith('from'):
                insert_pos = i
                break

        path_setup = '''import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

'''
        content_lines = content.split('\n')
        content_lines.insert(insert_pos, path_setup.rstrip())
        content = '\n'.join(content_lines)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def fix_strategy_test_errors():
    """修复策略层测试错误"""

    # 修复BaseStrategy构造函数调用
    strategy_files = [
        'tests/unit/strategy/test_strategy_coverage_boost_real.py'
    ]

    for file_path in strategy_files:
        if os.path.exists(file_path):
            print(f"修复策略文件: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 修复BaseStrategy构造函数调用
            content = re.sub(
                r'BaseStrategy\(config\)',
                r'BaseStrategy(\'test_strategy_001\', \'test_base_strategy\', \'base\')',
                content
            )

            # 修复参数设置方法调用
            content = re.sub(
                r'strategy\.set_parameter\([^,]+,\s*([^)]+)\)',
                r'strategy.set_parameters({\'\1\'})',
                content
            )

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)


if __name__ == '__main__':
    print("开始批量修复导入错误...")

    fix_adapter_imports()
    fix_strategy_test_errors()

    print("修复完成！")
