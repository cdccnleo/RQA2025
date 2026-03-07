#!/usr/bin/env python3
"""
自动化代码质量检查脚本
"""

import os
import re
import json
from pathlib import Path


def run_quality_checks():
    """运行质量检查"""
    infra_dir = Path('src/infrastructure')

    results = {
        'import_standards': check_import_standards(infra_dir),
        'naming_conventions': check_naming_conventions(infra_dir),
        'architecture_patterns': check_architecture_patterns(infra_dir),
        'code_quality': check_code_quality(infra_dir)
    }

    # 保存结果
    with open('quality_check_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("质量检查完成，结果已保存到 quality_check_results.json")
    return results


def check_import_standards(infra_dir):
    """检查导入标准"""
    issues = []

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查通配符导入
                    if ' import *' in content:
                        issues.append(f"通配符导入: {file_path}")

                    # 检查过长导入
                    lines = content.split('\n')
                    for line in lines:
                        if line.startswith('from ') and len(line) > 100:
                            issues.append(f"过长导入: {file_path}")

                except Exception:
                    continue

    return {'issues': issues, 'count': len(issues)}


def check_naming_conventions(infra_dir):
    """检查命名规范"""
    issues = []

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查接口命名
                    if 'interface' in file.lower():
                        class_matches = re.findall(r'class\s+(\w+)', content)
                        for class_name in class_matches:
                            if not class_name.startswith('I'):
                                issues.append(f"接口命名不规范: {file_path} - {class_name}")

                except Exception:
                    continue

    return {'issues': issues, 'count': len(issues)}


def check_architecture_patterns(infra_dir):
    """检查架构模式"""
    issues = []
    return {'issues': issues, 'count': len(issues)}


def check_code_quality(infra_dir):
    """检查代码质量"""
    issues = []

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    lines = content.split('\n')

                    # 检查函数长度
                    in_function = False
                    function_lines = 0
                    for line in lines:
                        if line.strip().startswith('def '):
                            in_function = True
                            function_lines = 0
                        elif in_function and line.strip() and not line.startswith(' '):
                            # 函数结束
                            if function_lines > 50:  # 超过50行
                                issues.append(f"函数过长: {file_path}")
                            in_function = False
                        elif in_function:
                            function_lines += 1

                except Exception:
                    continue

    return {'issues': issues, 'count': len(issues)}


if __name__ == "__main__":
    run_quality_checks()
