#!/usr/bin/env python3
"""
修复接口继承问题

自动修复关键的架构一致性问题
"""

import json
import re
from pathlib import Path


def fix_interface_inheritance():
    """修复接口继承问题"""
    print('🔧 开始修复接口继承问题')

    # 加载报告
    with open('infrastructure_code_review_report.json', 'r', encoding='utf-8') as f:
        report = json.load(f)

    # 获取前10个问题
    issues = report['detailed_results']['architecture_compliance']['interface_inheritance']['issues'][:10]

    fixed_count = 0
    failed_count = 0

    for issue in issues:
        file_path = Path('src/infrastructure') / issue['file'].replace('\\', '/')
        class_name = issue['class']
        expected_base = issue['expected_base']

        try:
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')

                # 检查是否需要修复
                if f'class {class_name}' in content and expected_base not in content:
                    print(f'🔧 修复: {class_name} -> {expected_base}')

                    # 添加继承
                    pattern = f'class {class_name}\\('
                    replacement = f'class {class_name}({expected_base}, '

                    if re.search(pattern, content):
                        content = re.sub(pattern, replacement, content)

                        # 写回文件
                        file_path.write_text(content, encoding='utf-8')
                        fixed_count += 1
                        print(f'✅ 修复成功: {file_path}')
                    else:
                        print(f'⚠️ 未找到类定义: {class_name}')
                        failed_count += 1
                else:
                    print(f'ℹ️ 已正确继承: {class_name}')
                    fixed_count += 1
            else:
                print(f'❌ 文件不存在: {file_path}')
                failed_count += 1

        except Exception as e:
            print(f'❌ 修复失败 {class_name}: {e}')
            failed_count += 1

    print(f'\\n📊 修复结果: {fixed_count} 成功, {failed_count} 失败')
    return fixed_count, failed_count


if __name__ == "__main__":
    fix_interface_inheritance()
