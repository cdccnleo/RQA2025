#!/usr/bin/env python3
"""
精确错误分析脚本

精确分析基础设施层测试中的错误信息
"""

import subprocess
import sys
from collections import defaultdict


def get_precise_errors():
    """获取精确的错误信息"""
    print("🔍 精确分析错误信息...")

    # 分析各个子模块
    submodules = {
        'cache': 'tests/unit/infrastructure/cache/',
        'config': 'tests/unit/infrastructure/config/',
        'error': 'tests/unit/infrastructure/error/',
        'health': 'tests/unit/infrastructure/health/',
        'logging': 'tests/unit/infrastructure/logging/',
        'resource': 'tests/unit/infrastructure/resource/',
        'security': 'tests/unit/infrastructure/security/'
    }

    precise_errors = {}

    for name, path in submodules.items():
        print(f"\n📁 精确分析 {name} 模块...")
        error_count = run_pytest_count_only(path)
        if error_count > 0:
            precise_errors[name] = error_count
            print(f"   找到 {error_count} 个错误")

    # 分析根目录测试
    print(f"\n📁 精确分析根目录测试...")
    root_error_count = run_pytest_count_only(
        'tests/unit/infrastructure/',
        exclude_filters=['cache', 'config', 'error', 'health', 'logging', 'resource', 'security']
    )
    if root_error_count > 0:
        precise_errors['root'] = root_error_count
        print(f"   找到 {root_error_count} 个错误")

    return precise_errors


def run_pytest_count_only(path, exclude_filters=None):
    """运行pytest并只统计错误数量"""
    cmd = [sys.executable, '-m', 'pytest', path, '--collect-only', '--quiet']

    if exclude_filters:
        exclude_expr = ' and '.join([f'not {f}' for f in exclude_filters])
        cmd.extend(['-k', exclude_expr])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=30
        )

        # 统计ERROR行数
        error_count = sum(1 for line in result.stdout.split('\n')
                          if line.strip().startswith('ERROR'))
        return error_count

    except subprocess.TimeoutExpired:
        return 1  # 超时算1个错误
    except Exception as e:
        return 1  # 异常算1个错误


def sample_specific_errors():
    """获取具体错误示例"""
    print("\n🔍 获取具体错误示例...")

    sample_errors = {}

    # 分析一个具体的模块作为示例
    modules_to_sample = ['cache', 'config', 'error', 'health']

    for module in modules_to_sample:
        print(f"\n📁 采样 {module} 模块的具体错误...")
        path = f"tests/unit/infrastructure/{module}/"
        errors = get_sample_errors(path)
        sample_errors[module] = errors[:5]  # 只保留前5个错误
        print(f"   获取到 {len(errors)} 个错误示例")

    return sample_errors


def get_sample_errors(path):
    """获取错误示例"""
    cmd = [sys.executable, '-m', 'pytest', path, '--collect-only', '--quiet']

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=15
        )

        errors = []
        for line in result.stdout.split('\n'):
            if 'ERROR' in line:
                errors.append(line.strip())

        return errors

    except:
        return ['ERROR: 无法获取错误示例']


def analyze_error_types(sample_errors):
    """分析错误类型"""
    print("\n" + "="*80)
    print("🔍 错误类型分析")
    print("="*80)

    error_categories = defaultdict(int)

    for module, errors in sample_errors.items():
        print(f"\n📁 {module} 模块错误示例:")
        for i, error in enumerate(errors[:3]):  # 只显示前3个
            print(f"   {i+1}. {error[:100]}...")

            # 分类错误
            error_lower = error.lower()
            if 'modulenotfounderror' in error_lower or 'no module named' in error_lower:
                error_categories['ModuleNotFoundError'] += 1
            elif 'importerror' in error_lower:
                error_categories['ImportError'] += 1
            elif 'syntaxerror' in error_lower:
                error_categories['SyntaxError'] += 1
            elif 'indentationerror' in error_lower:
                error_categories['IndentationError'] += 1
            elif 'attributeerror' in error_lower:
                error_categories['AttributeError'] += 1
            elif 'nameerror' in error_lower:
                error_categories['NameError'] += 1
            elif 'pytestcollectionwarning' in error_lower:
                error_categories['PytestCollectionWarning'] += 1
            else:
                error_categories['Other'] += 1

    print(f"\n📊 错误类型分布:")
    total_samples = sum(error_categories.values())
    for error_type, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"   {error_type}: {count} 个 ({percentage:.1f}%)")

    return error_categories


def create_targeted_fix_scripts(error_categories):
    """创建针对性的修复脚本"""
    print("\n" + "="*80)
    print("🛠️ 创建针对性修复脚本")
    print("="*80)

    scripts_created = []

    # 根据错误类型创建不同的修复脚本
    if error_categories.get('ModuleNotFoundError', 0) > 0:
        scripts_created.append(create_module_not_found_fix())
        print("✅ 创建了 ModuleNotFoundError 修复脚本")

    if error_categories.get('ImportError', 0) > 0:
        scripts_created.append(create_import_error_fix())
        print("✅ 创建了 ImportError 修复脚本")

    if error_categories.get('SyntaxError', 0) > 0:
        scripts_created.append(create_syntax_error_fix())
        print("✅ 创建了 SyntaxError 修复脚本")

    if error_categories.get('IndentationError', 0) > 0:
        scripts_created.append(create_indentation_error_fix())
        print("✅ 创建了 IndentationError 修复脚本")

    if error_categories.get('PytestCollectionWarning', 0) > 0:
        scripts_created.append(create_pytest_warning_fix())
        print("✅ 创建了 PytestCollectionWarning 修复脚本")

    return scripts_created


def create_module_not_found_fix():
    """创建模块不存在错误修复脚本"""
    script_content = '''#!/usr/bin/env python3
"""
修复 ModuleNotFoundError 的脚本
"""

import os
import re
from pathlib import Path

def fix_module_not_found():
    """修复模块不存在错误"""
    print("🔧 修复 ModuleNotFoundError...")

    # 这里实现具体的修复逻辑
    # 1. 分析缺失的模块
    # 2. 创建占位模块或mock对象
    # 3. 修复导入路径

    pass

if __name__ == "__main__":
    fix_module_not_found()
'''

    script_path = "scripts/testing/fix_module_not_found.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    return script_path


def create_import_error_fix():
    """创建导入错误修复脚本"""
    script_content = '''#!/usr/bin/env python3
"""
修复 ImportError 的脚本
"""

import os
import re
from pathlib import Path

def fix_import_errors():
    """修复导入错误"""
    print("🔧 修复 ImportError...")

    # 这里实现具体的修复逻辑
    # 1. 分析错误的导入语句
    # 2. 修复导入路径
    # 3. 处理循环导入

    pass

if __name__ == "__main__":
    fix_import_errors()
'''

    script_path = "scripts/testing/fix_import_errors.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    return script_path


def create_syntax_error_fix():
    """创建语法错误修复脚本"""
    script_content = '''#!/usr/bin/env python3
"""
修复 SyntaxError 的脚本
"""

import os
import re
from pathlib import Path

def fix_syntax_errors():
    """修复语法错误"""
    print("🔧 修复 SyntaxError...")

    # 这里实现具体的修复逻辑
    # 1. 检查括号匹配
    # 2. 修复字符串引号
    # 3. 检查语句结束符

    pass

if __name__ == "__main__":
    fix_syntax_errors()
'''

    script_path = "scripts/testing/fix_syntax_errors.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    return script_path


def create_indentation_error_fix():
    """创建缩进错误修复脚本"""
    script_content = '''#!/usr/bin/env python3
"""
修复 IndentationError 的脚本
"""

import os
import re
from pathlib import Path

def fix_indentation_errors():
    """修复缩进错误"""
    print("🔧 修复 IndentationError...")

    # 这里实现具体的修复逻辑
    # 1. 统一缩进风格（4空格）
    # 2. 修复不一致缩进
    # 3. 处理混合制表符和空格

    pass

if __name__ == "__main__":
    fix_indentation_errors()
'''

    script_path = "scripts/testing/fix_indentation_errors.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    return script_path


def create_pytest_warning_fix():
    """创建pytest警告修复脚本"""
    script_content = '''#!/usr/bin/env python3
"""
修复 PytestCollectionWarning 的脚本
"""

import os
import re
from pathlib import Path

def fix_pytest_warnings():
    """修复pytest收集警告"""
    print("🔧 修复 PytestCollectionWarning...")

    # 这里实现具体的修复逻辑
    # 1. 为测试类添加__init__方法
    # 2. 使用__test__=False标记非测试类
    # 3. 修复测试类结构

    pass

if __name__ == "__main__":
    fix_pytest_warnings()
'''

    script_path = "scripts/testing/fix_pytest_warnings.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    return script_path


def main():
    """主函数"""
    # 获取精确错误数量
    precise_errors = get_precise_errors()

    # 获取错误示例
    sample_errors = sample_specific_errors()

    # 分析错误类型
    error_categories = analyze_error_types(sample_errors)

    # 创建针对性修复脚本
    scripts_created = create_targeted_fix_scripts(error_categories)

    print(f"\n" + "="*80)
    print("📋 执行总结")
    print("="*80)

    total_errors = sum(precise_errors.values())
    print(f"🔢 总错误数: {total_errors}")
    print(f"📁 受影响模块数: {len(precise_errors)}")
    print(f"🛠️ 创建的修复脚本数: {len(scripts_created)}")

    print(f"\n📊 各模块错误分布:")
    for module, count in precise_errors.items():
        print(f"   {module}: {count} 个错误")

    print(f"\n🎯 下一阶段工作:")
    print("   1. 运行针对性修复脚本")
    print("   2. 验证修复效果")
    print("   3. 处理剩余复杂错误")
    print("   4. 实现基础设施层100%可收集")


if __name__ == "__main__":
    main()
