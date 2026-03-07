#!/usr/bin/env python3
"""
修复剩余测试失败的系统性脚本

分析所有失败测试，识别模式并批量修复
"""

import os
import subprocess
import re
from pathlib import Path


def run_test_and_capture_failures():
    """运行测试并捕获失败信息"""
    print("🔍 运行测试并分析失败模式...")

    # 运行测试并捕获输出
    cmd = ['pytest', 'tests/unit/infrastructure/', '--tb=no', '--maxfail=50', '-q']
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')

    # 解析失败的测试
    failures = []
    lines = result.stdout.split('\n')
    for line in lines:
        if line.startswith('FAILED'):
            test_path = line.replace('FAILED ', '').strip()
            failures.append(test_path)

    print(f"发现 {len(failures)} 个失败测试")
    return failures


def categorize_failures(failures):
    """对失败进行分类"""
    categories = {
        'import_error': [],
        'attribute_error': [],
        'type_error': [],
        'assertion_error': [],
        'other': []
    }

    for failure in failures[:10]:  # 先分析前10个
        try:
            # 运行单个测试获取详细错误信息
            cmd = ['pytest', failure, '--tb=short', '-q']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')

            error_output = result.stdout + result.stderr

            if 'ImportError' in error_output or 'cannot import' in error_output:
                categories['import_error'].append((failure, error_output))
            elif 'AttributeError' in error_output:
                categories['attribute_error'].append((failure, error_output))
            elif 'TypeError' in error_output:
                categories['type_error'].append((failure, error_output))
            elif 'AssertionError' in error_output:
                categories['assertion_error'].append((failure, error_output))
            else:
                categories['other'].append((failure, error_output))

        except Exception as e:
            print(f"分析失败 {failure}: {e}")

    return categories


def fix_import_errors(import_failures):
    """修复导入错误"""
    print(f"\n🔧 修复导入错误 ({len(import_failures)} 个)...")

    fixes_applied = 0

    for failure, error_output in import_failures:
        test_file = failure.split('::')[0]

        # 分析错误信息
        if 'ConfigProvider' in error_output and 'standard_interfaces' in error_output:
            # 修复ConfigProvider导入
            fix_config_provider_import(test_file)
            fixes_applied += 1

        elif 'IConfigProvider' in error_output:
            # 可能需要从其他地方导入
            fix_iconfig_provider_import(test_file)
            fixes_applied += 1

    print(f"应用了 {fixes_applied} 个导入修复")
    return fixes_applied


def fix_config_provider_import(test_file):
    """修复ConfigProvider导入"""
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换错误的导入
        old_import = """from src.infrastructure.interfaces.standard_interfaces import (
            IServiceProvider,
            ICacheProvider,
            ConfigProvider
        )"""

        new_import = """from src.infrastructure.interfaces.standard_interfaces import (
            IServiceProvider,
            ICacheProvider,
            IConfigProvider
        )"""

        if old_import in content:
            content = content.replace(old_import, new_import)
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 修复了 {test_file} 中的ConfigProvider导入")

    except Exception as e:
        print(f"✗ 修复失败 {test_file}: {e}")


def fix_iconfig_provider_import(test_file):
    """修复IConfigProvider导入"""
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否需要从config.tools.provider导入
        if 'from src.infrastructure.config.tools.provider import' not in content:
            # 添加正确的导入
            import_line = "from src.infrastructure.config.tools.provider import IConfigProvider\n"

            # 在文件开头添加导入
            if content.startswith('#!/usr/bin/env python3'):
                lines = content.split('\n')
                # 在shebang后添加导入
                lines.insert(1, import_line.rstrip())
                content = '\n'.join(lines)
            else:
                content = import_line + content

            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 添加了 {test_file} 中的IConfigProvider导入")

    except Exception as e:
        print(f"✗ 修复失败 {test_file}: {e}")


def fix_attribute_errors(attribute_failures):
    """修复属性错误"""
    print(f"\n🔧 修复属性错误 ({len(attribute_failures)} 个)...")

    fixes_applied = 0

    for failure, error_output in attribute_failures:
        test_file = failure.split('::')[0]

        # 常见的属性错误修复
        if "'str' object has no attribute '__name__'" in error_output:
            fix_string_name_error(test_file)
            fixes_applied += 1

        elif "has no attribute 'start_monitoring'" in error_output:
            fix_monitoring_attributes(test_file)
            fixes_applied += 1

    print(f"应用了 {fixes_applied} 个属性修复")
    return fixes_applied


def fix_string_name_error(test_file):
    """修复字符串没有__name__属性的错误"""
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找使用字符串作为服务类型的地方
        # 通常是 container.resolve("string") 应该改为 container.resolve(Class)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'resolve("' in line and '")' in line:
                # 提取服务名
                match = re.search(r'resolve\("([^"]+)"\)', line)
                if match:
                    service_name = match.group(1)
                    # 查找是否有对应的类定义
                    class_name = service_name.replace('_', ' ').title().replace(' ', '')

                    # 检查文件中是否有这个类
                    class_found = False
                    for j in range(max(0, i-20), min(len(lines), i+5)):
                        if f'class {class_name}:' in lines[j]:
                            class_found = True
                            break

                    if class_found:
                        # 替换字符串为类名
                        lines[i] = line.replace(f'resolve("{service_name}")', f'resolve({class_name})')
                        fixes_applied = True

        if fixes_applied:
            content = '\n'.join(lines)
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 修复了 {test_file} 中的字符串resolve调用")

    except Exception as e:
        print(f"✗ 修复失败 {test_file}: {e}")


def fix_monitoring_attributes(test_file):
    """修复监控相关属性错误"""
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换start_monitoring为record_log_processed
        content = content.replace('start_monitoring', 'record_log_processed')
        content = content.replace('get_stats', 'get_metrics')

        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 修复了 {test_file} 中的监控属性")

    except Exception as e:
        print(f"✗ 修复失败 {test_file}: {e}")


def run_final_verification():
    """运行最终验证"""
    print("\n✅ 运行最终验证...")

    cmd = ['pytest', 'tests/unit/infrastructure/', '--tb=no', '--maxfail=30', '-q']
    result = subprocess.run(cmd, cwd='.')

    if result.returncode == 0:
        print("🎉 所有测试通过！达到100%通过率目标！")
        return True
    else:
        print(f"⚠️  仍有失败测试，继续改进...")
        return False


def main():
    """主函数"""
    print("🚀 基础设施层测试修复系统")
    print("=" * 50)

    # 1. 运行测试并分析失败
    failures = run_test_and_capture_failures()
    if not failures:
        print("🎉 没有失败测试！")
        return

    # 2. 分类失败
    categories = categorize_failures(failures)

    print("\n📊 失败分类统计:")
    print(f"   导入错误: {len(categories['import_error'])}")
    print(f"   属性错误: {len(categories['attribute_error'])}")
    print(f"   类型错误: {len(categories['type_error'])}")
    print(f"   断言错误: {len(categories['assertion_error'])}")
    print(f"   其他错误: {len(categories['other'])}")

    # 3. 修复导入错误
    if categories['import_error']:
        fix_import_errors(categories['import_error'])

    # 4. 修复属性错误
    if categories['attribute_error']:
        fix_attribute_errors(categories['attribute_error'])

    # 5. 最终验证
    success = run_final_verification()

    if success:
        print("\n🎯 100%通过率目标达成！")
        print("接下来可以专注于提升覆盖率...")
    else:
        print("\n📋 需要继续修复剩余失败测试")


if __name__ == '__main__':
    main()
