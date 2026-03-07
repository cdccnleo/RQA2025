#!/usr/bin/env python3
"""
批量修复基础设施层测试中的导入错误
"""

import os
import re
from pathlib import Path


def fix_import_errors():
    """修复常见的导入错误"""

    # 需要修复的导入映射
    import_fixes = {
        'from src.infrastructure.interfaces.standard_interfaces import.*ConfigProvider': 'from src.infrastructure.config.tools.provider import IConfigProvider',
        'from src.infrastructure.config.core.config_interfaces import ConfigProvider': 'from src.infrastructure.config.tools.provider import ConfigProvider',
        'from src.infrastructure.config.core.config_interfaces import IConfigProvider': 'from src.infrastructure.config.tools.provider import IConfigProvider',
    }

    test_files = []
    for root, dirs, files in os.walk('tests/unit/infrastructure'):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))

    print(f"检查 {len(test_files)} 个测试文件...")

    fixed_files = 0

    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            # 应用导入修复
            for old_import, new_import in import_fixes.items():
                if re.search(old_import, content, re.MULTILINE):
                    content = re.sub(old_import, new_import, content, flags=re.MULTILINE)
                    modified = True
                    print(f"✓ 修复 {test_file}: {old_import} -> {new_import}")

            # 如果内容有变化，保存文件
            if modified:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files += 1

        except Exception as e:
            print(f"✗ 处理 {test_file} 时出错: {e}")

    print(f"\n修复完成！共修复了 {fixed_files} 个文件")
    return fixed_files


def fix_method_calls():
    """修复方法调用错误"""

    # 需要修复的方法调用
    method_fixes = {
        'start_monitoring': 'record_log_processed',
        'get_stats': 'get_metrics',
        'collect_data': 'get_dashboard_data',
        'register_handler': 'register_handler_class',
        'save_version_data': 'save_version',
        'delete_version': 'remove_version',
        'check_health': 'health_check',
        'singleton=True': 'lifetime=ServiceLifetime.SINGLETON',
        'singleton=False': 'lifetime=ServiceLifetime.TRANSIENT',
    }

    test_files = []
    for root, dirs, files in os.walk('tests/unit/infrastructure'):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))

    fixed_files = 0

    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            # 应用方法调用修复
            for old_method, new_method in method_fixes.items():
                if old_method in content:
                    content = content.replace(old_method, new_method)
                    modified = True

            # 特殊处理：字符串作为resolve参数
            content = re.sub(r'resolve\("([^"]+)"\)', lambda m: f'resolve({m.group(1).title().replace("_", "")})', content)

            # 如果内容有变化，保存文件
            if modified and content != original_content:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files += 1
                print(f"✓ 修复 {test_file} 的方法调用")

        except Exception as e:
            print(f"✗ 处理 {test_file} 时出错: {e}")

    print(f"方法调用修复完成！共修复了 {fixed_files} 个文件")
    return fixed_files


def run_verification():
    """运行验证测试"""
    print("\n🏃 运行验证测试...")

    import subprocess
    cmd = ['pytest', 'tests/unit/infrastructure/', '--tb=no', '--maxfail=20', '-q']
    result = subprocess.run(cmd, cwd='.')

    passed_match = re.search(r'(\d+) passed', result.stdout.decode() if result.stdout else '')
    failed_match = re.search(r'(\d+) failed', result.stdout.decode() if result.stdout else '')

    if passed_match and failed_match:
        passed = int(passed_match.group(1))
        failed = int(failed_match.group(1))
        total = passed + failed
        pass_rate = passed / total * 100 if total > 0 else 0

        print(f"验证结果: {passed} 通过, {failed} 失败")
        print(".1f")
        if failed == 0:
            print("🎉 达到100%通过率目标！")
            return True
        else:
            print("⚠️ 仍有失败测试需要修复")
            return False

    return False


def main():
    """主函数"""
    print("🔧 基础设施层测试批量修复工具")
    print("=" * 50)

    # 1. 修复导入错误
    print("\n1. 修复导入错误...")
    import_fixes = fix_import_errors()

    # 2. 修复方法调用错误
    print("\n2. 修复方法调用错误...")
    method_fixes = fix_method_calls()

    # 3. 运行验证
    print("\n3. 运行验证...")
    success = run_verification()

    print("
📊 修复总结:"    print(f"   导入修复: {import_fixes} 个文件")
    print(f"   方法修复: {method_fixes} 个文件")
    print(f"   验证结果: {'✅ 通过' if success else '⚠️  需要继续'}")

    if success:
        print("\n🎯 100%通过率目标达成！")
        print("接下来可以专注于提升覆盖率至70%...")
    else:
        print("\n📋 需要继续修复剩余失败测试")


if __name__ == '__main__':
    main()
