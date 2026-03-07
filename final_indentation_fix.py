#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终修复所有缩进错误
"""

import os
import re
from pathlib import Path


class FinalIndentationFixer:
    """最终缩进修复器"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_path = self.project_root / 'tests' / 'unit' / 'infrastructure' / 'health'

    def fix_all_indentation(self):
        """修复所有缩进问题"""
        error_files = [
            'test_all_monitoring_modules.py',
            'test_app_monitor_core_methods.py',
            'test_application_monitor_comprehensive.py',
            'test_application_monitor_real_methods.py',
            'test_critical_low_coverage.py',
            'test_health_checker_advanced_scenarios.py',
            'test_health_checker_complete_workflows.py',
            'test_integration_workflows.py',
            'test_metrics_business_logic.py',
            'test_performance_monitor_memory_tracking.py',
            'test_performance_monitor_real_code.py',
            'test_prometheus_integration_deep.py',
            'test_real_business_logic.py'
        ]

        fixed_count = 0

        for filename in error_files:
            file_path = self.tests_path / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    original_content = content

                    # 修复缩进问题：将单空格的pass语句改为4空格
                    content = re.sub(r'^\s{1}pass\s+#\s*Empty skip replaced\s*$',
                                   '        pass  # Empty skip replaced',
                                   content, flags=re.MULTILINE)

                    # 再次检查是否有其他缩进问题
                    lines = content.split('\n')
                    fixed_lines = []

                    for line in lines:
                        if 'pass  # Empty skip replaced' in line:
                            stripped = line.strip()
                            if stripped == 'pass  # Empty skip replaced':
                                # 确保正确的缩进
                                fixed_lines.append('        pass  # Empty skip replaced')
                            else:
                                fixed_lines.append(line)
                        else:
                            fixed_lines.append(line)

                    content = '\n'.join(fixed_lines)

                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixed_count += 1
                        print(f"✅ 修复缩进: {filename}")

                except Exception as e:
                    print(f"❌ 处理文件 {filename} 时出错: {e}")

        return fixed_count

    def verify_fixes(self):
        """验证修复结果"""
        import subprocess
        import sys

        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/unit/infrastructure/health/',
                '--collect-only', '--tb=no', '-q'
            ], capture_output=True, text=True, cwd=self.project_root, timeout=60)

            if result.returncode == 0:
                # 解析测试数量
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'tests collected' in line and 'errors' not in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            test_count = parts[0]
                            print(f"✅ 修复成功！共收集 {test_count} 个测试用例")
                            return True

                print("✅ 修复成功！测试收集正常")
                return True
            else:
                error_lines = [line for line in result.stdout.split('\n') if 'ERROR' in line]
                print(f"❌ 仍有 {len(error_lines)} 个错误")
                for error in error_lines[:5]:  # 只显示前5个
                    print(f"  {error.strip()}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ 验证超时")
            return False
        except Exception as e:
            print(f"❌ 验证错误: {e}")
            return False

    def run_final_fix(self):
        """运行最终修复"""
        print("🔧 开始最终缩进修复...")
        print("=" * 60)

        # 执行修复
        fixed_files = self.fix_all_indentation()
        print(f"✅ 修复了 {fixed_files} 个文件的缩进")

        # 验证修复
        print("🔍 验证修复结果...")
        success = self.verify_fixes()

        print("\n" + "=" * 60)
        print("🎉 最终缩进修复完成！")

        if success:
            print("✅ 所有缩进错误已修复")
        else:
            print("⚠️ 修复完成但仍需进一步处理")

        return success


def main():
    """主函数"""
    fixer = FinalIndentationFixer()
    success = fixer.run_final_fix()

    if success:
        print("\n🎉 缩进错误修复成功！")
        return 0
    else:
        print("\n⚠️ 修复完成但仍需进一步处理")
        return 1


if __name__ == "__main__":
    exit(main())
