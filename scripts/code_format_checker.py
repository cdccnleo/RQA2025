#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码格式化和质量检查脚本
运行Black、Flake8、isort进行代码格式化和质量检查
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, Any


class CodeFormatChecker:
    """代码格式检查器"""

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.results = {
            'black': {'status': 'pending', 'files_checked': 0, 'files_formatted': 0},
            'flake8': {'status': 'pending', 'violations': [], 'files_with_issues': 0},
            'isort': {'status': 'pending', 'files_sorted': 0, 'files_checked': 0}
        }

    def run_black_check(self, apply_fixes: bool = False) -> Dict[str, Any]:
        """运行Black代码格式检查"""
        print("🔍 运行Black代码格式检查...")

        # 主要源码目录
        src_dirs = [
            "src/infrastructure/cache",
            "src/infrastructure/config",
            "src/infrastructure/resource",
            "src/infrastructure/monitoring",
            "tests/unit/infrastructure",
            "scripts"
        ]

        result = {'status': 'success', 'files_checked': 0, 'files_formatted': 0, 'errors': []}

        for src_dir in src_dirs:
            full_path = self.project_root / src_dir
            if not full_path.exists():
                continue

            try:
                # 检查模式或应用模式
                cmd = [
                    sys.executable, "-m", "black",
                    "--line-length", "88",
                    "--target-version", "py38"
                ]

                if not apply_fixes:
                    cmd.append("--check")
                    cmd.append("--diff")

                cmd.append(str(full_path))

                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )

                if process.returncode == 0:
                    if apply_fixes:
                        print(f"  ✅ {src_dir}: 代码已格式化")
                        result['files_formatted'] += 1
                    else:
                        print(f"  ✅ {src_dir}: 代码格式正确")
                else:
                    if not apply_fixes:
                        print(f"  ⚠️ {src_dir}: 需要格式化")
                        if process.stdout:
                            print(f"    输出: {process.stdout[:200]}...")
                    else:
                        print(f"  ❌ {src_dir}: 格式化失败")
                        result['errors'].append(f"{src_dir}: {process.stderr}")

                result['files_checked'] += 1

            except Exception as e:
                result['errors'].append(f"{src_dir}: {str(e)}")
                print(f"  ❌ {src_dir}: 检查失败 - {e}")

        self.results['black'] = result
        return result

    def run_flake8_check(self) -> Dict[str, Any]:
        """运行Flake8代码质量检查"""
        print("🔍 运行Flake8代码质量检查...")

        # 配置选项
        cmd = [
            sys.executable, "-m", "flake8",
            "--max-line-length=88",
            "--ignore=E203,E501,W503,E402",  # 忽略一些常见的格式问题
            "--exclude=.git,__pycache__,build,dist,*.egg-info",
            "src/infrastructure",
            "tests/unit/infrastructure"
        ]

        result = {'status': 'success', 'violations': [], 'files_with_issues': 0}

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if process.returncode == 0:
                print("  ✅ 没有发现代码质量问题")
            else:
                violations = process.stdout.strip().split('\n') if process.stdout.strip() else []
                result['violations'] = violations
                result['files_with_issues'] = len(set(v.split(':')[0]
                                                  for v in violations if ':' in v))

                print(f"  ⚠️ 发现 {len(violations)} 个代码质量问题")
                for violation in violations[:10]:  # 只显示前10个
                    print(f"    - {violation}")

                if len(violations) > 10:
                    print(f"    ... 还有 {len(violations) - 10} 个问题")

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            print(f"  ❌ Flake8检查失败: {e}")

        self.results['flake8'] = result
        return result

    def run_isort_check(self, apply_fixes: bool = False) -> Dict[str, Any]:
        """运行isort导入排序检查"""
        print("🔍 运行isort导入排序检查...")

        # 目标目录
        src_dirs = [
            "src/infrastructure/cache",
            "src/infrastructure/config",
            "src/infrastructure/resource",
            "tests/unit/infrastructure"
        ]

        result = {'status': 'success', 'files_checked': 0, 'files_sorted': 0, 'errors': []}

        for src_dir in src_dirs:
            full_path = self.project_root / src_dir
            if not full_path.exists():
                continue

            try:
                cmd = [
                    sys.executable, "-m", "isort",
                    "--profile", "black",
                    "--line-length", "88"
                ]

                if not apply_fixes:
                    cmd.append("--check-only")
                    cmd.append("--diff")

                cmd.append(str(full_path))

                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )

                if process.returncode == 0:
                    print(f"  ✅ {src_dir}: 导入排序正确")
                else:
                    if not apply_fixes:
                        print(f"  ⚠️ {src_dir}: 需要重新排序导入")
                        if process.stdout:
                            print(f"    变更: {process.stdout[:100]}...")

                result['files_checked'] += 1
                if apply_fixes and process.returncode == 0:
                    result['files_sorted'] += 1

            except Exception as e:
                result['errors'].append(f"{src_dir}: {str(e)}")
                print(f"  ❌ {src_dir}: 检查失败 - {e}")

        self.results['isort'] = result
        return result

    def generate_report(self) -> Dict[str, Any]:
        """生成检查报告"""
        report = {
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'results': self.results,
            'summary': {}
        }

        # 计算总结
        total_issues = len(self.results['flake8'].get('violations', []))
        black_needs_formatting = self.results['black']['status'] != 'success' or self.results['black'].get(
            'files_formatted', 0) > 0
        isort_needs_sorting = self.results['isort'].get('files_sorted', 0) > 0

        report['summary'] = {
            'total_quality_issues': total_issues,
            'needs_black_formatting': black_needs_formatting,
            'needs_import_sorting': isort_needs_sorting,
            'overall_status': 'needs_fixes' if (total_issues > 0 or black_needs_formatting or isort_needs_sorting) else 'clean'
        }

        return report

    def run_baseline_check(self) -> Dict[str, Any]:
        """运行基线检查（不应用修复）"""
        print("🚀 开始代码格式基线检查...")

        self.run_black_check(apply_fixes=False)
        self.run_flake8_check()
        self.run_isort_check(apply_fixes=False)

        report = self.generate_report()

        print("\n📊 基线检查报告:")
        print(f"  - Black格式检查: {len(self.results['black'].get('errors', []))} 个错误")
        print(f"  - Flake8质量检查: {len(self.results['flake8'].get('violations', []))} 个问题")
        print(f"  - isort导入排序: {len(self.results['isort'].get('errors', []))} 个错误")
        print(f"  - 整体状态: {report['summary']['overall_status']}")

        return report

    def apply_formatting_fixes(self) -> Dict[str, Any]:
        """应用格式化修复"""
        print("🔧 开始应用代码格式化修复...")

        # 应用修复
        self.run_black_check(apply_fixes=True)
        self.run_isort_check(apply_fixes=True)

        # 重新检查
        self.run_flake8_check()

        report = self.generate_report()

        print("\n✅ 格式化修复完成:")
        print(f"  - Black格式化: {self.results['black']['files_formatted']} 个文件")
        print(f"  - isort排序: {self.results['isort']['files_sorted']} 个文件")
        print(f"  - 剩余Flake8问题: {len(self.results['flake8'].get('violations', []))}")

        return report


if __name__ == "__main__":
    checker = CodeFormatChecker()

    # 运行基线检查
    baseline_report = checker.run_baseline_check()

    # 保存基线报告
    report_file = checker.project_root / "reports" / "code_format_baseline.json"
    report_file.parent.mkdir(exist_ok=True)

    import json
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(baseline_report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 基线报告已保存: {report_file}")

    # 询问是否应用修复
    if baseline_report['summary']['overall_status'] == 'needs_fixes':
        print("\n🤔 检测到代码格式问题，是否应用自动修复？")
        print("   这将运行Black和isort来格式化代码")

        # 自动应用修复（在脚本模式下）
        if len(sys.argv) > 1 and sys.argv[1] == "--fix":
            print("🚀 自动应用修复...")
            checker.apply_formatting_fixes()
        else:
            print("💡 要应用修复，请运行: python scripts/code_format_checker.py --fix")
    else:
        print("\n🎉 代码格式检查通过！")
