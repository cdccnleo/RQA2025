#!/usr/bin/env python3
"""
RQA2025 预提交钩子脚本
在代码提交前检查测试覆盖率和代码质量
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict


class PreCommitHook:
    """预提交钩子检查器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.min_coverage = 75.0

    def get_changed_modules(self) -> List[str]:
        """获取变更的模块"""
        try:
            # 获取暂存区的文件
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            if result.returncode != 0:
                print("❌ 无法获取变更文件")
                return []

            changed_files = result.stdout.strip().split('\n')
            if not changed_files or changed_files == ['']:
                return []

            # 分析变更的模块
            modules = set()
            for file_path in changed_files:
                if file_path.startswith('src/'):
                    module = file_path.split('/')[1]
                    modules.add(module)

            return list(modules)

        except Exception as e:
            print(f"❌ 获取变更模块失败: {e}")
            return []

    def run_tests_for_module(self, module: str) -> Dict:
        """运行指定模块的测试"""
        print(f"🔍 运行 {module} 模块测试...")

        cmd = [
            "python", "scripts/testing/run_tests.py",
            "--env", "test",
            "--module", module,
            "--cov", f"src/{module}",
            "--pytest-args", "-v", "--timeout", "300"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=600
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "测试超时",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    def extract_coverage(self, stdout: str) -> float:
        """从测试输出中提取覆盖率"""
        try:
            lines = stdout.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        coverage_str = parts[3].replace('%', '')
                        return float(coverage_str)
        except:
            pass
        return 0.0

    def check_coverage_threshold(self, module: str, coverage: float) -> bool:
        """检查覆盖率是否达标"""
        # 不同模块的覆盖率要求
        thresholds = {
            "infrastructure": 80.0,
            "data": 80.0,
            "features": 80.0,
            "ensemble": 80.0,
            "trading": 80.0,
            "backtest": 80.0
        }

        threshold = thresholds.get(module, self.min_coverage)
        return coverage >= threshold

    def check_code_quality(self) -> bool:
        """检查代码质量"""
        print("🔍 检查代码质量...")

        # 检查是否有语法错误
        try:
            # 查找所有Python文件并检查语法
            python_files = []
            for root, dirs, files in os.walk(self.project_root / "src"):
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))

            if not python_files:
                print("ℹ️  没有找到Python文件")
                return True

            # 检查每个Python文件的语法
            for py_file in python_files:
                result = subprocess.run(
                    ["python", "-m", "py_compile", py_file],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )

                if result.returncode != 0:
                    print(f"❌ 发现语法错误 in {py_file}:")
                    print(result.stderr)
                    return False

        except Exception as e:
            print(f"❌ 代码质量检查失败: {e}")
            return False

        print("✅ 代码质量检查通过")
        return True

    def run_hook(self) -> bool:
        """运行预提交钩子检查"""
        print("🚀 开始预提交检查...")
        print("=" * 50)

        # 1. 检查代码质量
        if not self.check_code_quality():
            return False

        # 2. 获取变更的模块
        changed_modules = self.get_changed_modules()
        if not changed_modules:
            print("ℹ️  没有检测到代码变更，跳过测试检查")
            return True

        print(f"📦 检测到变更的模块: {', '.join(changed_modules)}")

        # 3. 运行相关模块的测试
        all_passed = True
        coverage_results = {}

        for module in changed_modules:
            result = self.run_tests_for_module(module)

            if result["success"]:
                coverage = self.extract_coverage(result["stdout"])
                coverage_results[module] = coverage

                if self.check_coverage_threshold(module, coverage):
                    print(f"✅ {module} 模块: {coverage:.1f}% (达标)")
                else:
                    print(f"❌ {module} 模块: {coverage:.1f}% (未达标)")
                    all_passed = False
            else:
                print(f"❌ {module} 模块测试失败")
                print(result["stderr"])
                all_passed = False

        # 4. 生成检查报告
        print("\n" + "=" * 50)
        print("📊 预提交检查结果")
        print("=" * 50)

        for module, coverage in coverage_results.items():
            status = "✅ 通过" if self.check_coverage_threshold(module, coverage) else "❌ 未通过"
            print(f"{module.title():15} {coverage:6.1f}% {status}")

        if all_passed:
            print("\n🎉 预提交检查通过！可以提交代码。")
            return True
        else:
            print("\n❌ 预提交检查失败！请修复问题后重新提交。")
            print("\n💡 建议:")
            print("- 添加更多测试用例以提高覆盖率")
            print("- 修复失败的测试")
            print("- 检查代码语法错误")
            return False


def main():
    """主函数"""
    hook = PreCommitHook()
    success = hook.run_hook()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
