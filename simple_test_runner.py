#!/usr/bin/env python3
"""
RQA2025 简单测试运行器
避免编码问题，专注于提升测试覆盖率
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

class SimpleTestRunner:
    """简单测试运行器"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {}

    def run_test_module(self, module_path, module_name):
        """运行单个测试模块"""
        print(f"\n🧪 运行 {module_name}...")

        try:
            # 使用环境变量设置编码
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            cmd = [
                sys.executable, "-m", "pytest",
                module_path,
                "-v", "--tb=short", "--maxfail=5",
                "--disable-warnings", "-q"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                timeout=180,  # 3分钟超时
                capture_output=True
            )

            # 尝试解码输出
            try:
                stdout = result.stdout.decode('utf-8', errors='ignore')
                stderr = result.stderr.decode('utf-8', errors='ignore')
            except:
                stdout = result.stdout.decode('gbk', errors='ignore') if result.stdout else ""
                stderr = result.stderr.decode('gbk', errors='ignore') if result.stderr else ""

            output = stdout + stderr

            # 解析结果
            passed = 0
            failed = 0
            skipped = 0

            for line in output.split('\n'):
                if 'passed' in line and 'failed' in line:
                    import re
                    match = re.search(r'(\d+)\s*passed.*?(\d+)\s*failed.*?(\d+)\s*skipped', line, re.IGNORECASE)
                    if match:
                        passed = int(match.group(1))
                        failed = int(match.group(2))
                        skipped = int(match.group(3))
                        break

            total_tests = passed + failed + skipped

            if total_tests > 0:
                pass_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
                print(f"   ✅ 通过: {passed}, ❌ 失败: {failed}, ⏭️  跳过: {skipped} | 通过率: {pass_rate:.1f}%")
                self.results[module_name] = {
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "total": total_tests,
                    "pass_rate": pass_rate,
                    "status": "成功" if failed == 0 else "部分失败"
                }
                return True
            else:
                print("   ⚠️  无测试执行")
                self.results[module_name] = {
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "total": 0,
                    "pass_rate": 0,
                    "status": "无测试"
                }
                return False

        except subprocess.TimeoutExpired:
            print("   ⏰ 测试超时")
            self.results[module_name] = {
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "total": 0,
                "pass_rate": 0,
                "status": "超时"
            }
            return False
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            self.results[module_name] = {
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "total": 0,
                "pass_rate": 0,
                "status": f"错误: {str(e)}"
            }
            return False

    def run_coverage_test(self, module_path, src_path, module_name):
        """运行覆盖率测试"""
        print(f"\n📊 计算 {module_name} 覆盖率...")

        try:
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'

            cmd = [
                sys.executable, "-m", "pytest",
                f"--cov={src_path}",
                "--cov-report=term-missing",
                module_path,
                "--tb=no", "--maxfail=5",
                "--disable-warnings", "-q"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                env=env,
                timeout=300,  # 5分钟超时
                capture_output=True
            )

            # 解码输出
            try:
                stdout = result.stdout.decode('utf-8', errors='ignore')
                stderr = result.stderr.decode('utf-8', errors='ignore')
            except:
                stdout = result.stdout.decode('gbk', errors='ignore') if result.stdout else ""
                stderr = result.stderr.decode('gbk', errors='ignore') if result.stderr else ""

            output = stdout + stderr

            # 查找TOTAL行
            total_coverage = 0.0
            for line in output.split('\n'):
                if line.startswith('TOTAL'):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            coverage_str = parts[-1].rstrip('%')
                            total_coverage = float(coverage_str)
                            break
                        except:
                            pass

            print(f"   📊 覆盖率: {total_coverage:.1f}%")
            return total_coverage

        except Exception as e:
            print(f"   ❌ 覆盖率计算失败: {e}")
            return 0.0

    def run_priority_tests(self):
        """按优先级运行测试"""
        print("🎯 RQA2025 测试覆盖率提升 - 优先级执行")
        print("="*60)

        # 高优先级模块（核心业务和基础设施）
        high_priority = [
            ("tests/unit/infrastructure/cache/", "src/infrastructure/cache", "基础设施层-缓存"),
            ("tests/unit/core/", "src/core", "核心服务层"),
        ]

        # 中优先级模块（业务逻辑层）
        medium_priority = [
            ("tests/unit/trading/", "src/trading", "交易层"),
            ("tests/unit/strategy/", "src/strategy", "策略层"),
            ("tests/unit/risk/", "src/risk", "风险控制层"),
            ("tests/unit/features/", "src/features", "特征分析层"),
        ]

        # 低优先级模块（支撑层）
        low_priority = [
            ("tests/unit/data/", "src/data", "数据管理层"),
            ("tests/unit/ml/", "src/ml", "机器学习层"),
            ("tests/unit/monitoring/", "src/monitoring", "监控层"),
            ("tests/unit/optimization/", "src/optimization", "优化层"),
        ]

        all_modules = high_priority + medium_priority + low_priority

        total_passed = 0
        total_failed = 0
        total_tests = 0
        successful_modules = 0

        for test_path, src_path, name in all_modules:
            # 首先运行测试
            success = self.run_test_module(test_path, name)

            if success:
                successful_modules += 1
                module_result = self.results[name]
                total_passed += module_result["passed"]
                total_failed += module_result["failed"]
                total_tests += module_result["total"]

                # 如果测试通过，计算覆盖率
                if module_result["failed"] == 0 and module_result["passed"] > 0:
                    coverage = self.run_coverage_test(test_path, src_path, name)
                    module_result["coverage"] = coverage
                else:
                    module_result["coverage"] = 0.0
            else:
                self.results[name]["coverage"] = 0.0

        # 生成报告
        self.generate_report(total_passed, total_failed, total_tests, successful_modules)

    def generate_report(self, total_passed, total_failed, total_tests, successful_modules):
        """生成报告"""
        print(f"\n{'='*60}")
        print("📊 测试执行汇总报告")
        print(f"{'='*60}")

        print(f"⏰ 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📦 成功模块数: {successful_modules}/{len(self.results)}")
        print(f"✅ 总通过测试: {total_passed}")
        print(f"❌ 总失败测试: {total_failed}")
        print(f"📊 总测试数: {total_tests}")

        if total_tests > 0:
            overall_pass_rate = (total_passed / total_tests) * 100
            print(f"📊 总体通过率: {overall_pass_rate:.1f}%")
            if overall_pass_rate >= 95:
                print("🎉 达到目标: 95%+ 通过率")
            elif overall_pass_rate >= 80:
                print("⚠️  接近目标: 需要继续修复")
            else:
                print("❌ 未达目标: 需要重点修复")

        print("\n🏆 各模块状态:")
        for name, result in self.results.items():
            status = result["status"]
            if result["total"] > 0:
                pass_rate = result["pass_rate"]
                coverage = result.get("coverage", 0.0)
                print(f"   {name}: 通过率 {pass_rate:.1f}%, 覆盖率 {coverage:.1f}% - {status}")
            else:
                print(f"   {name}: {status}")

        # 保存详细结果
        self.save_results()

    def save_results(self):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.project_root / "test_logs" / f"simple_test_results_{timestamp}.json"

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"\n📄 详细结果已保存: {result_file}")

def main():
    """主函数"""
    runner = SimpleTestRunner()
    runner.run_priority_tests()

if __name__ == "__main__":
    main()