#!/usr/bin/env python3
"""
RQA2025 自动化测试覆盖率流水线
集成CI/CD功能，自动化覆盖率报告和监控
"""

import subprocess
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict


class CoveragePipeline:
    """自动化测试覆盖率流水线"""

    def __init__(self, env: str = "test", timeout: int = 600):
        self.env = env
        self.timeout = timeout
        self.project_root = Path(__file__).parent.parent.parent
        self.reports_dir = self.project_root / "reports" / "testing"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_module_tests(self, module: str, cov_path: str) -> Dict:
        """运行指定模块的测试并收集覆盖率"""
        print(f"🔍 运行 {module} 层测试...")

        cmd = [
            "python", "scripts/testing/run_tests.py",
            "--env", self.env,
            "--module", module,
            "--cov", cov_path,
            "--pytest-args", "-v", "--timeout", str(self.timeout)
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=self.timeout + 60
            )
            duration = time.time() - start_time

            return {
                "module": module,
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "module": module,
                "success": False,
                "duration": self.timeout + 60,
                "stdout": "",
                "stderr": "测试超时",
                "returncode": -1
            }
        except Exception as e:
            return {
                "module": module,
                "success": False,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    def run_all_tests(self) -> Dict[str, Dict]:
        """运行所有层的测试"""
        modules = {
            "infrastructure": "src/infrastructure",
            "data": "src/data",
            "features": "src/features",
            "ensemble": "src/ensemble",
            "trading": "src/trading",
            "backtest": "src/backtest"
        }

        results = {}
        for module, cov_path in modules.items():
            results[module] = self.run_module_tests(module, cov_path)
            print(f"✅ {module} 层测试完成")

        return results

    def generate_coverage_report(self, results: Dict[str, Dict]) -> str:
        """生成覆盖率报告"""
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 解析覆盖率数据
        coverage_data = {}
        for module, result in results.items():
            if result["success"]:
                # 从stdout中提取覆盖率信息
                coverage = self._extract_coverage(result["stdout"])
                coverage_data[module] = coverage
            else:
                coverage_data[module] = {"total": 0, "missing": 0, "coverage": 0}

        # 计算总体覆盖率
        total_coverage = sum(data.get("coverage", 0) for data in coverage_data.values())
        avg_coverage = total_coverage / len(coverage_data) if coverage_data else 0

        # 生成报告
        report = f"""# RQA2025 自动化测试覆盖率报告

## 📊 执行摘要

**报告时间**: {report_time}  
**总体状态**: {'🚀 良好' if avg_coverage >= 50 else '⚠️ 需要改进'}  
**平均覆盖率**: {avg_coverage:.2f}%  
**测试状态**: {sum(1 for r in results.values() if r['success'])}/{len(results)} 层成功

## 📈 各层覆盖率详情

"""

        for module, data in coverage_data.items():
            status = "✅" if data.get("coverage", 0) >= 25 else "⚠️"
            report += f"""### {module.title()} 层
- **覆盖率**: {data.get('coverage', 0):.2f}% {status}
- **总行数**: {data.get('total', 0)}
- **未覆盖**: {data.get('missing', 0)}
- **测试状态**: {'✅ 成功' if results[module]['success'] else '❌ 失败'}

"""

        report += f"""
## 🎯 质量门禁检查

### 覆盖率门禁
- **目标覆盖率**: 25%
- **当前平均覆盖率**: {avg_coverage:.2f}%
- **门禁状态**: {'✅ 通过' if avg_coverage >= 25 else '❌ 未通过'}

### 测试稳定性门禁
- **成功层数**: {sum(1 for r in results.values() if r['success'])}/{len(results)}
- **门禁状态**: {'✅ 通过' if all(r['success'] for r in results.values()) else '❌ 未通过'}

## 📋 建议

"""

        if avg_coverage < 50:
            report += "- 🔧 需要提升测试覆盖率\n"
        if not all(r['success'] for r in results.values()):
            report += "- 🔧 需要修复失败的测试\n"
        if avg_coverage >= 50:
            report += "- 🎉 覆盖率表现良好，建议继续提升至80%\n"

        return report

    def _extract_coverage(self, stdout: str) -> Dict:
        """从测试输出中提取覆盖率信息"""
        try:
            # 查找覆盖率行
            lines = stdout.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    # 解析覆盖率数据
                    parts = line.split()
                    if len(parts) >= 4:
                        total = int(parts[1])
                        missing = int(parts[2])
                        coverage_str = parts[3].replace('%', '')
                        coverage = float(coverage_str)
                        return {
                            "total": total,
                            "missing": missing,
                            "coverage": coverage
                        }
        except:
            pass

        return {"total": 0, "missing": 0, "coverage": 0}

    def save_report(self, report: str, results: Dict[str, Dict]):
        """保存报告和结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存文本报告
        report_file = self.reports_dir / f"coverage_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        # 保存JSON结果
        json_file = self.reports_dir / f"coverage_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"📄 报告已保存: {report_file}")
        print(f"📄 结果已保存: {json_file}")

        return report_file, json_file

    def run_pipeline(self) -> bool:
        """运行完整的自动化流水线"""
        print("🚀 启动RQA2025自动化测试覆盖率流水线")
        print("=" * 60)

        # 运行所有测试
        results = self.run_all_tests()

        # 生成报告
        report = self.generate_coverage_report(results)

        # 保存报告
        report_file, json_file = self.save_report(report, results)

        # 打印摘要
        print("\n" + "=" * 60)
        print("📊 流水线执行摘要")
        print("=" * 60)

        success_count = sum(1 for r in results.values() if r['success'])
        total_duration = sum(r['duration'] for r in results.values())

        print(f"✅ 成功层数: {success_count}/{len(results)}")
        print(f"⏱️  总执行时间: {total_duration:.2f}秒")
        print(f"📄 报告文件: {report_file}")
        print(f"📄 结果文件: {json_file}")

        # 检查质量门禁
        all_success = all(r['success'] for r in results.values())
        if all_success:
            print("🎉 所有测试通过，质量门禁检查通过！")
            return True
        else:
            print("⚠️  部分测试失败，需要修复！")
            return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RQA2025自动化测试覆盖率流水线")
    parser.add_argument("--env", default="test", help="运行环境")
    parser.add_argument("--timeout", type=int, default=600, help="测试超时时间(秒)")
    parser.add_argument("--module", help="只运行指定模块")

    args = parser.parse_args()

    pipeline = CoveragePipeline(env=args.env, timeout=args.timeout)

    if args.module:
        # 只运行指定模块
        modules = {
            "infrastructure": "src/infrastructure",
            "data": "src/data",
            "features": "src/features",
            "ensemble": "src/ensemble",
            "trading": "src/trading",
            "backtest": "src/backtest"
        }

        if args.module not in modules:
            print(f"❌ 未知模块: {args.module}")
            sys.exit(1)

        result = pipeline.run_module_tests(args.module, modules[args.module])
        results = {args.module: result}
        report = pipeline.generate_coverage_report(results)
        pipeline.save_report(report, results)

        success = result["success"]
    else:
        # 运行所有模块
        success = pipeline.run_pipeline()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
