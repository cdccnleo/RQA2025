"""
使用pytest-cov准确统计17个模块的测试覆盖率

执行实际的代码覆盖率分析
"""

import subprocess
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class AccurateCoverageResult:
    """准确的覆盖率结果"""
    module_name: str
    priority: int
    target: float
    statements: int
    missing: int
    coverage: float
    is_达标: bool
    gap: float


class AccurateCoverageAnalyzer:
    """准确覆盖率分析器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent

        # 17个模块配置
        self.modules = {
            "monitoring": {"priority": 1, "target": 85},
            "config": {"priority": 1, "target": 85},
            "health": {"priority": 1, "target": 85},
            "resource": {"priority": 2, "target": 80},
            "security": {"priority": 1, "target": 85},
            "logging": {"priority": 1, "target": 85},
            "utils": {"priority": 2, "target": 80},
            "api": {"priority": 2, "target": 80},
            "cache": {"priority": 1, "target": 85},
            "error": {"priority": 1, "target": 85},
            "versioning": {"priority": 3, "target": 75},
            "distributed": {"priority": 2, "target": 80},
            "optimization": {"priority": 3, "target": 75},
            "constants": {"priority": 3, "target": 75},
            "core": {"priority": 1, "target": 85},
            "interfaces": {"priority": 1, "target": 90},
            "ops": {"priority": 3, "target": 75},
        }

    def run_coverage_for_module(self, module_name: str) -> AccurateCoverageResult:
        """为单个模块运行覆盖率测试"""

        src_path = f"src/infrastructure/{module_name}"
        test_path = f"tests/unit/infrastructure/{module_name}"

        config = self.modules[module_name]

        # 构建pytest-cov命令
        cmd = [
            "python", "-m", "pytest",
            test_path,
            f"--cov={src_path}",
            "--cov-report=term-missing",
            "--cov-report=json",
            "-q",
            "--no-header",
            "--tb=no",
        ]

        print(f"正在分析 {module_name} 模块...", end=" ", flush=True)

        try:
            # 运行pytest-cov
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                timeout=120
            )

            # 读取JSON报告
            coverage_file = self.project_root / ".coverage"
            json_file = self.project_root / "coverage.json"

            coverage = 0.0
            statements = 0
            missing = 0

            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    cov_data = json.load(f)

                    # 提取覆盖率数据
                    totals = cov_data.get('totals', {})
                    coverage = totals.get('percent_covered', 0.0)
                    statements = totals.get('num_statements', 0)
                    missing = totals.get('missing_lines', 0)

            print(f"✅ {coverage:.2f}%")

            gap = config["target"] - coverage
            is_达标 = coverage >= config["target"]

            return AccurateCoverageResult(
                module_name=module_name,
                priority=config["priority"],
                target=config["target"],
                statements=statements,
                missing=missing,
                coverage=round(coverage, 2),
                is_达标=is_达标,
                gap=round(gap, 2)
            )

        except subprocess.TimeoutExpired:
            print("⏱️  超时")
            return AccurateCoverageResult(
                module_name=module_name,
                priority=config["priority"],
                target=config["target"],
                statements=0,
                missing=0,
                coverage=0.0,
                is_达标=False,
                gap=config["target"]
            )
        except Exception as e:
            print(f"❌ 错误: {e}")
            return AccurateCoverageResult(
                module_name=module_name,
                priority=config["priority"],
                target=config["target"],
                statements=0,
                missing=0,
                coverage=0.0,
                is_达标=False,
                gap=config["target"]
            )

    def generate_report(self):
        """生成完整报告"""
        print("=" * 100)
        print(" " * 25 + "基础设施层17个模块准确测试覆盖率统计")
        print(" " * 35 + "(使用pytest-cov)")
        print("=" * 100)
        print()

        results = []

        # 分析所有模块
        for module_name in self.modules.keys():
            result = self.run_coverage_for_module(module_name)
            results.append(result)

        print()
        print("=" * 100)
        print("  📊 详细覆盖率统计")
        print("=" * 100)
        print()

        # 按优先级分组
        priorities = {1: "P1-核心关键", 2: "P2-重要", 3: "P3-一般"}

        for priority in [1, 2, 3]:
            priority_results = [r for r in results if r.priority == priority]
            if not priority_results:
                continue

            print(f"\n{priorities[priority]} ({len(priority_results)}个模块)")
            print("-" * 100)
            print(f"{'模块名':<15} {'代码行数':<10} {'缺失行数':<10} {'覆盖率':<12} {'目标':<10} {'差距':<12} {'状态':<10}")
            print("-" * 100)

            for r in sorted(priority_results, key=lambda x: -x.coverage):
                status = "✅ 达标" if r.is_达标 else "🔄 需提升"
                gap_str = f"{r.gap:+.2f}%" if r.gap != 0 else "0.00%"

                print(f"{r.module_name:<15} "
                    f"{r.statements:<10} "
                    f"{r.missing:<10} "
                    f"{r.coverage:>6.2f}%     "
                    f"{r.target:>5.1f}%    "
                    f"{gap_str:<12} "
                    f"{status:<10}")

            # 小计
            p_达标 = sum(1 for r in priority_results if r.is_达标)
            p_avg = sum(r.coverage for r in priority_results) / len(priority_results)
            print(f"\n  小计: {p_达标}/{len(priority_results)} 达标 ({p_达标/len(priority_results)*100:.1f}%), "
                f"平均覆盖率: {p_avg:.2f}%")

        # 总体统计
        print("\n" + "=" * 100)
        print("  🎯 总体统计")
        print("=" * 100)

        total_statements = sum(r.statements for r in results)
        total_missing = sum(r.missing for r in results)
        avg_coverage = sum(r.coverage for r in results) / len(results)
        达标_count = sum(1 for r in results if r.is_达标)

        print(f"\n  总代码行数: {total_statements:,}")
        print(f"  总缺失行数: {total_missing:,}")
        print(f"  平均覆盖率: {avg_coverage:.2f}%")
        print(f"  达标模块数: {达标_count}/17 ({达标_count/17*100:.1f}%)")

        if avg_coverage >= 85:
            print(f"  🎉 投产就绪: 是 (平均覆盖率 {avg_coverage:.2f}% ≥ 85%)")
        else:
            gap = 85 - avg_coverage
            print(f"  ⚠️  投产就绪: 否 (平均覆盖率 {avg_coverage:.2f}% < 85%, 差距 {gap:.2f}%)")

        # 保存结果
        report_data = {
            "summary": {
                "total_statements": total_statements,
                "total_missing": total_missing,
                "average_coverage": round(avg_coverage, 2),
                "达标_modules": 达标_count,
                "达标_rate": round(达标_count/17*100, 2),
            },
            "modules": [asdict(r) for r in results],
            "by_priority": {
                f"P{p}": [asdict(r) for r in results if r.priority == p]
                for p in [1, 2, 3]
            }
        }

        report_file = self.project_root / "test_logs" / "infrastructure_accurate_coverage_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\n  详细数据已保存: {report_file}")
        print()

        return results


def main():
    """主函数"""
    analyzer = AccurateCoverageAnalyzer()
    results = analyzer.generate_report()

    print("=" * 100)
    print("  分析完成！")
    print("=" * 100)


if __name__ == "__main__":
    main()
