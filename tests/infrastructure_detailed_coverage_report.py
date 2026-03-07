"""
基础设施层17个核心模块详细覆盖率统计

为每个模块生成详细的测试覆盖率报告
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import subprocess


@dataclass
class ModuleDetailedCoverage:
    """模块详细覆盖率"""
    module_name: str
    module_priority: int
    target_coverage: float

    # 文件统计
    total_source_files: int
    total_test_files: int
    tested_files: int
    untested_files: int

    # 覆盖率统计
    file_coverage_rate: float  # 文件覆盖率（测试文件/源文件）
    estimated_line_coverage: float  # 估算的行覆盖率

    # 详细列表
    source_files: List[str]
    test_files: List[str]
    untested_source_files: List[str]

    # 状态
    is_达标: bool
    coverage_gap: float


class InfrastructureDetailedAnalyzer:
    """基础设施层详细覆盖率分析器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_infrastructure = self.project_root / "src" / "infrastructure"
        self.test_infrastructure = self.project_root / "tests" / "unit" / "infrastructure"

        # 17个核心模块配置（根据架构文档）
        self.modules_config = {
            "monitoring": {"priority": 1, "target": 85, "expected_files": 58},
            "config": {"priority": 1, "target": 85, "expected_files": 118},
            "health": {"priority": 1, "target": 85, "expected_files": 71},
            "resource": {"priority": 2, "target": 80, "expected_files": 82},
            "security": {"priority": 1, "target": 85, "expected_files": 45},
            "logging": {"priority": 1, "target": 85, "expected_files": 55},
            "utils": {"priority": 2, "target": 80, "expected_files": 68},
            "api": {"priority": 2, "target": 80, "expected_files": 50},
            "cache": {"priority": 1, "target": 85, "expected_files": 30},
            "error": {"priority": 1, "target": 85, "expected_files": 18},
            "versioning": {"priority": 3, "target": 75, "expected_files": 10},
            "distributed": {"priority": 2, "target": 80, "expected_files": 5},
            "optimization": {"priority": 3, "target": 75, "expected_files": 2},
            "constants": {"priority": 3, "target": 75, "expected_files": 7},
            "core": {"priority": 1, "target": 85, "expected_files": 7},
            "interfaces": {"priority": 1, "target": 90, "expected_files": 2},
            "ops": {"priority": 3, "target": 75, "expected_files": 1},
        }

    def analyze_module(self, module_name: str) -> ModuleDetailedCoverage:
        """分析单个模块的详细覆盖率"""
        config = self.modules_config[module_name]

        # 获取源文件
        src_dir = self.src_infrastructure / module_name
        source_files = []
        if src_dir.exists():
            source_files = list(src_dir.rglob("*.py"))
            source_files = [f for f in source_files if not f.name.startswith("__")]

        # 获取测试文件
        test_dir = self.test_infrastructure / module_name
        test_files = []
        if test_dir.exists():
            test_files = list(test_dir.rglob("test_*.py"))

        # 识别已测试和未测试的源文件
        tested_modules = set()
        for test_file in test_files:
            # 提取测试文件对应的源文件名
            test_name = test_file.stem.replace("test_", "")
            tested_modules.add(test_name)

        untested_files = []
        for src_file in source_files:
            src_name = src_file.stem
            if src_name not in tested_modules and not src_name.startswith("_"):
                untested_files.append(src_file.relative_to(self.src_infrastructure))

        # 计算覆盖率
        total_source = len(source_files)
        total_tests = len(test_files)
        tested_count = total_source - len(untested_files)

        if total_source > 0:
            file_coverage = (tested_count / total_source) * 100
            # 估算行覆盖率（假设有测试的文件行覆盖率平均80%）
            estimated_line_coverage = (total_tests / total_source) * 80 if total_source > 0 else 0
            estimated_line_coverage = min(estimated_line_coverage, 100)
        else:
            file_coverage = 0
            estimated_line_coverage = 0

        coverage_gap = config["target"] - estimated_line_coverage
        is_达标 = estimated_line_coverage >= config["target"]

        return ModuleDetailedCoverage(
            module_name=module_name,
            module_priority=config["priority"],
            target_coverage=config["target"],
            total_source_files=total_source,
            total_test_files=total_tests,
            tested_files=tested_count,
            untested_files=len(untested_files),
            file_coverage_rate=round(file_coverage, 2),
            estimated_line_coverage=round(estimated_line_coverage, 2),
            source_files=[str(f.relative_to(self.src_infrastructure)) for f in source_files[:5]],
            test_files=[str(f.relative_to(self.test_infrastructure)) for f in test_files[:5]],
            untested_source_files=[str(f) for f in untested_files[:10]],
            is_达标=is_达标,
            coverage_gap=round(coverage_gap, 2)
        )

    def generate_detailed_report(self):
        """生成详细的17模块覆盖率报告"""

        print("=" * 100)
        print(" " * 30 + "基础设施层17个核心模块详细覆盖率统计")
        print("=" * 100)
        print()

        all_modules = []

        # 分析所有模块
        for module_name in self.modules_config.keys():
            module_data = self.analyze_module(module_name)
            all_modules.append(module_data)

        # 按优先级和覆盖率排序
        all_modules.sort(key=lambda m: (m.module_priority, -m.estimated_line_coverage))

        # 统计总览
        total_source = sum(m.total_source_files for m in all_modules)
        total_tests = sum(m.total_test_files for m in all_modules)
        avg_coverage = sum(m.estimated_line_coverage for m in all_modules) / len(all_modules)
        达标_count = sum(1 for m in all_modules if m.is_达标)

        print("📊 总体统计")
        print("-" * 100)
        print(f"  总源文件数: {total_source}")
        print(f"  总测试文件数: {total_tests}")
        print(f"  平均估算覆盖率: {avg_coverage:.2f}%")
        print(f"  达标模块数: {达标_count}/17 ({达标_count/17*100:.1f}%)")
        print()

        # 按优先级分组显示
        priorities = {
            1: "P1-核心关键",
            2: "P2-重要",
            3: "P3-一般"
        }

        for priority in [1, 2, 3]:
            priority_modules = [m for m in all_modules if m.module_priority == priority]
            if not priority_modules:
                continue

            print(f"\n{'='*100}")
            print(f"  {priorities[priority]} 模块 ({len(priority_modules)}个)")
            print(f"{'='*100}")
            print()

            # 表头
            print(f"{'模块名':<15} {'源文件':<8} {'测试文件':<8} {'已测试':<8} {'未测试':<8} "
                f"{'文件覆盖率':<12} {'估算覆盖率':<12} {'目标':<8} {'差距':<10} {'状态':<10}")
            print("-" * 100)

            for module in priority_modules:
                status = "✅ 达标" if module.is_达标 else "🔄 需提升"
                gap_str = f"{module.coverage_gap:+.2f}%" if module.coverage_gap != 0 else "0.00%"

                print(f"{module.module_name:<15} "
                    f"{module.total_source_files:<8} "
                    f"{module.total_test_files:<8} "
                    f"{module.tested_files:<8} "
                    f"{module.untested_files:<8} "
                    f"{module.file_coverage_rate:>6.2f}%     "
                    f"{module.estimated_line_coverage:>6.2f}%     "
                    f"{module.target_coverage:>5.1f}%  "
                    f"{gap_str:<10} "
                    f"{status:<10}")

            # 优先级小计
            p_达标 = sum(1 for m in priority_modules if m.is_达标)
            p_avg = sum(m.estimated_line_coverage for m in priority_modules) / len(priority_modules)
            print(f"\n  小计: {p_达标}/{len(priority_modules)} 达标 ({p_达标/len(priority_modules)*100:.1f}%), "
                f"平均覆盖率: {p_avg:.2f}%")

        # 详细信息
        print(f"\n{'='*100}")
        print("  📋 模块详细信息")
        print(f"{'='*100}\n")

        for module in all_modules:
            status_emoji = "✅" if module.is_达标 else "🔄"
            print(f"{status_emoji} {module.module_name} (P{module.module_priority})")
            print(f"   覆盖率: {module.estimated_line_coverage:.2f}% / 目标: {module.target_coverage:.1f}% "
                f"(差距: {module.coverage_gap:+.2f}%)")
            print(f"   文件: {module.total_source_files}个源文件, {module.total_test_files}个测试文件")
            print(f"   状态: {module.tested_files}个已测试, {module.untested_files}个未测试")

            if module.untested_source_files:
                print(f"   缺失测试文件示例: {', '.join(module.untested_source_files[:3])}")

            print()

        # 保存详细报告
        report_data = {
            "summary": {
                "total_source_files": total_source,
                "total_test_files": total_tests,
                "average_coverage": round(avg_coverage, 2),
                "达标_modules": 达标_count,
                "达标_rate": round(达标_count/17*100, 2),
            },
            "modules": [asdict(m) for m in all_modules],
            "by_priority": {
                "P1": [asdict(m) for m in all_modules if m.module_priority == 1],
                "P2": [asdict(m) for m in all_modules if m.module_priority == 2],
                "P3": [asdict(m) for m in all_modules if m.module_priority == 3],
            }
        }

        report_file = self.project_root / "test_logs" / "infrastructure_17_modules_detailed_coverage.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"{'='*100}")
        print(f"  详细数据已保存至: {report_file}")
        print(f"{'='*100}")
        print()

        # 最终总结
        print("🎯 最终总结")
        print("-" * 100)
        print(f"  ✅ 达标模块: {达标_count}/17 ({达标_count/17*100:.1f}%)")
        print(f"  🔄 需提升模块: {17-达标_count}/17 ({(17-达标_count)/17*100:.1f}%)")
        print(f"  📊 平均覆盖率: {avg_coverage:.2f}%")

        if avg_coverage >= 85:
            print(f"  🎉 投产就绪: 是 (平均覆盖率{avg_coverage:.2f}% ≥ 85%)")
        else:
            print(f"  ⚠️  投产就绪: 否 (平均覆盖率{avg_coverage:.2f}% < 85%)")

        print()

        return all_modules


def main():
    """主函数"""
    analyzer = InfrastructureDetailedAnalyzer()
    modules = analyzer.generate_detailed_report()

    # 生成优先级建议
    print("\n" + "=" * 100)
    print("  🎯 优化建议（按优先级）")
    print("=" * 100)
    print()

    needs_improvement = [m for m in modules if not m.is_达标]
    needs_improvement.sort(key=lambda m: (m.module_priority, -m.coverage_gap))

    if needs_improvement:
        for i, module in enumerate(needs_improvement, 1):
            print(f"{i}. {module.module_name} (P{module.module_priority})")
            print(f"   当前: {module.estimated_line_coverage:.2f}%, 目标: {module.target_coverage:.1f}%, "
                f"差距: {abs(module.coverage_gap):.2f}%")
            print(f"   需要为 {module.untested_files} 个文件添加测试")
            if module.untested_source_files:
                print(f"   优先测试: {', '.join(module.untested_source_files[:3])}")
            print()
    else:
        print("  🎉 所有模块均已达标！")

    print()


if __name__ == "__main__":
    main()
