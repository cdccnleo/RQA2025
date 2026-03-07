"""
基础设施层测试覆盖率系统性提升计划

流程：
1. 识别低覆盖模块
2. 添加缺失测试
3. 修复代码问题
4. 验证覆盖率提升

目标：达到投产要求的测试覆盖率标准
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ModuleCoverage:
    """模块覆盖率信息"""
    module_name: str
    file_count: int
    current_coverage: float
    target_coverage: float
    priority: int  # 1=最高, 5=最低
    test_files: List[str]
    missing_tests: List[str]


@dataclass
class CoverageBoostPlan:
    """覆盖率提升计划"""
    phase: int
    modules: List[str]
    target_coverage: float
    estimated_tests: int
    priority_level: str


class InfrastructureCoverageAnalyzer:
    """基础设施层覆盖率分析器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.src_infrastructure = self.project_root / "src" / "infrastructure"
        self.test_infrastructure = self.project_root / "tests" / "unit" / "infrastructure"

        # 基础设施层17个核心模块（根据架构文档）
        self.core_modules = {
            "monitoring": {"files": 58, "priority": 1, "target": 85},
            "config": {"files": 118, "priority": 1, "target": 85},
            "health": {"files": 71, "priority": 1, "target": 85},
            "resource": {"files": 82, "priority": 2, "target": 80},
            "security": {"files": 45, "priority": 1, "target": 85},
            "logging": {"files": 55, "priority": 1, "target": 85},
            "utils": {"files": 68, "priority": 2, "target": 80},
            "api": {"files": 50, "priority": 2, "target": 80},
            "cache": {"files": 30, "priority": 1, "target": 85},
            "error": {"files": 18, "priority": 1, "target": 85},
            "versioning": {"files": 10, "priority": 3, "target": 75},
            "distributed": {"files": 5, "priority": 2, "target": 80},
            "optimization": {"files": 2, "priority": 3, "target": 75},
            "constants": {"files": 7, "priority": 3, "target": 75},
            "core": {"files": 7, "priority": 1, "target": 85},
            "interfaces": {"files": 2, "priority": 1, "target": 90},
            "ops": {"files": 1, "priority": 3, "target": 75},
        }

    def analyze_current_coverage(self) -> Dict[str, ModuleCoverage]:
        """分析当前测试覆盖情况"""
        coverage_data = {}

        for module_name, module_info in self.core_modules.items():
            # 查找现有测试文件
            test_dir = self.test_infrastructure / module_name
            test_files = []
            if test_dir.exists():
                test_files = list(test_dir.rglob("test_*.py"))

            # 查找源代码文件
            src_dir = self.src_infrastructure / module_name
            src_files = []
            if src_dir.exists():
                src_files = list(src_dir.rglob("*.py"))
                src_files = [f for f in src_files if not f.name.startswith("__")]

            # 估算当前覆盖率（基于测试文件数量/源文件数量的简单估算）
            if src_files:
                current_coverage = (len(test_files) / len(src_files)) * 100
                current_coverage = min(current_coverage, 100)
            else:
                current_coverage = 0

            # 识别缺失测试的文件
            tested_modules = set(t.stem.replace("test_", "") for t in test_files)
            missing_tests = [
                f.stem for f in src_files
                if f.stem not in tested_modules and not f.stem.startswith("_")
            ]

            coverage_data[module_name] = ModuleCoverage(
                module_name=module_name,
                file_count=len(src_files),
                current_coverage=round(current_coverage, 2),
                target_coverage=module_info["target"],
                priority=module_info["priority"],
                test_files=[str(t) for t in test_files],
                missing_tests=missing_tests[:10]  # 只列出前10个
            )

        return coverage_data

    def identify_low_coverage_modules(self, coverage_data: Dict[str, ModuleCoverage]) -> List[ModuleCoverage]:
        """识别低覆盖率模块"""
        low_coverage = []

        for module in coverage_data.values():
            coverage_gap = module.target_coverage - module.current_coverage
            if coverage_gap > 10:  # 覆盖率差距超过10%
                low_coverage.append(module)

        # 按优先级和覆盖率差距排序
        low_coverage.sort(key=lambda m: (m.priority, -(m.target_coverage - m.current_coverage)))

        return low_coverage

    def generate_boost_phases(self, low_coverage: List[ModuleCoverage]) -> List[CoverageBoostPlan]:
        """生成分阶段提升计划"""
        phases = []

        # Phase 1: P1优先级模块（核心模块）
        p1_modules = [m for m in low_coverage if m.priority == 1]
        if p1_modules:
            phases.append(CoverageBoostPlan(
                phase=1,
                modules=[m.module_name for m in p1_modules],
                target_coverage=85.0,
                estimated_tests=sum(len(m.missing_tests) for m in p1_modules),
                priority_level="P1-核心关键"
            ))

        # Phase 2: P2优先级模块（重要模块）
        p2_modules = [m for m in low_coverage if m.priority == 2]
        if p2_modules:
            phases.append(CoverageBoostPlan(
                phase=2,
                modules=[m.module_name for m in p2_modules],
                target_coverage=80.0,
                estimated_tests=sum(len(m.missing_tests) for m in p2_modules),
                priority_level="P2-重要"
            ))

        # Phase 3: P3优先级模块（一般模块）
        p3_modules = [m for m in low_coverage if m.priority == 3]
        if p3_modules:
            phases.append(CoverageBoostPlan(
                phase=3,
                modules=[m.module_name for m in p3_modules],
                target_coverage=75.0,
                estimated_tests=sum(len(m.missing_tests) for m in p3_modules),
                priority_level="P3-一般"
            ))

        return phases

    def generate_report(self):
        """生成完整的分析报告"""
        print("=" * 80)
        print("基础设施层测试覆盖率系统性提升计划")
        print("=" * 80)
        print()

        # 1. 分析当前覆盖情况
        print("📊 Step 1: 识别当前覆盖情况")
        print("-" * 80)
        coverage_data = self.analyze_current_coverage()

        total_files = sum(m.file_count for m in coverage_data.values())
        total_tests = sum(len(m.test_files) for m in coverage_data.values())
        avg_coverage = sum(m.current_coverage for m in coverage_data.values()) / len(coverage_data)

        print(f"总源文件数: {total_files}")
        print(f"总测试文件数: {total_tests}")
        print(f"平均覆盖率: {avg_coverage:.2f}%")
        print()

        # 按覆盖率排序显示
        sorted_modules = sorted(coverage_data.values(), key=lambda m: m.current_coverage)
        print("各模块覆盖情况:")
        print(f"{'模块名':<15} {'源文件数':<10} {'测试数':<10} {'当前覆盖率':<12} {'目标覆盖率':<12} {'差距':<10} {'优先级':<10}")
        print("-" * 100)

        for module in sorted_modules:
            gap = module.target_coverage - module.current_coverage
            test_count = len(module.test_files)
            print(f"{module.module_name:<15} {module.file_count:<10} {test_count:<10} "
                f"{module.current_coverage:>6.2f}%     {module.target_coverage:>6.1f}%     "
                f"{gap:>6.2f}%    P{module.priority}")

        print()

        # 2. 识别低覆盖模块
        print("🔍 Step 2: 识别低覆盖模块")
        print("-" * 80)
        low_coverage = self.identify_low_coverage_modules(coverage_data)

        print(f"发现 {len(low_coverage)} 个低覆盖模块需要提升:")
        for module in low_coverage:
            gap = module.target_coverage - module.current_coverage
            print(f"  • {module.module_name}: {module.current_coverage:.2f}% → {module.target_coverage:.1f}% "
                f"(差距 {gap:.2f}%, 优先级 P{module.priority})")
            if module.missing_tests:
                print(f"    缺失测试示例: {', '.join(module.missing_tests[:5])}")

        print()

        # 3. 生成分阶段提升计划
        print("📋 Step 3: 生成分阶段提升计划")
        print("-" * 80)
        phases = self.generate_boost_phases(low_coverage)

        for phase in phases:
            print(f"\nPhase {phase.phase}: {phase.priority_level}")
            print(f"  涉及模块: {', '.join(phase.modules)}")
            print(f"  目标覆盖率: {phase.target_coverage}%")
            print(f"  预计新增测试: ~{phase.estimated_tests} 个")

        print()

        # 4. 保存详细报告
        print("💾 Step 4: 保存详细报告")
        print("-" * 80)

        report_data = {
            "summary": {
                "total_files": total_files,
                "total_tests": total_tests,
                "average_coverage": round(avg_coverage, 2),
                "low_coverage_modules": len(low_coverage),
            },
            "modules": {
                name: asdict(module)
                for name, module in coverage_data.items()
            },
            "low_coverage_modules": [
                asdict(module) for module in low_coverage
            ],
            "boost_phases": [
                asdict(phase) for phase in phases
            ],
        }

        report_file = self.project_root / "test_logs" / "infrastructure_coverage_boost_plan.json"
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"详细报告已保存至: {report_file}")
        print()

        # 5. 给出行动建议
        print("🎯 Step 5: 行动建议")
        print("-" * 80)
        print("建议按以下顺序提升测试覆盖率:")
        print()

        for i, phase in enumerate(phases, 1):
            print(f"{i}. Phase {phase.phase} ({phase.priority_level}):")
            for module_name in phase.modules:
                module = coverage_data[module_name]
                print(f"   • {module_name}:")
                print(f"     - 当前: {module.current_coverage:.2f}%, 目标: {module.target_coverage:.1f}%")
                print(f"     - 需要为以下文件添加测试: {', '.join(module.missing_tests[:3])}...")

        print()
        print("=" * 80)
        print("分析完成！可以开始按计划提升测试覆盖率。")
        print("=" * 80)


def main():
    """主函数"""
    analyzer = InfrastructureCoverageAnalyzer()
    analyzer.generate_report()


if __name__ == "__main__":
    main()
