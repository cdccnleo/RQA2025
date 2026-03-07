#!/usr/bin/env python3
"""
系统化测试覆盖率提升工具

按照标准流程提升测试覆盖率:
1. 识别低覆盖模块
2. 分析未覆盖代码
3. 生成测试用例模板
4. 验证覆盖率提升
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# 项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ModuleCoverage:
    """模块覆盖率信息"""
    module_name: str
    file_path: str
    total_statements: int
    covered_statements: int
    missing_lines: List[int] = field(default_factory=list)
    excluded_lines: List[int] = field(default_factory=list)
    
    @property
    def coverage_percentage(self) -> float:
        """计算覆盖率百分比"""
        if self.total_statements == 0:
            return 100.0
        return (self.covered_statements / self.total_statements) * 100
    
    @property
    def is_low_coverage(self) -> bool:
        """是否为低覆盖率模块"""
        return self.coverage_percentage < 85.0


@dataclass
class CoverageReport:
    """覆盖率报告"""
    timestamp: str
    total_coverage: float
    modules: List[ModuleCoverage] = field(default_factory=list)
    
    @property
    def low_coverage_modules(self) -> List[ModuleCoverage]:
        """低覆盖率模块列表"""
        return [m for m in self.modules if m.is_low_coverage]
    
    @property
    def high_coverage_modules(self) -> List[ModuleCoverage]:
        """高覆盖率模块列表"""
        return [m for m in self.modules if not m.is_low_coverage]


class SystematicCoverageImprover:
    """系统化覆盖率提升工具"""
    
    def __init__(self, target_path: str = "src/infrastructure", target_coverage: float = 85.0):
        self.target_path = Path(target_path)
        self.target_coverage = target_coverage
        self.coverage_data_dir = Path("test_logs/coverage")
        self.coverage_data_dir.mkdir(parents=True, exist_ok=True)
    
    def step1_identify_low_coverage_modules(self) -> CoverageReport:
        """
        步骤1: 识别低覆盖率模块
        
        运行测试并生成覆盖率报告，识别覆盖率<85%的模块
        """
        print("\n" + "=" * 80)
        print("📊 步骤1: 识别低覆盖率模块")
        print("=" * 80)
        
        # 运行测试获取覆盖率
        coverage_file = self.coverage_data_dir / "coverage.json"
        
        print(f"\n🔍 运行测试，目标路径: {self.target_path}")
        print(f"   覆盖率目标: {self.target_coverage}%")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            f"--cov={self.target_path}",
            f"--cov-report=json:{coverage_file}",
            "--cov-report=term-missing",
            "-v",
            "-n", "auto",
            "--tb=short"
        ]
        
        try:
            print("\n⏳ 执行测试... (这可能需要几分钟)")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                encoding='utf-8',
                errors='ignore'
            )
            
            # 解析覆盖率数据
            if coverage_file.exists():
                report = self._parse_coverage_report(coverage_file)
                self._print_coverage_summary(report)
                return report
            else:
                print("⚠️  覆盖率文件未生成")
                return None
                
        except subprocess.TimeoutExpired:
            print("⚠️  测试超时")
            return None
        except Exception as e:
            print(f"⚠️  执行失败: {e}")
            return None
    
    def _parse_coverage_report(self, coverage_file: Path) -> CoverageReport:
        """解析覆盖率报告"""
        with open(coverage_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        report = CoverageReport(
            timestamp=datetime.now().isoformat(),
            total_coverage=data.get('totals', {}).get('percent_covered', 0)
        )
        
        # 解析各模块覆盖率
        files_data = data.get('files', {})
        for file_path, file_data in files_data.items():
            # 只处理目标路径下的文件
            if str(self.target_path).replace('\\', '/') in file_path:
                summary = file_data.get('summary', {})
                
                module = ModuleCoverage(
                    module_name=Path(file_path).stem,
                    file_path=file_path,
                    total_statements=summary.get('num_statements', 0),
                    covered_statements=summary.get('covered_lines', 0),
                    missing_lines=file_data.get('missing_lines', []),
                    excluded_lines=file_data.get('excluded_lines', [])
                )
                report.modules.append(module)
        
        return report
    
    def _print_coverage_summary(self, report: CoverageReport):
        """打印覆盖率摘要"""
        print("\n" + "=" * 80)
        print("📊 覆盖率分析结果")
        print("=" * 80)
        
        print(f"\n总体覆盖率: {report.total_coverage:.2f}%")
        print(f"目标覆盖率: {self.target_coverage}%")
        
        gap = self.target_coverage - report.total_coverage
        if gap > 0:
            print(f"差距: {gap:.2f}% ⚠️")
        else:
            print(f"已达标: +{abs(gap):.2f}% ✅")
        
        # 低覆盖率模块
        low_coverage = report.low_coverage_modules
        print(f"\n低覆盖率模块 (<{self.target_coverage}%): {len(low_coverage)}个")
        
        if low_coverage:
            # 按覆盖率排序
            low_coverage.sort(key=lambda x: x.coverage_percentage)
            
            print("\n优先级排序（覆盖率从低到高）:")
            for i, module in enumerate(low_coverage[:20], 1):
                missing_count = len(module.missing_lines)
                status = "🔴" if module.coverage_percentage < 50 else "🟡"
                
                print(f"{i:2}. {status} {module.module_name:40} "
                      f"{module.coverage_percentage:5.1f}% "
                      f"({module.covered_statements}/{module.total_statements} lines, "
                      f"missing: {missing_count})")
        
        # 高覆盖率模块
        high_coverage = report.high_coverage_modules
        print(f"\n高覆盖率模块 (>={self.target_coverage}%): {len(high_coverage)}个 ✅")
    
    def step2_generate_test_templates(self, report: CoverageReport, limit: int = 5):
        """
        步骤2: 为低覆盖率模块生成测试模板
        
        Args:
            report: 覆盖率报告
            limit: 生成模板的模块数量限制
        """
        print("\n" + "=" * 80)
        print("📝 步骤2: 生成测试用例模板")
        print("=" * 80)
        
        low_coverage = report.low_coverage_modules[:limit]
        
        if not low_coverage:
            print("\n✅ 所有模块覆盖率已达标！")
            return
        
        print(f"\n为前{len(low_coverage)}个低覆盖率模块生成测试模板:")
        
        test_templates_dir = Path("tests/unit/infrastructure/generated")
        test_templates_dir.mkdir(parents=True, exist_ok=True)
        
        for module in low_coverage:
            self._generate_test_template(module, test_templates_dir)
    
    def _generate_test_template(self, module: ModuleCoverage, output_dir: Path):
        """为单个模块生成测试模板"""
        print(f"\n📄 生成测试模板: {module.module_name}")
        
        # 读取源代码
        try:
            source_file = Path(module.file_path)
            if not source_file.exists():
                print(f"   ⚠️  源文件不存在: {source_file}")
                return
            
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # 分析源代码找出需要测试的类和函数
            import ast
            try:
                tree = ast.parse(source_code)
            except SyntaxError:
                print(f"   ⚠️  语法错误，跳过")
                return
            
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    functions.append(node.name)
            
            # 生成测试模板
            test_file = output_dir / f"test_{module.module_name}.py"
            template = self._create_test_template(module, classes, functions)
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(template)
            
            print(f"   ✅ 已生成: {test_file}")
            print(f"   包含: {len(classes)}个类, {len(functions)}个函数")
            
        except Exception as e:
            print(f"   ⚠️  生成失败: {e}")
    
    def _create_test_template(self, module: ModuleCoverage, classes: List[str], functions: List[str]) -> str:
        """创建测试模板代码"""
        template_lines = [
            '"""',
            f'{module.module_name} 模块测试',
            '',
            '自动生成的测试模板，需要完善具体测试逻辑',
            f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'当前覆盖率: {module.coverage_percentage:.1f}%',
            f'目标覆盖率: 85.0%',
            f'需要覆盖的行数: {len(module.missing_lines)}',
            '"""',
            '',
            'import pytest',
            'from unittest.mock import Mock, patch, MagicMock',
            '',
            f'# 导入被测模块',
            f'# from {module.file_path.replace("/", ".").replace(".py", "")} import *',
            '',
            ''
        ]
        
        # 为每个类生成测试类
        for class_name in classes:
            template_lines.extend([
                '',
                f'class Test{class_name}:',
                f'    """测试{class_name}类"""',
                '',
                '    @pytest.fixture',
                '    def instance(self):',
                f'        """创建{class_name}实例"""',
                '        # TODO: 实现实例创建逻辑',
                '        pass',
                '',
                '    def test_initialization(self, instance):',
                f'        """测试{class_name}初始化"""',
                '        # TODO: 验证初始化逻辑',
                '        pass',
                '',
                '    def test_basic_operations(self, instance):',
                f'        """测试{class_name}基本操作"""',
                '        # TODO: 实现基本操作测试',
                '        pass',
                ''
            ])
        
        # 为独立函数生成测试
        if functions:
            template_lines.extend([
                '',
                'class TestModuleFunctions:',
                f'    """测试{module.module_name}模块函数"""',
                ''
            ])
            
            for func_name in functions[:10]:  # 限制数量
                template_lines.extend([
                    f'    def test_{func_name}(self):',
                    f'        """测试{func_name}函数"""',
                    '        # TODO: 实现测试逻辑',
                    '        pass',
                    ''
                ])
        
        # 添加覆盖率信息注释
        template_lines.extend([
            '',
            '# ============ 覆盖率改进指南 ============',
            f'# 当前覆盖率: {module.coverage_percentage:.1f}%',
            f'# 未覆盖行数: {len(module.missing_lines)}',
            '# ',
            '# 未覆盖的行号:',
            f'# {module.missing_lines[:50]}',  # 只显示前50行
            '# ',
            '# 改进建议:',
            '# 1. 为每个公共方法添加测试用例',
            '# 2. 测试正常流程和异常流程',
            '# 3. 测试边界条件',
            '# 4. 使用mock隔离外部依赖',
            '# 5. 运行测试验证覆盖率提升',
            ''
        ])
        
        return '\n'.join(template_lines)
    
    def step3_run_coverage_improvement_cycle(self, module_name: str):
        """
        步骤3: 运行覆盖率改进循环
        
        为指定模块运行测试，获取详细覆盖率信息
        """
        print("\n" + "=" * 80)
        print(f"🔄 步骤3: 覆盖率改进循环 - {module_name}")
        print("=" * 80)
        
        # 运行针对特定模块的测试
        print(f"\n运行{module_name}模块的测试...")
        
        # 这里可以运行pytest并获取详细覆盖率
        pass
    
    def step4_verify_coverage_improvement(self, before_report: CoverageReport) -> CoverageReport:
        """
        步骤4: 验证覆盖率提升
        
        运行测试并比较覆盖率变化
        """
        print("\n" + "=" * 80)
        print("✅ 步骤4: 验证覆盖率提升")
        print("=" * 80)
        
        # 重新运行覆盖率分析
        after_report = self.step1_identify_low_coverage_modules()
        
        if after_report and before_report:
            # 比较前后覆盖率
            improvement = after_report.total_coverage - before_report.total_coverage
            
            print(f"\n📈 覆盖率变化:")
            print(f"   之前: {before_report.total_coverage:.2f}%")
            print(f"   之后: {after_report.total_coverage:.2f}%")
            print(f"   提升: {improvement:+.2f}%")
            
            # 模块级别对比
            self._compare_module_coverage(before_report, after_report)
        
        return after_report
    
    def _compare_module_coverage(self, before: CoverageReport, after: CoverageReport):
        """比较模块级覆盖率变化"""
        before_map = {m.module_name: m for m in before.modules}
        after_map = {m.module_name: m for m in after.modules}
        
        improved = []
        degraded = []
        
        for name, after_module in after_map.items():
            if name in before_map:
                before_module = before_map[name]
                diff = after_module.coverage_percentage - before_module.coverage_percentage
                
                if diff > 1.0:
                    improved.append((name, diff))
                elif diff < -1.0:
                    degraded.append((name, diff))
        
        if improved:
            print(f"\n📈 改进的模块 ({len(improved)}个):")
            for name, diff in sorted(improved, key=lambda x: x[1], reverse=True)[:10]:
                print(f"   ✅ {name:40} +{diff:.2f}%")
        
        if degraded:
            print(f"\n📉 退化的模块 ({len(degraded)}个):")
            for name, diff in sorted(degraded, key=lambda x: x[1])[:5]:
                print(f"   ⚠️  {name:40} {diff:.2f}%")
    
    def generate_coverage_improvement_plan(self, report: CoverageReport) -> Dict[str, Any]:
        """生成覆盖率改进计划"""
        plan = {
            'timestamp': datetime.now().isoformat(),
            'current_coverage': report.total_coverage,
            'target_coverage': self.target_coverage,
            'gap': self.target_coverage - report.total_coverage,
            'modules_to_improve': [],
            'estimated_effort': '',
            'priority_actions': []
        }
        
        # 按覆盖率排序低覆盖模块
        low_coverage = sorted(report.low_coverage_modules, 
                            key=lambda x: x.coverage_percentage)
        
        for module in low_coverage:
            plan['modules_to_improve'].append({
                'module_name': module.module_name,
                'current_coverage': module.coverage_percentage,
                'missing_lines': len(module.missing_lines),
                'estimated_test_cases': self._estimate_test_cases(module),
                'priority': self._calculate_priority(module)
            })
        
        # 估算工作量
        total_test_cases = sum(m['estimated_test_cases'] for m in plan['modules_to_improve'])
        plan['estimated_effort'] = f"{total_test_cases * 15} 分钟 (~{total_test_cases * 15 / 60:.1f}小时)"
        
        # 优先行动
        plan['priority_actions'] = self._generate_priority_actions(low_coverage[:5])
        
        return plan
    
    def _estimate_test_cases(self, module: ModuleCoverage) -> int:
        """估算需要的测试用例数量"""
        # 简单估算：每20行缺失代码需要1个测试用例
        return max(1, len(module.missing_lines) // 20)
    
    def _calculate_priority(self, module: ModuleCoverage) -> str:
        """计算优先级"""
        if module.coverage_percentage < 50:
            return "🔴 极高"
        elif module.coverage_percentage < 70:
            return "🟡 高"
        else:
            return "🟢 中"
    
    def _generate_priority_actions(self, modules: List[ModuleCoverage]) -> List[str]:
        """生成优先行动列表"""
        actions = []
        
        for module in modules:
            action = f"为 {module.module_name} 添加测试 (当前: {module.coverage_percentage:.1f}%, 缺失: {len(module.missing_lines)}行)"
            actions.append(action)
        
        return actions
    
    def run_complete_improvement_cycle(self):
        """运行完整的改进循环"""
        print("\n🚀 系统化测试覆盖率提升流程")
        print("=" * 80)
        
        # 步骤1: 识别低覆盖模块
        initial_report = self.step1_identify_low_coverage_modules()
        
        if not initial_report:
            print("\n❌ 无法获取覆盖率报告")
            return
        
        # 步骤2: 生成测试模板
        self.step2_generate_test_templates(initial_report, limit=10)
        
        # 生成改进计划
        plan = self.generate_coverage_improvement_plan(initial_report)
        
        # 保存计划
        plan_file = self.coverage_data_dir / "improvement_plan.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 改进计划已保存: {plan_file}")
        
        # 打印计划摘要
        self._print_improvement_plan(plan)
        
        return plan
    
    def _print_improvement_plan(self, plan: Dict[str, Any]):
        """打印改进计划"""
        print("\n" + "=" * 80)
        print("📋 覆盖率改进计划")
        print("=" * 80)
        
        print(f"\n当前覆盖率: {plan['current_coverage']:.2f}%")
        print(f"目标覆盖率: {plan['target_coverage']:.2f}%")
        print(f"差距: {plan['gap']:.2f}%")
        
        print(f"\n需要改进的模块: {len(plan['modules_to_improve'])}个")
        print(f"预计工作量: {plan['estimated_effort']}")
        
        print("\n🎯 优先行动（Top 5）:")
        for i, action in enumerate(plan['priority_actions'], 1):
            print(f"   {i}. {action}")
        
        print("\n" + "=" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='系统化测试覆盖率提升工具')
    parser.add_argument('--target', default='src/infrastructure', help='目标路径')
    parser.add_argument('--threshold', type=float, default=85.0, help='目标覆盖率')
    parser.add_argument('--generate-templates', action='store_true', help='生成测试模板')
    parser.add_argument('--limit', type=int, default=10, help='生成模板数量限制')
    
    args = parser.parse_args()
    
    improver = SystematicCoverageImprover(
        target_path=args.target,
        target_coverage=args.threshold
    )
    
    # 运行完整改进循环
    improver.run_complete_improvement_cycle()
    
    print("\n✅ 覆盖率分析完成！")
    print("\n下一步:")
    print("1. 查看生成的测试模板: tests/unit/infrastructure/generated/")
    print("2. 完善测试模板中的TODO部分")
    print("3. 运行测试验证覆盖率提升")
    print("4. 重复此流程直到达标")


if __name__ == '__main__':
    main()

