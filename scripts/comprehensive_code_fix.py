#!/usr/bin/env python3
"""
全面修复代码质量问题的脚本

处理flake8检查出的所有问题类型
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict


class ComprehensiveCodeFixer:
    """全面代码质量修复器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.fixed_files: List[str] = []
        self.stats = {
            'files_processed': 0,
            'issues_fixed': 0,
            'f541_fixed': 0,
            'whitespace_fixed': 0,
            'indentation_fixed': 0,
            'syntax_fixed': 0
        }

    def fix_f541_fstring_placeholders(self, file_path: Path) -> bool:
        """修复F541: f-string is missing placeholders"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            # 修复双引号f字符串
            def fix_fstring_double(match):
                fstring_content = match.group(1)
                # 检查是否包含变量占位符
                if '{' in fstring_content and '}' in fstring_content:
                    # 检查是否有未闭合的大括号
                    if fstring_content.count('{') != fstring_content.count('}'):
                        # 尝试修复未闭合的大括号
                        return match.group(0)  # 暂时保持不变
                    return match.group(0)  # 正常f字符串
                else:
                    # 不包含变量，移除f前缀
                    nonlocal modified
                    modified = True
                    self.stats['f541_fixed'] += 1
                    return f'"{fstring_content}"'

            # 修复单引号f字符串
            def fix_fstring_single(match):
                fstring_content = match.group(1)
                if '{' in fstring_content and '}' in fstring_content:
                    if fstring_content.count('{') != fstring_content.count('}'):
                        return match.group(0)
                    return match.group(0)
                else:
                    nonlocal modified
                    modified = True
                    self.stats['f541_fixed'] += 1
                    return f"'{fstring_content}'"

            # 应用修复
            content = re.sub(r'f"([^"]*)"', fix_fstring_double, content)
            content = re.sub(r"f'([^']*)'", fix_fstring_single, content)

            if modified and content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.append(str(file_path))
                return True

        except Exception as e:
            print(f"处理F541时出错 {file_path}: {e}")

        return False

    def fix_whitespace_issues(self, file_path: Path) -> bool:
        """修复空白字符相关问题 (W291, W293, W391)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            lines = content.split('\n')

            # 修复每行的空白问题
            for i, line in enumerate(lines):
                original_line = line

                # W291: 行尾空白
                if line.rstrip('\n').endswith((' ', '\t')):
                    lines[i] = line.rstrip() + ('\n' if line.endswith('\n') else '')
                    modified = True
                    self.stats['whitespace_fixed'] += 1

                # W293: 只有空白的行
                if not line.strip() and line.strip('\n'):
                    lines[i] = '\n' if line.endswith('\n') else ''
                    modified = True
                    self.stats['whitespace_fixed'] += 1

            # W391: 文件末尾空白行
            while len(lines) >= 2 and lines[-1].strip() == '' and lines[-2].strip() == '':
                lines.pop()
                modified = True
                self.stats['whitespace_fixed'] += 1

            # 确保文件以换行符结尾
            if lines and lines[-1] and not lines[-1].endswith('\n'):
                lines[-1] += '\n'
                modified = True

            if modified:
                content = '\n'.join(lines)
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    if str(file_path) not in self.fixed_files:
                        self.fixed_files.append(str(file_path))
                    return True

        except Exception as e:
            print(f"处理空白问题时出错 {file_path}: {e}")

        return False

    def fix_indentation_issues(self, file_path: Path) -> bool:
        """修复缩进问题 (E128)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            original_lines = lines.copy()
            modified = False

            # 这里可以添加更复杂的缩进修复逻辑
            # 目前主要处理一些常见的缩进问题

            # 简单检查：确保连续行有合理的缩进
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue

                # 检查行是否以合理的缩进开始
                indent = len(line) - len(line.lstrip())
                if indent > 0 and indent % 4 != 0:
                    # 非标准缩进，尝试修复
                    new_indent = (indent // 4) * 4  # 向下取整到4的倍数
                    if new_indent != indent:
                        lines[i] = ' ' * new_indent + line.lstrip()
                        modified = True
                        self.stats['indentation_fixed'] += 1

            if modified and lines != original_lines:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                if str(file_path) not in self.fixed_files:
                    self.fixed_files.append(str(file_path))
                return True

        except Exception as e:
            print(f"处理缩进问题时出错 {file_path}: {e}")

        return False

    def fix_syntax_errors(self, file_path: Path) -> bool:
        """修复语法错误 (E999)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            # 修复f字符串语法错误
            lines = content.split('\n')
            for i, line in enumerate(lines):
                # 查找可能的问题行
                if 'f"' in line or "f'" in line:
                    # 检查f字符串语法
                    fstring_matches = re.findall(r'f"[^"]*"', line) + re.findall(r"f'[^']*'", line)
                    for match in fstring_matches:
                        # 检查大括号是否匹配
                        if match.count('{') != match.count('}'):
                            # 尝试简单修复
                            if match.count('{') > match.count('}'):
                                lines[i] = line.replace(match, match + '}')
                                modified = True
                                self.stats['syntax_fixed'] += 1

            if modified:
                content = '\n'.join(lines)
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    if str(file_path) not in self.fixed_files:
                        self.fixed_files.append(str(file_path))
                    return True

        except Exception as e:
            print(f"处理语法错误时出错 {file_path}: {e}")

        return False

    def remove_unused_variables(self, file_path: Path) -> bool:
        """移除未使用的变量 (F841)"""
        # 这个比较复杂，暂时跳过
        return False

    def process_file(self, file_path: Path) -> bool:
        """处理单个文件"""
        if not file_path.exists() or file_path.suffix != '.py':
            return False

        self.stats['files_processed'] += 1
        modified = False

        # 按优先级修复问题
        modified |= self.fix_syntax_errors(file_path)
        modified |= self.fix_f541_fstring_placeholders(file_path)
        modified |= self.fix_whitespace_issues(file_path)
        modified |= self.fix_indentation_issues(file_path)

        if modified:
            self.stats['issues_fixed'] += 1

        return modified

    def get_problematic_files(self) -> List[Path]:
        """获取有问题的文件列表"""
        # 基于提供的错误信息手动指定文件列表
        problematic_files = [
            "tests/deployment_launch_task.py",
            "tests/deployment_preparation.py",
            "tests/dev_env_setup_task.py",
            "tests/final_coverage_summary.py",
            "tests/fixtures/__init__.py",
            "tests/fixtures/config_storage_test_fixtures.py",
            "tests/fixtures/config_test_fixtures.py",
            "tests/fixtures/infrastructure_mocks.py",
            "tests/future_innovation_test_framework.py",
            "tests/import_manager.py",
            "tests/incremental_tester.py",
            "tests/incremental_tester_optimized.py",
            "tests/infrastructure_coverage_boost_plan.py",
            "tests/infrastructure_detailed_coverage_report.py",
            "tests/intelligent_reporter.py",
            "tests/integration/framework.py",
            "tests/infrastructure_coverage_boost_plan.py",
            "tests/kubernetes_tester.py",
            "tests/microservice_tester.py",
            "tests/ml_predictor.py",
            "tests/multilang_adapter.py",
            "tests/operations_monitoring_setup.py",
            "tests/performance/benchmark_framework.py",
            "tests/performance/load_stress_tester.py",
            "tests/performance/locustfile.py",
            "tests/performance/memory_leak_detector.py",
            "tests/performance/optimizer.py",
            "tests/performance/performance_monitor.py",
            "tests/performance_benchmark_framework.py",
            "tests/performance_monitor.py",
            "tests/performance_optimization.py",
            "tests/performance_test_runner.py",
            "tests/production_deployment.py",
            "tests/production_operations.py",
            "tests/production_readiness_assessment.py",
            "tests/project_delivery_generator.py",
            "tests/project_final_validation.py",
            "tests/project_management_tools_task.py",
            "tests/project_retrospective_task.py",
            "tests/quantum_lab_establishment.py",
            "tests/rqa2025_intelligent_ops.py",
            "tests/rqa2026_ai_developer.py",
            "tests/rqa_global_expansion_planner.py",
            "tests/rqa_knowledge_system_builder.py",
            "tests/rqa_post_project_documentation_generator.py",
            "tests/rqa_project_final_summary_generator.py",
            "tests/rqa_risk_management_optimizer.py",
            "tests/rqa_tech_innovation_explorer.py",
            "tests/run_accurate_coverage_report.py",
            "tests/run_enhanced_test_suite.py",
            "tests/run_ml_coverage_focused.py",
            "tests/run_parallel_coverage.py",
            "tests/run_smart_coverage.py",
            "tests/run_tests.py",
            "tests/simple_doc_generator.py",
            "tests/system_integration_task.py",
            "tests/team_division_task.py",
            "tests/tech_stack_evaluation_task.py",
            "tests/trading_execution_system_task.py",
            "tests/user_interface_development_task.py",
            # conftest.py文件
            "tests/fixtures/conftest.py",
            "tests/unit/adapters/conftest.py",
            "tests/unit/business/conftest.py",
            "tests/unit/core/conftest.py",
            "tests/unit/data/conftest.py",
            "tests/unit/distributed/conftest.py",
            "tests/unit/features/conftest.py",
            "tests/unit/gateway/conftest.py",
            "tests/unit/infrastructure/conftest.py",
            "tests/unit/ml/conftest.py",
            "tests/unit/mobile/conftest.py",
            "tests/unit/monitoring/conftest.py",
            "tests/unit/optimization/conftest.py",
            "tests/unit/risk/conftest.py",
            "tests/unit/streaming/conftest.py",
            "tests/unit/trading/broker/__init__.py",
            "tests/unit/trading/performance/__init__.py",
            "tests/unit/trading/portfolio/__init__.py",
            "tests/unit/trading/realtime/__init__.py",
            "tests/unit/trading/settlement/__init__.py",
            "tests/unit/trading/signal/__init__.py",
            "tests/unit/infrastructure/api/__init__.py",
            "tests/unit/infrastructure/base/__init__.py",
            "tests/unit/infrastructure/config/__init__.py",
            "tests/unit/infrastructure/core/__init__.py",
            "tests/unit/infrastructure/distributed/__init__.py",
            "tests/unit/infrastructure/error/__init__.py",
            "tests/unit/infrastructure/health/__init__.py",
            "tests/unit/infrastructure/interfaces/__init__.py",
            "tests/unit/infrastructure/logging/__init__.py",
            "tests/unit/infrastructure/monitoring/__init__.py",
            "tests/unit/infrastructure/ops/__init__.py",
            "tests/unit/infrastructure/optimization/__init__.py",
            "tests/unit/infrastructure/resource/__init__.py",
            "tests/unit/infrastructure/service/__init__.py",
            "tests/unit/infrastructure/utils/__init__.py",
            "tests/unit/infrastructure/versioning/__init__.py",
            "tests/unit/risk/pytest_import_helper.py",
            "tests/unit/streaming/engine/__init__.py"
        ]

        return [self.project_root / f for f in problematic_files if (self.project_root / f).exists()]

    def run_comprehensive_fix(self) -> Dict:
        """运行全面修复"""
        print("🔧 开始全面代码质量修复...")

        problematic_files = self.get_problematic_files()
        print(f"发现 {len(problematic_files)} 个有问题的文件")

        total_processed = 0
        total_fixed = 0

        for file_path in problematic_files:
            print(f"处理文件: {file_path.name}")
            if self.process_file(file_path):
                total_fixed += 1
                print("  ✅ 已修复")
            else:
                print("  ℹ️ 无需修复")
            total_processed += 1

        # 打印统计信息
        print(f"\n📊 修复完成!")
        print(f"📁 处理文件数: {total_processed}")
        print(f"🔧 修复文件数: {total_fixed}")
        print(f"F541修复数: {self.stats['f541_fixed']}")
        print(f"空白修复数: {self.stats['whitespace_fixed']}")
        print(f"缩进修复数: {self.stats['indentation_fixed']}")
        print(f"语法修复数: {self.stats['syntax_fixed']}")

        return self.stats


def main():
    """主函数"""
    project_root = Path(__file__).resolve().parent.parent

    fixer = ComprehensiveCodeFixer(project_root)
    stats = fixer.run_comprehensive_fix()

    if stats['issues_fixed'] > 0:
        print("\n🎉 建议运行以下命令验证修复结果:")
        print("flake8 tests/ --select=F541,E999,W291,W293,W391,E128 --show-source --statistics")
        print("\n或者运行完整的质量监控:")
        print("python scripts/quality_monitor.py --check code")
    else:
        print("\nℹ️ 没有发现需要修复的问题")


if __name__ == "__main__":
    main()
