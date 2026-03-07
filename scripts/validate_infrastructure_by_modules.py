#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础设施层子模块测试覆盖率和通过率验证脚本
按照子模块逐个运行测试并生成报告
"""

import subprocess
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# 基础设施层子模块定义
INFRASTRUCTURE_MODULES = [
    "cache",
    "config", 
    "constants",
    "core",
    "distributed",
    "error",
    "health",
    "logging",
    "monitoring",
    "ops",
    "resource",
    "security",
    "utils",
    "versioning",
]

# 投产要求标准
PRODUCTION_REQUIREMENTS = {
    "infrastructure_coverage": 52.0,  # 第一阶段目标
    "infrastructure_final_coverage": 55.0,  # 最终目标
    "overall_coverage": 70.0,
    "pass_rate": 98.0,
}


class InfrastructureValidator:
    """基础设施层验证器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.test_logs_dir = self.base_dir / "test_logs"
        self.test_logs_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_module_tests(self, module_name: str) -> Dict:
        """运行单个子模块的测试"""
        print(f"\n{'='*80}")
        print(f"正在验证子模块: {module_name}")
        print(f"{'='*80}")
        
        # 查找该模块的测试文件
        test_paths = []
        
        # tests/infrastructure/子模块
        infra_test_dir = self.base_dir / "tests" / "infrastructure"
        if infra_test_dir.exists():
            module_tests = list(infra_test_dir.glob(f"*{module_name}*.py"))
            test_paths.extend([str(p) for p in module_tests])
        
        # tests/unit/infrastructure/子模块
        unit_test_dir = self.base_dir / "tests" / "unit" / "infrastructure" / module_name
        if unit_test_dir.exists():
            test_paths.append(str(unit_test_dir))
        
        if not test_paths:
            print(f"⚠️  未找到 {module_name} 的测试文件")
            return {
                "module": module_name,
                "status": "no_tests",
                "coverage": 0,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
            }
        
        print(f"找到测试路径: {test_paths}")
        
        # 构建pytest命令
        coverage_json = self.test_logs_dir / f"coverage_{module_name}_{self.timestamp}.json"
        coverage_html = self.test_logs_dir / f"coverage_{module_name}_{self.timestamp}"
        
        cmd = [
            "pytest",
            *test_paths,
            "-v",
            "--tb=short",
            "-n", "auto",
            f"--cov=src/infrastructure/{module_name}",
            "--cov-report=term",
            f"--cov-report=html:{coverage_html}",
            f"--cov-report=json:{coverage_json}",
            "--maxfail=10",  # 最多失败10个就停止
            "-x",  # 遇到第一个错误停止（更快定位问题）
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
            )
            
            # 解析结果
            output = result.stdout + result.stderr
            
            # 解析覆盖率
            coverage = 0.0
            if coverage_json.exists():
                with open(coverage_json, 'r', encoding='utf-8') as f:
                    cov_data = json.load(f)
                    coverage = cov_data.get('totals', {}).get('percent_covered', 0.0)
            
            # 解析测试结果
            total, passed, failed, errors = self._parse_test_results(output)
            
            result_data = {
                "module": module_name,
                "status": "completed",
                "coverage": coverage,
                "total": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "pass_rate": (passed / total * 100) if total > 0 else 0,
                "coverage_file": str(coverage_json),
                "html_report": str(coverage_html / "index.html"),
            }
            
            print(f"\n✅ {module_name} 验证完成:")
            print(f"   覆盖率: {coverage:.2f}%")
            print(f"   测试总数: {total}")
            print(f"   通过: {passed}")
            print(f"   失败: {failed}")
            print(f"   错误: {errors}")
            print(f"   通过率: {result_data['pass_rate']:.2f}%")
            
            return result_data
            
        except subprocess.TimeoutExpired:
            print(f"⚠️  {module_name} 测试超时（5分钟）")
            return {
                "module": module_name,
                "status": "timeout",
                "coverage": 0,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
            }
        except Exception as e:
            print(f"❌ {module_name} 测试出错: {e}")
            return {
                "module": module_name,
                "status": "error",
                "error_msg": str(e),
                "coverage": 0,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
            }
    
    def _parse_test_results(self, output: str) -> Tuple[int, int, int, int]:
        """解析pytest输出获取测试结果"""
        total = passed = failed = errors = 0
        
        # 查找类似 "10 passed, 2 failed" 的行
        import re
        
        # 匹配 passed
        passed_match = re.search(r'(\d+) passed', output)
        if passed_match:
            passed = int(passed_match.group(1))
        
        # 匹配 failed
        failed_match = re.search(r'(\d+) failed', output)
        if failed_match:
            failed = int(failed_match.group(1))
        
        # 匹配 error
        error_match = re.search(r'(\d+) error', output)
        if error_match:
            errors = int(error_match.group(1))
        
        total = passed + failed + errors
        
        return total, passed, failed, errors
    
    def validate_all_modules(self) -> Dict:
        """验证所有子模块"""
        print("\n" + "="*80)
        print("基础设施层子模块测试覆盖率和通过率验证")
        print("="*80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"子模块数量: {len(INFRASTRUCTURE_MODULES)}")
        print("="*80)
        
        for module in INFRASTRUCTURE_MODULES:
            result = self.run_module_tests(module)
            self.results[module] = result
        
        # 生成汇总报告
        summary = self.generate_summary()
        
        # 保存结果
        self.save_results(summary)
        
        # 打印汇总
        self.print_summary(summary)
        
        return summary
    
    def generate_summary(self) -> Dict:
        """生成汇总报告"""
        total_coverage = 0
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        completed_modules = 0
        
        module_details = []
        
        for module, result in self.results.items():
            if result['status'] == 'completed':
                total_coverage += result['coverage']
                total_tests += result['total']
                total_passed += result['passed']
                total_failed += result['failed']
                total_errors += result['errors']
                completed_modules += 1
                
                module_details.append({
                    "module": module,
                    "coverage": result['coverage'],
                    "total": result['total'],
                    "passed": result['passed'],
                    "failed": result['failed'],
                    "errors": result['errors'],
                    "pass_rate": result['pass_rate'],
                    "达标状态": "✅" if result['coverage'] >= 50 else "⚠️"
                })
        
        avg_coverage = total_coverage / completed_modules if completed_modules > 0 else 0
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # 检查是否达标
        infrastructure_达标 = avg_coverage >= PRODUCTION_REQUIREMENTS['infrastructure_coverage']
        通过率达标 = pass_rate >= PRODUCTION_REQUIREMENTS['pass_rate']
        
        summary = {
            "验证时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "总子模块数": len(INFRASTRUCTURE_MODULES),
            "完成验证数": completed_modules,
            "平均覆盖率": avg_coverage,
            "测试总数": total_tests,
            "通过数": total_passed,
            "失败数": total_failed,
            "错误数": total_errors,
            "通过率": pass_rate,
            "子模块详情": module_details,
            "投产要求": PRODUCTION_REQUIREMENTS,
            "达标情况": {
                "基础设施覆盖率": {
                    "实际值": avg_coverage,
                    "目标值": PRODUCTION_REQUIREMENTS['infrastructure_coverage'],
                    "最终目标": PRODUCTION_REQUIREMENTS['infrastructure_final_coverage'],
                    "达标": infrastructure_达标,
                    "状态": "✅ 达标" if infrastructure_达标 else "⚠️ 未达标"
                },
                "测试通过率": {
                    "实际值": pass_rate,
                    "目标值": PRODUCTION_REQUIREMENTS['pass_rate'],
                    "达标": 通过率达标,
                    "状态": "✅ 达标" if 通过率达标 else "⚠️ 未达标"
                },
                "整体评估": "✅ 符合投产要求" if (infrastructure_达标 and 通过率达标) else "⚠️ 需要改进"
            }
        }
        
        return summary
    
    def save_results(self, summary: Dict):
        """保存验证结果"""
        # 保存JSON格式
        json_file = self.test_logs_dir / f"infrastructure_validation_{self.timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "detailed_results": self.results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 详细结果已保存: {json_file}")
        
        # 保存Markdown报告
        md_file = self.test_logs_dir / f"infrastructure_validation_{self.timestamp}.md"
        self.generate_markdown_report(summary, md_file)
        print(f"📄 Markdown报告已保存: {md_file}")
    
    def generate_markdown_report(self, summary: Dict, output_file: Path):
        """生成Markdown格式的报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 基础设施层测试覆盖率和通过率验证报告\n\n")
            f.write(f"**验证时间**: {summary['验证时间']}\n\n")
            
            f.write("## 📊 整体情况\n\n")
            f.write(f"- **总子模块数**: {summary['总子模块数']}\n")
            f.write(f"- **完成验证数**: {summary['完成验证数']}\n")
            f.write(f"- **平均覆盖率**: {summary['平均覆盖率']:.2f}%\n")
            f.write(f"- **测试总数**: {summary['测试总数']}\n")
            f.write(f"- **通过数**: {summary['通过数']}\n")
            f.write(f"- **失败数**: {summary['失败数']}\n")
            f.write(f"- **错误数**: {summary['错误数']}\n")
            f.write(f"- **通过率**: {summary['通过率']:.2f}%\n\n")
            
            f.write("## 🎯 投产要求达标情况\n\n")
            达标情况 = summary['达标情况']
            
            f.write("### 基础设施层覆盖率\n\n")
            infra_cov = 达标情况['基础设施覆盖率']
            f.write(f"- **实际值**: {infra_cov['实际值']:.2f}%\n")
            f.write(f"- **第一阶段目标**: {infra_cov['目标值']:.2f}%\n")
            f.write(f"- **最终目标**: {infra_cov['最终目标']:.2f}%\n")
            f.write(f"- **状态**: {infra_cov['状态']}\n\n")
            
            f.write("### 测试通过率\n\n")
            pass_rate = 达标情况['测试通过率']
            f.write(f"- **实际值**: {pass_rate['实际值']:.2f}%\n")
            f.write(f"- **目标值**: {pass_rate['目标值']:.2f}%\n")
            f.write(f"- **状态**: {pass_rate['状态']}\n\n")
            
            f.write(f"### 整体评估\n\n")
            f.write(f"**{达标情况['整体评估']}**\n\n")
            
            f.write("## 📋 子模块详情\n\n")
            f.write("| 子模块 | 覆盖率 | 测试总数 | 通过 | 失败 | 错误 | 通过率 | 状态 |\n")
            f.write("|--------|--------|----------|------|------|------|--------|------|\n")
            
            for detail in summary['子模块详情']:
                f.write(f"| {detail['module']} | {detail['coverage']:.2f}% | "
                       f"{detail['total']} | {detail['passed']} | {detail['failed']} | "
                       f"{detail['errors']} | {detail['pass_rate']:.2f}% | {detail['达标状态']} |\n")
            
            f.write("\n## 📝 说明\n\n")
            f.write("- ✅ 表示该子模块覆盖率达到50%以上\n")
            f.write("- ⚠️ 表示该子模块覆盖率低于50%，需要改进\n")
            f.write("- 投产要求：基础设施层平均覆盖率≥52%，测试通过率≥98%\n")
    
    def print_summary(self, summary: Dict):
        """打印汇总报告"""
        print("\n" + "="*80)
        print("📊 基础设施层验证汇总报告")
        print("="*80)
        
        print(f"\n整体情况:")
        print(f"  平均覆盖率: {summary['平均覆盖率']:.2f}%")
        print(f"  测试通过率: {summary['通过率']:.2f}%")
        print(f"  完成验证: {summary['完成验证数']}/{summary['总子模块数']} 个子模块")
        
        print(f"\n投产要求达标情况:")
        达标 = summary['达标情况']
        print(f"  基础设施覆盖率: {达标['基础设施覆盖率']['状态']}")
        print(f"  测试通过率: {达标['测试通过率']['状态']}")
        print(f"  整体评估: {达标['整体评估']}")
        
        print(f"\n子模块覆盖率TOP5:")
        sorted_modules = sorted(summary['子模块详情'], 
                               key=lambda x: x['coverage'], reverse=True)
        for i, module in enumerate(sorted_modules[:5], 1):
            print(f"  {i}. {module['module']}: {module['coverage']:.2f}%")
        
        print(f"\n需要改进的子模块:")
        low_coverage = [m for m in sorted_modules if m['coverage'] < 50]
        if low_coverage:
            for module in low_coverage:
                print(f"  ⚠️  {module['module']}: {module['coverage']:.2f}%")
        else:
            print("  ✅ 所有子模块覆盖率都达到50%以上")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    validator = InfrastructureValidator()
    summary = validator.validate_all_modules()
    
    # 返回退出码
    if summary['达标情况']['整体评估'] == "✅ 符合投产要求":
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())

