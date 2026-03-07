#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础设施层快速评估脚本
不运行完整测试，只统计代码和测试文件情况
"""

import os
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime


class QuickInfrastructureAssessment:
    """快速评估基础设施层情况"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.src_infra = self.base_dir / "src" / "infrastructure"
        self.test_infra = self.base_dir / "tests" / "infrastructure"
        self.test_unit_infra = self.base_dir / "tests" / "unit" / "infrastructure"
        
        # 基础设施层子模块
        self.modules = [
            "cache", "config", "constants", "core", "distributed",
            "error", "health", "logging", "monitoring", "ops",
            "resource", "security", "utils", "versioning"
        ]
    
    def count_python_files(self, directory: Path) -> int:
        """统计Python文件数量"""
        if not directory.exists():
            return 0
        return len(list(directory.rglob("*.py")))
    
    def count_lines_of_code(self, directory: Path) -> int:
        """统计代码行数"""
        if not directory.exists():
            return 0
        
        total_lines = 0
        for py_file in directory.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # 排除空行和注释行
                    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
                    total_lines += len(code_lines)
            except:
                pass
        
        return total_lines
    
    def assess_module(self, module_name: str) -> Dict:
        """评估单个子模块"""
        src_dir = self.src_infra / module_name
        test_dir = self.test_infra
        unit_test_dir = self.test_unit_infra / module_name
        
        # 统计源代码
        src_files = self.count_python_files(src_dir)
        src_lines = self.count_lines_of_code(src_dir)
        
        # 统计测试代码
        # tests/infrastructure/下与该模块相关的测试
        test_files = 0
        test_lines = 0
        if test_dir.exists():
            related_tests = list(test_dir.glob(f"*{module_name}*.py"))
            test_files += len(related_tests)
            for test_file in related_tests:
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
                        test_lines += len(code_lines)
                except:
                    pass
        
        # tests/unit/infrastructure/module/下的测试
        unit_test_files = self.count_python_files(unit_test_dir)
        unit_test_lines = self.count_lines_of_code(unit_test_dir)
        
        total_test_files = test_files + unit_test_files
        total_test_lines = test_lines + unit_test_lines
        
        # 计算测试覆盖指标（粗略估计）
        test_to_code_ratio = (total_test_lines / src_lines * 100) if src_lines > 0 else 0
        
        # 评估状态
        if src_files == 0:
            status = "无源码"
        elif total_test_files == 0:
            status = "❌ 无测试"
        elif test_to_code_ratio < 30:
            status = "⚠️ 测试不足"
        elif test_to_code_ratio < 60:
            status = "🟡 测试一般"
        else:
            status = "✅ 测试充分"
        
        return {
            "module": module_name,
            "src_files": src_files,
            "src_lines": src_lines,
            "test_files": total_test_files,
            "test_lines": total_test_lines,
            "test_to_code_ratio": test_to_code_ratio,
            "status": status,
            "priority": self._calculate_priority(src_lines, total_test_lines, test_to_code_ratio)
        }
    
    def _calculate_priority(self, src_lines: int, test_lines: int, ratio: float) -> str:
        """计算优先级"""
        if src_lines > 1000 and ratio < 30:
            return "🔴 高优先级"
        elif src_lines > 500 and ratio < 50:
            return "🟡 中优先级"
        elif ratio < 30:
            return "🟢 低优先级"
        else:
            return "✅ 已充分测试"
    
    def run_assessment(self) -> Dict:
        """运行评估"""
        print("\n" + "="*80)
        print("基础设施层快速评估")
        print("="*80)
        print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        results = []
        total_src_lines = 0
        total_test_lines = 0
        
        for module in self.modules:
            result = self.assess_module(module)
            results.append(result)
            total_src_lines += result['src_lines']
            total_test_lines += result['test_lines']
            
            print(f"{result['module']:15} | "
                  f"源码: {result['src_lines']:6}行 | "
                  f"测试: {result['test_lines']:6}行 | "
                  f"比率: {result['test_to_code_ratio']:5.1f}% | "
                  f"{result['status']:12} | {result['priority']}")
        
        overall_ratio = (total_test_lines / total_src_lines * 100) if total_src_lines > 0 else 0
        
        print("\n" + "="*80)
        print(f"整体情况:")
        print(f"  源代码总行数: {total_src_lines:,}")
        print(f"  测试代码总行数: {total_test_lines:,}")
        print(f"  测试/代码比率: {overall_ratio:.2f}%")
        print("="*80)
        
        # 优先级建议
        print("\n优先处理建议:")
        high_priority = [r for r in results if "高优先级" in r['priority']]
        medium_priority = [r for r in results if "中优先级" in r['priority']]
        
        if high_priority:
            print("\n🔴 高优先级模块（大型模块且测试不足）:")
            for r in high_priority:
                print(f"  - {r['module']}: {r['src_lines']}行代码，测试比率仅{r['test_to_code_ratio']:.1f}%")
        
        if medium_priority:
            print("\n🟡 中优先级模块（中型模块需要补充测试）:")
            for r in medium_priority:
                print(f"  - {r['module']}: {r['src_lines']}行代码，测试比率{r['test_to_code_ratio']:.1f}%")
        
        # 保存结果
        self.save_results(results, overall_ratio)
        
        return {
            "modules": results,
            "total_src_lines": total_src_lines,
            "total_test_lines": total_test_lines,
            "overall_ratio": overall_ratio
        }
    
    def save_results(self, results: List[Dict], overall_ratio: float):
        """保存评估结果"""
        test_logs_dir = self.base_dir / "test_logs"
        test_logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON格式
        json_file = test_logs_dir / f"infrastructure_assessment_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                "assessment_time": datetime.now().isoformat(),
                "modules": results,
                "overall_ratio": overall_ratio
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 评估结果已保存: {json_file}")
        
        # Markdown格式
        md_file = test_logs_dir / f"infrastructure_assessment_{timestamp}.md"
        self.save_markdown_report(results, overall_ratio, md_file)
        print(f"📄 Markdown报告已保存: {md_file}")
    
    def save_markdown_report(self, results: List[Dict], overall_ratio: float, output_file: Path):
        """保存Markdown报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 基础设施层快速评估报告\n\n")
            f.write(f"**评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 整体情况\n\n")
            total_src = sum(r['src_lines'] for r in results)
            total_test = sum(r['test_lines'] for r in results)
            f.write(f"- 源代码总行数: {total_src:,}\n")
            f.write(f"- 测试代码总行数: {total_test:,}\n")
            f.write(f"- 测试/代码比率: {overall_ratio:.2f}%\n\n")
            
            f.write("## 📋 子模块详情\n\n")
            f.write("| 子模块 | 源码行数 | 测试行数 | 测试比率 | 状态 | 优先级 |\n")
            f.write("|--------|----------|----------|----------|------|--------|\n")
            
            for r in sorted(results, key=lambda x: x['src_lines'], reverse=True):
                f.write(f"| {r['module']} | {r['src_lines']:,} | {r['test_lines']:,} | "
                       f"{r['test_to_code_ratio']:.1f}% | {r['status']} | {r['priority']} |\n")
            
            f.write("\n## 🎯 优先处理建议\n\n")
            
            high_priority = [r for r in results if "高优先级" in r['priority']]
            if high_priority:
                f.write("### 🔴 高优先级模块\n\n")
                for r in high_priority:
                    f.write(f"- **{r['module']}**: {r['src_lines']:,}行代码，测试比率仅{r['test_to_code_ratio']:.1f}%\n")
                f.write("\n")
            
            medium_priority = [r for r in results if "中优先级" in r['priority']]
            if medium_priority:
                f.write("### 🟡 中优先级模块\n\n")
                for r in medium_priority:
                    f.write(f"- **{r['module']}**: {r['src_lines']:,}行代码，测试比率{r['test_to_code_ratio']:.1f}%\n")


def main():
    """主函数"""
    assessor = QuickInfrastructureAssessment()
    result = assessor.run_assessment()
    return 0


if __name__ == "__main__":
    exit(main())

