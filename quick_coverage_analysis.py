#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速覆盖率分析工具
基于现有代码和测试文件，快速分析各层级覆盖情况
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import subprocess

class QuickCoverageAnalyzer:
    """快速覆盖率分析器"""
    
    LAYERS = [
        "infrastructure", "core", "data", "distributed", "adapters",
        "async", "automation", "boundary", "features", "gateway",
        "ml", "mobile", "monitoring", "optimization", "resilience",
        "risk", "security", "strategy", "streaming", "trading", "utils"
    ]
    
    def __init__(self):
        self.project_root = Path(os.getcwd())
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "reports" / "coverage_quick"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def count_files_and_lines(self, directory: Path, extensions: List[str] = ['.py']) -> Dict:
        """统计文件数和代码行数"""
        if not directory.exists():
            return {"files": 0, "lines": 0, "functions": 0, "classes": 0}
        
        files = 0
        lines = 0
        functions = 0
        classes = 0
        
        for ext in extensions:
            for file in directory.rglob(f"*{ext}"):
                if '__pycache__' in str(file) or '.backup' in str(file):
                    continue
                files += 1
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines += len(content.split('\n'))
                        functions += content.count('def ')
                        classes += content.count('class ')
                except:
                    pass
        
        return {"files": files, "lines": lines, "functions": functions, "classes": classes}
    
    def find_test_files(self, layer: str) -> List[Path]:
        """查找层级的测试文件"""
        test_files = []
        
        # 搜索所有测试目录
        test_dirs = [
            self.tests_dir / "unit",
            self.tests_dir / "integration",
            self.tests_dir / "functional",
            self.tests_dir / "infrastructure",
            self.tests_dir / "performance",
            self.tests_dir / "e2e"
        ]
        
        for test_dir in test_dirs:
            if not test_dir.exists():
                continue
            
            # 查找包含层级名称的测试文件
            for test_file in test_dir.rglob(f"*{layer}*.py"):
                if test_file.name.startswith("test_") and '__pycache__' not in str(test_file):
                    test_files.append(test_file)
        
        return test_files
    
    def analyze_layer(self, layer: str) -> Dict:
        """分析单个层级"""
        print(f"📊 分析层级: {layer}...")
        
        layer_src = self.src_dir / layer
        if not layer_src.exists():
            return {
                "layer": layer,
                "exists": False,
                "reason": "源代码目录不存在"
            }
        
        # 统计源代码
        src_stats = self.count_files_and_lines(layer_src)
        
        # 查找测试文件
        test_files = self.find_test_files(layer)
        
        # 统计测试代码
        test_stats = {"files": len(test_files), "lines": 0, "functions": 0, "classes": 0}
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    test_stats["lines"] += len(content.split('\n'))
                    test_stats["functions"] += content.count('def test_')
                    test_stats["classes"] += content.count('class Test')
            except:
                pass
        
        # 估算覆盖率（基于测试代码行数与源代码行数比例）
        if src_stats["lines"] > 0:
            # 简化的覆盖率估算：测试行数 / 源代码行数 * 基础系数
            coverage_estimate = min(100, (test_stats["lines"] / src_stats["lines"]) * 60)
        else:
            coverage_estimate = 0
        
        # 基于函数覆盖的估算
        if src_stats["functions"] > 0:
            func_coverage = min(100, (test_stats["functions"] / src_stats["functions"]) * 100)
        else:
            func_coverage = 0
        
        # 综合估算
        final_estimate = (coverage_estimate * 0.4 + func_coverage * 0.6)
        
        result = {
            "layer": layer,
            "exists": True,
            "source": {
                "files": src_stats["files"],
                "lines": src_stats["lines"],
                "functions": src_stats["functions"],
                "classes": src_stats["classes"]
            },
            "tests": {
                "files": test_stats["files"],
                "lines": test_stats["lines"],
                "test_functions": test_stats["functions"],
                "test_classes": test_stats["classes"],
                "test_file_list": [str(f.relative_to(self.project_root)) for f in test_files[:10]]
            },
            "coverage_estimate": round(final_estimate, 2),
            "test_to_code_ratio": round(test_stats["lines"] / src_stats["lines"], 2) if src_stats["lines"] > 0 else 0,
            "function_coverage_estimate": round(func_coverage, 2)
        }
        
        return result
    
    def run_quick_pytest_collection(self) -> Dict:
        """快速运行pytest收集，获取测试统计"""
        print("\n🚀 运行pytest收集...")
        
        try:
            result = subprocess.run(
                ["pytest", "--collect-only", "-q", str(self.tests_dir)],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout + result.stderr
            
            # 解析收集结果
            total_tests = 0
            for line in output.split('\n'):
                if 'test' in line.lower() and 'selected' in line.lower():
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            total_tests = int(part)
                            break
            
            return {"total_tests": total_tests, "status": "success"}
        except Exception as e:
            print(f"⚠️  pytest收集失败: {e}")
            return {"total_tests": 0, "status": "error", "error": str(e)}
    
    def generate_report(self, results: Dict):
        """生成报告"""
        print("\n📊 生成报告...")
        
        # 统计汇总
        total_src_files = sum(r['source']['files'] for r in results if r.get('exists'))
        total_src_lines = sum(r['source']['lines'] for r in results if r.get('exists'))
        total_test_files = sum(r['tests']['files'] for r in results if r.get('exists'))
        total_test_lines = sum(r['tests']['lines'] for r in results if r.get('exists'))
        
        avg_coverage = sum(r['coverage_estimate'] for r in results if r.get('exists')) / len([r for r in results if r.get('exists')])
        
        # 按覆盖率分组
        high_coverage = [r for r in results if r.get('coverage_estimate', 0) >= 80]
        medium_coverage = [r for r in results if 60 <= r.get('coverage_estimate', 0) < 80]
        low_coverage = [r for r in results if 0 < r.get('coverage_estimate', 0) < 60]
        no_tests = [r for r in results if r.get('exists') and r.get('tests', {}).get('files', 0) == 0]
        
        summary = {
            "report_type": "RQA2025 快速覆盖率分析",
            "generated_at": datetime.now().isoformat(),
            "disclaimer": "注意：这是基于代码结构的快速估算，非实际运行覆盖率",
            "overall": {
                "total_layers": len(results),
                "layers_with_code": len([r for r in results if r.get('exists')]),
                "average_coverage_estimate": round(avg_coverage, 2),
                "total_source_files": total_src_files,
                "total_source_lines": total_src_lines,
                "total_test_files": total_test_files,
                "total_test_lines": total_test_lines,
                "test_to_code_ratio": round(total_test_lines / total_src_lines, 2) if total_src_lines > 0 else 0
            },
            "coverage_distribution": {
                "high_coverage_80plus": {
                    "count": len(high_coverage),
                    "layers": [r['layer'] for r in high_coverage]
                },
                "medium_coverage_60_80": {
                    "count": len(medium_coverage),
                    "layers": [r['layer'] for r in medium_coverage]
                },
                "low_coverage_below_60": {
                    "count": len(low_coverage),
                    "layers": [r['layer'] for r in low_coverage]
                },
                "no_tests": {
                    "count": len(no_tests),
                    "layers": [r['layer'] for r in no_tests]
                }
            },
            "layer_details": results
        }
        
        # 保存JSON
        json_file = self.reports_dir / f"quick_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown
        md_file = self.reports_dir / f"quick_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        self._generate_markdown(summary, md_file)
        
        print(f"\n✅ 报告已生成:")
        print(f"   📄 JSON: {json_file}")
        print(f"   📄 Markdown: {md_file}")
        
        return summary
    
    def _generate_markdown(self, summary: Dict, output_file: Path):
        """生成Markdown报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RQA2025 快速覆盖率分析报告\n\n")
            f.write(f"**生成时间**: {summary['generated_at']}  \n")
            f.write(f"**报告类型**: {summary['report_type']}  \n\n")
            f.write(f"> ⚠️ **注意**: {summary['disclaimer']}\n\n")
            
            # 总体统计
            overall = summary['overall']
            f.write("## 📊 总体统计\n\n")
            f.write("| 指标 | 数值 |\n")
            f.write("|------|------|\n")
            f.write(f"| 总层级数 | {overall['total_layers']} |\n")
            f.write(f"| 有代码的层级 | {overall['layers_with_code']} |\n")
            f.write(f"| 平均覆盖率估算 | **{overall['average_coverage_estimate']:.2f}%** |\n")
            f.write(f"| 源代码文件数 | {overall['total_source_files']} |\n")
            f.write(f"| 源代码行数 | {overall['total_source_lines']:,} |\n")
            f.write(f"| 测试文件数 | {overall['total_test_files']} |\n")
            f.write(f"| 测试代码行数 | {overall['total_test_lines']:,} |\n")
            f.write(f"| 测试/代码比 | {overall['test_to_code_ratio']:.2f} |\n\n")
            
            # 覆盖率分布
            dist = summary['coverage_distribution']
            f.write("## 📈 覆盖率分布\n\n")
            f.write(f"- ✅ **高覆盖率 (≥80%)**: {dist['high_coverage_80plus']['count']} 个层级\n")
            if dist['high_coverage_80plus']['layers']:
                f.write(f"  - {', '.join(dist['high_coverage_80plus']['layers'])}\n")
            f.write(f"\n- 🟡 **中等覆盖率 (60-80%)**: {dist['medium_coverage_60_80']['count']} 个层级\n")
            if dist['medium_coverage_60_80']['layers']:
                f.write(f"  - {', '.join(dist['medium_coverage_60_80']['layers'])}\n")
            f.write(f"\n- ❌ **低覆盖率 (<60%)**: {dist['low_coverage_below_60']['count']} 个层级\n")
            if dist['low_coverage_below_60']['layers']:
                f.write(f"  - {', '.join(dist['low_coverage_below_60']['layers'])}\n")
            f.write(f"\n- ⚠️ **无测试**: {dist['no_tests']['count']} 个层级\n")
            if dist['no_tests']['layers']:
                f.write(f"  - {', '.join(dist['no_tests']['layers'])}\n")
            f.write("\n")
            
            # 详细表格
            f.write("## 📋 各层级详情\n\n")
            f.write("| 层级 | 源文件 | 源代码行 | 测试文件 | 测试行 | 覆盖率估算 | 状态 |\n")
            f.write("|------|--------|----------|----------|--------|------------|------|\n")
            
            for layer_data in summary['layer_details']:
                if not layer_data.get('exists'):
                    continue
                
                layer = layer_data['layer']
                src = layer_data['source']
                tests = layer_data['tests']
                cov = layer_data['coverage_estimate']
                
                if cov >= 80:
                    status = "✅"
                elif cov >= 60:
                    status = "🟡"
                else:
                    status = "❌"
                
                f.write(f"| {layer} | {src['files']} | {src['lines']:,} | "
                       f"{tests['files']} | {tests['lines']:,} | "
                       f"{cov:.1f}% | {status} |\n")
            
            f.write("\n")
            
            # 需要改进的层级
            needs_improvement = [r for r in summary['layer_details'] 
                               if r.get('exists') and r.get('coverage_estimate', 0) < 80]
            
            if needs_improvement:
                f.write("## ⚠️ 需要改进的层级\n\n")
                f.write("以下层级的覆盖率估算低于80%，需要增加测试：\n\n")
                
                for layer_data in sorted(needs_improvement, key=lambda x: x.get('coverage_estimate', 0)):
                    layer = layer_data['layer']
                    cov = layer_data['coverage_estimate']
                    tests = layer_data['tests']
                    
                    f.write(f"### {layer} ({cov:.1f}%)\n\n")
                    f.write(f"- 当前测试文件: {tests['files']} 个\n")
                    f.write(f"- 测试函数数: {tests['test_functions']} 个\n")
                    
                    if tests.get('test_file_list'):
                        f.write("- 现有测试文件:\n")
                        for test_file in tests['test_file_list'][:5]:
                            f.write(f"  - `{test_file}`\n")
                    else:
                        f.write("- ⚠️ **无测试文件，需要创建**\n")
                    
                    f.write("\n")
            
            f.write("\n---\n")
            f.write(f"\n**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
    
    def run(self):
        """运行分析"""
        print("="*80)
        print("🚀 RQA2025 快速覆盖率分析")
        print("="*80)
        print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 项目路径: {self.project_root}")
        print(f"📊 分析层级数: {len(self.LAYERS)}")
        print("="*80)
        
        results = []
        for i, layer in enumerate(self.LAYERS, 1):
            print(f"\n[{i}/{len(self.LAYERS)}] ", end='')
            result = self.analyze_layer(layer)
            results.append(result)
        
        # 生成报告
        summary = self.generate_report(results)
        
        # 打印汇总
        self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """打印汇总"""
        print("\n" + "="*80)
        print("📊 分析汇总")
        print("="*80 + "\n")
        
        overall = summary['overall']
        dist = summary['coverage_distribution']
        
        print(f"🎯 平均覆盖率估算: {overall['average_coverage_estimate']:.2f}%")
        print(f"📦 总层级数: {overall['total_layers']}")
        print(f"📝 源代码: {overall['total_source_files']} 文件, {overall['total_source_lines']:,} 行")
        print(f"🧪 测试代码: {overall['total_test_files']} 文件, {overall['total_test_lines']:,} 行")
        print(f"📊 测试/代码比: {overall['test_to_code_ratio']:.2f}")
        print()
        print(f"✅ 高覆盖率(≥80%): {dist['high_coverage_80plus']['count']} 个层级")
        print(f"🟡 中等覆盖率(60-80%): {dist['medium_coverage_60_80']['count']} 个层级")
        print(f"❌ 低覆盖率(<60%): {dist['low_coverage_below_60']['count']} 个层级")
        print(f"⚠️  无测试: {dist['no_tests']['count']} 个层级")
        print()
        
        if overall['average_coverage_estimate'] >= 80:
            print("🎉 " + "="*76)
            print("🎉 平均覆盖率估算已达标！(≥80%)")
            print("🎉 " + "="*76)
        else:
            gap = 80 - overall['average_coverage_estimate']
            print("⚠️  " + "="*76)
            print(f"⚠️  平均覆盖率估算未达标，距离目标还有 {gap:.2f}%")
            print("⚠️  " + "="*76)


def main():
    analyzer = QuickCoverageAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()

