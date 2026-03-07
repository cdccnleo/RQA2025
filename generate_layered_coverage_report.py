#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQA2025 分层覆盖率报告生成器
按照21个层级和子模块依次生成覆盖率报告
验证覆盖率≥80%，通过率≥98%
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import time

class LayeredCoverageReporter:
    """分层覆盖率报告生成器"""
    
    # 定义21个层级模块
    LAYERS = [
        "infrastructure",  # 基础设施层
        "core",            # 核心层
        "data",            # 数据层
        "distributed",     # 分布式层
        "adapters",        # 适配器层
        "async",           # 异步层
        "automation",      # 自动化层
        "boundary",        # 边界层
        "features",        # 特征层
        "gateway",         # 网关层
        "ml",              # 机器学习层
        "mobile",          # 移动层
        "monitoring",      # 监控层
        "optimization",    # 优化层
        "resilience",      # 弹性层
        "risk",            # 风控层
        "security",        # 安全层
        "strategy",        # 策略层
        "streaming",       # 流处理层
        "trading",         # 交易层
        "utils",           # 工具层
    ]
    
    def __init__(self, project_root: str = None):
        """初始化"""
        self.project_root = Path(project_root or os.getcwd())
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "reports" / "coverage_layered"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def run_tests_for_layer(self, layer: str) -> Tuple[Dict, bool]:
        """为指定层级运行测试并生成覆盖率报告"""
        print(f"\n{'='*80}")
        print(f"🔍 正在测试层级: {layer}")
        print(f"{'='*80}")
        
        layer_src = self.src_dir / layer
        if not layer_src.exists():
            print(f"⚠️  源代码目录不存在: {layer_src}")
            return {
                "layer": layer,
                "status": "skipped",
                "reason": "源代码目录不存在",
                "coverage": 0,
                "tests_total": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "pass_rate": 0
            }, False
        
        # 查找测试文件
        test_patterns = [
            f"tests/unit/test_{layer}*.py",
            f"tests/integration/test_{layer}*.py",
            f"tests/functional/test_{layer}*.py",
            f"tests/infrastructure/test_{layer}*.py",
        ]
        
        # 生成层级专属的覆盖率报告
        coverage_file = self.reports_dir / f".coverage.{layer}"
        html_dir = self.reports_dir / f"html_{layer}"
        json_file = self.reports_dir / f"coverage_{layer}_{self.timestamp}.json"
        
        # 构建pytest命令
        cmd = [
            "pytest",
            "-v",
            "--tb=short",
            f"--cov=src/{layer}",
            f"--cov-report=html:{html_dir}",
            f"--cov-report=json:{json_file}",
            f"--cov-report=term",
            "-n", "auto",  # 并行执行
        ]
        
        # 添加测试路径
        for pattern in test_patterns:
            test_path = self.project_root / pattern.replace("tests/", "")
            if test_path.parent.exists():
                cmd.extend([str(test_path.parent / f"test_{layer}*.py")])
        
        # 如果没有找到特定测试，尝试搜索所有相关测试
        if len(cmd) <= 8:  # 只有基础命令，没有测试文件
            print(f"⚠️  未找到 {layer} 层级的特定测试文件，搜索所有相关测试...")
            # 搜索所有可能包含该层级测试的文件
            all_test_files = []
            for test_dir in ["unit", "integration", "functional", "infrastructure"]:
                test_search_dir = self.tests_dir / test_dir
                if test_search_dir.exists():
                    for test_file in test_search_dir.rglob(f"*{layer}*.py"):
                        if test_file.name.startswith("test_"):
                            all_test_files.append(str(test_file))
            
            if all_test_files:
                cmd.extend(all_test_files)
                print(f"📁 找到 {len(all_test_files)} 个相关测试文件")
            else:
                # 如果还是找不到，运行所有测试但只统计该层级的覆盖率
                print(f"📁 未找到专属测试文件，将运行所有测试并统计 {layer} 层覆盖率")
                cmd.append(str(self.tests_dir))
        
        print(f"🚀 执行命令: {' '.join(cmd[:10])}...")
        
        # 运行测试
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            duration = time.time() - start_time
            
            # 解析测试结果
            output = result.stdout + result.stderr
            
            # 提取测试统计
            tests_total = 0
            tests_passed = 0
            tests_failed = 0
            tests_skipped = 0
            
            for line in output.split('\n'):
                if 'passed' in line.lower() or 'failed' in line.lower():
                    # pytest格式: "=== 123 passed, 4 failed, 2 skipped in 45.67s ==="
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'passed' in part and i > 0:
                            try:
                                tests_passed = int(parts[i-1])
                            except:
                                pass
                        elif 'failed' in part and i > 0:
                            try:
                                tests_failed = int(parts[i-1])
                            except:
                                pass
                        elif 'skipped' in part and i > 0:
                            try:
                                tests_skipped = int(parts[i-1])
                            except:
                                pass
            
            tests_total = tests_passed + tests_failed + tests_skipped
            pass_rate = (tests_passed / tests_total * 100) if tests_total > 0 else 0
            
            # 读取覆盖率JSON文件
            coverage_pct = 0
            coverage_details = {}
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        cov_data = json.load(f)
                        coverage_pct = cov_data.get('totals', {}).get('percent_covered', 0)
                        coverage_details = cov_data.get('files', {})
                except Exception as e:
                    print(f"⚠️  解析覆盖率JSON失败: {e}")
            
            # 状态判断
            status = "success"
            issues = []
            
            if coverage_pct < 80:
                status = "warning"
                issues.append(f"覆盖率 {coverage_pct:.2f}% < 80%")
            
            if pass_rate < 98:
                status = "warning"
                issues.append(f"通过率 {pass_rate:.2f}% < 98%")
            
            if tests_failed > 0:
                status = "warning"
                issues.append(f"{tests_failed} 个测试失败")
            
            layer_result = {
                "layer": layer,
                "status": status,
                "coverage": round(coverage_pct, 2),
                "tests_total": tests_total,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tests_skipped": tests_skipped,
                "pass_rate": round(pass_rate, 2),
                "duration": round(duration, 2),
                "html_report": str(html_dir.relative_to(self.project_root)),
                "json_report": str(json_file.relative_to(self.project_root)),
                "issues": issues,
                "coverage_details": coverage_details,
                "timestamp": datetime.now().isoformat()
            }
            
            # 打印结果
            self._print_layer_result(layer_result)
            
            return layer_result, (status == "success")
            
        except subprocess.TimeoutExpired:
            print(f"❌ 测试超时 (>600秒)")
            return {
                "layer": layer,
                "status": "timeout",
                "reason": "测试执行超时",
                "coverage": 0,
                "tests_total": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "pass_rate": 0,
                "duration": 600
            }, False
        except Exception as e:
            print(f"❌ 执行失败: {e}")
            return {
                "layer": layer,
                "status": "error",
                "reason": str(e),
                "coverage": 0,
                "tests_total": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "pass_rate": 0
            }, False
    
    def _print_layer_result(self, result: Dict):
        """打印层级结果"""
        status_emoji = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "skipped": "⏭️",
            "timeout": "⏱️"
        }
        
        print(f"\n{status_emoji.get(result['status'], '❓')} {result['layer']} 层级测试结果:")
        print(f"  📊 覆盖率: {result['coverage']:.2f}% {'✅' if result['coverage'] >= 80 else '❌'}")
        print(f"  🧪 测试总数: {result['tests_total']}")
        print(f"  ✅ 通过: {result['tests_passed']}")
        print(f"  ❌ 失败: {result['tests_failed']}")
        print(f"  ⏭️  跳过: {result.get('tests_skipped', 0)}")
        print(f"  📈 通过率: {result['pass_rate']:.2f}% {'✅' if result['pass_rate'] >= 98 else '❌'}")
        print(f"  ⏱️  耗时: {result.get('duration', 0):.2f}秒")
        
        if result.get('issues'):
            print(f"  ⚠️  问题:")
            for issue in result['issues']:
                print(f"     - {issue}")
    
    def generate_summary_report(self):
        """生成汇总报告"""
        print(f"\n{'='*80}")
        print("📊 生成汇总报告")
        print(f"{'='*80}\n")
        
        total_layers = len(self.results)
        success_layers = sum(1 for r in self.results.values() if r['status'] == 'success')
        warning_layers = sum(1 for r in self.results.values() if r['status'] == 'warning')
        error_layers = sum(1 for r in self.results.values() if r['status'] in ['error', 'timeout'])
        skipped_layers = sum(1 for r in self.results.values() if r['status'] == 'skipped')
        
        total_coverage = sum(r['coverage'] for r in self.results.values() if r['coverage'] > 0)
        tested_layers = sum(1 for r in self.results.values() if r['coverage'] > 0)
        avg_coverage = total_coverage / tested_layers if tested_layers > 0 else 0
        
        total_tests = sum(r['tests_total'] for r in self.results.values())
        total_passed = sum(r['tests_passed'] for r in self.results.values())
        total_failed = sum(r['tests_failed'] for r in self.results.values())
        total_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # 统计达标情况
        coverage_passed = sum(1 for r in self.results.values() if r['coverage'] >= 80)
        passrate_passed = sum(1 for r in self.results.values() if r['pass_rate'] >= 98)
        
        summary = {
            "report_type": "RQA2025 分层覆盖率报告",
            "generated_at": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "total_layers": total_layers,
            "layers_tested": tested_layers,
            "layers_summary": {
                "success": success_layers,
                "warning": warning_layers,
                "error": error_layers,
                "skipped": skipped_layers
            },
            "coverage_summary": {
                "average": round(avg_coverage, 2),
                "target": 80,
                "passed": coverage_passed,
                "total": tested_layers,
                "pass_percentage": round(coverage_passed / tested_layers * 100, 2) if tested_layers > 0 else 0
            },
            "tests_summary": {
                "total": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "pass_rate": round(total_pass_rate, 2),
                "target_pass_rate": 98,
                "passed_layers": passrate_passed,
                "pass_percentage": round(passrate_passed / tested_layers * 100, 2) if tested_layers > 0 else 0
            },
            "validation": {
                "coverage_target_met": avg_coverage >= 80,
                "passrate_target_met": total_pass_rate >= 98,
                "all_targets_met": avg_coverage >= 80 and total_pass_rate >= 98
            },
            "layer_results": self.results
        }
        
        # 保存JSON报告
        summary_file = self.reports_dir / f"summary_{self.timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        md_file = self.reports_dir / f"summary_{self.timestamp}.md"
        self._generate_markdown_report(summary, md_file)
        
        # 打印汇总
        self._print_summary(summary)
        
        return summary
    
    def _generate_markdown_report(self, summary: Dict, output_file: Path):
        """生成Markdown格式报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RQA2025 分层覆盖率报告\n\n")
            f.write(f"**生成时间**: {summary['generated_at']}  \n")
            f.write(f"**项目路径**: {summary['project_root']}  \n\n")
            
            # 总体结果
            f.write("## 📊 总体结果\n\n")
            f.write("| 指标 | 数值 | 目标 | 状态 |\n")
            f.write("|------|------|------|------|\n")
            
            cov = summary['coverage_summary']
            tests = summary['tests_summary']
            val = summary['validation']
            
            f.write(f"| 平均覆盖率 | **{cov['average']:.2f}%** | {cov['target']}% | {'✅ 达标' if val['coverage_target_met'] else '❌ 未达标'} |\n")
            f.write(f"| 测试通过率 | **{tests['pass_rate']:.2f}%** | {tests['target_pass_rate']}% | {'✅ 达标' if val['passrate_target_met'] else '❌ 未达标'} |\n")
            f.write(f"| 测试总数 | {tests['total']} | - | - |\n")
            f.write(f"| 通过测试 | {tests['passed']} | - | ✅ |\n")
            f.write(f"| 失败测试 | {tests['failed']} | 0 | {'✅' if tests['failed'] == 0 else '❌'} |\n")
            f.write(f"| 测试层级数 | {summary['layers_tested']}/{summary['total_layers']} | 21 | - |\n\n")
            
            # 验证结果
            f.write("## ✅ 验证结果\n\n")
            if val['all_targets_met']:
                f.write("🎉 **所有目标均已达标！**\n\n")
            else:
                f.write("⚠️ **部分目标未达标，需要改进：**\n\n")
                if not val['coverage_target_met']:
                    f.write(f"- ❌ 覆盖率 {cov['average']:.2f}% < 80%\n")
                if not val['passrate_target_met']:
                    f.write(f"- ❌ 通过率 {tests['pass_rate']:.2f}% < 98%\n")
                f.write("\n")
            
            # 层级详情
            f.write("## 📋 各层级详情\n\n")
            f.write("| 层级 | 覆盖率 | 测试数 | 通过 | 失败 | 通过率 | 状态 |\n")
            f.write("|------|--------|--------|------|------|--------|------|\n")
            
            for layer_name in self.LAYERS:
                if layer_name in self.results:
                    r = self.results[layer_name]
                    status_emoji = {
                        "success": "✅",
                        "warning": "⚠️",
                        "error": "❌",
                        "skipped": "⏭️",
                        "timeout": "⏱️"
                    }
                    emoji = status_emoji.get(r['status'], '❓')
                    
                    cov_status = "✅" if r['coverage'] >= 80 else "❌"
                    pass_status = "✅" if r['pass_rate'] >= 98 else "❌"
                    
                    f.write(f"| {layer_name} | {r['coverage']:.1f}% {cov_status} | "
                           f"{r['tests_total']} | {r['tests_passed']} | {r['tests_failed']} | "
                           f"{r['pass_rate']:.1f}% {pass_status} | {emoji} |\n")
            
            f.write("\n")
            
            # 需要改进的层级
            needs_improvement = [
                (name, r) for name, r in self.results.items()
                if r['coverage'] < 80 or r['pass_rate'] < 98 or r['tests_failed'] > 0
            ]
            
            if needs_improvement:
                f.write("## ⚠️ 需要改进的层级\n\n")
                for name, r in needs_improvement:
                    f.write(f"### {name}\n\n")
                    if r['issues']:
                        for issue in r['issues']:
                            f.write(f"- {issue}\n")
                    f.write(f"- 详细报告: [{r.get('html_report', 'N/A')}]({r.get('html_report', '#')})\n\n")
            
            # 报告文件列表
            f.write("## 📁 报告文件\n\n")
            f.write("各层级详细覆盖率报告:\n\n")
            for layer_name in self.LAYERS:
                if layer_name in self.results:
                    r = self.results[layer_name]
                    if r.get('html_report'):
                        f.write(f"- [{layer_name}]({r['html_report']}/index.html)\n")
            
            f.write("\n---\n")
            f.write(f"\n**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**报告生成器**: RQA2025 Layered Coverage Reporter v1.0\n")
    
    def _print_summary(self, summary: Dict):
        """打印汇总信息"""
        print("\n" + "="*80)
        print("📊 总体汇总")
        print("="*80 + "\n")
        
        cov = summary['coverage_summary']
        tests = summary['tests_summary']
        val = summary['validation']
        
        print(f"🎯 平均覆盖率: {cov['average']:.2f}% (目标: {cov['target']}%) {'✅ 达标' if val['coverage_target_met'] else '❌ 未达标'}")
        print(f"📊 覆盖率达标: {cov['passed']}/{cov['total']} 层级 ({cov['pass_percentage']:.1f}%)")
        print()
        print(f"🧪 测试总数: {tests['total']}")
        print(f"✅ 通过: {tests['passed']} ({tests['pass_rate']:.2f}%)")
        print(f"❌ 失败: {tests['failed']}")
        print(f"📈 通过率: {tests['pass_rate']:.2f}% (目标: {tests['target_pass_rate']}%) {'✅ 达标' if val['passrate_target_met'] else '❌ 未达标'}")
        print(f"📊 通过率达标: {tests['passed_layers']}/{summary['layers_tested']} 层级 ({tests['pass_percentage']:.1f}%)")
        print()
        
        layers_summary = summary['layers_summary']
        print(f"📦 层级汇总:")
        print(f"  ✅ 成功: {layers_summary['success']}")
        print(f"  ⚠️  警告: {layers_summary['warning']}")
        print(f"  ❌ 错误: {layers_summary['error']}")
        print(f"  ⏭️  跳过: {layers_summary['skipped']}")
        print()
        
        if val['all_targets_met']:
            print("🎉 " + "="*76)
            print("🎉 恭喜！所有目标均已达标！")
            print("🎉 " + "="*76)
        else:
            print("⚠️  " + "="*76)
            print("⚠️  部分目标未达标，请查看详细报告进行改进")
            print("⚠️  " + "="*76)
        
        print()
        print(f"📁 详细报告已保存至: {self.reports_dir}")
        print()
    
    def run_all(self):
        """运行所有层级的测试"""
        print("="*80)
        print("🚀 RQA2025 分层覆盖率报告生成器")
        print("="*80)
        print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 项目路径: {self.project_root}")
        print(f"📊 总层级数: {len(self.LAYERS)}")
        print("="*80)
        
        start_time = time.time()
        
        # 逐层测试
        for i, layer in enumerate(self.LAYERS, 1):
            print(f"\n进度: [{i}/{len(self.LAYERS)}]")
            result, success = self.run_tests_for_layer(layer)
            self.results[layer] = result
        
        # 生成汇总报告
        summary = self.generate_summary_report()
        
        total_duration = time.time() - start_time
        print(f"\n⏱️  总耗时: {total_duration:.2f}秒 ({total_duration/60:.1f}分钟)")
        print(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return summary


def main():
    """主函数"""
    reporter = LayeredCoverageReporter()
    summary = reporter.run_all()
    
    # 返回状态码
    if summary['validation']['all_targets_met']:
        print("\n✅ 退出状态: 0 (成功)")
        sys.exit(0)
    else:
        print("\n⚠️  退出状态: 1 (部分目标未达标)")
        sys.exit(1)


if __name__ == "__main__":
    main()

