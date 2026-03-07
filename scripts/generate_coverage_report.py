#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动化覆盖率报告生成脚本

生成详细的覆盖率报告，包括：
- 总体覆盖率统计
- 分层级覆盖率分析
- 趋势分析
- 质量评估报告
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import subprocess


class CoverageReportGenerator:
    """覆盖率报告生成器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "test_logs"
        self.reports_dir.mkdir(exist_ok=True)

    def run_coverage_tests(self) -> bool:
        """运行覆盖率测试"""
        print("🔍 运行覆盖率测试...")

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            "--cov=src",
            "--cov-report=xml",
            "--cov-report=html",
            "--cov-report=json",
            "--cov-report=term-missing",
            "-x", "--tb=short", "-q"
        ]

        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True, timeout=1800)
            if result.returncode == 0:
                print("✅ 覆盖率测试执行成功")
                return True
            else:
                print(f"❌ 覆盖率测试失败: {result.returncode}")
                print("STDOUT:", result.stdout[-500:])
                print("STDERR:", result.stderr[-500:])
                return False
        except subprocess.TimeoutExpired:
            print("❌ 覆盖率测试超时")
            return False
        except Exception as e:
            print(f"❌ 覆盖率测试执行异常: {e}")
            return False

    def parse_coverage_data(self) -> Dict[str, Any]:
        """解析覆盖率数据"""
        coverage_file = self.project_root / "coverage.json"
        if not coverage_file.exists():
            print("⚠️  coverage.json文件不存在")
            return {}

        try:
            with open(coverage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ 解析覆盖率数据失败: {e}")
            return {}

    def analyze_layer_coverage(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析分层级覆盖率"""
        layers = {
            'infrastructure': ['infrastructure'],
            'core': ['core'],
            'data': ['data'],
            'features': ['feature', 'features'],
            'ml': ['ml'],
            'strategy': ['strategy'],
            'trading': ['trading'],
            'risk': ['risk'],
            'monitoring': ['monitoring'],
            'streaming': ['streaming'],
            'gateway': ['gateway'],
            'optimization': ['optimization'],
            'adapter': ['adapter'],
            'automation': ['automation'],
            'resilience': ['resilience'],
            'testing': ['testing'],
            'utils': ['utils'],
            'distributed': ['distributed'],
            'async_processor': ['async_processor', 'async'],
            'mobile': ['mobile'],
            'boundary': ['boundary']
        }

        layer_stats = {}

        for layer_name, keywords in layers.items():
            layer_files = []
            total_lines = 0
            covered_lines = 0

            for file_path, file_data in coverage_data.get('files', {}).items():
                if any(keyword in file_path.lower() for keyword in keywords):
                    layer_files.append(file_path)
                    summary = file_data.get('summary', {})
                    total_lines += summary.get('num_statements', 0)
                    covered_lines += summary.get('covered_statements', 0)

            if total_lines > 0:
                coverage_percent = (covered_lines / total_lines) * 100
            else:
                coverage_percent = 0.0

            layer_stats[layer_name] = {
                'files': len(layer_files),
                'total_lines': total_lines,
                'covered_lines': covered_lines,
                'coverage_percent': round(coverage_percent, 2),
                'file_list': layer_files[:10]  # 只显示前10个文件
            }

        return layer_stats

    def generate_quality_assessment(self, layer_stats: Dict[str, Any]) -> Dict[str, Any]:
        """生成质量评估报告"""
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'overall_coverage': 0.0,
            'layers_above_30': 0,
            'layers_above_50': 0,
            'layers_above_80': 0,
            'total_layers': len(layer_stats),
            'quality_score': 0,
            'recommendations': []
        }

        total_weighted_coverage = 0
        total_weight = 0

        for layer_name, stats in layer_stats.items():
            coverage = stats['coverage_percent']
            weight = stats['total_lines']  # 按代码行数加权

            total_weighted_coverage += coverage * weight
            total_weight += weight

            if coverage >= 80:
                assessment['layers_above_80'] += 1
            elif coverage >= 50:
                assessment['layers_above_50'] += 1
            elif coverage >= 30:
                assessment['layers_above_30'] += 1

        if total_weight > 0:
            assessment['overall_coverage'] = round(total_weighted_coverage / total_weight, 2)

        # 计算质量评分 (0-100)
        coverage_score = min(assessment['overall_coverage'], 100)
        consistency_score = (assessment['layers_above_30'] / assessment['total_layers']) * 100
        assessment['quality_score'] = round((coverage_score * 0.7 + consistency_score * 0.3), 1)

        # 生成建议
        if assessment['overall_coverage'] < 30:
            assessment['recommendations'].append("⚠️ 整体覆盖率低于30%，建议优先提升核心业务层级的测试覆盖")
        elif assessment['overall_coverage'] < 50:
            assessment['recommendations'].append("📈 整体覆盖率在30-50%区间，建议继续完善测试用例")
        else:
            assessment['recommendations'].append("✅ 整体覆盖率良好，建议维护现有覆盖率水平")

        if assessment['layers_above_80'] < assessment['total_layers'] * 0.5:
            assessment['recommendations'].append("🔧 超过一半的层级覆盖率低于80%，建议重点关注低覆盖率层级")

        return assessment

    def generate_markdown_report(self, layer_stats: Dict[str, Any], assessment: Dict[str, Any]) -> str:
        """生成Markdown格式的报告"""
        report = f"""# 📊 RQA2025 测试覆盖率报告

**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

## 🎯 总体概览

- **整体覆盖率**: {assessment['overall_coverage']}%
- **质量评分**: {assessment['quality_score']}/100
- **达标层级** (≥30%): {assessment['layers_above_30']}/{assessment['total_layers']}
- **优良层级** (≥50%): {assessment['layers_above_50']}/{assessment['total_layers']}
- **优秀层级** (≥80%): {assessment['layers_above_80']}/{assessment['total_layers']}

## 📋 分层级覆盖率详情

| 层级 | 覆盖率 | 文件数 | 总行数 | 覆盖行数 | 状态 |
|------|--------|--------|--------|----------|------|
"""

        for layer_name, stats in sorted(layer_stats.items(), key=lambda x: x[1]['coverage_percent'], reverse=True):
            status = "🟢 优秀" if stats['coverage_percent'] >= 80 else \
                    "🟡 良好" if stats['coverage_percent'] >= 50 else \
                    "🟠 基础" if stats['coverage_percent'] >= 30 else \
                    "🔴 待提升"
            report += f"| {layer_name} | {stats['coverage_percent']}% | {stats['files']} | {stats['total_lines']} | {stats['covered_lines']} | {status} |\n"

        report += "\n## 💡 质量评估与建议\n\n"

        for rec in assessment['recommendations']:
            report += f"- {rec}\n"

        report += f"""
## 📈 覆盖率趋势分析

- **当前状态**: 整体覆盖率{assessment['overall_coverage']}%，质量评分{assessment['quality_score']}/100
- **达标情况**: {assessment['layers_above_30']}/{assessment['total_layers']}层级达到投产标准
- **优化空间**: {'有' if assessment['layers_above_80'] < assessment['total_layers'] else '无'}进一步优化空间

## 🎯 投产建议

"""

        if assessment['overall_coverage'] >= 30 and assessment['layers_above_30'] >= assessment['total_layers'] * 0.8:
            report += "✅ **推荐投产**: 项目已达到高质量投产标准，可以安全上线\n"
        else:
            report += "⚠️ **建议完善**: 建议继续提升测试覆盖率后再考虑投产\n"

        report += "\n---\n\n*此报告由自动化脚本生成*"

        return report

    def save_reports(self, layer_stats: Dict[str, Any], assessment: Dict[str, Any]):
        """保存报告文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存详细数据
        detailed_report = {
            'timestamp': timestamp,
            'assessment': assessment,
            'layer_stats': layer_stats
        }

        json_file = self.reports_dir / f"coverage_detailed_report_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)

        # 保存Markdown报告
        markdown_report = self.generate_markdown_report(layer_stats, assessment)
        md_file = self.reports_dir / f"coverage_report_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(markdown_report)

        # 复制HTML报告
        htmlcov_dir = self.project_root / "htmlcov"
        if htmlcov_dir.exists():
            html_report_dir = self.reports_dir / f"htmlcov_{timestamp}"
            shutil.copytree(htmlcov_dir, html_report_dir)

        print(f"📄 详细报告已保存: {json_file}")
        print(f"📄 Markdown报告已保存: {md_file}")
        if htmlcov_dir.exists():
            print(f"📄 HTML报告已保存: {html_report_dir}")

    def generate_badge(self, assessment: Dict[str, Any]):
        """生成覆盖率徽章"""
        coverage = assessment['overall_coverage']
        color = "brightgreen" if coverage >= 80 else \
               "green" if coverage >= 60 else \
               "yellow" if coverage >= 40 else \
               "orange" if coverage >= 20 else \
               "red"

        badge_url = f"https://img.shields.io/badge/coverage-{coverage}%25-{color}"
        badge_file = self.project_root / "coverage_badge.svg"

        # 下载徽章 (简化版本，直接生成文本徽章)
        badge_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="120" height="20">
  <rect width="50" height="20" fill="#555"/>
  <rect x="50" width="70" height="20" fill="#{color}"/>
  <text x="25" y="14" font-family="Arial" font-size="11" fill="white" text-anchor="middle">coverage</text>
  <text x="85" y="14" font-family="Arial" font-size="11" fill="white" text-anchor="middle">{coverage}%</text>
</svg>"""

        with open(badge_file, 'w', encoding='utf-8') as f:
            f.write(badge_content)

        print(f"🏷️  覆盖率徽章已生成: {badge_file}")

    def run(self):
        """执行完整的报告生成流程"""
        print("🚀 开始生成覆盖率报告...")

        # 1. 运行覆盖率测试
        if not self.run_coverage_tests():
            print("❌ 测试执行失败，无法生成报告")
            return

        # 2. 解析覆盖率数据
        coverage_data = self.parse_coverage_data()
        if not coverage_data:
            print("❌ 无法解析覆盖率数据")
            return

        # 3. 分析分层级覆盖率
        layer_stats = self.analyze_layer_coverage(coverage_data)
        print(f"📊 分析了 {len(layer_stats)} 个架构层级")

        # 4. 生成质量评估
        assessment = self.generate_quality_assessment(layer_stats)
        print(".1f")        # 5. 保存报告
        self.save_reports(layer_stats, assessment)

        # 6. 生成徽章
        self.generate_badge(assessment)

        print("✅ 覆盖率报告生成完成！")
        print(f"📈 整体覆盖率: {assessment['overall_coverage']}%")
        print(f"🏆 质量评分: {assessment['quality_score']}/100")


def main():
    """主入口"""
    project_root = Path(__file__).resolve().parent.parent
    generator = CoverageReportGenerator(project_root)
    generator.run()


if __name__ == "__main__":
    main()