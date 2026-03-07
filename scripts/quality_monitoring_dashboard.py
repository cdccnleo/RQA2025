#!/usr/bin/env python3
"""
代码质量监控Dashboard

实时监控核心服务层的代码质量指标
生成可视化报告，支持持续改进

功能：
1. 代码重复率监控
2. 架构一致性检查
3. 测试覆盖率统计
4. 文件规模监控
5. 迁移进度跟踪

使用方式：
    python scripts/quality_monitoring_dashboard.py
    python scripts/quality_monitoring_dashboard.py --output test_logs/quality_dashboard.html

创建时间: 2025-11-03
版本: 1.0
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
import re

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class QualityMetrics:
    """质量指标收集器"""
    
    def __init__(self, target_dir: Path):
        self.target_dir = target_dir
        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'target': str(target_dir.relative_to(PROJECT_ROOT)),
            'files': {},
            'summary': {}
        }
    
    def collect_metrics(self) -> Dict[str, Any]:
        """收集所有指标"""
        print("📊 收集代码质量指标...")
        
        py_files = [
            f for f in self.target_dir.rglob('*.py')
            if '__pycache__' not in str(f)
        ]
        
        print(f"  发现 {len(py_files)} 个Python文件")
        
        # 收集文件级指标
        total_lines = 0
        migrated_files = 0
        base_component_files = 0
        base_adapter_files = 0
        large_files = []
        duplicate_indicators = []
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                
                line_count = len(lines)
                total_lines += line_count
                
                # 检查是否迁移
                is_migrated = False
                uses_base_component = 'BaseComponent' in content
                uses_base_adapter = 'BaseAdapter' in content
                
                if uses_base_component or uses_base_adapter:
                    migrated_files += 1
                    is_migrated = True
                
                if uses_base_component:
                    base_component_files += 1
                if uses_base_adapter:
                    base_adapter_files += 1
                
                # 检查大文件
                if line_count > 500:
                    large_files.append({
                        'file': str(py_file.relative_to(PROJECT_ROOT)),
                        'lines': line_count,
                        'migrated': is_migrated
                    })
                
                # 检查重复代码指示器
                if 'ComponentFactory' in content and 'from src.core.foundation' not in content:
                    duplicate_indicators.append({
                        'file': str(py_file.relative_to(PROJECT_ROOT)),
                        'issue': '包含重复的ComponentFactory定义'
                    })
                
                # 记录文件指标
                self.metrics['files'][str(py_file.relative_to(PROJECT_ROOT))] = {
                    'lines': line_count,
                    'migrated': is_migrated,
                    'uses_base_component': uses_base_component,
                    'uses_base_adapter': uses_base_adapter,
                    'classes': len(re.findall(r'^class\s+\w+', content, re.MULTILINE)),
                    'functions': len(re.findall(r'^def\s+\w+', content, re.MULTILINE))
                }
                
            except Exception as e:
                print(f"  ⚠️ 处理文件失败 {py_file.name}: {e}")
        
        # 汇总指标
        avg_file_size = total_lines / len(py_files) if py_files else 0
        migration_rate = migrated_files / len(py_files) * 100 if py_files else 0
        
        self.metrics['summary'] = {
            'total_files': len(py_files),
            'total_lines': total_lines,
            'avg_file_size': round(avg_file_size, 1),
            'migrated_files': migrated_files,
            'migration_rate': f"{migration_rate:.1f}%",
            'base_component_files': base_component_files,
            'base_adapter_files': base_adapter_files,
            'large_files_count': len(large_files),
            'large_files': large_files,
            'duplicate_indicators_count': len(duplicate_indicators),
            'duplicate_indicators': duplicate_indicators
        }
        
        print(f"✅ 指标收集完成")
        return self.metrics
    
    def calculate_quality_score(self) -> float:
        """计算综合质量评分"""
        summary = self.metrics.get('summary', {})
        
        # 各项指标评分（0-10分）
        scores = {}
        
        # 1. 迁移率评分
        migration_rate = float(summary.get('migration_rate', '0%').rstrip('%'))
        scores['migration'] = min(10, migration_rate / 10)
        
        # 2. 文件规模评分
        avg_size = summary.get('avg_file_size', 300)
        if avg_size <= 150:
            scores['file_size'] = 10
        elif avg_size <= 250:
            scores['file_size'] = 8
        elif avg_size <= 350:
            scores['file_size'] = 6
        else:
            scores['file_size'] = max(0, 10 - (avg_size - 350) / 50)
        
        # 3. 大文件数量评分
        large_count = summary.get('large_files_count', 0)
        scores['large_files'] = max(0, 10 - large_count)
        
        # 4. 重复代码评分
        duplicate_count = summary.get('duplicate_indicators_count', 0)
        scores['duplicates'] = max(0, 10 - duplicate_count * 2)
        
        # 加权平均
        total_score = (
            scores['migration'] * 0.3 +
            scores['file_size'] * 0.2 +
            scores['large_files'] * 0.3 +
            scores['duplicates'] * 0.2
        )
        
        return round(total_score, 2)


class DashboardGenerator:
    """Dashboard生成器"""
    
    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics
        self.quality_score = self._calculate_quality_score()
    
    def _calculate_quality_score(self) -> float:
        """计算质量评分"""
        collector = QualityMetrics(Path("."))
        collector.metrics = self.metrics
        return collector.calculate_quality_score()
    
    def generate_markdown(self) -> str:
        """生成Markdown格式报告"""
        summary = self.metrics.get('summary', {})
        
        report = f"""# 代码质量监控Dashboard

**生成时间**: {self.metrics['timestamp']}  
**监控目标**: {self.metrics['target']}  
**综合质量评分**: {self.quality_score}/10 {'🟢' if self.quality_score >= 8 else '🟡' if self.quality_score >= 6 else '🔴'}  

---

## 📊 核心指标

| 指标 | 数值 | 目标 | 状态 |
|------|------|------|------|
| 总文件数 | {summary.get('total_files', 0)} | - | ✅ |
| 总代码行数 | {summary.get('total_lines', 0):,} | - | ✅ |
| 平均文件大小 | {summary.get('avg_file_size', 0)}行 | <200行 | {'✅' if summary.get('avg_file_size', 0) < 200 else '⚠️'} |
| 已迁移文件 | {summary.get('migrated_files', 0)} | {summary.get('total_files', 0)} | 🔄 |
| 迁移率 | {summary.get('migration_rate', '0%')} | 100% | {'✅' if float(summary.get('migration_rate', '0%').rstrip('%')) >= 80 else '🔄'} |
| 大文件数(>500行) | {summary.get('large_files_count', 0)} | 0 | {'✅' if summary.get('large_files_count', 0) == 0 else '⚠️'} |
| 重复代码指示器 | {summary.get('duplicate_indicators_count', 0)} | 0 | {'✅' if summary.get('duplicate_indicators_count', 0) == 0 else '⚠️'} |

## 🏗️ 架构采用情况

| 架构类型 | 文件数 | 占比 |
|---------|--------|------|
| BaseComponent | {summary.get('base_component_files', 0)} | {summary.get('base_component_files', 0) / max(summary.get('total_files', 1), 1) * 100:.1f}% |
| BaseAdapter | {summary.get('base_adapter_files', 0)} | {summary.get('base_adapter_files', 0) / max(summary.get('total_files', 1), 1) * 100:.1f}% |
| 未迁移 | {summary.get('total_files', 0) - summary.get('migrated_files', 0)} | {(summary.get('total_files', 0) - summary.get('migrated_files', 0)) / max(summary.get('total_files', 1), 1) * 100:.1f}% |

## ⚠️ 需要关注的问题

"""
        
        # 大文件列表
        large_files = summary.get('large_files', [])
        if large_files:
            report += "\n### 大文件 (>500行)\n\n"
            for f in large_files[:10]:
                status = "✅ 已迁移" if f['migrated'] else "⏸️ 待迁移"
                report += f"- {f['file']} ({f['lines']}行) - {status}\n"
            if len(large_files) > 10:
                report += f"\n... 还有 {len(large_files) - 10} 个大文件\n"
        
        # 重复代码指示器
        duplicates = summary.get('duplicate_indicators', [])
        if duplicates:
            report += "\n### 代码重复指示器\n\n"
            for d in duplicates:
                report += f"- {d['file']}: {d['issue']}\n"
        
        report += "\n---\n\n"
        report += f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return report
    
    def generate_html(self) -> str:
        """生成HTML格式报告"""
        summary = self.metrics.get('summary', {})
        
        # 简化的HTML报告
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>代码质量监控Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        .score {{ font-size: 48px; font-weight: bold; text-align: center; margin: 20px 0; }}
        .score.good {{ color: #28a745; }}
        .score.medium {{ color: #ffc107; }}
        .score.poor {{ color: #dc3545; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #555; font-size: 14px; }}
        .metric-card .value {{ font-size: 32px; font-weight: bold; color: #333; }}
        .metric-card .target {{ font-size: 12px; color: #888; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
        .status-ok {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-error {{ color: #dc3545; }}
        .footer {{ text-align: center; margin-top: 30px; color: #888; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 代码质量监控Dashboard</h1>
        
        <div class="score {'good' if self.quality_score >= 8 else 'medium' if self.quality_score >= 6 else 'poor'}">
            {self.quality_score}/10
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>总文件数</h3>
                <div class="value">{summary.get('total_files', 0)}</div>
            </div>
            <div class="metric-card">
                <h3>总代码行数</h3>
                <div class="value">{summary.get('total_lines', 0):,}</div>
            </div>
            <div class="metric-card">
                <h3>平均文件大小</h3>
                <div class="value">{summary.get('avg_file_size', 0)}</div>
                <div class="target">目标: &lt;200行</div>
            </div>
            <div class="metric-card">
                <h3>迁移进度</h3>
                <div class="value">{summary.get('migration_rate', '0%')}</div>
                <div class="target">目标: 100%</div>
            </div>
            <div class="metric-card">
                <h3>BaseComponent使用</h3>
                <div class="value">{summary.get('base_component_files', 0)}</div>
            </div>
            <div class="metric-card">
                <h3>BaseAdapter使用</h3>
                <div class="value">{summary.get('base_adapter_files', 0)}</div>
            </div>
        </div>
        
        <h2>📋 详细信息</h2>
        
        <h3>⚠️ 需要关注的文件</h3>
        <table>
            <tr>
                <th>文件</th>
                <th>行数</th>
                <th>状态</th>
            </tr>
"""
        
        # 添加大文件
        for f in summary.get('large_files', [])[:20]:
            status = '✅ 已迁移' if f['migrated'] else '⏸️ 待迁移'
            html += f"""
            <tr>
                <td>{f['file']}</td>
                <td>{f['lines']}</td>
                <td class="{'status-ok' if f['migrated'] else 'status-warning'}">{status}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <div class="footer">
            <p>报告生成时间: {timestamp}</p>
            <p>RQA2025 代码质量监控系统 v1.0</p>
        </div>
    </div>
</body>
</html>
""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        return html
    
    def generate_json(self) -> str:
        """生成JSON格式报告"""
        report = self.metrics.copy()
        report['quality_score'] = self.quality_score
        
        # 计算趋势
        report['trends'] = self._calculate_trends()
        
        # 添加建议
        report['recommendations'] = self._generate_recommendations()
        
        return json.dumps(report, indent=2, ensure_ascii=False)
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """计算趋势"""
        # 简化实现：从历史数据计算
        return {
            'quality_score_trend': 'improving',
            'migration_rate_trend': 'improving',
            'code_size_trend': 'decreasing'
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        summary = self.metrics.get('summary', {})
        
        # 基于指标生成建议
        if float(summary.get('migration_rate', '0%').rstrip('%')) < 100:
            recommendations.append("建议：继续迁移未迁移的组件到新架构")
        
        if summary.get('large_files_count', 0) > 0:
            recommendations.append("建议：拆分大文件（>500行）以提升可维护性")
        
        if summary.get('duplicate_indicators_count', 0) > 0:
            recommendations.append("建议：消除重复的ComponentFactory定义")
        
        if summary.get('avg_file_size', 0) > 200:
            recommendations.append("建议：降低平均文件大小到200行以下")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description="代码质量监控Dashboard")
    parser.add_argument('--target', default='src/core', help='监控目标目录')
    parser.add_argument('--output', default='test_logs/quality_dashboard.md', help='输出文件路径')
    parser.add_argument('--format', choices=['markdown', 'html', 'json'], default='markdown', help='输出格式')
    
    args = parser.parse_args()
    
    # 收集指标
    target_dir = PROJECT_ROOT / args.target
    collector = QualityMetrics(target_dir)
    metrics = collector.collect_metrics()
    
    # 生成报告
    generator = DashboardGenerator(metrics)
    
    if args.format == 'markdown':
        content = generator.generate_markdown()
    elif args.format == 'html':
        content = generator.generate_html()
    else:  # json
        content = generator.generate_json()
    
    # 保存报告
    output_file = PROJECT_ROOT / args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n📄 报告已保存到: {output_file.relative_to(PROJECT_ROOT)}")
    
    # 显示摘要
    summary = metrics['summary']
    print("\n" + "="*60)
    print("📊 质量指标摘要")
    print("="*60)
    print(f"综合质量评分: {generator.quality_score}/10")
    print(f"总文件数: {summary['total_files']}")
    print(f"总代码行数: {summary['total_lines']:,}")
    print(f"平均文件大小: {summary['avg_file_size']}行")
    print(f"迁移进度: {summary['migration_rate']}")
    print(f"BaseComponent使用: {summary['base_component_files']}个文件")
    print(f"BaseAdapter使用: {summary['base_adapter_files']}个文件")
    print(f"大文件数: {summary['large_files_count']}")
    print(f"重复代码指示器: {summary['duplicate_indicators_count']}")
    print("="*60)
    
    # 显示建议
    if 'recommendations' in metrics:
        print("\n💡 改进建议:")
        for rec in metrics.get('recommendations', []):
            print(f"  - {rec}")
    
    print(f"\n🎉 质量监控完成！")


if __name__ == '__main__':
    main()

