#!/usr/bin/env python3
"""
文件优化分析器

分析和优化文件数量过多的问题
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import hashlib


class FileOptimizationAnalyzer:
    """文件优化分析器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def analyze_layer_file_count(self, layer_path: str) -> Dict[str, Any]:
        """分析指定层的文件数量"""

        layer_dir = self.src_dir / layer_path
        if not layer_dir.exists():
            return {"error": f"目录不存在: {layer_dir}"}

        analysis = {
            "layer": layer_path,
            "total_files": 0,
            "python_files": 0,
            "other_files": 0,
            "sub_directories": {},
            "file_types": {},
            "empty_files": 0,
            "duplicate_files": {},
            "large_files": [],
            "optimization_suggestions": []
        }

        # 递归分析目录
        for root, dirs, files in os.walk(layer_dir):
            rel_root = Path(root).relative_to(self.src_dir)

            # 统计子目录
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                rel_path = dir_path.relative_to(self.src_dir)

                if str(rel_path).startswith(layer_path):
                    sub_analysis = self._analyze_directory(dir_path)
                    analysis['sub_directories'][str(rel_path)] = sub_analysis
                    analysis['total_files'] += sub_analysis['total_files']
                    analysis['python_files'] += sub_analysis['python_files']

        # 分析文件类型分布
        analysis['file_types'] = self._analyze_file_types(layer_dir)

        # 检查重复文件
        analysis['duplicate_files'] = self._find_duplicate_files(layer_dir)

        # 检查大文件
        analysis['large_files'] = self._find_large_files(layer_dir)

        # 检查空文件
        analysis['empty_files'] = self._count_empty_files(layer_dir)

        # 生成优化建议
        analysis['optimization_suggestions'] = self._generate_optimization_suggestions(
            analysis, layer_path)

        return analysis

    def _analyze_directory(self, directory: Path) -> Dict[str, Any]:
        """分析单个目录"""
        analysis = {
            "total_files": 0,
            "python_files": 0,
            "other_files": 0,
            "directories": 0
        }

        for item in directory.iterdir():
            if item.is_file():
                analysis['total_files'] += 1
                if item.suffix == '.py':
                    analysis['python_files'] += 1
                else:
                    analysis['other_files'] += 1
            elif item.is_dir():
                analysis['directories'] += 1

        return analysis

    def _analyze_file_types(self, directory: Path) -> Dict[str, int]:
        """分析文件类型分布"""
        file_types = {}

        for root, dirs, files in os.walk(directory):
            for file in files:
                suffix = Path(file).suffix
                file_types[suffix] = file_types.get(suffix, 0) + 1

        return file_types

    def _find_duplicate_files(self, directory: Path) -> Dict[str, List[str]]:
        """查找重复文件"""
        file_hashes = {}
        duplicates = {}

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            file_hash = hashlib.md5(content).hexdigest()

                        if file_hash in file_hashes:
                            if file_hash not in duplicates:
                                duplicates[file_hash] = [file_hashes[file_hash]]
                            duplicates[file_hash].append(str(file_path.relative_to(self.src_dir)))
                        else:
                            file_hashes[file_hash] = str(file_path.relative_to(self.src_dir))
                    except:
                        continue

        # 过滤出真正重复的文件（至少2个）
        return {k: v for k, v in duplicates.items() if len(v) >= 2}

    def _find_large_files(self, directory: Path, threshold_kb: int = 100) -> List[Dict[str, Any]]:
        """查找大文件"""
        large_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                try:
                    size_kb = file_path.stat().st_size / 1024
                    if size_kb > threshold_kb:
                        large_files.append({
                            "path": str(file_path.relative_to(self.src_dir)),
                            "size_kb": round(size_kb, 2)
                        })
                except:
                    continue

        return sorted(large_files, key=lambda x: x["size_kb"], reverse=True)

    def _count_empty_files(self, directory: Path) -> int:
        """统计空文件数量"""
        count = 0

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if not content:
                                count += 1
                    except:
                        continue

        return count

    def _generate_optimization_suggestions(self, analysis: Dict[str, Any], layer_path: str) -> List[str]:
        """生成优化建议"""
        suggestions = []

        # 文件数量过多建议
        expected_counts = {
            "infrastructure": 700,
            "features": 300,
            "trading": 150
        }

        current_count = analysis['python_files']
        expected = expected_counts.get(layer_path, 0)

        if expected > 0 and current_count > expected:
            excess = current_count - expected
            suggestions.append(f"文件数量过多: {current_count} > {expected}，建议减少{excess}个文件")

        # 重复文件建议
        if analysis['duplicate_files']:
            suggestions.append(f"发现{len(analysis['duplicate_files'])}组重复文件，建议合并或删除重复内容")

        # 大文件建议
        if analysis['large_files']:
            suggestions.append(f"发现{len(analysis['large_files'])}个大文件，建议拆分或优化")

        # 空文件建议
        if analysis['empty_files'] > 0:
            suggestions.append(f"发现{analysis['empty_files']}个空文件，建议删除")

        # 文件类型建议
        file_types = analysis['file_types']
        if file_types.get('.pyc', 0) > 0:
            suggestions.append("发现.pyc文件，建议清理编译文件")

        return suggestions

    def generate_optimization_plan(self, layer_path: str) -> Dict[str, Any]:
        """生成优化计划"""
        analysis = self.analyze_layer_file_count(layer_path)

        plan = {
            "layer": layer_path,
            "analysis_date": datetime.now().isoformat(),
            "current_status": {
                "total_files": analysis['python_files'],
                "sub_directories": len(analysis['sub_directories']),
                "duplicate_groups": len(analysis['duplicate_files']),
                "large_files": len(analysis['large_files']),
                "empty_files": analysis['empty_files']
            },
            "issues": analysis['optimization_suggestions'],
            "optimization_steps": self._generate_optimization_steps(analysis),
            "expected_improvement": self._calculate_expected_improvement(analysis, layer_path)
        }

        return plan

    def _generate_optimization_steps(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成优化步骤"""
        steps = []

        # 清理空文件
        if analysis['empty_files'] > 0:
            steps.append({
                "step": "清理空文件",
                "priority": "high",
                "description": f"删除{analysis['empty_files']}个空文件",
                "estimated_time": "10分钟"
            })

        # 合并重复文件
        if analysis['duplicate_files']:
            steps.append({
                "step": "合并重复文件",
                "priority": "medium",
                "description": f"分析并合并{len(analysis['duplicate_files'])}组重复文件",
                "estimated_time": "30分钟"
            })

        # 拆分大文件
        if analysis['large_files']:
            steps.append({
                "step": "拆分大文件",
                "priority": "medium",
                "description": f"拆分{len(analysis['large_files'])}个大文件",
                "estimated_time": "1小时"
            })

        # 重新组织目录结构
        if len(analysis['sub_directories']) > 8:
            steps.append({
                "step": "优化目录结构",
                "priority": "low",
                "description": "重新组织和合并子目录",
                "estimated_time": "2小时"
            })

        return steps

    def _calculate_expected_improvement(self, analysis: Dict[str, Any], layer_path: str) -> Dict[str, Any]:
        """计算预期改善"""
        current_count = analysis['python_files']
        expected_counts = {
            "infrastructure": 700,
            "features": 300,
            "trading": 150
        }
        expected = expected_counts.get(layer_path, current_count)

        improvement = {
            "target_file_count": expected,
            "estimated_reduction": max(0, current_count - expected),
            "estimated_completion_time": "1-2周",
            "difficulty": "medium"
        }

        return improvement

    def generate_report(self, layer_path: str) -> str:
        """生成优化报告"""
        plan = self.generate_optimization_plan(layer_path)

        report = f"""# 📊 {plan['layer']}层文件优化分析报告

## 📅 生成时间
{plan['analysis_date']}

## 📈 当前状态
- **Python文件总数**: {plan['current_status']['total_files']}
- **子目录数量**: {plan['current_status']['sub_directories']}
- **重复文件组数**: {plan['current_status']['duplicate_groups']}
- **大文件数量**: {plan['current_status']['large_files']}
- **空文件数量**: {plan['current_status']['empty_files']}

## 🚨 发现的问题
"""

        for issue in plan['issues']:
            report += f"- {issue}\n"

        report += f"""
## 🛠️ 优化步骤
"""

        for i, step in enumerate(plan['optimization_steps'], 1):
            report += f"""### {i}. {step['step']} ({step['priority']}优先级)
- **描述**: {step['description']}
- **预计时间**: {step['estimated_time']}

"""

        report += f"""
## 🎯 预期改善
- **目标文件数**: {plan['expected_improvement']['target_file_count']}
- **预计减少文件**: {plan['expected_improvement']['estimated_reduction']}
- **预计完成时间**: {plan['expected_improvement']['estimated_completion_time']}
- **难度评估**: {plan['expected_improvement']['difficulty']}

## 💡 优化建议
1. 优先处理高优先级任务（清理空文件）
2. 逐步进行文件合并和拆分
3. 保持代码质量和可读性
4. 定期备份和版本控制
"""

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='文件优化分析器')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--layer', required=True, help='要分析的层（infrastructure/features/trading）')
    parser.add_argument('--report', action='store_true', help='生成详细报告')

    args = parser.parse_args()

    analyzer = FileOptimizationAnalyzer(args.project)

    if args.report:
        report_content = analyzer.generate_report(args.layer)
        report_file = analyzer.reports_dir / \
            f"file_optimization_report_{args.layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"📊 优化报告已保存: {report_file}")
    else:
        plan = analyzer.generate_optimization_plan(args.layer)
        print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
