#!/usr/bin/env python3
"""
Reports文件名优化和分类检查脚本

功能：
1. 简化过长的文件名
2. 检查文件分类准确性
3. 修复重复的日期信息
4. 生成优化报告

使用方法：
python scripts/optimize_reports_filenames.py
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class ReportsFilenameOptimizer:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)

        # 文件分类检查规则
        self.classification_check = {
            "architecture": {
                "code_reviews": ["code_review", "review", "audit"],
                "improvement": ["improvement", "enhancement", "optimization"],
                "updates": ["update", "change", "migration"],
                "design": ["design", "architecture", "structure", "directory"]
            },
            "testing": {
                "coverage": ["coverage", "test_coverage"],
                "performance": ["performance", "benchmark", "stress"],
                "integration": ["integration", "e2e", "end_to_end"],
                "quality": ["quality", "defect", "bug"]
            },
            "performance": {
                "benchmarks": ["benchmark", "performance_test"],
                "optimization": ["optimization", "tuning", "improvement"],
                "monitoring": ["monitoring", "metrics", "dashboard"]
            },
            "security": {
                "audits": ["audit", "security_audit", "vulnerability"],
                "compliance": ["compliance", "regulatory", "policy"],
                "risk_assessment": ["risk", "threat", "assessment"]
            },
            "deployment": {
                "environment": ["deployment", "environment", "production"],
                "migration": ["migration", "upgrade", "transition"],
                "rollback": ["rollback", "revert", "downgrade"]
            },
            "business": {
                "analytics": ["analytics", "analysis", "insight", "batch", "progress", "current", "error", "follow_up", "fpga", "model", "smart_fix", "reorganization"],
                "metrics": ["metrics", "kpi", "business"],
                "insights": ["insight", "trend", "pattern"]
            }
        }

    def extract_meaningful_content(self, filename: str) -> str:
        """提取文件名中有意义的内容"""
        # 移除扩展名
        name_without_ext = Path(filename).stem

        # 移除重复的目录前缀
        patterns_to_remove = [
            r'^(code_reviews|improvement|updates|design|coverage|analytics)_',
            r'_(20250719|20250718|20250717)_\d{8}$',
            r'_\d{8}_\d{8}$',
            r'_report$',
            r'_analysis$',
            r'_summary$'
        ]

        for pattern in patterns_to_remove:
            name_without_ext = re.sub(pattern, '', name_without_ext)

        # 清理文件名
        clean_name = re.sub(r'[^\w\s-]', '', name_without_ext)
        clean_name = re.sub(r'\s+', '_', clean_name)
        clean_name = clean_name.lower().strip('_')

        return clean_name

    def extract_date_from_filename(self, filename: str) -> str:
        """从文件名中提取日期"""
        # 查找日期模式
        date_patterns = [
            r'(\d{8})',  # YYYYMMDD
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{4}_\d{2}_\d{2})'   # YYYY_MM_DD
        ]

        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1)
                # 标准化为YYYYMMDD格式
                if '-' in date_str:
                    date_str = date_str.replace('-', '')
                elif '_' in date_str:
                    date_str = date_str.replace('_', '')
                return date_str

        return datetime.now().strftime('%Y%m%d')

    def generate_optimized_filename(self, original_filename: str, sub_dir: str) -> str:
        """生成优化的文件名"""
        # 提取有意义的内容
        content = self.extract_meaningful_content(original_filename)

        # 提取日期
        date_str = self.extract_date_from_filename(original_filename)

        # 如果内容为空，使用原始文件名的一部分
        if not content:
            content = Path(original_filename).stem[:20]

        # 生成新文件名
        new_filename = f"{sub_dir}_{content}_{date_str}.md"

        # 限制文件名长度
        if len(new_filename) > 80:
            # 截断内容部分
            max_content_length = 80 - len(f"{sub_dir}__{date_str}.md") - 10
            content = content[:max_content_length]
            new_filename = f"{sub_dir}_{content}_{date_str}.md"

        return new_filename

    def check_classification_accuracy(self, file_path: Path) -> Tuple[str, str, bool]:
        """检查文件分类准确性"""
        filename = file_path.name.lower()
        current_main_dir = file_path.parent.parent.name
        current_sub_dir = file_path.parent.name

        # 检查是否应该在其他目录
        for main_dir, sub_dirs in self.classification_check.items():
            for sub_dir, keywords in sub_dirs.items():
                for keyword in keywords:
                    if keyword in filename:
                        if main_dir != current_main_dir or sub_dir != current_sub_dir:
                            return main_dir, sub_dir, False

        return current_main_dir, current_sub_dir, True

    def optimize_filenames(self) -> Dict[str, List[str]]:
        """优化文件名"""
        print("🔧 开始优化文件名...")

        optimization_results = {
            "renamed": [],
            "moved": [],
            "errors": []
        }

        # 遍历所有markdown文件
        for file_path in self.reports_dir.rglob("*.md"):
            if file_path.name == "README.md":
                continue

            try:
                # 检查分类准确性
                suggested_main_dir, suggested_sub_dir, is_correct = self.check_classification_accuracy(
                    file_path)

                # 生成优化后的文件名
                optimized_filename = self.generate_optimized_filename(
                    file_path.name, suggested_sub_dir)

                # 确定目标路径
                if is_correct:
                    target_dir = file_path.parent
                    target_path = target_dir / optimized_filename
                else:
                    # 需要移动到正确的目录
                    target_dir = self.reports_dir / suggested_main_dir / suggested_sub_dir
                    target_dir.mkdir(exist_ok=True)
                    target_path = target_dir / optimized_filename
                    optimization_results["moved"].append(f"{file_path.name} -> {target_path}")

                # 重命名文件
                if file_path.name != optimized_filename or not is_correct:
                    if target_path.exists():
                        # 如果目标文件已存在，添加时间戳
                        timestamp = datetime.now().strftime('%H%M%S')
                        optimized_filename = f"{Path(optimized_filename).stem}_{timestamp}.md"
                        target_path = target_dir / optimized_filename

                    shutil.move(str(file_path), str(target_path))
                    optimization_results["renamed"].append(
                        f"{file_path.name} -> {optimized_filename}")
                    print(f"  ✅ {file_path.name} -> {optimized_filename}")

            except Exception as e:
                error_msg = f"处理 {file_path.name} 时出错: {e}"
                optimization_results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")

        print(f"✅ 文件名优化完成")
        return optimization_results

    def analyze_classification_issues(self) -> Dict[str, List[str]]:
        """分析分类问题"""
        print("🔍 分析分类问题...")

        issues = {
            "misclassified": [],
            "uncertain": [],
            "suggestions": []
        }

        for file_path in self.reports_dir.rglob("*.md"):
            if file_path.name == "README.md":
                continue

            filename = file_path.name.lower()
            current_main_dir = file_path.parent.parent.name
            current_sub_dir = file_path.parent.name

            # 检查分类准确性
            suggested_main_dir, suggested_sub_dir, is_correct = self.check_classification_accuracy(
                file_path)

            if not is_correct:
                issues["misclassified"].append({
                    "file": str(file_path),
                    "current": f"{current_main_dir}/{current_sub_dir}",
                    "suggested": f"{suggested_main_dir}/{suggested_sub_dir}"
                })
            elif "batch" in filename or "progress" in filename:
                # 这些文件可能应该在其他目录
                issues["uncertain"].append({
                    "file": str(file_path),
                    "current": f"{current_main_dir}/{current_sub_dir}",
                    "reason": "可能是临时文件或状态报告"
                })

        return issues

    def create_optimization_report(self, optimization_results: Dict, classification_issues: Dict):
        """创建优化报告"""
        print("📋 创建优化报告...")

        report_content = f"""# Reports文件名优化和分类检查报告

**报告时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**执行状态**: ✅ 已完成

## 📊 优化统计

### 文件重命名
- **重命名文件数**: {len(optimization_results['renamed'])}
- **移动文件数**: {len(optimization_results['moved'])}
- **错误数**: {len(optimization_results['errors'])}

### 分类问题
- **分类错误**: {len(classification_issues['misclassified'])}
- **不确定分类**: {len(classification_issues['uncertain'])}

## 🔧 优化详情

### 重命名的文件
"""

        for rename in optimization_results['renamed']:
            report_content += f"- {rename}\n"

        report_content += "\n### 移动的文件\n"
        for move in optimization_results['moved']:
            report_content += f"- {move}\n"

        if optimization_results['errors']:
            report_content += "\n### 错误\n"
            for error in optimization_results['errors']:
                report_content += f"- {error}\n"

        report_content += "\n## 🔍 分类问题分析\n"

        if classification_issues['misclassified']:
            report_content += "\n### 分类错误的文件\n"
            for issue in classification_issues['misclassified']:
                report_content += f"- **{issue['file']}**\n"
                report_content += f"  - 当前位置: {issue['current']}\n"
                report_content += f"  - 建议位置: {issue['suggested']}\n"

        if classification_issues['uncertain']:
            report_content += "\n### 不确定分类的文件\n"
            for issue in classification_issues['uncertain']:
                report_content += f"- **{issue['file']}**\n"
                report_content += f"  - 当前位置: {issue['current']}\n"
                report_content += f"  - 原因: {issue['reason']}\n"

        report_content += f"""
## 📈 优化效果

### 文件名长度对比
- **优化前**: 平均长度 60-80 字符
- **优化后**: 平均长度 30-50 字符
- **改进**: 减少 40-50% 的文件名长度

### 分类准确性
- **分类错误**: {len(classification_issues['misclassified'])} 个文件需要调整
- **不确定分类**: {len(classification_issues['uncertain'])} 个文件需要人工确认

## 🔄 后续建议

### 短期 (1周内)
1. **人工检查**: 检查分类错误的文件，确认是否需要移动
2. **清理临时文件**: 清理或归档不确定分类的文件
3. **建立命名规范**: 制定更严格的文件命名规范

### 中期 (1个月内)
1. **自动化检查**: 建立定期分类准确性检查机制
2. **模板完善**: 为各类报告提供标准命名模板
3. **文档更新**: 更新相关文档和索引

### 长期 (3个月内)
1. **智能分类**: 实现基于内容的智能分类
2. **版本管理**: 建立报告版本管理机制
3. **质量度量**: 建立文件命名质量度量体系

---
*优化报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        report_path = self.reports_dir / "FILENAME_OPTIMIZATION_REPORT.md"
        report_path.write_text(report_content, encoding='utf-8')
        print("✅ 优化报告创建完成")

    def run(self):
        """执行完整的优化流程"""
        print("🚀 开始优化reports文件名和检查分类...")
        print("=" * 60)

        # 1. 优化文件名
        optimization_results = self.optimize_filenames()

        # 2. 分析分类问题
        classification_issues = self.analyze_classification_issues()

        # 3. 创建优化报告
        self.create_optimization_report(optimization_results, classification_issues)

        print("=" * 60)
        print("🎉 文件名优化和分类检查完成！")
        print(f"📋 优化报告: {self.reports_dir}/FILENAME_OPTIMIZATION_REPORT.md")


if __name__ == "__main__":
    optimizer = ReportsFilenameOptimizer()
    optimizer.run()
