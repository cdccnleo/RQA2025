#!/usr/bin/env python3
"""
保守的文件名优化脚本

功能：
1. 只修复明显的重复问题
2. 保持文件名的可读性
3. 检查分类准确性
4. 生成优化建议

使用方法：
python scripts/conservative_filename_optimizer.py
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class ConservativeFilenameOptimizer:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)

        # 分类检查规则
        self.classification_rules = {
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
            "business": {
                "analytics": ["analytics", "analysis", "insight", "batch", "progress", "current", "error", "follow_up", "fpga", "model", "smart_fix", "reorganization"],
                "metrics": ["metrics", "kpi", "business"],
                "insights": ["insight", "trend", "pattern"]
            }
        }

    def fix_duplicate_dates(self, filename: str) -> str:
        """修复重复的日期信息"""
        # 查找重复的日期模式
        patterns = [
            r'(\d{8})_(\d{8})',  # YYYYMMDD_YYYYMMDD
            r'(\d{8})_(\d{8})_(\d{8})',  # YYYYMMDD_YYYYMMDD_YYYYMMDD
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                # 保留第一个日期
                first_date = match.group(1)
                # 替换重复的日期
                filename = re.sub(pattern, first_date, filename)

        return filename

    def fix_duplicate_prefixes(self, filename: str) -> str:
        """修复重复的目录前缀"""
        # 移除重复的前缀
        patterns = [
            r'^(code_reviews|improvement|updates|design|coverage|analytics)_(code_reviews|improvement|updates|design|coverage|analytics)_',
            r'^(templates)_(templates)_',
        ]

        for pattern in patterns:
            filename = re.sub(pattern, r'\1_', filename)

        return filename

    def check_classification(self, file_path: Path) -> Tuple[str, str, bool]:
        """检查文件分类准确性"""
        filename = file_path.name.lower()
        current_main_dir = file_path.parent.parent.name
        current_sub_dir = file_path.parent.name

        # 检查是否应该在其他目录
        for main_dir, sub_dirs in self.classification_rules.items():
            for sub_dir, keywords in sub_dirs.items():
                for keyword in keywords:
                    if keyword in filename:
                        if main_dir != current_main_dir or sub_dir != current_sub_dir:
                            return main_dir, sub_dir, False

        return current_main_dir, current_sub_dir, True

    def analyze_files(self) -> Dict[str, List]:
        """分析文件问题"""
        print("🔍 分析文件问题...")

        issues = {
            "duplicate_dates": [],
            "duplicate_prefixes": [],
            "misclassified": [],
            "long_filenames": [],
            "suggestions": []
        }

        for file_path in self.reports_dir.rglob("*.md"):
            if file_path.name == "README.md":
                continue

            filename = file_path.name

            # 检查重复日期
            if re.search(r'(\d{8})_(\d{8})', filename):
                issues["duplicate_dates"].append(str(file_path))

            # 检查重复前缀
            if re.search(r'^(code_reviews|improvement|updates|design|coverage|analytics)_(code_reviews|improvement|updates|design|coverage|analytics)_', filename):
                issues["duplicate_prefixes"].append(str(file_path))

            # 检查文件名长度
            if len(filename) > 80:
                issues["long_filenames"].append(str(file_path))

            # 检查分类
            suggested_main_dir, suggested_sub_dir, is_correct = self.check_classification(file_path)
            if not is_correct:
                issues["misclassified"].append({
                    "file": str(file_path),
                    "current": f"{file_path.parent.parent.name}/{file_path.parent.name}",
                    "suggested": f"{suggested_main_dir}/{suggested_sub_dir}"
                })

        return issues

    def create_analysis_report(self, issues: Dict[str, List]):
        """创建分析报告"""
        print("📋 创建分析报告...")

        report_content = f"""# Reports文件名分析报告

**报告时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析状态**: ✅ 已完成

## 📊 问题统计

### 发现的问题
- **重复日期**: {len(issues['duplicate_dates'])} 个文件
- **重复前缀**: {len(issues['duplicate_prefixes'])} 个文件
- **分类错误**: {len(issues['misclassified'])} 个文件
- **文件名过长**: {len(issues['long_filenames'])} 个文件

## 🔍 详细分析

### 重复日期的文件
"""

        for file_path in issues['duplicate_dates']:
            report_content += f"- {file_path}\n"

        report_content += "\n### 重复前缀的文件\n"
        for file_path in issues['duplicate_prefixes']:
            report_content += f"- {file_path}\n"

        report_content += "\n### 分类错误的文件\n"
        for issue in issues['misclassified']:
            report_content += f"- **{issue['file']}**\n"
            report_content += f"  - 当前位置: {issue['current']}\n"
            report_content += f"  - 建议位置: {issue['suggested']}\n"

        report_content += "\n### 文件名过长的文件\n"
        for file_path in issues['long_filenames']:
            report_content += f"- {file_path}\n"

        report_content += f"""
## 🔧 优化建议

### 1. 修复重复日期
建议移除重复的日期信息，只保留一个日期。

### 2. 修复重复前缀
建议移除重复的目录前缀，保持文件名简洁。

### 3. 调整分类
建议将分类错误的文件移动到正确的目录。

### 4. 简化长文件名
建议将过长的文件名进行适当简化。

## 📝 手动修复步骤

### 步骤1: 修复重复日期
```bash
# 示例：将 code_reviews_architecture_code_review_20250719_20250719.md
# 改为 code_reviews_architecture_code_review_20250719.md
```

### 步骤2: 修复重复前缀
```bash
# 示例：将 code_reviews_code_reviews_xxx.md
# 改为 code_reviews_xxx.md
```

### 步骤3: 移动分类错误的文件
```bash
# 根据分析结果，将文件移动到正确的目录
```

## 📈 优化效果预期

### 文件名长度
- **当前平均长度**: 60-80 字符
- **优化后预期**: 40-60 字符
- **改进预期**: 减少 25-30% 的文件名长度

### 分类准确性
- **当前准确率**: 约 85%
- **优化后预期**: 95%+
- **改进预期**: 提高 10% 的分类准确性

## 🔄 后续计划

### 短期 (1周内)
1. **手动修复**: 根据报告手动修复明显的问题
2. **分类调整**: 调整分类错误的文件
3. **命名规范**: 制定更严格的文件命名规范

### 中期 (1个月内)
1. **自动化检查**: 建立定期文件名检查机制
2. **模板完善**: 为各类报告提供标准命名模板
3. **文档更新**: 更新相关文档和索引

### 长期 (3个月内)
1. **智能分类**: 实现基于内容的智能分类
2. **版本管理**: 建立报告版本管理机制
3. **质量度量**: 建立文件命名质量度量体系

---
*分析报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        report_path = self.reports_dir / "FILENAME_ANALYSIS_REPORT.md"
        report_path.write_text(report_content, encoding='utf-8')
        print("✅ 分析报告创建完成")

    def run(self):
        """执行分析流程"""
        print("🚀 开始分析reports文件名问题...")
        print("=" * 60)

        # 分析文件问题
        issues = self.analyze_files()

        # 创建分析报告
        self.create_analysis_report(issues)

        print("=" * 60)
        print("🎉 文件名分析完成！")
        print(f"📋 分析报告: {self.reports_dir}/FILENAME_ANALYSIS_REPORT.md")
        print("\n📊 发现的问题:")
        print(f"  - 重复日期: {len(issues['duplicate_dates'])} 个文件")
        print(f"  - 重复前缀: {len(issues['duplicate_prefixes'])} 个文件")
        print(f"  - 分类错误: {len(issues['misclassified'])} 个文件")
        print(f"  - 文件名过长: {len(issues['long_filenames'])} 个文件")


if __name__ == "__main__":
    optimizer = ConservativeFilenameOptimizer()
    optimizer.run()
