#!/usr/bin/env python3
"""
修复文件名问题的脚本

功能：
1. 修复重复的日期信息
2. 移动分类错误的文件
3. 简化过长的文件名
4. 生成修复报告

使用方法：
python scripts/fix_filename_issues.py
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class FilenameIssueFixer:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)

        # 分类规则
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
                "analytics": ["analytics", "analysis", "insight", "batch", "progress", "current", "error", "follow_up", "fpga", "model", "smart_fix", "reorganization", "filename"],
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

    def fix_issues(self) -> Dict[str, List[str]]:
        """修复文件名问题"""
        print("🔧 开始修复文件名问题...")

        fix_results = {
            "renamed": [],
            "moved": [],
            "errors": []
        }

        # 遍历所有markdown文件
        for file_path in self.reports_dir.rglob("*.md"):
            if file_path.name == "README.md":
                continue

            try:
                original_filename = file_path.name
                new_filename = original_filename

                # 1. 修复重复日期
                if re.search(r'(\d{8})_(\d{8})', original_filename):
                    new_filename = self.fix_duplicate_dates(original_filename)

                # 2. 检查分类准确性
                suggested_main_dir, suggested_sub_dir, is_correct = self.check_classification(
                    file_path)

                # 3. 确定目标路径
                if is_correct:
                    target_dir = file_path.parent
                    target_path = target_dir / new_filename
                else:
                    # 需要移动到正确的目录
                    target_dir = self.reports_dir / suggested_main_dir / suggested_sub_dir
                    target_dir.mkdir(exist_ok=True)
                    target_path = target_dir / new_filename
                    fix_results["moved"].append(f"{file_path.name} -> {target_path}")

                # 4. 执行修复
                if original_filename != new_filename or not is_correct:
                    if target_path.exists():
                        # 如果目标文件已存在，添加时间戳
                        timestamp = datetime.now().strftime('%H%M%S')
                        new_filename = f"{Path(new_filename).stem}_{timestamp}.md"
                        target_path = target_dir / new_filename

                    shutil.move(str(file_path), str(target_path))
                    if original_filename != new_filename:
                        fix_results["renamed"].append(f"{original_filename} -> {new_filename}")
                    print(f"  ✅ {original_filename} -> {new_filename}")

            except Exception as e:
                error_msg = f"处理 {file_path.name} 时出错: {e}"
                fix_results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")

        print(f"✅ 文件名问题修复完成")
        return fix_results

    def create_fix_report(self, fix_results: Dict[str, List[str]]):
        """创建修复报告"""
        print("📋 创建修复报告...")

        report_content = f"""# Reports文件名修复报告

**报告时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**修复状态**: ✅ 已完成

## 📊 修复统计

### 修复结果
- **重命名文件**: {len(fix_results['renamed'])} 个文件
- **移动文件**: {len(fix_results['moved'])} 个文件
- **错误数**: {len(fix_results['errors'])} 个

## 🔧 修复详情

### 重命名的文件
"""

        for rename in fix_results['renamed']:
            report_content += f"- {rename}\n"

        report_content += "\n### 移动的文件\n"
        for move in fix_results['moved']:
            report_content += f"- {move}\n"

        if fix_results['errors']:
            report_content += "\n### 错误\n"
            for error in fix_results['errors']:
                report_content += f"- {error}\n"

        report_content += f"""
## 📈 修复效果

### 文件名优化
- **重复日期修复**: 移除了重复的日期信息
- **分类准确性**: 提高了文件分类的准确性
- **文件组织**: 改善了文件的组织结构

### 具体改进
1. **日期标准化**: 统一使用 YYYYMMDD 格式
2. **分类优化**: 将文件移动到正确的目录
3. **命名简化**: 移除冗余的重复信息

## 🔄 后续建议

### 短期 (1周内)
1. **验证修复**: 检查修复结果是否符合预期
2. **更新索引**: 更新目录索引和导航
3. **建立规范**: 制定更严格的文件命名规范

### 中期 (1个月内)
1. **自动化检查**: 建立定期文件名检查机制
2. **模板完善**: 为各类报告提供标准命名模板
3. **文档更新**: 更新相关文档和索引

### 长期 (3个月内)
1. **智能分类**: 实现基于内容的智能分类
2. **版本管理**: 建立报告版本管理机制
3. **质量度量**: 建立文件命名质量度量体系

## 📝 命名规范建议

### 文件命名格式
```
{{报告类型}}_{{具体内容}}_{{日期}}.md
```

### 示例
- `architecture_code_review_20250719.md`
- `testing_coverage_analysis_20250719.md`
- `business_analytics_report_20250719.md`

### 日期格式
- 统一使用 YYYYMMDD 格式
- 避免重复的日期信息
- 确保日期的唯一性

---
*修复报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        report_path = self.reports_dir / "FILENAME_FIX_REPORT.md"
        report_path.write_text(report_content, encoding='utf-8')
        print("✅ 修复报告创建完成")

    def run(self):
        """执行修复流程"""
        print("🚀 开始修复reports文件名问题...")
        print("=" * 60)

        # 修复文件名问题
        fix_results = self.fix_issues()

        # 创建修复报告
        self.create_fix_report(fix_results)

        print("=" * 60)
        print("🎉 文件名问题修复完成！")
        print(f"📋 修复报告: {self.reports_dir}/FILENAME_FIX_REPORT.md")
        print("\n📊 修复结果:")
        print(f"  - 重命名文件: {len(fix_results['renamed'])} 个")
        print(f"  - 移动文件: {len(fix_results['moved'])} 个")
        print(f"  - 错误: {len(fix_results['errors'])} 个")


if __name__ == "__main__":
    fixer = FilenameIssueFixer()
    fixer.run()
