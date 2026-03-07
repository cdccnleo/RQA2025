#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
代码质量监控脚本

定期检查代码质量指标，包括：
1. 导入一致性
2. 代码重复检测
3. 架构分层合规性
4. 文档完整性
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import subprocess
import sys


class CodeQualityMonitor:
    """代码质量监控器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "reports" / "quality"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def check_import_consistency(self) -> Dict[str, Any]:
        """检查导入一致性"""
        print("🔍 检查导入一致性...")

        try:
            # 运行导入一致性检查
            result = subprocess.run([
                sys.executable,
                "scripts/development/check_import_consistency.py",
                "--output", str(self.reports_dir / "import_consistency_temp.md")
            ], capture_output=True, text=True, cwd=self.project_root)

            # 读取检查结果
            report_file = self.reports_dir / "import_consistency_temp.md"
            if report_file.exists():
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析结果
                issues_count = content.count("Line ")
                files_with_issues = len([line for line in content.split('\n')
                                         if line.startswith('src\\') and ':' in line])

                return {
                    "status": "completed",
                    "issues_count": issues_count,
                    "files_with_issues": files_with_issues,
                    "content": content
                }
            else:
                return {
                    "status": "failed",
                    "error": "检查脚本执行失败"
                }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def check_code_duplication(self) -> Dict[str, Any]:
        """检查代码重复"""
        print("🔍 检查代码重复...")

        try:
            # 运行代码重复检测
            result = subprocess.run([
                sys.executable,
                "scripts/development/migrate_imports.py",
                "--dry-run",
                "--verbose"
            ], capture_output=True, text=True, cwd=self.project_root)

            # 解析输出
            output = result.stdout
            changes_count = output.count("Would change:")

            return {
                "status": "completed",
                "changes_count": changes_count,
                "output": output
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def check_architecture_compliance(self) -> Dict[str, Any]:
        """检查架构分层合规性"""
        print("🔍 检查架构分层合规性...")

        issues = []

        # 检查关键文件
        key_files = [
            "src/utils/logger.py",
            "src/utils/date_utils.py",
            "src/infrastructure/utils/logger.py",
            "src/infrastructure/utils/date_utils.py"
        ]

        for file_path in key_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查重定向实现
                if "src/utils" in file_path:
                    if "from src.infrastructure.utils" in content:
                        issues.append(f"{file_path}: 包含基础设施层导入")

                # 检查基础设施层功能
                if "src/infrastructure" in file_path:
                    if "get_logger" in content and "def get_logger" in content:
                        issues.append(f"{file_path}: 包含标准功能实现")

        return {
            "status": "completed",
            "issues": issues,
            "issues_count": len(issues)
        }

    def check_documentation_completeness(self) -> Dict[str, Any]:
        """检查文档完整性"""
        print("🔍 检查文档完整性...")

        required_docs = [
            "docs/development/code_duplication_analysis.md",
            "docs/development/import_standards.md",
            "docs/development/code_review_guidelines.md",
            "scripts/development/migrate_imports.py",
            "scripts/development/check_import_consistency.py",
            "scripts/development/verify_migration.py"
        ]

        missing_docs = []
        for doc_path in required_docs:
            if not (self.project_root / doc_path).exists():
                missing_docs.append(doc_path)

        return {
            "status": "completed",
            "missing_docs": missing_docs,
            "missing_count": len(missing_docs),
            "total_required": len(required_docs)
        }

    def run_verification_tests(self) -> Dict[str, Any]:
        """运行验证测试"""
        print("🔍 运行验证测试...")

        try:
            result = subprocess.run([
                sys.executable,
                "scripts/development/verify_migration.py"
            ], capture_output=True, text=True, cwd=self.project_root)

            output = result.stdout
            passed_tests = output.count("✅")
            failed_tests = output.count("❌")

            return {
                "status": "completed",
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "output": output
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成质量报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"quality_report_{timestamp}.json"

        # 保存详细结果
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 生成摘要报告
        summary = []
        summary.append("# 代码质量监控报告")
        summary.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")

        # 导入一致性
        import_result = results.get("import_consistency", {})
        if import_result.get("status") == "completed":
            summary.append(f"## 导入一致性检查")
            summary.append(f"- 问题数量: {import_result.get('issues_count', 0)}")
            summary.append(f"- 有问题的文件数: {import_result.get('files_with_issues', 0)}")
            summary.append("")

        # 代码重复
        duplication_result = results.get("code_duplication", {})
        if duplication_result.get("status") == "completed":
            summary.append(f"## 代码重复检查")
            summary.append(f"- 需要变更的数量: {duplication_result.get('changes_count', 0)}")
            summary.append("")

        # 架构合规性
        arch_result = results.get("architecture_compliance", {})
        if arch_result.get("status") == "completed":
            summary.append(f"## 架构分层合规性")
            summary.append(f"- 问题数量: {arch_result.get('issues_count', 0)}")
            if arch_result.get("issues"):
                summary.append("### 发现的问题:")
                for issue in arch_result["issues"]:
                    summary.append(f"- {issue}")
            summary.append("")

        # 文档完整性
        doc_result = results.get("documentation_completeness", {})
        if doc_result.get("status") == "completed":
            summary.append(f"## 文档完整性")
            summary.append(f"- 缺失文档数: {doc_result.get('missing_count', 0)}")
            summary.append(f"- 总要求文档数: {doc_result.get('total_required', 0)}")
            if doc_result.get("missing_docs"):
                summary.append("### 缺失的文档:")
                for doc in doc_result["missing_docs"]:
                    summary.append(f"- {doc}")
            summary.append("")

        # 验证测试
        test_result = results.get("verification_tests", {})
        if test_result.get("status") == "completed":
            summary.append(f"## 验证测试")
            summary.append(f"- 通过测试数: {test_result.get('passed_tests', 0)}")
            summary.append(f"- 失败测试数: {test_result.get('failed_tests', 0)}")
            summary.append("")

        # 总体评估
        summary.append("## 总体评估")

        total_issues = (
            import_result.get("issues_count", 0) +
            duplication_result.get("changes_count", 0) +
            arch_result.get("issues_count", 0) +
            doc_result.get("missing_count", 0) +
            test_result.get("failed_tests", 0)
        )

        if total_issues == 0:
            summary.append("🎉 **优秀**: 所有检查项目都通过！")
        elif total_issues <= 5:
            summary.append("✅ **良好**: 大部分检查项目通过，有少量问题需要关注。")
        elif total_issues <= 10:
            summary.append("⚠️ **一般**: 存在一些问题需要修复。")
        else:
            summary.append("❌ **需要改进**: 存在较多问题需要优先处理。")

        summary.append(f"- 总问题数: {total_issues}")

        # 保存摘要报告
        summary_file = self.reports_dir / f"quality_summary_{timestamp}.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary))

        return str(summary_file)

    def run_full_check(self) -> str:
        """运行完整检查"""
        print("🚀 开始代码质量监控...")
        print("=" * 50)

        results = {
            "timestamp": datetime.now().isoformat(),
            "import_consistency": self.check_import_consistency(),
            "code_duplication": self.check_code_duplication(),
            "architecture_compliance": self.check_architecture_compliance(),
            "documentation_completeness": self.check_documentation_completeness(),
            "verification_tests": self.run_verification_tests()
        }

        # 生成报告
        report_file = self.generate_report(results)

        print("=" * 50)
        print(f"📊 质量监控完成！")
        print(f"📄 详细报告: {report_file}")

        return report_file


def main():
    parser = argparse.ArgumentParser(description='代码质量监控')
    parser.add_argument('--project-root', default='.', help='项目根目录')
    parser.add_argument('--output-dir', help='输出目录')

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"项目根目录不存在: {project_root}")
        return

    monitor = CodeQualityMonitor(project_root)
    report_file = monitor.run_full_check()

    print(f"✅ 监控完成，报告已生成: {report_file}")


if __name__ == "__main__":
    main()
