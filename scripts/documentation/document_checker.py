#!/usr/bin/env python3
"""
文档自动化检查脚本

功能：
1. 检查文档链接有效性
2. 验证文档结构完整性
3. 检查文档更新及时性
4. 生成文档质量报告
"""

import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
import argparse
import json


class DocumentChecker:
    """文档检查器"""

    def __init__(self, docs_root: str = "docs"):
        self.docs_root = Path(docs_root)
        self.issues = []
        self.stats = {
            "total_files": 0,
            "broken_links": 0,
            "missing_readme": 0,
            "outdated_docs": 0,
            "structure_issues": 0
        }

    def check_all(self) -> Dict:
        """执行所有检查"""
        print("🔍 开始文档检查...")

        # 检查文档结构
        self.check_document_structure()

        # 检查链接有效性
        self.check_link_validity()

        # 检查README文件
        self.check_readme_files()

        # 检查文档更新及时性
        self.check_document_freshness()

        # 生成报告
        return self.generate_report()

    def check_document_structure(self):
        """检查文档结构完整性"""
        print("📁 检查文档结构...")

        # 检查必需的目录
        required_dirs = [
            "core", "trading", "data", "models", "features",
            "adapters", "services", "engine", "backtest",
            "ensemble", "tuning", "acceleration", "utils",
            "infrastructure", "architecture", "configuration",
            "deployment", "testing", "reports", "api"
        ]

        for dir_name in required_dirs:
            dir_path = self.docs_root / dir_name
            if not dir_path.exists():
                self.issues.append(f"缺少必需目录: {dir_name}")
                self.stats["structure_issues"] += 1
            elif not (dir_path / "README.md").exists():
                self.issues.append(f"缺少README文件: {dir_name}/README.md")
                self.stats["missing_readme"] += 1

    def check_link_validity(self):
        """检查文档链接有效性"""
        print("🔗 检查链接有效性...")

        for md_file in self.docs_root.rglob("*.md"):
            self.stats["total_files"] += 1
            self._check_file_links(md_file)

    def _check_file_links(self, file_path: Path):
        """检查单个文件的链接"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找Markdown链接
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            links = re.findall(link_pattern, content)

            for link_text, link_url in links:
                if link_url.startswith('http'):
                    # 外部链接，暂时跳过
                    continue

                # 处理相对路径
                if link_url.startswith('./'):
                    link_url = link_url[2:]
                elif link_url.startswith('../'):
                    # 计算相对路径
                    relative_path = file_path.parent / link_url[3:]
                    if not relative_path.exists():
                        self.issues.append(f"无效链接: {file_path} -> {link_url}")
                        self.stats["broken_links"] += 1
                else:
                    # 相对于docs根目录的路径
                    target_path = self.docs_root / link_url
                    if not target_path.exists():
                        self.issues.append(f"无效链接: {file_path} -> {link_url}")
                        self.stats["broken_links"] += 1

        except Exception as e:
            self.issues.append(f"读取文件失败: {file_path} - {str(e)}")

    def check_readme_files(self):
        """检查README文件"""
        print("📖 检查README文件...")

        # 检查主要模块的README
        main_modules = [
            "core", "trading", "data", "models", "features",
            "adapters", "services", "engine", "backtest"
        ]

        for module in main_modules:
            readme_path = self.docs_root / module / "README.md"
            if not readme_path.exists():
                self.issues.append(f"缺少模块README: {module}/README.md")
                self.stats["missing_readme"] += 1
            else:
                # 检查README内容结构
                self._check_readme_structure(readme_path)

    def _check_readme_structure(self, readme_path: Path):
        """检查README文件结构"""
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查必需的章节
            required_sections = [
                "## 📋 模块概述",
                "## 🏗️ 模块结构",
                "## 📚 文档索引",
                "## 🔧 使用指南"
            ]

            for section in required_sections:
                if section not in content:
                    self.issues.append(f"README缺少章节: {readme_path} - {section}")
                    self.stats["structure_issues"] += 1

        except Exception as e:
            self.issues.append(f"读取README失败: {readme_path} - {str(e)}")

    def check_document_freshness(self):
        """检查文档更新及时性"""
        print("⏰ 检查文档更新及时性...")

        # 检查最近30天内的更新
        cutoff_date = datetime.now() - timedelta(days=30)

        for md_file in self.docs_root.rglob("*.md"):
            try:
                mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
                if mtime < cutoff_date:
                    self.issues.append(f"文档可能过时: {md_file} (最后更新: {mtime.strftime('%Y-%m-%d')})")
                    self.stats["outdated_docs"] += 1
            except Exception as e:
                self.issues.append(f"检查文件时间失败: {md_file} - {str(e)}")

    def generate_report(self) -> Dict:
        """生成检查报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats,
            "issues": self.issues,
            "summary": self._generate_summary()
        }

        # 保存报告
        report_path = self.docs_root / "document_check_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report

    def _generate_summary(self) -> str:
        """生成总结"""
        total_issues = len(self.issues)

        if total_issues == 0:
            return "✅ 文档检查通过，未发现任何问题！"

        summary = f"📊 文档检查完成\n"
        summary += f"📁 总文件数: {self.stats['total_files']}\n"
        summary += f"❌ 发现问题: {total_issues}\n"
        summary += f"🔗 无效链接: {self.stats['broken_links']}\n"
        summary += f"📖 缺少README: {self.stats['missing_readme']}\n"
        summary += f"⏰ 过时文档: {self.stats['outdated_docs']}\n"
        summary += f"🏗️ 结构问题: {self.stats['structure_issues']}\n"

        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文档自动化检查工具")
    parser.add_argument("--docs-root", default="docs", help="文档根目录")
    parser.add_argument("--output", help="输出报告文件路径")

    args = parser.parse_args()

    # 创建检查器
    checker = DocumentChecker(args.docs_root)

    # 执行检查
    report = checker.check_all()

    # 输出结果
    print("\n" + "="*50)
    print(report["summary"])
    print("="*50)

    if report["issues"]:
        print("\n🔍 详细问题:")
        for i, issue in enumerate(report["issues"], 1):
            print(f"{i}. {issue}")

    # 保存报告
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 报告已保存到: {args.output}")

    # 返回退出码
    exit_code = 0 if not report["issues"] else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
