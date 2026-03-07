#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 报告目录重新组织脚本

按照新的命名规范和组织结构重新整理reports目录
"""

import os
import shutil
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple


class ReportReorganizer:
    """报告重新组织器"""

    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.backup_dir = self.reports_dir / "backup_before_reorganization"

        # 新的目录结构定义
        self.new_structure = {
            "project": {
                "progress": [],
                "completion": [],
                "architecture": [],
                "deployment": []
            },
            "technical": {
                "testing": [],
                "performance": [],
                "security": [],
                "quality": [],
                "optimization": []
            },
            "business": {
                "analytics": [],
                "trading": [],
                "backtest": [],
                "compliance": []
            },
            "operational": {
                "monitoring": [],
                "deployment": [],
                "notification": [],
                "maintenance": []
            },
            "research": {
                "ml_integration": [],
                "deep_learning": [],
                "reinforcement_learning": [],
                "continuous_optimization": []
            }
        }

        # 文件分类规则
        self.classification_rules = {
            # 项目报告
            "project": {
                "progress": [
                    r"progress", r"milestone", r"status", r"update"
                ],
                "completion": [
                    r"completion", r"final", r"complete", r"done"
                ],
                "architecture": [
                    r"architecture", r"design", r"structure", r"code_review"
                ],
                "deployment": [
                    r"deployment", r"deploy", r"install", r"setup"
                ]
            },
            # 技术报告
            "technical": {
                "testing": [
                    r"test", r"testing", r"ast_analysis", r"data_loader"
                ],
                "performance": [
                    r"performance", r"benchmark", r"optimization", r"speed"
                ],
                "security": [
                    r"security", r"audit", r"vulnerability", r"risk"
                ],
                "quality": [
                    r"quality", r"code_quality", r"technical_debt"
                ],
                "optimization": [
                    r"optimization", r"improvement", r"enhancement"
                ]
            },
            # 业务报告
            "business": {
                "analytics": [
                    r"analytics", r"analysis", r"batch_report"
                ],
                "trading": [
                    r"trading", r"trade", r"strategy"
                ],
                "backtest": [
                    r"backtest", r"backtesting", r"consistency"
                ],
                "compliance": [
                    r"compliance", r"regulatory", r"regulation"
                ]
            },
            # 运维报告
            "operational": {
                "monitoring": [
                    r"monitoring", r"monitor", r"alert"
                ],
                "deployment": [
                    r"deployment", r"deploy", r"blue_green"
                ],
                "notification": [
                    r"notification", r"notify", r"alert"
                ],
                "maintenance": [
                    r"maintenance", r"maintain", r"support"
                ]
            },
            # 研究报告
            "research": {
                "ml_integration": [
                    r"ml_integration", r"machine_learning"
                ],
                "deep_learning": [
                    r"deep_learning", r"neural", r"nn"
                ],
                "reinforcement_learning": [
                    r"reinforcement_learning", r"rl_"
                ],
                "continuous_optimization": [
                    r"continuous_optimization", r"auto_optimization"
                ]
            }
        }

    def backup_current_structure(self) -> None:
        """备份当前目录结构"""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)

        shutil.copytree(self.reports_dir, self.backup_dir,
                        ignore=shutil.ignore_patterns("backup_*"))
        print(f"✅ 已备份当前结构到: {self.backup_dir}")

    def classify_file(self, file_path: Path) -> Tuple[str, str]:
        """分类文件到新的目录结构"""
        file_name = file_path.stem.lower()

        for category, subcategories in self.classification_rules.items():
            for subcategory, patterns in subcategories.items():
                for pattern in patterns:
                    if re.search(pattern, file_name, re.IGNORECASE):
                        return category, subcategory

        # 默认分类
        if "test" in file_name or "testing" in file_name:
            return "technical", "testing"
        elif "performance" in file_name or "benchmark" in file_name:
            return "technical", "performance"
        elif "security" in file_name or "audit" in file_name:
            return "technical", "security"
        elif "project" in file_name or "progress" in file_name:
            return "project", "progress"
        else:
            return "project", "progress"  # 默认分类

    def create_new_structure(self) -> None:
        """创建新的目录结构"""
        for category, subcategories in self.new_structure.items():
            category_dir = self.reports_dir / category
            category_dir.mkdir(exist_ok=True)

            for subcategory in subcategories.keys():
                subcategory_dir = category_dir / subcategory
                subcategory_dir.mkdir(exist_ok=True)

                # 创建README文件
                readme_file = subcategory_dir / "README.md"
                if not readme_file.exists():
                    self.create_readme_file(readme_file, category, subcategory)

        print("✅ 已创建新的目录结构")

    def create_readme_file(self, readme_path: Path, category: str, subcategory: str) -> None:
        """创建README文件"""
        category_names = {
            "project": "项目报告",
            "technical": "技术报告",
            "business": "业务报告",
            "operational": "运维报告",
            "research": "研究报告"
        }

        subcategory_names = {
            "progress": "进度报告",
            "completion": "完成报告",
            "architecture": "架构报告",
            "deployment": "部署报告",
            "testing": "测试报告",
            "performance": "性能报告",
            "security": "安全报告",
            "quality": "质量报告",
            "optimization": "优化报告",
            "analytics": "分析报告",
            "trading": "交易报告",
            "backtest": "回测报告",
            "compliance": "合规报告",
            "monitoring": "监控报告",
            "notification": "通知报告",
            "maintenance": "维护报告",
            "ml_integration": "机器学习集成",
            "deep_learning": "深度学习",
            "reinforcement_learning": "强化学习",
            "continuous_optimization": "持续优化"
        }

        content = f"""# {category_names.get(category, category)} - {subcategory_names.get(subcategory, subcategory)}

## 📋 概述

本目录包含{category_names.get(category, category)}中的{subcategory_names.get(subcategory, subcategory)}。

## 📁 文件列表

<!-- 文件列表将自动生成 -->

## 📊 统计信息

- 总文件数: 0
- 最后更新: {datetime.now().strftime('%Y-%m-%d')}

---

**维护者**: 项目团队  
**状态**: ✅ 活跃维护
"""

        readme_path.write_text(content, encoding='utf-8')

    def move_files_to_new_structure(self) -> Dict[str, int]:
        """将文件移动到新的目录结构"""
        moved_files = {}

        # 遍历所有文件
        for file_path in self.reports_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.md', '.json', '.html']:
                # 跳过备份目录和新创建的目录
                if "backup_" in str(file_path) or file_path.parent.name in self.new_structure:
                    continue

                # 分类文件
                category, subcategory = self.classify_file(file_path)

                # 目标路径
                target_dir = self.reports_dir / category / subcategory
                target_path = target_dir / file_path.name

                # 如果目标文件已存在，添加时间戳
                if target_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    name_parts = file_path.stem.split('_')
                    name_parts.append(timestamp)
                    new_name = '_'.join(name_parts) + file_path.suffix
                    target_path = target_dir / new_name

                # 移动文件
                try:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(target_path))

                    # 统计
                    key = f"{category}/{subcategory}"
                    moved_files[key] = moved_files.get(key, 0) + 1

                    print(f"📁 移动: {file_path.name} -> {target_path}")

                except Exception as e:
                    print(f"❌ 移动失败: {file_path.name} - {e}")

        return moved_files

    def update_index_files(self) -> None:
        """更新索引文件"""
        # 更新主索引
        self.update_main_index()

        # 更新各子目录的README
        self.update_subdirectory_readmes()

    def update_main_index(self) -> None:
        """更新主索引文件"""
        index_content = self.generate_main_index_content()
        index_file = self.reports_dir / "INDEX.md"
        index_file.write_text(index_content, encoding='utf-8')
        print("✅ 已更新主索引文件")

    def generate_main_index_content(self) -> str:
        """生成主索引内容"""
        content = """# RQA2025 项目报告索引

## 📋 报告概览

本文档提供了RQA2025项目所有报告的索引，按功能模块和报告类型进行分类。

"""

        # 统计各目录的文件
        stats = {}
        for category in self.new_structure.keys():
            category_dir = self.reports_dir / category
            if category_dir.exists():
                stats[category] = {}
                for subcategory in self.new_structure[category].keys():
                    subcategory_dir = category_dir / subcategory
                    if subcategory_dir.exists():
                        files = list(subcategory_dir.glob("*.md")) + \
                            list(subcategory_dir.glob("*.json"))
                        stats[category][subcategory] = len(files)

        # 生成索引内容
        for category, subcategories in stats.items():
            category_names = {
                "project": "项目报告",
                "technical": "技术报告",
                "business": "业务报告",
                "operational": "运维报告",
                "research": "研究报告"
            }

            content += f"## {category_names.get(category, category)} ({category}/)\n\n"

            for subcategory, count in subcategories.items():
                if count > 0:
                    subcategory_names = {
                        "progress": "进度报告",
                        "completion": "完成报告",
                        "architecture": "架构报告",
                        "deployment": "部署报告",
                        "testing": "测试报告",
                        "performance": "性能报告",
                        "security": "安全报告",
                        "quality": "质量报告",
                        "optimization": "优化报告",
                        "analytics": "分析报告",
                        "trading": "交易报告",
                        "backtest": "回测报告",
                        "compliance": "合规报告",
                        "monitoring": "监控报告",
                        "notification": "通知报告",
                        "maintenance": "维护报告",
                        "ml_integration": "机器学习集成",
                        "deep_learning": "深度学习",
                        "reinforcement_learning": "强化学习",
                        "continuous_optimization": "持续优化"
                    }

                    content += f"### {subcategory_names.get(subcategory, subcategory)} ({subcategory}/)\n"
                    content += f"- 文件数量: {count}个\n"
                    content += f"- [查看详情]({category}/{subcategory}/README.md)\n\n"

        content += """## 📋 报告状态

### 状态标识
- ✅ **已完成** - 报告已完成并可用
- 🔄 **进行中** - 报告正在编写中
- 📋 **待创建** - 报告计划但尚未创建
- 🗄️ **已归档** - 报告已移至归档目录

## 🔗 快速链接

### 常用报告
- [项目进度总览](project/progress/)
- [测试结果汇总](technical/testing/)
- [性能分析报告](technical/performance/)
- [安全审计报告](technical/security/)

---

**最后更新**: """ + datetime.now().strftime('%Y-%m-%d') + """  
**维护者**: 项目团队  
**状态**: ✅ 活跃维护
"""

        return content

    def update_subdirectory_readmes(self) -> None:
        """更新子目录的README文件"""
        for category, subcategories in self.new_structure.items():
            category_dir = self.reports_dir / category
            if category_dir.exists():
                for subcategory in subcategories.keys():
                    subcategory_dir = category_dir / subcategory
                    if subcategory_dir.exists():
                        self.update_subdirectory_readme(subcategory_dir, category, subcategory)

    def update_subdirectory_readme(self, subcategory_dir: Path, category: str, subcategory: str) -> None:
        """更新子目录的README文件"""
        files = list(subcategory_dir.glob("*.md")) + list(subcategory_dir.glob("*.json"))

        if not files:
            return

        # 读取现有README
        readme_file = subcategory_dir / "README.md"
        if readme_file.exists():
            content = readme_file.read_text(encoding='utf-8')
        else:
            content = ""

        # 更新文件列表
        file_list_start = content.find("## 📁 文件列表")
        if file_list_start != -1:
            file_list_end = content.find("## 📊 统计信息")
            if file_list_end == -1:
                file_list_end = len(content)

            new_file_list = "## 📁 文件列表\n\n"
            for file_path in sorted(files):
                file_name = file_path.name
                file_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')
                new_file_list += f"- [{file_name}]({file_name}) - {file_date}\n"

            new_file_list += "\n"

            # 替换文件列表部分
            content = content[:file_list_start] + new_file_list + content[file_list_end:]

            # 更新统计信息
            stats_start = content.find("## 📊 统计信息")
            if stats_start != -1:
                stats_end = content.find("\n---", stats_start)
                if stats_end == -1:
                    stats_end = len(content)

                new_stats = f"""## 📊 统计信息

- 总文件数: {len(files)}
- 最后更新: {datetime.now().strftime('%Y-%m-%d')}
- 文件类型: {', '.join(set(f.suffix for f in files))}

"""

                content = content[:stats_start] + new_stats + content[stats_end:]

        readme_file.write_text(content, encoding='utf-8')

    def cleanup_empty_directories(self) -> None:
        """清理空目录"""
        for root, dirs, files in os.walk(self.reports_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        print(f"🗑️ 删除空目录: {dir_path}")
                except Exception as e:
                    print(f"❌ 删除目录失败: {dir_path} - {e}")

    def reorganize(self) -> None:
        """执行重新组织"""
        print("🚀 开始重新组织reports目录...")

        # 1. 备份当前结构
        self.backup_current_structure()

        # 2. 创建新结构
        self.create_new_structure()

        # 3. 移动文件
        moved_files = self.move_files_to_new_structure()

        # 4. 更新索引
        self.update_index_files()

        # 5. 清理空目录
        self.cleanup_empty_directories()

        # 6. 输出统计
        print("\n📊 重新组织完成统计:")
        for category_subcategory, count in moved_files.items():
            print(f"  {category_subcategory}: {count}个文件")

        print(f"\n✅ 重新组织完成！备份位置: {self.backup_dir}")


def main():
    """主函数"""
    reorganizer = ReportReorganizer()
    reorganizer.reorganize()


if __name__ == "__main__":
    main()
