#!/usr/bin/env python3
"""
清理空目录脚本

检查并删除src/infrastructure目录中无代码文件的空目录
"""

import shutil
from pathlib import Path
from typing import Dict


class DirectoryCleanup:
    """目录清理器"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.infrastructure_dir = self.root_dir / "src" / "infrastructure"
        self.removed_dirs = []
        self.skipped_dirs = []

    def is_empty_of_code(self, dir_path: Path) -> bool:
        """
        检查目录是否为空代码文件

        目录为空代码文件如果：
        1. 目录下没有.py文件
        2. 或者只有__init__.py但没有其他.py文件
        3. 或者只有非代码文件（如.md, .txt等）
        """
        if not dir_path.exists() or not dir_path.is_dir():
            return False

        py_files = []
        non_code_files = []

        for item in dir_path.rglob("*"):
            if item.is_file():
                if item.name.endswith('.py'):
                    py_files.append(item)
                else:
                    non_code_files.append(item)

        # 如果没有Python文件，认为是空代码目录
        if not py_files:
            return True

        # 如果只有__init__.py一个Python文件，且没有其他代码文件，也认为是空代码目录
        if len(py_files) == 1 and py_files[0].name == '__init__.py':
            # 检查__init__.py的内容是否为空或只有基本导入
            try:
                with open(py_files[0], 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                # 如果__init__.py基本为空（只有注释或基本导入），也认为该目录为空
                if not content or content in ['', '"""', '"""', '# -*- coding: utf-8 -*-', 'from pathlib import Path\n\n__version__ = "1.0.0"']:
                    return True
            except:
                return True

        return False

    def cleanup_empty_directories(self) -> Dict[str, any]:
        """清理空目录"""
        print("🧹 开始清理空目录...")

        cleanup_result = {
            "removed_directories": [],
            "skipped_directories": [],
            "total_checked": 0,
            "total_removed": 0
        }

        # 获取所有子目录
        directories_to_check = []

        for item in self.infrastructure_dir.rglob("*"):
            if item.is_dir():
                directories_to_check.append(item)

        # 按深度倒序排列，确保先处理深层目录
        directories_to_check.sort(key=lambda x: len(x.parts), reverse=True)

        for dir_path in directories_to_check:
            # 跳过根目录
            if dir_path == self.infrastructure_dir:
                continue

            cleanup_result["total_checked"] += 1

            if self.is_empty_of_code(dir_path):
                try:
                    # 记录目录内容
                    dir_contents = list(dir_path.rglob("*"))
                    content_info = []

                    for item in dir_contents:
                        if item.is_file():
                            try:
                                size = item.stat().st_size
                                content_info.append(f"{item.name} ({size} bytes)")
                            except:
                                content_info.append(f"{item.name} (unknown size)")

                    # 删除目录
                    shutil.rmtree(dir_path)
                    cleanup_result["removed_directories"].append({
                        "path": str(dir_path.relative_to(self.infrastructure_dir)),
                        "contents": content_info
                    })
                    cleanup_result["total_removed"] += 1

                    print(f"  ✅ 删除空目录: {dir_path.relative_to(self.infrastructure_dir)}")

                except Exception as e:
                    cleanup_result["skipped_directories"].append({
                        "path": str(dir_path.relative_to(self.infrastructure_dir)),
                        "reason": f"删除失败: {e}"
                    })
                    print(f"  ❌ 删除失败: {dir_path.relative_to(self.infrastructure_dir)} - {e}")
            else:
                print(f"  ⏭️  跳过非空目录: {dir_path.relative_to(self.infrastructure_dir)}")

        print(
            f"✅ 目录清理完成，检查了 {cleanup_result['total_checked']} 个目录，删除了 {cleanup_result['total_removed']} 个空目录")
        return cleanup_result

    def generate_cleanup_report(self, cleanup_result: Dict[str, any]) -> str:
        """生成清理报告"""
        import datetime
        report = f"""# 空目录清理报告

## 📊 清理概览

**清理时间**: {datetime.datetime.now().isoformat()}
**检查目录数**: {cleanup_result['total_checked']} 个
**删除目录数**: {cleanup_result['total_removed']} 个
**跳过目录数**: {len(cleanup_result['skipped_directories'])} 个

---

## 🗑️ 删除的空目录

"""

        if cleanup_result['removed_directories']:
            for removed_dir in cleanup_result['removed_directories']:
                report += f"### `{removed_dir['path']}/`\n"
                if removed_dir['contents']:
                    report += "**目录内容**:\n"
                    for content in removed_dir['contents']:
                        report += f"- {content}\n"
                else:
                    report += "**目录内容**: 空目录\n"
                report += "\n"
        else:
            report += "无删除的空目录\n\n"

        if cleanup_result['skipped_directories']:
            report += "## ⏭️ 跳过的目录\n\n"
            for skipped_dir in cleanup_result['skipped_directories']:
                report += f"- `{skipped_dir['path']}` - {skipped_dir['reason']}\n"

        report += f"""

---

**清理工具**: scripts/cleanup_empty_dirs.py
**清理标准**: 无代码文件的空目录
**清理状态**: ✅ 完成
"""

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='空目录清理工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--output', help='输出报告文件')
    parser.add_argument('--dry-run', action='store_true', help='仅显示要删除的目录，不执行实际删除')

    args = parser.parse_args()

    cleanup = DirectoryCleanup(args.project)

    if args.dry_run:
        print("🔍 干运行模式 - 显示要删除的目录但不执行实际操作")

        # 模拟检查
        result = {
            "removed_directories": [],
            "skipped_directories": [],
            "total_checked": 0,
            "total_removed": 0
        }

        for item in cleanup.infrastructure_dir.rglob("*"):
            if item.is_dir() and item != cleanup.infrastructure_dir:
                result["total_checked"] += 1
                if cleanup.is_empty_of_code(item):
                    dir_contents = list(item.rglob("*"))
                    content_info = []
                    for content_item in dir_contents:
                        if content_item.is_file():
                            try:
                                size = content_item.stat().st_size
                                content_info.append(f"{content_item.name} ({size} bytes)")
                            except:
                                content_info.append(f"{content_item.name} (unknown size)")

                    result["removed_directories"].append({
                        "path": str(item.relative_to(cleanup.infrastructure_dir)),
                        "contents": content_info
                    })
                    print(f"  📁 将删除: {item.relative_to(cleanup.infrastructure_dir)}")

        print(f"🔍 干运行完成，发现 {len(result['removed_directories'])} 个空目录")

        if args.output:
            report = cleanup.generate_cleanup_report(result)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)

    else:
        result = cleanup.cleanup_empty_directories()

        report = cleanup.generate_cleanup_report(result)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
        else:
            print(report)


if __name__ == "__main__":
    main()
