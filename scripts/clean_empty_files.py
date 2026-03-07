#!/usr/bin/env python3
"""
清理空文件脚本

自动查找和删除项目中的空Python文件
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class EmptyFileCleaner:
    """空文件清理器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.deleted_files = []
        self.backup_dir = self.project_root / "backup" / \
            f"empty_files_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def find_empty_files(self, directory: Path) -> List[Path]:
        """查找空文件"""
        empty_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if not content:
                                empty_files.append(file_path)
                    except Exception as e:
                        print(f"❌ 读取文件失败 {file_path}: {e}")

        return empty_files

    def backup_file(self, file_path: Path) -> bool:
        """备份文件"""
        try:
            # 创建相对路径
            rel_path = file_path.relative_to(self.src_dir)
            backup_path = self.backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # 复制文件
            import shutil
            shutil.copy2(file_path, backup_path)
            return True
        except Exception as e:
            print(f"❌ 备份文件失败 {file_path}: {e}")
            return False

    def delete_file(self, file_path: Path) -> bool:
        """删除文件"""
        try:
            file_path.unlink()
            self.deleted_files.append(str(file_path.relative_to(self.src_dir)))
            return True
        except Exception as e:
            print(f"❌ 删除文件失败 {file_path}: {e}")
            return False

    def clean_empty_files(self, layer: str = None) -> Dict[str, Any]:
        """清理空文件"""
        if layer:
            target_dir = self.src_dir / layer
        else:
            target_dir = self.src_dir

        print("🔍 查找空文件...")
        empty_files = self.find_empty_files(target_dir)

        if not empty_files:
            return {
                "success": True,
                "empty_files_found": 0,
                "deleted_files": 0,
                "message": "没有发现空文件"
            }

        print(f"📋 发现 {len(empty_files)} 个空文件:")
        for file_path in empty_files:
            print(f"  - {file_path.relative_to(self.src_dir)}")

        # 备份并删除文件
        deleted_count = 0
        failed_count = 0

        for file_path in empty_files:
            print(f"📦 备份文件: {file_path.name}")
            if self.backup_file(file_path):
                print(f"🗑️ 删除文件: {file_path.name}")
                if self.delete_file(file_path):
                    deleted_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1

        return {
            "success": True,
            "empty_files_found": len(empty_files),
            "deleted_files": deleted_count,
            "failed_files": failed_count,
            "backup_directory": str(self.backup_dir),
            "deleted_file_list": self.deleted_files
        }

    def generate_report(self, result: Dict[str, Any]) -> str:
        """生成清理报告"""
        report = f"""# 🧹 空文件清理报告

## 📅 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 清理结果
- **发现空文件**: {result['empty_files_found']} 个
- **成功删除**: {result['deleted_files']} 个
- **删除失败**: {result.get('failed_files', 0)} 个
- **备份目录**: {result.get('backup_directory', 'N/A')}

## 📋 删除的文件列表
"""

        if result['deleted_file_list']:
            for file_path in result['deleted_file_list']:
                report += f"- {file_path}\n"
        else:
            report += "没有删除任何文件"

        report += """
## 💡 注意事项
1. 所有删除的文件都已备份到备份目录
2. 如需恢复文件，请从备份目录复制
3. 删除的都是完全空的Python文件
4. 此操作不会影响项目的功能
"""
        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='清理空文件')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--layer', help='指定要清理的层（可选）')
    parser.add_argument('--report', action='store_true', help='生成详细报告')

    args = parser.parse_args()

    cleaner = EmptyFileCleaner(args.project)

    print("🧹 开始清理空文件...")
    result = cleaner.clean_empty_files(args.layer)

    if result['success']:
        print("✅ 清理完成!")
        print(f"   发现空文件: {result['empty_files_found']} 个")
        print(f"   成功删除: {result['deleted_files']} 个")

        if result['deleted_files'] > 0:
            print(f"   备份目录: {result['backup_directory']}")

        if args.report:
            report_content = cleaner.generate_report(result)
            report_file = cleaner.project_root / "reports" / \
                f"empty_files_cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)

            print(f"📊 详细报告已保存: {report_file}")
    else:
        print("❌ 清理失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
