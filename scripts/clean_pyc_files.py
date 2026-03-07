#!/usr/bin/env python3
"""
清理.pyc文件脚本

自动查找和删除项目中的.pyc编译文件和__pycache__目录
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class PycFileCleaner:
    """.pyc文件清理器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.deleted_files = []
        self.deleted_dirs = []

    def find_pyc_files(self, directory: Path) -> List[Path]:
        """查找.pyc文件和__pycache__目录"""
        pyc_files = []
        pycache_dirs = []

        for root, dirs, files in os.walk(directory):
            # 检查__pycache__目录
            if '__pycache__' in dirs:
                pycache_path = Path(root) / '__pycache__'
                pycache_dirs.append(pycache_path)

            # 检查.pyc文件
            for file in files:
                if file.endswith('.pyc'):
                    pyc_files.append(Path(root) / file)

        return pyc_files, pycache_dirs

    def clean_pyc_files(self, layer: str = None) -> Dict[str, Any]:
        """清理.pyc文件和__pycache__目录"""
        if layer:
            target_dir = self.src_dir / layer
        else:
            target_dir = self.src_dir

        print("🔍 查找.pyc文件和__pycache__目录...")
        pyc_files, pycache_dirs = self.find_pyc_files(target_dir)

        total_items = len(pyc_files) + len(pycache_dirs)
        if total_items == 0:
            return {
                "success": True,
                "pyc_files_found": 0,
                "pycache_dirs_found": 0,
                "deleted_files": 0,
                "deleted_dirs": 0,
                "message": "没有发现.pyc文件或__pycache__目录"
            }

        print(f"📋 发现 {len(pyc_files)} 个.pyc文件和 {len(pycache_dirs)} 个__pycache__目录")

        if pyc_files:
            print(".pyc文件:")
            for file_path in pyc_files:
                print(f"  - {file_path.relative_to(self.src_dir)}")

        if pycache_dirs:
            print("__pycache__目录:")
            for dir_path in pycache_dirs:
                print(f"  - {dir_path.relative_to(self.src_dir)}")

        # 删除.pyc文件
        deleted_files = 0
        for file_path in pyc_files:
            try:
                file_path.unlink()
                self.deleted_files.append(str(file_path.relative_to(self.src_dir)))
                deleted_files += 1
                print(f"🗑️ 删除文件: {file_path.name}")
            except Exception as e:
                print(f"❌ 删除文件失败 {file_path}: {e}")

        # 删除__pycache__目录
        deleted_dirs = 0
        for dir_path in pycache_dirs:
            try:
                import shutil
                shutil.rmtree(dir_path)
                self.deleted_dirs.append(str(dir_path.relative_to(self.src_dir)))
                deleted_dirs += 1
                print(f"🗑️ 删除目录: {dir_path.name}")
            except Exception as e:
                print(f"❌ 删除目录失败 {dir_path}: {e}")

        return {
            "success": True,
            "pyc_files_found": len(pyc_files),
            "pycache_dirs_found": len(pycache_dirs),
            "deleted_files": deleted_files,
            "deleted_dirs": deleted_dirs,
            "deleted_file_list": self.deleted_files,
            "deleted_dir_list": self.deleted_dirs
        }

    def generate_report(self, result: Dict[str, Any]) -> str:
        """生成清理报告"""
        report = f"""# 🧹 .pyc文件清理报告

## 📅 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 清理结果
- **发现.pyc文件**: {result['pyc_files_found']} 个
- **发现__pycache__目录**: {result['pycache_dirs_found']} 个
- **删除.pyc文件**: {result['deleted_files']} 个
- **删除__pycache__目录**: {result['deleted_dirs']} 个

## 📋 删除的文件列表
"""

        if result['deleted_file_list']:
            for file_path in result['deleted_file_list']:
                report += f"- {file_path}\n"
        else:
            report += "没有删除.pyc文件"

        if result['deleted_dir_list']:
            report += """
## 📋 删除的目录列表
"""
            for dir_path in result['deleted_dir_list']:
                report += f"- {dir_path}\n"

        report += """
## 💡 注意事项
1. .pyc文件是Python的字节码编译文件，删除不会影响程序功能
2. __pycache__目录存储.pyc文件，删除后Python会重新生成
3. 此操作会加快git提交和项目备份速度
4. 下次运行Python程序时会自动重新生成.pyc文件
"""
        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='清理.pyc文件')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--layer', help='指定要清理的层（可选）')
    parser.add_argument('--report', action='store_true', help='生成详细报告')

    args = parser.parse_args()

    cleaner = PycFileCleaner(args.project)

    print("🧹 开始清理.pyc文件...")
    result = cleaner.clean_pyc_files(args.layer)

    if result['success']:
        print("✅ 清理完成!")
        print(f"   发现.pyc文件: {result['pyc_files_found']} 个")
        print(f"   删除.pyc文件: {result['deleted_files']} 个")
        print(f"   删除__pycache__目录: {result['deleted_dirs']} 个")

        if args.report:
            report_content = cleaner.generate_report(result)
            report_file = cleaner.project_root / "reports" / \
                f"pyc_cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)

            print(f"📊 详细报告已保存: {report_file}")
    else:
        print("❌ 清理失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
