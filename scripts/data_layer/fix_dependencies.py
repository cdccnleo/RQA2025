#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据层依赖关系修复脚本

修复数据层组件对上层组件的依赖问题，确保分层架构的正确性。
"""

import re
from pathlib import Path
from typing import List, Set


class DependencyFixer:
    """依赖关系修复器"""

    def __init__(self, data_layer_path: str):
        self.data_layer_path = Path(data_layer_path)
        self.fixed_files: Set[str] = set()
        self.backup_files: Set[str] = set()

    def find_files_with_engine_logging(self) -> List[Path]:
        """查找包含引擎层日志依赖的文件"""
        pattern = "from src\.engine\.logging"
        matching_files = []

        # 递归查找所有Python文件
        for py_file in self.data_layer_path.rglob("*.py"):
            # 跳过备份文件
            if any(backup in str(py_file) for backup in ['.backup', '.deep_backup']):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if re.search(pattern, content):
                        matching_files.append(py_file)
            except Exception as e:
                print(f"读取文件失败 {py_file}: {e}")

        return matching_files

    def fix_engine_logging_import(self, file_path: Path) -> bool:
        """修复单个文件的引擎层日志导入"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 创建备份
            backup_path = file_path.with_suffix('.backup')
            if not backup_path.exists():
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.backup_files.add(str(backup_path))

            # 替换引擎层日志导入
            original_content = content

            # 替换 get_unified_logger 导入
            content = re.sub(
                r'from src\.engine\.logging\.unified_logger import get_unified_logger',
                'from src.infrastructure.logging.infrastructure_logger import get_infrastructure_logger',
                content
            )

            # 替换 get_logger 导入
            content = re.sub(
                r'from src\.utils\.logger import get_logger',
                'from src.infrastructure.logging.infrastructure_logger import get_infrastructure_logger',
                content
            )

            # 替换日志器初始化
            content = re.sub(
                r'logger = get_unified_logger\((.*?)\)',
                r'logger = get_infrastructure_logger(\1)',
                content
            )

            content = re.sub(
                r'logger = get_logger\((.*?)\)',
                r'logger = get_infrastructure_logger(\1)',
                content
            )

            # 添加降级处理
            if 'get_infrastructure_logger' in content and 'try:' not in content[:content.find('get_infrastructure_logger')]:
                # 在文件开头添加降级处理
                import_section = content.find('import ')
                if import_section != -1:
                    # 找到第一个import语句，在其前添加降级处理
                    lines = content.split('\n')
                    insert_pos = 0
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            insert_pos = i
                            break

                    # 插入降级处理代码
                    downgrade_code = '''# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging.infrastructure_logger import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging
    def get_infrastructure_logger(name):
        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger
'''

                    lines.insert(insert_pos, downgrade_code)
                    content = '\n'.join(lines)

            # 如果内容有变化，写入文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.add(str(file_path))
                return True

            return False

        except Exception as e:
            print(f"修复文件失败 {file_path}: {e}")
            return False

    def fix_all_files(self) -> None:
        """修复所有文件"""
        print("🔍 查找包含引擎层日志依赖的文件...")
        files_to_fix = self.find_files_with_engine_logging()

        if not files_to_fix:
            print("✅ 没有找到需要修复的文件")
            return

        print(f"📝 找到 {len(files_to_fix)} 个需要修复的文件")
        for file_path in files_to_fix:
            print(f"🔧 修复文件: {file_path}")
            if self.fix_engine_logging_import(file_path):
                print("  ✅ 修复成功")
            else:
                print("  ❌ 修复失败")
        print("\n📊 修复统计:")
        print(f"  修复的文件数: {len(self.fixed_files)}")
        print(f"  创建的备份数: {len(self.backup_files)}")

        if self.backup_files:
            print("\n💾 备份文件列表:")
            for backup in sorted(self.backup_files):
                print(f"  {backup}")

    def cleanup_backups(self) -> None:
        """清理备份文件"""
        print("🧹 清理备份文件...")
        backup_count = 0
        for backup_file in self.data_layer_path.rglob("*.backup"):
            try:
                backup_file.unlink()
                backup_count += 1
            except Exception as e:
                print(f"删除备份文件失败 {backup_file}: {e}")

        print(f"✅ 已清理 {backup_count} 个备份文件")


def main():
    """主函数"""
    print("🚀 数据层依赖关系修复脚本")
    print("=" * 50)

    # 数据层路径
    data_layer_path = Path("src/data")

    if not data_layer_path.exists():
        print(f"❌ 数据层路径不存在: {data_layer_path}")
        return

    # 创建修复器
    fixer = DependencyFixer(data_layer_path)

    # 修复依赖关系
    fixer.fix_all_files()

    # 可选：清理备份文件
    if input("\n是否清理备份文件? (y/N): ").lower() == 'y':
        fixer.cleanup_backups()

    print("\n🎉 依赖关系修复完成！")


if __name__ == "__main__":
    main()
