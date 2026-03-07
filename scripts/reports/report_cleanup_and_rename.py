#!/usr/bin/env python3
"""
报告清理和重命名脚本

功能：
1. 移除文件名中的日期标识
2. 实施新的命名规范
3. 清理重复和过期的报告
4. 自动归档历史报告
"""

import re
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List


class ReportCleanupManager:
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.archive_dir = self.reports_dir / "archive"
        self.deprecated_dir = self.archive_dir / "deprecated"

        # 确保归档目录存在
        self.archive_dir.mkdir(exist_ok=True)
        self.deprecated_dir.mkdir(exist_ok=True)

        # 报告数量限制
        self.max_reports_per_dir = 10
        self.archive_days = 30

    def get_all_report_files(self) -> List[Path]:
        """获取所有报告文件"""
        report_files = []
        for file_path in self.reports_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.md', '.json', '.html']:
                # 排除归档目录
                if not str(file_path).startswith(str(self.archive_dir)):
                    report_files.append(file_path)
        return report_files

    def extract_date_from_filename(self, filename: str) -> str:
        """从文件名中提取日期"""
        # 匹配各种日期格式
        date_patterns = [
            r'_(\d{8})_',  # YYYYMMDD
            r'_(\d{4}\d{2}\d{2})_',  # YYYYMMDD
            r'_(\d{4}-\d{2}-\d{2})_',  # YYYY-MM-DD
            r'_(\d{4}_\d{2}_\d{2})_',  # YYYY_MM_DD
        ]

        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        return None

    def clean_filename(self, filename: str) -> str:
        """清理文件名，移除日期标识"""
        # 移除日期模式
        date_patterns = [
            r'_\d{8}_',  # YYYYMMDD
            r'_\d{4}\d{2}\d{2}_',  # YYYYMMDD
            r'_\d{4}-\d{2}-\d{2}_',  # YYYY-MM-DD
            r'_\d{4}_\d{2}_\d{2}_',  # YYYY_MM_DD
            r'_\d{10,}',  # 时间戳
        ]

        cleaned_name = filename
        for pattern in date_patterns:
            cleaned_name = re.sub(pattern, '_', cleaned_name)

        # 清理多余的下划线
        cleaned_name = re.sub(r'_+', '_', cleaned_name)
        cleaned_name = cleaned_name.strip('_')

        return cleaned_name

    def get_version_suffix(self, file_path: Path) -> str:
        """根据文件内容确定版本后缀"""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'report_metadata' in data and 'version' in data['report_metadata']:
                        return data['report_metadata']['version']

            # 检查文件名中是否已有版本标识
            if re.search(r'_v\d+$', file_path.stem):
                return ""

            # 根据文件修改时间判断版本
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime > datetime.now() - timedelta(days=7):
                return "_latest"
            else:
                return "_v1"

        except Exception:
            return "_v1"

    def rename_file(self, file_path: Path) -> Path:
        """重命名文件，移除日期并添加版本"""
        filename = file_path.name
        date = self.extract_date_from_filename(filename)

        if date:
            # 清理文件名
            cleaned_name = self.clean_filename(filename)

            # 添加版本后缀
            version_suffix = self.get_version_suffix(file_path)
            if version_suffix and not cleaned_name.endswith(version_suffix):
                cleaned_name = cleaned_name.replace(
                    file_path.suffix, f"{version_suffix}{file_path.suffix}")

            new_path = file_path.parent / cleaned_name

            # 如果新文件名已存在，添加数字后缀
            counter = 1
            original_new_path = new_path
            while new_path.exists():
                stem = original_new_path.stem
                suffix = original_new_path.suffix
                new_path = original_new_path.parent / f"{stem}_{counter}{suffix}"
                counter += 1

            return new_path

        return file_path

    def should_archive_file(self, file_path: Path) -> bool:
        """判断文件是否应该归档"""
        try:
            # 检查文件修改时间
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime < datetime.now() - timedelta(days=self.archive_days):
                return True

            # 检查文件名中的日期
            filename = file_path.name
            date_str = self.extract_date_from_filename(filename)
            if date_str:
                try:
                    # 尝试解析日期
                    if len(date_str) == 8:  # YYYYMMDD
                        file_date = datetime.strptime(date_str, '%Y%m%d')
                    elif len(date_str) == 10 and '-' in date_str:  # YYYY-MM-DD
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')
                    else:
                        return False

                    if file_date < datetime.now() - timedelta(days=self.archive_days):
                        return True
                except ValueError:
                    pass

            return False
        except Exception:
            return False

    def archive_file(self, file_path: Path) -> Path:
        """归档文件"""
        # 创建归档目录结构
        relative_path = file_path.relative_to(self.reports_dir)
        archive_path = self.archive_dir / relative_path

        # 确保归档目录存在
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        # 移动文件
        shutil.move(str(file_path), str(archive_path))
        return archive_path

    def limit_reports_per_directory(self):
        """限制每个目录的报告数量"""
        for dir_path in self.reports_dir.rglob("*"):
            if dir_path.is_dir() and not str(dir_path).startswith(str(self.archive_dir)):
                files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix in [
                    '.md', '.json', '.html']]

                if len(files) > self.max_reports_per_dir:
                    # 按修改时间排序，保留最新的文件
                    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                    # 归档多余的文件
                    for file_path in files[self.max_reports_per_dir:]:
                        self.archive_file(file_path)
                        print(f"归档文件: {file_path} -> {self.archive_dir}")

    def merge_similar_reports(self):
        """合并相似功能的报告"""
        # 按目录分组文件
        dir_files = {}
        for file_path in self.get_all_report_files():
            dir_key = str(file_path.parent)
            if dir_key not in dir_files:
                dir_files[dir_key] = []
            dir_files[dir_key].append(file_path)

        # 检查每个目录中的相似文件
        for dir_path, files in dir_files.items():
            if len(files) <= 1:
                continue

            # 按文件名前缀分组
            file_groups = {}
            for file_path in files:
                # 提取文件名前缀（移除版本和日期）
                prefix = re.sub(r'_(v\d+|latest|final|draft).*$', '', file_path.stem)
                prefix = re.sub(r'_\d{8}.*$', '', prefix)

                if prefix not in file_groups:
                    file_groups[prefix] = []
                file_groups[prefix].append(file_path)

            # 合并相似文件
            for prefix, similar_files in file_groups.items():
                if len(similar_files) > 1:
                    # 按修改时间排序，保留最新的
                    similar_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                    # 保留最新的文件，归档其他文件
                    for file_path in similar_files[1:]:
                        self.archive_file(file_path)
                        print(f"合并相似报告: {file_path} -> {self.archive_dir}")

    def update_index_files(self):
        """更新索引文件"""
        index_file = self.reports_dir / "INDEX.md"
        if index_file.exists():
            # 读取现有索引
            with open(index_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 更新文件链接（移除日期）
            updated_content = re.sub(
                r'\[([^\]]+)\]\(([^)]+)_\d{8}[^)]*\.(md|json|html)\)',
                r'[\1](\2_latest.\3)',
                content
            )

            # 写回文件
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)

    def run_cleanup(self):
        """执行完整的清理流程"""
        print("开始报告清理和重命名...")

        # 1. 重命名文件（移除日期）
        print("\n1. 重命名文件（移除日期标识）...")
        renamed_count = 0
        for file_path in self.get_all_report_files():
            new_path = self.rename_file(file_path)
            if new_path != file_path:
                try:
                    file_path.rename(new_path)
                    print(f"重命名: {file_path.name} -> {new_path.name}")
                    renamed_count += 1
                except Exception as e:
                    print(f"重命名失败: {file_path.name} - {e}")

        print(f"重命名完成: {renamed_count} 个文件")

        # 2. 归档过期文件
        print("\n2. 归档过期文件...")
        archived_count = 0
        for file_path in self.get_all_report_files():
            if self.should_archive_file(file_path):
                try:
                    self.archive_file(file_path)
                    print(f"归档: {file_path.name}")
                    archived_count += 1
                except Exception as e:
                    print(f"归档失败: {file_path.name} - {e}")

        print(f"归档完成: {archived_count} 个文件")

        # 3. 限制每个目录的报告数量
        print("\n3. 限制目录报告数量...")
        self.limit_reports_per_directory()

        # 4. 合并相似报告
        print("\n4. 合并相似报告...")
        self.merge_similar_reports()

        # 5. 更新索引文件
        print("\n5. 更新索引文件...")
        self.update_index_files()

        print("\n报告清理完成！")

        # 输出统计信息
        total_files = len(self.get_all_report_files())
        archive_files = len(list(self.archive_dir.rglob("*")))
        print(f"\n统计信息:")
        print(f"- 当前报告文件: {total_files}")
        print(f"- 归档文件: {archive_files}")


def main():
    """主函数"""
    manager = ReportCleanupManager()
    manager.run_cleanup()


if __name__ == "__main__":
    main()
