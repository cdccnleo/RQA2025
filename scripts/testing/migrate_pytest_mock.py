#!/usr/bin/env python3
"""
pytest-mock迁移脚本
将现有的pytest-mock用法替换为Python标准库unittest.mock
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict


class PytestMockMigrator:
    """pytest-mock迁移器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.tests_dir = self.project_root / "tests"

    def find_mock_files(self) -> List[Path]:
        """查找使用pytest-mock的文件"""
        mock_files = []

        for test_file in self.tests_dir.rglob("test_*.py"):
            content = test_file.read_text(encoding='utf-8')
            if 'mocker' in content or 'pytest_mock' in content:
                mock_files.append(test_file)

        return mock_files

    def analyze_mock_usage(self, file_path: Path) -> Dict[str, List[str]]:
        """分析文件中的mock使用情况"""
        content = file_path.read_text(encoding='utf-8')

        # 查找mocker参数
        mocker_params = re.findall(r'def test_\w+\([^)]*mocker[^)]*\):', content)

        # 查找mocker调用
        mocker_calls = re.findall(r'mocker\.\w+\([^)]*\)', content)

        # 查找patch调用
        patch_calls = re.findall(r'mocker\.patch\([^)]*\)', content)

        return {
            'mocker_params': mocker_params,
            'mocker_calls': mocker_calls,
            'patch_calls': patch_calls
        }

    def generate_migration_suggestions(self, file_path: Path) -> List[str]:
        """生成迁移建议"""
        suggestions = []
        usage = self.analyze_mock_usage(file_path)

        if usage['mocker_params']:
            suggestions.append(f"文件 {file_path.name} 使用了mocker参数，建议替换为:")
            suggestions.append("  from unittest.mock import Mock, patch, MagicMock")
            suggestions.append("  # 移除mocker参数，使用patch装饰器或上下文管理器")

        if usage['patch_calls']:
            suggestions.append("发现patch调用，建议替换为:")
            suggestions.append("  with patch('module.function') as mock_func:")
            suggestions.append("      mock_func.return_value = expected_value")
            suggestions.append("      # 执行测试")
            suggestions.append("      mock_func.assert_called_once()")

        return suggestions

    def create_migration_example(self, file_path: Path) -> str:
        """创建迁移示例"""
        content = file_path.read_text(encoding='utf-8')

        # 简单的替换示例
        example = f"""
# 原代码 (使用pytest-mock):
def test_example(mocker):
    mocker.patch('src.data.loader.stock_loader.akshare.stock_zh_a_hist', return_value=mock_data)
    loader = StockLoader()
    result = loader.load_data('000001')
    assert result is not None

# 迁移后 (使用标准库mock):
from unittest.mock import Mock, patch, MagicMock

def test_example():
    with patch('src.data.loader.stock_loader.akshare.stock_zh_a_hist') as mock_api:
        mock_api.return_value = mock_data
        loader = StockLoader()
        result = loader.load_data('000001')
        assert result is not None
        mock_api.assert_called_once()
"""
        return example

    def generate_migration_report(self) -> str:
        """生成迁移报告"""
        mock_files = self.find_mock_files()

        report = ["# pytest-mock迁移报告\n"]
        report.append(f"发现 {len(mock_files)} 个使用pytest-mock的文件\n")

        for file_path in mock_files:
            report.append(f"## {file_path.relative_to(self.project_root)}")
            usage = self.analyze_mock_usage(file_path)

            report.append(f"- mocker参数使用: {len(usage['mocker_params'])} 处")
            report.append(f"- mocker调用: {len(usage['mocker_calls'])} 处")
            report.append(f"- patch调用: {len(usage['patch_calls'])} 处")

            suggestions = self.generate_migration_suggestions(file_path)
            if suggestions:
                report.append("### 迁移建议:")
                report.extend(suggestions)

            report.append("")  # 空行分隔

        return "\n".join(report)

    def create_migration_script(self, file_path: Path) -> str:
        """为特定文件创建迁移脚本"""
        content = file_path.read_text(encoding='utf-8')

        # 基本替换规则
        replacements = [
            # 移除mocker参数
            (r'def test_\w+\([^)]*mocker[^)]*\):',
             lambda m: re.sub(r',?\s*mocker', '', m.group(0))),

            # 替换mocker.patch为patch
            (r'mocker\.patch\(', 'patch('),

            # 添加必要的导入
            (r'import pytest', 'import pytest\nfrom unittest.mock import Mock, patch, MagicMock'),
        ]

        migrated_content = content
        for pattern, replacement in replacements:
            if callable(replacement):
                migrated_content = re.sub(pattern, replacement, migrated_content)
            else:
                migrated_content = migrated_content.replace(pattern, replacement)

        return migrated_content


def main():
    parser = argparse.ArgumentParser(description='pytest-mock迁移工具')
    parser.add_argument('--project-root', default='.', help='项目根目录')
    parser.add_argument('--report', action='store_true', help='生成迁移报告')
    parser.add_argument('--migrate-file', help='迁移特定文件')
    parser.add_argument('--dry-run', action='store_true', help='仅显示迁移建议，不实际修改')

    args = parser.parse_args()

    migrator = PytestMockMigrator(args.project_root)

    if args.report:
        report = migrator.generate_migration_report()
        print(report)

        # 保存报告
        report_file = Path(args.project_root) / "pytest_mock_migration_report.md"
        report_file.write_text(report, encoding='utf-8')
        print(f"\n迁移报告已保存到: {report_file}")

    elif args.migrate_file:
        file_path = Path(args.project_root) / args.migrate_file
        if not file_path.exists():
            print(f"文件不存在: {file_path}")
            return

        if args.dry_run:
            suggestions = migrator.generate_migration_suggestions(file_path)
            print(f"\n文件 {file_path.name} 的迁移建议:")
            for suggestion in suggestions:
                print(f"  {suggestion}")
        else:
            migrated_content = migrator.create_migration_script(file_path)

            # 备份原文件
            backup_file = file_path.with_suffix('.py.bak')
            file_path.rename(backup_file)

            # 写入迁移后的内容
            file_path.write_text(migrated_content, encoding='utf-8')

            print(f"文件 {file_path} 已迁移")
            print(f"原文件备份为: {backup_file}")

    else:
        # 默认行为：显示所有需要迁移的文件
        mock_files = migrator.find_mock_files()
        print(f"发现 {len(mock_files)} 个使用pytest-mock的文件:")

        for file_path in mock_files:
            print(f"  {file_path.relative_to(args.project_root)}")

        print(f"\n使用 --report 生成详细迁移报告")
        print(f"使用 --migrate-file <file> 迁移特定文件")


if __name__ == "__main__":
    main()
