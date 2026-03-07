#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础设施层日志依赖批量修复脚本
替换所有对引擎层日志的依赖为基础设施层专用日志
"""

import re
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


class LoggingDependencyFixer:
    """日志依赖修复器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"
        self.backup_dir = self.project_root / "backup" / "logging_dependency_fix"
        self.log_file = self.backup_dir / "fix_log.json"

        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 修复模式 - 引擎层到基础设施层
        self.fix_patterns = [
            # 引擎层日志导入
            (
                r'from src\.engine\.logging\.unified_logger import get_unified_logger',
                'from ..logging.infrastructure_logger import get_unified_logger'
            ),
            (
                r'from src\.engine\.logging\.unified_logger import get_unified_logger, _engine_loggers',
                'from ..logging.infrastructure_logger import get_unified_logger'
            ),
            (
                r'from src\.engine\.logging\.unified_logger import \*',
                'from ..logging.infrastructure_logger import get_unified_logger'
            ),
            # 引擎层日志上下文导入
            (
                r'from src\.engine\.logging\.unified_context import UnifiedLogContext',
                'from ..logging.infrastructure_logger import InfrastructureLogContext'
            ),
            (
                r'from src\.engine\.logging\.unified_context import UnifiedLogContext as LogContext',
                'from ..logging.infrastructure_logger import InfrastructureLogContext as LogContext'
            ),
            (
                r'from src\.engine\.logging\.unified_context import \*',
                'from ..logging.infrastructure_logger import InfrastructureLogContext'
            ),
            # 引擎层日志配置导入
            (
                r'from src\.engine\.logging\.config import \*',
                'from ..logging.infrastructure_logger import InfrastructureLogConfig'
            ),
            (
                r'from src\.engine\.logging\.config import LogConfig',
                'from ..logging.infrastructure_logger import InfrastructureLogConfig'
            ),
            # 类名替换
            (
                r'\bUnifiedLogContext\b',
                'InfrastructureLogContext'
            ),
            (
                r'\bLogConfig\b',
                'InfrastructureLogConfig'
            ),
            # 日志级别常量
            (
                r'from src\.engine\.logging\.levels import \*',
                'from ..logging.infrastructure_logger import LogLevel'
            ),
        ]

        # 验证模式 - 检查是否还有引擎层依赖
        self.validation_patterns = [
            r'from src\.engine\.logging',
            r'import src\.engine\.logging',
            r'\bUnifiedLogContext\b',
            r'\bLogConfig\b',
        ]

    def find_files_to_fix(self) -> List[Path]:
        """查找需要修复的文件"""
        files_to_fix = []

        # 查找基础设施层文件
        if self.infrastructure_dir.exists():
            for py_file in self.infrastructure_dir.rglob("*.py"):
                if py_file.name != "__pycache__":
                    files_to_fix.append(py_file)

        # 查找测试文件
        test_dir = self.project_root / "tests" / "unit" / "infrastructure"
        if test_dir.exists():
            for py_file in test_dir.rglob("*.py"):
                if py_file.name != "__pycache__":
                    files_to_fix.append(py_file)

        # 查找集成测试文件
        integration_test_dir = self.project_root / "tests" / "integration" / "infrastructure"
        if integration_test_dir.exists():
            for py_file in integration_test_dir.rglob("*.py"):
                if py_file.name != "__pycache__":
                    files_to_fix.append(py_file)

        return files_to_fix

    def backup_file(self, file_path: Path) -> Path:
        """备份文件"""
        relative_path = file_path.relative_to(self.project_root)
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        return backup_path

    def validate_file(self, file_path: Path) -> List[str]:
        """验证文件是否还有引擎层依赖"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            issues = []
            for pattern in self.validation_patterns:
                if re.search(pattern, content):
                    issues.append(f"发现引擎层依赖: {pattern}")

            return issues
        except Exception as e:
            return [f"验证失败: {str(e)}"]

    def fix_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """修复单个文件"""
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            changes = []

            # 应用修复模式
            for old_pattern, new_pattern in self.fix_patterns:
                if re.search(old_pattern, content):
                    content = re.sub(old_pattern, new_pattern, content)
                    changes.append(f"替换: {old_pattern} -> {new_pattern}")

            # 如果有变化，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, changes

            return False, []

        except Exception as e:
            return False, [f"修复失败: {str(e)}"]

    def fix_all_files(self) -> Dict[str, List[str]]:
        """修复所有文件"""
        files_to_fix = self.find_files_to_fix()
        results = {
            'fixed': [],
            'skipped': [],
            'errors': [],
            'validation_issues': []
        }

        print(f"找到 {len(files_to_fix)} 个文件需要检查")

        for file_path in files_to_fix:
            try:
                # 备份文件
                backup_path = self.backup_file(file_path)

                # 修复文件
                fixed, changes = self.fix_file(file_path)

                # 验证修复结果
                validation_issues = self.validate_file(file_path)

                if fixed:
                    results['fixed'].append({
                        'file': str(file_path),
                        'backup': str(backup_path),
                        'changes': changes,
                        'validation_issues': validation_issues
                    })
                    print(f"✅ 修复: {file_path}")
                    for change in changes:
                        print(f"   {change}")
                    if validation_issues:
                        print(f"   ⚠️  验证问题: {len(validation_issues)} 个")
                else:
                    results['skipped'].append(str(file_path))
                    print(f"⏭️  跳过: {file_path}")

                if validation_issues:
                    results['validation_issues'].append({
                        'file': str(file_path),
                        'issues': validation_issues
                    })

            except Exception as e:
                results['errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
                print(f"❌ 错误: {file_path} - {str(e)}")

        return results

    def rollback_file(self, file_path: Path, backup_path: Path) -> bool:
        """回滚单个文件"""
        try:
            if backup_path.exists():
                shutil.copy2(backup_path, file_path)
                return True
            return False
        except Exception:
            return False

    def rollback_all(self) -> Dict[str, List[str]]:
        """回滚所有修复"""
        results = {
            'rolled_back': [],
            'failed': []
        }

        # 读取修复日志
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

                for item in log_data.get('fixed_files', []):
                    file_path = Path(item['file'])
                    backup_path = Path(item['backup'])

                    if self.rollback_file(file_path, backup_path):
                        results['rolled_back'].append(str(file_path))
                        print(f"✅ 回滚: {file_path}")
                    else:
                        results['failed'].append(str(file_path))
                        print(f"❌ 回滚失败: {file_path}")

            except Exception as e:
                print(f"❌ 读取修复日志失败: {str(e)}")

        return results

    def save_log(self, results: Dict[str, List[str]]):
        """保存修复日志"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'fixed_files': results['fixed'],
            'skipped_files': results['skipped'],
            'error_files': results['errors']
        }

        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    def generate_report(self, results: Dict[str, List[str]]) -> str:
        """生成修复报告"""
        report = []
        report.append("# 基础设施层日志依赖修复报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 修复统计
        report.append("## 修复统计")
        report.append(f"- 修复文件数: {len(results['fixed'])}")
        report.append(f"- 跳过文件数: {len(results['skipped'])}")
        report.append(f"- 错误文件数: {len(results['errors'])}")
        report.append(f"- 验证问题数: {len(results['validation_issues'])}")
        report.append("")

        # 修复的文件
        if results['fixed']:
            report.append("## 修复的文件")
            for item in results['fixed']:
                report.append(f"### {item['file']}")
                report.append(f"- 备份: {item['backup']}")
                for change in item['changes']:
                    report.append(f"- {change}")
                if item['validation_issues']:
                    report.append("- 验证问题:")
                    for issue in item['validation_issues']:
                        report.append(f"  - {issue}")
                report.append("")

        # 验证问题
        if results['validation_issues']:
            report.append("## 验证问题")
            for item in results['validation_issues']:
                report.append(f"### {item['file']}")
                for issue in item['issues']:
                    report.append(f"- {issue}")
                report.append("")

        # 错误的文件
        if results['errors']:
            report.append("## 修复失败的文件")
            for item in results['errors']:
                report.append(f"- {item['file']}: {item['error']}")
            report.append("")

        # 建议
        report.append("## 后续建议")
        report.append("1. 运行测试验证修复效果")
        report.append("2. 检查内存使用情况")
        report.append("3. 验证日志功能正常")
        report.append("4. 手动检查验证问题")
        report.append("5. 如有问题可运行回滚命令")

        return "\n".join(report)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="修复基础设施层日志依赖")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--dry-run", action="store_true", help="仅检查，不实际修改")
    parser.add_argument("--report", help="输出报告文件路径")
    parser.add_argument("--rollback", action="store_true", help="回滚所有修复")
    parser.add_argument("--validate", action="store_true", help="仅验证，不修复")

    args = parser.parse_args()

    # 创建修复器
    fixer = LoggingDependencyFixer(args.project_root)

    if args.rollback:
        print("🔄 开始回滚所有修复...")
        results = fixer.rollback_all()
        print(f"\n✅ 回滚完成！共回滚 {len(results['rolled_back'])} 个文件")
        if results['failed']:
            print(f"❌ 回滚失败 {len(results['failed'])} 个文件")
        return

    if args.dry_run:
        print("🔍 检查模式 - 不会实际修改文件")
        files_to_fix = fixer.find_files_to_fix()
        print(f"找到 {len(files_to_fix)} 个文件需要检查:")
        for file_path in files_to_fix:
            print(f"  - {file_path}")
    elif args.validate:
        print("🔍 验证模式 - 检查引擎层依赖")
        files_to_fix = fixer.find_files_to_fix()
        total_issues = 0
        for file_path in files_to_fix:
            issues = fixer.validate_file(file_path)
            if issues:
                print(f"⚠️  {file_path}:")
                for issue in issues:
                    print(f"   {issue}")
                total_issues += len(issues)
        print(f"\n发现 {total_issues} 个验证问题")
    else:
        print("🔧 开始修复基础设施层日志依赖...")
        results = fixer.fix_all_files()

        # 保存修复日志
        fixer.save_log(results)

        # 生成报告
        report = fixer.generate_report(results)

        # 输出报告
        if args.report:
            with open(args.report, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 报告已保存到: {args.report}")
        else:
            print("\n" + "="*50)
            print(report)

        print(f"\n✅ 修复完成！共修复 {len(results['fixed'])} 个文件")
        print(f"📝 修复日志已保存到: {fixer.log_file}")


if __name__ == "__main__":
    main()
