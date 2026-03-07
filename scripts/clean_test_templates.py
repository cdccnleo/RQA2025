#!/usr/bin/env python3
"""
基础设施层测试文件清理脚本

功能：
- 识别并清理空的测试模板文件
- 分析测试文件的有效性
- 安全删除无效文件并生成报告
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import shutil
from datetime import datetime


class TestFileCleaner:
    """测试文件清理器"""

    def __init__(self, test_root: str = "tests/unit/infrastructure"):
        self.test_root = Path(test_root)
        self.backup_dir = Path("test_files_backup") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.deleted_files: List[Path] = []
        self.analysis_results: Dict[str, any] = {}

    def analyze_file_content(self, file_path: Path) -> Dict[str, any]:
        """分析文件内容的有效性"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                'valid': False,
                'reason': f'无法读取文件: {e}',
                'size': file_path.stat().st_size,
                'type': 'unreadable'
            }

        analysis = {
            'size': len(content),
            'lines': len(content.split('\n')),
            'has_test_functions': bool(re.search(r'def test_\w+', content)),
            'has_test_classes': bool(re.search(r'class Test\w+', content)),
            'has_imports': bool(re.search(r'^(import|from).*', content, re.MULTILINE)),
            'has_skip_decorators': '@pytest.mark.skip' in content,
            'has_skip_calls': 'pytest.skip(' in content,
            'has_docstrings': '"""' in content or "'''" in content,
            'has_assertions': 'assert ' in content,
            'has_comments': '#' in content,
            'is_empty_or_whitespace': not content.strip(),
            'is_template_like': self._is_template_like(content)
        }

        # 判断文件是否应该被删除
        analysis['should_delete'] = self._should_delete_file(analysis, content)

        return analysis

    def _is_template_like(self, content: str) -> bool:
        """判断文件是否像是模板文件"""
        template_indicators = [
            'TODO: Implement test',
            'NotImplementedError',
            '# TODO',
            'pass  # TODO',
            'raise NotImplementedError',
            'def test_',
            'class Test',
            '# Write your test here'
        ]

        # 如果文件很小且只包含基本结构，很可能是模板
        if len(content.strip()) < 200:
            has_basic_structure = any(indicator in content for indicator in template_indicators[:4])
            has_minimal_content = len(content.split('\n')) < 10
            if has_basic_structure and has_minimal_content:
                return True

        # 如果只有类定义和pass语句，很可能是模板
        if re.search(r'class Test\w+:', content) and not re.search(r'def test_\w+', content):
            if 'pass' in content and len(content.split('\n')) < 15:
                return True

        return False

    def _should_delete_file(self, analysis: Dict[str, any], content: str) -> bool:
        """判断文件是否应该被删除"""

        # 完全空的或只有空白的文件
        if analysis['is_empty_or_whitespace']:
            return True

        # 模板文件：有类定义但没有实际测试内容
        if (analysis['has_test_classes'] and
            not analysis['has_test_functions'] and
            not analysis['has_assertions'] and
            analysis['is_template_like']):
            return True

        # 只有导入和基本结构的空文件
        if (analysis['size'] < 300 and
            analysis['has_imports'] and
            not analysis['has_test_functions'] and
            not analysis['has_assertions'] and
            not analysis['has_docstrings']):
            return True

        # 只有类定义和pass的文件
        if (re.search(r'class Test\w+:', content) and
            re.search(r'^\s*pass\s*$', content, re.MULTILINE) and
            not re.search(r'def test_\w+', content) and
            len(content.split('\n')) < 20):
            return True

        # 自动生成的模板文件：包含特定导入模式的空文件
        if (analysis['size'] < 1000 and
            not analysis['has_test_functions'] and
            not analysis['has_assertions'] and
            '自动修复导入问题' in content and
            'project_root = Path(__file__).resolve()' in content and
            'src_path_st' in content):
            return True

        # 只有基本导入和注释的文件
        if (analysis['size'] < 500 and
            not analysis['has_test_functions'] and
            not analysis['has_assertions'] and
            analysis['lines'] < 20 and
            not analysis['has_docstrings']):
            return True

        return False

    def create_backup(self, files_to_backup: List[Path]):
        """创建备份"""
        if not files_to_backup:
            return

        print(f"📦 创建备份目录: {self.backup_dir}")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files_to_backup:
            # 保持相对路径结构
            relative_path = file_path.relative_to(self.test_root)
            backup_path = self.backup_dir / relative_path

            # 创建备份目录
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # 复制文件
            shutil.copy2(file_path, backup_path)

        print(f"✅ 已备份 {len(files_to_backup)} 个文件到 {self.backup_dir}")

    def delete_files(self, files_to_delete: List[Path], create_backup: bool = True) -> int:
        """删除文件"""
        if not files_to_delete:
            print("ℹ️ 没有找到需要删除的文件")
            return 0

        if create_backup:
            self.create_backup(files_to_delete)

        deleted_count = 0
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                self.deleted_files.append(file_path)
                deleted_count += 1
                print(f"🗑️ 删除: {file_path}")
            except Exception as e:
                print(f"❌ 删除失败 {file_path}: {e}")

        return deleted_count

    def analyze_all_files(self) -> Dict[str, any]:
        """分析所有测试文件"""
        print("🔍 开始分析测试文件...")

        all_files = []
        valid_test_files = []
        template_files = []
        empty_files = []
        other_files = []

        # 收集所有Python文件
        for py_file in self.test_root.rglob('*.py'):
            if py_file.name in ['conftest.py', '__init__.py']:
                continue

            analysis = self.analyze_file_content(py_file)
            analysis['path'] = py_file

            all_files.append(analysis)

            if analysis['should_delete']:
                if analysis['is_empty_or_whitespace']:
                    empty_files.append(py_file)
                else:
                    template_files.append(py_file)
            elif analysis['has_test_functions'] and not (analysis['has_skip_decorators'] or analysis['has_skip_calls']):
                valid_test_files.append(py_file)
            else:
                other_files.append(py_file)

        results = {
            'total_files': len(all_files),
            'valid_test_files': len(valid_test_files),
            'template_files': len(template_files),
            'empty_files': len(empty_files),
            'other_files': len(other_files),
            'files_to_delete': template_files + empty_files,
            'valid_test_files_list': valid_test_files,
            'template_files_list': template_files,
            'empty_files_list': empty_files,
            'detailed_analysis': all_files
        }

        self.analysis_results = results
        return results

    def generate_report(self) -> str:
        """生成清理报告"""
        if not self.analysis_results:
            return "尚未进行文件分析"

        results = self.analysis_results

        report = f"""
# 测试文件清理报告

## 📊 分析结果

- **总文件数**: {results['total_files']}
- **有效测试文件**: {results['valid_test_files']} ({results['valid_test_files']/results['total_files']*100:.1f}%)
- **模板文件**: {results['template_files']} ({results['template_files']/results['total_files']*100:.1f}%)
- **空文件**: {results['empty_files']} ({results['empty_files']/results['total_files']*100:.1f}%)
- **其他文件**: {results['other_files']} ({results['other_files']/results['total_files']*100:.1f}%)

## 🗑️ 待删除文件

**模板文件 ({len(results['template_files_list'])} 个):**
"""

        for file_path in results['template_files_list'][:20]:  # 只显示前20个
            report += f"- {file_path.relative_to(self.test_root)}\n"

        if len(results['template_files_list']) > 20:
            report += f"- ... 还有 {len(results['template_files_list']) - 20} 个文件\n"

        report += f"""
**空文件 ({len(results['empty_files_list'])} 个):**
"""
        for file_path in results['empty_files_list'][:10]:  # 只显示前10个
            report += f"- {file_path.relative_to(self.test_root)}\n"

        if len(results['empty_files_list']) > 10:
            report += f"- ... 还有 {len(results['empty_files_list']) - 10} 个文件\n"

        report += f"""
## ✅ 保留的有效文件

**有效测试文件 ({len(results['valid_test_files_list'])} 个):**
"""
        for file_path in results['valid_test_files_list'][:15]:  # 只显示前15个
            report += f"- {file_path.relative_to(self.test_root)}\n"

        if len(results['valid_test_files_list']) > 15:
            report += f"- ... 还有 {len(results['valid_test_files_list']) - 15} 个文件\n"

        report += f"""
## 📋 执行建议

1. **备份文件**: 建议先备份待删除文件
2. **逐步删除**: 可以分批删除，避免误删重要文件
3. **验证删除**: 删除后运行测试确保没有破坏
4. **监控效果**: 观察测试收集数量的变化

## ⚠️ 重要提醒

- 删除操作不可逆，请确保备份
- 建议在删除前人工检查部分文件
- 删除后立即运行测试验证
- 如有疑问，可以恢复备份文件

---
*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        return report

    def save_report(self, report: str, filename: str = "test_cleanup_report.md"):
        """保存报告"""
        report_path = Path("test_logs") / filename
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 报告已保存到: {report_path}")
        return report_path

    def execute_cleanup(self, dry_run: bool = True, create_backup: bool = True) -> Dict[str, any]:
        """执行清理操作"""
        print("🧹 开始执行测试文件清理...")

        # 分析文件
        results = self.analyze_all_files()

        files_to_delete = results['files_to_delete']

        print(f"📊 分析完成: 发现 {len(files_to_delete)} 个可删除文件")

        if dry_run:
            print("🔍 这是预览模式，不会实际删除文件")
            print("要执行实际删除，请设置 dry_run=False")
        else:
            print("⚠️ 即将执行实际删除操作!")
            confirm = input("确认要删除这些文件吗? (yes/no): ")
            if confirm.lower() != 'yes':
                print("❌ 操作已取消")
                return results

        # 显示将要删除的文件
        print("\n🗑️ 将要删除的文件:")
        for i, file_path in enumerate(files_to_delete[:10], 1):
            print(f"  {i}. {file_path.relative_to(self.test_root)}")
        if len(files_to_delete) > 10:
            print(f"  ... 还有 {len(files_to_delete) - 10} 个文件")

        if not dry_run:
            deleted_count = self.delete_files(files_to_delete, create_backup)
            print(f"✅ 成功删除 {deleted_count} 个文件")
        else:
            print(f"📋 预览: 将删除 {len(files_to_delete)} 个文件")

        # 生成报告
        report = self.generate_report()
        self.save_report(report)

        return {
            'analysis': results,
            'deleted_files': self.deleted_files if not dry_run else [],
            'dry_run': dry_run,
            'backup_dir': str(self.backup_dir) if create_backup and not dry_run else None
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="基础设施层测试文件清理工具")
    parser.add_argument("--analyze", action="store_true", help="仅分析文件，不执行删除")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，显示将要删除的文件")
    parser.add_argument("--execute", action="store_true", help="执行实际删除操作")
    parser.add_argument("--no-backup", action="store_true", help="不创建备份")
    parser.add_argument("--report-only", action="store_true", help="只生成报告")

    args = parser.parse_args()

    cleaner = TestFileCleaner()

    if args.report_only:
        # 只生成报告
        results = cleaner.analyze_all_files()
        report = cleaner.generate_report()
        report_path = cleaner.save_report(report)
        print(f"报告已生成: {report_path}")
        return

    if args.analyze:
        # 只分析
        results = cleaner.analyze_all_files()
        print("\n📊 分析结果:")
        print(f"  总文件数: {results['total_files']}")
        print(f"  有效测试文件: {results['valid_test_files']}")
        print(f"  模板文件: {results['template_files']}")
        print(f"  空文件: {results['empty_files']}")
        print(f"  其他文件: {results['other_files']}")
        print(f"  可删除文件: {len(results['files_to_delete'])}")

        report = cleaner.generate_report()
        cleaner.save_report(report)
        return

    if args.execute:
        # 执行删除
        results = cleaner.execute_cleanup(
            dry_run=False,
            create_backup=not args.no_backup
        )
        return

    # 默认是预览模式
    results = cleaner.execute_cleanup(
        dry_run=True,
        create_backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
