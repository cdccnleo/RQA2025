#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础设施层优化脚本
自动修复审查中发现的问题，提升代码质量和架构合规性
"""

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


class InfrastructureOptimizer:
    """基础设施层优化器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"
        self.backup_dir = self.project_root / "backup" / "infrastructure_optimization"
        self.results = {
            'files_processed': 0,
            'issues_fixed': 0,
            'files_backed_up': 0,
            'optimizations': []
        }

        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 优化模式
        self.optimization_patterns = [
            # 编码声明修复
            (r'^#!.*\n', r'#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n'),
            # 移除多余空行
            (r'\n{3,}', '\n\n'),
            # 修复行长度
            (r'(.{120,})', lambda m: self._break_long_line(m.group(1))),
            # 修复导入语句格式
            (r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+([a-zA-Z_][a-zA-Z0-9_,\s]*)',
             lambda m: self._format_import(m.group(1), m.group(2))),
        ]

        # 禁止的依赖替换
        self.dependency_fixes = [
            (r'from src\.engine\.logging\.unified_logger import get_unified_logger',
             'from ..logging.infrastructure_logger import get_unified_logger'),
            (r'from src\.engine\.logging\.unified_context import UnifiedLogContext',
             'from ..logging.infrastructure_logger import InfrastructureLogContext'),
            (r'from src\.engine\.logging\.config import \*',
             'from ..logging.infrastructure_logger import InfrastructureLogConfig'),
        ]

    def backup_file(self, file_path: Path) -> Path:
        """备份文件"""
        relative_path = file_path.relative_to(self.project_root)
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        return backup_path

    def _break_long_line(self, line: str) -> str:
        """分割长行"""
        if len(line) <= 120:
            return line

        # 尝试在操作符处分割
        operators = [' + ', ' - ', ' * ', ' / ', ' // ',
                     ' % ', ' ** ', ' == ', ' != ', ' <= ', ' >= ']
        for op in operators:
            if op in line:
                parts = line.split(op)
                if len(parts) == 2:
                    return f"{parts[0]}{op}\n    {parts[1]}"

        # 尝试在逗号处分割
        if ',' in line:
            parts = line.split(',')
            if len(parts) == 2:
                return f"{parts[0]},\n    {parts[1]}"

        # 强制分割
        return line[:120] + '\n    ' + line[120:]

    def _format_import(self, module: str, imports: str) -> str:
        """格式化导入语句"""
        # 移除多余空格
        imports = re.sub(r'\s+', ' ', imports.strip())

        # 如果导入项过多，分行显示
        if len(imports) > 80:
            items = [item.strip() for item in imports.split(',')]
            items_str = ',\n    '.join(items)
            return f"from {module} import (\n    {items_str}\n)"
        else:
            return f"from {module} import {imports}"

    def add_docstring(self, content: str, file_path: Path) -> str:
        """添加文档字符串"""
        lines = content.split('\n')

        # 检查是否已有文档字符串
        for i, line in enumerate(lines[:10]):
            if '"""' in line or "'''" in line:
                return content

        # 生成模块文档字符串
        module_name = file_path.stem
        docstring = f'"""{module_name} 模块\n\n此模块提供基础设施层相关功能。\n"""'

        # 在编码声明后添加文档字符串
        if lines and lines[0].startswith('#!'):
            lines.insert(2, docstring)
        else:
            lines.insert(0, docstring)

        return '\n'.join(lines)

    def fix_code_style(self, content: str) -> str:
        """修复代码风格"""
        # 应用优化模式
        for pattern, replacement in self.optimization_patterns:
            if callable(replacement):
                content = re.sub(pattern, replacement, content)
            else:
                content = re.sub(pattern, replacement, content)

        return content

    def fix_dependencies(self, content: str) -> str:
        """修复依赖关系"""
        for old_pattern, new_pattern in self.dependency_fixes:
            content = re.sub(old_pattern, new_pattern, content)
        return content

    def optimize_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """优化单个文件"""
        try:
            # 备份文件
            backup_path = self.backup_file(file_path)
            self.results['files_backed_up'] += 1

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            changes = []

            # 添加文档字符串
            if not content.startswith('"""') and not content.startswith("'''"):
                content = self.add_docstring(content, file_path)
                changes.append("添加模块文档字符串")

            # 修复代码风格
            content = self.fix_code_style(content)
            if content != original_content:
                changes.append("修复代码风格")

            # 修复依赖关系
            content = self.fix_dependencies(content)
            if content != original_content:
                changes.append("修复依赖关系")

            # 如果有变化，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, changes

            return False, []

        except Exception as e:
            return False, [f"优化失败: {str(e)}"]

    def create_missing_modules(self) -> List[str]:
        """创建缺失的标准模块"""
        print("🔧 创建缺失的标准模块...")
        created_modules = []

        standard_modules = {
            'logging': {
                '__init__.py': '"""日志管理模块"""',
                'infrastructure_logger.py': '"""基础设施层日志器"""'
            },
            'config': {
                '__init__.py': '"""配置管理模块"""',
                'infrastructure_config.py': '"""基础设施层配置"""'
            },
            'database': {
                '__init__.py': '"""数据库管理模块"""',
                'infrastructure_database.py': '"""基础设施层数据库"""'
            },
            'cache': {
                '__init__.py': '"""缓存管理模块"""',
                'infrastructure_cache.py': '"""基础设施层缓存"""'
            },
            'messaging': {
                '__init__.py': '"""消息队列模块"""',
                'infrastructure_messaging.py': '"""基础设施层消息队列"""'
            },
            'monitoring': {
                '__init__.py': '"""监控管理模块"""',
                'infrastructure_monitoring.py': '"""基础设施层监控"""'
            },
            'security': {
                '__init__.py': '"""安全管理模块"""',
                'infrastructure_security.py': '"""基础设施层安全"""'
            },
            'utils': {
                '__init__.py': '"""工具函数模块"""',
                'infrastructure_utils.py': '"""基础设施层工具函数"""'
            }
        }

        for module_name, files in standard_modules.items():
            module_dir = self.infrastructure_dir / module_name
            if not module_dir.exists():
                module_dir.mkdir(parents=True, exist_ok=True)
                created_modules.append(module_name)

                for file_name, content in files.items():
                    file_path = module_dir / file_name
                    if not file_path.exists():
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(
                                f'#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n{content}\n')

        return created_modules

    def optimize_all_files(self) -> Dict:
        """优化所有文件"""
        print("🔧 开始优化基础设施层文件...")

        if not self.infrastructure_dir.exists():
            print("❌ 基础设施层目录不存在")
            return self.results

        # 创建缺失的模块
        created_modules = self.create_missing_modules()
        if created_modules:
            print(f"✅ 创建了 {len(created_modules)} 个缺失模块: {', '.join(created_modules)}")

        # 优化现有文件
        for py_file in self.infrastructure_dir.rglob("*.py"):
            if py_file.name != "__pycache__":
                try:
                    optimized, changes = self.optimize_file(py_file)

                    if optimized:
                        self.results['files_processed'] += 1
                        self.results['issues_fixed'] += len(changes)
                        self.results['optimizations'].append({
                            'file': str(py_file),
                            'changes': changes
                        })
                        print(f"✅ 优化: {py_file}")
                        for change in changes:
                            print(f"   {change}")
                    else:
                        print(f"⏭️  跳过: {py_file}")

                except Exception as e:
                    print(f"❌ 优化失败 {py_file}: {str(e)}")

        return self.results

    def generate_optimization_report(self) -> str:
        """生成优化报告"""
        report = []
        report.append("# 基础设施层优化报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 优化统计
        report.append("## 优化统计")
        report.append(f"- 处理文件数: {self.results['files_processed']}")
        report.append(f"- 修复问题数: {self.results['issues_fixed']}")
        report.append(f"- 备份文件数: {self.results['files_backed_up']}")
        report.append("")

        # 优化详情
        if self.results['optimizations']:
            report.append("## 优化详情")
            for opt in self.results['optimizations']:
                report.append(f"### {opt['file']}")
                for change in opt['changes']:
                    report.append(f"- {change}")
                report.append("")

        # 建议
        report.append("## 后续建议")
        report.append("1. 运行测试验证优化效果")
        report.append("2. 检查代码质量和架构合规性")
        report.append("3. 完善模块文档和测试用例")
        report.append("4. 定期运行优化脚本保持代码质量")

        return "\n".join(report)

    def rollback_optimizations(self) -> Dict:
        """回滚优化"""
        print("🔄 开始回滚优化...")
        results = {
            'rolled_back': 0,
            'failed': 0
        }

        if not self.backup_dir.exists():
            print("❌ 备份目录不存在，无法回滚")
            return results

        # 恢复备份文件
        for backup_file in self.backup_dir.rglob("*.py"):
            try:
                relative_path = backup_file.relative_to(self.backup_dir)
                original_file = self.project_root / relative_path

                if original_file.exists():
                    shutil.copy2(backup_file, original_file)
                    results['rolled_back'] += 1
                    print(f"✅ 回滚: {original_file}")
                else:
                    print(f"⚠️  原文件不存在: {original_file}")

            except Exception as e:
                results['failed'] += 1
                print(f"❌ 回滚失败 {backup_file}: {str(e)}")

        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化基础设施层")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--report", help="输出报告文件路径")
    parser.add_argument("--rollback", action="store_true", help="回滚所有优化")
    parser.add_argument("--dry-run", action="store_true", help="仅检查，不实际修改")

    args = parser.parse_args()

    # 创建优化器
    optimizer = InfrastructureOptimizer(args.project_root)

    if args.rollback:
        print("🔄 开始回滚优化...")
        results = optimizer.rollback_optimizations()
        print(f"\n✅ 回滚完成！共回滚 {results['rolled_back']} 个文件")
        if results['failed']:
            print(f"❌ 回滚失败 {results['failed']} 个文件")
        return

    if args.dry_run:
        print("🔍 检查模式 - 不会实际修改文件")
        if optimizer.infrastructure_dir.exists():
            files = list(optimizer.infrastructure_dir.rglob("*.py"))
            print(f"找到 {len(files)} 个文件需要检查:")
            for file_path in files:
                print(f"  - {file_path}")
    else:
        print("🔧 开始优化基础设施层...")
        results = optimizer.optimize_all_files()

        # 生成报告
        report = optimizer.generate_optimization_report()

        # 输出报告
        if args.report:
            with open(args.report, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 报告已保存到: {args.report}")
        else:
            print("\n" + "="*50)
            print(report)

        print(f"\n✅ 优化完成！共处理 {results['files_processed']} 个文件，修复 {results['issues_fixed']} 个问题")
        print(f"📝 备份文件保存在: {optimizer.backup_dir}")


if __name__ == "__main__":
    main()
