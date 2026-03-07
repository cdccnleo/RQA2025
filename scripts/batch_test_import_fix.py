#!/usr/bin/env python3
"""
批量测试导入修复工具

批量修复测试文件中的导入路径问题
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict


class BatchTestImportFixer:
    """批量测试导入修复工具"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.backup_dir = self.project_root / \
            f"backup/batch_test_import_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 创建扩展的导入路径映射
        self._build_extended_import_mapping()

    def _build_extended_import_mapping(self):
        """构建扩展的导入路径映射"""

        # 基础映射 - 基于分析结果
        self.import_mapping = {
            # 通用工具映射
            r'from src\.utils\.': r'from src.infrastructure.utils.',
            r'from src\.config\.': r'from src.infrastructure.config.',
            r'from src\.common\.': r'from src.infrastructure.utils.',
            r'from src\.helpers\.': r'from src.infrastructure.utils.',

            # 基础设施层映射
            r'from src\.infrastructure\.core\.': r'from src.infrastructure.',
            r'from src\.infrastructure\.base\.': r'from src.infrastructure.',

            # 数据层映射
            r'from src\.data\.core\.': r'from src.data.',
            r'from src\.data\.base\.': r'from src.data.',

            # 特征层映射
            r'from src\.features\.core\.': r'from src.features.',
            r'from src\.features\.base\.': r'from src.features.',

            # 模型层映射
            r'from src\.ml\.core\.': r'from src.ml.',
            r'from src\.ml\.base\.': r'from src.ml.',

            # 策略层映射
            r'from src\.core\.core\.': r'from src.core.',
            r'from src\.core\.base\.': r'from src.core.',

            # 风控层映射
            r'from src\.risk\.core\.': r'from src.risk.',
            r'from src\.risk\.base\.': r'from src.risk.',

            # 交易层映射
            r'from src\.trading\.core\.': r'from src.trading.',
            r'from src\.trading\.base\.': r'from src.trading.',

            # 回测层映射
            r'from src\.backtest\.core\.': r'from src.backtest.',
            r'from src\.backtest\.base\.': r'from src.backtest.',

            # 引擎层映射
            r'from src\.engine\.core\.': r'from src.engine.',
            r'from src\.engine\.base\.': r'from src.engine.',

            # API网关层映射
            r'from src\.gateway\.core\.': r'from src.gateway.',
            r'from src\.gateway\.base\.': r'from src.gateway.',

            # 具体模块映射
            r'from src\.models\.': r'from src.ml.models.',
            r'from src\.ensemble\.': r'from src.ml.ensemble.',
            r'from src\.monitoring\.': r'from src.infrastructure.monitoring.',
            r'from src\.integration\.': r'from src.core.integration.',
            r'from src\.services\.': r'from src.infrastructure.services.',
            r'from src\.adapters\.': r'from src.data.adapters.',

            # 导入语句映射
            r'import src\.utils': r'import src.infrastructure.utils',
            r'import src\.config': r'import src.infrastructure.config',
            r'import src\.common': r'import src.infrastructure.utils',
            r'import src\.models': r'import src.ml.models',
            r'import src\.ensemble': r'import src.ml.ensemble',
            r'import src\.monitoring': r'import src.infrastructure.monitoring',
            r'import src\.integration': r'import src.core.integration',
            r'import src\.services': r'import src.infrastructure.services',
            r'import src\.adapters': r'import src.data.adapters',
        }

    def find_broken_imports_batch(self, batch_size: int = 100) -> Dict[str, Any]:
        """批量查找损坏的导入"""

        analysis = {
            "total_files": 0,
            "files_with_broken_imports": 0,
            "broken_imports": [],
            "fixable_imports": [],
            "unfixable_imports": [],
            "import_pattern_stats": defaultdict(int)
        }

        print(f"🔍 批量分析测试文件导入（批次大小：{batch_size}）...")

        # 获取所有测试文件
        test_files = []
        for root, dirs, files in os.walk(self.tests_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    test_files.append(Path(root) / file)

        analysis["total_files"] = len(test_files)
        print(f"📋 发现 {len(test_files)} 个测试文件")

        # 分批处理
        for i in range(0, len(test_files), batch_size):
            batch = test_files[i:i + batch_size]
            print(f"📦 处理批次 {i//batch_size + 1}/{(len(test_files) + batch_size - 1)//batch_size}")

            for file_path in batch:
                try:
                    broken_imports = self._analyze_file_imports(file_path)
                    if broken_imports:
                        analysis["files_with_broken_imports"] += 1

                        for import_stmt in broken_imports:
                            analysis["broken_imports"].append({
                                "file": str(file_path.relative_to(self.project_root)),
                                "import": import_stmt
                            })

                            # 分类导入模式
                            pattern = self._classify_import_pattern(import_stmt)
                            analysis["import_pattern_stats"][pattern] += 1

                            # 检查是否可以修复
                            if self._can_fix_import(import_stmt):
                                analysis["fixable_imports"].append({
                                    "file": str(file_path.relative_to(self.project_root)),
                                    "import": import_stmt
                                })
                            else:
                                analysis["unfixable_imports"].append({
                                    "file": str(file_path.relative_to(self.project_root)),
                                    "import": import_stmt
                                })

                except Exception as e:
                    print(f"❌ 处理文件失败 {file_path}: {e}")

        return analysis

    def _analyze_file_imports(self, file_path: Path) -> List[str]:
        """分析单个文件的导入"""

        broken_imports = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找所有src开头的导入
            import_pattern = r'^(?:from|import)\s+src\.'
            imports = re.findall(import_pattern, content, re.MULTILINE)

            for imp in imports:
                if not self._validate_import_path(imp):
                    broken_imports.append(imp)

        except Exception as e:
            print(f"❌ 读取文件失败 {file_path}: {e}")

        return broken_imports

    def _validate_import_path(self, import_statement: str) -> bool:
        """验证导入路径"""

        # 提取模块路径
        match = re.match(r'(?:from|import)\s+src\.([\w\.]+)', import_statement)
        if not match:
            return False

        module_path = match.group(1)
        parts = module_path.split('.')

        # 尝试多种路径组合
        for i in range(len(parts)):
            test_path = '.'.join(parts[:i+1])
            target_path = self.src_dir / test_path.replace('.', '/')

            # 检查目录
            if target_path.exists() and target_path.is_dir():
                # 检查是否有__init__.py
                if (target_path / '__init__.py').exists():
                    return True

            # 检查文件
            target_file = self.src_dir / f"{test_path.replace('.', '/')}.py"
            if target_file.exists():
                return True

        # 如果是from导入，尝试验证类或函数是否存在
        if import_statement.startswith('from'):
            # 简单地认为如果模块路径存在，就是有效的
            # 这样可以减少误报
            if len(parts) >= 2:
                parent_path = '.'.join(parts[:-1])
                for i in range(len(parent_path.split('.'))):
                    test_path = '.'.join(parent_path.split('.')[:i+1])
                    target_path = self.src_dir / test_path.replace('.', '/')
                    if target_path.exists():
                        return True

        return False

    def _classify_import_pattern(self, import_statement: str) -> str:
        """分类导入模式"""

        patterns = {
            "infrastructure.core": "infrastructure.core",
            "infrastructure.base": "infrastructure.base",
            "data.core": "data.core",
            "data.base": "data.base",
            "features.core": "features.core",
            "features.base": "features.base",
            "core.core": "core.core",
            "core.base": "core.base",
            "trading.core": "trading.core",
            "trading.base": "trading.base",
            "utils": "utils",
            "config": "config",
            "common": "common"
        }

        for pattern, category in patterns.items():
            if pattern in import_statement:
                return category

        return "other"

    def _can_fix_import(self, import_statement: str) -> bool:
        """检查导入是否可以修复"""

        # 尝试应用映射规则
        for pattern, replacement in self.import_mapping.items():
            if re.search(pattern, import_statement):
                fixed_import = re.sub(pattern, replacement, import_statement)
                if self._validate_import_path(fixed_import):
                    return True

        return False

    def fix_imports_batch(self, batch_size: int = 50, dry_run: bool = True) -> Dict[str, Any]:
        """批量修复导入路径"""

        results = {
            "dry_run": dry_run,
            "total_files_processed": 0,
            "files_fixed": 0,
            "imports_fixed": 0,
            "errors": [],
            "fixes_applied": []
        }

        print(f"🔧 开始批量修复导入路径... ({'仅分析' if dry_run else '实际执行'})")

        # 获取所有测试文件
        test_files = []
        for root, dirs, files in os.walk(self.tests_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    test_files.append(Path(root) / file)

        # 分批处理
        for i in range(0, len(test_files), batch_size):
            batch = test_files[i:i + batch_size]
            print(f"📦 处理批次 {i//batch_size + 1}/{(len(test_files) + batch_size - 1)//batch_size}")

            for file_path in batch:
                results["total_files_processed"] += 1

                try:
                    fixes = self._fix_file_imports_batch(file_path, dry_run)
                    if fixes:
                        results["files_fixed"] += 1
                        results["imports_fixed"] += len(fixes)
                        results["fixes_applied"].extend(fixes)
                except Exception as e:
                    results["errors"].append(f"{file_path}: {e}")

        return results

    def _fix_file_imports_batch(self, file_path: Path, dry_run: bool = True) -> List[Dict[str, Any]]:
        """批量修复单个文件的导入"""

        fixes = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            modified = False

            # 逐行处理
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(('from src.', 'import src.')):
                    # 尝试修复导入
                    fixed_line = self._fix_single_import_batch(line)
                    if fixed_line != line:
                        lines[i] = fixed_line
                        modified = True
                        fixes.append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "original": line.strip(),
                            "fixed": fixed_line.strip(),
                            "line_number": i + 1
                        })

            if modified and not dry_run:
                new_content = '\n'.join(lines)

                # 备份原文件
                backup_path = self.backup_dir / file_path.relative_to(self.project_root)
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)

                # 写入修复后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

        except Exception as e:
            fixes.append({
                "file": str(file_path.relative_to(self.project_root)),
                "error": str(e)
            })

        return fixes

    def _fix_single_import_batch(self, import_line: str) -> str:
        """批量修复单个导入语句"""

        # 尝试应用映射规则
        for pattern, replacement in self.import_mapping.items():
            if re.search(pattern, import_line):
                fixed_import = re.sub(pattern, replacement, import_line)
                # 验证修复后的导入是否有效
                if self._validate_import_path(fixed_import):
                    return fixed_import

        return import_line

    def generate_batch_report(self, analysis: Dict[str, Any], fixes: Dict[str, Any]) -> str:
        """生成批量修复报告"""

        report = f"""# 🔧 测试导入批量修复报告

## 📅 生成时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 分析结果

### 测试文件概况
- **总测试文件数**: {analysis['total_files']}
- **有问题的文件数**: {analysis['files_with_broken_imports']}
- **损坏的导入数**: {len(analysis['broken_imports'])}
- **可修复的导入数**: {len(analysis['fixable_imports'])}
- **无法修复的导入数**: {len(analysis['unfixable_imports'])}

### 导入问题模式统计
"""

        for pattern, count in sorted(analysis['import_pattern_stats'].items(), key=lambda x: x[1], reverse=True):
            report += f"- **{pattern}**: {count} 个导入\n"

        report += f"""
## 🔧 修复结果

### 修复统计
- **处理的文件数**: {fixes['total_files_processed']}
- **修复的文件数**: {fixes['files_fixed']}
- **修复的导入数**: {fixes['imports_fixed']}
- **修复模式**: {'仅分析' if fixes['dry_run'] else '实际执行'}

### 应用的主要修复规则
"""

        for pattern, replacement in list(self.import_mapping.items())[:10]:  # 只显示前10个
            report += f"- `{pattern}` → `{replacement}`\n"

        if len(self.import_mapping) > 10:
            report += f"- ... 还有 {len(self.import_mapping) - 10} 个修复规则\n"

        if fixes['errors']:
            report += f"""
### 修复错误
"""
            for error in fixes['errors'][:10]:  # 只显示前10个错误
                report += f"- {error}\n"
            if len(fixes['errors']) > 10:
                report += f"... 还有 {len(fixes['errors']) - 10} 个错误\n"

        if not fixes['dry_run'] and fixes['fixes_applied']:
            report += f"""
### 修复示例
"""
            for fix in fixes['fixes_applied'][:5]:  # 只显示前5个修复
                if 'error' not in fix:
                    report += f"""
**文件**: {fix['file']}:{fix['line_number']}
- **修复前**: {fix['original']}
- **修复后**: {fix['fixed']}
"""
            if len(fixes['fixes_applied']) > 5:
                report += f"... 还有 {len(fixes['fixes_applied']) - 5} 个修复\n"

        # 提供后续建议
        report += f"""
## 💡 后续建议

### 优先级修复策略
1. **高优先级**: 修复核心业务逻辑的导入问题
2. **中优先级**: 修复基础设施和工具类的导入问题
3. **低优先级**: 修复测试辅助工具的导入问题

### 验证策略
1. **单元测试**: 运行关键单元测试验证修复效果
2. **集成测试**: 运行集成测试验证模块间交互
3. **端到端测试**: 运行E2E测试验证完整业务流程

### 监控建议
1. **测试覆盖率**: 确保修复后测试覆盖率不下降
2. **测试执行时间**: 监控测试执行时间的变化
3. **错误率**: 跟踪测试失败率的变化

## ⚠️ 备份信息
- **备份目录**: {self.backup_dir}
- **备份文件**: 所有修改的文件都会自动备份
- **恢复方法**: 从备份目录复制文件到原位置

---
*测试导入批量修复工具版本: v1.0*
*备份目录: {self.backup_dir}*
"""

        return report


def main():
    """主函数"""

    import argparse

    parser = argparse.ArgumentParser(description='测试导入批量修复工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--batch-size', type=int, default=100, help='批次大小')
    parser.add_argument('--dry-run', action='store_true', help='仅分析不执行修复')
    parser.add_argument('--report', action='store_true', help='生成详细报告')

    args = parser.parse_args()

    tool = BatchTestImportFixer(args.project)

    print("🔍 开始批量分析测试文件导入问题...")
    analysis = tool.find_broken_imports_batch(args.batch_size)

    print(f"✅ 发现 {analysis['total_files']} 个测试文件")
    print(f"✅ 发现 {analysis['files_with_broken_imports']} 个有问题的文件")
    print(f"✅ 损坏的导入: {len(analysis['broken_imports'])} 个")
    print(f"✅ 可修复的导入: {len(analysis['fixable_imports'])} 个")
    print(f"✅ 无法修复的导入: {len(analysis['unfixable_imports'])} 个")

    # 显示主要的导入问题模式
    print("\n📊 主要导入问题模式:")
    for pattern, count in sorted(analysis['import_pattern_stats'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {pattern}: {count} 个")

    if len(analysis['fixable_imports']) == 0:
        print("❌ 没有发现可以自动修复的导入问题")
        return

    print(f"\n🔧 开始批量修复导入路径... ({'仅分析' if args.dry_run else '实际执行'})")
    fixes = tool.fix_imports_batch(args.batch_size, dry_run=args.dry_run)

    print(f"✅ 处理了 {fixes['total_files_processed']} 个文件")
    print(f"✅ 修复了 {fixes['files_fixed']} 个文件")
    print(f"✅ 修复了 {fixes['imports_fixed']} 个导入")

    if args.report:
        report_content = tool.generate_batch_report(analysis, fixes)
        report_file = tool.project_root / "reports" / \
            f"batch_test_import_fix_{'analysis' if args.dry_run else 'execution'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📊 批量修复报告已保存: {report_file}")


if __name__ == "__main__":
    main()
