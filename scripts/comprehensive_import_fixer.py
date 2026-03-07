#!/usr/bin/env python3
"""
全面修复基础设施层相对导入问题

处理所有类型的相对导入，包括：
- from ..xxx (双点号相对导入)
- from .xxx (单点号相对导入)
- from ...xxx (三点号相对导入)

作者: RQA2025 Team
版本: 1.0.0
更新: 2025年9月21日
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple


class ComprehensiveImportFixer:
    """全面的导入修复器"""

    def __init__(self, infrastructure_path: str = "src/infrastructure"):
        self.infrastructure_path = Path(infrastructure_path)

        # 构建模块路径映射
        self.module_paths = self._build_module_path_mapping()

    def _build_module_path_mapping(self) -> Dict[str, str]:
        """构建模块路径映射"""

        mapping = {}

        # 遍历所有Python文件，构建相对路径到绝对路径的映射
        for file_path in self.infrastructure_path.rglob("*.py"):
            if file_path.name.startswith('__'):
                continue

            # 计算相对路径
            try:
                relative_path = file_path.relative_to(self.infrastructure_path)
                module_parts = list(relative_path.with_suffix('').parts)

                # 为每个层级创建映射
                for i in range(len(module_parts)):
                    current_parts = module_parts[:i+1]
                    module_name = '.'.join(current_parts)

                    # 绝对路径
                    abs_path = f"infrastructure.{module_name}"

                    # 为不同级别的相对导入创建映射
                    for level in range(1, len(module_parts) + 1):
                        if i >= level - 1:
                            relative_parts = current_parts[-(level):]
                            if relative_parts:
                                relative_key = '.'.join([''] + relative_parts)
                                if len(relative_key) > 1:  # 避免单个点
                                    mapping[relative_key] = abs_path

            except ValueError:
                continue

        return mapping

    def fix_all_relative_imports(self, dry_run: bool = False) -> Tuple[int, int, int]:
        """修复所有相对导入

        Returns:
            (修复的文件数, 修复的导入数, 剩余问题数)
        """

        print(f"🔧 {'预览' if dry_run else '开始'}全面修复相对导入...")
        print("=" * 60)

        fixed_files = 0
        total_fixed_imports = 0
        remaining_issues = 0

        # 遍历所有Python文件
        for file_path in self.infrastructure_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content
                fixed_imports_in_file = 0

                # 处理每一行
                lines = content.split('\n')
                modified_lines = []

                for line in lines:
                    original_line = line
                    line = line.strip()

                    # 处理相对导入
                    if re.match(r'from \.\.?', line):
                        fixed_line = self._fix_single_import_line(line, file_path)
                        if fixed_line != original_line:
                            fixed_imports_in_file += 1
                            line = fixed_line.strip() if fixed_line.strip() else original_line
                        else:
                            # 如果无法修复，保持原样但记录问题
                            remaining_issues += 1

                    modified_lines.append(line)

                # 保存修改
                if fixed_imports_in_file > 0 and not dry_run:
                    new_content = '\n'.join(modified_lines)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                    # 校验文件语法
                    syntax_valid, syntax_error = self._validate_file_syntax(file_path)
                    if syntax_valid:
                        fixed_files += 1
                        total_fixed_imports += fixed_imports_in_file
                        print(f"  ✅ {file_path.name}: 修复了 {fixed_imports_in_file} 个导入")
                    else:
                        print(
                            f"  ⚠️ {file_path.name}: 修复了 {fixed_imports_in_file} 个导入 - 语法错误: {syntax_error}")
                        # 恢复原内容
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(original_content)

                elif fixed_imports_in_file > 0 and dry_run:
                    total_fixed_imports += fixed_imports_in_file
                    print(f"  👁️  {file_path.name}: 需要修复 {fixed_imports_in_file} 个导入")

            except Exception as e:
                print(f"  ❌ 处理文件失败 {file_path}: {e}")
                remaining_issues += 1

        # 最终验证
        final_remaining = self._validate_fixes()

        print("\n📊 修复总结:")
        print(f"  处理文件数: {fixed_files}")
        print(f"  修复导入数: {total_fixed_imports}")
        print(f"  剩余问题数: {final_remaining}")
        print(f"  执行模式: {'预览模式' if dry_run else '实际修复'}")

        return fixed_files, total_fixed_imports, final_remaining

    def _validate_file_syntax(self, file_path: str) -> Tuple[bool, str]:
        """校验文件语法

        Returns:
            (is_valid, error_message)
        """
        try:
            import py_compile
            py_compile.compile(file_path, doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            return False, str(e)
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"未知错误: {e}"

    def _fix_single_import_line(self, import_line: str, file_path: Path) -> str:
        """修复单个导入行"""

        # 计算当前文件的相对路径
        try:
            rel_path = file_path.relative_to(self.infrastructure_path)
            path_parts = list(rel_path.parent.parts) if rel_path.parent != Path('.') else []
        except ValueError:
            return import_line

        # 处理不同的相对导入模式
        if import_line.startswith('from .'):
            return self._fix_single_dot_import(import_line, path_parts)
        elif import_line.startswith('from ..'):
            return self._fix_double_dot_import(import_line, path_parts)
        elif import_line.startswith('from ...'):
            return self._fix_triple_dot_import(import_line, path_parts)

        return import_line

    def _fix_single_dot_import(self, import_line: str, path_parts: List[str]) -> str:
        """修复单点号相对导入 (from .xxx)"""

        # from .module -> from infrastructure.current_package.module
        match = re.match(r'from \.(\w+)', import_line)
        if match:
            module = match.group(1)
            if path_parts:
                package_path = '.'.join(path_parts)
                return import_line.replace(f'from .{module}', f'from infrastructure.{package_path}.{module}')
            else:
                return import_line.replace(f'from .{module}', f'from infrastructure.{module}')

        # from .submodule.module -> from infrastructure.current_package.submodule.module
        match = re.match(r'from \.([\w\.]+)', import_line)
        if match:
            submodule_path = match.group(1)
            if path_parts:
                package_path = '.'.join(path_parts)
                return import_line.replace(f'from .{submodule_path}', f'from infrastructure.{package_path}.{submodule_path}')
            else:
                return import_line.replace(f'from .{submodule_path}', f'from infrastructure.{submodule_path}')

        return import_line

    def _fix_double_dot_import(self, import_line: str, path_parts: List[str]) -> str:
        """修复双点号相对导入 (from ..xxx)"""

        # from ..module -> from infrastructure.parent_package.module
        match = re.match(r'from \.\.(\w+)', import_line)
        if match:
            module = match.group(1)
            if len(path_parts) >= 1:
                parent_package = '.'.join(path_parts[:-1]) if len(path_parts) > 1 else ''
                if parent_package:
                    return import_line.replace(f'from ..{module}', f'from infrastructure.{parent_package}.{module}')
                else:
                    return import_line.replace(f'from ..{module}', f'from infrastructure.{module}')
            else:
                return import_line.replace(f'from ..{module}', f'from infrastructure.{module}')

        # from ..submodule.module -> from infrastructure.parent_package.submodule.module
        match = re.match(r'from \.\.([\w\.]+)', import_line)
        if match:
            submodule_path = match.group(1)
            if len(path_parts) >= 1:
                parent_package = '.'.join(path_parts[:-1]) if len(path_parts) > 1 else ''
                if parent_package:
                    return import_line.replace(f'from ..{submodule_path}', f'from infrastructure.{parent_package}.{submodule_path}')
                else:
                    return import_line.replace(f'from ..{submodule_path}', f'from infrastructure.{submodule_path}')
            else:
                return import_line.replace(f'from ..{submodule_path}', f'from infrastructure.{submodule_path}')

        return import_line

    def _fix_triple_dot_import(self, import_line: str, path_parts: List[str]) -> str:
        """修复三点号相对导入 (from ...xxx)"""

        # from ...module -> from infrastructure.grandparent_package.module
        match = re.match(r'from \.\.\.(\w+)', import_line)
        if match:
            module = match.group(1)
            if len(path_parts) >= 2:
                grandparent_package = '.'.join(path_parts[:-2]) if len(path_parts) > 2 else ''
                if grandparent_package:
                    return import_line.replace(f'from ...{module}', f'from infrastructure.{grandparent_package}.{module}')
                else:
                    return import_line.replace(f'from ...{module}', f'from infrastructure.{module}')
            else:
                return import_line.replace(f'from ...{module}', f'from infrastructure.{module}')

        return import_line

    def _validate_fixes(self) -> int:
        """验证修复结果"""

        remaining_issues = 0

        for file_path in self.infrastructure_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if re.match(r'from \.\.?', line):
                        remaining_issues += 1

            except Exception:
                remaining_issues += 1

        return remaining_issues

    def analyze_remaining_issues(self) -> Dict[str, List[str]]:
        """分析剩余问题"""

        issues = {}

        for file_path in self.infrastructure_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                file_issues = []

                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if re.match(r'from \.\.?', line):
                        file_issues.append(f"Line {i}: {line}")

                if file_issues:
                    issues[str(file_path)] = file_issues

            except Exception as e:
                issues[str(file_path)] = [f"Error: {e}"]

        return issues


def main():
    """主函数"""

    print("🔧 基础设施层全面相对导入修复工具")
    print("=" * 50)

    fixer = ComprehensiveImportFixer()

    # 首先分析模块路径映射
    print("1️⃣ 构建模块路径映射...")
    module_count = len(fixer.module_paths)
    print(f"   📊 构建了 {module_count} 个模块路径映射")

    # 预览需要修复的问题
    print("\\n2️⃣ 分析剩余相对导入问题...")
    remaining_before = fixer._validate_fixes()
    print(f"   📊 发现 {remaining_before} 个相对导入需要修复")

    # 显示一些示例
    issues = fixer.analyze_remaining_issues()
    if issues:
        print("\\n📋 问题示例:")
        example_count = 0
        for file_path, file_issues in list(issues.items())[:3]:
            print(f"\\n📄 {Path(file_path).name}:")
            for issue in file_issues[:2]:
                print(f"  {issue}")
                example_count += 1
                if example_count >= 6:
                    break
            if example_count >= 6:
                break

    # 执行修复
    print("\\n3️⃣ 执行全面修复...")
    fixed_files, fixed_imports, remaining_after = fixer.fix_all_relative_imports(dry_run=False)

    # 生成报告
    print("\n📋 最终修复报告:")
    print(f"  修复文件数: {fixed_files}")
    print(f"  修复导入数: {fixed_imports}")
    print(f"  修复前问题: {remaining_before}")
    print(f"  修复后剩余: {remaining_after}")
    success_rate = ((remaining_before - remaining_after) /
                    remaining_before * 100) if remaining_before > 0 else 0.0
    print(f"  修复成功率: {success_rate:.1f}%")

    if remaining_after == 0:
        print("\\n🎉 所有相对导入修复完成!")
    else:
        print(f"\\n⚠️  还有 {remaining_after} 个相对导入需要手动处理")

        # 显示剩余问题
        print("\\n📋 剩余问题文件:")
        remaining_issues = fixer.analyze_remaining_issues()
        for file_path, file_issues in list(remaining_issues.items())[:5]:
            print(f"  • {Path(file_path).name}: {len(file_issues)} 个问题")
        if len(remaining_issues) > 5:
            print(f"  ... 还有 {len(remaining_issues) - 5} 个文件")


if __name__ == "__main__":
    main()
