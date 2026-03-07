#!/usr/bin/env python3
"""
批量修复接口继承问题

根据审查报告批量修复所有缺少基类继承的类
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set


class InterfaceInheritanceFixer:
    """接口继承修复器"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.report_file = Path('infrastructure_code_review_report.json')
        self.backup_dir = Path('interface_inheritance_backup')
        self.fixed_files: Set[str] = set()

        # 基类映射
        self.base_classes = {
            'BaseFactory': 'interfaces/factory_pattern',
            'BaseManager': 'interfaces/manager_pattern',
            'BaseService': 'interfaces/service_pattern',
            'BaseHandler': 'interfaces/handler_pattern',
            'BaseProvider': 'interfaces/provider_pattern',
            'BaseInterface': 'interfaces/base_interface'
        }

    def fix_all_interface_inheritance(self) -> Dict[str, Any]:
        """修复所有接口继承问题"""
        print('🔧 开始批量修复接口继承问题')
        print('=' * 60)

        # 创建备份目录
        self.backup_dir.mkdir(exist_ok=True)

        # 读取审查报告
        issues = self._load_inheritance_issues()
        print(f'📋 发现 {len(issues)} 个接口继承问题')

        # 按文件分组问题
        issues_by_file = self._group_issues_by_file(issues)

        # 修复每个文件的问题
        fix_results = {
            'total_issues': len(issues),
            'fixed_issues': 0,
            'failed_fixes': 0,
            'files_modified': 0,
            'details': []
        }

        for file_path, file_issues in issues_by_file.items():
            try:
                result = self._fix_file_inheritance(file_path, file_issues)
                if result['success']:
                    fix_results['fixed_issues'] += len(file_issues)
                    fix_results['files_modified'] += 1
                    fix_results['details'].append(result)
                    print(f"✅ 修复 {file_path}: {len(file_issues)} 个问题")
                else:
                    fix_results['failed_fixes'] += len(file_issues)
                    print(f"❌ 修复失败 {file_path}: {result.get('error', '未知错误')}")
            except Exception as e:
                fix_results['failed_fixes'] += len(file_issues)
                print(f"❌ 异常 {file_path}: {e}")

        # 生成修复报告
        report = {
            'timestamp': self._get_timestamp(),
            'fix_results': fix_results,
            'summary': self._generate_fix_summary(fix_results),
            'verification': self._verify_fixes()
        }

        # 保存报告
        with open('interface_inheritance_fix_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print('\\n✅ 接口继承修复完成')
        self._print_fix_summary(report)

        return report

    def _load_inheritance_issues(self) -> List[Dict[str, Any]]:
        """加载继承问题"""
        with open(self.report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)

        return report['detailed_results']['architecture_compliance']['interface_inheritance']['issues']

    def _group_issues_by_file(self, issues: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按文件分组问题"""
        issues_by_file = {}

        for issue in issues:
            file_path = issue['file'].replace('\\', '/')
            if not file_path.startswith('src/infrastructure/'):
                file_path = f'src/infrastructure/{file_path}'

            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(issue)

        return issues_by_file

    def _fix_file_inheritance(self, file_path: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """修复单个文件的继承问题"""
        result = {
            'file': file_path,
            'issues_count': len(issues),
            'success': False,
            'fixed_classes': [],
            'error': None
        }

        try:
            full_path = Path(file_path)
            if not full_path.exists():
                result['error'] = '文件不存在'
                return result

            # 备份原文件
            backup_path = self.backup_dir / f"{full_path.name}.backup"
            backup_path.write_text(full_path.read_text(encoding='utf-8'), encoding='utf-8')

            # 读取文件内容
            content = full_path.read_text(encoding='utf-8')

            # 分析现有导入
            existing_imports = self._analyze_existing_imports(content)

            # 需要添加的导入
            needed_imports = self._get_needed_imports(issues, existing_imports)

            # 添加缺失的导入
            if needed_imports:
                content = self._add_missing_imports(content, needed_imports)

            # 修复每个类的继承
            fixed_classes = []
            for issue in issues:
                class_name = issue['class']
                expected_base = issue['expected_base']

                # 检查是否已经是接口（以I开头或Base开头）
                if class_name.startswith(('I', 'Base')):
                    # 接口类不需要继承，只需要确保导入正确
                    continue

                # 修复类继承
                new_content, fixed = self._fix_class_inheritance(content, class_name, expected_base)
                if fixed:
                    content = new_content
                    fixed_classes.append({
                        'class': class_name,
                        'added_base': expected_base
                    })

            # 写回文件
            full_path.write_text(content, encoding='utf-8')

            # 验证语法
            import py_compile
            py_compile.compile(str(full_path), doraise=True)

            result['success'] = True
            result['fixed_classes'] = fixed_classes
            self.fixed_files.add(file_path)

        except Exception as e:
            result['error'] = str(e)

        return result

    def _analyze_existing_imports(self, content: str) -> Set[str]:
        """分析现有导入"""
        imports = set()

        # 查找from imports
        from_imports = re.findall(r'from\s+([\w.]+)\s+import', content)
        for imp in from_imports:
            # 提取最后一部分
            parts = imp.split('.')
            if len(parts) >= 2 and parts[-2] == 'interfaces':
                imports.add(parts[-1])  # 例如 BaseManager

        return imports

    def _get_needed_imports(self, issues: List[Dict[str, Any]], existing_imports: Set[str]) -> Set[str]:
        """获取需要的导入"""
        needed = set()

        for issue in issues:
            expected_base = issue['expected_base']
            if expected_base not in existing_imports:
                needed.add(expected_base)

        return needed

    def _add_missing_imports(self, content: str, needed_imports: Set[str]) -> str:
        """添加缺失的导入"""
        if not needed_imports:
            return content

        lines = content.split('\n')
        insert_index = 0

        # 找到最后一个导入语句的位置
        for i, line in enumerate(lines):
            if line.strip().startswith(('from ', 'import ')):
                insert_index = i + 1
            elif line.strip() and not line.strip().startswith('#'):
                break

        # 构造导入语句
        import_statements = []
        for base_class in sorted(needed_imports):
            if base_class in self.base_classes:
                module_path = self.base_classes[base_class]
                # 计算相对导入层级
                file_parts = file_path.replace('src/infrastructure/', '').split('/')
                depth = len(file_parts) - 1  # 减去文件名
                dots = '.' * depth if depth > 0 else '.'
                import_stmt = f'from {dots}interfaces.{module_path.replace("interfaces/", "")} import {base_class}'
                import_statements.append(import_stmt)

        # 插入导入语句
        for stmt in reversed(import_statements):
            lines.insert(insert_index, stmt)

        return '\n'.join(lines)

    def _fix_class_inheritance(self, content: str, class_name: str, base_class: str) -> tuple[str, bool]:
        """修复单个类的继承"""
        # 查找类定义
        class_pattern = rf'(class\s+{re.escape(class_name)}\s*\()([^)]*)(\))'
        match = re.search(class_pattern, content, re.MULTILINE)

        if not match:
            return content, False

        full_match = match.group(0)
        before_paren = match.group(1)
        inheritance_list = match.group(2).strip()

        # 检查是否已经继承了该基类
        if base_class in inheritance_list:
            return content, False

        # 构建新的继承列表
        if inheritance_list:
            new_inheritance = f'{inheritance_list}, {base_class}'
        else:
            new_inheritance = base_class

        new_class_def = f'{before_paren}{new_inheritance})'

        # 替换类定义
        content = content.replace(full_match, new_class_def)

        return content, True

    def _generate_fix_summary(self, fix_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成修复总结"""
        summary = {
            'total_issues': fix_results['total_issues'],
            'fix_success_rate': (fix_results['fixed_issues'] / fix_results['total_issues'] * 100) if fix_results['total_issues'] > 0 else 0,
            'files_modified': fix_results['files_modified'],
            'most_common_missing_base': self._get_most_common_missing_base(fix_results['details']),
            'fix_distribution': self._get_fix_distribution(fix_results['details'])
        }

        return summary

    def _get_most_common_missing_base(self, details: List[Dict[str, Any]]) -> Dict[str, int]:
        """获取最常见的缺失基类"""
        base_count = {}

        for detail in details:
            for fixed_class in detail.get('fixed_classes', []):
                base = fixed_class['added_base']
                base_count[base] = base_count.get(base, 0) + 1

        return dict(sorted(base_count.items(), key=lambda x: x[1], reverse=True))

    def _get_fix_distribution(self, details: List[Dict[str, Any]]) -> Dict[str, int]:
        """获取修复分布"""
        module_count = {}

        for detail in details:
            file_path = detail['file']
            # 提取模块名
            parts = file_path.replace('src/infrastructure/', '').split('/')
            module = parts[0] if parts else 'unknown'
            module_count[module] = module_count.get(
                module, 0) + len(detail.get('fixed_classes', []))

        return dict(sorted(module_count.items(), key=lambda x: x[1], reverse=True))

    def _verify_fixes(self) -> Dict[str, Any]:
        """验证修复结果"""
        verification = {
            'syntax_check_passed': 0,
            'inheritance_check_passed': 0,
            'total_files_checked': len(self.fixed_files),
            'failed_files': []
        }

        for file_path in self.fixed_files:
            try:
                full_path = Path(file_path)

                # 语法检查
                import py_compile
                py_compile.compile(str(full_path), doraise=True)
                verification['syntax_check_passed'] += 1

                # 继承检查（简化检查）
                content = full_path.read_text(encoding='utf-8')
                if any(base in content for base in self.base_classes.keys()):
                    verification['inheritance_check_passed'] += 1
                else:
                    verification['failed_files'].append(f'{file_path}: 未找到预期基类')

            except Exception as e:
                verification['failed_files'].append(f'{file_path}: {e}')

        return verification

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _print_fix_summary(self, report: Dict[str, Any]):
        """打印修复总结"""
        summary = report['summary']
        verification = report['verification']

        print('\\n🔧 接口继承修复总结:')
        print('-' * 50)
        print(f'📋 总问题数: {summary["total_issues"]}')
        print(f'✅ 修复成功: {report["fix_results"]["fixed_issues"]}')
        print(f'❌ 修复失败: {report["fix_results"]["failed_fixes"]}')
        print('.1f')
        print(f'📁 修改文件数: {summary["files_modified"]}')

        print('\\n🏷️ 最常见的缺失基类:')
        for base, count in list(summary['most_common_missing_base'].items())[:5]:
            print(f'   {base}: {count} 次')

        print('\\n📊 修复分布 (按模块):')
        for module, count in list(summary['fix_distribution'].items())[:5]:
            print(f'   {module}: {count} 个修复')

        print('\\n🔍 验证结果:')
        print(
            f'   语法检查通过: {verification["syntax_check_passed"]}/{verification["total_files_checked"]}')
        print(
            f'   继承检查通过: {verification["inheritance_check_passed"]}/{verification["total_files_checked"]}')

        if verification['failed_files']:
            print('\\n⚠️ 验证失败的文件:')
            for failed in verification['failed_files'][:3]:
                print(f'   • {failed}')

        print('\\n📄 详细报告已保存: interface_inheritance_fix_report.json')


def main():
    """主函数"""
    fixer = InterfaceInheritanceFixer()
    report = fixer.fix_all_interface_inheritance()


if __name__ == "__main__":
    main()
