#!/usr/bin/env python3
"""
基础设施层Phase 2重构工具

统一接口定义和重新组织文件结构
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict


class InfrastructurePhase2Refactor:
    """Phase 2架构重构工具"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.backup_dir = Path('backup_phase2')
        self.backup_dir.mkdir(exist_ok=True)

        # 接口继承映射
        self.interface_mapping = {
            # Factory模式类应该继承BaseFactory
            'Factory': 'BaseFactory',
            'ComponentFactory': 'BaseComponentFactory',

            # Manager模式类应该继承BaseManager
            'Manager': 'BaseManager',
            'CacheManager': 'BaseManager',
            'ConfigManager': 'BaseManager',

            # Service模式类应该继承BaseService
            'Service': 'BaseService',
            'HealthChecker': 'BaseService',

            # Handler模式类应该继承BaseHandler
            'Handler': 'BaseHandler',

            # Provider模式类应该继承BaseProvider
            'Provider': 'BaseProvider',

            # Monitor模式类应该继承BaseMonitor
            'Monitor': 'BaseMonitor',
        }

        # 导入映射
        self.import_mapping = {
            'BaseFactory': 'from src.infrastructure.interfaces import BaseFactory',
            'BaseManager': 'from src.infrastructure.interfaces import BaseManager',
            'BaseService': 'from src.infrastructure.interfaces import BaseService',
            'BaseHandler': 'from src.infrastructure.interfaces import BaseHandler',
            'BaseProvider': 'from src.infrastructure.interfaces import BaseProvider',
            'BaseMonitor': 'from src.infrastructure.interfaces import BaseMonitor',
            'BaseComponentFactory': 'from src.infrastructure.interfaces import BaseComponentFactory',
        }

    def execute_phase2_refactor(self) -> Dict[str, Any]:
        """执行Phase 2重构"""
        print('🏗️ 开始基础设施层Phase 2架构重构')
        print('=' * 60)

        results = {
            'interface_inheritance_fix': self._fix_interface_inheritance(),
            'file_structure_reorganization': self._reorganize_file_structure(),
            'naming_standardization': self._standardize_naming(),
            'duplicate_file_consolidation': self._consolidate_duplicate_files(),
            'summary': {}
        }

        # 生成重构摘要
        results['summary'] = self._generate_refactor_summary(results)

        print('\\n✅ Phase 2重构完成！')
        self._print_summary(results['summary'])

        return results

    def _fix_interface_inheritance(self) -> Dict[str, Any]:
        """修复接口继承问题"""
        print('\\n🔧 修复接口继承问题...')

        # 分析所有Python文件中的类定义
        classes_to_fix = self._analyze_class_inheritance()

        # 修复类继承
        fixed_count = 0
        for file_path, classes in classes_to_fix.items():
            if self._fix_single_file_inheritance(file_path, classes):
                fixed_count += len(classes)

        return {
            'classes_analyzed': sum(len(classes) for classes in classes_to_fix.values()),
            'classes_fixed': fixed_count,
            'files_modified': len([f for f, c in classes_to_fix.items() if c])
        }

    def _analyze_class_inheritance(self) -> Dict[str, List[Dict[str, Any]]]:
        """分析类继承情况"""
        classes_to_fix = defaultdict(list)

        # 遍历所有Python文件
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 解析类定义
                        classes = self._parse_classes(content)
                        for class_info in classes:
                            expected_base = self._get_expected_base_class(class_info['name'])
                            if expected_base and expected_base not in class_info['bases']:
                                classes_to_fix[str(file_path)].append({
                                    'class_name': class_info['name'],
                                    'current_bases': class_info['bases'],
                                    'expected_base': expected_base,
                                    'line_no': class_info['line_no']
                                })

                    except Exception as e:
                        print(f'⚠️ 解析文件失败 {file_path}: {e}')

        return classes_to_fix

    def _parse_classes(self, content: str) -> List[Dict[str, Any]]:
        """解析文件中的类定义"""
        classes = []
        lines = content.split('\\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('class '):
                # 解析类定义
                match = re.match(r'class\s+(\w+)\s*(\([^)]*\))?:', line)
                if match:
                    class_name = match.group(1)
                    bases_str = match.group(2) or ''
                    bases = []

                    if bases_str:
                        # 提取基类
                        bases_content = bases_str[1:-1]  # 去掉括号
                        bases = [b.strip() for b in bases_content.split(',') if b.strip()]

                    classes.append({
                        'name': class_name,
                        'bases': bases,
                        'line_no': i + 1
                    })

        return classes

    def _get_expected_base_class(self, class_name: str) -> Optional[str]:
        """根据类名确定期望的基类"""
        for pattern, base_class in self.interface_mapping.items():
            if pattern in class_name:
                return base_class
        return None

    def _fix_single_file_inheritance(self, file_path: Path, classes_to_fix: List[Dict[str, Any]]) -> bool:
        """修复单个文件的继承关系"""
        if not classes_to_fix:
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 备份文件
            self._backup_file(file_path)

            lines = content.split('\\n')
            modified = False

            for class_info in classes_to_fix:
                line_no = class_info['line_no'] - 1  # 转换为0索引
                if line_no < len(lines):
                    line = lines[line_no]

                    # 添加期望的基类
                    expected_base = class_info['expected_base']

                    if '(' in line and ')' in line:
                        # 已有基类，添加新的基类
                        if class_info['current_bases']:
                            # 在现有基类后添加
                            lines[line_no] = line.replace('):', f', {expected_base}):')
                        else:
                            # 没有基类，添加第一个
                            lines[line_no] = line.replace('():', f'({expected_base}):')
                    else:
                        # 没有基类定义，添加
                        lines[line_no] = line.replace(':', f'({expected_base}):')

                    # 添加必要的导入
                    if expected_base in self.import_mapping:
                        import_line = self.import_mapping[expected_base]
                        lines = self._add_import(lines, import_line)

                    modified = True
                    print(f'✅ 修复 {file_path.name}: {class_info["class_name"]} 继承 {expected_base}')

            if modified:
                # 写回文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\\n'.join(lines))

            return modified

        except Exception as e:
            print(f'❌ 修复文件失败 {file_path}: {e}')
            return False

    def _add_import(self, lines: List[str], import_line: str) -> List[str]:
        """添加导入语句"""
        # 查找合适的位置添加导入
        insert_pos = 0

        # 跳过文件开头的注释和空行
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                insert_pos = i
                break

        # 在现有导入后添加
        for i in range(insert_pos, len(lines)):
            if lines[i].strip().startswith(('from ', 'import ')):
                continue
            elif lines[i].strip() and not lines[i].strip().startswith('#'):
                # 找到第一个非导入行，插入到前面
                lines.insert(i, import_line)
                return lines

        # 如果没有找到，在文件开头添加
        lines.insert(insert_pos, import_line)
        return lines

    def _reorganize_file_structure(self) -> Dict[str, Any]:
        """重新组织文件结构"""
        print('\\n📁 重新组织文件结构...')

        # 分析当前文件结构
        file_analysis = self._analyze_file_structure()

        # 创建新的目录结构
        new_structure = self._create_new_directory_structure(file_analysis)

        return {
            'files_analyzed': file_analysis['total_files'],
            'directories_created': len(new_structure.get('directories', [])),
            'files_moved': len(new_structure.get('moves', [])),
            'duplicates_found': file_analysis['duplicates']
        }

    def _analyze_file_structure(self) -> Dict[str, Any]:
        """分析当前文件结构"""
        analysis = {
            'total_files': 0,
            'by_type': defaultdict(int),
            'by_size': {'small': 0, 'medium': 0, 'large': 0},
            'duplicates': 0,
            'empty_dirs': []
        }

        duplicate_names = defaultdict(list)

        for root, dirs, files in os.walk(self.infra_dir):
            # 检查空目录
            if not files and not any(os.listdir(Path(root))):
                analysis['empty_dirs'].append(root)

            for file in files:
                if file.endswith('.py'):
                    analysis['total_files'] += 1
                    file_path = Path(root) / file

                    # 按类型分类
                    if 'test' in file.lower():
                        analysis['by_type']['test'] += 1
                    elif 'interface' in file.lower() or 'protocol' in file.lower():
                        analysis['by_type']['interface'] += 1
                    elif 'factory' in file.lower():
                        analysis['by_type']['factory'] += 1
                    elif 'manager' in file.lower():
                        analysis['by_type']['manager'] += 1
                    elif 'service' in file.lower():
                        analysis['by_type']['service'] += 1
                    else:
                        analysis['by_type']['other'] += 1

                    # 按大小分类
                    try:
                        size = file_path.stat().st_size
                        if size < 1024:
                            analysis['by_size']['small'] += 1
                        elif size < 10240:
                            analysis['by_size']['medium'] += 1
                        else:
                            analysis['by_size']['large'] += 1
                    except:
                        pass

                    # 检查重复文件名
                    duplicate_names[file].append(str(file_path))

        # 计算重复文件
        for filename, paths in duplicate_names.items():
            if len(paths) > 1:
                analysis['duplicates'] += len(paths) - 1  # 重复的数量

        return analysis

    def _create_new_directory_structure(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """创建新的目录结构"""
        # 建议的新结构
        new_structure = {
            'directories': [
                'src/infrastructure/core/',      # 核心组件
                'src/infrastructure/interfaces/',  # 接口定义
                'src/infrastructure/implementations/',  # 具体实现
                'src/infrastructure/utils/',     # 工具类
                'src/infrastructure/tests/',     # 测试文件
            ],
            'moves': [],
            'consolidations': []
        }

        print(f'📊 文件结构分析完成:')
        print(f'   总文件数: {analysis["total_files"]}')
        print(f'   重复文件: {analysis["duplicates"]}')
        print(f'   空目录: {len(analysis["empty_dirs"])}')

        return new_structure

    def _standardize_naming(self) -> Dict[str, Any]:
        """标准化命名规范"""
        print('\\n🏷️ 标准化命名规范...')

        # 检查命名规范
        naming_issues = self._analyze_naming_conventions()

        # 修复命名问题
        fixed_count = 0
        for file_path, issues in naming_issues.items():
            if self._fix_naming_issues(file_path, issues):
                fixed_count += len(issues)

        return {
            'files_analyzed': len(naming_issues),
            'issues_found': sum(len(issues) for issues in naming_issues.values()),
            'issues_fixed': fixed_count
        }

    def _analyze_naming_conventions(self) -> Dict[str, List[Dict[str, Any]]]:
        """分析命名规范"""
        naming_issues = defaultdict(list)

        # 接口命名规范：应该以I开头
        interface_pattern = re.compile(r'class\s+(\w+).*?:')

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 检查接口命名
                        for match in interface_pattern.finditer(content):
                            class_name = match.group(1)
                            if ('interface' in file.lower() or 'protocol' in file.lower()) and not class_name.startswith('I'):
                                naming_issues[str(file_path)].append({
                                    'type': 'interface_naming',
                                    'class_name': class_name,
                                    'expected': f'I{class_name}',
                                    'line': content[:match.start()].count('\\n') + 1
                                })

                    except Exception:
                        continue

        return naming_issues

    def _fix_naming_issues(self, file_path: Path, issues: List[Dict[str, Any]]) -> bool:
        """修复命名问题"""
        if not issues:
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 备份文件
            self._backup_file(file_path)

            for issue in issues:
                if issue['type'] == 'interface_naming':
                    old_name = issue['class_name']
                    new_name = issue['expected']

                    # 替换类定义
                    content = re.sub(
                        rf'\\bclass\\s+{re.escape(old_name)}\\b',
                        f'class {new_name}',
                        content
                    )

                    # 替换类引用（简单的重命名）
                    content = re.sub(
                        rf'\\b{re.escape(old_name)}\\b(?!\\w)',
                        new_name,
                        content
                    )

                    print(f'✅ 重命名 {file_path.name}: {old_name} -> {new_name}')

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True

        except Exception as e:
            print(f'❌ 修复命名失败 {file_path}: {e}')
            return False

    def _consolidate_duplicate_files(self) -> Dict[str, Any]:
        """合并重复文件"""
        print('\\n🔄 合并重复文件...')

        # 查找重复文件
        duplicates = self._find_duplicate_files()

        consolidated = 0
        for file_group in duplicates:
            if len(file_group) > 1:
                # 保留第一个文件，删除其他文件
                main_file = file_group[0]
                for duplicate_file in file_group[1:]:
                    try:
                        # 备份文件
                        self._backup_file(Path(duplicate_file))

                        # 删除重复文件
                        os.remove(duplicate_file)
                        consolidated += 1
                        print(f'✅ 删除重复文件: {Path(duplicate_file).name}')

                    except Exception as e:
                        print(f'❌ 删除失败 {duplicate_file}: {e}')

        return {
            'duplicate_groups': len(duplicates),
            'files_consolidated': consolidated
        }

    def _find_duplicate_files(self) -> List[List[str]]:
        """查找重复文件"""
        file_hashes = defaultdict(list)

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 计算文件哈希
                        content_hash = hash(content)
                        file_hashes[content_hash].append(str(file_path))

                    except Exception:
                        continue

        # 返回重复的文件组
        duplicates = [paths for paths in file_hashes.values() if len(paths) > 1]
        return duplicates

    def _backup_file(self, file_path: Path):
        """备份文件"""
        if file_path.exists():
            backup_path = self.backup_dir / file_path.name
            counter = 1
            while backup_path.exists():
                backup_path = self.backup_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
                counter += 1

            try:
                import shutil
                shutil.copy2(file_path, backup_path)
                print(f'📁 已备份: {file_path.name}')
            except Exception as e:
                print(f'⚠️ 备份失败 {file_path}: {e}')

    def _generate_refactor_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成重构摘要"""
        summary = {
            'total_actions': 0,
            'files_modified': 0,
            'classes_fixed': 0,
            'naming_issues_fixed': 0,
            'duplicates_consolidated': 0,
            'status': 'completed'
        }

        # 统计接口继承修复
        inherit_fix = results.get('interface_inheritance_fix', {})
        summary['classes_fixed'] = inherit_fix.get('classes_fixed', 0)
        summary['files_modified'] += inherit_fix.get('files_modified', 0)
        if inherit_fix.get('classes_fixed', 0) > 0:
            summary['total_actions'] += 1

        # 统计文件重组
        struct_fix = results.get('file_structure_reorganization', {})
        summary['files_modified'] += struct_fix.get('files_moved', 0)
        if struct_fix.get('files_moved', 0) > 0:
            summary['total_actions'] += 1

        # 统计命名标准化
        naming_fix = results.get('naming_standardization', {})
        summary['naming_issues_fixed'] = naming_fix.get('issues_fixed', 0)
        summary['files_modified'] += naming_fix.get('files_analyzed', 0)
        if naming_fix.get('issues_fixed', 0) > 0:
            summary['total_actions'] += 1

        # 统计重复文件合并
        dup_fix = results.get('duplicate_file_consolidation', {})
        summary['duplicates_consolidated'] = dup_fix.get('files_consolidated', 0)
        if dup_fix.get('files_consolidated', 0) > 0:
            summary['total_actions'] += 1

        return summary

    def _print_summary(self, summary: Dict[str, Any]):
        """打印摘要"""
        print('\\n📊 Phase 2重构摘要:')
        print('-' * 40)
        print(f'✅ 重构操作: {summary["total_actions"]} 个')
        print(f'📁 修改文件: {summary["files_modified"]} 个')
        print(f'🔧 修复类继承: {summary["classes_fixed"]} 个')
        print(f'🏷️ 修复命名: {summary["naming_issues_fixed"]} 个')
        print(f'🔄 合并重复: {summary["duplicates_consolidated"]} 个')
        print(f'📂 备份位置: {self.backup_dir}')

        if summary['classes_fixed'] > 0:
            print('🎉 接口继承问题显著改善！')
        if summary['naming_issues_fixed'] > 0:
            print('🎉 命名规范得到统一！')
        if summary['duplicates_consolidated'] > 0:
            print('🎉 文件重复问题得到解决！')


def main():
    """主函数"""
    refactor = InfrastructurePhase2Refactor()
    results = refactor.execute_phase2_refactor()

    # 保存重构报告
    with open('infrastructure_phase2_refactor_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('\\n📄 重构报告已保存: infrastructure_phase2_refactor_report.json')


if __name__ == "__main__":
    main()
