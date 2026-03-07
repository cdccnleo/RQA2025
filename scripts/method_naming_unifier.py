#!/usr/bin/env python3
"""
方法命名规范统一工具

将不符合蛇形命名规范的方法重命名为标准格式
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Any


class MethodNamingUnifier:
    """方法命名统一器"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.backup_dir = Path('method_naming_backup')

        # 需要重命名的方法映射
        self.rename_map = {
            'validate': 'validate_config',
            'initialize': 'initialize_component',
            'shutdown': 'shutdown_service',
            'create': 'create_instance',
            'build': 'build_component',
            'setup': 'setup_environment',
            'cleanup': 'cleanup_resources',
            'update': 'update_data',
            'process': 'process_data',
            'handle': 'handle_request',
            'check': 'check_status',
            'load': 'load_config',
            'save': 'save_config',
            'start': 'start_service',
            'stop': 'stop_service',
            'run': 'run_process',
            'execute': 'execute_task',
            'get': 'get_data',
            'set': 'set_data',
            'add': 'add_item',
            'remove': 'remove_item',
            'clear': 'clear_data',
            'reset': 'reset_state',
            'connect': 'connect_service',
            'disconnect': 'disconnect_service',
            'send': 'send_data',
            'receive': 'receive_data'
        }

    def unify_method_names(self) -> Dict[str, Any]:
        """统一方法命名"""
        print('🏷️ 开始统一方法命名规范')
        print('=' * 50)

        # 创建备份目录
        self.backup_dir.mkdir(exist_ok=True)

        # 扫描所有Python文件
        python_files = self._get_python_files()
        print(f'📁 发现 {len(python_files)} 个Python文件')

        # 分析方法命名问题
        naming_analysis = self._analyze_method_naming(python_files)
        print(f'🔍 发现 {len(naming_analysis["violations"])} 个命名违规')

        # 执行重命名
        rename_results = self._perform_renames(naming_analysis["violations"])

        # 生成报告
        report = {
            'timestamp': self._get_timestamp(),
            'analysis': naming_analysis,
            'rename_results': rename_results,
            'summary': self._generate_naming_summary(naming_analysis, rename_results)
        }

        # 保存报告
        with open('method_naming_unification_report.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False)

        print('\\n✅ 方法命名规范统一完成')
        self._print_naming_summary(report)

        return report

    def _get_python_files(self) -> List[Path]:
        """获取Python文件列表"""
        python_files = []
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files

    def _analyze_method_naming(self, files: List[Path]) -> Dict[str, Any]:
        """分析方法命名"""
        analysis = {
            'total_methods': 0,
            'snake_case_methods': 0,
            'camel_case_methods': 0,
            'violations': [],
            'rename_candidates': []
        }

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 查找方法定义
                method_matches = re.findall(r'\s+def\s+(\w+)\s*\(', content)

                for method_name in method_matches:
                    analysis['total_methods'] += 1

                    # 检查方法命名规范
                    if not method_name.startswith('_'):  # 跳过私有方法
                        if '_' in method_name and method_name.islower():
                            analysis['snake_case_methods'] += 1
                        elif not '_' in method_name and method_name.islower():
                            # 可能是需要重命名的简单方法名
                            if method_name in self.rename_map:
                                analysis['violations'].append({
                                    'file': str(file_path),
                                    'method': method_name,
                                    'current': method_name,
                                    'suggested': self.rename_map[method_name],
                                    'reason': '方法名过于简单，建议使用更描述性的名称'
                                })
                                analysis['rename_candidates'].append({
                                    'file': str(file_path),
                                    'method': method_name,
                                    'new_name': self.rename_map[method_name]
                                })
                        else:
                            analysis['camel_case_methods'] += 1

            except Exception as e:
                print(f'⚠️ 分析文件失败 {file_path}: {e}')

        return analysis

    def _perform_renames(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行重命名"""
        results = {
            'total_renames_attempted': len(violations),
            'successful_renames': 0,
            'failed_renames': 0,
            'renamed_methods': [],
            'errors': []
        }

        # 按文件分组
        renames_by_file = {}
        for violation in violations:
            file_path = violation['file']
            if file_path not in renames_by_file:
                renames_by_file[file_path] = []
            renames_by_file[file_path].append(violation)

        # 处理每个文件
        for file_path, file_renames in renames_by_file.items():
            try:
                result = self._rename_methods_in_file(file_path, file_renames)
                if result['success']:
                    results['successful_renames'] += len(file_renames)
                    results['renamed_methods'].extend(result['renamed'])
                    print(f"✅ 重命名 {file_path}: {len(file_renames)} 个方法")
                else:
                    results['failed_renames'] += len(file_renames)
                    results['errors'].extend(result['errors'])
                    print(f"❌ 重命名失败 {file_path}: {len(file_renames)} 个方法")
            except Exception as e:
                results['failed_renames'] += len(file_renames)
                results['errors'].append(f'{file_path}: {e}')
                print(f"❌ 异常 {file_path}: {e}")

        return results

    def _rename_methods_in_file(self, file_path: str, renames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """在文件中重命名方法"""
        result = {
            'success': False,
            'renamed': [],
            'errors': []
        }

        try:
            full_path = Path(file_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 备份原文件
            backup_path = self.backup_dir / f"{full_path.name}.backup"
            backup_path.write_text(content, encoding='utf-8')

            # 执行重命名
            renamed_methods = []
            for rename in renames:
                old_name = rename['method']
                new_name = rename['suggested']

                # 重命名方法定义
                def_pattern = rf'(\s+)def\s+{re.escape(old_name)}\s*\('
                def_replacement = rf'\1def {new_name}('

                if re.search(def_pattern, content):
                    content = re.sub(def_pattern, def_replacement, content)

                    # 重命名方法调用（在同一文件中）
                    call_pattern = rf'(\W){re.escape(old_name)}\s*\('
                    call_replacement = rf'\1{new_name}('
                    content = re.sub(call_pattern, call_replacement, content)

                    renamed_methods.append({
                        'old_name': old_name,
                        'new_name': new_name,
                        'file': file_path
                    })

            # 写回文件
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # 验证语法
            import py_compile
            py_compile.compile(str(full_path), doraise=True)

            result['success'] = True
            result['renamed'] = renamed_methods

        except Exception as e:
            result['errors'].append(str(e))

        return result

    def _generate_naming_summary(self, analysis: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        """生成命名总结"""
        summary = {
            'total_methods_analyzed': analysis['total_methods'],
            'snake_case_compliance': (analysis['snake_case_methods'] / analysis['total_methods'] * 100) if analysis['total_methods'] > 0 else 0,
            'violations_found': len(analysis['violations']),
            'renames_attempted': results['total_renames_attempted'],
            'renames_successful': results['successful_renames'],
            'rename_success_rate': (results['successful_renames'] / results['total_renames_attempted'] * 100) if results['total_renames_attempted'] > 0 else 0
        }

        return summary

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _print_naming_summary(self, report: Dict[str, Any]):
        """打印命名总结"""
        analysis = report['analysis']
        results = report['rename_results']
        summary = report['summary']

        print('\\n🏷️ 方法命名规范统一总结:')
        print('-' * 50)
        print(f'📊 分析方法数: {summary["total_methods_analyzed"]}')
        print('.1f')
        print(f'⚠️ 发现违规: {summary["violations_found"]}')
        print(f'🔄 重命名尝试: {summary["renames_attempted"]}')
        print(f'✅ 重命名成功: {summary["renames_successful"]}')
        print('.1f')

        print('\\n📋 主要重命名映射:')
        for old, new in list(self.rename_map.items())[:8]:  # 显示前8个
            print(f'   {old} → {new}')

        if results['renamed_methods']:
            print('\\n✅ 成功重命名的方法:')
            for method in results['renamed_methods'][:5]:  # 显示前5个
                print(f'   • {method["old_name"]} → {method["new_name"]}')

        print('\\n📄 详细报告已保存: method_naming_unification_report.json')


def main():
    """主函数"""
    unifier = MethodNamingUnifier()
    report = unifier.unify_method_names()


if __name__ == "__main__":
    main()
