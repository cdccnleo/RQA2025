#!/usr/bin/env python3
"""
代码文件组织优化工具

合并过小的文件，重组目录结构
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


class FileOrganizationOptimizer:
    """文件组织优化器"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.backup_dir = Path('file_organization_backup')
        self.small_file_threshold = 50  # 少于50行的文件认为过小
        self.merge_candidates = []

    def optimize_file_organization(self) -> Dict[str, Any]:
        """优化文件组织"""
        print('📦 开始文件组织优化')
        print('=' * 50)

        # 创建备份
        self.backup_dir.mkdir(exist_ok=True)

        # 分析文件大小分布
        file_analysis = self._analyze_file_sizes()
        print(f'📊 分析了 {file_analysis["total_files"]} 个文件')

        # 识别可合并的文件
        merge_plan = self._identify_merge_candidates(file_analysis)
        print(f'🔍 发现 {len(merge_plan["merge_groups"])} 个合并组')

        # 执行合并
        merge_results = self._execute_merges(merge_plan)

        # 清理空目录
        cleanup_results = self._cleanup_empty_directories()

        # 生成优化报告
        optimization_report = {
            'timestamp': self._get_timestamp(),
            'file_analysis': file_analysis,
            'merge_plan': merge_plan,
            'merge_results': merge_results,
            'cleanup_results': cleanup_results,
            'optimization_summary': self._generate_optimization_summary(
                file_analysis, merge_results, cleanup_results
            )
        }

        # 保存报告
        with open('file_organization_optimization_report.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(optimization_report, f, indent=2, ensure_ascii=False)

        print('\\n✅ 文件组织优化完成')
        self._print_optimization_summary(optimization_report)

        return optimization_report

    def _analyze_file_sizes(self) -> Dict[str, Any]:
        """分析文件大小"""
        analysis = {
            'total_files': 0,
            'size_distribution': {
                'tiny': [],      # < 10行
                'small': [],     # 10-50行
                'medium': [],    # 50-200行
                'large': [],     # 200-500行
                'huge': []       # > 500行
            },
            'empty_files': [],
            'binary_files': []
        }

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    analysis['total_files'] += 1

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        lines = content.split('\n')
                        line_count = len([line for line in lines if line.strip()])

                        # 分类文件大小
                        if line_count < 10:
                            analysis['size_distribution']['tiny'].append({
                                'path': str(file_path),
                                'lines': line_count,
                                'size_kb': file_path.stat().st_size / 1024
                            })
                        elif line_count < 50:
                            analysis['size_distribution']['small'].append({
                                'path': str(file_path),
                                'lines': line_count,
                                'size_kb': file_path.stat().st_size / 1024
                            })
                        elif line_count < 200:
                            analysis['size_distribution']['medium'].append({
                                'path': str(file_path),
                                'lines': line_count,
                                'size_kb': file_path.stat().st_size / 1024
                            })
                        elif line_count < 500:
                            analysis['size_distribution']['large'].append({
                                'path': str(file_path),
                                'lines': line_count,
                                'size_kb': file_path.stat().st_size / 1024
                            })
                        else:
                            analysis['size_distribution']['huge'].append({
                                'path': str(file_path),
                                'lines': line_count,
                                'size_kb': file_path.stat().st_size / 1024
                            })

                        # 检查空文件
                        if line_count == 0:
                            analysis['empty_files'].append(str(file_path))

                    except UnicodeDecodeError:
                        analysis['binary_files'].append(str(file_path))
                    except Exception as e:
                        print(f'⚠️ 分析文件失败 {file_path}: {e}')

        return analysis

    def _identify_merge_candidates(self, file_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """识别可合并的文件"""
        merge_plan = {
            'merge_groups': [],
            'merge_candidates': [],
            'unmergeable_files': []
        }

        # 找出小文件
        small_files = (file_analysis['size_distribution']['tiny'] +
                       file_analysis['size_distribution']['small'])

        # 按目录分组
        files_by_dir = defaultdict(list)
        for file_info in small_files:
            dir_path = str(Path(file_info['path']).parent)
            files_by_dir[dir_path].append(file_info)

        # 为每个目录创建合并组
        for dir_path, files in files_by_dir.items():
            if len(files) >= 2:  # 至少2个文件才能合并
                merge_group = {
                    'directory': dir_path,
                    'files': files,
                    'total_lines': sum(f['lines'] for f in files),
                    'total_size_kb': sum(f['size_kb'] for f in files),
                    'merge_strategy': self._determine_merge_strategy(files),
                    'target_file': self._suggest_target_filename(dir_path, files)
                }
                merge_plan['merge_groups'].append(merge_group)
            else:
                merge_plan['unmergeable_files'].extend([f['path'] for f in files])

        return merge_plan

    def _determine_merge_strategy(self, files: List[Dict[str, Any]]) -> str:
        """确定合并策略"""
        # 分析文件内容相似性
        file_types = set()
        for file_info in files:
            file_path = Path(file_info['path'])
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 判断文件类型
                if 'class ' in content:
                    file_types.add('class')
                elif 'def ' in content and 'class ' not in content:
                    file_types.add('function')
                else:
                    file_types.add('utility')
            except:
                file_types.add('unknown')

        if len(file_types) == 1:
            if 'class' in file_types:
                return 'merge_classes'
            elif 'function' in file_types:
                return 'merge_functions'
            else:
                return 'merge_utilities'
        else:
            return 'create_module'

    def _suggest_target_filename(self, dir_path: str, files: List[Dict[str, Any]]) -> str:
        """建议目标文件名"""
        dir_name = Path(dir_path).name

        # 根据目录名和文件内容建议合并后的文件名
        if 'interfaces' in dir_name:
            return f"{dir_path}/combined_interfaces.py"
        elif 'utils' in dir_name or 'helpers' in dir_name:
            return f"{dir_path}/utilities.py"
        elif 'core' in dir_name:
            return f"{dir_path}/core_components.py"
        else:
            return f"{dir_path}/combined_{dir_name}.py"

    def _execute_merges(self, merge_plan: Dict[str, Any]) -> Dict[str, Any]:
        """执行合并"""
        merge_results = {
            'successful_merges': [],
            'failed_merges': [],
            'total_files_merged': 0,
            'total_lines_saved': 0
        }

        for merge_group in merge_plan['merge_groups']:
            try:
                result = self._merge_file_group(merge_group)
                if result['success']:
                    merge_results['successful_merges'].append(result)
                    merge_results['total_files_merged'] += len(merge_group['files'])
                    merge_results['total_lines_saved'] += result.get('lines_saved', 0)
                else:
                    merge_results['failed_merges'].append(result)
            except Exception as e:
                merge_results['failed_merges'].append({
                    'group': merge_group,
                    'error': str(e)
                })

        return merge_results

    def _merge_file_group(self, merge_group: Dict[str, Any]) -> Dict[str, Any]:
        """合并一组文件"""
        result = {
            'success': False,
            'target_file': merge_group['target_file'],
            'merged_files': [f['path'] for f in merge_group['files']],
            'strategy': merge_group['merge_strategy']
        }

        target_path = Path(merge_group['target_file'])
        target_path.parent.mkdir(parents=True, exist_ok=True)

        merged_content = []
        total_lines = 0

        # 合并文件内容
        for file_info in merge_group['files']:
            file_path = Path(file_info['path'])

            # 备份原文件
            backup_path = self.backup_dir / f"{file_path.name}.backup"
            if file_path.exists():
                shutil.copy2(file_path, backup_path)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 添加文件分割注释
                merged_content.append(f"\n# ===== Merged from: {file_path.name} =====\n")
                merged_content.append(content)
                total_lines += len(content.split('\n'))

            except Exception as e:
                result['error'] = f'读取文件失败 {file_path}: {e}'
                return result

        # 写入合并后的文件
        try:
            final_content = ''.join(merged_content)
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(final_content)

            # 删除原文件
            for file_info in merge_group['files']:
                file_path = Path(file_info['path'])
                if file_path.exists():
                    file_path.unlink()

            result['success'] = True
            result['lines_saved'] = total_lines
            result['final_file'] = str(target_path)

        except Exception as e:
            result['error'] = f'写入合并文件失败: {e}'

        return result

    def _cleanup_empty_directories(self) -> Dict[str, Any]:
        """清理空目录"""
        cleanup_results = {
            'removed_directories': [],
            'errors': []
        }

        # 递归清理空目录
        for root, dirs, files in os.walk(self.infra_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.rglob('*')):  # 检查目录是否为空
                        dir_path.rmdir()
                        cleanup_results['removed_directories'].append(str(dir_path))
                        print(f'🗑️ 删除空目录: {dir_path}')
                except Exception as e:
                    cleanup_results['errors'].append(f'删除目录失败 {dir_path}: {e}')

        return cleanup_results

    def _generate_optimization_summary(self, file_analysis: Dict[str, Any],
                                       merge_results: Dict[str, Any],
                                       cleanup_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成优化总结"""
        summary = {
            'original_file_count': file_analysis['total_files'],
            'files_merged': merge_results['total_files_merged'],
            'directories_removed': len(cleanup_results['removed_directories']),
            'final_file_count': file_analysis['total_files'] - merge_results['total_files_merged'],
            'optimization_metrics': {
                'merge_success_rate': (len(merge_results['successful_merges']) /
                                       len(merge_results['successful_merges'] +
                                           merge_results['failed_merges'])
                                       if merge_results['successful_merges'] or merge_results['failed_merges'] else 0),
                'lines_saved': merge_results['total_lines_saved'],
                'empty_dirs_cleaned': len(cleanup_results['removed_directories'])
            }
        }

        return summary

    def _get_timestamp(self) -> str:
        """获取时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _print_optimization_summary(self, report: Dict[str, Any]):
        """打印优化摘要"""
        summary = report['optimization_summary']

        print('\\n📦 文件组织优化摘要:')
        print('-' * 40)
        print(f'📁 原始文件数: {summary["original_file_count"]}')
        print(f'🔀 合并文件数: {summary["files_merged"]}')
        print(f'🗑️ 删除目录数: {summary["directories_removed"]}')
        print(f'📄 最终文件数: {summary["final_file_count"]}')

        metrics = summary['optimization_metrics']
        print(f'\\n📊 优化指标:')
        print('.1%')
        print(f'   节省行数: {metrics["lines_saved"]}')
        print(f'   清理空目录: {metrics["empty_dirs_cleaned"]}')

        if report['merge_results']['successful_merges']:
            print('\\n✅ 成功合并:')
            for merge in report['merge_results']['successful_merges'][:3]:
                print(f'   • {merge["target_file"]} ({len(merge["merged_files"])} 个文件)')

        if report['merge_results']['failed_merges']:
            print('\\n❌ 合并失败:')
            for failed in report['merge_results']['failed_merges'][:3]:
                print(f'   • {failed.get("target_file", "未知")}')

        print('\\n📄 详细报告已保存: file_organization_optimization_report.json')


def main():
    """主函数"""
    optimizer = FileOrganizationOptimizer()
    report = optimizer.optimize_file_organization()


if __name__ == "__main__":
    main()
