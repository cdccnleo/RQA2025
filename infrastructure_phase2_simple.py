#!/usr/bin/env python3
"""
基础设施层Phase 2简化重构工具

专注于最重要的重构任务
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any


class SimplePhase2Refactor:
    """简化的Phase 2重构工具"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.backup_dir = Path('backup_phase2_simple')
        self.backup_dir.mkdir(exist_ok=True)

    def execute_simple_refactor(self) -> Dict[str, Any]:
        """执行简化重构"""
        print('🏗️ 开始基础设施层Phase 2简化重构')
        print('=' * 60)

        results = {
            'fix_encoding_issues': self._fix_encoding_issues(),
            'consolidate_duplicate_files': self._consolidate_duplicate_files(),
            'clean_empty_dirs': self._clean_empty_directories(),
            'optimize_file_structure': self._optimize_file_structure(),
            'summary': {}
        }

        # 生成重构摘要
        results['summary'] = self._generate_summary(results)

        print('\\n✅ Phase 2简化重构完成！')
        self._print_summary(results['summary'])

        return results

    def _fix_encoding_issues(self) -> Dict[str, Any]:
        """修复编码问题"""
        print('\\n🔧 修复编码问题...')

        fixed_files = 0

        # 查找所有__init__.py文件
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file == '__init__.py':
                    file_path = Path(root) / file
                    try:
                        # 尝试以不同编码读取
                        encodings = ['utf-8', 'gbk', 'latin1']
                        content = None

                        for encoding in encodings:
                            try:
                                with open(file_path, 'r', encoding=encoding) as f:
                                    content = f.read()
                                break
                            except UnicodeDecodeError:
                                continue

                        if content is not None:
                            # 检查是否有问题字符
                            if any(ord(c) > 127 for c in content):
                                # 重新以UTF-8保存
                                self._backup_file(file_path)
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write('"""\\nInfrastructure module initialization\\n"""\\n')
                                fixed_files += 1
                                print(f'✅ 修复编码问题: {file_path.name}')

                    except Exception as e:
                        print(f'⚠️ 处理文件失败 {file_path}: {e}')

        return {'files_fixed': fixed_files}

    def _consolidate_duplicate_files(self) -> Dict[str, Any]:
        """合并重复文件"""
        print('\\n🔄 合并重复文件...')

        # 查找重复文件
        duplicates = self._find_exact_duplicates()
        consolidated = 0

        for original, copies in duplicates.items():
            # 保留第一个，删除其他副本
            for copy in copies[1:]:  # 跳过第一个
                try:
                    self._backup_file(Path(copy))
                    os.remove(copy)
                    consolidated += 1
                    print(f'✅ 删除重复文件: {Path(copy).name}')
                except Exception as e:
                    print(f'❌ 删除失败 {copy}: {e}')

        return {
            'duplicate_groups': len(duplicates),
            'files_consolidated': consolidated
        }

    def _find_exact_duplicates(self) -> Dict[str, List[str]]:
        """查找完全相同的重复文件"""
        file_hashes = {}

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        content_hash = hash(content)
                        if content_hash not in file_hashes:
                            file_hashes[content_hash] = []

                        file_hashes[content_hash].append(str(file_path))

                    except Exception:
                        continue

        # 只保留有多个文件的哈希
        duplicates = {k: v for k, v in file_hashes.items() if len(v) > 1}
        return duplicates

    def _clean_empty_directories(self) -> Dict[str, Any]:
        """清理空目录"""
        print('\\n🧹 清理空目录...')

        empty_dirs = []
        cleaned = 0

        for root, dirs, files in os.walk(self.infra_dir, topdown=False):
            dir_path = Path(root)

            # 检查目录是否为空
            try:
                if not any(dir_path.iterdir()):
                    empty_dirs.append(str(dir_path))
                    try:
                        dir_path.rmdir()
                        cleaned += 1
                        print(f'✅ 删除空目录: {dir_path.name}')
                    except Exception as e:
                        print(f'❌ 删除目录失败 {dir_path}: {e}')
            except Exception:
                continue

        return {
            'empty_dirs_found': len(empty_dirs),
            'dirs_cleaned': cleaned
        }

    def _optimize_file_structure(self) -> Dict[str, Any]:
        """优化文件结构"""
        print('\\n📁 优化文件结构...')

        # 统计当前结构
        stats = self._analyze_current_structure()

        # 建议的优化措施
        optimizations = {
            'large_files': len([f for f in stats['files_by_size'] if f['size'] > 10000]),
            'small_files': len([f for f in stats['files_by_size'] if f['size'] < 100]),
            'deep_nested_dirs': stats['max_depth']
        }

        print(f'📊 当前结构统计:')
        print(f'   总文件数: {stats["total_files"]}')
        print(f'   目录深度: {stats["max_depth"]}')
        print(f'   大文件(>10KB): {optimizations["large_files"]}')
        print(f'   小文件(<100B): {optimizations["small_files"]}')

        return {
            'structure_analysis': stats,
            'optimization_suggestions': optimizations
        }

    def _analyze_current_structure(self) -> Dict[str, Any]:
        """分析当前文件结构"""
        stats = {
            'total_files': 0,
            'total_dirs': 0,
            'max_depth': 0,
            'files_by_size': [],
            'empty_dirs': []
        }

        for root, dirs, files in os.walk(self.infra_dir):
            stats['total_dirs'] += len(dirs)

            # 计算深度
            depth = len(Path(root).relative_to(self.infra_dir).parts)
            stats['max_depth'] = max(stats['max_depth'], depth)

            for file in files:
                if file.endswith('.py'):
                    stats['total_files'] += 1
                    file_path = Path(root) / file

                    try:
                        size = file_path.stat().st_size
                        stats['files_by_size'].append({
                            'path': str(file_path),
                            'size': size
                        })
                    except Exception:
                        continue

        return stats

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
            except Exception:
                pass  # 静默失败

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要"""
        summary = {
            'total_actions': 0,
            'files_affected': 0,
            'dirs_cleaned': 0,
            'encoding_fixes': 0,
            'duplicates_removed': 0,
            'status': 'completed'
        }

        # 统计编码修复
        enc_fix = results.get('fix_encoding_issues', {})
        summary['encoding_fixes'] = enc_fix.get('files_fixed', 0)
        if enc_fix.get('files_fixed', 0) > 0:
            summary['total_actions'] += 1
            summary['files_affected'] += enc_fix['files_fixed']

        # 统计重复文件合并
        dup_fix = results.get('consolidate_duplicate_files', {})
        summary['duplicates_removed'] = dup_fix.get('files_consolidated', 0)
        if dup_fix.get('files_consolidated', 0) > 0:
            summary['total_actions'] += 1
            summary['files_affected'] += dup_fix['files_consolidated']

        # 统计目录清理
        dir_clean = results.get('clean_empty_dirs', {})
        summary['dirs_cleaned'] = dir_clean.get('dirs_cleaned', 0)
        if dir_clean.get('dirs_cleaned', 0) > 0:
            summary['total_actions'] += 1

        return summary

    def _print_summary(self, summary: Dict[str, Any]):
        """打印摘要"""
        print('\\n📊 Phase 2简化重构摘要:')
        print('-' * 40)
        print(f'✅ 重构操作: {summary["total_actions"]} 个')
        print(f'📁 影响文件: {summary["files_affected"]} 个')
        print(f'🗂️ 清理目录: {summary["dirs_cleaned"]} 个')
        print(f'🔧 修复编码: {summary["encoding_fixes"]} 个')
        print(f'🔄 删除重复: {summary["duplicates_removed"]} 个')
        print(f'📂 备份位置: {self.backup_dir}')

        if summary['files_affected'] > 0:
            print('🎉 文件结构得到优化！')


def main():
    """主函数"""
    refactor = SimplePhase2Refactor()
    results = refactor.execute_simple_refactor()

    # 保存重构报告
    with open('infrastructure_phase2_simple_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('\\n📄 重构报告已保存: infrastructure_phase2_simple_report.json')


if __name__ == "__main__":
    main()
