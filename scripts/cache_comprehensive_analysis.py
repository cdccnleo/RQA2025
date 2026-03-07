#!/usr/bin/env python3
"""
Cache目录全面分析和优化脚本

分析所有重复的模板文件，制定优化策略
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any


class CacheDirectoryAnalyzer:
    """Cache目录分析器"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.analysis_results = {}

    def analyze_file_patterns(self) -> Dict[str, Any]:
        """分析文件命名模式"""
        print("🔍 分析文件命名模式...")

        patterns = {
            'cache_files': [],
            'client_files': [],
            'service_files': [],
            'strategy_files': [],
            'optimizer_files': [],
            'other_files': []
        }

        for file_path in self.cache_dir.glob("*.py"):
            if file_path.name.startswith('__'):
                continue

            filename = file_path.name

            if re.match(r'cache_\d+\.py$', filename):
                patterns['cache_files'].append(filename)
            elif re.match(r'client_\d+\.py$', filename):
                patterns['client_files'].append(filename)
            elif re.match(r'service_\d+\.py$', filename):
                patterns['service_files'].append(filename)
            elif re.match(r'strategy_\d+\.py$', filename):
                patterns['strategy_files'].append(filename)
            elif re.match(r'optimizer_\d+\.py$', filename):
                patterns['optimizer_files'].append(filename)
            else:
                patterns['other_files'].append(filename)

        return patterns

    def analyze_file_sizes(self) -> Dict[str, List[Dict[str, Any]]]:
        """分析文件大小分布"""
        print("📊 分析文件大小...")

        size_analysis = {
            'template_files': [],  # < 2KB, 可能是模板
            'small_files': [],     # 2-10KB, 小文件
            'medium_files': [],    # 10-50KB, 中等文件
            'large_files': [],     # > 50KB, 大文件
            'functional_files': []  # 有实际功能的文件
        }

        for file_path in self.cache_dir.glob("*.py"):
            if file_path.name.startswith('__'):
                continue

            size_kb = file_path.stat().st_size / 1024
            file_info = {
                'name': file_path.name,
                'size_kb': round(size_kb, 1),
                'path': str(file_path)
            }

            if size_kb < 2:
                size_analysis['template_files'].append(file_info)
            elif size_kb < 10:
                size_analysis['small_files'].append(file_info)
            elif size_kb < 50:
                size_analysis['medium_files'].append(file_info)
            else:
                size_analysis['large_files'].append(file_info)

            # 检查是否有实际功能（通过文件大小和内容特征）
            if self._is_functional_file(file_path):
                size_analysis['functional_files'].append(file_info)

        return size_analysis

    def _is_functional_file(self, file_path: Path) -> bool:
        """判断文件是否有实际功能"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()

            # 功能性文件的特征
            functional_indicators = [
                'async def', 'await', 'class.*:', 'def.*:', 'import.*',
                'from.*import', 'try:', 'except:', 'if __name__',
                'logging', 'config', 'database', 'redis', 'memory'
            ]

            indicator_count = sum(1 for indicator in functional_indicators if indicator in content)
            return indicator_count > 3 or len(content) > 1000

        except:
            return False

    def analyze_content_similarity(self) -> Dict[str, Any]:
        """分析内容相似性"""
        print("🔍 分析内容相似性...")

        similarity_analysis = {
            'identical_groups': [],  # 完全相同的文件组
            'similar_groups': [],    # 相似的文件组
            'unique_files': []       # 独特的文件
        }

        # 获取所有模板化文件的哈希
        file_hashes = {}
        for file_path in self.cache_dir.glob("*_*.py"):
            if file_path.name.startswith('__'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_hash = hash(content)
                    if file_hash not in file_hashes:
                        file_hashes[file_hash] = []
                    file_hashes[file_hash].append(file_path.name)
            except:
                continue

        # 分析哈希组
        for file_hash, files in file_hashes.items():
            if len(files) > 1:
                similarity_analysis['identical_groups'].append({
                    'hash': file_hash,
                    'files': files,
                    'count': len(files)
                })

        return similarity_analysis

    def generate_optimization_plan(self) -> Dict[str, Any]:
        """生成优化计划"""
        print("📋 生成优化计划...")

        patterns = self.analyze_file_patterns()
        sizes = self.analyze_file_sizes()
        similarities = self.analyze_content_similarity()

        plan = {
            'current_state': {
                'total_files': len(list(self.cache_dir.glob("*.py"))),
                'pattern_distribution': {k: len(v) for k, v in patterns.items()},
                'size_distribution': {k: len(v) for k, v in sizes.items()}
            },
            'optimization_targets': {
                'template_files_to_remove': [f['name'] for f in sizes['template_files'] if f['name'] not in ['cache_components.py']],
                'files_to_merge': {},
                'files_to_keep': [f['name'] for f in sizes['functional_files']]
            },
            'estimated_benefits': {
                'files_to_remove': len(sizes['template_files']),
                'code_reduction_percent': 0,
                'maintenance_cost_reduction': 0
            }
        }

        # 计算优化效益
        template_size = sum(f['size_kb'] for f in sizes['template_files'])
        total_size = sum(f['size_kb'] for files in sizes.values() for f in files)
        if total_size > 0:
            plan['estimated_benefits']['code_reduction_percent'] = round(
                (template_size / total_size) * 100, 1)

        plan['estimated_benefits']['maintenance_cost_reduction'] = min(
            80, plan['estimated_benefits']['code_reduction_percent'] * 0.8)

        return plan

    def run_analysis(self):
        """运行完整分析"""
        print("🚀 开始Cache目录全面分析...")
        print("="*60)

        try:
            patterns = self.analyze_file_patterns()
            sizes = self.analyze_file_sizes()
            similarities = self.analyze_content_similarity()
            plan = self.generate_optimization_plan()

            print("\n" + "="*60)
            print("✅ Cache目录分析完成！")
            print("="*60)

            print("\n📊 当前状态:")
            print(f"   总文件数: {plan['current_state']['total_files']}")
            print(f"   文件类型分布: {plan['current_state']['pattern_distribution']}")
            print(f"   文件大小分布: {plan['current_state']['size_distribution']}")

            print("\n🎯 优化目标:")
            print(f"   待删除模板文件: {len(plan['optimization_targets']['template_files_to_remove'])}个")
            print(f"   保留功能文件: {len(plan['optimization_targets']['files_to_keep'])}个")

            # 更新统计信息
            plan['optimization_targets']['files_to_remove'] = len(
                plan['optimization_targets']['template_files_to_remove'])

            print("\n📈 预期效益:")
            print(f"   代码量减少: {plan['estimated_benefits']['code_reduction_percent']}%")
            print(f"   维护成本降低: {plan['estimated_benefits']['maintenance_cost_reduction']}%")

            return plan

        except Exception as e:
            print(f"\n❌ 分析过程中出错: {e}")
            return None


def main():
    """主函数"""
    cache_dir = "src/infrastructure/cache"

    if not os.path.exists(cache_dir):
        print("❌ Cache目录不存在")
        return

    analyzer = CacheDirectoryAnalyzer(cache_dir)
    result = analyzer.run_analysis()

    if result:
        print("\n🎉 Cache目录分析成功完成！")
        print(f"发现 {result['optimization_targets']['files_to_remove']} 个可优化模板文件")
    else:
        print("\n❌ Cache目录分析失败！")


if __name__ == "__main__":
    main()
