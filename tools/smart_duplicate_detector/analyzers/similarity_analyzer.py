"""
相似度分析器

提供高级的代码相似度计算和分析功能。
"""

import time
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.code_fragment import CodeFragment
from ..core.similarity_metrics import SimilarityMetrics
from ..core.config import SmartDuplicateConfig
from .base_analyzer import BaseAnalyzer


class SimilarityAnalyzer(BaseAnalyzer):
    """
    相似度分析器

    计算代码片段之间的相似度，支持多种相似度算法。
    """

    def __init__(self, config: SmartDuplicateConfig):
        super().__init__(config)
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        self.metrics = SimilarityMetrics()

    def analyze(self, fragments: List[CodeFragment]) -> Dict[Tuple[int, int], float]:
        """
        分析片段间的相似度

        Args:
            fragments: 代码片段列表

        Returns:
            Dict[Tuple[int, int], float]: 相似度矩阵
        """
        start_time = time.time()

        if len(fragments) < 2:
            return {}

        similarity_matrix = {}

        if self.config.performance.parallel_processing:
            similarity_matrix = self._calculate_similarity_parallel(fragments)
        else:
            similarity_matrix = self._calculate_similarity_sequential(fragments)

        duration = time.time() - start_time
        self.logger.info(".2f"
                         f"缓存命中: {len(self.similarity_cache)}")

        return similarity_matrix

    def _calculate_similarity_sequential(self, fragments: List[CodeFragment]) -> Dict[Tuple[int, int], float]:
        """
        顺序计算相似度

        Args:
            fragments: 代码片段列表

        Returns:
            Dict[Tuple[int, int], float]: 相似度矩阵
        """
        similarity_matrix = {}
        total_comparisons = len(fragments) * (len(fragments) - 1) // 2

        comparison_count = 0
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                similarity = self._get_similarity_score(fragments[i], fragments[j])
                if similarity >= self.config.similarity.weak_clone:
                    similarity_matrix[(i, j)] = similarity

                comparison_count += 1
                if comparison_count % 1000 == 0:
                    self.logger.debug(f"已完成 {comparison_count}/{total_comparisons} 次比较")

        return similarity_matrix

    def _calculate_similarity_parallel(self, fragments: List[CodeFragment]) -> Dict[Tuple[int, int], float]:
        """
        并行计算相似度

        Args:
            fragments: 代码片段列表

        Returns:
            Dict[Tuple[int, int], float]: 相似度矩阵
        """
        similarity_matrix = {}
        total_comparisons = len(fragments) * (len(fragments) - 1) // 2

        # 生成比较任务
        tasks = []
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                tasks.append((i, j, fragments[i], fragments[j]))

        # 使用线程池执行比较
        with ThreadPoolExecutor(max_workers=self.config.performance.max_workers) as executor:
            future_to_task = {
                executor.submit(self._compare_fragments, task): task
                for task in tasks
            }

            completed_count = 0
            for future in as_completed(future_to_task):
                try:
                    i, j, similarity = future.result()
                    if similarity >= self.config.similarity.weak_clone:
                        similarity_matrix[(i, j)] = similarity

                    completed_count += 1
                    if completed_count % 1000 == 0:
                        self.logger.debug(f"已完成 {completed_count}/{total_comparisons} 次比较")

                except Exception as e:
                    self.logger.error(f"相似度计算失败: {e}")

        return similarity_matrix

    def _compare_fragments(self, task: Tuple[int, int, CodeFragment, CodeFragment]) -> Tuple[int, int, float]:
        """
        比较两个片段

        Args:
            task: (索引i, 索引j, 片段1, 片段2)

        Returns:
            Tuple[int, int, float]: (i, j, 相似度)
        """
        i, j, frag1, frag2 = task
        similarity = self._get_similarity_score(frag1, frag2)
        return i, j, similarity

    def _get_similarity_score(self, frag1: CodeFragment, frag2: CodeFragment) -> float:
        """
        获取两个片段的相似度分数

        Args:
            frag1: 片段1
            frag2: 片段2

        Returns:
            float: 相似度分数
        """
        # 创建缓存键
        cache_key = self._create_cache_key(frag1, frag2)

        # 检查缓存
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # 计算相似度
        similarity_scores = self.metrics.comprehensive_similarity(frag1, frag2)

        # 计算加权综合得分
        comprehensive_score = (
            similarity_scores.get('text', 0.0) * self.config.similarity.text_weight +
            similarity_scores.get('ast', 0.0) * self.config.similarity.ast_weight +
            similarity_scores.get('semantic', 0.0) * self.config.similarity.semantic_weight +
            similarity_scores.get('normalized', 0.0) * self.config.similarity.normalized_weight
        )

        # 缓存结果
        if len(self.similarity_cache) < self.config.performance.similarity_cache_size:
            self.similarity_cache[cache_key] = comprehensive_score

        return comprehensive_score

    def _create_cache_key(self, frag1: CodeFragment, frag2: CodeFragment) -> Tuple[str, str]:
        """
        创建缓存键

        Args:
            frag1: 片段1
            frag2: 片段2

        Returns:
            Tuple[str, str]: 有序的哈希对
        """
        hash1 = frag1.ast_hash or frag1.semantic_hash
        hash2 = frag2.ast_hash or frag2.semantic_hash

        # 确保键的顺序一致性
        if hash1 <= hash2:
            return (hash1, hash2)
        else:
            return (hash2, hash1)

    def find_similar_groups(self, fragments: List[CodeFragment],
                            similarity_matrix: Dict[Tuple[int, int], float]) -> List[List[CodeFragment]]:
        """
        从相似度矩阵中找出相似组

        Args:
            fragments: 代码片段列表
            similarity_matrix: 相似度矩阵

        Returns:
            List[List[CodeFragment]]: 相似组列表
        """
        # 使用并查集算法聚类相似片段
        parent = list(range(len(fragments)))
        rank = [0] * len(fragments)

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if rank[px] < rank[py]:
                    parent[px] = py
                elif rank[px] > rank[py]:
                    parent[py] = px
                else:
                    parent[py] = px
                    rank[px] += 1

        # 根据相似度阈值合并组
        for (i, j), similarity in similarity_matrix.items():
            if similarity >= self.config.similarity.similar_clone:
                union(i, j)

        # 收集每个组的片段
        groups = {}
        for i in range(len(fragments)):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(fragments[i])

        # 过滤掉单一片段的组
        similar_groups = [group for group in groups.values() if len(group) >= 2]

        self.logger.info(f"找到 {len(similar_groups)} 个相似组")
        return similar_groups

    def calculate_group_similarity_stats(self, groups: List[List[CodeFragment]]) -> Dict[str, Any]:
        """
        计算相似组的统计信息

        Args:
            groups: 相似组列表

        Returns:
            Dict[str, Any]: 统计信息
        """
        if not groups:
            return {
                'total_groups': 0,
                'avg_group_size': 0,
                'max_group_size': 0,
                'total_fragments': 0,
                'similarity_distribution': {}
            }

        group_sizes = [len(group) for group in groups]
        total_fragments = sum(group_sizes)

        # 计算相似度分布
        similarity_distribution = {
            'exact': 0,      # >= 0.95
            'high': 0,       # >= 0.8
            'medium': 0,     # >= 0.6
            'low': 0         # >= 0.4
        }

        for group in groups:
            # 计算组内平均相似度
            similarities = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    sim = self._get_similarity_score(group[i], group[j])
                    similarities.append(sim)

            if similarities:
                avg_similarity = sum(similarities) / len(similarities)

                if avg_similarity >= 0.95:
                    similarity_distribution['exact'] += 1
                elif avg_similarity >= 0.8:
                    similarity_distribution['high'] += 1
                elif avg_similarity >= 0.6:
                    similarity_distribution['medium'] += 1
                else:
                    similarity_distribution['low'] += 1

        return {
            'total_groups': len(groups),
            'avg_group_size': total_fragments / len(groups),
            'max_group_size': max(group_sizes),
            'total_fragments': total_fragments,
            'similarity_distribution': similarity_distribution
        }

    def get_similarity_details(self, frag1: CodeFragment,
                               frag2: CodeFragment) -> Dict[str, Any]:
        """
        获取两个片段的详细相似度信息

        Args:
            frag1: 片段1
            frag2: 片段2

        Returns:
            Dict[str, Any]: 详细相似度信息
        """
        similarity_scores = self.metrics.comprehensive_similarity(frag1, frag2)

        return {
            'overall_similarity': sum(
                similarity_scores.get(metric, 0.0) * weight
                for metric, weight in [
                    ('text', self.config.similarity.text_weight),
                    ('ast', self.config.similarity.ast_weight),
                    ('semantic', self.config.similarity.semantic_weight),
                    ('normalized', self.config.similarity.normalized_weight)
                ]
            ),
            'component_scores': similarity_scores,
            'is_exact_clone': similarity_scores.get('comprehensive', 0) >= self.config.similarity.exact_clone,
            'is_similar_clone': similarity_scores.get('comprehensive', 0) >= self.config.similarity.similar_clone,
            'is_semantic_clone': similarity_scores.get('comprehensive', 0) >= self.config.similarity.semantic_clone,
        }

    def clear_similarity_cache(self) -> None:
        """清除相似度缓存"""
        self.similarity_cache.clear()
