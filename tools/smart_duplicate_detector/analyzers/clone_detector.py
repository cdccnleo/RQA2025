"""
克隆检测器

智能代码克隆检测的核心组件，整合所有分析功能。
"""

import time
import uuid
from typing import Dict, Any, List

from ..core.config import SmartDuplicateConfig
from ..core.detection_result import DetectionResult, CloneGroup
from ..core.code_fragment import CodeFragment
from .fragment_extractor import FragmentExtractor
from .similarity_analyzer import SimilarityAnalyzer
from .base_analyzer import BaseAnalyzer


class CloneDetector(BaseAnalyzer):
    """
    智能克隆检测器

    整合片段提取、相似度分析等功能，提供完整的克隆检测解决方案。
    """

    def __init__(self, config: SmartDuplicateConfig):
        super().__init__(config)
        self.fragment_extractor = FragmentExtractor(config)
        self.similarity_analyzer = SimilarityAnalyzer(config)

    def analyze(self, target_path: str) -> DetectionResult:
        """
        执行完整的克隆检测分析

        Args:
            target_path: 检测目标路径

        Returns:
            DetectionResult: 检测结果
        """
        result = DetectionResult()
        start_time = time.time()

        try:
            self.logger.info(f"开始检测克隆代码: {target_path}")

            # 1. 提取代码片段
            self.logger.info("步骤1: 提取代码片段...")
            fragments = self.fragment_extractor.analyze(target_path)
            result.total_fragments_analyzed = len(fragments)

            if not fragments:
                self.logger.warning("未找到有效的代码片段")
                result.set_end_time()
                return result

            # 2. 计算相似度矩阵
            self.logger.info(f"步骤2: 计算相似度 (共{len(fragments)}个片段)...")
            similarity_matrix = self.similarity_analyzer.analyze(fragments)

            # 3. 识别相似组
            self.logger.info("步骤3: 识别相似组...")
            similar_groups = self.similarity_analyzer.find_similar_groups(
                fragments, similarity_matrix
            )

            # 4. 生成克隆组
            self.logger.info(f"步骤4: 生成克隆组 (共{len(similar_groups)}个相似组)...")
            clone_groups = self._create_clone_groups(similar_groups)

            # 5. 设置结果
            result.clone_groups = clone_groups
            result.total_clone_groups = len(clone_groups)
            result.analysis_duration = time.time() - start_time
            result.config_used = self.config.to_dict()

            # 获取文件统计
            python_files = self.get_python_files(target_path)
            result.total_files_analyzed = len(python_files)

            self.logger.info(f"检测完成，发现{len(clone_groups)}个克隆组")

        except Exception as e:
            self.logger.error(f"克隆检测失败: {e}")
            result.analysis_duration = time.time() - start_time

        result.set_end_time()
        return result

    def _create_clone_groups(self, similar_groups: List[List[CodeFragment]]) -> List[CloneGroup]:
        """
        从相似组创建克隆组

        Args:
            similar_groups: 相似组列表

        Returns:
            List[CloneGroup]: 克隆组列表
        """
        clone_groups = []

        for group_fragments in similar_groups:
            # 生成唯一组ID
            group_id = str(uuid.uuid4())[:8]

            # 创建克隆组
            clone_group = CloneGroup(
                group_id=group_id,
                fragments=group_fragments
            )

            # 如果启用了重构建议，分析组已经生成了建议
            # 这里可以添加额外的分析逻辑

            clone_groups.append(clone_group)

        return clone_groups

    def analyze_file_pair(self, file1: str, file2: str) -> Dict[str, Any]:
        """
        分析两个文件之间的克隆关系

        Args:
            file1: 文件1路径
            file2: 文件2路径

        Returns:
            Dict[str, Any]: 分析结果
        """
        # 提取两个文件的片段
        fragments1 = self.fragment_extractor._extract_from_file(file1)
        fragments2 = self.fragment_extractor._extract_from_file(file2)

        all_fragments = fragments1 + fragments2

        if len(all_fragments) < 2:
            return {'clones_found': 0, 'details': []}

        # 计算相似度
        similarity_matrix = {}
        for i in range(len(fragments1)):
            for j in range(len(fragments2)):
                frag1, frag2 = fragments1[i], fragments2[j]
                idx1, idx2 = i, len(fragments1) + j

                similarity = self.similarity_analyzer._get_similarity_score(frag1, frag2)
                if similarity >= self.config.similarity.weak_clone:
                    similarity_matrix[(idx1, idx2)] = similarity

        # 找出克隆对
        clones = []
        for (i, j), similarity in similarity_matrix.items():
            if i < len(fragments1) and j >= len(fragments1):
                frag1 = all_fragments[i]
                frag2 = all_fragments[j]

                clones.append({
                    'fragment1': frag1.__str__(),
                    'fragment2': frag2.__str__(),
                    'similarity': similarity,
                    'clone_type': self._classify_clone_type(similarity)
                })

        return {
            'files_analyzed': [file1, file2],
            'fragments_analyzed': len(all_fragments),
            'clones_found': len(clones),
            'details': clones
        }

    def _classify_clone_type(self, similarity: float) -> str:
        """
        根据相似度分类克隆类型

        Args:
            similarity: 相似度分数

        Returns:
            str: 克隆类型
        """
        if similarity >= self.config.similarity.exact_clone:
            return 'exact'
        elif similarity >= self.config.similarity.similar_clone:
            return 'similar'
        elif similarity >= self.config.similarity.semantic_clone:
            return 'semantic'
        else:
            return 'weak'

    def get_detection_stats(self, result: DetectionResult) -> Dict[str, Any]:
        """
        获取检测统计信息

        Args:
            result: 检测结果

        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = result.get_statistics()

        # 添加额外的统计信息
        total_lines_cloned = sum(
            sum(len(frag) for frag in group.fragments)
            for group in result.clone_groups
        )

        files_affected = set()
        for group in result.clone_groups:
            for frag in group.fragments:
                files_affected.add(frag.file_path)

        # 计算重构机会
        refactoring_opportunities = result.get_refactoring_opportunities()

        stats.update({
            'total_lines_cloned': total_lines_cloned,
            'files_affected_count': len(files_affected),
            'refactoring_opportunities_count': len(refactoring_opportunities),
            'analysis_time_seconds': result.analysis_duration,
            'fragments_per_second': result.total_fragments_analyzed / max(result.analysis_duration, 0.1),
        })

        return stats

    def export_results(self, result: DetectionResult, format: str = 'json') -> str:
        """
        导出检测结果

        Args:
            result: 检测结果
            format: 导出格式 ('json', 'xml', 'html')

        Returns:
            str: 导出的结果字符串
        """
        if format == 'json':
            import json
            return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        elif format == 'xml':
            return self._export_xml(result)
        elif format == 'html':
            return self._export_html(result)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    def _export_xml(self, result: DetectionResult) -> str:
        """导出XML格式"""
        # 简化的XML导出实现
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<clone_detection>']

        for group in result.clone_groups:
            xml_parts.append(f'  <clone_group id="{group.group_id}" type="{group.clone_type}">')
            for frag in group.fragments:
                xml_parts.append(f'    <fragment file="{frag.file_path}" '
                                 f'start="{frag.start_line}" end="{frag.end_line}"/>')
            xml_parts.append('  </clone_group>')

        xml_parts.append('</clone_detection>')
        return '\n'.join(xml_parts)

    def _export_html(self, result: DetectionResult) -> str:
        """导出HTML格式"""
        # 简化的HTML导出实现
        html_parts = [
            '<!DOCTYPE html>',
            '<html><head><title>代码克隆检测报告</title></head><body>',
            '<h1>代码克隆检测报告</h1>',
            f'<p>共发现 {len(result.clone_groups)} 个克隆组</p>',
            '<table border="1"><tr><th>组ID</th><th>类型</th><th>片段数量</th><th>相似度</th></tr>'
        ]

        for group in result.clone_groups:
            html_parts.append(
                f'<tr><td>{group.group_id}</td><td>{group.clone_type}</td>'
                f'<td>{len(group)}</td><td>{group.similarity_score:.2f}</td></tr>'
            )

        html_parts.extend(['</table>', '</body></html>'])
        return '\n'.join(html_parts)
