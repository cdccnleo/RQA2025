"""
检测结果数据结构

定义克隆组和检测结果的表示。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from datetime import datetime
from .code_fragment import CodeFragment


@dataclass
class CloneGroup:
    """
    克隆代码组

    表示一组相似的代码片段。
    """

    group_id: str
    fragments: List[CodeFragment] = field(default_factory=list)
    similarity_score: float = 0.0
    clone_type: str = "unknown"  # exact, similar, semantic

    # 分析结果
    analysis: Dict[str, Any] = field(default_factory=dict)

    # 重构建议
    refactoring_suggestions: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """后初始化处理"""
        self._analyze_group()

    def _analyze_group(self) -> None:
        """分析克隆组"""
        if len(self.fragments) < 2:
            return

        # 计算平均相似度
        total_similarity = 0.0
        comparisons = 0

        for i, frag1 in enumerate(self.fragments):
            for frag2 in self.fragments[i+1:]:
                similarity = frag1.get_similarity_score(frag2)
                total_similarity += similarity
                comparisons += 1

        if comparisons > 0:
            self.similarity_score = total_similarity / comparisons

        # 确定克隆类型
        if self.similarity_score >= 0.95:
            self.clone_type = "exact"
        elif self.similarity_score >= 0.8:
            self.clone_type = "similar"
        elif self.similarity_score >= 0.6:
            self.clone_type = "semantic"
        else:
            self.clone_type = "weak"

        # 生成重构建议
        self._generate_refactoring_suggestions()

    def _generate_refactoring_suggestions(self) -> None:
        """生成重构建议"""
        suggestions = []

        # 公共方法提取建议
        if len(self.fragments) >= 3:
            suggestions.append({
                'type': 'extract_method',
                'description': f'将重复代码提取为公共方法，共{len(self.fragments)}处使用',
                'impact': 'high',
                'complexity': 'medium'
            })

        # 父类方法建议
        if all(frag.fragment_type.value == 'method' for frag in self.fragments):
            suggestions.append({
                'type': 'extract_superclass_method',
                'description': '将重复方法提取到父类中',
                'impact': 'high',
                'complexity': 'low'
            })

        # 工具类建议
        if len(self.fragments) >= 4:
            suggestions.append({
                'type': 'create_utility_class',
                'description': '创建工具类封装重复逻辑',
                'impact': 'medium',
                'complexity': 'high'
            })

        # 配置文件建议
        if self._is_configuration_code():
            suggestions.append({
                'type': 'extract_configuration',
                'description': '将配置逻辑提取到配置文件中',
                'impact': 'medium',
                'complexity': 'low'
            })

        self.refactoring_suggestions = suggestions

    def _is_configuration_code(self) -> bool:
        """判断是否为配置代码"""
        config_keywords = ['config', 'setting', 'parameter', 'option']
        for fragment in self.fragments:
            content_lower = fragment.raw_content.lower()
            if any(keyword in content_lower for keyword in config_keywords):
                return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        """获取组摘要"""
        return {
            'group_id': self.group_id,
            'fragment_count': len(self.fragments),
            'similarity_score': self.similarity_score,
            'clone_type': self.clone_type,
            'files': list(set(frag.file_path for frag in self.fragments)),
            'line_range': f"{min(f.start_line for f in self.fragments)}-{max(f.end_line for f in self.fragments)}",
            'total_lines': sum(len(frag) for frag in self.fragments),
            'suggestion_count': len(self.refactoring_suggestions)
        }

    def __len__(self) -> int:
        return len(self.fragments)


@dataclass
class DetectionResult:
    """
    检测结果

    包含所有发现的克隆组和统计信息。
    """

    clone_groups: List[CloneGroup] = field(default_factory=list)
    detection_time: datetime = field(default_factory=datetime.now)

    # 统计信息
    total_files_analyzed: int = 0
    total_fragments_analyzed: int = 0
    total_clone_groups: int = 0

    # 性能指标
    analysis_duration: float = 0.0
    memory_usage: float = 0.0

    # 配置信息
    config_used: Dict[str, Any] = field(default_factory=dict)

    def add_clone_group(self, group: CloneGroup) -> None:
        """添加克隆组"""
        self.clone_groups.append(group)
        self.total_clone_groups = len(self.clone_groups)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.clone_groups:
            return {
                'total_groups': 0,
                'total_fragments': 0,
                'avg_similarity': 0.0,
                'clone_types': {},
                'files_affected': 0
            }

        # 计算各种统计
        total_fragments = sum(len(group) for group in self.clone_groups)
        avg_similarity = sum(
            group.similarity_score for group in self.clone_groups) / len(self.clone_groups)

        clone_types = {}
        files_affected = set()

        for group in self.clone_groups:
            clone_types[group.clone_type] = clone_types.get(group.clone_type, 0) + 1
            files_affected.update(frag.file_path for frag in group.fragments)

        return {
            'total_groups': len(self.clone_groups),
            'total_fragments': total_fragments,
            'avg_similarity': avg_similarity,
            'clone_types': clone_types,
            'files_affected': len(files_affected),
            'avg_group_size': total_fragments / len(self.clone_groups),
            'exact_clones': clone_types.get('exact', 0),
            'similar_clones': clone_types.get('similar', 0),
            'semantic_clones': clone_types.get('semantic', 0),
        }

    def get_top_clone_groups(self, limit: int = 10) -> List[CloneGroup]:
        """获取最大的克隆组"""
        return sorted(self.clone_groups,
                      key=lambda g: len(g) * g.similarity_score,
                      reverse=True)[:limit]

    def filter_by_similarity(self, min_similarity: float) -> List[CloneGroup]:
        """按相似度过滤克隆组"""
        return [group for group in self.clone_groups
                if group.similarity_score >= min_similarity]

    def filter_by_type(self, clone_type: str) -> List[CloneGroup]:
        """按克隆类型过滤"""
        return [group for group in self.clone_groups
                if group.clone_type == clone_type]

    def get_refactoring_opportunities(self) -> List[Dict[str, Any]]:
        """获取重构机会"""
        opportunities = []

        for group in self.clone_groups:
            if group.refactoring_suggestions:
                opportunities.extend([{
                    'group_id': group.group_id,
                    'similarity': group.similarity_score,
                    'fragment_count': len(group),
                    **suggestion
                } for suggestion in group.refactoring_suggestions])

        return sorted(opportunities,
                      key=lambda x: (x['impact'] == 'high', x['fragment_count']),
                      reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'detection_time': self.detection_time.isoformat(),
            'total_files_analyzed': self.total_files_analyzed,
            'total_fragments_analyzed': self.total_fragments_analyzed,
            'total_clone_groups': self.total_clone_groups,
            'analysis_duration': self.analysis_duration,
            'memory_usage': self.memory_usage,
            'config_used': self.config_used,
            'statistics': self.get_statistics(),
            'clone_groups': [group.get_summary() for group in self.clone_groups],
            'refactoring_opportunities': self.get_refactoring_opportunities()
        }

    def set_end_time(self) -> None:
        """设置结束时间（兼容性方法）"""
        # 这个方法主要用于兼容旧的CheckResult接口

    def __len__(self) -> int:
        return len(self.clone_groups)
