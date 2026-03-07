"""
反馈分析器组件
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass

from ...base import BaseComponent

logger = logging.getLogger(__name__)


@dataclass
class FeedbackItem:
    """反馈项"""

    id: str
    user: str
    category: str
    content: str
    rating: int
    timestamp: float
    status: str = "pending"


@dataclass
class PerformanceMetric:
    """性能指标"""

    name: str
    value: float
    unit: str
    timestamp: float
    category: str = "general"


class FeedbackAnalyzer(BaseComponent):
    """反馈分析器"""

    def __init__(self):
        super().__init__("FeedbackAnalyzer")
        logger.info("反馈分析器初始化完成")

    def shutdown(self) -> bool:
        """关闭反馈分析器"""
        try:
            logger.info("开始关闭反馈分析器")
            logger.info("反馈分析器关闭完成")
            return True
        except Exception as e:
            logger.error(f"关闭反馈分析器失败: {e}")
            return False

    def analyze_feedback(self, feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析反馈"""
        logger.info(f"开始分析 {len(feedback)} 条反馈")

        if not feedback:
            return {"analysis": "no_feedback", "suggestions": []}

        # 按类别分组
        categories = {}
        ratings = []

        for item in feedback:
            category = item.get("category", "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
            ratings.append(item.get("rating", 0))

        # 计算统计信息
        analysis = {
            "total_feedback": len(feedback),
            "categories": {cat: len(items) for cat, items in categories.items()},
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "rating_distribution": self._calculate_rating_distribution(ratings),
            "top_concerns": self._identify_top_concerns(categories),
            "improvement_areas": self._identify_improvement_areas(categories),
        }

        logger.info(
            f"反馈分析完成: {analysis['total_feedback']} 条反馈，平均评分 {analysis['average_rating']:.2f}"
        )

        return analysis

    def _calculate_rating_distribution(self, ratings: List[int]) -> Dict[int, int]:
        """计算评分分布"""
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for rating in ratings:
            if rating in distribution:
                distribution[rating] += 1
        return distribution

    def _identify_top_concerns(
        self, categories: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """识别主要关注点"""
        concerns = []
        for category, items in categories.items():
            low_ratings = [item for item in items if item.get("rating", 0) <= 3]
            if low_ratings:
                concerns.append(f"{category}: {len(low_ratings)} 条低评分反馈")
        return concerns

    def _identify_improvement_areas(
        self, categories: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """识别改进领域"""
        areas = []
        for category, items in categories.items():
            avg_rating = sum(item.get("rating", 0) for item in items) / len(items)
            if avg_rating < 4.0:
                areas.append(f"{category} (平均评分: {avg_rating:.2f})")
        return areas

    def generate_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        suggestions = []

        # 基于评分分布生成建议
        rating_dist = analysis.get("rating_distribution", {})
        if rating_dist.get(1, 0) + rating_dist.get(2, 0) > 0:
            suggestions.append("存在较多低评分反馈，建议优先处理用户关注的问题")

        # 基于改进领域生成建议
        improvement_areas = analysis.get("improvement_areas", [])
        for area in improvement_areas:
            suggestions.append(f"重点关注 {area} 的改进")

        # 基于类别分布生成建议
        categories = analysis.get("categories", {})
        if "performance" in categories:
            suggestions.append("性能相关反馈较多，建议加强性能优化")
        if "documentation" in categories:
            suggestions.append("文档相关反馈较多，建议完善文档和示例")
        if "usability" in categories:
            suggestions.append("易用性相关反馈较多，建议改进用户体验")

        return suggestions
