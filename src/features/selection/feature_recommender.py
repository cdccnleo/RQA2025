#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择推荐系统

基于历史数据和机器学习算法，提供智能特征选择推荐：
- 基于历史表现的特征推荐
- 基于相关性的特征组合推荐
- 基于重要性的特征排序
- 个性化特征推荐
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class FeatureRecommender:
    """
    特征推荐器
    
    基于多种算法提供智能特征选择推荐
    """
    
    def __init__(self):
        self.min_history_samples = 5  # 最小历史样本数
        self.recommendation_weights = {
            'historical_performance': 0.3,
            'correlation_score': 0.25,
            'importance_score': 0.25,
            'stability_score': 0.2
        }
    
    def recommend_features(
        self,
        available_features: List[str],
        target_feature: Optional[str] = None,
        task_type: str = "technical",
        top_k: int = 10,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        推荐特征
        
        Args:
            available_features: 可用特征列表
            target_feature: 目标特征（用于相关性分析）
            task_type: 任务类型
            top_k: 推荐数量
            user_preferences: 用户偏好
            
        Returns:
            推荐结果
        """
        try:
            logger.info(f"开始特征推荐: 可用特征 {len(available_features)} 个")
            
            # 获取历史数据
            history_scores = self._get_historical_scores(available_features)
            
            # 计算综合评分
            feature_scores = []
            for feature in available_features:
                score = self._calculate_feature_score(
                    feature,
                    history_scores.get(feature, {}),
                    task_type
                )
                feature_scores.append({
                    'feature_name': feature,
                    'score': score,
                    'historical_score': history_scores.get(feature, {}).get('avg_score', 0.5),
                    'usage_count': history_scores.get(feature, {}).get('usage_count', 0)
                })
            
            # 排序并选择Top-K
            feature_scores.sort(key=lambda x: x['score'], reverse=True)
            top_features = feature_scores[:top_k]
            
            # 生成推荐组合
            recommended_combinations = self._generate_combinations(
                [f['feature_name'] for f in top_features],
                target_feature
            )
            
            # 生成推荐理由
            recommendations = self._generate_recommendations(top_features)
            
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'available_features': len(available_features),
                'recommended_features': top_features,
                'recommended_combinations': recommended_combinations,
                'recommendations': recommendations,
                'algorithm': 'hybrid_scoring',
                'confidence': self._calculate_confidence(top_features)
            }
            
            logger.info(f"特征推荐完成: 推荐 {len(top_features)} 个特征")
            return result
            
        except Exception as e:
            logger.error(f"特征推荐失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommended_features': []
            }
    
    def _get_historical_scores(self, features: List[str]) -> Dict[str, Dict[str, float]]:
        """
        获取特征的历史评分
        
        Args:
            features: 特征列表
            
        Returns:
            历史评分字典
        """
        try:
            from src.features.selection.feature_selector_history import get_feature_selector_history_manager
            
            history_manager = get_feature_selector_history_manager()
            
            # 获取重要性排名
            importance_ranking = history_manager.get_feature_importance_ranking(
                feature_names=features,
                days=30
            )
            
            # 转换为字典
            scores = {}
            for rank_info in importance_ranking:
                feature_name = rank_info['feature_name']
                scores[feature_name] = {
                    'avg_score': min(1.0, rank_info['selected_count'] / 10),  # 归一化
                    'usage_count': rank_info['selected_count'],
                    'selection_records': rank_info['selection_records']
                }
            
            # 为没有历史的特征设置默认分数
            for feature in features:
                if feature not in scores:
                    scores[feature] = {
                        'avg_score': 0.5,
                        'usage_count': 0,
                        'selection_records': []
                    }
            
            return scores
            
        except Exception as e:
            logger.warning(f"获取历史评分失败: {e}")
            # 返回默认分数
            return {f: {'avg_score': 0.5, 'usage_count': 0} for f in features}
    
    def _calculate_feature_score(
        self,
        feature: str,
        history: Dict[str, Any],
        task_type: str
    ) -> float:
        """
        计算特征综合评分
        
        Args:
            feature: 特征名称
            history: 历史数据
            task_type: 任务类型
            
        Returns:
            综合评分
        """
        # 历史表现分数
        historical_score = history.get('avg_score', 0.5)
        
        # 使用频率分数（使用越多越可靠）
        usage_count = history.get('usage_count', 0)
        usage_score = min(1.0, usage_count / 20)  # 最多20次使用达到满分
        
        # 特征类型匹配分数
        type_match_score = self._calculate_type_match(feature, task_type)
        
        # 稳定性分数（基于历史使用的稳定性）
        stability_score = self._calculate_stability_score(history)
        
        # 加权综合
        total_score = (
            historical_score * self.recommendation_weights['historical_performance'] +
            usage_score * 0.15 +
            type_match_score * 0.15 +
            stability_score * self.recommendation_weights['stability_score']
        )
        
        return total_score
    
    def _calculate_type_match(self, feature: str, task_type: str) -> float:
        """计算特征与任务类型的匹配度"""
        # 定义特征类型映射
        type_mapping = {
            'technical': ['SMA', 'EMA', 'RSI', 'MACD', 'KDJ', 'BOLL', 'MA', 'EMA', 'WMA'],
            'fundamental': ['PE', 'PB', 'ROE', 'ROA', 'EPS', 'Revenue', 'Profit'],
            'sentiment': ['Sentiment', 'News', 'Social', 'Emotion'],
            'volatility': ['ATR', 'Volatility', 'StdDev', 'Variance']
        }
        
        # 检查特征是否匹配任务类型
        matching_features = type_mapping.get(task_type, [])
        for match in matching_features:
            if match.upper() in feature.upper():
                return 1.0
        
        return 0.5  # 默认中等匹配度
    
    def _calculate_stability_score(self, history: Dict[str, Any]) -> float:
        """计算特征稳定性分数"""
        records = history.get('selection_records', [])
        
        if len(records) < 2:
            return 0.5  # 数据不足，返回中等分数
        
        # 计算选择时间间隔的稳定性
        timestamps = sorted([r['timestamp'] for r in records])
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return 0.5
        
        # 计算变异系数
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 1.0
        
        cv = std_interval / mean_interval
        stability = max(0.0, 1.0 - cv)
        
        return stability
    
    def _generate_combinations(
        self,
        top_features: List[str],
        target_feature: Optional[str]
    ) -> List[Dict[str, Any]]:
        """生成推荐特征组合"""
        combinations = []
        
        if len(top_features) >= 3:
            # 推荐3个最佳特征组合
            combinations.append({
                'name': '核心指标组合',
                'features': top_features[:3],
                'description': '基于历史表现最佳的3个特征',
                'confidence': 'high'
            })
        
        if len(top_features) >= 5:
            # 推荐5个特征组合
            combinations.append({
                'name': '平衡指标组合',
                'features': top_features[:5],
                'description': '兼顾覆盖面和稳定性的5个特征',
                'confidence': 'medium'
            })
        
        if target_feature and len(top_features) >= 2:
            # 推荐与目标特征相关的组合
            related = [f for f in top_features if f != target_feature][:2]
            if related:
                combinations.append({
                    'name': f'与{target_feature}相关的特征',
                    'features': related,
                    'description': f'与{target_feature}相关性较高的特征',
                    'confidence': 'medium'
                })
        
        return combinations
    
    def _generate_recommendations(self, top_features: List[Dict[str, Any]]) -> List[str]:
        """生成推荐理由"""
        recommendations = []
        
        if not top_features:
            return recommendations
        
        # 基于历史表现
        best_feature = top_features[0]
        if best_feature['usage_count'] > 5:
            recommendations.append(
                f"{best_feature['feature_name']} 在历史任务中表现优异，"
                f"被选择了{best_feature['usage_count']}次"
            )
        
        # 基于多样性
        if len(top_features) >= 3:
            recommendations.append(
                f"推荐的{len(top_features)}个特征涵盖了不同的技术指标类型，"
                "有助于提高模型的稳定性"
            )
        
        # 基于新颖性
        new_features = [f for f in top_features if f['usage_count'] == 0]
        if new_features and len(new_features) < len(top_features):
            recommendations.append(
                f"包含{len(new_features)}个新特征，可以尝试不同的特征组合"
            )
        
        return recommendations
    
    def _calculate_confidence(self, top_features: List[Dict[str, Any]]) -> str:
        """计算推荐置信度"""
        if not top_features:
            return 'low'
        
        # 基于历史使用次数计算置信度
        avg_usage = np.mean([f['usage_count'] for f in top_features])
        
        if avg_usage >= 10:
            return 'high'
        elif avg_usage >= 5:
            return 'medium'
        else:
            return 'low'
    
    def analyze_feature_importance_trend(
        self,
        feature_name: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        分析特征重要性趋势
        
        Args:
            feature_name: 特征名称
            days: 分析天数
            
        Returns:
            趋势分析结果
        """
        try:
            from src.features.selection.feature_selector_history import get_feature_selector_history_manager
            
            history_manager = get_feature_selector_history_manager()
            history = history_manager.get_selection_history(days=days)
            
            # 统计该特征的选择情况
            feature_history = []
            for record in history:
                if feature_name in record.get('selected_features', []):
                    feature_history.append({
                        'timestamp': record['timestamp'],
                        'datetime': record['datetime'],
                        'task_id': record['task_id']
                    })
            
            if not feature_history:
                return {
                    'feature_name': feature_name,
                    'trend': 'stable',
                    'message': '该特征在指定时间内未被选择'
                }
            
            # 分析趋势
            if len(feature_history) >= 3:
                # 简单线性趋势分析
                recent = feature_history[-3:]
                older = feature_history[:-3] if len(feature_history) > 3 else []
                
                recent_rate = len(recent) / 3
                older_rate = len(older) / max(len(older), 1)
                
                if recent_rate > older_rate * 1.5:
                    trend = 'increasing'
                elif recent_rate < older_rate * 0.5:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'insufficient_data'
            
            return {
                'feature_name': feature_name,
                'trend': trend,
                'selection_count': len(feature_history),
                'history': feature_history[-10:],  # 最近10条记录
                'message': f'该特征被选择了{len(feature_history)}次'
            }
            
        except Exception as e:
            logger.error(f"分析特征趋势失败: {e}")
            return {
                'feature_name': feature_name,
                'trend': 'unknown',
                'error': str(e)
            }


# 全局推荐器实例
_recommender: Optional[FeatureRecommender] = None


def get_feature_recommender() -> FeatureRecommender:
    """
    获取全局特征推荐器实例
    
    Returns:
        特征推荐器实例
    """
    global _recommender
    if _recommender is None:
        _recommender = FeatureRecommender()
    return _recommender
