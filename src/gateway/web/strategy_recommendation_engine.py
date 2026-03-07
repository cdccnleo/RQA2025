"""
策略智能推荐引擎模块
提供基于回测结果和历史数据的智能推荐功能
"""

import json
import logging
import os
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """推荐项"""
    recommendation_id: str
    strategy_id: str
    recommendation_type: str
    title: str
    description: str
    confidence: float  # 置信度 0-1
    priority: int  # 优先级 1-5
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_read: bool = False
    is_applied: bool = False


class StrategyRecommendationEngine:
    """策略推荐引擎"""
    
    def __init__(self, recommendations_dir: str = "data/recommendations"):
        self.recommendations_dir = recommendations_dir
        self._ensure_directory()
        
        # 推荐阈值配置
        self.thresholds = {
            'sharpe_ratio_low': 0.5,
            'sharpe_ratio_good': 1.0,
            'sharpe_ratio_excellent': 1.5,
            'max_drawdown_warning': 0.2,
            'max_drawdown_critical': 0.3,
            'win_rate_low': 0.4,
            'win_rate_good': 0.6,
            'volatility_high': 0.25,
            'return_risk_ratio_low': 1.0
        }
    
    def _ensure_directory(self):
        """确保推荐目录存在"""
        if not os.path.exists(self.recommendations_dir):
            os.makedirs(self.recommendations_dir)
            logger.info(f"创建推荐目录: {self.recommendations_dir}")
    
    def _get_strategy_recommendations_dir(self, strategy_id: str) -> str:
        """获取策略推荐目录"""
        strategy_dir = os.path.join(self.recommendations_dir, strategy_id)
        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir)
        return strategy_dir
    
    def _generate_recommendation_id(self) -> str:
        """生成推荐ID"""
        return f"rec_{int(time.time())}_{hash(str(time.time())) % 10000}"
    
    def _save_recommendation(self, recommendation: Recommendation):
        """保存推荐到文件"""
        try:
            strategy_dir = self._get_strategy_recommendations_dir(recommendation.strategy_id)
            filepath = os.path.join(strategy_dir, f"{recommendation.recommendation_id}.json")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'recommendation_id': recommendation.recommendation_id,
                    'strategy_id': recommendation.strategy_id,
                    'recommendation_type': recommendation.recommendation_type,
                    'title': recommendation.title,
                    'description': recommendation.description,
                    'confidence': recommendation.confidence,
                    'priority': recommendation.priority,
                    'created_at': recommendation.created_at,
                    'metadata': recommendation.metadata,
                    'is_read': recommendation.is_read,
                    'is_applied': recommendation.is_applied
                }, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存推荐失败: {e}")
    
    def analyze_backtest_result(self, strategy_id: str, backtest_result: Dict) -> List[Recommendation]:
        """分析回测结果并生成推荐"""
        recommendations = []
        
        try:
            metrics = backtest_result.get('metrics', {})
            performance = backtest_result.get('performance', {})
            
            # 1. 夏普比率分析
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe < self.thresholds['sharpe_ratio_low']:
                recommendations.append(Recommendation(
                    recommendation_id=self._generate_recommendation_id(),
                    strategy_id=strategy_id,
                    recommendation_type='performance_improvement',
                    title='夏普比率偏低',
                    description=f'当前夏普比率 {sharpe:.2f} 低于建议值 {self.thresholds["sharpe_ratio_low"]}，建议优化参数或调整策略逻辑',
                    confidence=0.85,
                    priority=4,
                    created_at=time.time(),
                    metadata={'current_sharpe': sharpe, 'threshold': self.thresholds['sharpe_ratio_low']}
                ))
            elif sharpe > self.thresholds['sharpe_ratio_excellent']:
                recommendations.append(Recommendation(
                    recommendation_id=self._generate_recommendation_id(),
                    strategy_id=strategy_id,
                    recommendation_type='performance_good',
                    title='夏普比率优秀',
                    description=f'当前夏普比率 {sharpe:.2f} 表现优秀，可以考虑增加仓位或进入模拟交易',
                    confidence=0.9,
                    priority=2,
                    created_at=time.time(),
                    metadata={'current_sharpe': sharpe}
                ))
            
            # 2. 最大回撤分析
            max_drawdown = metrics.get('max_drawdown', 0)
            if max_drawdown > self.thresholds['max_drawdown_critical']:
                recommendations.append(Recommendation(
                    recommendation_id=self._generate_recommendation_id(),
                    strategy_id=strategy_id,
                    recommendation_type='risk_warning',
                    title='最大回撤过高',
                    description=f'当前最大回撤 {max_drawdown*100:.1f}% 超过警戒线 {self.thresholds["max_drawdown_critical"]*100:.1f}%，建议增加止损机制',
                    confidence=0.9,
                    priority=5,
                    created_at=time.time(),
                    metadata={'current_drawdown': max_drawdown, 'threshold': self.thresholds['max_drawdown_critical']}
                ))
            elif max_drawdown > self.thresholds['max_drawdown_warning']:
                recommendations.append(Recommendation(
                    recommendation_id=self._generate_recommendation_id(),
                    strategy_id=strategy_id,
                    recommendation_type='risk_warning',
                    title='最大回撤偏高',
                    description=f'当前最大回撤 {max_drawdown*100:.1f}% 偏高，建议优化风险控制',
                    confidence=0.75,
                    priority=3,
                    created_at=time.time(),
                    metadata={'current_drawdown': max_drawdown}
                ))
            
            # 3. 胜率分析
            win_rate = metrics.get('win_rate', 0)
            if win_rate < self.thresholds['win_rate_low']:
                recommendations.append(Recommendation(
                    recommendation_id=self._generate_recommendation_id(),
                    strategy_id=strategy_id,
                    recommendation_type='strategy_improvement',
                    title='胜率偏低',
                    description=f'当前胜率 {win_rate*100:.1f}% 低于建议值 {self.thresholds["win_rate_low"]*100:.1f}%，建议优化入场条件',
                    confidence=0.8,
                    priority=4,
                    created_at=time.time(),
                    metadata={'current_win_rate': win_rate}
                ))
            
            # 4. 收益风险比分析
            total_return = metrics.get('total_return', 0)
            volatility = metrics.get('volatility', 0)
            if volatility > 0:
                return_risk_ratio = total_return / volatility
                if return_risk_ratio < self.thresholds['return_risk_ratio_low']:
                    recommendations.append(Recommendation(
                        recommendation_id=self._generate_recommendation_id(),
                        strategy_id=strategy_id,
                        recommendation_type='risk_adjustment',
                        title='收益风险比偏低',
                        description=f'当前收益风险比 {return_risk_ratio:.2f} 偏低，建议调整仓位管理',
                        confidence=0.7,
                        priority=3,
                        created_at=time.time(),
                        metadata={'return_risk_ratio': return_risk_ratio}
                    ))
            
            # 5. 波动率分析
            if volatility > self.thresholds['volatility_high']:
                recommendations.append(Recommendation(
                    recommendation_id=self._generate_recommendation_id(),
                    strategy_id=strategy_id,
                    recommendation_type='volatility_warning',
                    title='波动率过高',
                    description=f'当前波动率 {volatility*100:.1f}% 较高，策略可能过于激进',
                    confidence=0.75,
                    priority=3,
                    created_at=time.time(),
                    metadata={'volatility': volatility}
                ))
            
            # 6. 交易频率分析
            total_trades = metrics.get('total_trades', 0)
            if total_trades < 10:
                recommendations.append(Recommendation(
                    recommendation_id=self._generate_recommendation_id(),
                    strategy_id=strategy_id,
                    recommendation_type='sample_size_warning',
                    title='交易样本不足',
                    description=f'当前交易次数 {total_trades} 较少，统计结果可能不可靠，建议延长回测时间',
                    confidence=0.9,
                    priority=2,
                    created_at=time.time(),
                    metadata={'total_trades': total_trades}
                ))
            
            # 保存所有推荐
            for rec in recommendations:
                self._save_recommendation(rec)
            
            logger.info(f"为策略 {strategy_id} 生成 {len(recommendations)} 条推荐")
            return recommendations
            
        except Exception as e:
            logger.error(f"分析回测结果失败: {e}")
            return []
    
    def recommend_optimization_direction(self, strategy_id: str, 
                                        backtest_history: List[Dict]) -> List[Recommendation]:
        """基于回测历史推荐优化方向"""
        recommendations = []
        
        try:
            if len(backtest_history) < 2:
                return recommendations
            
            # 分析趋势
            sharpe_trend = []
            return_trend = []
            drawdown_trend = []
            
            for result in sorted(backtest_history, key=lambda x: x.get('timestamp', 0)):
                metrics = result.get('metrics', {})
                sharpe_trend.append(metrics.get('sharpe_ratio', 0))
                return_trend.append(metrics.get('total_return', 0))
                drawdown_trend.append(metrics.get('max_drawdown', 0))
            
            # 1. 夏普比率趋势分析
            if len(sharpe_trend) >= 3:
                recent_avg = np.mean(sharpe_trend[-3:])
                older_avg = np.mean(sharpe_trend[:-3])
                
                if recent_avg < older_avg * 0.9:  # 下降超过10%
                    recommendations.append(Recommendation(
                        recommendation_id=self._generate_recommendation_id(),
                        strategy_id=strategy_id,
                        recommendation_type='trend_warning',
                        title='夏普比率呈下降趋势',
                        description='近期夏普比率持续下降，建议检查策略是否适应最新市场环境',
                        confidence=0.8,
                        priority=4,
                        created_at=time.time(),
                        metadata={'trend': 'declining', 'recent_avg': recent_avg, 'older_avg': older_avg}
                    ))
            
            # 2. 回撤趋势分析
            if len(drawdown_trend) >= 3:
                recent_max_dd = max(drawdown_trend[-3:])
                if recent_max_dd > 0.15:  # 近期回撤超过15%
                    recommendations.append(Recommendation(
                        recommendation_id=self._generate_recommendation_id(),
                        strategy_id=strategy_id,
                        recommendation_type='risk_trend_warning',
                        title='回撤风险上升',
                        description='近期最大回撤增加，建议加强风险控制或降低仓位',
                        confidence=0.85,
                        priority=4,
                        created_at=time.time(),
                        metadata={'recent_max_drawdown': recent_max_dd}
                    ))
            
            # 保存推荐
            for rec in recommendations:
                self._save_recommendation(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"推荐优化方向失败: {e}")
            return []
    
    def recommend_parameter_range(self, strategy_id: str, 
                                  current_params: Dict,
                                  optimization_history: List[Dict]) -> List[Recommendation]:
        """推荐参数优化范围"""
        recommendations = []
        
        try:
            if not optimization_history:
                return recommendations
            
            # 分析历史优化结果
            param_performance = defaultdict(list)
            
            for opt_result in optimization_history:
                results = opt_result.get('results', [])
                if results:
                    best = results[0]
                    params = best.get('params', {})
                    performance = best.get('performance', {})
                    sharpe = performance.get('sharpe', 0)
                    
                    for param_name, param_value in params.items():
                        param_performance[param_name].append({
                            'value': param_value,
                            'sharpe': sharpe
                        })
            
            # 为每个参数生成推荐
            for param_name, performances in param_performance.items():
                if len(performances) < 3:
                    continue
                
                # 找出最佳参数值范围
                sorted_by_sharpe = sorted(performances, key=lambda x: x['sharpe'], reverse=True)
                top_performers = sorted_by_sharpe[:max(3, len(sorted_by_sharpe) // 4)]
                
                if top_performers:
                    values = [p['value'] for p in top_performers]
                    min_val = min(values)
                    max_val = max(values)
                    avg_val = np.mean(values)
                    
                    current_value = current_params.get(param_name)
                    
                    if current_value is not None:
                        # 检查当前值是否在推荐范围内
                        if current_value < min_val or current_value > max_val:
                            recommendations.append(Recommendation(
                                recommendation_id=self._generate_recommendation_id(),
                                strategy_id=strategy_id,
                                recommendation_type='parameter_optimization',
                                title=f'参数 {param_name} 优化建议',
                                description=f'当前值 {current_value} 不在最优范围 [{min_val:.2f}, {max_val:.2f}] 内，建议调整到 {avg_val:.2f} 附近',
                                confidence=0.75,
                                priority=3,
                                created_at=time.time(),
                                metadata={
                                    'param_name': param_name,
                                    'current_value': current_value,
                                    'recommended_range': [min_val, max_val],
                                    'recommended_value': avg_val
                                }
                            ))
            
            # 保存推荐
            for rec in recommendations:
                self._save_recommendation(rec)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"推荐参数范围失败: {e}")
            return []
    
    def generate_performance_alert(self, strategy_id: str, 
                                   current_metrics: Dict,
                                   baseline_metrics: Dict) -> Optional[Recommendation]:
        """生成性能预警"""
        try:
            alerts = []
            
            # 1. 夏普比率下降预警
            current_sharpe = current_metrics.get('sharpe_ratio', 0)
            baseline_sharpe = baseline_metrics.get('sharpe_ratio', 0)
            
            if baseline_sharpe > 0 and current_sharpe < baseline_sharpe * 0.8:
                alerts.append({
                    'type': 'sharpe_decline',
                    'severity': 'high' if current_sharpe < baseline_sharpe * 0.5 else 'medium',
                    'message': f'夏普比率从 {baseline_sharpe:.2f} 下降到 {current_sharpe:.2f}'
                })
            
            # 2. 回撤增加预警
            current_dd = current_metrics.get('max_drawdown', 0)
            baseline_dd = baseline_metrics.get('max_drawdown', 0)
            
            if current_dd > baseline_dd * 1.3:
                alerts.append({
                    'type': 'drawdown_increase',
                    'severity': 'high' if current_dd > baseline_dd * 1.5 else 'medium',
                    'message': f'最大回撤从 {baseline_dd*100:.1f}% 增加到 {current_dd*100:.1f}%'
                })
            
            # 3. 胜率下降预警
            current_wr = current_metrics.get('win_rate', 0)
            baseline_wr = baseline_metrics.get('win_rate', 0)
            
            if current_wr < baseline_wr * 0.85:
                alerts.append({
                    'type': 'winrate_decline',
                    'severity': 'medium',
                    'message': f'胜率从 {baseline_wr*100:.1f}% 下降到 {current_wr*100:.1f}%'
                })
            
            if alerts:
                # 选择最高优先级的预警
                high_priority = [a for a in alerts if a['severity'] == 'high']
                alert = high_priority[0] if high_priority else alerts[0]
                
                rec = Recommendation(
                    recommendation_id=self._generate_recommendation_id(),
                    strategy_id=strategy_id,
                    recommendation_type='performance_alert',
                    title='策略性能预警',
                    description=alert['message'] + '，建议立即检查策略状态',
                    confidence=0.9,
                    priority=5 if alert['severity'] == 'high' else 4,
                    created_at=time.time(),
                    metadata={'alerts': alerts}
                )
                
                self._save_recommendation(rec)
                return rec
            
            return None
            
        except Exception as e:
            logger.error(f"生成性能预警失败: {e}")
            return None
    
    def get_recommendations(self, strategy_id: str, 
                           recommendation_type: Optional[str] = None,
                           unread_only: bool = False) -> List[Recommendation]:
        """获取推荐列表"""
        try:
            strategy_dir = self._get_strategy_recommendations_dir(strategy_id)
            recommendations = []
            
            for filename in os.listdir(strategy_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(strategy_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            rec = Recommendation(
                                recommendation_id=data['recommendation_id'],
                                strategy_id=data['strategy_id'],
                                recommendation_type=data['recommendation_type'],
                                title=data['title'],
                                description=data['description'],
                                confidence=data['confidence'],
                                priority=data['priority'],
                                created_at=data['created_at'],
                                metadata=data.get('metadata', {}),
                                is_read=data.get('is_read', False),
                                is_applied=data.get('is_applied', False)
                            )
                            
                            # 筛选
                            if recommendation_type and rec.recommendation_type != recommendation_type:
                                continue
                            if unread_only and rec.is_read:
                                continue
                            
                            recommendations.append(rec)
                    except Exception as e:
                        logger.warning(f"加载推荐文件失败 {filename}: {e}")
            
            # 按优先级和时间排序
            recommendations.sort(key=lambda r: (-r.priority, -r.created_at))
            return recommendations
            
        except Exception as e:
            logger.error(f"获取推荐列表失败: {e}")
            return []
    
    def mark_recommendation_read(self, strategy_id: str, recommendation_id: str) -> bool:
        """标记推荐为已读"""
        try:
            strategy_dir = self._get_strategy_recommendations_dir(strategy_id)
            filepath = os.path.join(strategy_dir, f"{recommendation_id}.json")
            
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data['is_read'] = True
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"标记推荐已读失败: {e}")
            return False
    
    def mark_recommendation_applied(self, strategy_id: str, recommendation_id: str) -> bool:
        """标记推荐为已应用"""
        try:
            strategy_dir = self._get_strategy_recommendations_dir(strategy_id)
            filepath = os.path.join(strategy_dir, f"{recommendation_id}.json")
            
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data['is_applied'] = True
            data['applied_at'] = time.time()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"标记推荐已应用失败: {e}")
            return False


# 全局推荐引擎实例
recommendation_engine = StrategyRecommendationEngine()


# 便捷的API函数
def analyze_backtest_and_recommend(strategy_id: str, backtest_result: Dict) -> List[Recommendation]:
    """分析回测并生成推荐"""
    return recommendation_engine.analyze_backtest_result(strategy_id, backtest_result)


def get_strategy_recommendations(strategy_id: str, unread_only: bool = False) -> List[Recommendation]:
    """获取策略推荐"""
    return recommendation_engine.get_recommendations(strategy_id, unread_only=unread_only)


def generate_performance_alert(strategy_id: str, current: Dict, baseline: Dict) -> Optional[Recommendation]:
    """生成性能预警"""
    return recommendation_engine.generate_performance_alert(strategy_id, current, baseline)
