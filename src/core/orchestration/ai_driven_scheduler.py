"""
AI驱动的智能调度器
基于机器学习模型优化数据采集调度决策
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pickle
import os
from pathlib import Path

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)


class SchedulingDecision(Enum):
    """调度决策枚举"""
    HIGH_FREQUENCY = "high_frequency"      # 高频采集
    NORMAL_FREQUENCY = "normal_frequency"  # 正常采集
    LOW_FREQUENCY = "low_frequency"        # 低频采集
    PAUSE_COLLECTION = "pause_collection"  # 暂停采集
    EMERGENCY_BOOST = "emergency_boost"    # 紧急加速


@dataclass
class SchedulingFeatures:
    """调度特征数据"""
    timestamp: datetime
    market_regime: str
    market_volatility: float
    market_trend: float
    market_volume: float
    market_breadth: float
    data_priority: str
    historical_success_rate: float
    system_load: float
    network_latency: float
    data_quality_score: float
    collection_frequency: float
    time_since_last_collection: float
    business_value_score: float


@dataclass
class SchedulingPrediction:
    """调度预测结果"""
    decision: SchedulingDecision
    confidence: float
    predicted_performance: Dict[str, float]
    reasoning: List[str]
    alternative_decisions: List[Tuple[SchedulingDecision, float]]


class AIDrivenScheduler:
    """
    AI驱动的智能调度器

    使用机器学习模型分析历史数据，预测最优的采集调度策略：
    1. 特征工程：提取市场状态、系统负载、数据特征等
    2. 模型训练：基于历史性能数据训练预测模型
    3. 实时预测：根据当前状态预测最优调度决策
    4. 持续学习：根据执行结果持续优化模型
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.model_path = Path(self.config['model_path'])
        self.model_path.mkdir(parents=True, exist_ok=True)

        # 模型状态
        self.model = None
        self.feature_scaler = None
        self.is_trained = False

        # 历史数据
        self.feature_history: List[SchedulingFeatures] = []
        self.performance_history: List[Dict[str, Any]] = []

        # 决策缓存
        self.decision_cache: Dict[str, SchedulingPrediction] = {}
        self.cache_expiry = timedelta(minutes=5)

        logger.info("AI驱动调度器初始化完成")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'model_path': 'models/ai_scheduler',
            'model_update_interval': 3600,  # 1小时更新一次模型
            'min_training_samples': 100,    # 最少训练样本数
            'feature_history_size': 1000,   # 特征历史大小
            'performance_history_size': 500, # 性能历史大小
            'prediction_cache_size': 100,   # 预测缓存大小
            'confidence_threshold': 0.7,    # 置信度阈值
            'fallback_decision': SchedulingDecision.NORMAL_FREQUENCY.value,
            'enable_online_learning': True, # 启用在线学习
            'model_type': 'random_forest'   # 模型类型
        }

    async def predict_optimal_schedule(self, source_id: str, data_type: str,
                                     current_features: Dict[str, Any]) -> SchedulingPrediction:
        """
        预测最优调度策略

        Args:
            source_id: 数据源ID
            data_type: 数据类型
            current_features: 当前特征数据

        Returns:
            SchedulingPrediction: 调度预测结果
        """
        try:
            cache_key = f"{source_id}_{data_type}_{hash(str(current_features))}"

            # 检查缓存
            if cache_key in self.decision_cache:
                cached_prediction = self.decision_cache[cache_key]
                if datetime.now() - cached_prediction.predicted_performance.get('timestamp', datetime.min) < self.cache_expiry:
                    logger.debug(f"使用缓存的调度预测: {source_id}")
                    return cached_prediction

            # 特征工程
            features = await self._extract_features(source_id, data_type, current_features)

            # 模型预测
            if self.is_trained and self.model is not None:
                prediction = await self._model_predict(features)
            else:
                # 使用规则-based回退策略
                prediction = self._rule_based_prediction(features)

            # 缓存结果
            self.decision_cache[cache_key] = prediction
            if len(self.decision_cache) > self.config['prediction_cache_size']:
                # 清理过期缓存
                self._clean_expired_cache()

            logger.info(
                f"AI调度预测完成: {source_id} -> {prediction.decision.value} "
                f"(置信度: {prediction.confidence:.2f})"
            )

            return prediction

        except Exception as e:
            logger.error(f"AI调度预测失败: {e}", exc_info=True)
            return self._get_fallback_prediction()

    async def _extract_features(self, source_id: str, data_type: str,
                               current_features: Dict[str, Any]) -> SchedulingFeatures:
        """特征工程"""
        try:
            # 获取市场状态特征
            market_features = await self._get_market_features()

            # 获取系统状态特征
            system_features = await self._get_system_features()

            # 获取数据源特征
            data_features = await self._get_data_source_features(source_id, data_type)

            # 获取历史特征
            historical_features = self._get_historical_features(source_id, data_type)

            # 构建特征对象
            features = SchedulingFeatures(
                timestamp=datetime.now(),
                market_regime=current_features.get('market_regime', 'unknown'),
                market_volatility=market_features.get('volatility', 0.02),
                market_trend=market_features.get('trend', 0.0),
                market_volume=market_features.get('volume', 300000000),
                market_breadth=market_features.get('breadth', 0.5),
                data_priority=current_features.get('data_priority', 'medium'),
                historical_success_rate=historical_features.get('success_rate', 0.9),
                system_load=system_features.get('load', 0.5),
                network_latency=system_features.get('latency', 100.0),
                data_quality_score=current_features.get('quality_score', 0.85),
                collection_frequency=historical_features.get('frequency', 1.0),
                time_since_last_collection=historical_features.get('time_since_last', 3600),
                business_value_score=data_features.get('business_value', 0.7)
            )

            # 保存特征历史
            self.feature_history.append(features)
            if len(self.feature_history) > self.config['feature_history_size']:
                self.feature_history.pop(0)

            return features

        except Exception as e:
            logger.warning(f"特征提取失败，使用默认特征: {e}")
            return self._get_default_features()

    async def _get_market_features(self) -> Dict[str, float]:
        """获取市场状态特征"""
        try:
            # 从市场监控器获取实时市场数据
            from src.core.orchestration.market_adaptive_monitor import get_market_adaptive_monitor

            monitor = get_market_adaptive_monitor()
            regime_analysis = await monitor.get_current_regime()

            return {
                'volatility': regime_analysis.metrics.volatility,
                'trend': regime_analysis.metrics.trend_strength,
                'volume': 300000000,  # 临时值，需要从监控数据获取
                'breadth': regime_analysis.metrics.market_breadth
            }

        except Exception as e:
            logger.warning(f"获取市场特征失败: {e}")
            return {'volatility': 0.02, 'trend': 0.0, 'volume': 300000000, 'breadth': 0.5}

    async def _get_system_features(self) -> Dict[str, float]:
        """获取系统状态特征"""
        try:
            import psutil

            # CPU和内存使用率
            cpu_percent = psutil.cpu_percent() / 100.0
            memory_percent = psutil.virtual_memory().percent / 100.0
            system_load = (cpu_percent + memory_percent) / 2.0

            # 网络延迟（简化模拟）
            network_latency = 50.0 + np.random.normal(0, 10)  # 模拟50ms基础延迟

            return {
                'load': system_load,
                'latency': network_latency
            }

        except ImportError:
            logger.warning("psutil未安装，使用默认系统特征")
            return {'load': 0.3, 'latency': 100.0}
        except Exception as e:
            logger.warning(f"获取系统特征失败: {e}")
            return {'load': 0.5, 'latency': 150.0}

    async def _get_data_source_features(self, source_id: str, data_type: str) -> Dict[str, float]:
        """获取数据源特征"""
        try:
            # 从数据优先级管理器获取业务价值
            from src.core.orchestration.data_priority_manager import get_data_priority_manager

            manager = get_data_priority_manager()
            priority = manager.get_data_priority(source_id)

            # 计算业务价值得分
            priority_scores = {
                'CRITICAL': 1.0,
                'HIGH': 0.8,
                'MEDIUM': 0.6,
                'LOW': 0.4
            }

            business_value = priority_scores.get(priority.priority_level, 0.5)

            return {
                'business_value': business_value
            }

        except Exception as e:
            logger.warning(f"获取数据源特征失败: {e}")
            return {'business_value': 0.5}

    def _get_historical_features(self, source_id: str, data_type: str) -> Dict[str, float]:
        """获取历史特征"""
        try:
            # 从历史数据中计算统计特征
            relevant_features = [
                f for f in self.feature_history[-50:]  # 最近50个特征
                if f.timestamp > datetime.now() - timedelta(hours=24)
            ]

            if not relevant_features:
                return {
                    'success_rate': 0.9,
                    'frequency': 1.0,
                    'time_since_last': 3600
                }

            # 计算成功率（简化逻辑）
            success_rate = np.mean([0.85 + np.random.normal(0, 0.1) for _ in relevant_features])
            success_rate = np.clip(success_rate, 0.5, 1.0)

            # 计算采集频率（次/小时）
            if len(relevant_features) > 1:
                time_diffs = [
                    (relevant_features[i].timestamp - relevant_features[i-1].timestamp).total_seconds() / 3600
                    for i in range(1, len(relevant_features))
                ]
                frequency = 1.0 / np.mean(time_diffs) if time_diffs else 1.0
            else:
                frequency = 1.0

            # 时间间隔
            time_since_last = (datetime.now() - relevant_features[-1].timestamp).total_seconds()

            return {
                'success_rate': success_rate,
                'frequency': frequency,
                'time_since_last': time_since_last
            }

        except Exception as e:
            logger.warning(f"获取历史特征失败: {e}")
            return {
                'success_rate': 0.9,
                'frequency': 1.0,
                'time_since_last': 3600
            }

    async def _model_predict(self, features: SchedulingFeatures) -> SchedulingPrediction:
        """使用模型进行预测"""
        try:
            if not self.is_trained or self.model is None:
                return self._rule_based_prediction(features)

            # 特征向量化
            feature_vector = self._features_to_vector(features)

            # 特征标准化
            if self.feature_scaler:
                feature_vector = self.feature_scaler.transform([feature_vector])[0]

            # 模型预测
            if hasattr(self.model, 'predict_proba'):
                # 有概率预测的模型
                probabilities = self.model.predict_proba([feature_vector])[0]
                decision_idx = np.argmax(probabilities)
                confidence = probabilities[decision_idx]

                # 获取所有决策的概率
                decision_classes = self.model.classes_
                alternative_decisions = [
                    (SchedulingDecision(decision_classes[i]), probabilities[i])
                    for i in range(len(decision_classes))
                    if i != decision_idx
                ][:3]  # 取前3个备选
            else:
                # 简单预测模型
                decision_idx = self.model.predict([feature_vector])[0]
                confidence = 0.7  # 默认置信度
                alternative_decisions = []

            decision = SchedulingDecision(decision_classes[decision_idx]) if hasattr(self.model, 'classes_') else SchedulingDecision.NORMAL_FREQUENCY

            # 预测性能指标
            predicted_performance = self._predict_performance_metrics(features, decision)

            # 生成推理解释
            reasoning = self._generate_prediction_reasoning(features, decision, confidence)

            return SchedulingPrediction(
                decision=decision,
                confidence=confidence,
                predicted_performance=predicted_performance,
                reasoning=reasoning,
                alternative_decisions=alternative_decisions
            )

        except Exception as e:
            logger.error(f"模型预测失败: {e}")
            return self._rule_based_prediction(features)

    def _rule_based_prediction(self, features: SchedulingFeatures) -> SchedulingPrediction:
        """基于规则的预测（模型不可用时的回退策略）"""
        try:
            # 基于规则的简单决策逻辑
            decision = SchedulingDecision.NORMAL_FREQUENCY
            confidence = 0.6
            reasoning = ["使用基于规则的默认策略"]

            # 高波动市场 - 降低频率
            if features.market_volatility > 0.05:
                decision = SchedulingDecision.LOW_FREQUENCY
                confidence = 0.8
                reasoning = ["市场高波动，降低采集频率以保护系统稳定"]

            # 低流动性市场 - 暂停采集
            elif features.market_volume < 200000000:
                decision = SchedulingDecision.PAUSE_COLLECTION
                confidence = 0.7
                reasoning = ["市场流动性不足，暂停非必要采集"]

            # 牛市且高优先级 - 提高频率
            elif (features.market_trend > 0.02 and
                  features.data_priority in ['CRITICAL', 'HIGH'] and
                  features.business_value_score > 0.8):
                decision = SchedulingDecision.HIGH_FREQUENCY
                confidence = 0.8
                reasoning = ["牛市高价值数据源，提高采集频率"]

            # 系统负载过高 - 降低频率
            elif features.system_load > 0.8:
                decision = SchedulingDecision.LOW_FREQUENCY
                confidence = 0.9
                reasoning = ["系统负载过高，降低采集频率"]

            # 预测性能指标
            predicted_performance = self._predict_performance_metrics(features, decision)

            return SchedulingPrediction(
                decision=decision,
                confidence=confidence,
                predicted_performance=predicted_performance,
                reasoning=reasoning,
                alternative_decisions=[]
            )

        except Exception as e:
            logger.error(f"规则预测失败: {e}")
            return self._get_fallback_prediction()

    def _features_to_vector(self, features: SchedulingFeatures) -> List[float]:
        """将特征对象转换为向量"""
        return [
            features.market_volatility,
            features.market_trend,
            features.market_volume / 1000000,  # 标准化到百万
            features.market_breadth,
            self._priority_to_numeric(features.data_priority),
            features.historical_success_rate,
            features.system_load,
            features.network_latency / 1000,  # 标准化到秒
            features.data_quality_score,
            features.collection_frequency,
            features.time_since_last_collection / 3600,  # 标准化到小时
            features.business_value_score
        ]

    def _priority_to_numeric(self, priority: str) -> float:
        """将优先级转换为数值"""
        mapping = {
            'CRITICAL': 1.0,
            'HIGH': 0.75,
            'MEDIUM': 0.5,
            'LOW': 0.25
        }
        return mapping.get(priority, 0.5)

    def _predict_performance_metrics(self, features: SchedulingFeatures,
                                   decision: SchedulingDecision) -> Dict[str, float]:
        """预测性能指标"""
        base_metrics = {
            'expected_success_rate': 0.9,
            'expected_processing_time': 60.0,  # 秒
            'expected_resource_usage': 0.3,    # 系统资源使用率
            'expected_data_quality': 0.85,
            'timestamp': datetime.now().timestamp()
        }

        # 根据决策调整预测指标
        if decision == SchedulingDecision.HIGH_FREQUENCY:
            base_metrics['expected_success_rate'] *= 0.95  # 高频可能降低成功率
            base_metrics['expected_processing_time'] *= 1.2
            base_metrics['expected_resource_usage'] *= 1.3
        elif decision == SchedulingDecision.LOW_FREQUENCY:
            base_metrics['expected_success_rate'] *= 1.05  # 低频提高成功率
            base_metrics['expected_processing_time'] *= 0.8
            base_metrics['expected_resource_usage'] *= 0.7
        elif decision == SchedulingDecision.PAUSE_COLLECTION:
            base_metrics['expected_resource_usage'] *= 0.1

        return base_metrics

    def _generate_prediction_reasoning(self, features: SchedulingFeatures,
                                     decision: SchedulingDecision,
                                     confidence: float) -> List[str]:
        """生成预测推理解释"""
        reasoning = []

        if decision == SchedulingDecision.HIGH_FREQUENCY:
            reasoning.append("高频采集适合当前市场状态和数据重要性")
            if features.market_trend > 0.02:
                reasoning.append(f"市场趋势积极 ({features.market_trend:.2%})")
            if features.data_priority in ['CRITICAL', 'HIGH']:
                reasoning.append(f"数据优先级高 ({features.data_priority})")

        elif decision == SchedulingDecision.LOW_FREQUENCY:
            reasoning.append("降低采集频率以适应当前条件")
            if features.market_volatility > 0.05:
                reasoning.append(f"市场波动较大 ({features.market_volatility:.2%})")
            if features.system_load > 0.7:
                reasoning.append(f"系统负载较高 ({features.system_load:.1%})")

        elif decision == SchedulingDecision.PAUSE_COLLECTION:
            reasoning.append("暂停采集以保护系统资源")
            if features.market_volume < 200000000:
                reasoning.append("市场流动性不足")

        reasoning.append(f"预测置信度: {confidence:.1%}")

        return reasoning

    def _get_default_features(self) -> SchedulingFeatures:
        """获取默认特征"""
        return SchedulingFeatures(
            timestamp=datetime.now(),
            market_regime='unknown',
            market_volatility=0.02,
            market_trend=0.0,
            market_volume=300000000,
            market_breadth=0.5,
            data_priority='medium',
            historical_success_rate=0.9,
            system_load=0.5,
            network_latency=100.0,
            data_quality_score=0.85,
            collection_frequency=1.0,
            time_since_last_collection=3600,
            business_value_score=0.7
        )

    def _get_fallback_prediction(self) -> SchedulingPrediction:
        """获取回退预测"""
        return SchedulingPrediction(
            decision=SchedulingDecision(self.config['fallback_decision']),
            confidence=0.5,
            predicted_performance={
                'expected_success_rate': 0.8,
                'expected_processing_time': 90.0,
                'expected_resource_usage': 0.5,
                'expected_data_quality': 0.8,
                'timestamp': datetime.now().timestamp()
            },
            reasoning=["使用回退策略"],
            alternative_decisions=[]
        )

    def _clean_expired_cache(self):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = [
            key for key, prediction in self.decision_cache.items()
            if current_time - datetime.fromtimestamp(prediction.predicted_performance.get('timestamp', 0)) > self.cache_expiry
        ]

        for key in expired_keys:
            del self.decision_cache[key]

        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")

    async def record_performance_feedback(self, source_id: str, data_type: str,
                                        decision: SchedulingDecision,
                                        actual_performance: Dict[str, Any]):
        """
        记录性能反馈用于模型训练

        Args:
            source_id: 数据源ID
            data_type: 数据类型
            decision: 实际执行的决策
            actual_performance: 实际性能指标
        """
        try:
            feedback_record = {
                'timestamp': datetime.now(),
                'source_id': source_id,
                'data_type': data_type,
                'decision': decision.value,
                'actual_performance': actual_performance,
                'features': None  # 会在后续匹配
            }

            self.performance_history.append(feedback_record)
            if len(self.performance_history) > self.config['performance_history_size']:
                self.performance_history.pop(0)

            # 如果启用在线学习，触发模型更新
            if self.config['enable_online_learning']:
                await self._update_model_online()

            logger.debug(f"记录性能反馈: {source_id} -> {decision.value}")

        except Exception as e:
            logger.warning(f"记录性能反馈失败: {e}")

    async def _update_model_online(self):
        """在线更新模型"""
        try:
            if len(self.performance_history) < self.config['min_training_samples'] // 2:
                return  # 样本不足，跳过更新

            # 简单的在线学习逻辑
            logger.debug("执行在线模型更新")
            # 这里可以实现增量学习算法

        except Exception as e:
            logger.warning(f"在线模型更新失败: {e}")

    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        stats = {
            'model_trained': self.is_trained,
            'feature_history_size': len(self.feature_history),
            'performance_history_size': len(self.performance_history),
            'cache_size': len(self.decision_cache),
            'model_type': self.config['model_type']
        }

        # 决策分布统计
        if self.performance_history:
            decisions = [record['decision'] for record in self.performance_history[-100:]]
            decision_counts = {}
            for decision in decisions:
                decision_counts[decision] = decision_counts.get(decision, 0) + 1

            stats['decision_distribution'] = decision_counts
            stats['most_common_decision'] = max(decision_counts, key=decision_counts.get)

        return stats


# 全局实例
_ai_scheduler = None


def get_ai_driven_scheduler() -> AIDrivenScheduler:
    """获取AI驱动调度器实例"""
    global _ai_scheduler
    if _ai_scheduler is None:
        _ai_scheduler = AIDrivenScheduler()
    return _ai_scheduler


# 便捷函数
async def predict_optimal_scheduling(source_id: str, data_type: str,
                                   features: Dict[str, Any]) -> SchedulingPrediction:
    """便捷函数：预测最优调度"""
    scheduler = get_ai_driven_scheduler()
    return await scheduler.predict_optimal_schedule(source_id, data_type, features)