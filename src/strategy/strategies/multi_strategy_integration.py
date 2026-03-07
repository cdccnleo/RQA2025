#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多策略集成框架
提供多策略协调、权重分配、性能监控和动态调整功能
"""

import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict, deque

# 使用标准logging避免导入错误
import logging
from strategy.interfaces.strategy_interfaces import StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class StrategyInfo:

    """策略信息"""
    strategy_id: str
    strategy_name: str
    strategy_class: type
    config: StrategyConfig
    weight: float = 0.0
    status: str = "active"  # active, paused, stopped
    performance_score: float = 0.0
    last_update: datetime = None
    metadata: Dict[str, Any] = None


@dataclass
class IntegrationConfig:

    """集成配置"""
    integration_id: str
    strategy_weights: Dict[str, float]
    rebalance_frequency: int = 24  # 小时
    performance_window: int = 30  # 天
    min_weight: float = 0.05
    max_weight: float = 0.5
    risk_budget: float = 0.2
    correlation_threshold: float = 0.7
    diversity_factor: float = 0.3
    auto_rebalance: bool = True
    performance_metrics: List[str] = None


@dataclass
class IntegrationResult:

    """集成结果"""
    timestamp: datetime
    strategy_predictions: Dict[str, Any]
    ensemble_prediction: Any
    weights: Dict[str, float]
    confidence: float
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]


class MultiStrategyIntegration:

    """多策略集成框架"""

    def __init__(self, config: IntegrationConfig):
        """
        初始化多策略集成框架

        Args:
            config: 集成配置
        """
        self.config = config
        self.strategies: Dict[str, StrategyInfo] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.weight_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.lock = threading.RLock()

        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        self.weight_optimizer = WeightOptimizer()
        self.risk_manager = RiskManager()

        logger.info(f"多策略集成框架初始化完成: {config.integration_id}")

    def add_strategy(self, strategy_id: str, strategy_name: str,


                     strategy_class: type, config: StrategyConfig,
                     initial_weight: float = 0.0) -> bool:
        """
        添加策略到集成框架

        Args:
            strategy_id: 策略ID
            strategy_name: 策略名称
            strategy_class: 策略类
            config: 策略配置
            initial_weight: 初始权重

        Returns:
            bool: 是否添加成功
        """
        with self.lock:
            if strategy_id in self.strategies:
                logger.warning(f"策略 {strategy_id} 已存在")
                return False

            strategy_info = StrategyInfo(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                strategy_class=strategy_class,
                config=config,
                weight=initial_weight,
                last_update=datetime.now(),
                metadata={}
            )

            self.strategies[strategy_id] = strategy_info
            logger.info(f"策略 {strategy_id} 添加成功，权重: {initial_weight}")
            return True

    def remove_strategy(self, strategy_id: str) -> bool:
        """
        移除策略

        Args:
            strategy_id: 策略ID

        Returns:
            bool: 是否移除成功
        """
        with self.lock:
            if strategy_id not in self.strategies:
                logger.warning(f"策略 {strategy_id} 不存在")
                return False

            del self.strategies[strategy_id]
            logger.info(f"策略 {strategy_id} 移除成功")
            return True

    def update_strategy_weight(self, strategy_id: str, weight: float) -> bool:
        """
        更新策略权重

        Args:
            strategy_id: 策略ID
            weight: 新权重

        Returns:
            bool: 是否更新成功
        """
        with self.lock:
            if strategy_id not in self.strategies:
                logger.warning(f"策略 {strategy_id} 不存在")
                return False

            # 验证权重范围
            if not (self.config.min_weight <= weight <= self.config.max_weight):
                logger.warning(
                    f"权重 {weight} 超出范围 [{self.config.min_weight}, {self.config.max_weight}]")
                return False

            self.strategies[strategy_id].weight = weight
            self.strategies[strategy_id].last_update = datetime.now()

            # 记录权重历史
            self.weight_history[strategy_id].append({
                'timestamp': datetime.now(),
                'weight': weight
            })

            logger.info(f"策略 {strategy_id} 权重更新为 {weight}")
            return True

    def generate_ensemble_prediction(self, data: pd.DataFrame) -> IntegrationResult:
        """
        生成集成预测

        Args:
            data: 输入数据

        Returns:
            IntegrationResult: 集成结果
        """
        with self.lock:
            if not self.strategies:
                raise ValueError("没有可用的策略")

            strategy_predictions = {}
            active_strategies = {}

            # 获取各策略预测
            for strategy_id, strategy_info in self.strategies.items():
                if strategy_info.status != "active":
                    continue

                try:
                    # 创建策略实例
                    strategy = strategy_info.strategy_class(strategy_info.config)
                    prediction = strategy.generate_signals(data)

                    strategy_predictions[strategy_id] = prediction
                    active_strategies[strategy_id] = strategy_info

                except Exception as e:
                    logger.error(f"策略 {strategy_id} 预测失败: {e}")
                    continue

            if not active_strategies:
                raise ValueError("没有活跃的策略")

            # 计算权重
            weights = self._calculate_weights(active_strategies)

            # 生成集成预测
            ensemble_prediction = self._combine_predictions(
                strategy_predictions, weights
            )

            # 计算风险指标
            risk_metrics = self._calculate_risk_metrics(
                strategy_predictions, weights
            )

            # 计算性能指标
            performance_metrics = self._calculate_performance_metrics(
                strategy_predictions, weights
            )

            # 计算置信度
            confidence = self._calculate_confidence(
                strategy_predictions, weights, risk_metrics
            )

            result = IntegrationResult(
                timestamp=datetime.now(),
                strategy_predictions=strategy_predictions,
                ensemble_prediction=ensemble_prediction,
                weights=weights,
                confidence=confidence,
                risk_metrics=risk_metrics,
                performance_metrics=performance_metrics
            )

            # 更新性能历史
            self._update_performance_history(result)

            return result

    def _calculate_weights(self, active_strategies: Dict[str, StrategyInfo]) -> Dict[str, float]:
        """
        计算策略权重

        Args:
            active_strategies: 活跃策略字典

        Returns:
            Dict[str, float]: 权重字典
        """
        if len(active_strategies) == 1:
            strategy_id = list(active_strategies.keys())[0]
            return {strategy_id: 1.0}

        # 获取性能分数
        performance_scores = {}
        for strategy_id, strategy_info in active_strategies.items():
            score = self._calculate_performance_score(strategy_id)
            performance_scores[strategy_id] = score

        # 使用权重优化器计算权重
        weights = self.weight_optimizer.optimize_weights(
            performance_scores,
            self.correlation_matrix,
            self.config
        )

        return weights

    def _combine_predictions(self, predictions: Dict[str, Any],


                             weights: Dict[str, float]) -> Any:
        """
        组合预测结果

        Args:
            predictions: 策略预测字典
            weights: 权重字典

        Returns:
            Any: 组合预测结果
        """
        if not predictions:
            return None

        # 如果是DataFrame预测，进行加权平均
        if isinstance(list(predictions.values())[0], pd.DataFrame):
            combined = pd.DataFrame()

            for strategy_id, prediction in predictions.items():
                weight = weights.get(strategy_id, 0.0)
                if weight > 0:
                    # 只对数值列进行加权
                    numeric_pred = prediction.select_dtypes(include=[np.number])
                    if not numeric_pred.empty:
                        weighted_pred = numeric_pred * weight
                        if combined.empty:
                            combined = weighted_pred
                        else:
                            combined += weighted_pred
                    else:
                        # 如果没有数值列，使用第一个预测
                        if combined.empty:
                            combined = prediction.copy()

            return combined

        # 如果是数值预测，进行加权平均
        elif isinstance(list(predictions.values())[0], (int, float, np.number)):
            combined = 0.0
            total_weight = 0.0

            for strategy_id, prediction in predictions.items():
                weight = weights.get(strategy_id, 0.0)
                combined += prediction * weight
                total_weight += weight

            return combined / total_weight if total_weight > 0 else 0.0

        else:
            # 其他类型，返回第一个预测
            return list(predictions.values())[0]

    def _calculate_risk_metrics(self, predictions: Dict[str, Any],


                                weights: Dict[str, float]) -> Dict[str, float]:
        """
        计算风险指标

        Args:
            predictions: 策略预测字典
            weights: 权重字典

        Returns:
            Dict[str, float]: 风险指标字典
        """
        risk_metrics = {}

        # 计算组合波动率
        if len(predictions) > 1:
            returns = []
        for strategy_id, prediction in predictions.items():
            if isinstance(prediction, pd.DataFrame) and 'returns' in prediction.columns:
                returns.append(prediction['returns'])

        if returns:
            returns_df = pd.concat(returns, axis=1)
            portfolio_vol = self.risk_manager.calculate_portfolio_volatility(
                returns_df, weights
            )
            risk_metrics['portfolio_volatility'] = portfolio_vol

        # 计算最大回撤
        risk_metrics['max_drawdown'] = self.risk_manager.calculate_max_drawdown(
            predictions, weights
        )

        # 计算VaR
        risk_metrics['var_95'] = self.risk_manager.calculate_var(
            predictions, weights, confidence_level=0.95
        )

        return risk_metrics

    def _calculate_performance_metrics(self, predictions: Dict[str, Any],


                                       weights: Dict[str, float]) -> Dict[str, float]:
        """
        计算性能指标

        Args:
            predictions: 策略预测字典
            weights: 权重字典

        Returns:
            Dict[str, float]: 性能指标字典
        """
        performance_metrics = {}

        # 计算加权平均性能
        total_performance = 0.0
        total_weight = 0.0

        for strategy_id, weight in weights.items():
            if strategy_id in self.performance_history:
                recent_performance = list(self.performance_history[strategy_id])[-1]
                if isinstance(recent_performance, dict) and 'score' in recent_performance:
                    total_performance += recent_performance['score'] * weight
                    total_weight += weight

        if total_weight > 0:
            performance_metrics['weighted_performance'] = total_performance / total_weight

        # 计算多样性指标
        if len(predictions) > 1:
            diversity_score = self._calculate_diversity_score(predictions)
            performance_metrics['diversity_score'] = diversity_score

        return performance_metrics

    def _calculate_confidence(self, predictions: Dict[str, Any],


                              weights: Dict[str, float],
                              risk_metrics: Dict[str, float]) -> float:
        """
        计算置信度

        Args:
            predictions: 策略预测字典
            weights: 权重字典
            risk_metrics: 风险指标字典

        Returns:
            float: 置信度
        """
        # 基于权重分布的置信度
        weight_confidence = 1.0 - np.std(list(weights.values()))

        # 基于风险指标的置信度
        risk_confidence = 1.0
        if 'portfolio_volatility' in risk_metrics:
            vol = risk_metrics['portfolio_volatility']
            risk_confidence = max(0.0, 1.0 - vol)

        # 基于预测一致性的置信度
        consistency_confidence = self._calculate_prediction_consistency(predictions)

        # 综合置信度
        confidence = (weight_confidence + risk_confidence + consistency_confidence) / 3

        return max(0.0, min(1.0, confidence))

    def _calculate_performance_score(self, strategy_id: str) -> float:
        """
        计算策略性能分数

        Args:
            strategy_id: 策略ID

        Returns:
            float: 性能分数
        """
        if strategy_id not in self.performance_history:
            return 0.0

        recent_performances = list(self.performance_history[strategy_id])
        if not recent_performances:
            return 0.0

        # 计算最近性能的平均值
        scores = []
        for perf in recent_performances[-10:]:  # 最近10次
            if isinstance(perf, dict) and 'score' in perf:
                scores.append(perf['score'])

        return np.mean(scores) if scores else 0.0

    def _calculate_diversity_score(self, predictions: Dict[str, Any]) -> float:
        """
        计算多样性分数

        Args:
            predictions: 策略预测字典

        Returns:
            float: 多样性分数
        """
        if len(predictions) < 2:
            return 0.0

        # 计算预测之间的相关性
        correlations = []
        prediction_list = list(predictions.values())

        for i in range(len(prediction_list)):
            for j in range(i + 1, len(prediction_list)):
                if isinstance(prediction_list[i], pd.DataFrame) and isinstance(prediction_list[j], pd.DataFrame):
                    # 计算DataFrame之间的相关性
                    corr = self._calculate_dataframe_correlation(
                        prediction_list[i], prediction_list[j]
                    )
                    correlations.append(abs(corr))

        if correlations:
            # 多样性分数 = 1 - 平均相关性
            diversity_score = 1.0 - np.mean(correlations)
            return max(0.0, diversity_score)

        return 0.0

    def _calculate_dataframe_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        计算两个DataFrame之间的相关性

        Args:
            df1: 第一个DataFrame
            df2: 第二个DataFrame

        Returns:
            float: 相关性系数
        """
        try:
            # 找到共同的列
            common_columns = set(df1.columns) & set(df2.columns)
            if not common_columns:
                return 0.0

            # 计算相关性
            correlations = []
            for col in common_columns:
                if col in df1.columns and col in df2.columns:
                    corr = df1[col].corr(df2[col])
                    if not pd.isna(corr):
                        correlations.append(corr)

            return np.mean(correlations) if correlations else 0.0

        except Exception:
            return 0.0

    def _calculate_prediction_consistency(self, predictions: Dict[str, Any]) -> float:
        """
        计算预测一致性

        Args:
            predictions: 策略预测字典

        Returns:
            float: 一致性分数
        """
        if len(predictions) < 2:
            return 1.0

        # 计算预测方向的一致性
        directions = []
        for prediction in predictions.values():
            if isinstance(prediction, pd.DataFrame) and 'signal' in prediction.columns:
                # 提取信号方向
                signals = prediction['signal'].fillna(0)
                directions.append(signals)
            elif isinstance(prediction, (int, float, np.number)):
                directions.append([1 if prediction > 0 else -1 if prediction < 0 else 0])

        if directions:
            # 计算方向一致性
            consistency_scores = []
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                if len(directions[i]) == len(directions[j]):
                    agreement = np.mean(np.array(directions[i]) == np.array(directions[j]))
                    consistency_scores.append(agreement)

            return np.mean(consistency_scores) if consistency_scores else 0.0

        return 0.0

    def _update_performance_history(self, result: IntegrationResult):
        """
        更新性能历史

        Args:
            result: 集成结果
        """
        timestamp = result.timestamp

        for strategy_id, prediction in result.strategy_predictions.items():
            # 计算策略性能分数
            performance_score = self._calculate_strategy_performance_score(
                strategy_id, prediction, result
            )

            self.performance_history[strategy_id].append({
                'timestamp': timestamp,
                'score': performance_score,
                'prediction': prediction
            })

    def _calculate_strategy_performance_score(self, strategy_id: str,


                                              prediction: Any,
                                              result: IntegrationResult) -> float:
        """
        计算策略性能分数

        Args:
            strategy_id: 策略ID
            prediction: 策略预测
            result: 集成结果

        Returns:
            float: 性能分数
        """
        # 基础分数
        base_score = 0.5

        # 权重贡献
        weight = result.weights.get(strategy_id, 0.0)
        weight_score = weight * 0.3

        # 预测质量（基于置信度）
        confidence_score = result.confidence * 0.2

        # 风险贡献
        risk_score = 0.0
        if 'portfolio_volatility' in result.risk_metrics:
            vol = result.risk_metrics['portfolio_volatility']
            risk_score = max(0.0, 1.0 - vol) * 0.1

        total_score = base_score + weight_score + confidence_score + risk_score
        return min(1.0, total_score)

    def rebalance_weights(self) -> Dict[str, float]:
        """
        重新平衡权重

        Returns:
            Dict[str, float]: 新的权重字典
        """
        with self.lock:
            if not self.strategies:
                return {}

            # 计算新的权重
            new_weights = self.weight_optimizer.rebalance_weights(
                self.strategies,
                self.performance_history,
                self.correlation_matrix,
                self.config
            )

            # 更新权重
            for strategy_id, weight in new_weights.items():
                self.update_strategy_weight(strategy_id, weight)

            logger.info(f"权重重新平衡完成: {new_weights}")
            return new_weights

    def get_integration_status(self) -> Dict[str, Any]:
        """
        获取集成状态

        Returns:
            Dict[str, Any]: 集成状态信息
        """
        with self.lock:
            status = {
                'integration_id': self.config.integration_id,
                'total_strategies': len(self.strategies),
                'active_strategies': len([s for s in self.strategies.values() if s.status == 'active']),
                'current_weights': {s.strategy_id: s.weight for s in self.strategies.values()},
                'last_rebalance': getattr(self, '_last_rebalance', None),
                'performance_summary': self._get_performance_summary()
            }
            return status

    def _get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要

        Returns:
            Dict[str, Any]: 性能摘要
        """
        summary = {}

        for strategy_id, strategy_info in self.strategies.items():
            if strategy_id in self.performance_history:
                recent_scores = [
                    perf['score'] for perf in self.performance_history[strategy_id]
                    if isinstance(perf, dict) and 'score' in perf
                ]

                if recent_scores:
                    summary[strategy_id] = {
                        'avg_score': np.mean(recent_scores),
                        'max_score': np.max(recent_scores),
                        'min_score': np.min(recent_scores),
                        'recent_trend': 'up' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[-2] else 'down'
                    }

        return summary


class PerformanceMonitor:

    """性能监控器"""

    def __init__(self):

        self.metrics = defaultdict(list)
        self.alerts = []

    def record_metric(self, strategy_id: str, metric_name: str, value: float):
        """记录指标"""
        self.metrics[f"{strategy_id}_{metric_name}"].append({
            'timestamp': datetime.now(),
            'value': value
        })

    def get_metrics(self, strategy_id: str, metric_name: str,


                    lookback_days: int = 30) -> List[float]:
        """获取指标历史"""
        key = f"{strategy_id}_{metric_name}"
        if key not in self.metrics:
            return []

        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        recent_metrics = [
            m['value'] for m in self.metrics[key]
            if m['timestamp'] >= cutoff_time
        ]

        return recent_metrics


class WeightOptimizer:

    """权重优化器"""

    def __init__(self):

        self.optimization_history = []

    def optimize_weights(self, performance_scores: Dict[str, float],


                         correlation_matrix: pd.DataFrame,
                         config: IntegrationConfig) -> Dict[str, float]:
        """
        优化权重

        Args:
            performance_scores: 性能分数字典
            correlation_matrix: 相关性矩阵
            config: 集成配置

        Returns:
            Dict[str, float]: 优化后的权重
        """
        if not performance_scores:
            return {}

        # 基于性能的权重分配
        total_score = sum(performance_scores.values())
        if total_score == 0:
            # 等权重分配
            n_strategies = len(performance_scores)
            weights = {strategy_id: 1.0 / n_strategies for strategy_id in performance_scores.keys()}
        else:
            # 基于性能的权重分配
            weights = {strategy_id: score / total_score for strategy_id,
                       score in performance_scores.items()}

        # 应用权重限制
        weights = self._apply_weight_constraints(weights, config)

        # 考虑多样性
        if correlation_matrix is not None and not correlation_matrix.empty:
            weights = self._adjust_for_diversity(weights, correlation_matrix, config)

        return weights

    def rebalance_weights(self, strategies: Dict[str, StrategyInfo],


                          performance_history: Dict[str, deque],
                          correlation_matrix: pd.DataFrame,
                          config: IntegrationConfig) -> Dict[str, float]:
        """
        重新平衡权重

        Args:
            strategies: 策略字典
            performance_history: 性能历史
            correlation_matrix: 相关性矩阵
            config: 集成配置

        Returns:
            Dict[str, float]: 新的权重
        """
        # 计算性能分数
        performance_scores = {}
        for strategy_id, strategy_info in strategies.items():
            if strategy_info.status == "active":
                score = self._calculate_recent_performance(strategy_id, performance_history)
                performance_scores[strategy_id] = score

        # 优化权重
        new_weights = self.optimize_weights(performance_scores, correlation_matrix, config)

        return new_weights

    def _apply_weight_constraints(self, weights: Dict[str, float],


                                  config: IntegrationConfig) -> Dict[str, float]:
        """
        应用权重约束

        Args:
            weights: 原始权重
            config: 集成配置

        Returns:
            Dict[str, float]: 约束后的权重
        """
        constrained_weights = {}

        for strategy_id, weight in weights.items():
            # 应用最小和最大权重限制
            constrained_weight = max(config.min_weight, min(config.max_weight, weight))
            constrained_weights[strategy_id] = constrained_weight

        # 重新归一化
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                strategy_id: weight / total_weight
                for strategy_id, weight in constrained_weights.items()
            }

        return constrained_weights

    def _adjust_for_diversity(self, weights: Dict[str, float],


                              correlation_matrix: pd.DataFrame,
                              config: IntegrationConfig) -> Dict[str, float]:
        """
        基于多样性调整权重

        Args:
            weights: 原始权重
            correlation_matrix: 相关性矩阵
            config: 集成配置

        Returns:
            Dict[str, float]: 调整后的权重
        """
        # 计算策略间的相关性
        strategy_ids = list(weights.keys())
        if len(strategy_ids) < 2:
            return weights

        # 调整权重以增加多样性
        adjusted_weights = weights.copy()

        for i, strategy_id1 in enumerate(strategy_ids):
            for j, strategy_id2 in enumerate(strategy_ids[i + 1:], i + 1):
                if strategy_id1 in correlation_matrix.index and strategy_id2 in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[strategy_id1, strategy_id2]

                    # 如果相关性过高，减少权重
                    if abs(correlation) > config.correlation_threshold:
                        # 减少权重较高的策略的权重
                        if adjusted_weights[strategy_id1] > adjusted_weights[strategy_id2]:
                            reduction = config.diversity_factor * adjusted_weights[strategy_id1]
                            adjusted_weights[strategy_id1] -= reduction
                            adjusted_weights[strategy_id2] += reduction
                        else:
                            reduction = config.diversity_factor * adjusted_weights[strategy_id2]
                            adjusted_weights[strategy_id2] -= reduction
                            adjusted_weights[strategy_id1] += reduction

        # 重新归一化
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                strategy_id: weight / total_weight
                for strategy_id, weight in adjusted_weights.items()
            }

        return adjusted_weights

    def _calculate_recent_performance(self, strategy_id: str,


                                      performance_history: Dict[str, deque]) -> float:
        """
        计算最近性能

        Args:
            strategy_id: 策略ID
            performance_history: 性能历史

        Returns:
            float: 最近性能分数
        """
        if strategy_id not in performance_history:
            return 0.0

        recent_performances = list(performance_history[strategy_id])
        if not recent_performances:
            return 0.0

        # 计算最近10次的平均性能
        recent_scores = []
        for perf in recent_performances[-10:]:
            if isinstance(perf, dict) and 'score' in perf:
                recent_scores.append(perf['score'])

        return np.mean(recent_scores) if recent_scores else 0.0


class RiskManager:

    """风险管理器"""

    def __init__(self):

        self.risk_metrics = {}

    def calculate_portfolio_volatility(self, returns: pd.DataFrame,


                                       weights: Dict[str, float]) -> float:
        """
        计算组合波动率

        Args:
            returns: 收益率DataFrame
            weights: 权重字典

        Returns:
            float: 组合波动率
        """
        if returns.empty or not weights:
            return 0.0

        try:
            # 计算加权收益率
            weighted_returns = pd.Series(0.0, index=returns.index)
            for strategy_id, weight in weights.items():
                if strategy_id in returns.columns:
                    weighted_returns += returns[strategy_id] * weight

            # 计算波动率
            volatility = weighted_returns.std() * np.sqrt(252)  # 年化
            return float(volatility)

        except Exception as e:
            logger.error(f"计算组合波动率失败: {e}")
            return 0.0

    def calculate_max_drawdown(self, predictions: Dict[str, Any],


                               weights: Dict[str, float]) -> float:
        """
        计算最大回撤

        Args:
            predictions: 预测字典
            weights: 权重字典

        Returns:
            float: 最大回撤
        """
        try:
            # 计算组合收益
            portfolio_returns = pd.Series(0.0)

            for strategy_id, prediction in predictions.items():
                if isinstance(prediction, pd.DataFrame) and 'returns' in prediction.columns:
                    weight = weights.get(strategy_id, 0.0)
                    portfolio_returns += prediction['returns'] * weight

            if portfolio_returns.empty:
                return 0.0

            # 计算累积收益
            cumulative_returns = (1 + portfolio_returns).cumprod()

            # 计算最大回撤
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            return float(max_drawdown)

        except Exception as e:
            logger.error(f"计算最大回撤失败: {e}")
            return 0.0

    def calculate_var(self, predictions: Dict[str, Any],


                      weights: Dict[str, float],
                      confidence_level: float = 0.95) -> float:
        """
        计算VaR

        Args:
            predictions: 预测字典
            weights: 权重字典
            confidence_level: 置信水平

        Returns:
            float: VaR值
        """
        try:
            # 计算组合收益
            portfolio_returns = pd.Series(0.0)

            for strategy_id, prediction in predictions.items():
                if isinstance(prediction, pd.DataFrame) and 'returns' in prediction.columns:
                    weight = weights.get(strategy_id, 0.0)
                    portfolio_returns += prediction['returns'] * weight

            if portfolio_returns.empty:
                return 0.0

            # 计算VaR
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            return float(var)

        except Exception as e:
            logger.error(f"计算VaR失败: {e}")
            return 0.0
