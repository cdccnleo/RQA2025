#!/usr/bin/env python3
"""
AI驱动优化器
基于机器学习的数据采集策略优化和预测性调度
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import os
from pathlib import Path
import json

from src.core.orchestration.performance_optimizer import PerformanceMetrics
from src.core.cache.redis_cache import RedisCache


@dataclass
class OptimizationFeatures:
    """优化特征"""
    # 系统状态特征
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    active_tasks: int = 0

    # 任务特征
    task_priority: int = 1  # 1-5
    data_volume_estimate: int = 0  # 预估数据量
    historical_success_rate: float = 0.0  # 历史成功率
    average_processing_time: float = 0.0  # 平均处理时间

    # 时间特征
    hour_of_day: int = 0
    day_of_week: int = 0
    is_trading_hours: bool = False

    # 市场特征
    market_volatility: float = 0.0  # 市场波动率
    data_source_reliability: float = 0.0  # 数据源可靠性

    def to_array(self) -> np.ndarray:
        """转换为特征数组"""
        return np.array([
            self.cpu_usage,
            self.memory_usage,
            self.network_latency,
            self.active_tasks,
            self.task_priority,
            self.data_volume_estimate,
            self.historical_success_rate,
            self.average_processing_time,
            self.hour_of_day,
            self.day_of_week,
            int(self.is_trading_hours),
            self.market_volatility,
            self.data_source_reliability
        ])


@dataclass
class OptimizationPrediction:
    """优化预测结果"""
    optimal_concurrency: int = 1
    predicted_duration: float = 0.0  # 预测执行时间（秒）
    success_probability: float = 0.0  # 成功概率
    recommended_schedule_time: Optional[datetime] = None  # 推荐调度时间
    confidence_score: float = 0.0  # 置信度分数


@dataclass
class TrainingData:
    """训练数据"""
    features: List[OptimizationFeatures] = field(default_factory=list)
    labels: Dict[str, List[float]] = field(default_factory=dict)  # 不同目标的标签
    timestamps: List[datetime] = field(default_factory=list)


class AIModelManager:
    """AI模型管理器"""

    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # 模型文件名
        self.concurrency_model_file = self.model_dir / "concurrency_predictor.pkl"
        self.duration_model_file = self.model_dir / "duration_predictor.pkl"
        self.success_model_file = self.model_dir / "success_predictor.pkl"
        self.scaler_file = self.model_dir / "feature_scaler.pkl"

        # 模型实例
        self.concurrency_model: Optional[RandomForestRegressor] = None
        self.duration_model: Optional[RandomForestRegressor] = None
        self.success_model: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None

        # 加载现有模型
        self._load_models()

    def _load_models(self):
        """加载模型"""
        try:
            if self.concurrency_model_file.exists():
                self.concurrency_model = joblib.load(self.concurrency_model_file)
            if self.duration_model_file.exists():
                self.duration_model = joblib.load(self.duration_model_file)
            if self.success_model_file.exists():
                self.success_model = joblib.load(self.success_model_file)
            if self.scaler_file.exists():
                self.scaler = joblib.load(self.scaler_file)

            logging.info("AI模型加载完成")

        except Exception as e:
            logging.warning(f"加载AI模型失败: {e}")
            # 初始化默认模型
            self._initialize_default_models()

    def _initialize_default_models(self):
        """初始化默认模型"""
        self.concurrency_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.duration_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.success_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()

    def _save_models(self):
        """保存模型"""
        try:
            if self.concurrency_model:
                joblib.dump(self.concurrency_model, self.concurrency_model_file)
            if self.duration_model:
                joblib.dump(self.duration_model, self.duration_model_file)
            if self.success_model:
                joblib.dump(self.success_model, self.success_model_file)
            if self.scaler:
                joblib.dump(self.scaler, self.scaler_file)

            logging.info("AI模型保存完成")

        except Exception as e:
            logging.error(f"保存AI模型失败: {e}")

    def train_models(self, training_data: TrainingData, test_size: float = 0.2):
        """
        训练模型

        Args:
            training_data: 训练数据
            test_size: 测试集比例
        """
        if len(training_data.features) < 10:
            logging.warning("训练数据不足，跳过模型训练")
            return

        try:
            # 准备特征数据
            X = np.array([f.to_array() for f in training_data.features])

            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)

            # 训练并发度预测模型
            if 'concurrency' in training_data.labels:
                y_concurrency = np.array(training_data.labels['concurrency'])
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_concurrency, test_size=test_size, random_state=42
                )

                self.concurrency_model.fit(X_train, y_train)

                # 评估模型
                y_pred = self.concurrency_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                logging.info(f"并发度预测模型MAE: {mae:.2f}")

            # 训练持续时间预测模型
            if 'duration' in training_data.labels:
                y_duration = np.array(training_data.labels['duration'])
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_duration, test_size=test_size, random_state=42
                )

                self.duration_model.fit(X_train, y_train)

                y_pred = self.duration_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                logging.info(f"持续时间预测模型MAE: {mae:.2f}")

            # 训练成功率预测模型
            if 'success' in training_data.labels:
                y_success = np.array(training_data.labels['success'])
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_success, test_size=test_size, random_state=42
                )

                self.success_model.fit(X_train, y_train)

                y_pred = self.success_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                logging.info(f"成功率预测模型准确率: {accuracy:.2f}")

            # 保存模型
            self._save_models()

            logging.info("AI模型训练完成")

        except Exception as e:
            logging.error(f"AI模型训练失败: {e}")

    def predict_optimization(self, features: OptimizationFeatures) -> OptimizationPrediction:
        """
        预测优化结果

        Args:
            features: 优化特征

        Returns:
            预测结果
        """
        try:
            X = features.to_array().reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            prediction = OptimizationPrediction()

            # 预测最优并发度
            if self.concurrency_model:
                concurrency_pred = self.concurrency_model.predict(X_scaled)[0]
                prediction.optimal_concurrency = max(1, int(round(concurrency_pred)))

            # 预测执行时间
            if self.duration_model:
                duration_pred = self.duration_model.predict(X_scaled)[0]
                prediction.predicted_duration = max(0, duration_pred)

            # 预测成功概率
            if self.success_model:
                success_prob = self.success_model.predict_proba(X_scaled)[0]
                prediction.success_probability = success_prob[1] if len(success_prob) > 1 else 0.5

            # 计算置信度（基于预测一致性）
            prediction.confidence_score = min(0.95, max(0.1, prediction.success_probability))

            # 推荐调度时间（避开高峰期）
            prediction.recommended_schedule_time = self._recommend_schedule_time(features)

            return prediction

        except Exception as e:
            logging.error(f"AI预测失败: {e}")
            # 返回保守的默认预测
            return OptimizationPrediction(
                optimal_concurrency=1,
                predicted_duration=300.0,  # 5分钟
                success_probability=0.5,
                confidence_score=0.1
            )

    def _recommend_schedule_time(self, features: OptimizationFeatures) -> Optional[datetime]:
        """推荐调度时间"""
        now = datetime.now()

        # 如果当前是交易高峰期，推荐延后到非高峰期
        if features.is_trading_hours and features.cpu_usage > 70:
            # 延后到晚上非交易时间
            evening_time = now.replace(hour=19, minute=0, second=0, microsecond=0)
            if evening_time > now:
                return evening_time
            else:
                # 次日晚上
                return evening_time + timedelta(days=1)

        return None


class PerformanceDataCollector:
    """性能数据收集器"""

    def __init__(self, redis_config: Dict[str, Any]):
        self.redis = RedisCache(redis_config)
        self.data_key = "ai_optimizer:performance_data"
        self.logger = logging.getLogger(__name__)

    async def collect_performance_data(self, metrics: PerformanceMetrics,
                                     features: OptimizationFeatures) -> TrainingData:
        """
        收集性能数据用于训练

        Args:
            metrics: 性能指标
            features: 优化特征

        Returns:
            训练数据
        """
        training_data = TrainingData()

        try:
            # 从Redis加载历史数据
            historical_data = await self.redis.get_json(self.data_key) or {
                "features": [],
                "labels": {"concurrency": [], "duration": [], "success": []},
                "timestamps": []
            }

            # 添加新数据点
            historical_data["features"].append(features.__dict__)
            historical_data["labels"]["concurrency"].append(features.active_tasks)  # 使用实际并发度作为标签
            historical_data["labels"]["duration"].append(metrics.duration_seconds)
            historical_data["labels"]["success"].append(1.0 if metrics.successful_requests > 0 else 0.0)
            historical_data["timestamps"].append(metrics.start_time.isoformat())

            # 限制历史数据大小（最近1000个数据点）
            max_history_size = 1000
            if len(historical_data["features"]) > max_history_size:
                for key in historical_data:
                    if isinstance(historical_data[key], list):
                        historical_data[key] = historical_data[key][-max_history_size:]

            # 保存回Redis
            await self.redis.set_json(self.data_key, historical_data)

            # 转换为TrainingData对象
            training_data.features = [OptimizationFeatures(**f) for f in historical_data["features"]]
            training_data.labels = historical_data["labels"]
            training_data.timestamps = [datetime.fromisoformat(t) for t in historical_data["timestamps"]]

            self.logger.info(f"性能数据收集完成，当前数据点: {len(training_data.features)}")

        except Exception as e:
            self.logger.error(f"性能数据收集失败: {e}")

        return training_data

    async def get_training_data(self) -> TrainingData:
        """获取训练数据"""
        try:
            data = await self.redis.get_json(self.data_key)
            if data:
                training_data = TrainingData()
                training_data.features = [OptimizationFeatures(**f) for f in data["features"]]
                training_data.labels = data["labels"]
                training_data.timestamps = [datetime.fromisoformat(t) for t in data["timestamps"]]
                return training_data

        except Exception as e:
            self.logger.error(f"获取训练数据失败: {e}")

        return TrainingData()


class MarketConditionAnalyzer:
    """市场状况分析器"""

    def __init__(self, redis_config: Dict[str, Any]):
        self.redis = RedisCache(redis_config)
        self.market_data_key = "market_conditions"

    async def analyze_market_conditions(self) -> Dict[str, float]:
        """
        分析市场状况

        Returns:
            市场特征字典
        """
        try:
            # 这里应该从市场数据源获取实时市场数据
            # 暂时返回模拟数据
            market_data = await self.redis.get_json(self.market_data_key) or {}

            # 计算波动率（基于最近价格变动）
            volatility = market_data.get('volatility', 0.15)  # 默认15%波动率

            # 市场活跃度
            activity_level = market_data.get('activity', 0.7)  # 默认70%活跃度

            # 节假日影响
            holiday_effect = self._calculate_holiday_effect()

            return {
                'volatility': volatility,
                'activity_level': activity_level,
                'holiday_effect': holiday_effect,
                'trading_hours': self._is_trading_hours()
            }

        except Exception as e:
            logging.error(f"市场状况分析失败: {e}")
            return {
                'volatility': 0.15,
                'activity_level': 0.7,
                'holiday_effect': 0.0,
                'trading_hours': True
            }

    def _calculate_holiday_effect(self) -> float:
        """计算节假日影响"""
        now = datetime.now()

        # 检查是否接近节假日（简化实现）
        # 实际应该查询节假日日历
        if now.month in [1, 2, 9, 10]:  # 春节、国庆前后
            return 0.3  # 30%的影响
        elif now.weekday() >= 5:  # 周末
            return 0.8  # 80%的影响
        else:
            return 0.0

    def _is_trading_hours(self) -> bool:
        """检查是否交易时间"""
        now = datetime.now()
        weekday = now.weekday()  # 0-6, 周一到周日

        # 周一到周五，9:30-11:30, 13:00-15:00
        if weekday < 5:
            morning_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
            morning_end = now.replace(hour=11, minute=30, second=0, microsecond=0)
            afternoon_start = now.replace(hour=13, minute=0, second=0, microsecond=0)
            afternoon_end = now.replace(hour=15, minute=0, second=0, microsecond=0)

            return (morning_start <= now <= morning_end) or (afternoon_start <= now <= afternoon_end)

        return False


class AIDrivenOptimizer:
    """AI驱动优化器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 初始化组件
        self.model_manager = AIModelManager(config.get('model_dir', './models'))
        self.data_collector = PerformanceDataCollector(config.get('redis_config', {}))
        self.market_analyzer = MarketConditionAnalyzer(config.get('redis_config', {}))

        # 配置参数
        self.retraining_interval = timedelta(hours=24)  # 每24小时重新训练一次
        self.last_training_time: Optional[datetime] = None

        # 性能监控
        self.optimization_attempts = 0
        self.optimization_successes = 0

    async def optimize_task_execution(self, task_payload: Dict[str, Any]) -> OptimizationPrediction:
        """
        优化任务执行

        Args:
            task_payload: 任务负载

        Returns:
            优化预测结果
        """
        try:
            # 收集当前系统特征
            features = await self._collect_current_features(task_payload)

            # 检查是否需要重新训练模型
            await self._check_model_retraining()

            # AI预测优化结果
            prediction = self.model_manager.predict_optimization(features)

            # 应用业务规则调整
            prediction = await self._apply_business_rules(prediction, features)

            # 记录优化尝试
            self.optimization_attempts += 1
            if prediction.confidence_score > 0.7:
                self.optimization_successes += 1

            self.logger.info(f"AI优化完成: 并发度={prediction.optimal_concurrency}, "
                           f"预测时间={prediction.predicted_duration:.1f}s, "
                           f"成功率={prediction.success_probability:.2%}")

            return prediction

        except Exception as e:
            self.logger.error(f"AI优化失败: {e}")
            # 返回保守的默认值
            return OptimizationPrediction(
                optimal_concurrency=1,
                predicted_duration=300.0,
                success_probability=0.5,
                confidence_score=0.1
            )

    async def _collect_current_features(self, task_payload: Dict[str, Any]) -> OptimizationFeatures:
        """收集当前系统特征"""
        features = OptimizationFeatures()

        try:
            # 系统状态特征
            import psutil
            features.cpu_usage = psutil.cpu_percent(interval=1)
            features.memory_usage = psutil.virtual_memory().percent
            features.network_latency = 50.0  # 模拟网络延迟，实际应该测量
            features.active_tasks = len(asyncio.all_tasks())

            # 任务特征
            features.task_priority = task_payload.get('priority', 1)
            features.data_volume_estimate = task_payload.get('estimated_records', 1000)

            # 从历史数据获取统计信息
            symbol = task_payload.get('symbol', '')
            if symbol:
                historical_stats = await self._get_historical_task_stats(symbol)
                features.historical_success_rate = historical_stats.get('success_rate', 0.8)
                features.average_processing_time = historical_stats.get('avg_duration', 300.0)

            # 时间特征
            now = datetime.now()
            features.hour_of_day = now.hour
            features.day_of_week = now.weekday()

            # 市场特征
            market_conditions = await self.market_analyzer.analyze_market_conditions()
            features.market_volatility = market_conditions['volatility']
            features.data_source_reliability = 0.9  # 模拟数据源可靠性
            features.is_trading_hours = market_conditions['trading_hours']

        except Exception as e:
            self.logger.error(f"特征收集失败: {e}")

        return features

    async def _get_historical_task_stats(self, symbol: str) -> Dict[str, float]:
        """获取历史任务统计"""
        try:
            # 从Redis获取历史统计
            stats_key = f"task_stats:{symbol}"
            stats = await self.data_collector.redis.get_json(stats_key) or {}

            return {
                'success_rate': stats.get('success_rate', 0.8),
                'avg_duration': stats.get('avg_duration', 300.0),
                'total_executions': stats.get('total_executions', 0)
            }

        except Exception as e:
            self.logger.error(f"获取历史统计失败: {e}")
            return {'success_rate': 0.8, 'avg_duration': 300.0, 'total_executions': 0}

    async def _check_model_retraining(self):
        """检查模型重新训练"""
        now = datetime.now()

        if (self.last_training_time is None or
            (now - self.last_training_time) > self.retraining_interval):

            try:
                self.logger.info("开始AI模型重新训练")

                # 获取训练数据
                training_data = await self.data_collector.get_training_data()

                if len(training_data.features) >= 10:  # 最小训练样本数
                    # 训练模型
                    self.model_manager.train_models(training_data)
                    self.last_training_time = now
                    self.logger.info("AI模型重新训练完成")
                else:
                    self.logger.info("训练数据不足，跳过重新训练")

            except Exception as e:
                self.logger.error(f"AI模型重新训练失败: {e}")

    async def _apply_business_rules(self, prediction: OptimizationPrediction,
                                  features: OptimizationFeatures) -> OptimizationPrediction:
        """应用业务规则调整预测结果"""
        # 规则1: 高CPU使用率时降低并发度
        if features.cpu_usage > 80:
            prediction.optimal_concurrency = max(1, prediction.optimal_concurrency // 2)
            prediction.confidence_score *= 0.8

        # 规则2: 大数据量任务适当降低并发度
        if features.data_volume_estimate > 100000:
            prediction.optimal_concurrency = max(1, prediction.optimal_concurrency - 1)

        # 规则3: 高优先级任务可以适当提高并发度
        if features.task_priority >= 4:
            prediction.optimal_concurrency = min(10, prediction.optimal_concurrency + 1)

        # 规则4: 非交易时间可以提高并发度
        if not features.is_trading_hours:
            prediction.optimal_concurrency = min(15, prediction.optimal_concurrency + 2)

        # 规则5: 高波动率市场适当降低并发度（减少对市场的影响）
        if features.market_volatility > 0.25:
            prediction.optimal_concurrency = max(1, prediction.optimal_concurrency - 1)

        return prediction

    async def record_execution_result(self, task_payload: Dict[str, Any],
                                    prediction: OptimizationPrediction,
                                    actual_metrics: PerformanceMetrics):
        """
        记录执行结果用于模型训练

        Args:
            task_payload: 任务负载
            prediction: AI预测结果
            actual_metrics: 实际执行指标
        """
        try:
            # 收集特征
            features = await self._collect_current_features(task_payload)

            # 记录训练数据
            await self.data_collector.collect_performance_data(actual_metrics, features)

            # 更新历史统计
            await self._update_historical_stats(task_payload, actual_metrics)

            self.logger.info("执行结果记录完成")

        except Exception as e:
            self.logger.error(f"执行结果记录失败: {e}")

    async def _update_historical_stats(self, task_payload: Dict[str, Any],
                                     metrics: PerformanceMetrics):
        """更新历史统计"""
        try:
            symbol = task_payload.get('symbol', 'unknown')
            stats_key = f"task_stats:{symbol}"

            # 获取现有统计
            existing_stats = await self.data_collector.redis.get_json(stats_key) or {
                'total_executions': 0,
                'successful_executions': 0,
                'total_duration': 0.0,
                'success_rate': 0.0,
                'avg_duration': 0.0
            }

            # 更新统计
            existing_stats['total_executions'] += 1
            existing_stats['total_duration'] += metrics.duration_seconds

            if metrics.successful_requests > metrics.failed_requests:
                existing_stats['successful_executions'] += 1

            # 重新计算平均值
            if existing_stats['total_executions'] > 0:
                existing_stats['success_rate'] = (
                    existing_stats['successful_executions'] / existing_stats['total_executions']
                )
                existing_stats['avg_duration'] = (
                    existing_stats['total_duration'] / existing_stats['total_executions']
                )

            # 保存统计
            await self.data_collector.redis.set_json(stats_key, existing_stats)

        except Exception as e:
            self.logger.error(f"更新历史统计失败: {e}")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计"""
        total_attempts = self.optimization_attempts
        success_rate = self.optimization_successes / total_attempts if total_attempts > 0 else 0

        return {
            'total_optimization_attempts': total_attempts,
            'optimization_success_rate': success_rate,
            'model_last_training': self.last_training_time.isoformat() if self.last_training_time else None,
            'model_performance': {
                'concurrency_model_available': self.model_manager.concurrency_model is not None,
                'duration_model_available': self.model_manager.duration_model is not None,
                'success_model_available': self.model_manager.success_model is not None
            }
        }

    async def predict_optimal_schedule(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        预测最优调度时间

        Args:
            task_payload: 任务负载

        Returns:
            调度建议
        """
        try:
            features = await self._collect_current_features(task_payload)
            prediction = self.model_manager.predict_optimization(features)

            recommendation = {
                'recommended_concurrency': prediction.optimal_concurrency,
                'predicted_duration_seconds': prediction.predicted_duration,
                'success_probability': prediction.success_probability,
                'confidence_score': prediction.confidence_score,
                'suggested_schedule_time': (
                    prediction.recommended_schedule_time.isoformat()
                    if prediction.recommended_schedule_time else None
                ),
                'reasoning': self._generate_schedule_reasoning(prediction, features)
            }

            return recommendation

        except Exception as e:
            self.logger.error(f"调度预测失败: {e}")
            return {
                'recommended_concurrency': 1,
                'predicted_duration_seconds': 300.0,
                'success_probability': 0.5,
                'confidence_score': 0.1,
                'suggested_schedule_time': None,
                'reasoning': '预测失败，使用保守策略'
            }

    def _generate_schedule_reasoning(self, prediction: OptimizationPrediction,
                                   features: OptimizationFeatures) -> str:
        """生成调度推理说明"""
        reasons = []

        if features.cpu_usage > 70:
            reasons.append("CPU使用率较高，降低并发度")
        if features.memory_usage > 80:
            reasons.append("内存使用率较高，降低并发度")
        if features.is_trading_hours and features.market_volatility > 0.2:
            reasons.append("交易时间市场波动较大，谨慎调度")
        if not features.is_trading_hours:
            reasons.append("非交易时间可以提高并发度")
        if features.task_priority >= 4:
            reasons.append("高优先级任务，适当提高并发度")

        if not reasons:
            reasons.append("基于历史数据和当前系统状态的AI预测")

        return "; ".join(reasons)