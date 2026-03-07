#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoML引擎
AutoML Engine

自动化机器学习策略生成和优化。
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available, using basic optimization")

from ..interfaces.strategy_interfaces import StrategyConfig, StrategyType

logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:

    """AutoML配置"""
    task_type: str = "classification"  # classification, regression
    time_limit: int = 3600  # 时间限制(秒)
    max_models: int = 10
    cv_folds: int = 5
    random_state: int = 42
    metric: str = "accuracy"
    enable_feature_selection: bool = True
    enable_hyperparameter_tuning: bool = True
    ensemble_methods: List[str] = field(default_factory=lambda: ["voting", "stacking"])


@dataclass
class ModelCandidate:

    """模型候选"""
    name: str
    model_class: Any
    param_space: Dict[str, Any]
    preprocessing_steps: List[Any] = field(default_factory=list)


@dataclass
class AutoMLResult:

    """AutoML结果"""
    best_model: Dict[str, Any]
    model_candidates: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    performance_metrics: Dict[str, Any]
    training_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyGenerationResult:

    """策略生成结果"""
    strategy_config: StrategyConfig
    expected_performance: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    generation_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class ModelLibrary:

    """模型库"""

    @staticmethod
    def get_classification_models() -> List[ModelCandidate]:
        """获取分类模型"""
        return [
            ModelCandidate(
                name="RandomForest",
                model_class=RandomForestClassifier,
                param_space={
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            ),
            ModelCandidate(
                name="GradientBoosting",
                model_class=GradientBoostingClassifier,
                param_space={
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            ),
            ModelCandidate(
                name="LogisticRegression",
                model_class=LogisticRegression,
                param_space={
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            ),
            ModelCandidate(
                name="SVM",
                model_class=SVC,
                param_space={
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            )
        ]

    @staticmethod
    def get_regression_models() -> List[ModelCandidate]:
        """获取回归模型"""
        # 这里可以添加回归模型
        return []


class AutoMLTrainer:

    """AutoML训练器"""

    def __init__(self, config: AutoMLConfig):

        self.config = config
        self.models = []
        self.best_model = None
        self.best_score = float('-inf')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder() if config.task_type == "classification" else None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AutoMLResult:
        """训练AutoML模型"""
        start_time = asyncio.get_event_loop().time()

        try:
            logger.info("Starting AutoML training...")

            # 数据预处理
            X_processed, y_processed = self._preprocess_data(X, y)

            # 特征选择
            if self.config.enable_feature_selection:
                X_processed = self._select_features(X_processed, y_processed)

            # 获取模型候选
            model_candidates = self._get_model_candidates()

            # 训练和评估模型
            trained_models = []
            for candidate in model_candidates[:self.config.max_models]:
                model_result = self._train_and_evaluate_model(
                    candidate, X_processed, y_processed
                )
                trained_models.append(model_result)

                # 更新最佳模型
            if model_result['score'] > self.best_score:
                self.best_score = model_result['score']
                self.best_model = model_result

            # 特征重要性分析
            feature_importance = self._analyze_feature_importance(X_processed, y_processed)

            training_time = asyncio.get_event_loop().time() - start_time

            result = AutoMLResult(
                best_model=self.best_model,
                model_candidates=trained_models,
                feature_importance=feature_importance,
                performance_metrics=self._calculate_performance_metrics(trained_models),
                training_time=training_time
            )

            logger.info(f"AutoML training completed in {training_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"AutoML training failed: {e}")
            raise

    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """数据预处理"""
        try:
            # 处理缺失值
            X = X.fillna(X.mean())

            # 编码分类变量
            X = pd.get_dummies(X, drop_first=True)

            # 标准化数值特征
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                X[numeric_columns] = self.scaler.fit_transform(X[numeric_columns])

            # 编码目标变量(如果是分类)
            if self.label_encoder and self.config.task_type == "classification":
                y = pd.Series(self.label_encoder.fit_transform(y))

            return X, y

        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """特征选择"""
        try:
            # 使用随机森林进行特征重要性排序
            rf = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
            rf.fit(X, y)

            # 选择最重要的特征
            feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
            top_features = feature_importance.nlargest(20).index  # 选择前20个特征

            return X[top_features]

        except Exception as e:
            logger.warning(f"Feature selection failed, using all features: {e}")
            return X

    def _get_model_candidates(self) -> List[ModelCandidate]:
        """获取模型候选"""
        if self.config.task_type == "classification":
            return ModelLibrary.get_classification_models()
        elif self.config.task_type == "regression":
            return ModelLibrary.get_regression_models()
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")

    def _train_and_evaluate_model(self, candidate: ModelCandidate,


                                  X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """训练和评估模型"""
        try:
            # 超参数调优
            if self.config.enable_hyperparameter_tuning and OPTUNA_AVAILABLE:
                best_params = self._optimize_hyperparameters(candidate, X, y)
            else:
                # 使用默认参数
                best_params = {}
                for param, values in candidate.param_space.items():
                    best_params[param] = values[0] if values else None

            # 创建模型管道
            pipeline_steps = candidate.preprocessing_steps + [
                ('classifier', candidate.model_class(**best_params))
            ]
            pipeline = Pipeline(pipeline_steps)

            # 交叉验证
            cv_scores = cross_val_score(
                pipeline, X, y,
                cv=self.config.cv_folds,
                scoring=self.config.metric
            )

            # 在完整数据集上训练最终模型
            pipeline.fit(X, y)

            # 预测和评估
            y_pred = pipeline.predict(X)
            accuracy = accuracy_score(y, y_pred)

            result = {
                'name': candidate.name,
                'params': best_params,
                'score': cv_scores.mean(),
                'cv_scores': cv_scores.tolist(),
                'accuracy': accuracy,
                'model': pipeline,
                'feature_importance': self._get_model_feature_importance(pipeline, X.columns)
            }

            logger.info(f"Model {candidate.name} trained with score: {cv_scores.mean():.4f}")
            return result

        except Exception as e:
            logger.error(f"Model training failed for {candidate.name}: {e}")
            return {
                'name': candidate.name,
                'params': {},
                'score': 0.0,
                'cv_scores': [],
                'accuracy': 0.0,
                'model': None,
                'feature_importance': {}
            }

    def _optimize_hyperparameters(self, candidate: ModelCandidate,


                                  X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """超参数优化"""

        def objective(trial):

            params = {}
            for param, values in candidate.param_space.items():
                if isinstance(values[0], int):
                    params[param] = trial.suggest_int(param, min(values), max(values))
                elif isinstance(values[0], float):
                    params[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    params[param] = trial.suggest_categorical(param, values)

            # 创建模型
            model = candidate.model_class(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring=self.config.metric)
            return scores.mean()

        # 创建Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        return study.best_params

    def _get_model_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """获取模型特征重要性"""
        try:
            if hasattr(model.named_steps.get('classifier', model), 'feature_importances_'):
                importance = model.named_steps.get('classifier', model).feature_importances_
                return dict(zip(feature_names, importance))
            else:
                return {}
        except Exception:
            return {}

    def _analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """分析特征重要性"""
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
            rf.fit(X, y)

            importance = pd.Series(rf.feature_importances_, index=X.columns)
            return importance.to_dict()

        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
            return {}

    def _calculate_performance_metrics(self, trained_models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算性能指标"""
        scores = [model['score'] for model in trained_models if model['score'] > 0]
        accuracies = [model['accuracy'] for model in trained_models if model['accuracy'] > 0]

        return {
            'mean_score': np.mean(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0,
            'best_score': max(scores) if scores else 0,
            'mean_accuracy': np.mean(accuracies) if accuracies else 0,
            'models_trained': len(trained_models),
            'successful_models': len([m for m in trained_models if m['score'] > 0])
        }


class StrategyGenerator:

    """策略生成器"""

    def __init__(self, automl_config: AutoMLConfig = None):

        self.automl_config = automl_config or AutoMLConfig()
        self.automl_trainer = AutoMLTrainer(self.automl_config)

    async def generate_strategy_from_data(self, historical_data: pd.DataFrame,
                                          target_column: str,
                                          feature_columns: List[str] = None) -> StrategyGenerationResult:
        """从数据生成策略"""
        try:
            logger.info("Starting strategy generation from data...")

            # 准备数据
            if feature_columns:
                X = historical_data[feature_columns]
            else:
                X = historical_data.drop(columns=[target_column])

            y = historical_data[target_column]

            # AutoML训练
            automl_result = await asyncio.get_event_loop().run_in_executor(
                None, self.automl_trainer.fit, X, y
            )

            # 生成策略配置
            strategy_config = self._generate_strategy_config(automl_result, target_column)

            # 评估预期性能
            expected_performance = self._assess_expected_performance(automl_result)

            # 风险评估
            risk_assessment = self._assess_strategy_risk(automl_result, historical_data)

            # 生成元数据
            generation_metadata = {
                'automl_result': automl_result,
                'data_shape': historical_data.shape,
                'feature_columns': list(X.columns),
                'target_column': target_column,
                'generation_method': 'automl_based'
            }

            result = StrategyGenerationResult(
                strategy_config=strategy_config,
                expected_performance=expected_performance,
                risk_assessment=risk_assessment,
                generation_metadata=generation_metadata
            )

            logger.info(f"Strategy generation completed: {strategy_config.strategy_id}")
            return result

        except Exception as e:
            logger.error(f"Strategy generation failed: {e}")
            raise

    def _generate_strategy_config(self, automl_result: AutoMLResult,


                                  target_column: str) -> StrategyConfig:
        """生成策略配置"""
        strategy_id = f"automl_strategy_{int(asyncio.get_event_loop().time())}"

        # 根据最佳模型确定策略类型
        model_name = automl_result.best_model['name']
        if 'RandomForest' in model_name:
            strategy_type = StrategyType.MOMENTUM
        elif 'GradientBoosting' in model_name:
            strategy_type = StrategyType.MEAN_REVERSION
        else:
            strategy_type = StrategyType.MOMENTUM

        # 生成策略参数
        params = {
            'model_type': model_name,
            'model_params': automl_result.best_model['params'],
            'feature_importance': automl_result.feature_importance,
            'expected_accuracy': automl_result.best_model['accuracy'],
            'target_column': target_column
        }

        return StrategyConfig(
            strategy_id=strategy_id,
            strategy_name=f"AutoML Generated Strategy ({model_name})",
            strategy_type=strategy_type,
            parameters=params
        )

    def _assess_expected_performance(self, automl_result: AutoMLResult) -> Dict[str, Any]:
        """评估预期性能"""
        return {
            'accuracy': automl_result.best_model['accuracy'],
            'cross_validation_score': automl_result.best_model['score'],
            'training_time': automl_result.training_time,
            'model_complexity': len(automl_result.best_model['params']),
            'feature_count': len(automl_result.feature_importance),
            'performance_stability': 1.0 - automl_result.performance_metrics['std_score']
        }

    def _assess_strategy_risk(self, automl_result: AutoMLResult,


                              historical_data: pd.DataFrame) -> Dict[str, Any]:
        """评估策略风险"""
        # 计算夏普比率
        returns = historical_data.get('returns', pd.Series(
            np.secrets.normal(0.001, 0.02, len(historical_data))))
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # 年化

        # 计算最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 计算VaR
        var_95 = np.percentile(returns, 5)

        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'value_at_risk_95': var_95,
            'volatility': returns.std() * np.sqrt(252),
            'expected_return': returns.mean() * 252,
            'risk_score': self._calculate_risk_score(sharpe_ratio, max_drawdown, var_95)
        }

    def _calculate_risk_score(self, sharpe_ratio: float, max_drawdown: float,


                              var_95: float) -> float:
        """计算风险评分"""
        # 基于多个风险指标计算综合评分
        sharpe_score = min(max(sharpe_ratio / 2.0, 0), 1)  # 夏普比率标准化
        drawdown_score = 1 - min(abs(max_drawdown), 0.5) * 2  # 回撤评分
        var_score = 1 - min(abs(var_95) * 20, 1)  # VaR评分

        risk_score = (sharpe_score + drawdown_score + var_score) / 3
        return risk_score


class AutoMLPipeline:

    """AutoML管道"""

    def __init__(self, config: AutoMLConfig = None):

        self.config = config or AutoMLConfig()
        self.trainer = AutoMLTrainer(self.config)
        self.generator = StrategyGenerator(self.config)

    async def run_automl_pipeline(self, data: pd.DataFrame,
                                  target_column: str) -> Dict[str, Any]:
        """运行完整的AutoML管道"""
        try:
            logger.info("Starting AutoML pipeline...")

            # 1. 数据验证
            validation_result = self._validate_data(data, target_column)
            if not validation_result['valid']:
                raise ValueError(f"Data validation failed: {validation_result['errors']}")

            # 2. AutoML训练
            automl_result = await asyncio.get_event_loop().run_in_executor(
                None, self.trainer.fit, data.drop(columns=[target_column]), data[target_column]
            )

            # 3. 策略生成
            strategy_result = await self.generator.generate_strategy_from_data(
                data, target_column
            )

            # 4. 结果整合
            pipeline_result = {
                'automl_result': automl_result,
                'strategy_result': strategy_result,
                'data_validation': validation_result,
                'pipeline_metadata': {
                    'execution_time': asyncio.get_event_loop().time(),
                    'data_shape': data.shape,
                    'config': self.config.__dict__
                }
            }

            logger.info("AutoML pipeline completed successfully")
            return pipeline_result

        except Exception as e:
            logger.error(f"AutoML pipeline failed: {e}")
            raise

    def _validate_data(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """验证数据"""
        errors = []

        # 检查目标列是否存在
        if target_column not in data.columns:
            errors.append(f"Target column '{target_column}' not found")

        # 检查数据量
        if len(data) < 100:
            errors.append("Insufficient data: minimum 100 samples required")

        # 检查缺失值
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_ratio > 0.3:
            errors.append(f"Too many missing values: {missing_ratio:.2%}")

        # 检查特征数量
        if data.shape[1] < 3:
            errors.append("Insufficient features: minimum 3 features required")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'data_info': {
                'shape': data.shape,
                'missing_ratio': missing_ratio,
                'dtypes': data.dtypes.to_dict()
            }
        }


# 全局实例
_automl_engine = None
_strategy_generator = None
_automl_pipeline = None


def get_automl_engine(config: AutoMLConfig = None) -> AutoMLTrainer:
    """获取AutoML引擎实例"""
    global _automl_engine
    if _automl_engine is None:
        _automl_engine = AutoMLTrainer(config or AutoMLConfig())
    return _automl_engine


def get_strategy_generator(config: AutoMLConfig = None) -> StrategyGenerator:
    """获取策略生成器实例"""
    global _strategy_generator
    if _strategy_generator is None:
        _strategy_generator = StrategyGenerator(config)
    return _strategy_generator


def get_automl_pipeline(config: AutoMLConfig = None) -> AutoMLPipeline:
    """获取AutoML管道实例"""
    global _automl_pipeline
    if _automl_pipeline is None:
        _automl_pipeline = AutoMLPipeline(config)
    return _automl_pipeline
