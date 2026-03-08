import time
#!/usr/bin/env python3
"""
RQA2025 ML步骤执行器

提供各种ML业务流程步骤的具体执行器实现，
支持模型训练、预测、评估等操作的自动化执行。
"""

import logging
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from abc import ABC, abstractmethod

from .process_orchestrator import ProcessStep
from src.ml.model_manager import ModelManager, ModelType
from .feature_engineering import FeatureEngineer
from .inference_service import InferenceService
from .performance_monitor import record_inference_performance, record_model_performance
from .error_handling import handle_ml_error, InferenceError, TrainingError

try:  # pragma: no cover
    from src.infrastructure.integration import get_models_adapter as _get_models_adapter
except ImportError:  # pragma: no cover
    import logging

    class _FallbackModelsAdapter:
        def get_models_logger(self):
            return logging.getLogger(__name__)

    def _get_models_adapter():
        return _FallbackModelsAdapter()

get_models_adapter = _get_models_adapter

# 获取统一基础设施集成层的模型层适配器
try:
    models_adapter = _get_models_adapter()
    logger = models_adapter.get_models_logger()
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


class BaseMLStepExecutor(ABC):

    """ML步骤执行器基类"""

    def __init__(self):

        self.model_manager = ModelManager()
        self.feature_engineer = FeatureEngineer()
        self.inference_service = InferenceService()

    @abstractmethod
    def execute(self, step: ProcessStep, context: Dict[str, Any]) -> Any:
        """执行步骤"""

    def validate(self, step: ProcessStep) -> bool:
        """验证步骤配置"""
        # 基础验证
        if not step.config:
            logger.warning(f"步骤 {step.step_id} 缺少配置信息")
            return False

        return True

    def get_dependencies(self, step: ProcessStep) -> List[str]:
        """获取步骤依赖"""
        return step.dependencies

    def _get_process_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """获取流程数据"""
        process = context.get('process')
        if process and hasattr(process, 'metadata'):
            return process.metadata
        if isinstance(process, dict):
            return process.setdefault('metadata', {})
        return {}

    def _get_process_id(self, context: Dict[str, Any]) -> Any:
        """获取流程 ID"""
        process = context.get('process')
        if isinstance(process, dict):
            return process.get('process_id')
        if hasattr(process, 'process_id'):
            return getattr(process, 'process_id')
        return None

    def _update_process_metrics(self, context: Dict[str, Any], metrics: Dict[str, Any]):
        """更新流程指标"""
        process = context.get('process')
        if process and hasattr(process, 'metrics'):
            process.metrics.update(metrics)
        elif isinstance(process, dict):
            process.setdefault('metrics', {}).update(metrics)


class DataLoadingExecutor(BaseMLStepExecutor):

    """数据加载执行器"""

    def execute(self, step: ProcessStep, context: Dict[str, Any]) -> Any:
        """执行数据加载"""
        config = step.config

        data_source = config.get('data_source')
        data_format = config.get('data_format', 'csv')
        data_path = config.get('data_path')
        query = config.get('query')
        limit = config.get('limit')

        logger.info(f"开始加载数据: {data_source}")

        try:
            if data_source == 'file':
                if data_format == 'csv':
                    data = pd.read_csv(data_path)
                elif data_format == 'json':
                    data = pd.read_json(data_path, orient='records', typ='frame', lines=None)
                elif data_format == 'parquet':
                    data = pd.read_parquet(data_path)
                else:
                    raise ValueError(f"不支持的数据格式: {data_format}")
            elif data_source == 'database':
                # 这里应该实现数据库查询逻辑
                raise NotImplementedError("数据库数据加载暂未实现")
            else:
                raise ValueError(f"不支持的数据源: {data_source}")

            if limit:
                data = data.head(limit)

            process_data = self._get_process_data(context)
            process_data['raw_data'] = data

            metrics = {
                'data_rows': len(data),
                'data_columns': len(data.columns),
                'data_size_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
            }
            self._update_process_metrics(context, metrics)

            logger.info(f"数据加载完成: {len(data)} 行, {len(data.columns)} 列")
            return data

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def validate(self, step: ProcessStep) -> bool:
        """验证数据加载配置"""
        if not super().validate(step):
            return False

        config = step.config
        required_fields = ['data_source']

        if config.get('data_source') == 'file':
            required_fields.append('data_path')

        for field in required_fields:
            if field not in config:
                logger.error(f"数据加载步骤缺少必需字段: {field}")
                return False

        return True


class FeatureEngineeringExecutor(BaseMLStepExecutor):

    """特征工程执行器"""

    def execute(self, step: ProcessStep, context: Dict[str, Any]) -> Any:
        """执行特征工程"""
        config = step.config
        process_data = self._get_process_data(context)

        # 获取输入数据
        input_data = process_data.get('raw_data')
        if input_data is None:
            raise ValueError("未找到输入数据，请确保数据加载步骤已执行")

        target_column = config.get('target_column')
        feature_types = config.get('feature_types', {})
        scaling_method = config.get('scaling_method', 'standard')
        encoding_method = config.get('encoding_method', 'onehot')

        logger.info("开始特征工程处理")

        try:
            # 分离特征和目标
            if target_column and target_column in input_data.columns:
                X = input_data.drop(columns=[target_column])
                y = input_data[target_column]
            else:
                X = input_data
                y = None

            # 特征工程处理
            processed_X = self.feature_engineer.process_features(
                X,
                feature_types=feature_types,
                scaling_method=scaling_method,
                encoding_method=encoding_method,
            )

            # 更新流程数据
            process_data['features'] = processed_X
            process_data['target'] = y
            process_data['feature_columns'] = list(processed_X.columns)

            # 记录指标
            metrics = {
                'original_features': len(X.columns),
                'processed_features': len(processed_X.columns),
                'feature_engineering_time': time.time(),
            }
            self._update_process_metrics(context, metrics)

            logger.info(f"特征工程完成: {len(X.columns)} -> {len(processed_X.columns)} 个特征")
            return processed_X

        except Exception as e:
            logger.error(f"特征工程失败: {e}")
            raise


class ModelTrainingExecutor(BaseMLStepExecutor):

    """模型训练执行器"""

    def execute(self, step: ProcessStep, context: Dict[str, Any]) -> Any:
        """执行模型训练"""
        config = step.config
        process_data = self._get_process_data(context)

        # 获取训练数据
        X = process_data.get('features')
        y = process_data.get('target')

        if X is None:
            raise ValueError("未找到特征数据，请确保特征工程步骤已执行")

        model_type = config.get('model_type', 'random_forest')
        model_config = config.get('model_config', {})
        validation_split = config.get('validation_split', 0.2)
        cross_validation = config.get('cross_validation', False)
        cv_folds = config.get('cv_folds', 5)

        logger.info(f"开始训练模型: {model_type}")

        try:
            # 创建模型
            model = self.model_manager.create_model(
                model_type=ModelType(model_type),
                config=model_config
            )

            # 训练模型
            if cross_validation:
                results = self.model_manager.train_with_cv(
                    model, X, y, cv_folds=cv_folds
                )
            else:
                results = self.model_manager.train_model(
                    model, X, y, validation_split=validation_split
                )

            model_id = f"{model_type}_{int(datetime.now().timestamp())}"
            self.model_manager.save_model(model, model_id)

            process_data['trained_model'] = model
            process_data['model_id'] = model_id
            process_data['training_results'] = results

            metrics = {
                'model_type': model_type,
                'training_samples': len(X),
                'model_id': model_id,
                'training_score': results.get('score', 0),
                'cross_validation': cross_validation
            }
            if cross_validation:
                metrics['cv_scores'] = results.get('cv_scores', [])

            self._update_process_metrics(context, metrics)

            logger.info(f"模型训练完成: {model_type}, 得分: {results.get('score', 0):.4f}")
            return {
                'model': model,
                'model_id': model_id,
                'results': results
            }

        except Exception as e:
            # 使用统一的错误处理机制
            training_error = TrainingError(
                f"模型训练失败: {e}",
                context={
                    'model_type': model_type,
                    'training_samples': len(X),
                    'process_id': self._get_process_id(context),
                    'step_id': step.step_id
                }
            )
            handle_ml_error(training_error)
            logger.error(f"模型训练失败: {e}")
            raise


class ModelEvaluationExecutor(BaseMLStepExecutor):

    """模型评估执行器"""

    def execute(self, step: ProcessStep, context: Dict[str, Any]) -> Any:
        """执行模型评估"""
        config = step.config
        process_data = self._get_process_data(context)

        # 获取模型和测试数据
        model = process_data.get('trained_model')
        if model is None:
            model_id = process_data.get('model_id')
            if model_id:
                model = self.model_manager.load_model(model_id)
            else:
                raise ValueError("未找到训练好的模型")

        test_data = process_data.get('test_data')
        if test_data is None:
            # 如果没有单独的测试数据，使用训练数据的一部分
            features = process_data.get('features')
            target = process_data.get('target')
            if features is not None and target is not None:
                from sklearn.model_selection import train_test_split
                _, test_data_X, _, test_data_y = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )
                test_data = (test_data_X, test_data_y)

        if test_data is None:
            raise ValueError("未找到测试数据")

        evaluation_metrics = config.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
        confusion_matrix = config.get('confusion_matrix', True)
        classification_report = config.get('classification_report', True)

        logger.info("开始模型评估")

        try:
            # 执行评估
            X_test, y_test = test_data
            predictions = self.model_manager.predict(model, X_test)

            results = self.model_manager.evaluate_model(
                model, X_test, y_test,
                metrics=evaluation_metrics,
                confusion_matrix=confusion_matrix,

                classification_report=classification_report
            )

            # 记录模型性能指标
            eval_metrics = results.get('metrics', {})
            accuracy = eval_metrics.get('accuracy', 0)
            precision = eval_metrics.get('precision', 0)
            recall = eval_metrics.get('recall', 0)
            f1_score = eval_metrics.get('f1_score', 0)

            model_id = process_data.get('model_id', '')
            record_model_performance(accuracy, precision, recall, f1_score, model_id)

            # 更新流程数据
            process_data['evaluation_results'] = results
            process_data['predictions'] = predictions

            # 记录指标
            metrics = {
                'evaluation_samples': len(X_test),
                'evaluation_metrics': evaluation_metrics,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
            metrics.update(eval_metrics)
            self._update_process_metrics(context, metrics)

            logger.info(f"模型评估完成，主要指标: {eval_metrics}")
            return results

        except Exception as e:
            logger.error(f"模型评估失败: {e}")
            raise


class ModelPredictionExecutor(BaseMLStepExecutor):

    """模型预测执行器"""

    def execute(self, step: ProcessStep, context: Dict[str, Any]) -> Any:
        """执行模型预测"""
        config = step.config
        process_data = self._get_process_data(context)

        # 获取模型
        model = process_data.get('trained_model')
        if model is None:
            model_id = process_data.get('model_id')
            if model_id:
                model = self.model_manager.load_model(model_id)
            else:
                raise ValueError("未找到模型")

        prediction_data = process_data.get('prediction_data')
        if prediction_data is None:
            features = process_data.get('features')
            if features is not None:
                prediction_data = features
            else:
                raise ValueError("未找到预测数据")

        prediction_method = config.get('prediction_method', 'predict')
        batch_size = config.get('batch_size', 1000)
        output_format = config.get('output_format', 'dataframe')

        predictions = []

        logger.info(f"开始模型预测: {prediction_method}")

        try:
            start_time = datetime.now()

            # 执行预测
            if len(prediction_data) > batch_size:
                for i in range(0, len(prediction_data), batch_size):
                    batch = prediction_data.iloc[i:i + batch_size]
                    batch_predictions = self.model_manager.predict(
                        model, batch, method=prediction_method
                    )
                    predictions.extend(batch_predictions)
            else:
                predictions = self.model_manager.predict(
                    model, prediction_data, method=prediction_method
                )

            # 计算执行时间
            execution_time = (datetime.now() - start_time).total_seconds() * 1000  # 毫秒

            # 记录推理性能
            model_id = process_data.get('model_id', '')
            record_inference_performance(execution_time / len(prediction_data), model_id)

            # 格式化输出
            if output_format == 'dataframe':
                result = pd.DataFrame({
                    'predictions': predictions
                })
            elif output_format == 'dict':
                result = {'predictions': predictions}
            else:
                result = predictions

            # 更新流程数据
            process_data['predictions'] = result

            # 记录指标
            metrics = {
                'prediction_samples': len(prediction_data),
                'prediction_method': prediction_method,
                'batch_size': batch_size,
                'execution_time_ms': execution_time,
                'avg_latency_ms': execution_time / len(prediction_data)
            }
            self._update_process_metrics(context, metrics)

            logger.info(
                f"模型预测完成: {len(prediction_data)} 个样本, 平均延迟: {execution_time / len(prediction_data):.2f}ms")
            return result

        except Exception as e:
            # 记录推理错误
            model_id = process_data.get('model_id', '')
            record_inference_performance(0, model_id, str(e))

            # 使用统一的错误处理机制
            inference_error = InferenceError(
                f"模型预测失败: {e}",
                context={
                    'model_id': model_id,
                    'prediction_samples': len(prediction_data),
                    'process_id': self._get_process_id(context),
                    'step_id': step.step_id
                }
            )
            handle_ml_error(inference_error)

            logger.error(f"模型预测失败: {e}")
            raise


class ModelDeploymentExecutor(BaseMLStepExecutor):

    """模型部署执行器"""

    def execute(self, step: ProcessStep, context: Dict[str, Any]) -> Any:
        """执行模型部署"""
        config = step.config
        process_data = self._get_process_data(context)

        # 获取模型
        model = process_data.get('trained_model')
        model_id = process_data.get('model_id')

        if model is None and model_id:
            model = self.model_manager.load_model(model_id)

        if model is None:
            raise ValueError("未找到要部署的模型")

        deployment_target = config.get('deployment_target', 'local')
        service_name = config.get('service_name', f"ml_service_{model_id}")
        enable_monitoring = config.get('enable_monitoring', True)
        enable_logging = config.get('enable_logging', True)

        logger.info(f"开始模型部署: {service_name}")

        try:
            # 部署模型
            deployment_result = self.inference_service.deploy_model(
                model=model,
                model_id=model_id,
                service_name=service_name,
                target=deployment_target,
                enable_monitoring=enable_monitoring,
                enable_logging=enable_logging
            )

            # 更新流程数据
            process_data['deployment_result'] = deployment_result
            process_data['service_name'] = service_name

            # 记录指标
            metrics = {
                'deployment_target': deployment_target,
                'service_name': service_name,
                'deployment_time': datetime.now().isoformat(),
                'model_id': model_id
            }
            self._update_process_metrics(context, metrics)

            logger.info(f"模型部署完成: {service_name}")
            return deployment_result

        except Exception as e:
            logger.error(f"模型部署失败: {e}")
            raise


class HyperparameterTuningExecutor(BaseMLStepExecutor):

    """超参数调优执行器"""

    def execute(self, step: ProcessStep, context: Dict[str, Any]) -> Any:
        """执行超参数调优"""
        config = step.config
        process_data = self._get_process_data(context)

        # 获取训练数据
        X = process_data.get('features')
        y = process_data.get('target')

        if X is None or y is None:
            raise ValueError("未找到训练数据")

        model_type = config.get('model_type', 'random_forest')
        param_space = config.get('param_space', {})
        tuning_method = config.get('tuning_method', 'grid')
        cv_folds = config.get('cv_folds', 5)
        max_evals = config.get('max_evals', 50)
        scoring_metric = config.get('scoring_metric', 'accuracy')

        logger.info(f"开始超参数调优: {model_type} ({tuning_method})")

        try:
            # 执行调优
            best_params, best_score, tuning_results = self.model_manager.tune_hyperparameters(
                model_type=ModelType(model_type),
                X=X,
                y=y,
                param_space=param_space,
                method=tuning_method,
                cv_folds=cv_folds,
                max_evals=max_evals,
                scoring=scoring_metric
            )

            # 使用最佳参数重新训练模型
            best_model = self.model_manager.create_model(
                model_type=ModelType(model_type),
                config=best_params
            )

            final_results = self.model_manager.train_model(best_model, X, y)

            # 保存最佳模型
            best_model_id = f"{model_type}_tuned_{int(datetime.now().timestamp())}"
            self.model_manager.save_model(best_model, best_model_id)

            # 更新流程数据
            process_data['tuned_model'] = best_model
            process_data['best_params'] = best_params
            process_data['tuning_results'] = tuning_results
            process_data['best_model_id'] = best_model_id

            # 记录指标
            metrics = {
                'model_type': model_type,
                'tuning_method': tuning_method,
                'best_score': best_score,
                'tuning_evaluations': len(tuning_results),
                'best_model_id': best_model_id
            }
            self._update_process_metrics(context, metrics)

            logger.info(f"超参数调优完成，最佳得分: {best_score:.4f}")
            return {
                'best_model': best_model,
                'best_params': best_params,
                'best_score': best_score,
                'tuning_results': tuning_results
            }

        except Exception as e:
            logger.error(f"超参数调优失败: {e}")
            raise

            # 注册执行器到全局注册表


def register_ml_step_executors(orchestrator):
    """注册ML步骤执行器"""
    executors = {
        'data_loading': DataLoadingExecutor(),
        'feature_engineering': FeatureEngineeringExecutor(),
        'model_training': ModelTrainingExecutor(),
        'model_evaluation': ModelEvaluationExecutor(),
        'model_prediction': ModelPredictionExecutor(),
        'model_deployment': ModelDeploymentExecutor(),
        'hyperparameter_tuning': HyperparameterTuningExecutor()
    }

    for step_type, executor in executors.items():
        orchestrator.register_step_executor(step_type, executor)
        logger.info(f"已注册ML步骤执行器: {step_type}")


# 便捷函数
__all__ = [
    # 执行器类
    'DataLoadingExecutor',
    'FeatureEngineeringExecutor',
    'ModelTrainingExecutor',
    'ModelEvaluationExecutor',
    'ModelPredictionExecutor',
    'ModelDeploymentExecutor',
    'HyperparameterTuningExecutor',

    # 注册函数
    'register_ml_step_executors'
]
