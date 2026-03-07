#!/usr/bin/env python3
"""
RQA2025机器学习核心
提供机器学习模型的核心功能
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Any as AnyType
import numpy as np
import pandas as pd
from datetime import datetime

import os

from .exceptions import MLException, ModelNotFoundError, DataValidationError

_FORCE_CORE_FALLBACK = os.environ.get("ML_CORE_FORCE_FALLBACK") == "1"

try:  # pragma: no cover - 主要用于降级场景
    if _FORCE_CORE_FALLBACK:
        raise ImportError("Forced MLCore fallback for testing")
    from src.core.integration import get_models_adapter as _get_models_adapter
except ImportError:  # pragma: no cover
    class _FallbackModelsAdapter:
        def get_models_cache_manager(self):
            return None

        def get_models_config_manager(self):
            return None

        def get_models_logger(self):
            return logging.getLogger(__name__)

    def _get_models_adapter():
        return _FallbackModelsAdapter()

get_models_adapter = _get_models_adapter

from .constants import *
from .exceptions import *

# 获取统一基础设施集成层的模型层适配器
try:
    models_adapter = _get_models_adapter()
    cache_manager = models_adapter.get_models_cache_manager()
    config_manager = models_adapter.get_models_config_manager()
    logger = models_adapter.get_models_logger()
except Exception:
    logger = logging.getLogger(__name__)
    cache_manager = None
    config_manager = None


class MLCore:

    """机器学习核心类（符合架构设计：使用BusinessProcessOrchestrator和统一适配器）"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化ML核心

        Args:
        config: 配置参数
        """
        self.config = dict(config or {})
        self.models = {}
        self.feature_processors = {}

        # 初始化业务流程编排器（符合架构设计：业务流程编排）
        try:
            from src.core.orchestration.orchestrator_refactored import BusinessProcessOrchestrator
            self.orchestrator = BusinessProcessOrchestrator()
            logger.info("ML业务流程编排器已初始化")
        except Exception as e:
            logger.warning(f"业务流程编排器初始化失败: {e}")
            self.orchestrator = None

        # 优先使用统一适配器工厂（符合架构设计：统一基础设施集成）
        try:
            from src.core.integration.business_adapters import get_unified_adapter_factory
            from src.core.integration.unified_business_adapters import BusinessLayerType
            adapter_factory = get_unified_adapter_factory()
            if adapter_factory:
                # 通过统一适配器工厂获取ML层适配器
                self.ml_adapter = adapter_factory.get_adapter(BusinessLayerType.ML)
                if self.ml_adapter:
                    self.cache_manager = self.ml_adapter.get_cache_manager() if hasattr(self.ml_adapter, 'get_cache_manager') else None
                    self.config_manager = self.ml_adapter.get_config_manager() if hasattr(self.ml_adapter, 'get_config_manager') else None
                    self.logger = self.ml_adapter.get_logger() if hasattr(self.ml_adapter, 'get_logger') else logging.getLogger(__name__)
                    logger.info("使用统一适配器工厂访问ML层服务")
                else:
                    raise Exception("无法获取ML层适配器")
            else:
                raise Exception("统一适配器工厂不可用")
        except Exception as e:
            logger.debug(f"统一适配器工厂不可用，降级到get_models_adapter: {e}")
            # 降级方案：使用get_models_adapter
            try:
                if os.environ.get("ML_CORE_FORCE_FALLBACK") == "1":
                    raise RuntimeError("Forced MLCore fallback at init")

                self.models_adapter = _get_models_adapter()
                self.cache_manager = self.models_adapter.get_models_cache_manager()
                self.config_manager = self.models_adapter.get_models_config_manager()
                self.logger = self.models_adapter.get_models_logger()
            except Exception:
                self._apply_default_services()
            else:
                if self.cache_manager is None and self.config_manager is None:
                    self._apply_default_services()

        if "model_cache_enabled" not in self.config:
            self._apply_default_services()

        self.logger.info("初始化ML核心")

    def _apply_default_services(self):
        """应用降级配置和默认依赖"""
        self.logger = logging.getLogger(__name__)
        self.cache_manager = None
        self.config_manager = None

        self.default_config = {
            'model_cache_enabled': True,
            'model_cache_ttl': CACHE_TTL_SECONDS,
            'feature_preprocessing': True,
            'auto_feature_selection': False,
            'cross_validation_folds': DEFAULT_CV_FOLDS,
            'test_size': DEFAULT_TEST_SIZE,
            'random_state': DEFAULT_RANDOM_STATE,
            'gpu_enabled': False,
            'batch_size': DEFAULT_BATCH_SIZE,
            'epochs': DEFAULT_EPOCHS
        }
        self.default_config.update(self.config)
        self.config = self.default_config

    @handle_ml_exception
    def train_model(self, X: Union[pd.DataFrame, np.ndarray],
                    y: Union[pd.Series, np.ndarray],
                    model_type: str = 'linear',
                    model_params: Optional[Dict[str, Any]] = None,
                    feature_names: Optional[List[str]] = None) -> str:
        """
        训练模型（符合架构设计：使用BusinessProcessOrchestrator管理训练流程）

        Args:
            X: 特征数据
            y: 目标变量
            model_type: 模型类型 ('linear', 'rf', 'xgb', 'lstm')
            model_params: 模型参数
            feature_names: 特征名称

        Returns:
            模型ID

        Raises:
            DataValidationError: 数据验证失败
            ModelTrainingError: 模型训练失败
        """
        # 如果业务流程编排器可用，启动训练流程（符合架构设计）
        process_id = None
        if self.orchestrator:
            try:
                # 使用MLProcessOrchestrator管理训练流程（符合架构设计）
                from src.ml.core.process_orchestrator import get_ml_process_orchestrator, MLProcessType, ProcessPriority, MLProcess
                import time
                ml_orchestrator = get_ml_process_orchestrator()
                if ml_orchestrator:
                    # 创建ML流程实例
                    process = MLProcess(
                        process_id=f"model_training_{int(time.time() * 1000)}",
                        process_type=MLProcessType.MODEL_TRAINING,
                        process_name=f"Model Training: {model_type}",
                        config={
                            "model_type": model_type,
                            "model_params": model_params,
                            "feature_count": X.shape[1] if hasattr(X, 'shape') else len(X[0]) if X else 0
                        },
                        priority=ProcessPriority.NORMAL
                    )
                    process_id = ml_orchestrator.submit_process(process)
                    self.logger.info(f"模型训练流程已启动: {process_id}")
            except Exception as e:
                self.logger.debug(f"启动训练流程失败: {e}")
        
        # 数据验证
        validate_data_shape(X, y)

        self.logger.info(f"开始训练 {model_type} 模型")

        try:
            # 数据预处理
            X_processed, feature_names = self._prepare_features(X, feature_names)
            y_processed = self._prepare_target(y)

            # 创建和训练模型
            model = self._create_model(model_type, model_params)
            model.fit(X_processed, y_processed)

            # 保存模型
            model_id = self._save_trained_model(model, model_type, feature_names, model_params)

            # 如果业务流程编排器可用，完成训练流程（符合架构设计）
            if self.orchestrator and process_id:
                try:
                    from src.ml.core.process_orchestrator import get_ml_process_orchestrator, ProcessStatus
                    ml_orchestrator = get_ml_process_orchestrator()
                    if ml_orchestrator:
                        # 流程状态由编排器自动管理，这里只记录日志
                        process_status = ml_orchestrator.get_process_status(process_id)
                        if process_status:
                            self.logger.info(f"模型训练流程已完成: {process_id}, 状态: {process_status.get('status')}")
                except Exception as e:
                    self.logger.debug(f"完成训练流程失败: {e}")

            self.logger.info(f"模型训练完成，ID: {model_id}")
            return model_id

        except Exception as e:
            error_msg = f"模型训练失败: {str(e)}"
            self.logger.error(error_msg)
            
            # 如果业务流程编排器可用，标记流程失败（符合架构设计）
            if self.orchestrator and process_id:
                try:
                    from src.ml.core.process_orchestrator import get_ml_process_orchestrator
                    ml_orchestrator = get_ml_process_orchestrator()
                    if ml_orchestrator:
                        # 流程状态由编排器自动管理，这里只记录日志
                        self.logger.warning(f"模型训练流程失败: {process_id}")
                except Exception:
                    pass
            
            raise ModelTrainingError(error_msg, model_type) from e

    def _prepare_features(self, X: Union[pd.DataFrame, np.ndarray],
                          feature_names: Optional[List[str]]) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[List[str]]]:
        """准备特征数据"""
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_processed = self._preprocess_features(X, feature_names)
        else:
            X_processed = X
        return X_processed, feature_names

    def _prepare_target(self, y: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """准备目标变量"""
        if isinstance(y, pd.Series):
            return y.values
        return y

    def _save_trained_model(self, model: AnyType, model_type: str,
                            feature_names: Optional[List[str]],
                            model_params: Optional[Dict[str, Any]]) -> str:
        """保存训练好的模型"""
        model_id = self._generate_model_id()
        self.models[model_id] = {
            'model': model,
            'model_type': model_type,
            'feature_names': feature_names,
            'created_at': datetime.now(),
            'model_params': model_params or {}
        }
        return model_id

    def predict(self, model_id: str, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        模型预测

        Args:
            model_id: 模型ID
            X: 特征数据

        Returns:
            预测结果
        """
        if model_id not in self.models:
            raise ModelNotFoundError(model_id)

        model_info = self.models[model_id]
        model = model_info['model']

        try:
            if isinstance(X, pd.DataFrame):
                feature_names = model_info.get('feature_names')
                X_processed = self._preprocess_features(X, feature_names)
            else:
                X_processed = X

            predictions = model.predict(X_processed)
            self.logger.info(f"模型 {model_id} 预测完成")
            return predictions
        except Exception as e:
            self.logger.error(f"模型预测失败: {e}")
            raise

    def evaluate_model(self, model_id: str, X: Union[pd.DataFrame, np.ndarray],


                       y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        评估模型性能

        Args:
        model_id: 模型ID
        X: 特征数据
        y: 目标变量

        Returns:
        评估指标
        """
        try:
            predictions = self.predict(model_id, X)

            y_true = y.values if isinstance(y, pd.Series) else y
            metrics = self._calculate_metrics(y_true, predictions)

            self.logger.info(f"模型 {model_id} 评估完成: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"模型评估失败: {e}")
            raise

    def _create_model(self, model_type: str, model_params: Optional[Dict[str, Any]] = None):
        """创建模型实例"""
        try:
            if model_params is None:
                model_params = {}

            if model_type == 'linear':
                from sklearn.linear_model import LinearRegression
                return LinearRegression(**model_params)

            elif model_type == 'rf':
                from sklearn.ensemble import RandomForestRegressor

                default_params = {'n_estimators': 100, 'random_state': self.config['random_state']}

                default_params.update(model_params)
                return RandomForestRegressor(**default_params)

            elif model_type == 'xgb':
                try:
                    import xgboost as xgb

                    default_params = {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'random_state': self.config['random_state']
                    }

                    default_params.update(model_params)
                    return xgb.XGBRegressor(**default_params)
                except ImportError:
                    self.logger.warning("XGBoost未安装，使用随机森林作为替代")
                    from sklearn.ensemble import RandomForestRegressor
                    return RandomForestRegressor(**model_params)

            elif model_type == 'lstm':
                # 这里可以实现LSTM模型
                # 目前返回一个简单的占位符
                return self._create_simple_neural_network(model_params)

            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

        except Exception as e:
            self.logger.error(f"创建模型失败 {model_type}: {e}")
            raise

    def _create_simple_neural_network(self, model_params: Dict[str, Any]):
        """创建简单的神经网络模型"""
        try:
            from sklearn.neural_network import MLPRegressor

            default_params = {
                'hidden_layer_sizes': (64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'random_state': self.config['random_state'],
                'max_iter': self.config['epochs']
            }

            default_params.update(model_params)
            return MLPRegressor(**default_params)
        except ImportError:
            self.logger.warning("MLPRegressor不可用，使用线性回归作为替代")
            from sklearn.linear_model import LinearRegression
            return LinearRegression()

    def _preprocess_features(self, X: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        """特征预处理"""
        try:
            # 确保特征顺序一致
            if feature_names:
                X = X[feature_names]

                # 处理缺失值
                X = X.fillna(X.mean())

                # 转换为numpy数组
                return X.values

        except Exception as e:
            self.logger.error(f"特征预处理失败: {e}")
            return X.values

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标

        优化版本：避免重复计算，提高性能
        """
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            # 避免重复计算MSE
            mse = mean_squared_error(y_true, y_pred)

            metrics = {
                'mse': mse,
                'rmse': np.sqrt(mse),  # 直接使用已计算的MSE
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }

            return metrics

        except Exception as e:
            self.logger.error(f"计算评估指标失败: {e}")
            return {}

    def _generate_model_id(self) -> str:
        """生成模型ID"""
        import uuid
        return f"model_{uuid.uuid4().hex[:8]}"

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        return self.models.get(model_id)

    def list_models(self) -> List[str]:
        """列出所有模型"""
        return list(self.models.keys())

    def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        try:
            if model_id in self.models:
                del self.models[model_id]
                self.logger.info(f"模型 {model_id} 已删除")
                return True
            return False
        except Exception as e:
            self.logger.error(f"删除模型失败 {model_id}: {e}")
            return False

    def save_model(self, model_id: str, filepath: str) -> bool:
        """保存模型"""
        try:
            import joblib

            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")

            model_info = self.models[model_id]
            joblib.dump({"model_id": model_id, **model_info}, filepath)

            self.logger.info(f"模型 {model_id} 已保存到 {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"保存模型失败 {model_id}: {e}")
            return False

    def load_model(self, filepath: str) -> Optional[str]:
        """加载模型"""
        try:
            import joblib

            model_info = joblib.load(filepath)
            model_id = model_info.get('model_id', self._generate_model_id())

            self.models[model_id] = model_info
            self.logger.info(f"模型已从 {filepath} 加载，ID: {model_id}")
            return model_id

        except Exception as e:
            self.logger.error(f"加载模型失败 {filepath}: {e}")
            return None

    def create_feature_processor(self, processor_type: str = 'standard',


                                 config: Optional[Dict[str, Any]] = None) -> str:
        """
        创建特征处理器

        Args:
        processor_type: 处理器类型 ('standard', 'robust', 'minmax')
        config: 处理器配置

        Returns:
        处理器ID
        """
        try:
            from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

            if config is None:
                config = {}

            if processor_type == 'standard':
                processor = StandardScaler(**config)
            elif processor_type == 'robust':
                processor = RobustScaler(**config)
            elif processor_type == 'minmax':
                processor = MinMaxScaler(**config)
            else:
                raise ValueError(f"不支持的处理器类型: {processor_type}")

            processor_id = f"processor_{processor_type}_{self._generate_model_id()}"

            self.feature_processors[processor_id] = {
                'processor': processor,
                'type': processor_type,
                'config': config,
                'created_at': datetime.now()
            }

            self.logger.info(f"特征处理器创建成功: {processor_id}")
            return processor_id

        except Exception as e:
            self.logger.error(f"创建特征处理器失败: {e}")
            raise

    def fit_feature_processor(self, processor_id: str, X: Union[pd.DataFrame, np.ndarray]):
        """拟合特征处理器"""
        try:
            if processor_id not in self.feature_processors:
                raise ValueError(f"特征处理器 {processor_id} 不存在")

            processor_info = self.feature_processors[processor_id]
            processor = processor_info['processor']

            X_fit = X.values if isinstance(X, pd.DataFrame) else X
            processor.fit(X_fit)
            self.logger.info(f"特征处理器 {processor_id} 拟合完成")

        except Exception as e:
            self.logger.error(f"拟合特征处理器失败: {e}")
            raise

    def transform_features(self, processor_id: str, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """转换特征"""
        try:
            if processor_id not in self.feature_processors:
                raise ValueError(f"特征处理器 {processor_id} 不存在")

            processor_info = self.feature_processors[processor_id]
            processor = processor_info['processor']

            X_transform = X.values if isinstance(X, pd.DataFrame) else X
            X_transformed = processor.transform(X_transform)
            self.logger.info(f"特征转换完成，形状: {X_transformed.shape}")
            return X_transformed

        except Exception as e:
            self.logger.error(f"特征转换失败: {e}")
            raise

    def get_feature_importance(self, model_id: str) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        try:
            if model_id not in self.models:
                return None

            model_info = self.models[model_id]
            model = model_info['model']
            feature_names = model_info.get('feature_names', [])

            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_values = np.abs(model.coef_)
            else:
                return None

            if feature_names and len(feature_names) == len(importance_values):
                importance_dict = dict(zip(feature_names, importance_values))
            else:
                importance_dict = {f'feature_{i}': val for i, val in enumerate(importance_values)}

            return importance_dict

        except Exception as e:
            self.logger.error(f"获取特征重要性失败: {e}")
            return None

    def process_data(self, data: Any) -> Dict[str, Any]:
        """
        处理输入数据

        Args:
            data: 输入数据

        Returns:
            处理后的数据字典

        Raises:
            ValueError: 当输入数据无效时
        """
        try:
            if data is None:
                raise ValueError("输入数据不能为None")

            if isinstance(data, list) and len(data) == 0:
                return {}

            # 这里可以添加数据预处理逻辑
            # 目前返回基本的处理结果
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return {"processed_data": data, "count": len(data)}
            else:
                return {"processed_data": data}

        except Exception as e:
            self.logger.error(f"数据处理失败: {e}")
            raise ValueError(f"数据处理失败: {str(e)}")

    def cross_validate(self, X: Union[pd.DataFrame, np.ndarray],


                       y: Union[pd.Series, np.ndarray],
                       model_type: str = 'linear',
                       model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        交叉验证

        Args:
        X: 特征数据
        y: 目标变量
        model_type: 模型类型
        model_params: 模型参数

        Returns:
        交叉验证结果
        """
        try:
            from sklearn.model_selection import cross_val_score

            # 检查数据集是否为空
            if len(X) == 0 or len(y) == 0:
                from .exceptions import MLException
                raise MLException("数据集为空，无法进行交叉验证", model_type)

            if isinstance(X, pd.DataFrame):
                X_processed = X.values
            else:
                X_processed = X

            if isinstance(y, pd.Series):
                y_processed = y.values
            else:
                y_processed = y

            model = self._create_model(model_type, model_params)

            # 执行交叉验证
            cv_scores = cross_val_score(
                model, X_processed, y_processed,
                cv=self.config['cross_validation_folds'],
                scoring='neg_mean_squared_error'
            )

            cv_results = {
                'mean_score': -cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': -cv_scores,
                'folds': self.config['cross_validation_folds']
            }

            self.logger.info(f"交叉验证完成: 平均分数={cv_results['mean_score']:.4f}")
            return cv_results

        except Exception as e:
            self.logger.error(f"交叉验证失败: {e}")
            raise
