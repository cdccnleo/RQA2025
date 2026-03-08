#!/usr/bin/env python3
"""
精简版 ML 核心服务，提供基础的启动、推理与模型管理能力。
"""

from __future__ import annotations

import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

try:  # pragma: no cover
    from src.core.foundation.interfaces.ml_strategy_interfaces import (  # type: ignore
        IMLService,
        IMLFeatureEngineering,
        MLFeatures,
        MLInferenceRequest,
        MLInferenceResponse,
    )
except ImportError:  # pragma: no cover
    @dataclass
    class MLFeatures:
        data: Dict[str, Any]

    @dataclass
    class MLInferenceRequest:
        model_id: str
        features: MLFeatures
        inference_type: str = "sync"

    @dataclass
    class MLInferenceResponse:
        request_id: str
        success: bool
        prediction: Optional[Any] = None
        confidence: Optional[float] = None
        processing_time_ms: float = 0.0
        error_message: Optional[str] = None
        metadata: Optional[Dict[str, Any]] = None

    class IMLFeatureEngineering:
        def process(self, features: MLFeatures) -> MLFeatures:
            return features

    class IMLService:
        pass


try:  # pragma: no cover
    from src.infrastructure.integration import get_models_adapter as _get_models_adapter
except ImportError:  # pragma: no cover
    class _FallbackModelsAdapter:
        def get_models_logger(self):
            return logging.getLogger(__name__)

        def get_models_cache_manager(self):
            return None

        def get_models_config_manager(self):
            return None

    def _get_models_adapter():
        return _FallbackModelsAdapter()


get_models_adapter = _get_models_adapter


class MLServiceStatus(Enum):
    """服务状态枚举"""

    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


class _DefaultFeatureEngineering(IMLFeatureEngineering):
    def extract_features(self, raw_data, feature_config: Dict[str, Any]) -> MLFeatures:
        """默认特征提取实现"""
        # 简单地将DataFrame转换为MLFeatures格式
        features_dict = {}
        if hasattr(raw_data, 'iloc') and len(raw_data) > 0:
            # 如果是DataFrame，取第一行作为特征
            row = raw_data.iloc[0]
            features_dict = {col: float(val) if isinstance(val, (int, float)) else 0.0
                           for col, val in row.items()}

        return MLFeatures(
            timestamp=datetime.now(),
            symbol="default",
            features=features_dict
        )

    def preprocess_features(self, features: MLFeatures,
                           preprocessing_config: Dict[str, Any]) -> MLFeatures:
        """默认特征预处理实现"""
        # 默认不进行任何预处理，直接返回
        return features

    def select_features(self, features: MLFeatures,
                       selection_config: Dict[str, Any]) -> MLFeatures:
        """默认特征选择实现"""
        # 默认选择所有特征
        return features

    def process(self, features: MLFeatures) -> MLFeatures:
        """兼容性方法"""
        return features


class _DefaultModelManager:
    def list_models(self):
        return []

    def get_model_info(self, model_id: str):
        return None


class _DefaultInferenceService:
    def start(self):
        return True

    async def start_async(self):
        return True

    def stop(self):
        return True

    def predict(self, data, mode=None):
        raise RuntimeError("Inference service not configured")


class MLService(IMLService):
    """轻量级 ML 核心服务实现"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.status = MLServiceStatus.STOPPED

        try:  # pragma: no cover
            adapter = get_models_adapter()
            self.logger = adapter.get_models_logger()
        except Exception:  # pragma: no cover
            self.logger = logging.getLogger(__name__)

        self.max_workers = self.config.get("max_workers", 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.inference_service = self.config.get("inference_service") or _DefaultInferenceService()
        self.model_manager = self.config.get("model_manager") or _DefaultModelManager()
        self.feature_engineering = self.config.get("feature_engineering") or _DefaultFeatureEngineering()

        # 初始化模型存储
        self._models = {}  # model_id -> model_instance
        self._model_configs = {}  # model_id -> config
        self._model_performance = {}  # model_id -> performance_metrics

        # 初始化模型缓存
        try:
            from src.data.cache.hot_data_cache import get_hot_data_cache
            self._model_cache = get_hot_data_cache('model_cache')
            self._enable_model_cache = True
        except Exception as e:
            self.logger.warning(f"Failed to initialize model cache: {e}")
            self._model_cache = None
            self._enable_model_cache = False

        self.stats = {
            "inference_requests": 0,
            "inference_success": 0,
            "inference_failed": 0,
            "model_loads": 0,
            "model_unloads": 0,
            "training_sessions": 0,
            "batch_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    # ------------------------------------------------------------------ #
    # 生命周期管理
    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        if self.status == MLServiceStatus.RUNNING:
            return True

        try:
            if hasattr(self.inference_service, "start"):
                self.inference_service.start()
            self.status = MLServiceStatus.RUNNING
            return True
        except Exception as exc:  # pragma: no cover
            self.logger.error("启动 MLService 失败: %s", exc)
            self.status = MLServiceStatus.ERROR
            return False

    async def start_async(self) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.start)

    def stop(self) -> None:
        if self.status != MLServiceStatus.RUNNING:
            return
        if hasattr(self.inference_service, "stop"):
            self.inference_service.stop()
        # 停止模型缓存
        if self._model_cache:
            try:
                self._model_cache.stop()
            except Exception as e:
                self.logger.warning(f"Failed to stop model cache: {e}")
        self.status = MLServiceStatus.STOPPED

    def _generate_cache_key(self, model_id: str, data: Any) -> str:
        """生成缓存键"""
        import hashlib
        import pickle
        
        try:
            # 尝试序列化数据
            if hasattr(data, 'iloc'):  # DataFrame
                # 对于DataFrame，只使用前几行和列名作为缓存键
                if len(data) > 5:
                    sample_data = data.head(5)
                else:
                    sample_data = data
                data_hash = hashlib.md5(pickle.dumps((list(sample_data.columns), sample_data.values))).hexdigest()
            elif isinstance(data, (np.ndarray, list, dict)):
                # 对于数组、列表和字典，直接序列化
                data_hash = hashlib.md5(pickle.dumps(data)).hexdigest()
            else:
                # 对于其他类型，使用字符串表示
                data_hash = hashlib.md5(str(data).encode()).hexdigest()
            
            return f"model_{model_id}_{data_hash}"
        except Exception:
            # 如果序列化失败，使用时间戳生成唯一键
            import time
            return f"model_{model_id}_timestamp_{int(time.time() * 1000)}"

    # ------------------------------------------------------------------ #
    # 服务信息
    # ------------------------------------------------------------------ #
    def get_service_info(self) -> Dict[str, Any]:
        models = self.list_models()
        return {
            "status": self.status.value,
            "config": self.config,
            "models": {
                "total_models": len(models),
                "items": models,
            },
            "stats": self.stats.copy(),
        }

    # ------------------------------------------------------------------ #
    # 推理相关
    # ------------------------------------------------------------------ #
    def predict(self, data: Any, mode: Optional[str] = None) -> Any:
        if self.status != MLServiceStatus.RUNNING:
            raise RuntimeError("MLService 未启动")

        self.stats["inference_requests"] += 1
        try:
            # 如果有已加载的模型，使用模型进行预测
            if self._models:
                model_id = list(self._models.keys())[0]  # 使用第一个模型
                
                # 尝试从缓存获取预测结果
                cache_key = self._generate_cache_key(model_id, data)
                if self._enable_model_cache and self._model_cache:
                    cached_result = self._model_cache.get(cache_key)
                    if cached_result is not None:
                        self.stats["cache_hits"] += 1
                        self.stats["inference_success"] += 1
                        return cached_result
                    else:
                        self.stats["cache_misses"] += 1

                model = self._models[model_id]

                # 将DataFrame转换为特征格式
                n_samples = 1
                if hasattr(data, 'iloc'):  # DataFrame
                    n_samples = len(data)
                    if n_samples == 1:
                        # 单样本预测
                        feature_dict = {}
                        for col in data.columns:
                            # 处理字符串和非字符串列名
                            col_name = str(col).lower() if hasattr(col, 'lower') else str(col)
                            if col_name not in ['target', 'y', 'label']:  # 排除目标列
                                feature_dict[col] = float(data[col].iloc[0])
                        X = np.array(list(feature_dict.values())).reshape(1, -1)
                    else:
                        # 多样本预测
                        feature_list = []
                        for col in data.columns:
                            col_name = str(col).lower() if hasattr(col, 'lower') else str(col)
                            if col_name not in ['target', 'y', 'label']:  # 排除目标列
                                feature_list.append(data[col].values)
                        X = np.column_stack(feature_list)
                else:
                    X = np.array(data).reshape(1, -1) if not isinstance(data, np.ndarray) else data

                prediction = model.predict(X)
                
                # 将预测结果缓存
                if self._enable_model_cache and self._model_cache:
                    try:
                        self._model_cache.set(cache_key, prediction, ttl=3600)  # 缓存1小时
                    except Exception as cache_exc:
                        self.logger.warning(f"Failed to cache prediction: {cache_exc}")

                self.stats["inference_success"] += 1
                # 如果是单样本预测，返回标量；如果是多样本预测，返回数组
                if n_samples == 1 and hasattr(prediction, '__len__') and len(prediction) == 1:
                    return prediction[0]
                else:
                    return prediction
            else:
                # 回退到推理服务
                prediction = self.inference_service.predict(data, mode=mode)
                self.stats["inference_success"] += 1
                return prediction
        except Exception as exc:
            self.stats["inference_failed"] += 1
            self.logger.error("推理失败: %s", exc)
            raise

    # ------------------------------------------------------------------ #
    # 模型管理
    # ------------------------------------------------------------------ #
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有已加载的模型"""
        # 如果有自定义的model_manager，使用它
        if hasattr(self, 'model_manager') and self.model_manager != self and hasattr(self.model_manager, 'list_models'):
            models = self.model_manager.list_models()
            # 如果返回的是字符串列表，转换为字典列表格式
            if models and isinstance(models[0], str):
                return [{"model_id": model_id} for model_id in models]
            return models

        # 否则使用内部模型存储
        models = []
        for model_id, model in self._models.items():
            config = self._model_configs.get(model_id, {})
            performance = self._model_performance.get(model_id, {})

            models.append({
                "id": model_id,
                "algorithm": config.get("algorithm", "unknown"),
                "status": "loaded",
                "performance": performance,
                "config": config,
            })

        # 如果有外部模型管理器，也包含其模型
        if hasattr(self.model_manager, "list_models"):
            external_models = self.model_manager.list_models()
            for model_info in external_models:
                if isinstance(model_info, dict) and model_info.get("id") not in [m["id"] for m in models]:
                    models.append(model_info)

        return models

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        # 如果有自定义的model_manager，使用它
        if hasattr(self, 'model_manager') and self.model_manager != self and hasattr(self.model_manager, 'get_model_info'):
            info = self.model_manager.get_model_info(model_id)
            # 如果返回的是字典但格式不同，转换为标准格式
            if info and isinstance(info, dict) and "model_id" not in info:
                info["model_id"] = model_id
            return info

        # 否则使用内部模型存储
        if model_id in self._models:
            config = self._model_configs.get(model_id, {})
            performance = self._model_performance.get(model_id, {})

            return {
                "id": model_id,
                "algorithm": config.get("algorithm", "unknown"),
                "status": "loaded",
                "loaded_at": config.get("loaded_at"),
                "performance": performance,
                "config": config,
            }

        # 尝试从外部模型管理器获取
        if hasattr(self.model_manager, "get_model_info"):
            return self.model_manager.get_model_info(model_id)

        return None

    def load_model(self, model_id: str, model_config: Dict[str, Any]) -> bool:
        """加载模型"""
        try:
            algorithm = model_config.get("algorithm", "linear_regression")

            # 创建模型实例（这里使用简单的模拟实现）
            if algorithm == "linear_regression":
                model = self._create_linear_regression_model(model_config)
            elif algorithm == "random_forest":
                model = self._create_random_forest_model(model_config)
            elif algorithm == "xgboost":
                model = self._create_xgboost_model(model_config)
            elif algorithm == "voting":
                model = self._create_voting_model(model_config)
            else:
                # 对于无效算法，抛出异常
                raise ValueError(f"不支持的算法类型: {algorithm}")

            # 存储模型
            self._models[model_id] = model
            self._model_configs[model_id] = {
                **model_config,
                "loaded_at": datetime.now().isoformat(),
            }

            self.stats["model_loads"] += 1
            self.logger.info("模型 %s 加载成功", model_id)
            return True

        except Exception as exc:
            self.logger.error("加载模型 %s 失败: %s", model_id, exc)
            return False

    def unload_model(self, model_id: str) -> bool:
        """卸载模型"""
        try:
            if model_id in self._models:
                del self._models[model_id]
                if model_id in self._model_configs:
                    del self._model_configs[model_id]
                if model_id in self._model_performance:
                    del self._model_performance[model_id]

                self.stats["model_unloads"] += 1
                self.logger.info("模型 %s 卸载成功", model_id)
                return True
            else:
                self.logger.warning("模型 %s 未找到", model_id)
                return False

        except Exception as exc:
            self.logger.error("卸载模型 %s 失败: %s", model_id, exc)
            return False

    def train_model(self, model_id: str, training_data: Any, model_config: Dict[str, Any]) -> bool:
        """训练模型"""
        try:
            self.stats["training_sessions"] += 1

            # 创建和训练模型
            success = self.load_model(model_id, model_config)
            if not success:
                return False

            model = self._models.get(model_id)
            if model is None:
                return False

            # 执行训练
            training_result = self._train_model_instance(model, training_data, model_config)

            # 确保性能指标被正确设置
            if training_result.get("status") == "completed":
                self._model_performance[model_id] = training_result.get("performance", {})
                self.logger.info("模型 %s 训练完成", model_id)
                return True
            else:
                error_msg = training_result.get("error", "未知错误")
                if not isinstance(error_msg, str):
                    error_msg = str(error_msg)
                self.logger.error("模型 %s 训练失败: %s", model_id, error_msg)
                return False

        except Exception as exc:
            self.logger.error("训练模型 %s 失败: %s", model_id, exc)
            return False

    def predict_batch(self, requests: List[MLInferenceRequest]) -> List[MLInferenceResponse]:
        """批量推理"""
        if self.status != MLServiceStatus.RUNNING:
            raise RuntimeError("MLService 未启动")

        self.stats["batch_predictions"] += 1
        responses = []

        try:
            for request in requests:
                try:
                    # 获取模型
                    model = self._models.get(request.model_id)
                    if model is None:
                        responses.append(MLInferenceResponse(
                            request_id=f"batch_{len(responses)}",
                            success=False,
                            error_message=f"模型 {request.model_id} 未找到"
                        ))
                        continue

                    # 执行推理
                    prediction = self._predict_with_model(model, request.features)

                    responses.append(MLInferenceResponse(
                        request_id=f"batch_{len(responses)}",
                        success=True,
                        prediction=prediction,
                        confidence=0.8,  # 模拟置信度
                        processing_time_ms=10.0
                    ))

                except Exception as exc:
                    responses.append(MLInferenceResponse(
                        request_id=f"batch_{len(responses)}",
                        success=False,
                        error_message=str(exc)
                    ))

            return responses

        except Exception as exc:
            self.logger.error("批量推理失败: %s", exc)
            return []

    def optimize_hyperparameters(self, model_id: str, param_space: Dict[str, Any], training_data: Any) -> Dict[str, Any]:
        """超参数优化"""
        try:
            # 验证训练数据
            if not hasattr(training_data, 'iloc') or len(training_data) == 0:
                return {"error": "训练数据无效或为空"}

            # 验证参数空间
            if not param_space:
                return {"error": "参数空间为空"}

            # 简单的网格搜索实现
            best_params = {}
            best_score = float('-inf')

            # 生成参数组合（简化版）
            param_combinations = self._generate_param_combinations(param_space)

            if not param_combinations:
                return {"error": "无法生成参数组合"}

            for i, params in enumerate(param_combinations):
                try:
                    # 使用当前参数训练模型并评估
                    temp_config = {"algorithm": "linear_regression", "params": params}
                    # 使用索引替代哈希，避免 unhashable type: 'dict' 错误
                    temp_model_id = f"{model_id}_temp_{i}"

                    # 训练临时模型
                    success = self.train_model(temp_model_id, training_data, temp_config)
                    if success:
                        # 评估模型性能
                        score = self._evaluate_model_performance(temp_model_id, training_data)

                        if score > best_score:
                            best_score = score
                            best_params = params

                    # 清理临时模型
                    self.unload_model(temp_model_id)

                except Exception as temp_exc:
                    self.logger.warning("参数组合 %s 训练失败: %s", params, temp_exc)
                    continue

            if not best_params:
                return {"error": "所有参数组合训练均失败"}

            return {
                "best_params": best_params,
                "best_score": best_score,
                "total_combinations": len(param_combinations)
            }

        except Exception as exc:
            self.logger.error("超参数优化失败: %s", exc)
            return {"error": str(exc)}

    def get_model_performance(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型性能指标"""
        return self._model_performance.get(model_id)

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "status": self.status.value,
            "stats": self.stats.copy(),
            "max_workers": self.max_workers,
            "loaded_models": len(self._models),
            "available_models": list(self._models.keys()),
            "uptime": "unknown",  # 可以后续添加
        }

    # ------------------------------------------------------------------ #
    # 私有辅助方法
    # ------------------------------------------------------------------ #
    def _create_linear_regression_model(self, config: Dict[str, Any]) -> Any:
        """创建线性回归模型"""
        class SimpleLinearRegression:
            def __init__(self, params=None):
                self.params = params or {}
                self.coef_ = None
                self.intercept_ = 0.0

            def fit(self, X, y):
                # 简单的线性回归实现
                X = np.array(X) if not isinstance(X, np.ndarray) else X
                y = np.array(y) if not isinstance(y, np.ndarray) else y

                # 使用最小二乘法
                if X.ndim == 1:
                    X = X.reshape(-1, 1)

                # 添加偏置项
                X_bias = np.column_stack([np.ones(X.shape[0]), X])

                # 计算系数
                try:
                    self.coef_ = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
                    self.intercept_ = self.coef_[0]
                    self.coef_ = self.coef_[1:]
                except np.linalg.LinAlgError:
                    # 如果矩阵不可逆，使用默认值
                    self.coef_ = np.ones(X.shape[1]) * 0.1
                    self.intercept_ = np.mean(y)

            def predict(self, X):
                X = np.array(X) if not isinstance(X, np.ndarray) else X
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                return X @ self.coef_ + self.intercept_

        return SimpleLinearRegression(config.get("params", {}))

    def _create_random_forest_model(self, config: Dict[str, Any]) -> Any:
        """创建随机森林模型"""
        class SimpleRandomForest:
            def __init__(self, params=None):
                self.params = params or {}
                self.n_estimators = self.params.get("n_estimators", 10)
                self.models = []

            def fit(self, X, y):
                # 简化的随机森林实现
                X = np.array(X)
                y = np.array(y)

                for _ in range(self.n_estimators):
                    # 随机采样
                    indices = np.random.choice(len(X), size=len(X), replace=True)
                    X_sample = X[indices]
                    y_sample = y[indices]

                    # 训练简单决策树（这里用线性模型替代）
                    model = self._create_simple_tree()
                    model.fit(X_sample, y_sample)
                    self.models.append(model)

            def predict(self, X):
                X = np.array(X)
                predictions = np.array([model.predict(X) for model in self.models])
                return np.mean(predictions, axis=0)

            def _create_simple_tree(self):
                # 直接创建线性回归模型
                class SimpleTree:
                    def __init__(self):
                        self.coef_ = None
                        self.intercept_ = 0.0

                    def fit(self, X, y):
                        X = np.array(X)
                        y = np.array(y)
                        if X.ndim == 1:
                            X = X.reshape(-1, 1)

                        # 添加偏置项
                        X_bias = np.column_stack([np.ones(X.shape[0]), X])

                        try:
                            coef = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
                            self.intercept_ = coef[0]
                            self.coef_ = coef[1:]
                        except np.linalg.LinAlgError:
                            self.coef_ = np.ones(X.shape[1]) * 0.1
                            self.intercept_ = np.mean(y)

                    def predict(self, X):
                        X = np.array(X)
                        if X.ndim == 1:
                            X = X.reshape(-1, 1)
                        return X @ self.coef_ + self.intercept_

                return SimpleTree()

        return SimpleRandomForest(config.get("params", {}))

    def _create_xgboost_model(self, config: Dict[str, Any]) -> Any:
        """创建XGBoost模型"""
        # XGBoost实现与随机森林类似，但这里简化为随机森林的变体
        return self._create_random_forest_model(config)

    def _create_voting_model(self, config: Dict[str, Any]) -> Any:
        """创建投票集成模型"""
        class SimpleBaseModel:
            """简单的基准模型"""
            def __init__(self, model_type="linear"):
                self.model_type = model_type
                self.coef_ = None
                self.intercept_ = 0.0

            def fit(self, X, y):
                X = np.array(X) if not isinstance(X, np.ndarray) else X
                y = np.array(y) if not isinstance(y, np.ndarray) else y

                if self.model_type == "linear":
                    # 简单的线性回归
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    X_bias = np.column_stack([np.ones(X.shape[0]), X])
                    try:
                        coef = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
                        self.intercept_ = coef[0]
                        self.coef_ = coef[1:]
                    except:
                        self.coef_ = np.ones(X.shape[1]) * 0.1
                        self.intercept_ = np.mean(y)
                else:
                    # 随机森林简化版
                    self.trees = []
                    n_estimators = 5
                    for _ in range(n_estimators):
                        # 简单的决策树替代
                        tree = SimpleBaseModel("linear")
                        tree.fit(X, y)
                        self.trees.append(tree)

            def predict(self, X):
                X = np.array(X) if not isinstance(X, np.ndarray) else X

                if self.model_type == "linear":
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    return X @ self.coef_ + self.intercept_
                else:
                    # 随机森林预测
                    predictions = np.array([tree.predict(X) for tree in self.trees])
                    return np.mean(predictions, axis=0)

        class SimpleVotingClassifier:
            def __init__(self, params=None):
                self.params = params or {}
                self.estimators = self.params.get("estimators", ["random_forest", "linear_regression"])
                self.models = []
                self.is_trained = False

            def fit(self, X, y):
                """训练多个基础模型"""
                self.models = []
                X = np.array(X) if not isinstance(X, np.ndarray) else X
                y = np.array(y) if not isinstance(y, np.ndarray) else y

                # 根据配置创建多个基础模型
                for estimator_type in self.estimators:
                    if estimator_type == "random_forest":
                        model = SimpleBaseModel("forest")
                    elif estimator_type == "svm":
                        model = SimpleBaseModel("linear")
                    elif estimator_type == "xgboost":
                        model = SimpleBaseModel("forest")
                    else:
                        # 默认使用线性回归
                        model = SimpleBaseModel("linear")

                    # 训练模型
                    model.fit(X, y)
                    self.models.append(model)

                self.is_trained = True

            def predict(self, X):
                """进行投票预测"""
                if not self.is_trained:
                    raise RuntimeError("模型尚未训练")

                X = np.array(X) if not isinstance(X, np.ndarray) else X
                if X.ndim == 1:
                    X = X.reshape(1, -1)

                # 获取所有模型的预测
                predictions = []
                for model in self.models:
                    pred = model.predict(X)
                    # 将连续预测转换为类别标签（这里简化为四舍五入到最近的整数）
                    pred_classes = np.round(pred).astype(int)
                    predictions.append(pred_classes)

                predictions = np.array(predictions)  # shape: (n_models, n_samples)

                # 多分类投票：多数投票
                final_predictions = []
                for i in range(X.shape[0]):
                    # 对于每个样本，收集所有模型的预测
                    sample_predictions = predictions[:, i]

                    # 使用多数投票
                    unique, counts = np.unique(sample_predictions, return_counts=True)
                    majority_vote = unique[np.argmax(counts)]
                    final_predictions.append(majority_vote)

                return np.array(final_predictions)

        return SimpleVotingClassifier(config.get("params", {}))

    def _train_model_instance(self, model: Any, training_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """训练模型实例"""
        try:
            # 解析训练数据
            if hasattr(training_data, 'iloc'):  # DataFrame
                # 找到目标列
                target_cols = []
                for col in training_data.columns:
                    col_str = str(col).lower() if hasattr(col, 'lower') else str(col)
                    if 'target' in col_str or col_str in ['y', 'label']:
                        target_cols.append(col)

                if target_cols:
                    y_col = target_cols[0]
                    X = training_data.drop(columns=[y_col])
                    y = training_data[y_col]
                else:
                    # 如果没有找到目标列，使用最后一列作为目标
                    X = training_data.iloc[:, :-1]
                    y = training_data.iloc[:, -1]
            else:
                # 假设是字典格式
                X = training_data.get('X', training_data.get('features', []))
                y = training_data.get('y', training_data.get('target', []))

            # 训练模型
            model.fit(X, y)

            # 计算训练性能
            y_pred = model.predict(X)
            mse = np.mean((np.array(y) - np.array(y_pred)) ** 2)
            rmse = np.sqrt(mse)
            r2 = 1 - mse / np.var(y) if np.var(y) > 0 else 0

            return {
                "performance": {
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "r2_score": float(r2),
                    "training_samples": len(y)
                },
                "training_time": 1.0,  # 模拟训练时间
                "status": "completed"
            }

        except Exception as exc:
            return {
                "performance": {},
                "error": str(exc),
                "status": "failed"
            }

    def _predict_with_model(self, model: Any, features: MLFeatures) -> Any:
        """使用模型进行预测"""
        # 将MLFeatures转换为模型输入格式
        feature_dict = features.features if hasattr(features, 'features') else features.data

        # 转换为numpy数组
        feature_values = list(feature_dict.values())
        X = np.array(feature_values).reshape(1, -1)

        return model.predict(X)[0]

    def _generate_param_combinations(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成参数组合"""
        # 简化实现，只返回几个预定义的组合
        combinations = []

        if 'alpha' in param_space:
            for alpha in [0.01, 0.1, 1.0, 10.0]:
                combinations.append({'alpha': alpha})

        if 'max_depth' in param_space:
            for depth in [3, 5, 7, 10]:
                combinations.append({'max_depth': depth})

        # 如果没有找到参数，返回默认组合
        if not combinations:
            combinations = [{'default': 1.0}]

        return combinations

    def _evaluate_model_performance(self, model_id: str, validation_data: Any) -> float:
        """评估模型性能"""
        try:
            model = self._models.get(model_id)
            if model is None:
                return 0.0

            # 使用验证数据进行预测
            if hasattr(validation_data, 'iloc'):
                X = validation_data.drop(columns=[col for col in validation_data.columns if 'target' in col.lower()])
                y_true = validation_data[[col for col in validation_data.columns if 'target' in col.lower() or col.lower() in ['y', 'label']][0]]
            else:
                X = validation_data.get('X', [])
                y_true = validation_data.get('y', [])

            y_pred = model.predict(X)

            # 计算R²分数作为性能指标
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return max(0, r2)  # 确保返回非负值

        except Exception:
            return 0.0


__all__ = [
    "MLService",
    "MLServiceStatus",
    "get_models_adapter",
]