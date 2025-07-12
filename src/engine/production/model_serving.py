import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    """预测请求数据结构"""
    model_id: str                   # 模型标识
    features: Dict[str, float]      # 特征字典
    request_id: Optional[str]       # 请求ID(可选)
    timestamp: Optional[datetime]  # 时间戳(可选)

class PredictionResponse(BaseModel):
    """预测响应数据结构"""
    request_id: str                 # 请求ID
    prediction: Union[float, Dict]  # 预测结果
    model_version: str              # 模型版本
    timestamp: datetime             # 响应时间
    metadata: Optional[Dict]       # 元数据

class ModelWrapper:
    """模型包装器"""

    def __init__(self, model_path: str, version: str = "1.0"):
        """
        初始化模型包装器

        Args:
            model_path: 模型文件路径
            version: 模型版本
        """
        self.model = self._load_model(model_path)
        self.version = version
        self.feature_names = self._get_feature_names()
        self.last_used = datetime.now()

    def _load_model(self, model_path: str):
        """加载模型"""
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def _get_feature_names(self) -> List[str]:
        """获取模型特征名称"""
        if hasattr(self.model, 'feature_names_in_'):
            return list(self.model.feature_names_in_)
        elif hasattr(self.model, 'get_booster') and hasattr(self.model.get_booster(), 'feature_names'):
            return self.model.get_booster().feature_names
        else:
            logger.warning("Feature names not available in model")
            return []

    def preprocess(self, features: Dict[str, float]) -> np.ndarray:
        """特征预处理"""
        # 确保特征顺序与训练时一致
        if self.feature_names:
            return np.array([features.get(f, 0) for f in self.feature_names])
        else:
            return np.array(list(features.values()))

    def predict(self, features: Dict[str, float]) -> Union[float, Dict]:
        """执行预测"""
        try:
            # 特征预处理
            X = self.preprocess(features)
            X = X.reshape(1, -1)

            # 执行预测
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[0]
                return {
                    'prediction': self.model.predict(X)[0],
                    'probabilities': proba.tolist()
                }
            else:
                return float(self.model.predict(X)[0])

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Prediction error: {str(e)}")

class ModelService:
    """模型服务"""

    def __init__(self, max_workers: int = 4):
        """
        初始化模型服务

        Args:
            max_workers: 最大并发工作线程数
        """
        self.models: Dict[str, ModelWrapper] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.app = FastAPI(title="Quant Model Serving")
        self._setup_routes()

    def _setup_routes(self):
        """设置API路由"""
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            return await self._handle_prediction(request)

        @self.app.get("/models")
        async def list_models():
            return {
                "models": list(self.models.keys()),
                "count": len(self.models)
            }

        @self.app.get("/model/{model_id}")
        async def get_model_info(model_id: str):
            if model_id not in self.models:
                raise HTTPException(status_code=404, detail="Model not found")
            return {
                "model_id": model_id,
                "version": self.models[model_id].version,
                "last_used": self.models[model_id].last_used,
                "feature_names": self.models[model_id].feature_names
            }

    async def _handle_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """处理预测请求"""
        if request.model_id not in self.models:
            raise HTTPException(status_code=404, detail="Model not found")

        try:
            # 更新最后使用时间
            self.models[request.model_id].last_used = datetime.now()

            # 执行预测
            prediction = await self.executor.submit(
                self.models[request.model_id].predict,
                request.features
            )

            return PredictionResponse(
                request_id=request.request_id or "N/A",
                prediction=prediction,
                model_version=self.models[request.model_id].version,
                timestamp=datetime.now(),
                metadata={
                    "model_type": type(self.models[request.model_id].model).__name__,
                    "features_used": list(request.features.keys())
                }
            )
        except Exception as e:
            logger.error(f"Prediction failed for {request.model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def load_model(self, model_id: str, model_path: str, version: str = "1.0"):
        """加载模型到服务"""
        if model_id in self.models:
            logger.warning(f"Model {model_id} already loaded, will be replaced")

        try:
            self.models[model_id] = ModelWrapper(model_path, version)
            logger.info(f"Model {model_id} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def unload_model(self, model_id: str):
        """卸载模型"""
        if model_id in self.models:
            del self.models[model_id]
            logger.info(f"Model {model_id} unloaded")
        else:
            logger.warning(f"Model {model_id} not found")

    def get_model(self, model_id: str) -> Optional[ModelWrapper]:
        """获取模型实例"""
        return self.models.get(model_id)

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """启动服务"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

class ABTestManager:
    """AB测试管理器"""

    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.experiments: Dict[str, Dict] = {}

    def create_experiment(self, exp_id: str, models: Dict[str, float]):
        """
        创建AB测试实验

        Args:
            exp_id: 实验ID
            models: 模型权重映射 {model_id: weight}
        """
        total_weight = sum(models.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")

        for model_id in models:
            if model_id not in self.model_service.models:
                raise ValueError(f"Model {model_id} not loaded")

        self.experiments[exp_id] = {
            "models": models,
            "stats": {m: {"requests": 0} for m in models}
        }
        logger.info(f"AB test experiment {exp_id} created")

    def predict(self, exp_id: str, features: Dict) -> Dict:
        """
        执行AB测试预测

        Args:
            exp_id: 实验ID
            features: 输入特征

        Returns:
            {
                "model_id": 选择的模型ID,
                "prediction": 预测结果,
                "experiment_id": 实验ID
            }
        """
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")

        # 根据权重随机选择模型
        models = self.experiments[exp_id]["models"]
        model_id = np.random.choice(
            list(models.keys()),
            p=list(models.values())
        )

        # 记录使用统计
        self.experiments[exp_id]["stats"][model_id]["requests"] += 1

        # 执行预测
        prediction = self.model_service.models[model_id].predict(features)

        return {
            "model_id": model_id,
            "prediction": prediction,
            "experiment_id": exp_id
        }
