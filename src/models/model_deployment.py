import os
import logging
import pickle
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import onnxruntime as ort
from pydantic import BaseModel
import hashlib
from datetime import datetime
import json
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelFormat(Enum):
    PICKLE = "pickle"
    ONNX = "onnx"
    PMML = "pmml"

class ModelDeployment:
    """模型部署核心类"""

    def __init__(self, model_dir: str = "deployed_models"):
        """
        Args:
            model_dir: 模型存储目录
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # 内存中的模型缓存
        self.model_cache = {}

        # 模型元数据存储
        self.metadata_file = os.path.join(model_dir, "metadata.json")
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """加载模型元数据"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """保存模型元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _generate_version(self) -> str:
        """生成版本号"""
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def _get_model_path(self, model_name: str, version: str, fmt: ModelFormat) -> str:
        """获取模型存储路径"""
        return os.path.join(
            self.model_dir,
            f"{model_name}_{version}.{fmt.value}"
        )

    def export_model(self,
                   model: Any,
                   model_name: str,
                   input_sample: np.ndarray,
                   fmt: ModelFormat = ModelFormat.ONNX) -> str:
        """
        导出模型到指定格式

        Args:
            model: 待导出模型
            model_name: 模型名称
            input_sample: 输入样本(用于ONNX转换)
            fmt: 导出格式

        Returns:
            模型版本号
        """
        version = self._generate_version()
        model_path = self._get_model_path(model_name, version, fmt)

        try:
            if fmt == ModelFormat.PICKLE:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

            elif fmt == ModelFormat.ONNX:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType

                # 简化处理: 假设输入都是float32
                initial_type = [('float_input', FloatTensorType([None, input_sample.shape[1]]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)

                with open(model_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())

            else:
                raise ValueError(f"Unsupported format: {fmt}")

            # 记录元数据
            self.metadata[f"{model_name}_{version}"] = {
                "name": model_name,
                "version": version,
                "format": fmt.value,
                "path": model_path,
                "create_time": datetime.now().isoformat(),
                "input_shape": input_sample.shape,
                "hash": hashlib.md5(open(model_path, 'rb').read()).hexdigest()
            }
            self._save_metadata()

            return version

        except Exception as e:
            logger.error(f"Failed to export model: {str(e)}")
            if os.path.exists(model_path):
                os.remove(model_path)
            raise

    def load_model(self, model_name: str, version: str) -> Any:
        """
        加载指定版本的模型

        Args:
            model_name: 模型名称
            version: 模型版本

        Returns:
            加载的模型
        """
        cache_key = f"{model_name}_{version}"

        # 检查缓存
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        # 查找模型文件
        model_info = None
        for k, v in self.metadata.items():
            if v["name"] == model_name and v["version"] == version:
                model_info = v
                break

        if not model_info:
            raise ValueError(f"Model {model_name} version {version} not found")

        try:
            model_path = model_info["path"]
            fmt = ModelFormat(model_info["format"])

            if fmt == ModelFormat.PICKLE:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

            elif fmt == ModelFormat.ONNX:
                model = ort.InferenceSession(model_path)

            else:
                raise ValueError(f"Unsupported format: {fmt}")

            # 更新缓存
            self.model_cache[cache_key] = model
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict(self,
               model_name: str,
               version: str,
               input_data: np.ndarray) -> np.ndarray:
        """
        使用指定模型进行预测

        Args:
            model_name: 模型名称
            version: 模型版本
            input_data: 输入数据

        Returns:
            预测结果
        """
        model = self.load_model(model_name, version)

        try:
            if isinstance(model, ort.InferenceSession):
                # ONNX模型预测
                input_name = model.get_inputs()[0].name
                output_name = model.get_outputs()[0].name
                return model.run([output_name], {input_name: input_data.astype(np.float32)})[0]
            else:
                # 普通模型预测
                return model.predict(input_data)

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def list_models(self) -> Dict[str, Any]:
        """列出所有可用模型"""
        return self.metadata

    def delete_model(self, model_name: str, version: str):
        """删除指定模型"""
        cache_key = f"{model_name}_{version}"

        # 查找模型信息
        model_info = None
        for k, v in self.metadata.items():
            if v["name"] == model_name and v["version"] == version:
                model_info = v
                break

        if not model_info:
            raise ValueError(f"Model {model_name} version {version} not found")

        try:
            # 删除文件
            os.remove(model_info["path"])

            # 更新元数据
            del self.metadata[cache_key]
            self._save_metadata()

            # 清理缓存
            if cache_key in self.model_cache:
                del self.model_cache[cache_key]

        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}")
            raise

class PredictionRequest(BaseModel):
    """预测请求数据模型"""
    model_name: str
    version: str
    data: list

class DeploymentService:
    """模型部署服务"""

    def __init__(self, deployment: ModelDeployment):
        self.app = FastAPI(
            title="RQA Model Deployment API",
            description="API for model deployment and prediction",
            version="1.0.0"
        )
        self.deployment = deployment

        # 注册路由
        self.app.post("/predict")(self.predict)
        self.app.get("/models")(self.list_models)
        self.app.delete("/models/{model_name}/{version}")(self.delete_model)

    async def predict(self, request: PredictionRequest):
        """预测接口"""
        try:
            data = np.array(request.data)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)

            result = self.deployment.predict(
                request.model_name,
                request.version,
                data
            )
            return {"prediction": result.tolist()}

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def list_models(self):
        """列出模型接口"""
        return self.deployment.list_models()

    async def delete_model(self, model_name: str, version: str):
        """删除模型接口"""
        try:
            self.deployment.delete_model(model_name, version)
            return {"status": "success"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行服务"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)

class ABTestFramework:
    """AB测试框架"""

    def __init__(self, deployment: ModelDeployment):
        self.deployment = deployment
        self.active_experiments = {}

    def start_experiment(self,
                       experiment_name: str,
                       model_a: str,
                       version_a: str,
                       model_b: str,
                       version_b: str,
                       traffic_split: float = 0.5):
        """
        开始AB测试实验

        Args:
            experiment_name: 实验名称
            model_a: 模型A名称
            version_a: 模型A版本
            model_b: 模型B名称
            version_b: 模型B版本
            traffic_split: 流量分配比例(0-1)
        """
        if experiment_name in self.active_experiments:
            raise ValueError(f"Experiment {experiment_name} already exists")

        self.active_experiments[experiment_name] = {
            "model_a": (model_a, version_a),
            "model_b": (model_b, version_b),
            "traffic_split": traffic_split,
            "stats": {
                "total_requests": 0,
                "a_requests": 0,
                "b_requests": 0
            }
        }

    def predict(self,
               experiment_name: str,
               input_data: np.ndarray) -> Dict[str, Any]:
        """
        AB测试预测

        Args:
            experiment_name: 实验名称
            input_data: 输入数据

        Returns:
            包含预测结果和模型信息的字典
        """
        if experiment_name not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        exp = self.active_experiments[experiment_name]
        exp["stats"]["total_requests"] += 1

        # 分配流量
        use_a = np.random.random() < exp["traffic_split"]

        if use_a:
            model_name, version = exp["model_a"]
            exp["stats"]["a_requests"] += 1
        else:
            model_name, version = exp["model_b"]
            exp["stats"]["b_requests"] += 1

        prediction = self.deployment.predict(model_name, version, input_data)

        return {
            "prediction": prediction,
            "model_used": model_name,
            "version_used": version,
            "experiment": experiment_name
        }

    def get_experiment_stats(self, experiment_name: str) -> Dict[str, Any]:
        """获取实验统计信息"""
        if experiment_name not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        return self.active_experiments[experiment_name]["stats"]

    def end_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """结束实验并返回最终统计"""
        if experiment_name not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_name} not found")

        stats = self.active_experiments.pop(experiment_name)
        return stats
