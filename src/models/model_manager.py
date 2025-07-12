# src/models/model_manager.py
import importlib
import inspect
from pathlib import Path

import joblib
import shap
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from typing import Dict, Any, Optional, Tuple
import datetime
import numpy as np
import pandas as pd
import torch
import arviz as az
import pymc as pm

from src.models.base_model import BaseModel
from src.models.model_lstm import LSTMModelWrapper
from src.models.rf import RandomForestModel
from src.models.nn import NeuralNetworkModel
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)


class ModelExplainer:
    """模型解释性工具类"""

    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.is_trained = self._check_if_model_is_trained()

    def _check_if_model_is_trained(self) -> bool:
        """检查模型是否已经训练完成"""
        if hasattr(self.model, 'is_trained'):
            return self.model.is_trained
        elif hasattr(self.model, '_is_trained'):
            return self.model._is_trained
        else:
            logger.warning("模型没有 is_trained 或 _is_trained 属性，假设模型已经训练完成")
            return True

    def explain_with_shap(self, X):
        """使用SHAP值解释模型"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，无法解释")

        # 删除对随机森林模型的额外检查
        if isinstance(self.model, (LSTMModelWrapper, NeuralNetworkModel)):
            # 为深度学习模型提供背景数据集
            background = X[np.random.choice(X.shape[0], 100, replace=False)]
            explainer = shap.DeepExplainer(self.model, background)
        elif isinstance(self.model, RandomForestModel):
            # 对于随机森林模型，使用 TreeExplainer
            explainer = shap.TreeExplainer(self.model.model)
        else:
            # 对于其他模型，使用通用Explainer
            explainer = shap.Explainer(self.model)

        shap_values = explainer.shap_values(X)
        return shap_values

    def get_feature_importance(self, X, y=None):
        """获取特征重要性"""
        if isinstance(self.model, RandomForestModel):
            # 随机森林模型的特征重要性
            return pd.Series(self.model.model.feature_importances_, index=self.feature_names)
        elif hasattr(self.model, 'feature_importances_'):
            # 其他支持 feature_importances_ 的模型
            return pd.Series(self.model.feature_importances_, index=self.feature_names)
        else:
            # 对于不支持 feature_importances_ 的模型，使用置换重要性
            return self._permutation_importance(X, y)

    def feature_importance(self, x, y):
        """特征重要性分析"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.model.feature_importances_
        else:
            # 对于神经网络模型，使用 permutation importance
            importances = self._permutation_importance(x, y)
        return dict(zip(self.feature_names, importances))

    def _permutation_importance(self, X, y):
        """计算特征的置换重要性"""
        if y is None:
            raise ValueError("置换重要性计算需要目标变量 y")

        baseline = self._model_performance(X, y)
        importances = []

        for col in X.columns:
            # 保存原始列
            original = X[col].copy()
            # 置换列
            X[col] = np.random.permutation(X[col])
            # 评估性能下降
            performance = self._model_performance(X, y)
            # 恢复原始列
            X[col] = original
            # 计算重要性
            importances.append(baseline - performance)

        return pd.Series(importances, index=X.columns)

    def _model_performance(self, X, y):
        """评估模型性能"""
        y_pred = self.model.predict(X)
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)[:, 1]
            return roc_auc_score(y, y_proba)
        else:
            if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
                return mean_squared_error(y.values, y_pred)
            else:
                # 检查是否为分类任务
                if set(np.unique(y_pred)).issubset({0, 1}):
                    return accuracy_score(y, np.round(y_pred))
                else:
                    return mean_squared_error(y, y_pred)

    def analyze_feature_interactions(self, features: pd.DataFrame) -> pd.DataFrame:
        """特征交互分析"""
        if not self.is_trained or self.model is None:
            raise RuntimeError("模型尚未训练")

        # 初始化 TreeExplainer
        explainer = shap.TreeExplainer(self.model)

        # 计算 SHAP 交互值
        shap_interaction_values = explainer.shap_interaction_values(features)

        # 取第一个输出的交互值（针对回归任务）
        interaction_matrix = np.mean(np.abs(shap_interaction_values[0]), axis=0)

        # 转换为 DataFrame
        interaction_df = pd.DataFrame(
            interaction_matrix,
            columns=features.columns,
            index=features.columns
        )

        return interaction_df


class ModelDriftDetector:
    """模型漂移检测类"""

    def __init__(self):
        pass

    def calculate_psi(self, expected, actual, buckets=10):
        """计算PSI指标"""
        expected_percentile = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        actual_percentile = np.percentile(actual, np.linspace(0, 100, buckets + 1))

        expected_dist = np.histogram(expected, bins=expected_percentile)[0] / len(expected)
        actual_dist = np.histogram(actual, bins=expected_percentile)[0] / len(actual)

        psi = 0
        for e, a in zip(expected_dist, actual_dist):
            if e == 0:
                e = 0.0001
            if a == 0:
                a = 0.0001
            psi += (a - e) * np.log(a / e)
        return psi

    def calculate_pd(self, expected, actual, metric='js'):
        """计算分布差异指标"""
        if metric == 'js':
            expected_dist = np.histogram(expected, bins=10)[0] / len(expected)
            actual_dist = np.histogram(actual, bins=10)[0] / len(actual)
            return self._js_divergence(expected_dist, actual_dist)
        else:
            raise ValueError("Unsupported metric")

    def _js_divergence(self, p, q):
        """计算Jensen-Shannon Divergence"""
        M = (p + q) / 2
        return 0.5 * (self._kl_divergence(p, M) + self._kl_divergence(q, M))

    def _kl_divergence(self, p, q):
        """计算KL散度"""
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))


class ModelEnsembler:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.models = {
            "lstm": LSTMModelWrapper,
            "random_forest": RandomForestModel,
            "neural_net": NeuralNetworkModel
        }
        self.weights = self._load_weights()

    def _load_config(self, path: str) -> Dict:
        # 加载配置文件
        pass

    def _load_weights(self) -> Dict:
        # 从配置加载模型权重
        return {
            "lstm": 0.5,
            "random_forest": 0.3,
            "neural_net": 0.2
        }

    def ensemble_predict(self, features: pd.DataFrame, model_types: list) -> pd.Series:
        """多模型集成预测"""
        predictions = []
        for model_type in model_types:
            model = self.models[model_type]()
            preds = model.predict(features)
            predictions.append(preds * self.weights[model_type])
        return pd.concat(predictions, axis=1).mean(axis=1)

    @staticmethod
    def bayesian_ensemble(models, features, true_targets, n_samples=500):
        """贝叶斯模型集成"""
        with pm.Model() as model:
            # 定义 Dirichlet 先验分布
            weights = pm.Dirichlet('weights', a=np.ones(len(models)))

            # 构建加权平均预测
            mu = 0
            for i, mdl in enumerate(models):
                mu += weights[i] * pm.Normal(f'pred_{i}', mu=mdl.predict(features), sigma=1)

            # 定义观测变量
            y_pred = pm.Normal('y_pred', mu=mu, sigma=1, observed=true_targets)

            # 采样
            trace = pm.sample(
                draws=n_samples,
                tune=1000,
                chains=2,
                return_inferencedata=True,
                random_seed=42
            )

            # 生成后验预测
            posterior_predictive = pm.sample_posterior_predictive(
                trace,
                var_names=['y_pred'],
                random_seed=42
            )

            # 返回后验预测数据
            return posterior_predictive

    @staticmethod
    def calculate_uncertainty(trace, var_name='y_pred'):
        """计算后验预测的均值和标准差"""
        # 提取后验预测数据
        posterior_pred = az.extract(trace, group="posterior_predictive")[var_name]

        # 检查维度名称并正确计算
        if 'chain' in posterior_pred.dims and 'draw' in posterior_pred.dims:
            # 标准维度结构
            mean_per_obs = posterior_pred.mean(dim=['chain', 'draw']).values
            std_per_obs = posterior_pred.std(dim=['chain', 'draw']).values
        elif 'sample' in posterior_pred.dims:
            # 备选维度名称
            mean_per_obs = posterior_pred.mean(dim='sample').values
            std_per_obs = posterior_pred.std(dim='sample').values
        else:
            # 回退方案：使用所有维度计算
            dims_to_reduce = [dim for dim in posterior_pred.dims if dim != 'obs']
            mean_per_obs = posterior_pred.mean(dim=dims_to_reduce).values
            std_per_obs = posterior_pred.std(dim=dims_to_reduce).values

        return mean_per_obs, std_per_obs


class ModelMonitor:
    """模型性能监控类"""

    def __init__(self, model, validation_data, metrics):
        self.model = model
        self.validation_data = validation_data
        self.metrics = metrics
        self.performance_history = []

    def monitor_performance(self):
        """监控模型性能"""
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)

        performance = {}
        for metric in self.metrics:
            if metric == 'accuracy':
                performance[metric] = accuracy_score(y_val, y_pred.round())
            elif metric == 'mse':
                performance[metric] = mean_squared_error(y_val, y_pred)
            elif metric == 'auc':
                if hasattr(self.model, 'predict_proba'):
                    y_proba = self.model.predict_proba(X_val)[:, 1]
                    performance[metric] = roc_auc_score(y_val, y_proba)
                else:
                    performance[metric] = None
            else:
                performance[metric] = None

        self.performance_history.append({
            'timestamp': datetime.datetime.now(),
            'performance': performance
        })

        # 保存性能历史
        self._save_performance_history()

    def _save_performance_history(self):
        """保存性能历史"""
        joblib.dump(self.performance_history, 'performance_history.pkl')


class ModelManager:
    """模型生命周期管理类，负责版本控制与元数据存储

    属性：
        base_path (Path): 模型存储根目录
        metadata_store (Path): 元数据存储目录
        device (torch.device): 当前设备配置
    """

    def __init__(self, base_path: str = "models", device: str = "auto"):
        """初始化模型管理器

        参数：
            base_path (str): 模型存储根目录，默认为"models"
            device (str): 设备配置，"auto"自动选择，默认为"auto"
        """
        self.base_path = Path(base_path)
        self.metadata_store = self.base_path / "metadata"
        self._ensure_directories()

        # 添加设备配置逻辑
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 确保设备选择正确
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        self.explainer = None
        self.drift_detector = ModelDriftDetector()
        self.monitor = None
        self.MODEL_MODULE_PREFIX = "src.models"  # 在配置中定义模型类前缀

    def _ensure_directories(self):
        """创建必要的存储目录"""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_store.mkdir(parents=True, exist_ok=True)

    def save_model(
            self,
            model: Any,
            model_name: str,
            version: str,
            feature_columns: list,
            metadata: Optional[Dict] = None,
            overwrite: bool = False
    ) -> Path:
        """保存模型及元数据"""
        # 创建版本目录
        dir_path = self.base_path / f"{model_name}_v{version}"
        dir_path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        if isinstance(model, LSTMModelWrapper):
            # LSTM模型包装类保存逻辑
            model_path = dir_path / f"{model_name}.pt"
            scaler_path = dir_path / f"{model_name}_scaler.pkl"

            if model_path.exists() and not overwrite:
                raise FileExistsError(f"模型文件已存在: {model_path}")

            torch.save({
                'model_state_dict': model.model.state_dict(),
                'config': model.config,
                'is_trained': model.is_trained,
                'feature_names_': model.feature_names_
            }, model_path)

            joblib.dump(model.scaler, scaler_path)
        elif isinstance(model, NeuralNetworkModel):
            # 神经网络模型保存逻辑
            model_path = dir_path / f"{model_name}.pt"
            scaler_path = dir_path / f"{model_name}_scaler.pkl"

            if model_path.exists() and not overwrite:
                raise FileExistsError(f"模型文件已存在: {model_path}")

            torch.save({
                'model_state_dict': model.model.state_dict(),
                'config': model.config,
                'is_trained': model.is_trained,
                'feature_names_': model.feature_names_
            }, model_path)

            joblib.dump(model.scaler, scaler_path)
        elif isinstance(model, RandomForestModel):
            # 随机森林模型保存逻辑
            model_path = dir_path / f"{model_name}.pkl"

            if model_path.exists() and not overwrite:
                raise FileExistsError(f"模型文件已存在: {model_path}")

            joblib.dump({
                'model': model.model,
                'config': model.config,
                'is_trained': model.is_trained,
                'feature_names_': model.feature_names_
            }, model_path)
        else:
            # 通用保存逻辑
            if hasattr(model, 'save'):
                model.save(dir_path, overwrite=overwrite)
            else:
                model_path = dir_path / f"{model_name}.pt"
                torch.save(model, model_path)

        # 保存元数据
        meta = {
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.datetime.now().isoformat(),
            "feature_columns": feature_columns,
            "metadata": metadata or {},
            "model_type": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "config": getattr(model, 'config', {}),
            "is_trained": getattr(model, 'is_trained', False)
        }

        metadata_path = self.metadata_store / f"{model_name}_v{version}.pkl"
        joblib.dump(meta, metadata_path)

        return dir_path

    def load_model(self, model_name: str, version: str = "latest") -> Tuple[Any, Dict]:
        """加载模型及元数据"""
        if version == "latest":
            version = self.get_latest_version(model_name)

        model_dir = self.base_path / f"{model_name}_v{version}"
        metadata_path = self.metadata_store / f"{model_name}_v{version}.pkl"

        if not model_dir.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Model {model_name} v{version} not found")

        logger.info(f"Loading model from: {model_dir}")
        logger.info(f"Loading metadata from: {metadata_path}")

        metadata = joblib.load(metadata_path)

        # 动态导入模型类
        model_type = metadata.get('model_type', 'src.models.lstm.LSTMModelWrapper')
        module_path, class_name = model_type.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)

        # 根据模型类型选择加载方式
        model_files = list(model_dir.glob(f"{model_name}.*"))
        if not model_files:
            raise FileNotFoundError(f"No model file found for {model_name} in {model_dir}")

        model_file = model_files[0]
        logger.info(f"Loading model from: {model_file}")

        if model_file.suffix == ".pt":
            map_location = self.device  # 使用管理器中的设备配置
            # 移除 model_name 参数和其他非初始化参数
            config = {k: v for k, v in metadata.get('config', {}).items()
                      if k in inspect.getfullargspec(model_class.__init__).args}
            model = model_class(**config)

            # 确保模型结构正确初始化
            if hasattr(model, 'build_model'):
                model.build_model()

            # 加载模型状态字典
            checkpoint = torch.load(model_file, map_location=map_location)
            if hasattr(model, 'model') and isinstance(model.model, torch.nn.Module):
                model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # 显式设置 is_trained 和 feature_names_
            model._is_trained = checkpoint.get('is_trained', False)
            model.feature_names_ = checkpoint.get('feature_names_', None)

            # 加载标准化器（如果存在）
            if hasattr(model, 'scaler'):
                scaler_path = model_dir / f"{model_name}_scaler.pkl"
                if scaler_path.exists():
                    model.scaler = joblib.load(scaler_path)
        elif model_file.suffix == ".pkl":
            model_data = joblib.load(model_file)
            if isinstance(model_data, dict):
                model = model_class(**model_data.get('config', {}))
                model.model = model_data.get('model')
                model._is_trained = model_data.get('is_trained', False)
                model.feature_names_ = model_data.get('feature_names_', None)
            else:
                model = model_data
        else:
            model = torch.load(model_file, map_location=self.device)

        # 确保元数据包含 'metadata' 字段
        if 'metadata' not in metadata:
            metadata['metadata'] = {}

        return model, metadata

    def _load_sklearn(self, path: Path) -> BaseModel:
        """加载Scikit-learn模型"""
        if path.suffix not in [".pkl"]:
            raise NotImplementedError(f"不支持的文件扩展名: {path.suffix}")
        model = joblib.load(path)
        instance = self._create_model_instance(model)
        instance.model = model
        instance._is_trained = True
        return instance

    def _load_pytorch(self, path: Path) -> BaseModel:
        """加载PyTorch模型"""
        if path.suffix not in [".pt"]:
            raise NotImplementedError(f"不支持的文件扩展名: {path.suffix}")
        checkpoint = torch.load(path, map_location=self.device)
        model_config = checkpoint.get('config', {})
        instance = self._create_model_instance(model_config)
        instance.build_model()
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance._is_trained = checkpoint.get('is_trained', False)
        return instance

    def _create_model_instance(self, config: Dict) -> BaseModel:
        """根据配置创建模型实例"""
        model_name = config.get('model_name', 'random_forest')
        if model_name == 'lstm':
            return LSTMModelWrapper(**config)
        elif model_name == 'neural_network':
            return NeuralNetworkModel(**config)
        elif model_name == 'random_forest':
            return RandomForestModel(**config)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def get_latest_version(self, model_name: str) -> str:
        """获取模型最新版本号"""
        versions = []
        # 确保 metadata_store 存在
        if not self.metadata_store.exists():
            return "0.0.0"

        for meta_file in self.metadata_store.glob(f"{model_name}_v*.pkl"):
            version = meta_file.stem.split("_")[-1][1:]  # 提取版本号（去掉 "v" 前缀）
            if all(part.isdigit() for part in version.split('.')[:3]):
                versions.append(version)
        if not versions:
            return "0.0.0"
        # 按语义化版本排序规则比较
        return max(versions, key=lambda v: tuple(map(int, v.split('.'))))

    def validate_model(
            self,
            model_name: str,
            version: str,
            checks: Dict[str, Any]
    ) -> bool:
        """验证模型是否符合要求，新增特征列顺序验证

        Args:
            model_name (str): 模型名称
            version (str): 版本号
            checks (Dict[str, Any]): 验证条件（如数据架构、性能指标）

        Returns:
            bool: 是否通过验证
        """
        _, metadata = self.load_model(model_name, version)

        # 数据架构验证
        if 'data_schema' in checks:
            stored_schema = metadata.get('data_schema', {})
            if not self._validate_schema(stored_schema, checks['data_schema']):
                return False

        # 验证特征列顺序
        required_features = checks.get('feature_columns', [])
        if required_features:
            stored_features = metadata.get('feature_columns', [])
            if stored_features != required_features:
                logger.warning("特征列顺序不匹配，可能影响模型性能")
                return False

        # 验证性能指标
        if 'metrics' in checks:
            model_metrics = metadata.get('metrics', {})
            for metric, threshold in checks['metrics'].items():
                if model_metrics.get(metric, 0) < threshold:
                    return False

        return True

    def _validate_schema(self, stored: Dict, current: Dict) -> bool:
        """验证数据架构兼容性"""
        return stored.get('feature_columns', []) == current.get('feature_columns', [])

    def _get_model_path(self, model_name: str, version: str) -> Path:
        """生成模型文件路径"""
        return self.base_path / f"{model_name}_v{version}.pkl"

    def _get_metadata_path(self, model_name: str, version: str) -> Path:
        """生成元数据文件路径"""
        return self.metadata_store / f"{model_name}_metadata_v{version}.pkl"

    def explain_model(self, model, features: pd.DataFrame):
        """解释模型"""
        self.explainer = ModelExplainer(model, features.columns)
        shap_values = self.explainer.explain_with_shap(features)
        feature_importance = self.explainer.get_feature_importance(features)
        return shap_values, feature_importance

    def detect_drift(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict:
        """检测特征漂移"""
        drift_report = {}
        for feature in train_data.columns:
            psi = self.drift_detector.calculate_psi(train_data[feature], test_data[feature])
            drift_report[feature] = psi
        return {"feature_drift": drift_report}

    def monitor_model(self, model, validation_data, metrics):
        """监控模型性能"""
        self.monitor = ModelMonitor(model, validation_data, metrics)
        self.monitor.monitor_performance()

    def _load_metadata(self, model_name: str, version: str) -> Dict:
        """加载模型元数据"""
        metadata_path = self.metadata_store / f"{model_name}_v{version}.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        return joblib.load(metadata_path)
