"""
模型训练阶段模块

负责模型训练、超参数搜索和交叉验证
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path

from .base import PipelineStage
from ..exceptions import ModelTrainingException, StageExecutionException
from ..config import StageConfig


class ModelTrainingStage(PipelineStage):
    """
    模型训练阶段
    
    功能：
    - 数据准备和分割
    - 超参数搜索
    - 模型训练
    - 交叉验证
    - 模型保存
    """
    
    def __init__(self, config: Optional[StageConfig] = None):
        super().__init__("model_training", config)
        self._model: Any = None
        self._training_metrics: Dict[str, Any] = {}
        self._model_info: Dict[str, Any] = {}
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行模型训练
        
        Args:
            context: 包含features的上下文
            
        Returns:
            包含model, model_path, training_metrics的输出
        """
        self.logger.info("开始模型训练阶段")
        
        # 获取输入数据
        features_df = context.get("features")
        if features_df is None:
            raise StageExecutionException(
                message="缺少features输入",
                stage_name=self.name
            )
        
        if isinstance(features_df, dict):
            df = pd.DataFrame(features_df)
        else:
            df = features_df.copy()
        
        self.logger.info(f"输入特征: {df.shape}")
        
        # 获取配置
        model_type = self.config.config.get("model_type", "xgboost")
        target_col = self.config.config.get("target_col", "target")
        hyperparameter_search = self.config.config.get("hyperparameter_search", False)
        
        # 1. 准备数据
        self.logger.info("准备训练数据")
        X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_data(df, target_col)
        
        # 2. 超参数搜索
        if hyperparameter_search:
            self.logger.info("执行超参数搜索")
            best_params = self._hyperparameter_search(X_train, y_train, model_type)
        else:
            best_params = self._get_default_params(model_type)
        
        # 3. 训练模型
        self.logger.info(f"训练 {model_type} 模型")
        model = self._train_model(X_train, y_train, model_type, best_params)
        self._model = model
        
        # 4. 评估训练结果
        self.logger.info("评估训练结果")
        train_metrics = self._evaluate_model(model, X_train, y_train)
        val_metrics = self._evaluate_model(model, X_val, y_val)
        test_metrics = self._evaluate_model(model, X_test, y_test) if X_test is not None else {}
        
        self._training_metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics
        }
        
        # 5. 保存模型
        model_path = self._save_model(model, context)
        
        # 6. 记录模型信息
        self._model_info = {
            "model_type": model_type,
            "model_path": model_path,
            "hyperparameters": best_params,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "feature_count": X_train.shape[1],
            "training_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"模型训练完成，验证集准确率: {val_metrics.get('accuracy', 'N/A')}")
        
        return {
            "model": model,
            "model_path": model_path,
            "model_info": self._model_info,
            "training_metrics": self._training_metrics,
            "feature_columns": list(X_train.columns)
        }
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], pd.Series, pd.Series, Optional[pd.Series]]:
        """
        准备训练数据
        
        Args:
            df: 特征数据框
            target_col: 目标列名
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # 确定特征列和目标列
        exclude_cols = ["timestamp", target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise StageExecutionException(
                message="没有可用的特征列",
                stage_name=self.name
            )
        
        # 创建目标变量（如果没有）
        if target_col not in df.columns:
            self.logger.info(f"创建目标变量: {target_col}")
            df = self._create_target(df, target_col)
        
        # 删除包含NaN的行
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # 时间序列分割
        train_size = int(len(df_clean) * 0.7)
        val_size = int(len(df_clean) * 0.15)
        
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        X_val = X.iloc[train_size:train_size + val_size]
        y_val = y.iloc[train_size:train_size + val_size]
        
        X_test = X.iloc[train_size + val_size:] if train_size + val_size < len(df_clean) else None
        y_test = y.iloc[train_size + val_size:] if train_size + val_size < len(df_clean) else None
        
        self.logger.info(f"数据分割: 训练{len(X_train)}, 验证{len(X_val)}, 测试{len(X_test) if X_test is not None else 0}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _create_target(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """创建目标变量（价格涨跌）"""
        df = df.copy()
        
        if "close" in df.columns:
            # 未来N期收益率
            future_return = df["close"].shift(-5) / df["close"] - 1
            df[target_col] = (future_return > 0).astype(int)
        else:
            raise StageExecutionException(
                message="无法创建目标变量：缺少close列",
                stage_name=self.name
            )
        
        return df
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """获取默认超参数"""
        if model_type == "xgboost":
            return {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }
        elif model_type == "lightgbm":
            return {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }
        elif model_type == "random_forest":
            return {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42
            }
        else:
            return {}
    
    def _hyperparameter_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str
    ) -> Dict[str, Any]:
        """超参数搜索"""
        try:
            from sklearn.model_selection import RandomizedSearchCV
            
            if model_type == "xgboost":
                from xgboost import XGBClassifier
                model = XGBClassifier()
                param_distributions = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0]
                }
            elif model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
                param_distributions = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            else:
                return self._get_default_params(model_type)
            
            search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=10,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                random_state=42
            )
            
            search.fit(X_train, y_train)
            
            self.logger.info(f"最佳参数: {search.best_params_}")
            self.logger.info(f"最佳分数: {search.best_score_:.4f}")
            
            return search.best_params_
            
        except ImportError as e:
            self.logger.warning(f"超参数搜索依赖缺失: {e}")
            return self._get_default_params(model_type)
        except Exception as e:
            self.logger.error(f"超参数搜索失败: {e}")
            return self._get_default_params(model_type)
    
    def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str,
        params: Dict[str, Any]
    ) -> Any:
        """训练模型"""
        try:
            if model_type == "xgboost":
                from xgboost import XGBClassifier
                model = XGBClassifier(**params)
            elif model_type == "lightgbm":
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(**params)
            elif model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params)
            elif model_type == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(**params, max_iter=1000)
            else:
                raise ModelTrainingException(
                    message=f"不支持的模型类型: {model_type}",
                    model_type=model_type
                )
            
            model.fit(X_train, y_train)
            return model
            
        except ImportError as e:
            self.logger.error(f"模型库导入失败: {e}")
            raise ModelTrainingException(
                message=f"模型训练失败: {e}",
                model_type=model_type,
                cause=e
            )
    
    def _evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """评估模型"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y, y_pred, average="binary", zero_division=0)
        }
        
        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y, y_prob)
            except:
                pass
        
        return metrics
    
    def _save_model(self, model: Any, context: Dict[str, Any]) -> str:
        """保存模型"""
        model_dir = context.get("model_dir", "models")
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{model_dir}/model_{timestamp}.joblib"
        
        joblib.dump(model, model_path)
        self.logger.info(f"模型已保存: {model_path}")
        
        return model_path
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取阶段指标"""
        return {
            "model_type": self._model_info.get("model_type"),
            "training_samples": self._model_info.get("training_samples"),
            "feature_count": self._model_info.get("feature_count"),
            "validation_accuracy": self._training_metrics.get("validation", {}).get("accuracy"),
            "training_accuracy": self._training_metrics.get("train", {}).get("accuracy")
        }
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """回滚模型训练阶段"""
        self.logger.info("回滚模型训练阶段")
        
        # 删除保存的模型文件
        if self._model_info.get("model_path"):
            try:
                Path(self._model_info["model_path"]).unlink(missing_ok=True)
            except Exception as e:
                self.logger.warning(f"删除模型文件失败: {e}")
        
        self._model = None
        self._training_metrics = {}
        self._model_info = {}
        return True
