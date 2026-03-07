# src / features / intelligent / intelligent_enhancement_manager.py
"""
智能化增强功能管理器
整合自动特征选择、智能告警和机器学习模型集成功能
"""

import logging
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# 使用统一基础设施集成层
try:
    from src.core.integration import get_features_layer_adapter
    _features_adapter = get_features_layer_adapter()
    logger = logging.getLogger(__name__)
except ImportError:
    # 降级到直接导入
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
from ..core.config_integration import get_config_integration_manager, ConfigScope
from .auto_feature_selector import AutoFeatureSelector
from .smart_alert_system import SmartAlertSystem, AlertRule
from .ml_model_integration import MLModelIntegration

logger = get_unified_logger(__name__)


class IntelligentEnhancementManager:

    """智能化增强功能管理器"""

    def __init__(


        self,
        config_manager=None,
        enable_auto_feature_selection: bool = True,
        enable_smart_alerts: bool = True,
        enable_ml_integration: bool = True
    ):
        """
        初始化智能化增强功能管理器

        Args:
            config_manager: 配置管理器
            enable_auto_feature_selection: 是否启用自动特征选择
            enable_smart_alerts: 是否启用智能告警
            enable_ml_integration: 是否启用机器学习集成
        """
        # 配置管理集成
        self.config_manager = config_manager or get_config_integration_manager()
        self.config_manager.register_config_watcher(ConfigScope.PROCESSING, self._on_config_change)

        # 功能开关
        self.enable_auto_feature_selection = enable_auto_feature_selection
        self.enable_smart_alerts = enable_smart_alerts
        self.enable_ml_integration = enable_ml_integration

        # 初始化组件
        self.auto_feature_selector: Optional[AutoFeatureSelector] = None
        self.smart_alert_system: Optional[SmartAlertSystem] = None
        self.ml_model_integration: Optional[MLModelIntegration] = None

        # 状态跟踪
        self.enhancement_history: List[Dict[str, Any]] = []
        self.current_features: Optional[List[str]] = None
        self.current_alerts: List[Dict[str, Any]] = []
        self.model_performance: Dict[str, Any] = {}

        # 初始化组件
        self._init_components()

        logger.info("智能化增强功能管理器初始化完成")

    def _on_config_change(self, scope: ConfigScope, key: str, value: Any) -> None:
        """配置变更处理"""
        if scope == ConfigScope.PROCESSING:
            if key == "enable_auto_feature_selection":
                self.enable_auto_feature_selection = value
                logger.info(f"更新自动特征选择状态: {value}")
            elif key == "enable_smart_alerts":
                self.enable_smart_alerts = value
                logger.info(f"更新智能告警状态: {value}")
            elif key == "enable_ml_integration":
                self.enable_ml_integration = value
                logger.info(f"更新机器学习集成状态: {value}")

    def _init_components(self) -> None:
        """初始化各个组件"""
        # 初始化自动特征选择器
        if self.enable_auto_feature_selection:
            self.auto_feature_selector = AutoFeatureSelector(
                strategy="auto",
                task_type="classification",
                config_manager=self.config_manager
            )
            logger.info("自动特征选择器初始化完成")

        # 初始化智能告警系统
        if self.enable_smart_alerts:
            self.smart_alert_system = SmartAlertSystem(
                config_manager=self.config_manager,
                enable_adaptive_thresholds=True,
                enable_trend_analysis=True,
                enable_anomaly_detection=True
            )
            # 创建默认告警规则
            self.smart_alert_system.create_default_rules()
            logger.info("智能告警系统初始化完成")

        # 初始化机器学习模型集成
        if self.enable_ml_integration:
            self.ml_model_integration = MLModelIntegration(
                task_type="classification",
                ensemble_method="voting",
                enable_auto_tuning=True,
                config_manager=self.config_manager
            )
            logger.info("机器学习模型集成初始化完成")

    def enhance_features(


        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_features: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        智能化特征增强

        Args:
            X: 原始特征数据
            y: 目标变量
            target_features: 目标特征数量

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (增强后的特征数据, 增强信息)
        """
        enhancement_info = {
            "timestamp": datetime.now().isoformat(),
            "original_features": list(X.columns),
            "original_shape": X.shape,
            "enhancement_steps": []
        }

        # 1. 自动特征选择
        if self.enable_auto_feature_selection and self.auto_feature_selector:
            try:
                logger.info("开始自动特征选择")
                X_selected, selected_features, selection_info = self.auto_feature_selector.select_features(
                    X, y, target_features
                )

                enhancement_info["feature_selection"] = {
                    "selected_features": selected_features,
                    "selection_info": selection_info,
                    "reduction_ratio": 1 - len(selected_features) / len(X.columns)
                }

                X = X_selected
                self.current_features = selected_features
                enhancement_info["enhancement_steps"].append("auto_feature_selection")

                logger.info(f"自动特征选择完成，选择了 {len(selected_features)} 个特征")

            except Exception as e:
                logger.error(f"自动特征选择失败: {e}")
                enhancement_info["feature_selection"] = {"error": str(e)}

        # 2. 智能告警检查
        if self.enable_smart_alerts and self.smart_alert_system:
            try:
                logger.info("开始智能告警检查")
                alerts = self._check_feature_alerts(X, y)
                enhancement_info["alerts"] = alerts
                self.current_alerts.extend(alerts)

                if alerts:
                    logger.info(f"检测到 {len(alerts)} 个告警")
                else:
                    logger.info("未检测到告警")

            except Exception as e:
                logger.error(f"智能告警检查失败: {e}")
                enhancement_info["alerts"] = {"error": str(e)}

        # 3. 机器学习模型集成
        if self.enable_ml_integration and self.ml_model_integration:
            try:
                logger.info("开始机器学习模型集成")
                performance = self.ml_model_integration.train_models(X, y)
                enhancement_info["ml_integration"] = {
                    "performance": performance,
                    "best_model": self.ml_model_integration._get_best_model()
                }
                self.model_performance = performance
                enhancement_info["enhancement_steps"].append("ml_integration")

                logger.info("机器学习模型集成完成")

            except Exception as e:
                logger.error(f"机器学习模型集成失败: {e}")
                enhancement_info["ml_integration"] = {"error": str(e)}

        # 记录增强历史
        self.enhancement_history.append(enhancement_info)

        return X, enhancement_info

    def _check_feature_alerts(self, X: pd.DataFrame, y: pd.Series) -> List[Dict[str, Any]]:
        """检查特征相关告警"""
        alerts = []

        # 检查特征数量
        if self.smart_alert_system:
            alerts.extend(
                self.smart_alert_system.check_metric("feature_count", len(X.columns))
            )

        # 检查数据质量
        if not X.empty:
            # 检查缺失值比例
            missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
            alerts.extend(
                self.smart_alert_system.check_metric("missing_ratio", missing_ratio)
            )

            # 检查特征方差
            variance_ratio = (X.var() < 0.01).sum() / len(X.columns)
            alerts.extend(
                self.smart_alert_system.check_metric("low_variance_ratio", variance_ratio)
            )

        # 检查目标变量分布
        if not y.empty:
            if y.dtype in ['object', 'category']:
                # 分类任务

                class_counts = y.value_counts()
                imbalance_ratio = class_counts.min() / class_counts.max()
                alerts.extend(
                    self.smart_alert_system.check_metric("class_imbalance", 1 - imbalance_ratio)
                )
            else:
                # 回归任务
                y_std = y.std()
                if y_std < 0.1:
                    alerts.extend(
                        self.smart_alert_system.check_metric("low_target_variance", y_std)
                    )

        return [alert.__dict__ for alert in alerts]

    def predict_with_enhanced_model(


        self,
        X: pd.DataFrame,
        use_ensemble: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用增强模型进行预测

        Args:
            X: 特征数据
            use_ensemble: 是否使用集成模型

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (预测结果, 预测信息)
        """
        prediction_info = {
            "timestamp": datetime.now().isoformat(),
            "input_shape": X.shape,
            "prediction_method": "ensemble" if use_ensemble else "single_model"
        }

        if not self.ml_model_integration:
            raise ValueError("机器学习模型集成未启用")

        try:
            # 特征选择（如果已训练）
            if self.current_features and self.auto_feature_selector:
                X = X[self.current_features]
                prediction_info["feature_selection_applied"] = True
                prediction_info["selected_features"] = self.current_features
            else:
                prediction_info["feature_selection_applied"] = False

            # 进行预测
            predictions = self.ml_model_integration.predict(X, use_ensemble)
            prediction_info["predictions_shape"] = predictions.shape

            # 添加模型性能信息
            if self.model_performance:
                prediction_info["model_performance"] = self.model_performance

            logger.info(f"预测完成，结果形状: {predictions.shape}")

        except Exception as e:
            logger.error(f"预测失败: {e}")
            prediction_info["error"] = str(e)
            raise

        return predictions, prediction_info

    def get_enhancement_summary(self) -> Dict[str, Any]:
        """获取增强功能摘要"""
        summary = {
            "enable_auto_feature_selection": self.enable_auto_feature_selection,
            "enable_smart_alerts": self.enable_smart_alerts,
            "enable_ml_integration": self.enable_ml_integration,
            "current_features": self.current_features,
            "current_alerts_count": len(self.current_alerts),
            "enhancement_history_count": len(self.enhancement_history),
            "model_performance": self.model_performance
        }

        # 添加各组件状态
        if self.auto_feature_selector:
            summary["auto_feature_selector"] = {
                "is_fitted": self.auto_feature_selector.is_fitted,
                "strategy": self.auto_feature_selector.strategy,
                "task_type": self.auto_feature_selector.task_type
            }

        if self.smart_alert_system:
            summary["smart_alert_system"] = {
                "active_rules": len([r for r in self.smart_alert_system.rules.values() if r.enabled]),
                "total_alerts": len(self.smart_alert_system.alerts),
                "statistics": self.smart_alert_system.get_alert_statistics()
            }

        if self.ml_model_integration:
            summary["ml_integration"] = self.ml_model_integration.get_model_summary()

        return summary

    def save_enhancement_state(self, filepath: Union[str, Path]) -> None:
        """保存增强状态"""
        state = {
            "current_features": self.current_features,
            "current_alerts": self.current_alerts,
            "model_performance": self.model_performance,
            "enhancement_history": self.enhancement_history,
            "config": {
                "enable_auto_feature_selection": self.enable_auto_feature_selection,
                "enable_smart_alerts": self.enable_smart_alerts,
                "enable_ml_integration": self.enable_ml_integration
            }
        }

        with open(filepath, 'w', encoding='utf - 8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"增强状态已保存到: {filepath}")

    def load_enhancement_state(self, filepath: Union[str, Path]) -> None:
        """加载增强状态"""
        with open(filepath, 'r', encoding='utf - 8') as f:
            state = json.load(f)

        self.current_features = state.get("current_features")
        self.current_alerts = state.get("current_alerts", [])
        self.model_performance = state.get("model_performance", {})
        self.enhancement_history = state.get("enhancement_history", [])

        config = state.get("config", {})
        self.enable_auto_feature_selection = config.get("enable_auto_feature_selection", True)
        self.enable_smart_alerts = config.get("enable_smart_alerts", True)
        self.enable_ml_integration = config.get("enable_ml_integration", True)

        logger.info(f"增强状态已从 {filepath} 加载")

    def add_custom_alert_rule(self, rule: AlertRule) -> None:
        """添加自定义告警规则"""
        if self.smart_alert_system:
            self.smart_alert_system.add_rule(rule)
            logger.info(f"添加自定义告警规则: {rule.name}")

    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取最近的告警"""
        if not self.smart_alert_system:
            return []

        cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
        recent_alerts = self.smart_alert_system.get_alerts(start_time=cutoff_time)
        return [alert.__dict__ for alert in recent_alerts]

    def export_enhancement_report(self, filepath: Union[str, Path]) -> None:
        """导出增强功能报告"""
        report = {
            "summary": self.get_enhancement_summary(),
            "recent_alerts": self.get_recent_alerts(),
            "enhancement_history": self.enhancement_history[-10:],  # 最近10次
            "generated_at": datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf - 8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"增强功能报告已导出到: {filepath}")

    def reset_enhancement_state(self) -> None:
        """重置增强状态"""
        self.current_features = None
        self.current_alerts = []
        self.model_performance = {}
        self.enhancement_history = []

        # 重置各组件
        if self.auto_feature_selector:
            self.auto_feature_selector = AutoFeatureSelector(
                strategy="auto",
                task_type="classification",
                config_manager=self.config_manager
            )

        if self.smart_alert_system:
            self.smart_alert_system = SmartAlertSystem(
                config_manager=self.config_manager
            )
            self.smart_alert_system.create_default_rules()

        if self.ml_model_integration:
            self.ml_model_integration = MLModelIntegration(
                task_type="classification",
                ensemble_method="voting",
                enable_auto_tuning=True,
                config_manager=self.config_manager
            )

        logger.info("增强状态已重置")
