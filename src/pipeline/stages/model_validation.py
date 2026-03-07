"""
模型验证阶段模块

负责A/B测试设置、影子验证和业务规则验证
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .base import PipelineStage
from ..exceptions import StageExecutionException, StageValidationException
from ..config import StageConfig


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    ab_test_result: Optional[Dict[str, Any]] = None
    shadow_test_result: Optional[Dict[str, Any]] = None
    business_rules_passed: bool = True
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "ab_test_result": self.ab_test_result,
            "shadow_test_result": self.shadow_test_result,
            "business_rules_passed": self.business_rules_passed,
            "validation_timestamp": self.validation_timestamp
        }


class ModelValidationStage(PipelineStage):
    """
    模型验证阶段
    
    功能：
    - A/B测试设置和对比
    - 影子验证（Shadow Mode）
    - 业务规则验证
    - 通过/失败判定
    """
    
    def __init__(self, config: Optional[StageConfig] = None):
        super().__init__("model_validation", config)
        self._validation_result: Optional[ValidationResult] = None
        self._baseline_model: Optional[Any] = None
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行模型验证
        
        Args:
            context: 包含model, features, evaluation_metrics的上下文
            
        Returns:
            包含validation_result的输出
        """
        self.logger.info("开始模型验证阶段")
        
        # 获取输入
        model = context.get("model")
        evaluation_passed = context.get("evaluation_passed", False)
        
        if not evaluation_passed:
            self.logger.warning("模型评估未通过，跳过验证")
            return {
                "validation_result": ValidationResult(
                    passed=False,
                    business_rules_passed=False
                ).to_dict()
            }
        
        # 获取配置
        run_ab_test = self.config.config.get("ab_test", True)
        run_shadow = self.config.config.get("shadow_mode", True)
        validate_business_rules = self.config.config.get("validate_business_rules", True)
        
        ab_result = None
        shadow_result = None
        business_passed = True
        
        # 1. A/B测试
        if run_ab_test:
            self.logger.info("执行A/B测试")
            ab_result = self._run_ab_test(model, context)
        
        # 2. 影子验证
        if run_shadow:
            self.logger.info("执行影子验证")
            shadow_result = self._run_shadow_validation(model, context)
        
        # 3. 业务规则验证
        if validate_business_rules:
            self.logger.info("执行业务规则验证")
            business_passed = self._validate_business_rules(model, context)
        
        # 4. 综合判定
        passed = self._evaluate_validation_result(ab_result, shadow_result, business_passed)
        
        self._validation_result = ValidationResult(
            passed=passed,
            ab_test_result=ab_result,
            shadow_test_result=shadow_result,
            business_rules_passed=business_passed
        )
        
        self.logger.info(f"模型验证完成，结果: {'通过' if passed else '未通过'}")
        
        return {
            "validation_result": self._validation_result.to_dict(),
            "validation_passed": passed
        }
    
    def _run_ab_test(self, model: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行A/B测试"""
        # 获取基线模型（上一个版本）
        baseline_model = self._load_baseline_model(context)
        
        if baseline_model is None:
            self.logger.warning("未找到基线模型，跳过A/B测试")
            return {"skipped": True, "reason": "no_baseline_model"}
        
        features_df = context.get("features")
        if features_df is None:
            return {"error": "缺少features数据"}
        
        # 获取特征列
        feature_cols = context.get("feature_columns", [])
        if not feature_cols:
            exclude_cols = ["timestamp", "target", "open", "high", "low", "close", "volume"]
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # 使用最近7天数据进行A/B测试
        test_days = self.config.config.get("ab_test_days", 7)
        test_df = features_df.tail(test_days * 24 * 60)  # 假设分钟级数据
        
        if len(test_df) == 0:
            return {"error": "测试数据不足"}
        
        X_test = test_df[feature_cols].dropna()
        
        # 两个模型分别预测
        new_predictions = model.predict(X_test)
        baseline_predictions = baseline_model.predict(X_test)
        
        # 计算预测一致性
        agreement = (new_predictions == baseline_predictions).mean()
        
        # 如果新模型有概率输出，计算置信度差异
        new_confidence = None
        baseline_confidence = None
        if hasattr(model, "predict_proba") and hasattr(baseline_model, "predict_proba"):
            new_proba = model.predict_proba(X_test)
            baseline_proba = baseline_model.predict_proba(X_test)
            new_confidence = np.max(new_proba, axis=1).mean()
            baseline_confidence = np.max(baseline_proba, axis=1).mean()
        
        result = {
            "test_samples": len(X_test),
            "prediction_agreement": float(agreement),
            "new_model_confidence": float(new_confidence) if new_confidence else None,
            "baseline_confidence": float(baseline_confidence) if baseline_confidence else None
        }
        
        # 判定标准：预测一致性应大于70%
        result["passed"] = agreement >= 0.7
        
        self.logger.info(f"A/B测试 - 预测一致性: {agreement:.2%}, 通过: {result['passed']}")
        
        return result
    
    def _run_shadow_validation(self, model: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行影子验证"""
        features_df = context.get("features")
        if features_df is None:
            return {"error": "缺少features数据"}
        
        # 获取特征列
        feature_cols = context.get("feature_columns", [])
        if not feature_cols:
            exclude_cols = ["timestamp", "target", "open", "high", "low", "close", "volume"]
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # 使用影子数据（模拟生产环境数据）
        shadow_size = self.config.config.get("shadow_test_size", 1000)
        shadow_df = features_df.tail(shadow_size)
        
        X_shadow = shadow_df[feature_cols].dropna()
        
        # 执行预测
        predictions = model.predict(X_shadow)
        
        # 分析预测分布
        unique, counts = np.unique(predictions, return_counts=True)
        prediction_distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        # 计算预测延迟
        import time
        start_time = time.time()
        _ = model.predict(X_shadow.head(100))
        inference_time = time.time() - start_time
        avg_latency = inference_time / 100 * 1000  # 转换为毫秒
        
        result = {
            "shadow_samples": len(X_shadow),
            "prediction_distribution": prediction_distribution,
            "avg_inference_latency_ms": round(avg_latency, 2),
            "predictions_per_second": round(100 / inference_time, 2)
        }
        
        # 判定标准：延迟应小于100ms
        result["passed"] = avg_latency < 100
        
        self.logger.info(f"影子验证 - 平均延迟: {avg_latency:.2f}ms, 通过: {result['passed']}")
        
        return result
    
    def _validate_business_rules(self, model: Any, context: Dict[str, Any]) -> bool:
        """验证业务规则"""
        evaluation_report = context.get("evaluation_report", {})
        backtest_metrics = evaluation_report.get("backtest_metrics", {})
        
        # 业务规则检查
        rules = []
        
        # 规则1: 最大回撤不能超过20%
        max_dd = backtest_metrics.get("max_drawdown", 0)
        rules.append(("max_drawdown", max_dd <= 0.2, max_dd))
        
        # 规则2: Sharpe比率应大于0.5
        sharpe = backtest_metrics.get("sharpe_ratio", 0)
        rules.append(("sharpe_ratio", sharpe >= 0.5, sharpe))
        
        # 规则3: 胜率应大于40%
        win_rate = backtest_metrics.get("win_rate", 0)
        rules.append(("win_rate", win_rate >= 0.4, win_rate))
        
        # 规则4: 总交易次数应大于10次
        total_trades = backtest_metrics.get("total_trades", 0)
        rules.append(("min_trades", total_trades >= 10, total_trades))
        
        all_passed = all(passed for _, passed, _ in rules)
        
        for rule_name, passed, value in rules:
            status = "通过" if passed else "未通过"
            self.logger.info(f"业务规则 {rule_name}: {value:.4f} - {status}")
        
        return all_passed
    
    def _load_baseline_model(self, context: Dict[str, Any]) -> Optional[Any]:
        """加载基线模型"""
        # 从模型存储中加载上一个版本的模型
        model_dir = context.get("model_dir", "models")
        try:
            import joblib
            from pathlib import Path
            
            # 查找最新的模型文件（排除当前模型）
            model_files = list(Path(model_dir).glob("model_*.joblib"))
            if len(model_files) >= 2:
                # 返回倒数第二个模型作为基线
                baseline_path = sorted(model_files)[-2]
                return joblib.load(baseline_path)
            
        except Exception as e:
            self.logger.warning(f"加载基线模型失败: {e}")
        
        return None
    
    def _evaluate_validation_result(
        self,
        ab_result: Optional[Dict[str, Any]],
        shadow_result: Optional[Dict[str, Any]],
        business_passed: bool
    ) -> bool:
        """评估验证结果"""
        # 业务规则必须通过
        if not business_passed:
            return False
        
        # A/B测试检查
        if ab_result and not ab_result.get("skipped", False):
            if not ab_result.get("passed", False):
                return False
        
        # 影子验证检查
        if shadow_result:
            if not shadow_result.get("passed", False):
                return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取阶段指标"""
        if self._validation_result is None:
            return {}
        
        return {
            "validation_passed": self._validation_result.passed,
            "ab_test_passed": self._validation_result.ab_test_result.get("passed") if self._validation_result.ab_test_result else None,
            "shadow_test_passed": self._validation_result.shadow_test_result.get("passed") if self._validation_result.shadow_test_result else None,
            "business_rules_passed": self._validation_result.business_rules_passed
        }
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """回滚模型验证阶段"""
        self.logger.info("回滚模型验证阶段")
        self._validation_result = None
        self._baseline_model = None
        return True
