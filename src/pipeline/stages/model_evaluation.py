"""
模型评估阶段模块

负责模型技术指标评估、业务指标评估和回测评估
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field

from .base import PipelineStage
from ..exceptions import StageExecutionException
from ..config import StageConfig


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: float
    equity_curve: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_trade_return": self.avg_trade_return
        }


class ModelEvaluationStage(PipelineStage):
    """
    模型评估阶段
    
    功能：
    - 技术指标评估（Accuracy, F1, ROC-AUC等）
    - 业务指标评估（Sharpe Ratio, Max Drawdown等）
    - 回测评估
    - 生成评估报告
    """
    
    def __init__(self, config: Optional[StageConfig] = None):
        super().__init__("model_evaluation", config)
        self._evaluation_metrics: Dict[str, Any] = {}
        self._backtest_result: Optional[BacktestResult] = None
        self._evaluation_report: Dict[str, Any] = {}
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行模型评估
        
        Args:
            context: 包含model, features, training_metrics的上下文
            
        Returns:
            包含evaluation_metrics, backtest_result, evaluation_report的输出
        """
        self.logger.info("开始模型评估阶段")
        
        # 获取输入
        model = context.get("model")
        features_df = context.get("features")
        training_metrics = context.get("training_metrics", {})
        
        if model is None:
            raise StageExecutionException(
                message="缺少model输入",
                stage_name=self.name
            )
        
        if features_df is None:
            raise StageExecutionException(
                message="缺少features输入",
                stage_name=self.name
            )
        
        # 获取配置
        metrics_to_evaluate = self.config.config.get("metrics", ["accuracy", "f1", "sharpe_ratio"])
        run_backtest = self.config.config.get("backtest", True)
        
        # 1. 技术指标评估
        self.logger.info("执行技术指标评估")
        technical_metrics = self._evaluate_technical_metrics(model, features_df, context)
        
        # 2. 回测评估（业务指标）
        backtest_result = None
        if run_backtest:
            self.logger.info("执行回测评估")
            backtest_result = self._run_backtest(model, features_df, context)
            self._backtest_result = backtest_result
        
        # 3. 综合评估
        self.logger.info("生成综合评估报告")
        evaluation_report = self._generate_evaluation_report(
            technical_metrics,
            backtest_result,
            training_metrics
        )
        
        self._evaluation_metrics = technical_metrics
        self._evaluation_report = evaluation_report
        
        # 4. 判断是否通过评估
        passed = self._evaluate_pass_criteria(technical_metrics, backtest_result)
        
        self.logger.info(f"模型评估完成，是否通过: {passed}")
        
        return {
            "evaluation_metrics": technical_metrics,
            "backtest_result": backtest_result.to_dict() if backtest_result else None,
            "evaluation_report": evaluation_report,
            "evaluation_passed": passed
        }
    
    def _evaluate_technical_metrics(
        self,
        model: Any,
        features_df: pd.DataFrame,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """评估技术指标"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, log_loss, confusion_matrix
        )
        
        # 准备测试数据
        feature_cols = context.get("feature_columns", [])
        if not feature_cols:
            # 自动识别特征列
            exclude_cols = ["timestamp", "target", "open", "high", "low", "close", "volume"]
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # 创建目标变量
        if "target" not in features_df.columns and "close" in features_df.columns:
            features_df = features_df.copy()
            future_return = features_df["close"].shift(-5) / features_df["close"] - 1
            features_df["target"] = (future_return > 0).astype(int)
        
        # 分割测试集（使用最后20%作为测试集）
        test_size = int(len(features_df) * 0.2)
        test_df = features_df.iloc[-test_size:].dropna()
        
        if len(test_df) == 0:
            self.logger.warning("测试集为空，使用训练集评估")
            test_df = features_df.dropna()
        
        X_test = test_df[feature_cols]
        y_test = test_df["target"]
        
        # 预测
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # 计算指标
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
                metrics["log_loss"] = log_loss(y_test, y_prob)
            except:
                pass
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["true_positive"] = int(cm[1, 1]) if cm.shape[0] > 1 else 0
        metrics["false_positive"] = int(cm[0, 1]) if cm.shape[0] > 1 else 0
        metrics["true_negative"] = int(cm[0, 0]) if cm.shape[0] > 0 else 0
        metrics["false_negative"] = int(cm[1, 0]) if cm.shape[0] > 1 else 0
        
        return metrics
    
    def _run_backtest(
        self,
        model: Any,
        features_df: pd.DataFrame,
        context: Dict[str, Any]
    ) -> BacktestResult:
        """执行回测"""
        # 准备数据
        feature_cols = context.get("feature_columns", [])
        if not feature_cols:
            exclude_cols = ["timestamp", "target", "open", "high", "low", "close", "volume"]
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # 使用最后30%数据作为回测数据
        backtest_size = int(len(features_df) * 0.3)
        backtest_df = features_df.iloc[-backtest_size:].copy()
        
        if "close" not in backtest_df.columns:
            raise StageExecutionException(
                message="回测需要close价格数据",
                stage_name=self.name
            )
        
        # 预测信号
        X = backtest_df[feature_cols].dropna()
        predictions = model.predict(X)
        
        # 对齐数据
        backtest_df = backtest_df.loc[X.index].copy()
        backtest_df["signal"] = predictions
        
        # 执行回测
        trades = []
        equity = [1.0]  # 初始资金
        position = 0  # 0: 空仓, 1: 多头
        entry_price = 0
        
        for i in range(len(backtest_df) - 1):
            current_price = backtest_df["close"].iloc[i]
            next_price = backtest_df["close"].iloc[i + 1]
            signal = backtest_df["signal"].iloc[i]
            
            # 信号变化时交易
            if signal == 1 and position == 0:
                # 买入
                position = 1
                entry_price = next_price
                trades.append({
                    "type": "buy",
                    "entry_time": backtest_df.index[i + 1],
                    "entry_price": entry_price
                })
            elif signal == 0 and position == 1:
                # 卖出
                exit_price = next_price
                trade_return = (exit_price - entry_price) / entry_price
                equity.append(equity[-1] * (1 + trade_return))
                
                trades[-1]["exit_time"] = backtest_df.index[i + 1]
                trades[-1]["exit_price"] = exit_price
                trades[-1]["return"] = trade_return
                
                position = 0
                entry_price = 0
            else:
                # 持仓不变，权益不变
                if len(equity) > 0:
                    equity.append(equity[-1])
        
        # 如果最后还有持仓，强制平仓
        if position == 1 and len(trades) > 0:
            exit_price = backtest_df["close"].iloc[-1]
            trade_return = (exit_price - entry_price) / entry_price
            equity[-1] = equity[-2] * (1 + trade_return) if len(equity) > 1 else 1 + trade_return
            
            trades[-1]["exit_time"] = backtest_df.index[-1]
            trades[-1]["exit_price"] = exit_price
            trades[-1]["return"] = trade_return
        
        # 计算回测指标
        returns = [t["return"] for t in trades if "return" in t]
        winning_trades = [r for r in returns if r > 0]
        losing_trades = [r for r in returns if r <= 0]
        
        total_return = equity[-1] - 1 if len(equity) > 0 else 0
        
        # 年化收益率
        days = (backtest_df.index[-1] - backtest_df.index[0]).days if len(backtest_df) > 1 else 1
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if total_return > -1 else -1
        
        # 最大回撤
        peak = 0
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe Ratio
        if len(returns) > 1:
            returns_series = pd.Series(returns)
            sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252) if returns_series.std() > 0 else 0
        else:
            sharpe = 0
        
        # Profit Factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=len(winning_trades) / len(returns) if len(returns) > 0 else 0,
            profit_factor=profit_factor,
            total_trades=len(returns),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_return=np.mean(returns) if len(returns) > 0 else 0,
            equity_curve=equity,
            trades=trades
        )
    
    def _generate_evaluation_report(
        self,
        technical_metrics: Dict[str, float],
        backtest_result: Optional[BacktestResult],
        training_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成评估报告"""
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "technical_metrics": technical_metrics,
            "training_metrics": training_metrics,
            "backtest_metrics": backtest_result.to_dict() if backtest_result else None,
            "overall_score": self._calculate_overall_score(technical_metrics, backtest_result)
        }
        
        return report
    
    def _calculate_overall_score(
        self,
        technical_metrics: Dict[str, float],
        backtest_result: Optional[BacktestResult]
    ) -> float:
        """计算综合评分"""
        scores = []
        
        # 技术指标评分
        if "accuracy" in technical_metrics:
            scores.append(technical_metrics["accuracy"] * 25)
        if "f1" in technical_metrics:
            scores.append(technical_metrics["f1"] * 25)
        
        # 业务指标评分
        if backtest_result:
            # Sharpe ratio评分
            sharpe_score = min(backtest_result.sharpe_ratio / 2 * 25, 25)
            scores.append(sharpe_score)
            
            # 最大回撤评分
            dd_score = max(0, (1 - backtest_result.max_drawdown) * 25)
            scores.append(dd_score)
        
        return np.mean(scores) if scores else 0
    
    def _evaluate_pass_criteria(
        self,
        technical_metrics: Dict[str, float],
        backtest_result: Optional[BacktestResult]
    ) -> bool:
        """评估是否通过标准"""
        # 获取通过标准
        min_accuracy = self.config.config.get("min_accuracy", 0.55)
        min_sharpe = self.config.config.get("min_sharpe", 0.5)
        max_drawdown = self.config.config.get("max_drawdown", 0.2)
        
        # 检查技术指标
        if technical_metrics.get("accuracy", 0) < min_accuracy:
            self.logger.warning(f"准确率 {technical_metrics.get('accuracy', 0):.4f} 低于阈值 {min_accuracy}")
            return False
        
        # 检查业务指标
        if backtest_result:
            if backtest_result.sharpe_ratio < min_sharpe:
                self.logger.warning(f"Sharpe比率 {backtest_result.sharpe_ratio:.4f} 低于阈值 {min_sharpe}")
                return False
            
            if backtest_result.max_drawdown > max_drawdown:
                self.logger.warning(f"最大回撤 {backtest_result.max_drawdown:.4f} 超过阈值 {max_drawdown}")
                return False
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取阶段指标"""
        return {
            "accuracy": self._evaluation_metrics.get("accuracy"),
            "f1_score": self._evaluation_metrics.get("f1"),
            "sharpe_ratio": self._backtest_result.sharpe_ratio if self._backtest_result else None,
            "max_drawdown": self._backtest_result.max_drawdown if self._backtest_result else None,
            "overall_score": self._evaluation_report.get("overall_score"),
            "evaluation_passed": self._evaluation_report.get("evaluation_passed")
        }
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """回滚模型评估阶段"""
        self.logger.info("回滚模型评估阶段")
        self._evaluation_metrics = {}
        self._backtest_result = None
        self._evaluation_report = {}
        return True
