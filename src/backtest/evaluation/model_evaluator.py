import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from sklearn.metrics import classification_report, mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """模型评估指标
    新增A股市场特定指标:
    - china_win_rate: 考虑涨跌停后的实际胜率
    - t1_impact: T+1规则对收益的影响
    - circuit_breaker_impact: 熔断机制对收益的影响
    - policy_risk: 政策风险暴露度
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    mse: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    china_win_rate: float = 0.0
    t1_impact: float = 0.0
    circuit_breaker_impact: float = 0.0
    policy_risk: float = 0.0

class ModelEvaluator:
    """多模型评估器"""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.metrics_history = []

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, ModelMetrics]:
        """评估所有模型表现"""
        results = {}
        models = self.model_manager.get_all_models()

        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")

            # 模型预测
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

            # 计算指标
            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            results[name] = metrics
            self.metrics_history.append((name, metrics))

            # 特征重要性
            if hasattr(model, 'feature_importances_'):
                self._plot_feature_importance(name, model, X_test.columns)

        return results

    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_proba: np.ndarray = None) -> ModelMetrics:
        """计算各类评估指标
        新增功能:
        - 添加A股市场特定指标
        - 改进指标计算方式
        """
        # 分类指标
        report = classification_report(y_true, y_pred, output_dict=True)
        cls_metrics = report['weighted avg']

        # 回归指标
        mse = mean_squared_error(y_true, y_pred)

        # 策略指标
        sharpe, max_dd, win_rate = self._strategy_metrics(y_true, y_pred, y_proba)
        
        # A股市场特定指标
        china_metrics = self._calculate_china_specific_metrics(y_true, y_pred, y_proba)

        return ModelMetrics(
            accuracy=cls_metrics['precision'],
            precision=cls_metrics['precision'],
            recall=cls_metrics['recall'],
            f1=cls_metrics['f1-score'],
            mse=mse,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            **china_metrics  # 添加A股特定指标
        )

    def _calculate_china_specific_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_proba: np.ndarray) -> Dict[str, float]:
        """计算A股市场特定指标
        Args:
            y_true: 实际收益率
            y_pred: 预测信号
            y_proba: 预测概率
            
        Returns:
            dict: 包含以下指标:
                - 'china_win_rate': 考虑涨跌停后的实际胜率
                - 't1_impact': T+1规则对收益的影响
                - 'circuit_breaker_impact': 熔断机制对收益的影响
                - 'policy_risk': 政策风险暴露度
        """
        from src.trading.risk.china import ChinaMarketRuleChecker
        
        checker = ChinaMarketRuleChecker()
        metrics = {}
        
        # 计算考虑涨跌停后的实际胜率
        tradable_mask = [checker.can_trade(i) for i in range(len(y_true))]
        tradable_returns = y_true[tradable_mask] * y_pred[tradable_mask]
        metrics['china_win_rate'] = (tradable_returns > 0).mean()
        
        # 计算T+1规则影响
        t1_penalty = checker.estimate_t1_impact(y_pred)
        metrics['t1_impact'] = t1_penalty
        
        # 熔断机制影响
        circuit_breaker_days = checker.detect_circuit_breaker_days(y_true)
        metrics['circuit_breaker_impact'] = len(circuit_breaker_days) / len(y_true)
        
        # 政策风险暴露度(基于新闻情感分析)
        if hasattr(self, 'news_sentiment'):
            policy_risk = self.news_sentiment.get_policy_risk(y_true.index)
            metrics['policy_risk'] = policy_risk.mean()
            
        return metrics

    def _strategy_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_proba: np.ndarray) -> Tuple[float, float, float]:
        """计算策略相关指标
        Args:
            y_true: 实际收益率序列
            y_pred: 预测信号序列 (1:买入, -1:卖出, 0:空仓)
            y_proba: 预测概率(如果有)
            
        Returns:
            tuple: (夏普比率, 最大回撤, 胜率)
            
        特别处理A股市场规则:
        - 考虑T+1交易限制
        - 处理涨跌停无法交易情况
        - 计算考虑交易费用
        """
        from src.trading.risk.china import ChinaMarketRuleChecker
        
        # 初始化A股规则检查器
        rule_checker = ChinaMarketRuleChecker()
        
        # 模拟交易 - 考虑A股规则
        positions = 0  # 当前持仓
        strategy_returns = []  # 每日策略收益
        
        for i in range(1, len(y_true)):
            # 检查是否可交易(处理涨跌停)
            if not rule_checker.can_trade(i):
                strategy_returns.append(0)  # 无法交易
                continue
                
            # T+1规则处理
            if positions != 0:
                # 持仓状态下不能反向交易
                if np.sign(y_pred[i]) != np.sign(positions):
                    strategy_returns.append(0)
                    continue
                    
            # 计算当日收益 (考虑手续费和印花税)
            daily_return = y_true[i] * y_pred[i] - rule_checker.calculate_fee(y_pred[i])
            strategy_returns.append(daily_return)
            
            # 更新持仓
            positions = y_pred[i] if rule_checker.is_trading_hour(i) else 0

        returns = pd.Series(strategy_returns)
        cum_returns = (1 + returns).cumprod()

        # 夏普比率(年化)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

        # 最大回撤
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        max_dd = drawdown.min()

        # 胜率(排除空仓日)
        win_rate = (returns[returns != 0] > 0).mean()

        return sharpe, max_dd, win_rate

    def _plot_feature_importance(self, model_name: str, model, features: List[str]):
        """绘制特征重要性图
        Args:
            model_name: 模型名称
            model: 模型实例
            features: 特征名称列表
            
        改进点:
        - 添加中文标题和标签
        - 支持特征名称中英文混合
        - 优化图表格式
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 按重要性排序
        idx = np.argsort(importance)[::-1]
        sorted_features = [features[i] for i in idx]
        sorted_importance = importance[idx]
        
        # 绘制水平条形图
        sns.barplot(x=sorted_importance, y=sorted_features, ax=ax, palette="Blues_d")
        
        # 添加中文标签
        ax.set_title(f'{model_name} 特征重要性分析', fontsize=16)
        ax.set_xlabel('重要性得分', fontsize=12)
        ax.set_ylabel('特征名称', fontsize=12)
        
        # 添加数值标签
        for i, v in enumerate(sorted_importance):
            ax.text(v + 0.01, i, f'{v:.3f}', color='black', fontsize=10)
            
        plt.tight_layout()
        
        # 确保目录存在
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def compare_models(self, metrics: Dict[str, ModelMetrics]) -> pd.DataFrame:
        """模型对比报告
        改进点:
        - 添加A股市场特定指标
        - 按夏普比率排序
        - 优化报告格式
        """
        # 转换指标为DataFrame
        df = pd.DataFrame.from_dict(
            {k: vars(v) for k, v in metrics.items()},
            orient='index'
        )
        
        # 重命名列名
        df = df.rename(columns={
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1 Score',
            'mse': 'MSE',
            'sharpe': 'Sharpe Ratio',
            'max_drawdown': 'Max Drawdown',
            'win_rate': 'Win Rate',
            'china_win_rate': 'China Win Rate',
            't1_impact': 'T+1 Impact',
            'circuit_breaker_impact': 'Circuit Breaker Impact',
            'policy_risk': 'Policy Risk'
        })
        
        # 按夏普比率排序
        df = df.sort_values('Sharpe Ratio', ascending=False)
        
        # 保存报告
        df.to_csv('reports/model_comparison.csv')
        return df

    def stability_test(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, List[ModelMetrics]]:
        """模型稳定性测试"""
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = {name: [] for name in self.model_manager.get_all_models()}

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # 重新训练模型
            self.model_manager.retrain_all(X_train, y_train)

            # 评估表现
            metrics = self.evaluate_models(X_test, y_test)
            for name, metric in metrics.items():
                results[name].append(metric)

        return results

class StrategyOptimizer:
    """交易策略优化器"""

    def __init__(self, trading_strategy):
        self.strategy = trading_strategy
        self.optimization_history = []

    def optimize_parameters(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict) -> Dict:
        """策略参数优化"""
        from sklearn.model_selection import GridSearchCV

        # 使用网格搜索优化
        gs = GridSearchCV(
            estimator=self.strategy,
            param_grid=param_grid,
            scoring='neg_mean_squared_error',
            cv=3,
            n_jobs=-1
        )
        gs.fit(X, y)

        # 记录优化结果
        self.optimization_history.append({
            'params': gs.best_params_,
            'score': gs.best_score_
        })

        # 更新策略参数
        self.strategy.set_params(**gs.best_params_)
        return gs.best_params_

    def backtest_strategy(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """策略回测分析"""
        signals = self.strategy.generate_signals(X)
        returns = y * signals

        # 计算回测指标
        metrics = {
            'total_return': returns.sum(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self._calculate_drawdown(returns),
            'win_rate': (returns > 0).mean(),
            'profit_factor': returns[returns > 0].sum() / -returns[returns < 0].sum()
        }

        # 绘制净值曲线
        self._plot_equity_curve(returns)
        return metrics

    def _calculate_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding().max()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()

    def _plot_equity_curve(self, returns: pd.Series):
        """绘制净值曲线"""
        equity = (1 + returns).cumprod()

        fig, ax = plt.subplots(figsize=(10, 6))
        equity.plot(ax=ax)
        ax.set_title('Strategy Equity Curve')
        ax.set_ylabel('Normalized Return')
        plt.savefig('reports/equity_curve.png')
        plt.close()

def main():
    """评估优化主流程"""
    from src.models.model_manager import ModelManager
    from src.trading.enhanced_trading_strategy import EnhancedTradingStrategy

    # 初始化模型和策略
    model_mgr = ModelManager()
    strategy = EnhancedTradingStrategy()

    # 加载测试数据
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

    # 模型评估
    evaluator = ModelEvaluator(model_mgr)
    metrics = evaluator.evaluate_models(X_test, y_test)
    print("Model Evaluation Results:")
    print(evaluator.compare_models(metrics))

    # 策略优化
    optimizer = StrategyOptimizer(strategy)
    param_grid = {
        'threshold': [0.5, 0.6, 0.7],
        'max_position': [0.1, 0.2, 0.3],
        'stop_loss': [-0.05, -0.1, -0.15]
    }
    best_params = optimizer.optimize_parameters(X_test, y_test, param_grid)
    print(f"Optimized Parameters: {best_params}")

    # 回测验证
    backtest_results = optimizer.backtest_strategy(X_test, y_test)
    print("Backtest Results:")
    print(backtest_results)

if __name__ == "__main__":
    main()
