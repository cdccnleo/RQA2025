import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import optuna
from typing import Dict, Optional
from pathlib import Path


class TuningVisualizer:

    """调参可视化工具"""

    @staticmethod
    def plot_optimization_history(trials: pd.DataFrame,


        metric_name: str = 'value',
        save_path: Optional[str] = None):
        """绘制优化历史曲线"""
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=trials, x='number', y=metric_name)
        plt.title('Optimization History')
        plt.xlabel('Trial')
        plt.ylabel(metric_name)

        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return plt.gcf()

    @staticmethod
    def plot_param_importance(importance: Dict[str, float],

                              save_path: Optional[str] = None):
        """绘制参数重要性"""
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(importance.values()),
        y=list(importance.keys()),
        orient='h')
        plt.title('Hyperparameter Importance')
        plt.xlabel('Importance')
        plt.ylabel('Parameter')

        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return plt.gcf()

    @staticmethod
    def plot_parallel_coordinate(study: optuna.Study,
                                params: list,
                                save_path: Optional[str] = None):
        """绘制平行坐标图"""
        plt.figure(figsize=(12, 8))
        fig = optuna.visualization.plot_parallel_coordinate(
            study,
            params=params
        )
        fig.show()

        if save_path:
            Path(save_path).parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return plt.gcf()
