from typing import Dict, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_squared_error
)

class ModelEvaluator:
    """模型评估器，支持分类和回归任务的评估"""

    def __init__(self, model: object, task_type: str = 'classification'):
        """
        初始化评估器

        Args:
            model: 待评估的模型对象
            task_type: 任务类型 ('classification' 或 'regression')
        """
        self.model = model
        self.task_type = task_type.lower()
        self.metrics: Dict[str, float] = {}
        self._validate_task_type()

    def _validate_task_type(self):
        """验证任务类型是否有效"""
        if self.task_type not in ['classification', 'regression']:
            raise ValueError(
                f"Invalid task type: {self.task_type}. "
                "Must be 'classification' or 'regression'"
            )

    def evaluate(
        self,
        X: Union[np.ndarray, list],
        y_true: Union[np.ndarray, list]
    ) -> Dict[str, float]:
        """
        执行模型评估并返回指标字典

        Args:
            X: 测试数据
            y_true: 真实标签/值

        Returns:
            包含各项评估指标的字典
        """
        y_pred = self.model.predict(X)

        if self.task_type == 'classification':
            self.metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='macro'),
                'recall': recall_score(y_true, y_pred, average='macro'),
                'f1_score': f1_score(y_true, y_pred, average='macro'),
                'roc_auc': roc_auc_score(y_true, y_pred)
                    if len(np.unique(y_true)) == 2 else None
            }
        else:
            self.metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }

        return self.metrics

    def get_metrics(self) -> Dict[str, float]:
        """获取最后一次评估的指标结果"""
        if not self.metrics:
            raise RuntimeError("No evaluation has been performed yet")
        return self.metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """绘制混淆矩阵 (仅分类任务)"""
        if self.task_type != 'classification':
            raise RuntimeError("Confusion matrix is only for classification tasks")

        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def generate_report(self) -> str:
        """生成文本格式的评估报告"""
        metrics = self.get_metrics()
        report = f"Model Evaluation Report\n{'='*30}\n"
        report += f"Task Type: {self.task_type}\n"

        for name, value in metrics.items():
            report += f"{name.replace('_', ' ').title()}: {value:.4f}\n"

        return report
