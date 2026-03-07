from typing import Optional

from ..optimizers.base import ObjectiveDirection


class EarlyStopping:

    """早停机制实现"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0, direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE):
        """
        参数:
        patience: 容忍轮数
        min_delta: 最小改进量
        direction: 优化方向
        """
        self.patience = patience
        self.min_delta = min_delta
        self.direction = direction
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """检查是否需要早停"""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.direction == ObjectiveDirection.MAXIMIZE:
            improvement = score > self.best_score + self.min_delta
        else:
            improvement = score < self.best_score - self.min_delta

        if improvement:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True

            return self.early_stop

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
