# src/models/utils.py
import numpy as np
import torch
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """早停类，用于防止过拟合"""

    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.patience_counter = 0

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            self.val_loss_min = val_loss
        elif val_loss >= self.val_loss_min - self.delta:  # 修复比较方向
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.val_loss_min = val_loss
            self.patience_counter = 0
        return self.early_stop


class DeviceManager:
    """设备选择管理器，根据运行环境自动选择设备"""

    @staticmethod
    def get_device(device_str: str = "auto") -> torch.device:
        """根据设备字符串和环境自动选择设备"""
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("自动选择设备: CUDA (GPU)")
            else:
                device = torch.device("cpu")
                logger.info("自动选择设备: CPU (CUDA不可用)")
        elif device_str == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("用户指定设备: CUDA (GPU)")
            else:
                # 修复：修改日志消息以匹配测试期望
                logger.warning("用户指定CUDA但不可用，回退到CPU设备")  # 修改这里
                device = torch.device("cpu")
        else:
            device = torch.device(device_str)
            logger.info(f"用户指定设备: {device_str}")

        # 记录设备详细信息
        if device.type == "cuda":
            try:
                logger.info(f"GPU设备: {torch.cuda.get_device_name(device)}")
                logger.info(f"CUDA版本: {torch.version.cuda}")
            except RuntimeError as e:
                logger.error(f"无法获取GPU设备信息: {str(e)}")
                logger.warning("回退到CPU设备")
                device = torch.device("cpu")

        return device
