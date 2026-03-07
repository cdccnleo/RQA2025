"""
深度学习优化器

AI模型优化、GPU资源管理等。

从deep_learning_predictor.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import torch
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class GPUResourceManager:
    """
    GPU资源管理器

    负责GPU资源的分配和管理，支持多GPU训练
    """

    def __init__(self):
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.current_device = 0 if self.device_count > 0 else -1

        logger.info(f"GPU资源管理器初始化，可用GPU数: {self.device_count}")

    def get_device(self) -> torch.device:
        """获取可用设备"""
        if self.device_count > 0:
            return torch.device(f'cuda:{self.current_device}')
        else:
            return torch.device('cpu')

    def get_memory_info(self) -> Dict[str, Any]:
        """获取GPU内存信息"""
        if self.device_count == 0:
            return {'available': False}

        try:
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory

            return {
                'available': True,
                'device_id': device,
                'memory_allocated': memory_allocated / (1024 ** 3),  # GB
                'memory_reserved': memory_reserved / (1024 ** 3),  # GB
                'total_memory': total_memory / (1024 ** 3),  # GB
                'utilization': memory_allocated / total_memory if total_memory > 0 else 0
            }

        except Exception as e:
            logger.error(f"获取GPU内存信息失败: {e}")
            return {'available': False, 'error': str(e)}

    def clear_cache(self):
        """清理GPU缓存"""
        if self.device_count > 0:
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清理")


class AIModelOptimizer:
    """
    AI模型优化器

    提供模型量化、剪枝、蒸馏等优化功能
    """

    def __init__(self):
        self.optimized_models: Dict[str, Any] = {}
        logger.info("AI模型优化器初始化完成")

    def quantize_model(self, model: torch.nn.Module, dtype=torch.qint8) -> torch.nn.Module:
        """
        量化模型

        将模型转换为低精度格式，减少内存占用和计算量
        """
        try:
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.LSTM},
                dtype=dtype
            )
            logger.info(f"模型量化完成: {type(model).__name__}")
            return quantized_model

        except Exception as e:
            logger.error(f"模型量化失败: {e}")
            return model

    def prune_model(self, model: torch.nn.Module, amount: float = 0.3) -> torch.nn.Module:
        """
        剪枝模型

        移除不重要的权重，减少模型复杂度
        """
        try:
            import torch.nn.utils.prune as prune

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=amount)
                    prune.remove(module, 'weight')

            logger.info(f"模型剪枝完成，剪枝比例: {amount}")
            return model

        except Exception as e:
            logger.error(f"模型剪枝失败: {e}")
            return model

    def get_model_size(self, model: torch.nn.Module) -> Dict[str, Any]:
        """获取模型大小信息"""
        try:
            param_count = sum(p.numel() for p in model.parameters())
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)  # MB

            return {
                'param_count': param_count,
                'param_size_mb': param_size,
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }

        except Exception as e:
            logger.error(f"获取模型大小失败: {e}")
            return {}


class DynamicBatchOptimizer:
    """
    动态批量优化器

    根据系统负载自动调整批量大小
    """

    def __init__(self, initial_batch_size: int = 32,
                 min_batch_size: int = 8,
                 max_batch_size: int = 256):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.gpu_manager = GPUResourceManager()

        logger.info(f"动态批量优化器初始化，初始批大小: {initial_batch_size}")

    def adjust_batch_size(self) -> int:
        """根据GPU使用率调整批量大小"""
        memory_info = self.gpu_manager.get_memory_info()

        if not memory_info.get('available'):
            return self.batch_size

        utilization = memory_info.get('utilization', 0)

        if utilization > 0.9:
            # GPU使用率过高，减小批量
            self.batch_size = max(self.min_batch_size, self.batch_size // 2)
            logger.info(f"GPU使用率过高({utilization:.1%})，减小批量至 {self.batch_size}")

        elif utilization < 0.5:
            # GPU使用率较低，增大批量
            self.batch_size = min(self.max_batch_size, self.batch_size * 2)
            logger.info(f"GPU使用率较低({utilization:.1%})，增大批量至 {self.batch_size}")

        return self.batch_size

    def get_batch_size(self) -> int:
        """获取当前批量大小"""
        return self.batch_size


__all__ = [
    'GPUResourceManager',
    'AIModelOptimizer',
    'DynamicBatchOptimizer'
]

