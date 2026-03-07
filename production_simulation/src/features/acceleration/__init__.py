"""
RQA2025加速层模块
包含FPGA和GPU加速功能

主要模块:
- fpga: FPGA加速模块
- gpu: GPU加速模块

使用示例:
    from src.acceleration.fpga import FpgaManager
    from src.acceleration.gpu import GPUManager, CUDAComputeEngine

版本历史:
- v1.0 (2025 - 07 - 19): 初始版本，从trading层迁移FPGA模块
- v1.1 (2025 - 07 - 19): 添加GPU加速模块实现
"""

from .fpga import FPGAManager as FpgaManager, FPGAAccelerator as FpgaAccelerator
from .gpu import (
    GPUManager,
    GPUAccelerator,
    GPUHealthMonitor,
    GPUComputeEngine,
    CUDAComputeEngine,
    OpenCLComputeEngine
)

__all__ = [
    'FpgaManager',
    'FpgaAccelerator',

    'GPUManager',
    'GPUAccelerator',
    'GPUHealthMonitor',
    'GPUComputeEngine',
    'CUDAComputeEngine',
    'OpenCLComputeEngine'
]
