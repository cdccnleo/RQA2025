"""
GPU加速模块
提供GPU硬件加速功能

主要类:
- GPUManager: GPU管理器
- GPUAccelerator: GPU加速器
- GPUHealthMonitor: GPU健康监视器
- GPUComputeEngine: GPU计算引擎抽象基类
- CUDAComputeEngine: CUDA计算引擎
- OpenCLComputeEngine: OpenCL计算引擎
- GPUScheduler: GPU资源调度器
- TensorRTEngine: TensorRT推理优化引擎

使用示例:
    from src.acceleration.gpu import GPUManager, CUDAComputeEngine, TaskPriority

    manager = GPUManager(enable_scheduler=True, enable_tensorrt=True)
    cuda_engine = CUDAComputeEngine(manager, use_tensorrt=True)
    result = cuda_engine.matrix_multiply(a, b)

    # 提交GPU任务
    success = manager.submit_task(
        task_id="model_inference_001",
        model_id="lstm_model",
        priority=TaskPriority.HIGH,
        memory_required=2048
    )
"""

from .gpu_accelerator import (
    GPUManager,
    GPUAccelerator,
    GPUHealthMonitor,
    GPUComputeEngine,
    CUDAComputeEngine,
    OpenCLComputeEngine,
    TensorRTEngine,
)

from .gpu_scheduler import (
    GPUScheduler,
    TaskPriority,
    TaskStatus,
    SchedulingPolicy,
    GPUTask,
    GPUResource,
)

__all__ = [
    "GPUManager",
    "GPUAccelerator",
    "GPUHealthMonitor",
    "GPUComputeEngine",
    "CUDAComputeEngine",
    "OpenCLComputeEngine",
    "TensorRTEngine",
    "GPUScheduler",
    "TaskPriority",
    "TaskStatus",
    "SchedulingPolicy",
    "GPUTask",
    "GPUResource",
]
