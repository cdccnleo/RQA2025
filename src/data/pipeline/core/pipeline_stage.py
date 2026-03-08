"""
管道阶段基类模块

定义所有管道阶段的统一接口和基础功能，提供阶段执行、验证和回滚的标准化框架。

主要功能:
    - 定义阶段执行接口 execute(context)
    - 定义阶段验证接口 validate(context)
    - 定义阶段回滚接口 rollback(context)
    - 提供阶段状态管理和指标收集
    - 支持阶段重试和超时控制

使用示例:
    >>> class MyStage(PipelineStage):
    ...     def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
    ...         # 实现阶段逻辑
    ...         return {"result": "success"}
    ...
    ...     def validate(self, context: Dict[str, Any]) -> bool:
    ...         # 实现验证逻辑
    ...         return True
    ...
    ...     def rollback(self, context: Dict[str, Any]) -> bool:
    ...         # 实现回滚逻辑
    ...         return True
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time


class StageStatus(Enum):
    """
    阶段状态枚举
    
    定义管道阶段在执行生命周期中的各种状态。
    
    Attributes:
        PENDING: 等待执行
        RUNNING: 执行中
        COMPLETED: 成功完成
        FAILED: 执行失败
        ROLLED_BACK: 已回滚
        SKIPPED: 被跳过（未启用或条件不满足）
    """
    PENDING = auto()      # 等待执行
    RUNNING = auto()      # 执行中
    COMPLETED = auto()    # 完成
    FAILED = auto()       # 失败
    ROLLED_BACK = auto()  # 已回滚
    SKIPPED = auto()      # 跳过


@dataclass
class StageResult:
    """
    阶段执行结果数据类
    
    保存阶段执行的完整结果信息，包括状态、输出数据、指标和日志。
    
    Attributes:
        stage_name: 阶段名称
        status: 执行状态
        start_time: 开始时间
        end_time: 结束时间
        output: 阶段输出数据字典
        metrics: 阶段执行指标字典
        error: 错误信息（如果有）
        logs: 执行日志列表
    
    Properties:
        duration_seconds: 执行时长（秒）
        is_success: 是否成功完成
    
    Example:
        >>> result = StageResult(
        ...     stage_name="data_preparation",
        ...     status=StageStatus.COMPLETED,
        ...     start_time=datetime.now(),
        ...     output={"data_shape": (1000, 10)}
        ... )
        >>> print(f"阶段耗时: {result.duration_seconds:.2f}秒")
    """
    stage_name: str
    status: StageStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    output: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """
        计算执行时长（秒）
        
        Returns:
            执行时长，如果开始或结束时间为None则返回None
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_success(self) -> bool:
        """
        检查是否成功完成
        
        Returns:
            True 如果状态为 COMPLETED
        """
        return self.status == StageStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将结果转换为字典格式
        
        Returns:
            包含所有结果信息的字典
        """
        return {
            "stage_name": self.stage_name,
            "status": self.status.name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "output": self.output,
            "metrics": self.metrics,
            "error": self.error,
            "logs": self.logs
        }


class PipelineStage(ABC):
    """
    管道阶段抽象基类
    
    所有管道阶段必须继承此类并实现 execute 方法。
    提供阶段执行、验证、回滚的标准化框架，支持重试和超时控制。
    
    Attributes:
        name: 阶段名称
        config: 阶段配置字典
        logger: 日志记录器
        _result: 当前执行结果
    
    Example:
        >>> class DataPreparationStage(PipelineStage):
        ...     def __init__(self):
        ...         super().__init__("data_preparation")
        ...
        ...     def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ...         data = context.get("raw_data")
        ...         processed = self._process_data(data)
        ...         return {"processed_data": processed}
        ...
        ...     def validate(self, context: Dict[str, Any]) -> bool:
        ...         output = context.get("output", {})
        ...         return "processed_data" in output
        ...
        ...     def rollback(self, context: Dict[str, Any]) -> bool:
        ...         # 清理临时文件等
        ...         return True
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化管道阶段
        
        Args:
            name: 阶段名称，用于标识和日志记录
            config: 阶段配置字典，包含重试次数、超时时间等参数
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"pipeline.stage.{name}")
        self._result: Optional[StageResult] = None
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行阶段任务（抽象方法，子类必须实现）
        
        执行阶段的核心逻辑，处理输入数据并返回输出结果。
        
        Args:
            context: 管道上下文字典，包含之前阶段的输出和全局配置
                - pipeline_id: 管道实例ID
                - global_config: 全局配置
                - 之前阶段的输出数据
        
        Returns:
            阶段输出数据字典，将被传递给后续阶段
        
        Raises:
            StageExecutionException: 执行失败时抛出
        
        Example:
            >>> def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
            ...     raw_data = context.get("raw_data")
            ...     processed = self._transform(raw_data)
            ...     return {"processed_data": processed}
        """
        pass
    
    def validate(self, context: Dict[str, Any]) -> bool:
        """
        验证阶段执行结果（可选重写）
        
        验证阶段输出是否满足要求，可以在 execute 后自动调用。
        
        Args:
            context: 管道上下文字典，包含阶段输出
        
        Returns:
            True 如果验证通过
        
        Raises:
            StageValidationException: 验证失败时抛出
        
        Example:
            >>> def validate(self, context: Dict[str, Any]) -> bool:
            ...     output = context.get("output", {})
            ...     if "accuracy" not in output:
            ...         raise StageValidationException("缺少 accuracy 指标")
            ...     return output["accuracy"] > 0.8
        """
        return True
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """
        回滚阶段操作（可选重写）
        
        在阶段执行失败或管道需要回滚时调用，清理已执行的操作。
        
        Args:
            context: 管道上下文字典
        
        Returns:
            True 如果回滚成功
        
        Example:
            >>> def rollback(self, context: Dict[str, Any]) -> bool:
            ...     # 删除临时文件
            ...     if self._temp_file.exists():
            ...         self._temp_file.unlink()
            ...     return True
        """
        self.logger.info(f"阶段 {self.name} 回滚操作（默认空实现）")
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取阶段执行指标（可选重写）
        
        返回阶段的性能指标和业务指标，用于监控和报告。
        
        Returns:
            指标字典
        
        Example:
            >>> def get_metrics(self) -> Dict[str, Any]:
            ...     return {
            ...         "processed_rows": len(self._data),
            ...         "processing_time_ms": self._processing_time
            ...     }
        """
        if self._result:
            return self._result.metrics
        return {}
    
    def run(
        self,
        context: Dict[str, Any],
        retry_count: int = 0,
        retry_delay: float = 1.0
    ) -> StageResult:
        """
        运行阶段（包含重试逻辑）
        
        执行阶段并处理重试逻辑，记录执行结果和日志。
        
        Args:
            context: 管道上下文字典
            retry_count: 失败重试次数
            retry_delay: 重试间隔（秒）
        
        Returns:
            StageResult: 阶段执行结果
        
        Raises:
            StageExecutionException: 所有重试失败后抛出
        """
        start_time = datetime.now()
        logs = []
        
        self.logger.info(f"开始执行阶段: {self.name}")
        logs.append(f"[{start_time.isoformat()}] 阶段开始执行")
        
        result = StageResult(
            stage_name=self.name,
            status=StageStatus.RUNNING,
            start_time=start_time,
            logs=logs
        )
        self._result = result
        
        # 执行重试逻辑
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                if attempt > 0:
                    retry_msg = f"第 {attempt} 次重试..."
                    self.logger.info(retry_msg)
                    logs.append(f"[{datetime.now().isoformat()}] {retry_msg}")
                    time.sleep(retry_delay)
                
                # 执行阶段
                output = self.execute(context)
                
                # 更新上下文用于验证
                validation_context = dict(context)
                validation_context["output"] = output
                
                # 验证输出
                self.validate(validation_context)
                
                # 更新结果
                result.status = StageStatus.COMPLETED
                result.end_time = datetime.now()
                result.output = output
                result.metrics = self.get_metrics()
                
                success_msg = f"阶段 {self.name} 执行成功"
                self.logger.info(success_msg)
                logs.append(f"[{result.end_time.isoformat()}] {success_msg}")
                
                return result
                
            except Exception as e:
                last_error = e
                error_msg = f"阶段执行失败: {str(e)}"
                self.logger.error(error_msg)
                logs.append(f"[{datetime.now().isoformat()}] ERROR: {error_msg}")
        
        # 所有重试失败
        result.status = StageStatus.FAILED
        result.end_time = datetime.now()
        result.error = str(last_error) if last_error else "未知错误"
        
        from ..exceptions import StageExecutionException
        raise StageExecutionException(
            message=f"阶段 {self.name} 执行失败（已重试 {retry_count} 次）",
            stage_name=self.name,
            context={"attempts": retry_count + 1},
            cause=last_error
        )
    
    def cleanup(self) -> None:
        """
        清理阶段资源（可选重写）
        
        在阶段执行完成后调用，用于释放临时资源。
        
        Example:
            >>> def cleanup(self) -> None:
            ...     self._temp_buffer.clear()
            ...     self._cache = None
        """
        self.logger.debug(f"清理阶段 {self.name} 资源")
    
    def get_dependencies(self) -> List[str]:
        """
        获取阶段依赖（可选重写）
        
        返回此阶段依赖的其他阶段名称列表，用于构建执行顺序。
        
        Returns:
            依赖阶段名称列表
        
        Example:
            >>> def get_dependencies(self) -> List[str]:
            ...     return ["data_preparation"]  # 依赖数据准备阶段
        """
        return self.config.get("dependencies", [])
    
    def __str__(self) -> str:
        """字符串表示"""
        status = self._result.status.name if self._result else "NOT_STARTED"
        return f"PipelineStage(name={self.name}, status={status})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


class CompositeStage(PipelineStage):
    """
    复合阶段类
    
    包含多个子阶段的复合阶段，可以嵌套使用，按顺序执行所有子阶段。
    
    Attributes:
        stages: 子阶段列表
        _stage_results: 子阶段结果字典
    
    Example:
        >>> stage1 = DataPreparationStage()
        >>> stage2 = FeatureEngineeringStage()
        >>> composite = CompositeStage("preprocessing", [stage1, stage2])
        >>> result = composite.run(context)
    """
    
    def __init__(
        self,
        name: str,
        stages: List[PipelineStage],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化复合阶段
        
        Args:
            name: 阶段名称
            stages: 子阶段列表，按顺序执行
            config: 阶段配置
        """
        super().__init__(name, config)
        self.stages = stages
        self._stage_results: Dict[str, StageResult] = {}
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        按顺序执行所有子阶段
        
        Args:
            context: 管道上下文
        
        Returns:
            合并后的输出数据
        
        Raises:
            StageExecutionException: 子阶段执行失败时抛出
        """
        combined_output = dict(context)
        
        for stage in self.stages:
            self.logger.info(f"执行子阶段: {stage.name}")
            
            # 运行子阶段
            result = stage.run(combined_output)
            
            # 保存结果
            self._stage_results[stage.name] = result
            
            # 合并输出
            if result.is_success:
                combined_output.update(result.output)
            else:
                from ..exceptions import StageExecutionException
                raise StageExecutionException(
                    message=f"子阶段 {stage.name} 执行失败",
                    stage_name=self.name,
                    context={"failed_stage": stage.name}
                )
        
        return combined_output
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """
        回滚所有子阶段（逆序）
        
        Args:
            context: 管道上下文
        
        Returns:
            True 如果所有子阶段回滚成功
        """
        success = True
        
        # 逆序回滚
        for stage in reversed(self.stages):
            try:
                if not stage.rollback(context):
                    success = False
                    self.logger.warning(f"子阶段 {stage.name} 回滚失败")
            except Exception as e:
                success = False
                self.logger.error(f"子阶段 {stage.name} 回滚异常: {e}")
        
        return success
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取所有子阶段的指标
        
        Returns:
            合并的指标字典
        """
        metrics = {
            "sub_stages": {}
        }
        
        for stage_name, result in self._stage_results.items():
            metrics["sub_stages"][stage_name] = {
                "status": result.status.name,
                "duration_seconds": result.duration_seconds,
                "metrics": result.metrics
            }
        
        return metrics
