"""
管道阶段基类模块

定义所有管道阶段的统一接口和基础功能
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

from ..exceptions import (
    PipelineException,
    StageExecutionException,
    StageValidationException
)
from ..config import StageConfig


class StageStatus(Enum):
    """阶段状态枚举"""
    PENDING = auto()      # 等待执行
    RUNNING = auto()      # 执行中
    COMPLETED = auto()    # 完成
    FAILED = auto()       # 失败
    ROLLED_BACK = auto()  # 已回滚
    SKIPPED = auto()      # 跳过


@dataclass
class StageResult:
    """
    阶段执行结果
    
    Attributes:
        stage_name: 阶段名称
        status: 执行状态
        start_time: 开始时间
        end_time: 结束时间
        output: 阶段输出数据
        metrics: 阶段指标
        error: 错误信息
        logs: 执行日志
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
        """计算执行时长"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_success(self) -> bool:
        """是否成功完成"""
        return self.status == StageStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
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
    
    所有管道阶段必须继承此类并实现execute方法
    
    Attributes:
        name: 阶段名称
        config: 阶段配置
        logger: 日志记录器
    """
    
    def __init__(self, name: str, config: Optional[StageConfig] = None):
        """
        初始化阶段
        
        Args:
            name: 阶段名称
            config: 阶段配置
        """
        self.name = name
        self.config = config or StageConfig(name=name)
        self.logger = logging.getLogger(f"pipeline.stage.{name}")
        self._result: Optional[StageResult] = None
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行阶段任务
        
        Args:
            context: 管道上下文，包含之前阶段的输出
            
        Returns:
            阶段输出数据
            
        Raises:
            StageExecutionException: 执行失败
        """
        pass
    
    def validate(self, output: Dict[str, Any]) -> bool:
        """
        验证阶段输出
        
        Args:
            output: 阶段输出数据
            
        Returns:
            验证是否通过
            
        Raises:
            StageValidationException: 验证失败
        """
        # 默认实现：检查输出不为空
        if output is None:
            raise StageValidationException(
                message="阶段输出不能为空",
                stage_name=self.name
            )
        return True
    
    def rollback(self, context: Dict[str, Any]) -> bool:
        """
        回滚阶段操作
        
        Args:
            context: 管道上下文
            
        Returns:
            回滚是否成功
        """
        self.logger.info(f"阶段 {self.name} 回滚操作（默认空实现）")
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取阶段指标
        
        Returns:
            指标字典
        """
        if self._result:
            return self._result.metrics
        return {}
    
    def run(self, context: Dict[str, Any]) -> StageResult:
        """
        运行阶段（包含重试逻辑）
        
        Args:
            context: 管道上下文
            
        Returns:
            阶段执行结果
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
        
        # 检查是否启用
        if not self.config.enabled:
            result.status = StageStatus.SKIPPED
            result.end_time = datetime.now()
            logs.append(f"[{result.end_time.isoformat()}] 阶段已跳过（未启用）")
            self.logger.info(f"阶段 {self.name} 已跳过")
            return result
        
        # 执行重试逻辑
        last_error = None
        for attempt in range(self.config.retry_count + 1):
            try:
                if attempt > 0:
                    retry_msg = f"第 {attempt} 次重试..."
                    self.logger.info(retry_msg)
                    logs.append(f"[{datetime.now().isoformat()}] {retry_msg}")
                    time.sleep(self.config.retry_delay_seconds)
                
                # 执行阶段
                output = self.execute(context)
                
                # 验证输出
                self.validate(output)
                
                # 更新结果
                result.status = StageStatus.COMPLETED
                result.end_time = datetime.now()
                result.output = output
                result.metrics = self.get_metrics()
                
                success_msg = f"阶段 {self.name} 执行成功"
                self.logger.info(success_msg)
                logs.append(f"[{result.end_time.isoformat()}] {success_msg}")
                
                return result
                
            except PipelineException as e:
                last_error = e
                error_msg = f"阶段执行失败: {e.message}"
                self.logger.error(error_msg)
                logs.append(f"[{datetime.now().isoformat()}] ERROR: {error_msg}")
                
                # 如果是验证错误，不重试
                if isinstance(e, StageValidationException):
                    break
                    
            except Exception as e:
                last_error = e
                error_msg = f"阶段执行异常: {str(e)}"
                self.logger.error(error_msg)
                logs.append(f"[{datetime.now().isoformat()}] ERROR: {error_msg}")
        
        # 所有重试失败
        result.status = StageStatus.FAILED
        result.end_time = datetime.now()
        result.error = str(last_error) if last_error else "未知错误"
        
        # 抛出异常
        raise StageExecutionException(
            message=f"阶段 {self.name} 执行失败（已重试 {self.config.retry_count} 次）",
            stage_name=self.name,
            context={"attempts": self.config.retry_count + 1},
            cause=last_error
        )
    
    def cleanup(self) -> None:
        """
        清理阶段资源
        
        子类可以重写此方法进行资源清理
        """
        self.logger.debug(f"清理阶段 {self.name} 资源")
    
    def get_dependencies(self) -> List[str]:
        """
        获取阶段依赖
        
        Returns:
            依赖阶段名称列表
        """
        return self.config.dependencies
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"PipelineStage(name={self.name}, status={self._result.status if self._result else 'NOT_STARTED'})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


class CompositeStage(PipelineStage):
    """
    复合阶段
    
    包含多个子阶段的复合阶段，可以嵌套使用
    """
    
    def __init__(self, name: str, stages: List[PipelineStage], config: Optional[StageConfig] = None):
        """
        初始化复合阶段
        
        Args:
            name: 阶段名称
            stages: 子阶段列表
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
            合并后的输出
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
            回滚是否成功
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
