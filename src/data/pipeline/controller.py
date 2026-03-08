"""
ML管道控制器模块

提供端到端的机器学习训练管道编排功能，支持8个阶段的自动化执行
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import json

from .config import PipelineConfig, StageConfig
from .exceptions import (
    PipelineException,
    StageExecutionException,
    RollbackException,
    ConfigurationException
)
from .stages.base import PipelineStage, StageResult, StageStatus


class PipelineStatus(Enum):
    """管道执行状态"""
    PENDING = auto()          # 等待执行
    RUNNING = auto()          # 执行中
    COMPLETED = auto()        # 完成
    FAILED = auto()           # 失败
    ROLLING_BACK = auto()     # 回滚中
    ROLLED_BACK = auto()      # 已回滚
    PARTIAL = auto()          # 部分完成


@dataclass
class PipelineContext:
    """
    管道执行上下文
    
    保存管道执行过程中的所有状态和数据
    
    Attributes:
        pipeline_id: 管道实例ID
        pipeline_name: 管道名称
        start_time: 开始时间
        end_time: 结束时间
        stage_results: 各阶段执行结果
        global_context: 全局上下文数据
        metadata: 元数据
    """
    pipeline_id: str
    pipeline_name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    global_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_stage_output(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """获取指定阶段的输出"""
        if stage_name in self.stage_results:
            return self.stage_results[stage_name].output
        return None
    
    def get_all_outputs(self) -> Dict[str, Any]:
        """获取所有阶段的合并输出"""
        combined = dict(self.global_context)
        for result in self.stage_results.values():
            if result.is_success:
                combined.update(result.output)
        return combined
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "stage_results": {
                name: result.to_dict() 
                for name, result in self.stage_results.items()
            },
            "metadata": self.metadata
        }


@dataclass
class PipelineExecutionResult:
    """
    管道执行结果
    
    Attributes:
        pipeline_id: 管道实例ID
        status: 执行状态
        context: 执行上下文
        error: 错误信息
        summary: 执行摘要
    """
    pipeline_id: str
    status: PipelineStatus
    context: PipelineContext
    error: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        """是否成功完成"""
        return self.status == PipelineStatus.COMPLETED
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """执行时长"""
        if self.context.start_time and self.context.end_time:
            return (self.context.end_time - self.context.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.name,
            "duration_seconds": self.duration_seconds,
            "context": self.context.to_dict(),
            "error": self.error,
            "summary": self.summary
        }


class MLPipelineController:
    """
    ML管道控制器
    
    负责管道的整体编排和执行管理，支持：
    - 阶段依赖管理
    - 错误处理和重试
    - 自动回滚
    - 状态持久化
    - 并发控制
    
    Attributes:
        config: 管道配置
        stages: 阶段字典
        logger: 日志记录器
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        stages: Optional[Dict[str, PipelineStage]] = None
    ):
        """
        初始化管道控制器
        
        Args:
            config: 管道配置
            stages: 预定义的阶段字典（可选）
        """
        self.config = config
        self.stages = stages or {}
        self.logger = logging.getLogger(f"pipeline.controller.{config.name}")
        self._running_pipelines: Dict[str, PipelineContext] = {}
        
        # 验证配置
        errors = config.validate()
        if errors:
            raise ConfigurationException(
                message=f"管道配置验证失败: {'; '.join(errors)}",
                config_key="pipeline_config"
            )
        
        self.logger.info(f"管道控制器初始化完成: {config.name} v{config.version}")
    
    def register_stage(self, stage: PipelineStage) -> None:
        """
        注册阶段
        
        Args:
            stage: 阶段实例
        """
        self.stages[stage.name] = stage
        self.logger.debug(f"注册阶段: {stage.name}")
    
    def register_stages(self, stages: List[PipelineStage]) -> None:
        """
        批量注册阶段
        
        Args:
            stages: 阶段实例列表
        """
        for stage in stages:
            self.register_stage(stage)
    
    def _build_execution_order(self) -> List[str]:
        """
        构建阶段执行顺序（拓扑排序）
        
        Returns:
            按依赖关系排序的阶段名称列表
        """
        # 构建依赖图
        in_degree = {stage.name: 0 for stage in self.config.stages}
        dependencies = {stage.name: [] for stage in self.config.stages}
        
        for stage_config in self.config.stages:
            for dep in stage_config.dependencies:
                if dep in in_degree:
                    dependencies[dep].append(stage_config.name)
                    in_degree[stage_config.name] += 1
        
        # 拓扑排序
        queue = [name for name, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            for dependent in dependencies[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # 检查是否有循环依赖
        if len(execution_order) != len(self.config.stages):
            raise ConfigurationException(
                message="阶段依赖存在循环",
                config_key="stage_dependencies"
            )
        
        return execution_order
    
    def execute(
        self,
        initial_context: Optional[Dict[str, Any]] = None,
        pipeline_id: Optional[str] = None
    ) -> PipelineExecutionResult:
        """
        执行管道
        
        Args:
            initial_context: 初始上下文数据
            pipeline_id: 管道实例ID（自动生成如果不提供）
            
        Returns:
            管道执行结果
        """
        pipeline_id = pipeline_id or str(uuid.uuid4())
        
        # 创建执行上下文
        context = PipelineContext(
            pipeline_id=pipeline_id,
            pipeline_name=self.config.name,
            start_time=datetime.now(),
            global_context=initial_context or {}
        )
        
        self._running_pipelines[pipeline_id] = context
        
        self.logger.info(f"开始执行管道: {self.config.name} (ID: {pipeline_id})")
        
        try:
            # 构建执行顺序
            execution_order = self._build_execution_order()
            self.logger.info(f"执行顺序: {' -> '.join(execution_order)}")
            
            # 按顺序执行阶段
            for stage_name in execution_order:
                stage_config = self.config.get_stage_config(stage_name)
                
                if not stage_config:
                    raise ConfigurationException(
                        message=f"阶段配置不存在: {stage_name}",
                        config_key=f"stage.{stage_name}"
                    )
                
                # 检查阶段是否已注册
                if stage_name not in self.stages:
                    raise ConfigurationException(
                        message=f"阶段未注册: {stage_name}",
                        config_key=f"stage.{stage_name}"
                    )
                
                stage = self.stages[stage_name]
                
                # 执行阶段
                self.logger.info(f"执行阶段: {stage_name}")
                
                try:
                    # 获取当前上下文（包含之前阶段的输出）
                    current_context = context.get_all_outputs()
                    
                    # 运行阶段
                    result = stage.run(current_context)
                    
                    # 保存结果
                    context.stage_results[stage_name] = result
                    
                    if not result.is_success:
                        raise StageExecutionException(
                            message=f"阶段 {stage_name} 执行失败",
                            stage_name=stage_name
                        )
                    
                    self.logger.info(f"阶段 {stage_name} 完成，耗时: {result.duration_seconds:.2f}秒")
                    
                except Exception as e:
                    self.logger.error(f"阶段 {stage_name} 执行异常: {e}")
                    
                    # 触发回滚
                    if self.config.rollback.enabled:
                        self._rollback(context, stage_name)
                    
                    raise
            
            # 所有阶段完成
            context.end_time = datetime.now()
            del self._running_pipelines[pipeline_id]
            
            # 生成执行摘要
            summary = self._generate_summary(context)
            
            self.logger.info(f"管道执行完成，总耗时: {summary.get('total_duration_seconds', 0):.2f}秒")
            
            return PipelineExecutionResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.COMPLETED,
                context=context,
                summary=summary
            )
            
        except Exception as e:
            context.end_time = datetime.now()
            del self._running_pipelines[pipeline_id]
            
            error_msg = str(e)
            self.logger.error(f"管道执行失败: {error_msg}")
            
            return PipelineExecutionResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.FAILED,
                context=context,
                error=error_msg,
                summary=self._generate_summary(context)
            )
    
    def _rollback(self, context: PipelineContext, failed_stage: str) -> bool:
        """
        执行回滚
        
        Args:
            context: 管道上下文
            failed_stage: 失败的阶段名称
            
        Returns:
            回滚是否成功
        """
        self.logger.warning(f"开始回滚管道，失败阶段: {failed_stage}")
        
        success = True
        
        # 逆序回滚已完成的阶段
        completed_stages = [
            name for name, result in context.stage_results.items()
            if result.is_success
        ]
        
        for stage_name in reversed(completed_stages):
            try:
                stage = self.stages[stage_name]
                if not stage.rollback(context.get_all_outputs()):
                    success = False
                    self.logger.warning(f"阶段 {stage_name} 回滚失败")
            except Exception as e:
                success = False
                self.logger.error(f"阶段 {stage_name} 回滚异常: {e}")
        
        if success:
            self.logger.info("管道回滚完成")
        else:
            self.logger.error("管道回滚部分失败")
        
        return success
    
    def _generate_summary(self, context: PipelineContext) -> Dict[str, Any]:
        """
        生成执行摘要
        
        Args:
            context: 管道上下文
            
        Returns:
            执行摘要字典
        """
        total_duration = 0
        stage_summaries = {}
        
        for stage_name, result in context.stage_results.items():
            duration = result.duration_seconds or 0
            total_duration += duration
            
            stage_summaries[stage_name] = {
                "status": result.status.name,
                "duration_seconds": duration,
                "metrics": result.metrics
            }
        
        return {
            "total_duration_seconds": total_duration,
            "stages_completed": sum(1 for r in context.stage_results.values() if r.is_success),
            "stages_total": len(context.stage_results),
            "stage_summaries": stage_summaries
        }
    
    def get_status(self, pipeline_id: str) -> Optional[PipelineStatus]:
        """
        获取管道执行状态
        
        Args:
            pipeline_id: 管道实例ID
            
        Returns:
            管道状态，如果不存在返回None
        """
        if pipeline_id in self._running_pipelines:
            return PipelineStatus.RUNNING
        return None
    
    def save_state(self, file_path: Union[str, Path]) -> None:
        """
        保存管道状态到文件
        
        Args:
            file_path: 保存路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "config": self.config.to_dict(),
            "running_pipelines": {
                pid: ctx.to_dict() 
                for pid, ctx in self._running_pipelines.items()
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"管道状态已保存: {file_path}")
    
    @classmethod
    def load_state(
        cls,
        file_path: Union[str, Path],
        stages: Optional[Dict[str, PipelineStage]] = None
    ) -> 'MLPipelineController':
        """
        从文件加载管道状态
        
        Args:
            file_path: 状态文件路径
            stages: 阶段字典
            
        Returns:
            管道控制器实例
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        config = PipelineConfig.from_dict(state["config"])
        controller = cls(config, stages or {})
        
        # 恢复运行中的管道上下文
        for pid, ctx_data in state.get("running_pipelines", {}).items():
            context = PipelineContext(
                pipeline_id=ctx_data["pipeline_id"],
                pipeline_name=ctx_data["pipeline_name"],
                global_context=ctx_data.get("global_context", {}),
                metadata=ctx_data.get("metadata", {})
            )
            controller._running_pipelines[pid] = context
        
        return controller
    
    def stop_pipeline(self, pipeline_id: str) -> bool:
        """
        停止运行中的管道
        
        Args:
            pipeline_id: 管道实例ID
            
        Returns:
            是否成功停止
        """
        if pipeline_id not in self._running_pipelines:
            self.logger.warning(f"管道 {pipeline_id} 不在运行中")
            return False
        
        # TODO: 实现实际的停止逻辑
        self.logger.info(f"停止管道: {pipeline_id}")
        del self._running_pipelines[pipeline_id]
        return True
