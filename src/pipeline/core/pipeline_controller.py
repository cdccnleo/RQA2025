"""
ML管道控制器模块

提供端到端的机器学习训练管道编排功能，支持8个阶段的自动化执行、失败处理和状态跟踪。

主要功能:
    - 8阶段执行流程管理（数据准备→特征工程→模型训练→评估→验证→金丝雀部署→全量部署→监控）
    - 阶段依赖管理和拓扑排序
    - 失败处理和自动回滚
    - 状态持久化和恢复
    - 与ModelManager、FeatureManager等模块集成

使用示例:
    >>> config = create_default_config()
    >>> controller = MLPipelineController(config)
    >>> controller.register_stage(DataPreparationStage())
    >>> result = controller.execute()
    >>> print(f"管道执行{'成功' if result.is_success else '失败'}")
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .pipeline_stage import PipelineStage, StageResult, StageStatus
from .pipeline_state import PipelineState, PipelineStatus, PipelineStateManager
from .pipeline_config import PipelineConfig, StageConfig


class PipelineExecutionResult:
    """
    管道执行结果类
    
    保存管道执行的完整结果信息。
    
    Attributes:
        pipeline_id: 管道实例ID
        status: 执行状态
        state: 管道状态对象
        error: 错误信息
        summary: 执行摘要
    
    Properties:
        is_success: 是否成功完成
        duration_seconds: 执行时长
    
    Example:
        >>> result = controller.execute()
        >>> if result.is_success:
        ...     print(f"耗时: {result.duration_seconds:.2f}秒")
    """
    
    def __init__(
        self,
        pipeline_id: str,
        status: PipelineStatus,
        state: PipelineState,
        error: Optional[str] = None,
        summary: Optional[Dict[str, Any]] = None
    ):
        """
        初始化执行结果
        
        Args:
            pipeline_id: 管道实例ID
            status: 执行状态
            state: 管道状态对象
            error: 错误信息
            summary: 执行摘要
        """
        self.pipeline_id = pipeline_id
        self.status = status
        self.state = state
        self.error = error
        self.summary = summary or {}
    
    @property
    def is_success(self) -> bool:
        """
        检查是否成功完成
        
        Returns:
            True 如果状态为 COMPLETED
        """
        return self.status == PipelineStatus.COMPLETED
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """
        计算执行时长
        
        Returns:
            执行时长（秒），如果无法计算则返回None
        """
        return self.state.duration_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            结果字典
        """
        return {
            "pipeline_id": self.pipeline_id,
            "status": self.status.name,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "summary": self.summary,
            "state": self.state.to_dict()
        }


class MLPipelineController:
    """
    ML管道控制器
    
    负责管道的整体编排和执行管理，支持8个阶段的自动化执行。
    
    8个标准阶段:
        1. data_preparation: 数据准备
        2. feature_engineering: 特征工程
        3. model_training: 模型训练
        4. model_evaluation: 模型评估
        5. model_validation: 模型验证
        6. canary_deployment: 金丝雀部署
        7. full_deployment: 全量部署
        8. monitoring: 监控
    
    Attributes:
        config: 管道配置
        stages: 阶段字典
        state_manager: 状态管理器
        logger: 日志记录器
    
    Example:
        >>> config = create_default_config()
        >>> controller = MLPipelineController(config)
        >>> 
        >>> # 注册阶段
        >>> controller.register_stage(DataPreparationStage())
        >>> controller.register_stage(FeatureEngineeringStage())
        >>> 
        >>> # 执行管道
        >>> result = controller.execute()
        >>> 
        >>> # 保存状态
        >>> controller.save_state()
    """
    
    # 8个标准阶段的执行顺序
    DEFAULT_STAGE_ORDER = [
        "data_preparation",
        "feature_engineering",
        "model_training",
        "model_evaluation",
        "model_validation",
        "canary_deployment",
        "full_deployment",
        "monitoring"
    ]
    
    def __init__(
        self,
        config: PipelineConfig,
        state_manager: Optional[PipelineStateManager] = None
    ):
        """
        初始化管道控制器
        
        Args:
            config: 管道配置
            state_manager: 状态管理器（可选，默认创建新实例）
        
        Raises:
            ValueError: 配置验证失败
        """
        self.config = config
        self.stages: Dict[str, PipelineStage] = {}
        self.state_manager = state_manager or PipelineStateManager()
        self.logger = logging.getLogger(f"pipeline.controller.{config.name}")
        
        # 验证配置
        errors = config.validate()
        if errors:
            raise ValueError(f"管道配置验证失败: {'; '.join(errors)}")
        
        self.logger.info(f"管道控制器初始化完成: {config.name} v{config.version}")
    
    def register_stage(self, stage: PipelineStage) -> None:
        """
        注册阶段
        
        将阶段实例注册到控制器，用于后续执行。
        
        Args:
            stage: 阶段实例
        
        Example:
            >>> controller.register_stage(DataPreparationStage())
        """
        self.stages[stage.name] = stage
        self.logger.debug(f"注册阶段: {stage.name}")
    
    def register_stages(self, stages: List[PipelineStage]) -> None:
        """
        批量注册阶段
        
        Args:
            stages: 阶段实例列表
        
        Example:
            >>> stages = [Stage1(), Stage2(), Stage3()]
            >>> controller.register_stages(stages)
        """
        for stage in stages:
            self.register_stage(stage)
    
    def _build_execution_order(self) -> List[str]:
        """
        构建阶段执行顺序（拓扑排序）
        
        根据阶段依赖关系计算正确的执行顺序。
        
        Returns:
            按依赖关系排序的阶段名称列表
        
        Raises:
            ValueError: 存在循环依赖
        """
        # 获取启用的阶段
        enabled_stages = [
            s for s in self.config.stages
            if s.enabled
        ]
        
        if not enabled_stages:
            return []
        
        # 构建依赖图
        in_degree = {stage.name: 0 for stage in enabled_stages}
        dependencies = {stage.name: [] for stage in enabled_stages}
        
        for stage in enabled_stages:
            for dep in stage.dependencies:
                if dep in in_degree:
                    dependencies[dep].append(stage.name)
                    in_degree[stage.name] += 1
        
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
        
        # 检查循环依赖
        if len(execution_order) != len(enabled_stages):
            raise ValueError("阶段依赖存在循环")
        
        return execution_order
    
    def execute(
        self,
        initial_context: Optional[Dict[str, Any]] = None,
        pipeline_id: Optional[str] = None,
        resume_from: Optional[str] = None
    ) -> PipelineExecutionResult:
        """
        执行管道
        
        按顺序执行所有阶段，处理依赖关系和失败情况。
        
        Args:
            initial_context: 初始上下文数据
            pipeline_id: 管道实例ID（自动生成如果不提供）
            resume_from: 从指定阶段恢复执行（可选）
        
        Returns:
            PipelineExecutionResult: 执行结果
        
        Example:
            >>> result = controller.execute(
            ...     initial_context={"symbols": ["AAPL", "GOOGL"]},
            ...     pipeline_id="my_pipeline_001"
            ... )
            >>> print(f"结果: {result.status}")
        """
        pipeline_id = pipeline_id or str(uuid.uuid4())
        
        # 创建或恢复状态
        if resume_from:
            state = self.state_manager.load_state(pipeline_id)
            if not state:
                self.logger.warning(f"无法恢复管道 {pipeline_id}，创建新状态")
                state = self._create_state(pipeline_id)
        else:
            state = self._create_state(pipeline_id)
        
        # 更新初始上下文
        if initial_context:
            state.context.update(initial_context)
        
        self.logger.info(f"开始执行管道: {self.config.name} (ID: {pipeline_id})")
        
        try:
            # 构建执行顺序
            execution_order = self._build_execution_order()
            self.logger.info(f"执行顺序: {' -> '.join(execution_order)}")
            
            # 如果从指定阶段恢复，跳过已完成的阶段
            if resume_from and resume_from in execution_order:
                resume_idx = execution_order.index(resume_from)
                execution_order = execution_order[resume_idx:]
                self.logger.info(f"从阶段 {resume_from} 恢复执行")
            
            # 按顺序执行阶段
            for stage_name in execution_order:
                stage_config = self.config.get_stage_config(stage_name)
                
                if not stage_config:
                    raise ValueError(f"阶段配置不存在: {stage_name}")
                
                # 检查阶段是否已注册
                if stage_name not in self.stages:
                    raise ValueError(f"阶段未注册: {stage_name}")
                
                stage = self.stages[stage_name]
                
                # 执行阶段
                success = self._execute_stage(stage, stage_config, state)
                
                if not success:
                    # 执行失败，触发回滚
                    if self.config.rollback.enabled:
                        self._rollback(state, stage_name)
                    
                    return PipelineExecutionResult(
                        pipeline_id=pipeline_id,
                        status=PipelineStatus.FAILED,
                        state=state,
                        error=f"阶段 {stage_name} 执行失败",
                        summary=self._generate_summary(state)
                    )
            
            # 所有阶段完成
            state.mark_completed()
            self.state_manager.save_state(state)
            
            # 生成执行摘要
            summary = self._generate_summary(state)
            
            self.logger.info(f"管道执行完成，总耗时: {summary.get('total_duration_seconds', 0):.2f}秒")
            
            return PipelineExecutionResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.COMPLETED,
                state=state,
                summary=summary
            )
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"管道执行失败: {error_msg}")
            
            state.mark_failed(error_msg)
            self.state_manager.save_state(state)
            
            return PipelineExecutionResult(
                pipeline_id=pipeline_id,
                status=PipelineStatus.FAILED,
                state=state,
                error=error_msg,
                summary=self._generate_summary(state)
            )
    
    def _create_state(self, pipeline_id: str) -> PipelineState:
        """
        创建新的管道状态
        
        Args:
            pipeline_id: 管道ID
        
        Returns:
            新创建的管道状态
        """
        state = self.state_manager.create_state(
            pipeline_name=self.config.name,
            pipeline_id=pipeline_id,
            total_stages=len([s for s in self.config.stages if s.enabled])
        )
        self.state_manager.save_state(state)
        return state
    
    def _execute_stage(
        self,
        stage: PipelineStage,
        stage_config: StageConfig,
        state: PipelineState
    ) -> bool:
        """
        执行单个阶段
        
        Args:
            stage: 阶段实例
            stage_config: 阶段配置
            state: 管道状态
        
        Returns:
            True 如果执行成功
        """
        stage_name = stage.name
        
        self.logger.info(f"执行阶段: {stage_name}")
        state.mark_stage_start(stage_name)
        self.state_manager.save_state(state)
        
        try:
            # 构建阶段上下文
            stage_context = self._build_stage_context(state)
            
            # 运行阶段
            result = stage.run(
                stage_context,
                retry_count=stage_config.retry_count,
                retry_delay=stage_config.retry_delay_seconds
            )
            
            # 更新状态
            state.update_stage_result(stage_name, result)
            state.mark_stage_complete(stage_name)
            self.state_manager.save_state(state)
            
            if result.is_success:
                self.logger.info(f"阶段 {stage_name} 完成，耗时: {result.duration_seconds:.2f}秒")
                return True
            else:
                self.logger.error(f"阶段 {stage_name} 执行失败")
                return False
                
        except Exception as e:
            self.logger.error(f"阶段 {stage_name} 执行异常: {e}")
            
            # 创建失败结果
            failed_result = StageResult(
                stage_name=stage_name,
                status=StageStatus.FAILED,
                error=str(e)
            )
            state.update_stage_result(stage_name, failed_result)
            self.state_manager.save_state(state)
            
            return False
    
    def _build_stage_context(self, state: PipelineState) -> Dict[str, Any]:
        """
        构建阶段执行上下文
        
        合并全局上下文和所有已完成阶段的输出。
        
        Args:
            state: 管道状态
        
        Returns:
            阶段执行上下文
        """
        context = dict(state.context)
        context["pipeline_id"] = state.pipeline_id
        context["global_config"] = self.config.global_config
        context["integration_config"] = self.config.integration.to_dict()
        
        # 合并已完成阶段的输出
        for stage_name, result in state.stage_results.items():
            if result.is_success:
                context.update(result.output)
        
        return context
    
    def _rollback(self, state: PipelineState, failed_stage: str) -> bool:
        """
        执行回滚
        
        逆序回滚已完成的阶段。
        
        Args:
            state: 管道状态
            failed_stage: 失败的阶段名称
        
        Returns:
            True 如果回滚成功
        """
        self.logger.warning(f"开始回滚管道，失败阶段: {failed_stage}")
        state.status = PipelineStatus.ROLLING_BACK
        
        success = True
        context = self._build_stage_context(state)
        
        # 逆序回滚已完成的阶段
        completed_stages = [
            name for name, result in state.stage_results.items()
            if result.is_success
        ]
        
        for stage_name in reversed(completed_stages):
            try:
                stage = self.stages.get(stage_name)
                if stage and not stage.rollback(context):
                    success = False
                    self.logger.warning(f"阶段 {stage_name} 回滚失败")
            except Exception as e:
                success = False
                self.logger.error(f"阶段 {stage_name} 回滚异常: {e}")
        
        if success:
            state.status = PipelineStatus.ROLLED_BACK
            self.logger.info("管道回滚完成")
        else:
            self.logger.error("管道回滚部分失败")
        
        self.state_manager.save_state(state)
        return success
    
    def _generate_summary(self, state: PipelineState) -> Dict[str, Any]:
        """
        生成执行摘要
        
        Args:
            state: 管道状态
        
        Returns:
            执行摘要字典
        """
        total_duration = 0.0
        stage_summaries = {}
        
        for stage_name, result in state.stage_results.items():
            duration = result.duration_seconds or 0.0
            total_duration += duration
            
            stage_summaries[stage_name] = {
                "status": result.status.name,
                "duration_seconds": duration,
                "metrics": result.metrics
            }
        
        return {
            "total_duration_seconds": total_duration,
            "stages_completed": sum(1 for r in state.stage_results.values() if r.is_success),
            "stages_total": len(state.stage_results),
            "stage_summaries": stage_summaries,
            "pipeline_name": self.config.name,
            "pipeline_version": self.config.version
        }
    
    def get_state(self, pipeline_id: str) -> Optional[PipelineState]:
        """
        获取管道状态
        
        Args:
            pipeline_id: 管道ID
        
        Returns:
            管道状态或None
        """
        return self.state_manager.load_state(pipeline_id)
    
    def save_state(self, pipeline_id: Optional[str] = None) -> None:
        """
        保存管道状态
        
        Args:
            pipeline_id: 管道ID（可选）
        """
        # 这个方法在状态管理器中自动调用
        self.logger.debug("状态已自动保存")
    
    def pause_pipeline(self, pipeline_id: str) -> bool:
        """
        暂停管道执行
        
        Args:
            pipeline_id: 管道ID
        
        Returns:
            True 如果成功暂停
        """
        state = self.state_manager.load_state(pipeline_id)
        if not state or not state.is_active:
            return False
        
        state.status = PipelineStatus.PAUSED
        self.state_manager.save_state(state)
        self.logger.info(f"管道已暂停: {pipeline_id}")
        return True
    
    def resume_pipeline(self, pipeline_id: str) -> PipelineExecutionResult:
        """
        恢复管道执行
        
        Args:
            pipeline_id: 管道ID
        
        Returns:
            执行结果
        """
        state = self.state_manager.load_state(pipeline_id)
        if not state:
            raise ValueError(f"管道状态不存在: {pipeline_id}")
        
        if state.status != PipelineStatus.PAUSED:
            raise ValueError(f"管道状态不是暂停: {state.status}")
        
        # 从当前阶段恢复
        resume_stage = state.current_stage
        if not resume_stage and state.completed_stages:
            # 找到下一个要执行的阶段
            execution_order = self._build_execution_order()
            for stage_name in execution_order:
                if stage_name not in state.completed_stages:
                    resume_stage = stage_name
                    break
        
        return self.execute(
            pipeline_id=pipeline_id,
            resume_from=resume_stage
        )
    
    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        取消管道执行
        
        Args:
            pipeline_id: 管道ID
        
        Returns:
            True 如果成功取消
        """
        state = self.state_manager.load_state(pipeline_id)
        if not state or state.is_terminal:
            return False
        
        state.status = PipelineStatus.CANCELLED
        state.end_time = datetime.now()
        self.state_manager.save_state(state)
        self.logger.info(f"管道已取消: {pipeline_id}")
        return True
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineStatus]:
        """
        获取管道状态
        
        Args:
            pipeline_id: 管道ID
        
        Returns:
            管道状态或None
        """
        state = self.state_manager.load_state(pipeline_id)
        return state.status if state else None
    
    def list_pipelines(
        self,
        status: Optional[PipelineStatus] = None
    ) -> List[PipelineState]:
        """
        列出管道
        
        Args:
            status: 按状态过滤
        
        Returns:
            管道状态列表
        """
        return self.state_manager.list_states(
            status=status,
            pipeline_name=self.config.name
        )
    
    def cleanup_old_states(self, max_age_days: int = 30) -> int:
        """
        清理旧的状态文件
        
        Args:
            max_age_days: 最大保留天数
        
        Returns:
            删除的文件数量
        """
        return self.state_manager.cleanup_old_states(max_age_days)


def create_pipeline_controller(
    config_path: Optional[str] = None,
    config: Optional[PipelineConfig] = None
) -> MLPipelineController:
    """
    创建管道控制器的工厂函数
    
    Args:
        config_path: 配置文件路径（与config二选一）
        config: 管道配置对象（与config_path二选一）
    
    Returns:
        MLPipelineController 实例
    
    Raises:
        ValueError: 配置参数错误
    
    Example:
        >>> # 从配置文件创建
        >>> controller = create_pipeline_controller("pipeline.yaml")
        >>> 
        >>> # 从配置对象创建
        >>> config = create_default_config()
        >>> controller = create_pipeline_controller(config=config)
    """
    if config_path and config:
        raise ValueError("不能同时指定 config_path 和 config")
    
    if config_path:
        from .pipeline_config import load_config
        config = load_config(config_path)
    elif config:
        config = config
    else:
        from .pipeline_config import create_default_config
        config = create_default_config()
    
    return MLPipelineController(config)
