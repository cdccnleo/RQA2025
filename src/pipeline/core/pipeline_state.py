"""
管道状态管理模块

提供管道执行状态的持久化、恢复和查询功能，支持断点续执行和故障恢复。

主要功能:
    - 管道状态持久化到文件/数据库
    - 状态恢复和断点续执行
    - 状态查询和监控
    - 历史执行记录管理

使用示例:
    >>> state_manager = PipelineStateManager("./states")
    >>> state = PipelineState(pipeline_id="123", status=PipelineStatus.RUNNING)
    >>> state_manager.save_state(state)
    >>> restored_state = state_manager.load_state("123")
"""

import json
import logging
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

from .pipeline_stage import StageResult, StageStatus


class PipelineStatus(Enum):
    """
    管道执行状态枚举
    
    定义管道在执行生命周期中的各种状态。
    
    Attributes:
        PENDING: 等待执行
        RUNNING: 执行中
        COMPLETED: 成功完成
        FAILED: 执行失败
        ROLLING_BACK: 回滚中
        ROLLED_BACK: 已回滚
        PAUSED: 已暂停
        CANCELLED: 已取消
    """
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ROLLING_BACK = auto()
    ROLLED_BACK = auto()
    PAUSED = auto()
    CANCELLED = auto()


@dataclass
class PipelineState:
    """
    管道状态数据类
    
    保存管道执行的完整状态信息，支持序列化和持久化。
    
    Attributes:
        pipeline_id: 管道实例唯一标识
        pipeline_name: 管道名称
        status: 当前执行状态
        current_stage: 当前执行的阶段名称
        completed_stages: 已完成的阶段列表
        stage_results: 各阶段执行结果
        context: 执行上下文数据
        start_time: 开始时间
        end_time: 结束时间
        metadata: 元数据信息
        error: 错误信息（如果有）
    
    Properties:
        duration_seconds: 执行时长
        progress_percentage: 执行进度百分比
        is_active: 是否处于活动状态
    
    Example:
        >>> state = PipelineState(
        ...     pipeline_id="pipe_001",
        ...     pipeline_name="ml_training",
        ...     status=PipelineStatus.RUNNING,
        ...     current_stage="model_training"
        ... )
        >>> print(f"进度: {state.progress_percentage}%")
    """
    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus = PipelineStatus.PENDING
    current_stage: Optional[str] = None
    completed_stages: List[str] = field(default_factory=list)
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """
        计算执行时长（秒）
        
        Returns:
            执行时长，如果开始时间为None则返回None
        """
        if self.start_time:
            end = self.end_time or datetime.now()
            return (end - self.start_time).total_seconds()
        return None
    
    @property
    def progress_percentage(self) -> float:
        """
        计算执行进度百分比
        
        Returns:
            进度百分比（0-100）
        """
        total_stages = self.metadata.get("total_stages", 0)
        if total_stages == 0:
            return 0.0
        return (len(self.completed_stages) / total_stages) * 100
    
    @property
    def is_active(self) -> bool:
        """
        检查是否处于活动状态
        
        Returns:
            True 如果状态为 RUNNING 或 PAUSED
        """
        return self.status in (PipelineStatus.RUNNING, PipelineStatus.PAUSED)
    
    @property
    def is_terminal(self) -> bool:
        """
        检查是否处于终止状态
        
        Returns:
            True 如果状态为 COMPLETED, FAILED, ROLLED_BACK 或 CANCELLED
        """
        return self.status in (
            PipelineStatus.COMPLETED,
            PipelineStatus.FAILED,
            PipelineStatus.ROLLED_BACK,
            PipelineStatus.CANCELLED
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将状态转换为字典格式
        
        Returns:
            可序列化的字典
        """
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status.name,
            "current_stage": self.current_stage,
            "completed_stages": self.completed_stages,
            "stage_results": {
                name: result.to_dict()
                for name, result in self.stage_results.items()
            },
            "context": self._serialize_context(self.context),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "progress_percentage": self.progress_percentage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        """
        从字典创建状态实例
        
        Args:
            data: 状态字典
        
        Returns:
            PipelineState 实例
        """
        # 解析阶段结果
        stage_results = {}
        for name, result_data in data.get("stage_results", {}).items():
            stage_results[name] = StageResult(
                stage_name=result_data.get("stage_name", name),
                status=StageStatus[result_data.get("status", "PENDING")],
                start_time=datetime.fromisoformat(result_data["start_time"]) if result_data.get("start_time") else None,
                end_time=datetime.fromisoformat(result_data["end_time"]) if result_data.get("end_time") else None,
                output=result_data.get("output", {}),
                metrics=result_data.get("metrics", {}),
                error=result_data.get("error"),
                logs=result_data.get("logs", [])
            )
        
        return cls(
            pipeline_id=data["pipeline_id"],
            pipeline_name=data.get("pipeline_name", "unknown"),
            status=PipelineStatus[data.get("status", "PENDING")],
            current_stage=data.get("current_stage"),
            completed_stages=data.get("completed_stages", []),
            stage_results=stage_results,
            context=data.get("context", {}),
            start_time=datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            metadata=data.get("metadata", {}),
            error=data.get("error")
        )
    
    def _serialize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        序列化上下文（处理不可序列化的对象）
        
        Args:
            context: 原始上下文
        
        Returns:
            可序列化的上下文
        """
        serialized = {}
        for key, value in context.items():
            try:
                # 尝试JSON序列化
                json.dumps({key: value})
                serialized[key] = value
            except (TypeError, ValueError):
                # 不可序列化的对象，保存类型信息
                serialized[key] = {
                    "_type": type(value).__name__,
                    "_module": type(value).__module__,
                    "_repr": repr(value)[:200]  # 限制长度
                }
        return serialized
    
    def update_stage_result(self, stage_name: str, result: StageResult) -> None:
        """
        更新阶段执行结果
        
        Args:
            stage_name: 阶段名称
            result: 阶段执行结果
        """
        self.stage_results[stage_name] = result
        if result.is_success and stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)
    
    def mark_stage_start(self, stage_name: str) -> None:
        """
        标记阶段开始
        
        Args:
            stage_name: 阶段名称
        """
        self.current_stage = stage_name
        self.status = PipelineStatus.RUNNING
        if not self.start_time:
            self.start_time = datetime.now()
    
    def mark_stage_complete(self, stage_name: str) -> None:
        """
        标记阶段完成
        
        Args:
            stage_name: 阶段名称
        """
        if stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)
        self.current_stage = None
    
    def mark_failed(self, error: str) -> None:
        """
        标记管道失败
        
        Args:
            error: 错误信息
        """
        self.status = PipelineStatus.FAILED
        self.error = error
        self.end_time = datetime.now()
    
    def mark_completed(self) -> None:
        """标记管道完成"""
        self.status = PipelineStatus.COMPLETED
        self.current_stage = None
        self.end_time = datetime.now()


class PipelineStateManager:
    """
    管道状态管理器
    
    管理管道状态的持久化、恢复和查询，支持文件系统存储。
    
    Attributes:
        storage_path: 状态存储路径
        logger: 日志记录器
    
    Example:
        >>> manager = PipelineStateManager("./pipeline_states")
        >>> state = manager.create_state("ml_pipeline")
        >>> manager.save_state(state)
        >>> loaded = manager.load_state(state.pipeline_id)
    """
    
    def __init__(self, storage_path: Union[str, Path] = "./pipeline_states"):
        """
        初始化状态管理器
        
        Args:
            storage_path: 状态存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("pipeline.state_manager")
        self.logger.info(f"状态管理器初始化，存储路径: {self.storage_path}")
    
    def create_state(
        self,
        pipeline_name: str,
        pipeline_id: Optional[str] = None,
        total_stages: int = 0
    ) -> PipelineState:
        """
        创建新的管道状态
        
        Args:
            pipeline_name: 管道名称
            pipeline_id: 管道ID（自动生成如果不提供）
            total_stages: 总阶段数
        
        Returns:
            新创建的管道状态
        """
        import uuid
        
        state = PipelineState(
            pipeline_id=pipeline_id or str(uuid.uuid4()),
            pipeline_name=pipeline_name,
            status=PipelineStatus.PENDING,
            start_time=datetime.now(),
            metadata={"total_stages": total_stages}
        )
        
        self.logger.info(f"创建管道状态: {state.pipeline_id}")
        return state
    
    def save_state(self, state: PipelineState, filename: Optional[str] = None) -> Path:
        """
        保存管道状态到文件
        
        Args:
            state: 管道状态
            filename: 文件名（默认使用 pipeline_id.json）
        
        Returns:
            保存的文件路径
        """
        filename = filename or f"{state.pipeline_id}.json"
        file_path = self.storage_path / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"状态已保存: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
            raise
    
    def load_state(self, pipeline_id: str) -> Optional[PipelineState]:
        """
        从文件加载管道状态
        
        Args:
            pipeline_id: 管道ID
        
        Returns:
            管道状态，如果不存在返回 None
        """
        file_path = self.storage_path / f"{pipeline_id}.json"
        
        if not file_path.exists():
            self.logger.warning(f"状态文件不存在: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = PipelineState.from_dict(data)
            self.logger.debug(f"状态已加载: {pipeline_id}")
            return state
        except Exception as e:
            self.logger.error(f"加载状态失败: {e}")
            return None
    
    def list_states(
        self,
        status: Optional[PipelineStatus] = None,
        pipeline_name: Optional[str] = None
    ) -> List[PipelineState]:
        """
        列出所有管道状态
        
        Args:
            status: 按状态过滤
            pipeline_name: 按名称过滤
        
        Returns:
            管道状态列表
        """
        states = []
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                state = PipelineState.from_dict(data)
                
                # 应用过滤条件
                if status and state.status != status:
                    continue
                if pipeline_name and state.pipeline_name != pipeline_name:
                    continue
                
                states.append(state)
            except Exception as e:
                self.logger.warning(f"读取状态文件失败 {file_path}: {e}")
        
        # 按开始时间排序
        states.sort(key=lambda s: s.start_time or datetime.min, reverse=True)
        return states
    
    def delete_state(self, pipeline_id: str) -> bool:
        """
        删除管道状态
        
        Args:
            pipeline_id: 管道ID
        
        Returns:
            True 如果删除成功
        """
        file_path = self.storage_path / f"{pipeline_id}.json"
        
        if not file_path.exists():
            return False
        
        try:
            file_path.unlink()
            self.logger.info(f"状态已删除: {pipeline_id}")
            return True
        except Exception as e:
            self.logger.error(f"删除状态失败: {e}")
            return False
    
    def get_latest_state(
        self,
        pipeline_name: Optional[str] = None
    ) -> Optional[PipelineState]:
        """
        获取最新的管道状态
        
        Args:
            pipeline_name: 管道名称过滤
        
        Returns:
            最新的管道状态
        """
        states = self.list_states(pipeline_name=pipeline_name)
        return states[0] if states else None
    
    def get_resumable_states(self) -> List[PipelineState]:
        """
        获取可恢复的状态（失败或暂停的管道）
        
        Returns:
            可恢复的管道状态列表
        """
        resumable_statuses = [
            PipelineStatus.FAILED,
            PipelineStatus.PAUSED,
            PipelineStatus.ROLLING_BACK
        ]
        
        states = []
        for status in resumable_statuses:
            states.extend(self.list_states(status=status))
        
        return states
    
    def cleanup_old_states(self, max_age_days: int = 30) -> int:
        """
        清理旧的状态文件
        
        Args:
            max_age_days: 最大保留天数
        
        Returns:
            删除的文件数量
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        deleted_count = 0
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                # 检查文件修改时间
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
                    self.logger.debug(f"清理旧状态文件: {file_path}")
            except Exception as e:
                self.logger.warning(f"清理文件失败 {file_path}: {e}")
        
        self.logger.info(f"清理完成，删除 {deleted_count} 个旧状态文件")
        return deleted_count
    
    def export_state(
        self,
        pipeline_id: str,
        export_path: Union[str, Path]
    ) -> bool:
        """
        导出状态到指定路径
        
        Args:
            pipeline_id: 管道ID
            export_path: 导出路径
        
        Returns:
            True 如果导出成功
        """
        state = self.load_state(pipeline_id)
        if not state:
            return False
        
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"导出状态失败: {e}")
            return False


class PipelineCheckpoint:
    """
    管道检查点类
    
    在关键节点创建检查点，支持从检查点恢复执行。
    
    Attributes:
        state: 管道状态
        checkpoint_name: 检查点名称
        created_at: 创建时间
    """
    
    def __init__(self, state: PipelineState, checkpoint_name: str):
        """
        初始化检查点
        
        Args:
            state: 管道状态
            checkpoint_name: 检查点名称
        """
        self.state = state
        self.checkpoint_name = checkpoint_name
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "checkpoint_name": self.checkpoint_name,
            "created_at": self.created_at.isoformat(),
            "state": self.state.to_dict()
        }
