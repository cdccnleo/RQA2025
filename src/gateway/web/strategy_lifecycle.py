"""
策略生命周期管理模块
提供策略从设计到实盘交易的完整生命周期管理
"""

import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# 策略生命周期状态
class LifecycleStatus(Enum):
    """策略生命周期状态枚举"""
    DRAFT = "draft"                    # 草稿
    DESIGN = "design"                  # 设计中
    BACKTESTING = "backtesting"        # 回测中
    OPTIMIZING = "optimizing"          # 优化中
    PAPER_TRADING = "paper_trading"    # 模拟交易中
    LIVE_TRADING = "live_trading"      # 实盘交易中
    PAUSED = "paused"                  # 暂停
    ARCHIVED = "archived"              # 已归档
    FAILED = "failed"                  # 失败

# 生命周期状态转换规则
LIFECYCLE_TRANSITIONS = {
    LifecycleStatus.DRAFT: [LifecycleStatus.DESIGN],
    LifecycleStatus.DESIGN: [LifecycleStatus.BACKTESTING, LifecycleStatus.DRAFT],
    LifecycleStatus.BACKTESTING: [LifecycleStatus.OPTIMIZING, LifecycleStatus.PAPER_TRADING, LifecycleStatus.FAILED],
    LifecycleStatus.OPTIMIZING: [LifecycleStatus.BACKTESTING, LifecycleStatus.PAPER_TRADING, LifecycleStatus.FAILED],
    LifecycleStatus.PAPER_TRADING: [LifecycleStatus.LIVE_TRADING, LifecycleStatus.PAUSED, LifecycleStatus.FAILED],
    LifecycleStatus.LIVE_TRADING: [LifecycleStatus.PAUSED, LifecycleStatus.ARCHIVED, LifecycleStatus.FAILED],
    LifecycleStatus.PAUSED: [LifecycleStatus.PAPER_TRADING, LifecycleStatus.LIVE_TRADING, LifecycleStatus.ARCHIVED],
    LifecycleStatus.FAILED: [LifecycleStatus.DESIGN, LifecycleStatus.BACKTESTING, LifecycleStatus.OPTIMIZING],
    LifecycleStatus.ARCHIVED: []  # 终态
}


@dataclass
class LifecycleEvent:
    """生命周期事件"""
    event_id: str
    event_type: str
    from_status: Optional[str]
    to_status: str
    timestamp: float
    operator: str = "system"
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StrategyLifecycle:
    """策略生命周期实例"""
    strategy_id: str
    strategy_name: str
    current_status: LifecycleStatus
    events: List[LifecycleEvent] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 各阶段时间记录
    design_completed_at: Optional[float] = None
    backtest_completed_at: Optional[float] = None
    optimize_completed_at: Optional[float] = None
    paper_trading_started_at: Optional[float] = None
    live_trading_started_at: Optional[float] = None
    archived_at: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'current_status': self.current_status.value,
            'events': [e.to_dict() for e in self.events],
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metadata': self.metadata,
            'design_completed_at': self.design_completed_at,
            'backtest_completed_at': self.backtest_completed_at,
            'optimize_completed_at': self.optimize_completed_at,
            'paper_trading_started_at': self.paper_trading_started_at,
            'live_trading_started_at': self.live_trading_started_at,
            'archived_at': self.archived_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyLifecycle':
        lifecycle = cls(
            strategy_id=data['strategy_id'],
            strategy_name=data['strategy_name'],
            current_status=LifecycleStatus(data['current_status']),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            metadata=data.get('metadata', {}),
            design_completed_at=data.get('design_completed_at'),
            backtest_completed_at=data.get('backtest_completed_at'),
            optimize_completed_at=data.get('optimize_completed_at'),
            paper_trading_started_at=data.get('paper_trading_started_at'),
            live_trading_started_at=data.get('live_trading_started_at'),
            archived_at=data.get('archived_at')
        )
        lifecycle.events = [LifecycleEvent(**e) for e in data.get('events', [])]
        return cls


class StrategyLifecycleManager:
    """策略生命周期管理器"""
    
    def __init__(self, lifecycle_dir: str = "data/lifecycles"):
        self.lifecycle_dir = lifecycle_dir
        self.active_lifecycles: Dict[str, StrategyLifecycle] = {}
        self._ensure_directory()
        self._load_active_lifecycles()
    
    def _ensure_directory(self):
        """确保生命周期目录存在"""
        if not os.path.exists(self.lifecycle_dir):
            os.makedirs(self.lifecycle_dir)
            logger.info(f"创建生命周期目录: {self.lifecycle_dir}")
    
    def _load_active_lifecycles(self):
        """加载活跃的生命周期"""
        try:
            for filename in os.listdir(self.lifecycle_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.lifecycle_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            lifecycle = StrategyLifecycle.from_dict(data)
                            # 只加载未归档的
                            if lifecycle.current_status != LifecycleStatus.ARCHIVED:
                                self.active_lifecycles[lifecycle.strategy_id] = lifecycle
                    except Exception as e:
                        logger.warning(f"加载生命周期文件失败 {filename}: {e}")
        except Exception as e:
            logger.error(f"加载生命周期目录失败: {e}")
    
    def _save_lifecycle(self, lifecycle: StrategyLifecycle):
        """保存生命周期到文件"""
        try:
            filepath = os.path.join(self.lifecycle_dir, f"{lifecycle.strategy_id}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(lifecycle.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存生命周期失败 {lifecycle.strategy_id}: {e}")
    
    def create_lifecycle(self, strategy_id: str, strategy_name: str,
                         initial_status: LifecycleStatus = LifecycleStatus.DRAFT) -> StrategyLifecycle:
        """
        创建新策略生命周期

        Args:
            strategy_id: 策略ID
            strategy_name: 策略名称
            initial_status: 初始状态，默认为 DRAFT
        """
        lifecycle = StrategyLifecycle(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            current_status=initial_status
        )

        # 添加初始事件
        lifecycle.events.append(LifecycleEvent(
            event_id=f"evt_{int(time.time())}",
            event_type="create",
            from_status=None,
            to_status=initial_status.value,
            timestamp=time.time(),
            reason="策略创建"
        ))

        self.active_lifecycles[strategy_id] = lifecycle
        self._save_lifecycle(lifecycle)

        logger.info(f"创建策略生命周期: {strategy_id}, 初始状态: {initial_status.value}")
        return lifecycle
    
    def get_lifecycle(self, strategy_id: str) -> Optional[StrategyLifecycle]:
        """获取策略生命周期"""
        return self.active_lifecycles.get(strategy_id)
    
    def can_transition(self, lifecycle: StrategyLifecycle, new_status: LifecycleStatus) -> bool:
        """检查状态转换是否允许"""
        return new_status in LIFECYCLE_TRANSITIONS.get(lifecycle.current_status, [])
    
    def transition_status(self, strategy_id: str, new_status: LifecycleStatus, 
                         operator: str = "system", reason: str = "") -> bool:
        """转换生命周期状态"""
        lifecycle = self.get_lifecycle(strategy_id)
        if not lifecycle:
            logger.error(f"生命周期不存在: {strategy_id}")
            return False
        
        if not self.can_transition(lifecycle, new_status):
            logger.warning(f"状态转换不允许: {lifecycle.current_status.value} -> {new_status.value}")
            return False
        
        old_status = lifecycle.current_status
        lifecycle.current_status = new_status
        lifecycle.updated_at = time.time()
        
        # 记录事件
        lifecycle.events.append(LifecycleEvent(
            event_id=f"evt_{int(time.time())}",
            event_type="transition",
            from_status=old_status.value,
            to_status=new_status.value,
            timestamp=time.time(),
            operator=operator,
            reason=reason
        ))
        
        # 更新各阶段时间
        if new_status == LifecycleStatus.DESIGN:
            pass  # 设计阶段开始
        elif new_status == LifecycleStatus.BACKTESTING:
            pass  # 回测阶段开始
        elif new_status == LifecycleStatus.OPTIMIZING:
            pass  # 优化阶段开始
        elif new_status == LifecycleStatus.PAPER_TRADING:
            lifecycle.paper_trading_started_at = time.time()
        elif new_status == LifecycleStatus.LIVE_TRADING:
            lifecycle.live_trading_started_at = time.time()
        elif new_status == LifecycleStatus.ARCHIVED:
            lifecycle.archived_at = time.time()
            if strategy_id in self.active_lifecycles:
                del self.active_lifecycles[strategy_id]
            
            # 停止策略执行
            self._stop_strategy_execution(strategy_id, operator, reason)
        
        self._save_lifecycle(lifecycle)
        
        logger.info(f"策略生命周期状态转换: {strategy_id} {old_status.value} -> {new_status.value}")
        return True
    
    def _stop_strategy_execution(self, strategy_id: str, operator: str = "system", reason: str = ""):
        """
        停止策略执行
        
        在策略退市时调用，确保策略从实时引擎中停止并更新执行状态
        """
        try:
            import asyncio
            
            # 1. 尝试从实时引擎停止策略
            try:
                from .strategy_execution_service import get_realtime_engine
                engine = asyncio.run(get_realtime_engine())
                if engine and strategy_id in engine.strategies:
                    strategy = engine.strategies[strategy_id]
                    if hasattr(strategy, 'is_active'):
                        strategy.is_active = False
                    # 从引擎中注销策略
                    if hasattr(engine, 'unregister_strategy'):
                        engine.unregister_strategy(strategy_id)
                    elif hasattr(engine, 'strategies') and strategy_id in engine.strategies:
                        del engine.strategies[strategy_id]
                    logger.info(f"策略 {strategy_id} 已从实时引擎停止并注销")
            except Exception as e:
                logger.warning(f"从实时引擎停止策略失败 {strategy_id}: {e}")
            
            # 2. 更新执行状态持久化存储
            try:
                from .execution_persistence import load_execution_state, save_execution_state
                exec_state = load_execution_state(strategy_id)
                if exec_state:
                    exec_state['status'] = 'stopped'
                    exec_state['stopped_at'] = time.time()
                    exec_state['stopped_by'] = operator
                    exec_state['stop_reason'] = reason or '策略退市'
                    save_execution_state(strategy_id, exec_state)
                    logger.info(f"策略 {strategy_id} 执行状态已更新为 stopped")
                else:
                    # 如果执行状态不存在，创建一个停止状态
                    save_execution_state(strategy_id, {
                        'strategy_id': strategy_id,
                        'name': strategy_id,
                        'status': 'stopped',
                        'stopped_at': time.time(),
                        'stopped_by': operator,
                        'stop_reason': reason or '策略退市'
                    })
                    logger.info(f"策略 {strategy_id} 已创建停止状态")
            except Exception as e:
                logger.warning(f"更新执行状态失败 {strategy_id}: {e}")
                
        except Exception as e:
            logger.error(f"停止策略执行失败 {strategy_id}: {e}")
    
    def get_lifecycle_timeline(self, strategy_id: str) -> List[Dict]:
        """获取策略生命周期时间线"""
        lifecycle = self.get_lifecycle(strategy_id)
        if not lifecycle:
            return []
        
        timeline = []
        for event in lifecycle.events:
            timeline.append({
                'time': datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                'event': event.event_type,
                'from': event.from_status,
                'to': event.to_status,
                'operator': event.operator,
                'reason': event.reason
            })
        
        return timeline
    
    def get_lifecycle_stats(self, strategy_id: str) -> Dict:
        """获取策略生命周期统计"""
        lifecycle = self.get_lifecycle(strategy_id)
        if not lifecycle:
            return {"error": "生命周期不存在"}
        
        # 计算各阶段耗时
        design_time = 0
        backtest_time = 0
        optimize_time = 0
        paper_trading_time = 0
        
        if lifecycle.design_completed_at and lifecycle.created_at:
            design_time = lifecycle.design_completed_at - lifecycle.created_at
        
        if lifecycle.backtest_completed_at and lifecycle.design_completed_at:
            backtest_time = lifecycle.backtest_completed_at - lifecycle.design_completed_at
        
        if lifecycle.optimize_completed_at and lifecycle.backtest_completed_at:
            optimize_time = lifecycle.optimize_completed_at - lifecycle.backtest_completed_at
        
        if lifecycle.paper_trading_started_at:
            if lifecycle.live_trading_started_at:
                paper_trading_time = lifecycle.live_trading_started_at - lifecycle.paper_trading_started_at
            else:
                paper_trading_time = time.time() - lifecycle.paper_trading_started_at
        
        return {
            'strategy_id': strategy_id,
            'strategy_name': lifecycle.strategy_name,
            'current_status': lifecycle.current_status.value,
            'created_at': lifecycle.created_at,
            'updated_at': lifecycle.updated_at,
            'total_duration': time.time() - lifecycle.created_at,
            'stage_durations': {
                'design': design_time,
                'backtest': backtest_time,
                'optimize': optimize_time,
                'paper_trading': paper_trading_time
            },
            'event_count': len(lifecycle.events)
        }
    
    def list_lifecycles(self, status: Optional[LifecycleStatus] = None) -> List[StrategyLifecycle]:
        """列出生命周期"""
        lifecycles = list(self.active_lifecycles.values())
        
        if status:
            lifecycles = [l for l in lifecycles if l.current_status == status]
        
        return sorted(lifecycles, key=lambda l: l.updated_at, reverse=True)
    
    def archive_strategy(self, strategy_id: str, reason: str = "") -> bool:
        """归档策略"""
        return self.transition_status(
            strategy_id, 
            LifecycleStatus.ARCHIVED, 
            operator="user", 
            reason=reason or "策略归档"
        )
    
    def pause_strategy(self, strategy_id: str, reason: str = "") -> bool:
        """暂停策略"""
        return self.transition_status(
            strategy_id, 
            LifecycleStatus.PAUSED, 
            operator="user", 
            reason=reason or "策略暂停"
        )
    
    def resume_strategy(self, strategy_id: str, target_status: LifecycleStatus, reason: str = "") -> bool:
        """恢复策略"""
        return self.transition_status(
            strategy_id, 
            target_status, 
            operator="user", 
            reason=reason or "策略恢复"
        )


# 全局生命周期管理器实例
lifecycle_manager = StrategyLifecycleManager()


# 便捷的API函数
def create_strategy_lifecycle(strategy_id: str, strategy_name: str) -> StrategyLifecycle:
    """创建策略生命周期"""
    return lifecycle_manager.create_lifecycle(strategy_id, strategy_name)


def get_strategy_lifecycle(strategy_id: str) -> Optional[StrategyLifecycle]:
    """获取策略生命周期"""
    return lifecycle_manager.get_lifecycle(strategy_id)


def transition_strategy_status(strategy_id: str, new_status: str,
                               operator: str = "system", reason: str = "") -> bool:
    """
    转换策略状态

    如果策略生命周期不存在，会自动创建一个默认生命周期：
    - 如果策略正在运行，初始状态为 LIVE_TRADING
    - 否则初始状态为 DRAFT

    特殊处理：如果要退市（archived）但当前状态不允许直接转换，
    会先转换到 LIVE_TRADING，然后再退市
    """
    try:
        status = LifecycleStatus(new_status)

        # 检查生命周期是否存在，如果不存在则创建
        lifecycle = lifecycle_manager.get_lifecycle(strategy_id)
        if not lifecycle:
            logger.warning(f"策略 {strategy_id} 生命周期不存在，尝试从执行状态获取信息")

            # 尝试从执行状态获取策略信息
            initial_status = LifecycleStatus.DRAFT
            strategy_name = strategy_id

            try:
                from .execution_persistence import load_execution_state
                exec_state = load_execution_state(strategy_id)
                if exec_state:
                    strategy_name = exec_state.get('name', strategy_id)
                    # 如果策略正在运行，设置为 LIVE_TRADING 以便可以退市
                    if exec_state.get('status') == 'running':
                        initial_status = LifecycleStatus.LIVE_TRADING
                        logger.info(f"策略 {strategy_id} 正在运行，设置初始状态为 LIVE_TRADING")
            except Exception as e:
                logger.debug(f"从执行状态获取策略信息失败: {e}")

            lifecycle = lifecycle_manager.create_lifecycle(
                strategy_id, strategy_name, initial_status
            )

        # 检查是否可以直接转换
        if not lifecycle_manager.can_transition(lifecycle, status):
            # 如果要退市但当前状态不允许，需要通过完整路径转换到 LIVE_TRADING
            if status == LifecycleStatus.ARCHIVED:
                logger.warning(f"策略 {strategy_id} 当前状态 {lifecycle.current_status.value} 不能直接退市，需要通过完整路径转换")
                
                # 定义完整转换路径：DRAFT -> DESIGN -> BACKTESTING -> PAPER_TRADING -> LIVE_TRADING -> ARCHIVED
                transition_path = [
                    (LifecycleStatus.DESIGN, "进入设计阶段"),
                    (LifecycleStatus.BACKTESTING, "进入回测阶段"),
                    (LifecycleStatus.PAPER_TRADING, "进入模拟交易"),
                    (LifecycleStatus.LIVE_TRADING, "进入实盘交易")
                ]
                
                # 获取当前生命周期对象（可能在之前的转换中已经更新）
                current_lifecycle = lifecycle_manager.get_lifecycle(strategy_id)
                current_status = current_lifecycle.current_status if current_lifecycle else LifecycleStatus.DRAFT
                
                # 按路径逐步转换
                for target_status, reason_text in transition_path:
                    # 跳过已经经过的状态
                    if current_status == target_status:
                        continue
                    
                    # 检查是否可以直接转换到目标状态
                    if lifecycle_manager.can_transition(current_lifecycle, target_status):
                        if not lifecycle_manager.transition_status(strategy_id, target_status, operator, reason_text):
                            logger.error(f"策略 {strategy_id} 转换到 {target_status.value} 失败")
                            return False
                        logger.info(f"策略 {strategy_id} 已转换到 {target_status.value}")
                        
                        # 更新当前生命周期对象和状态
                        current_lifecycle = lifecycle_manager.get_lifecycle(strategy_id)
                        current_status = current_lifecycle.current_status
                    else:
                        logger.warning(f"策略 {strategy_id} 无法从 {current_status.value} 转换到 {target_status.value}")
                        # 尝试找到中间状态
                        # 如果当前是 DRAFT，目标是 BACKTESTING，需要先经过 DESIGN
                        if current_status == LifecycleStatus.DRAFT and target_status == LifecycleStatus.BACKTESTING:
                            if lifecycle_manager.can_transition(current_lifecycle, LifecycleStatus.DESIGN):
                                if not lifecycle_manager.transition_status(strategy_id, LifecycleStatus.DESIGN, operator, "准备进入回测"):
                                    logger.error(f"策略 {strategy_id} 转换到 DESIGN 失败")
                                    return False
                                logger.info(f"策略 {strategy_id} 已转换到 DESIGN")
                                current_lifecycle = lifecycle_manager.get_lifecycle(strategy_id)
                                current_status = current_lifecycle.current_status
                                
                                # 再次尝试转换到 BACKTESTING
                                if lifecycle_manager.can_transition(current_lifecycle, LifecycleStatus.BACKTESTING):
                                    if not lifecycle_manager.transition_status(strategy_id, LifecycleStatus.BACKTESTING, operator, "进入回测阶段"):
                                        logger.error(f"策略 {strategy_id} 转换到 BACKTESTING 失败")
                                        return False
                                    logger.info(f"策略 {strategy_id} 已转换到 BACKTESTING")
                                    current_lifecycle = lifecycle_manager.get_lifecycle(strategy_id)
                                    current_status = current_lifecycle.current_status
                        else:
                            logger.error(f"策略 {strategy_id} 无法完成状态转换路径")
                            return False
                
                # 现在可以退市了
                return lifecycle_manager.transition_status(strategy_id, status, operator, reason)
            else:
                logger.warning(f"状态转换不允许: {lifecycle.current_status.value} -> {new_status}")
                return False

        return lifecycle_manager.transition_status(strategy_id, status, operator, reason)
    except ValueError:
        logger.error(f"无效的生命周期状态: {new_status}")
        return False


def get_strategy_lifecycle_timeline(strategy_id: str) -> List[Dict]:
    """获取策略生命周期时间线"""
    return lifecycle_manager.get_lifecycle_timeline(strategy_id)


def get_strategy_lifecycle_stats(strategy_id: str) -> Dict:
    """获取策略生命周期统计"""
    return lifecycle_manager.get_lifecycle_stats(strategy_id)
