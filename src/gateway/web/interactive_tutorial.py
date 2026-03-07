"""
交互式教程系统模块

功能：
- 引导式用户教程
- 步骤化学习路径
- 交互式提示
- 进度跟踪
- 成就系统
- 上下文帮助

技术栈：
- dataclasses: 数据模型
- json: 教程配置
- typing: 类型提示

作者: Claude
创建日期: 2026-02-21
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TutorialType(Enum):
    """教程类型"""
    ONBOARDING = "onboarding"           # 新手引导
    FEATURE_GUIDE = "feature_guide"     # 功能指南
    WORKFLOW_TUTORIAL = "workflow"      # 工作流教程
    QUICK_START = "quick_start"         # 快速入门
    ADVANCED = "advanced"               # 高级教程


class StepType(Enum):
    """步骤类型"""
    INTRO = "intro"                     # 介绍
    HIGHLIGHT = "highlight"             # 高亮元素
    INPUT = "input"                     # 用户输入
    ACTION = "action"                   # 执行操作
    CONFIRM = "confirm"                 # 确认
    COMPLETION = "completion"           # 完成


class TutorialStatus(Enum):
    """教程状态"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class TutorialStep:
    """教程步骤"""
    step_id: str
    step_number: int
    step_type: StepType
    title: str
    content: str
    target_element: Optional[str]        # CSS选择器
    position: str = "bottom"             # 提示位置
    action_required: bool = False
    action_type: Optional[str] = None    # click, input, etc.
    validation: Optional[str] = None     # 验证规则
    next_step: Optional[str] = None
    skippable: bool = True
    delay_ms: int = 0


@dataclass
class Tutorial:
    """教程定义"""
    tutorial_id: str
    tutorial_type: TutorialType
    title: str
    description: str
    steps: List[TutorialStep]
    estimated_duration_minutes: int
    difficulty: str = "beginner"
    prerequisites: List[str] = field(default_factory=list)
    rewards: List[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class UserProgress:
    """用户进度"""
    user_id: str
    tutorial_id: str
    status: TutorialStatus
    current_step: int
    completed_steps: List[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_time_seconds: int
    achievements: List[str]


@dataclass
class Achievement:
    """成就"""
    achievement_id: str
    name: str
    description: str
    icon: str
    condition: str
    reward_points: int


class TutorialLibrary:
    """教程库"""
    
    def __init__(self):
        self.tutorials: Dict[str, Tutorial] = {}
        self.achievements: Dict[str, Achievement] = {}
        self._init_default_tutorials()
        self._init_default_achievements()
    
    def _init_default_tutorials(self) -> None:
        """初始化默认教程"""
        # 新手入门教程
        onboarding_steps = [
            TutorialStep(
                step_id="welcome",
                step_number=1,
                step_type=StepType.INTRO,
                title="欢迎使用RQA2025",
                content="这是一个专业的量化投资研究平台。让我们通过几个简单步骤了解主要功能。",
                position="center",
                skippable=False
            ),
            TutorialStep(
                step_id="dashboard_intro",
                step_number=2,
                step_type=StepType.HIGHLIGHT,
                title="仪表盘",
                content="这是您的个性化仪表盘，可以自定义显示您关注的数据和图表。",
                target_element="#dashboard-container",
                position="bottom"
            ),
            TutorialStep(
                step_id="market_data",
                step_number=3,
                step_type=StepType.HIGHLIGHT,
                title="市场数据",
                content="在这里查看实时行情、股票信息和技术指标。",
                target_element="#market-data-panel",
                position="right"
            ),
            TutorialStep(
                step_id="strategy_center",
                step_number=4,
                step_type=StepType.HIGHLIGHT,
                title="策略中心",
                content="创建、回测和部署您的交易策略。",
                target_element="#strategy-center",
                position="right"
            ),
            TutorialStep(
                step_id="natural_language",
                step_number=5,
                step_type=StepType.ACTION,
                title="自然语言查询",
                content="试试用自然语言查询数据，例如：'查询茅台最近一周的行情'",
                target_element="#nlq-input",
                position="bottom",
                action_required=True,
                action_type="input"
            ),
            TutorialStep(
                step_id="complete",
                step_number=6,
                step_type=StepType.COMPLETION,
                title="恭喜完成！",
                content="您已完成新手引导。现在可以开始探索更多功能了！",
                position="center",
                skippable=False
            )
        ]
        
        self.tutorials["onboarding"] = Tutorial(
            tutorial_id="onboarding",
            tutorial_type=TutorialType.ONBOARDING,
            title="新手入门",
            description="了解平台基本功能和界面布局",
            steps=onboarding_steps,
            estimated_duration_minutes=5,
            difficulty="beginner",
            rewards=["first_step", "explorer"]
        )
        
        # 策略创建教程
        strategy_steps = [
            TutorialStep(
                step_id="strategy_intro",
                step_number=1,
                step_type=StepType.INTRO,
                title="创建您的第一个策略",
                content="学习如何创建、配置和回测交易策略。",
                position="center"
            ),
            TutorialStep(
                step_id="create_strategy",
                step_number=2,
                step_type=StepType.ACTION,
                title="新建策略",
                content="点击'新建策略'按钮开始创建。",
                target_element="#btn-create-strategy",
                position="bottom",
                action_required=True,
                action_type="click"
            ),
            TutorialStep(
                step_id="configure_params",
                step_number=3,
                step_type=StepType.INPUT,
                title="配置参数",
                content="设置策略的基本参数，如股票池、回测时间范围等。",
                target_element="#strategy-config-panel",
                position="right"
            ),
            TutorialStep(
                step_id="run_backtest",
                step_number=4,
                step_type=StepType.ACTION,
                title="运行回测",
                content="点击'运行回测'查看策略表现。",
                target_element="#btn-run-backtest",
                position="bottom",
                action_required=True,
                action_type="click"
            ),
            TutorialStep(
                step_id="analyze_results",
                step_number=5,
                step_type=StepType.HIGHLIGHT,
                title="分析结果",
                content="查看回测报告，包括收益曲线、风险指标等。",
                target_element="#backtest-results",
                position="left"
            ),
            TutorialStep(
                step_id="strategy_complete",
                step_number=6,
                step_type=StepType.COMPLETION,
                title="策略创建完成！",
                content="您已成功创建并回测了一个策略。可以尝试部署到模拟交易环境。",
                position="center"
            )
        ]
        
        self.tutorials["create_strategy"] = Tutorial(
            tutorial_id="create_strategy",
            tutorial_type=TutorialType.WORKFLOW_TUTORIAL,
            title="创建交易策略",
            description="学习如何创建和回测交易策略",
            steps=strategy_steps,
            estimated_duration_minutes=10,
            difficulty="intermediate",
            prerequisites=["onboarding"],
            rewards=["strategy_creator"]
        )
        
        # 自然语言查询教程
        nlq_steps = [
            TutorialStep(
                step_id="nlq_intro",
                step_number=1,
                step_type=StepType.INTRO,
                title="自然语言查询",
                content="学习如何使用自然语言快速查询数据。",
                position="center"
            ),
            TutorialStep(
                step_id="nlq_example_1",
                step_number=2,
                step_type=StepType.ACTION,
                title="查询股票信息",
                content="输入：'查询茅台的市盈率'",
                target_element="#nlq-input",
                position="bottom",
                action_required=True,
                action_type="input"
            ),
            TutorialStep(
                step_id="nlq_example_2",
                step_number=3,
                step_type=StepType.ACTION,
                title="查询行情",
                content="输入：'腾讯最近一周的走势'",
                target_element="#nlq-input",
                position="bottom",
                action_required=True,
                action_type="input"
            ),
            TutorialStep(
                step_id="nlq_example_3",
                step_number=4,
                step_type=StepType.ACTION,
                title="对比分析",
                content="输入：'对比茅台和五粮液的财务指标'",
                target_element="#nlq-input",
                position="bottom",
                action_required=True,
                action_type="input"
            ),
            TutorialStep(
                step_id="nlq_complete",
                step_number=5,
                step_type=StepType.COMPLETION,
                title="掌握自然语言查询！",
                content="您已学会使用自然语言查询数据。支持的查询类型包括：股票信息、行情走势、技术指标、财务数据等。",
                position="center"
            )
        ]
        
        self.tutorials["natural_language_query"] = Tutorial(
            tutorial_id="natural_language_query",
            tutorial_type=TutorialType.FEATURE_GUIDE,
            title="自然语言查询指南",
            description="学习使用自然语言查询数据",
            steps=nlq_steps,
            estimated_duration_minutes=5,
            difficulty="beginner",
            rewards=["nlq_master"]
        )
    
    def _init_default_achievements(self) -> None:
        """初始化默认成就"""
        self.achievements["first_step"] = Achievement(
            achievement_id="first_step",
            name="第一步",
            description="完成新手入门教程",
            icon="🏃",
            condition="complete_tutorial:onboarding",
            reward_points=10
        )
        
        self.achievements["explorer"] = Achievement(
            achievement_id="explorer",
            name="探索者",
            description="探索平台所有主要功能",
            icon="🔍",
            condition="complete_tutorial:onboarding",
            reward_points=20
        )
        
        self.achievements["strategy_creator"] = Achievement(
            achievement_id="strategy_creator",
            name="策略师",
            description="创建第一个交易策略",
            icon="📊",
            condition="complete_tutorial:create_strategy",
            reward_points=50
        )
        
        self.achievements["nlq_master"] = Achievement(
            achievement_id="nlq_master",
            name="查询大师",
            description="掌握自然语言查询功能",
            icon="💬",
            condition="complete_tutorial:natural_language_query",
            reward_points=30
        )
        
        self.achievements["quick_learner"] = Achievement(
            achievement_id="quick_learner",
            name="快速学习者",
            description="一天内完成3个教程",
            icon="⚡",
            condition="complete_3_tutorials_in_one_day",
            reward_points=100
        )
    
    def get_tutorial(self, tutorial_id: str) -> Optional[Tutorial]:
        """获取教程"""
        return self.tutorials.get(tutorial_id)
    
    def list_tutorials(self, tutorial_type: Optional[TutorialType] = None) -> List[Tutorial]:
        """列出教程"""
        tutorials = list(self.tutorials.values())
        if tutorial_type:
            tutorials = [t for t in tutorials if t.tutorial_type == tutorial_type]
        return [t for t in tutorials if t.is_active]
    
    def get_achievement(self, achievement_id: str) -> Optional[Achievement]:
        """获取成就"""
        return self.achievements.get(achievement_id)


class TutorialManager:
    """教程管理器"""
    
    def __init__(self, storage_path: str = "tutorial_progress"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.library = TutorialLibrary()
        self.user_progress: Dict[str, Dict[str, UserProgress]] = {}
        self._load_progress()
    
    def _load_progress(self) -> None:
        """加载用户进度"""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    user_id = data['user_id']
                    if user_id not in self.user_progress:
                        self.user_progress[user_id] = {}
                    
                    for progress_data in data.get('progress', []):
                        progress = UserProgress(
                            user_id=user_id,
                            tutorial_id=progress_data['tutorial_id'],
                            status=TutorialStatus(progress_data['status']),
                            current_step=progress_data['current_step'],
                            completed_steps=progress_data.get('completed_steps', []),
                            started_at=datetime.fromisoformat(progress_data['started_at']) if progress_data.get('started_at') else None,
                            completed_at=datetime.fromisoformat(progress_data['completed_at']) if progress_data.get('completed_at') else None,
                            total_time_seconds=progress_data.get('total_time_seconds', 0),
                            achievements=progress_data.get('achievements', [])
                        )
                        self.user_progress[user_id][progress.tutorial_id] = progress
            except Exception as e:
                logger.error(f"加载教程进度失败 {file_path}: {e}")
    
    def _save_progress(self, user_id: str) -> None:
        """保存用户进度"""
        file_path = self.storage_path / f"{user_id}.json"
        try:
            data = {
                'user_id': user_id,
                'progress': []
            }
            
            for progress in self.user_progress.get(user_id, {}).values():
                data['progress'].append({
                    'tutorial_id': progress.tutorial_id,
                    'status': progress.status.value,
                    'current_step': progress.current_step,
                    'completed_steps': progress.completed_steps,
                    'started_at': progress.started_at.isoformat() if progress.started_at else None,
                    'completed_at': progress.completed_at.isoformat() if progress.completed_at else None,
                    'total_time_seconds': progress.total_time_seconds,
                    'achievements': progress.achievements
                })
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存教程进度失败: {e}")
    
    def start_tutorial(self, user_id: str, tutorial_id: str) -> Optional[TutorialStep]:
        """
        开始教程
        
        Returns:
            第一个步骤
        """
        tutorial = self.library.get_tutorial(tutorial_id)
        if not tutorial:
            return None
        
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}
        
        progress = UserProgress(
            user_id=user_id,
            tutorial_id=tutorial_id,
            status=TutorialStatus.IN_PROGRESS,
            current_step=0,
            completed_steps=[],
            started_at=datetime.now(),
            completed_at=None,
            total_time_seconds=0,
            achievements=[]
        )
        
        self.user_progress[user_id][tutorial_id] = progress
        self._save_progress(user_id)
        
        logger.info(f"用户 {user_id} 开始教程 {tutorial_id}")
        return tutorial.steps[0] if tutorial.steps else None
    
    def get_current_step(self, user_id: str, tutorial_id: str) -> Optional[TutorialStep]:
        """获取当前步骤"""
        tutorial = self.library.get_tutorial(tutorial_id)
        if not tutorial:
            return None
        
        progress = self.user_progress.get(user_id, {}).get(tutorial_id)
        if not progress:
            return None
        
        if progress.status == TutorialStatus.COMPLETED:
            return None
        
        if progress.current_step < len(tutorial.steps):
            return tutorial.steps[progress.current_step]
        
        return None
    
    def complete_step(self, user_id: str, tutorial_id: str, 
                     step_id: str) -> Optional[TutorialStep]:
        """
        完成步骤
        
        Returns:
            下一步骤或None
        """
        tutorial = self.library.get_tutorial(tutorial_id)
        if not tutorial:
            return None
        
        progress = self.user_progress.get(user_id, {}).get(tutorial_id)
        if not progress:
            return None
        
        # 标记步骤完成
        if step_id not in progress.completed_steps:
            progress.completed_steps.append(step_id)
        
        # 移动到下一步
        progress.current_step += 1
        
        # 检查是否完成
        if progress.current_step >= len(tutorial.steps):
            progress.status = TutorialStatus.COMPLETED
            progress.completed_at = datetime.now()
            self._award_achievements(user_id, tutorial)
            logger.info(f"用户 {user_id} 完成教程 {tutorial_id}")
        
        self._save_progress(user_id)
        
        # 返回下一步
        return self.get_current_step(user_id, tutorial_id)
    
    def _award_achievements(self, user_id: str, tutorial: Tutorial) -> None:
        """授予成就"""
        progress = self.user_progress[user_id][tutorial.tutorial_id]
        
        for achievement_id in tutorial.rewards:
            achievement = self.library.get_achievement(achievement_id)
            if achievement and achievement_id not in progress.achievements:
                progress.achievements.append(achievement_id)
                logger.info(f"用户 {user_id} 获得成就: {achievement.name}")
    
    def skip_tutorial(self, user_id: str, tutorial_id: str) -> bool:
        """跳过教程"""
        progress = self.user_progress.get(user_id, {}).get(tutorial_id)
        if not progress:
            return False
        
        progress.status = TutorialStatus.SKIPPED
        self._save_progress(user_id)
        return True
    
    def get_progress(self, user_id: str, tutorial_id: str) -> Optional[UserProgress]:
        """获取进度"""
        return self.user_progress.get(user_id, {}).get(tutorial_id)
    
    def get_user_achievements(self, user_id: str) -> List[Achievement]:
        """获取用户成就"""
        achievements = []
        for progress in self.user_progress.get(user_id, {}).values():
            for achievement_id in progress.achievements:
                achievement = self.library.get_achievement(achievement_id)
                if achievement:
                    achievements.append(achievement)
        return achievements
    
    def get_recommended_tutorials(self, user_id: str) -> List[Tutorial]:
        """获取推荐教程"""
        completed = set()
        for tutorial_id, progress in self.user_progress.get(user_id, {}).items():
            if progress.status == TutorialStatus.COMPLETED:
                completed.add(tutorial_id)
        
        recommendations = []
        for tutorial in self.library.list_tutorials():
            if tutorial.tutorial_id not in completed:
                # 检查先决条件
                if all(prereq in completed for prereq in tutorial.prerequisites):
                    recommendations.append(tutorial)
        
        return recommendations[:5]
    
    def get_tutorial_summary(self, user_id: str) -> Dict[str, Any]:
        """获取教程摘要"""
        total_tutorials = len(self.library.tutorials)
        completed = 0
        in_progress = 0
        
        for progress in self.user_progress.get(user_id, {}).values():
            if progress.status == TutorialStatus.COMPLETED:
                completed += 1
            elif progress.status == TutorialStatus.IN_PROGRESS:
                in_progress += 1
        
        achievements = self.get_user_achievements(user_id)
        total_points = sum(a.reward_points for a in achievements)
        
        return {
            'total_tutorials': total_tutorials,
            'completed': completed,
            'in_progress': in_progress,
            'not_started': total_tutorials - completed - in_progress,
            'completion_rate': completed / total_tutorials if total_tutorials > 0 else 0,
            'achievements_count': len(achievements),
            'total_points': total_points,
            'achievements': [
                {
                    'id': a.achievement_id,
                    'name': a.name,
                    'description': a.description,
                    'icon': a.icon,
                    'points': a.reward_points
                }
                for a in achievements
            ]
        }


# 便捷函数
def get_tutorial_manager(storage_path: str = "tutorial_progress") -> TutorialManager:
    """获取教程管理器实例"""
    return TutorialManager(storage_path)


# 单例实例
_tutorial_manager_instance: Optional[TutorialManager] = None


def get_tutorial_manager_singleton() -> TutorialManager:
    """获取教程管理器单例"""
    global _tutorial_manager_instance
    if _tutorial_manager_instance is None:
        _tutorial_manager_instance = TutorialManager()
    return _tutorial_manager_instance
