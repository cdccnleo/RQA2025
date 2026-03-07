"""
强化学习动态优化系统

基于强化学习算法，实现自适应的质量优化决策：
1. 质量优化环境建模 - 将质量优化问题建模为强化学习环境
2. 智能体训练 - 训练质量优化智能体学习最优策略
3. 动态优化执行 - 实时执行强化学习决策
4. 持续学习更新 - 基于反馈持续改进优化策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
from collections import deque
import pickle
import json

logger = logging.getLogger(__name__)


class QualityOptimizationEnv(gym.Env):
    """质量优化强化学习环境"""

    def __init__(self, historical_quality_data: pd.DataFrame):
        super(QualityOptimizationEnv, self).__init__()

        self.historical_data = historical_quality_data
        self.current_step = 0
        self.max_steps = 100  # 每个episode的最大步数

        # 定义状态空间 (质量指标 + 系统状态)
        self.state_features = [
            'test_coverage', 'test_success_rate', 'code_quality_score',
            'performance_score', 'error_rate', 'response_time',
            'cpu_usage', 'memory_usage', 'active_connections'
        ]

        # 状态空间：9个连续特征
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.state_features),), dtype=np.float32
        )

        # 定义动作空间 (优化措施)
        self.actions = [
            'increase_test_coverage',      # 增加测试覆盖率
            'optimize_performance',        # 性能优化
            'reduce_error_rate',          # 降低错误率
            'improve_response_time',      # 改善响应时间
            'scale_resources',            # 扩展资源
            'update_code_quality',        # 提升代码质量
            'optimize_architecture',      # 架构优化
            'enhance_monitoring',         # 加强监控
            'no_action'                   # 不采取行动
        ]
        self.action_space = spaces.Discrete(len(self.actions))

        # 奖励函数参数
        self.reward_weights = {
            'quality_improvement': 10.0,    # 质量提升奖励
            'performance_gain': 8.0,        # 性能提升奖励
            'stability_bonus': 5.0,         # 稳定性奖励
            'resource_penalty': -2.0,       # 资源消耗惩罚
            'disruption_penalty': -5.0,     # 干扰惩罚
            'failure_penalty': -10.0        # 失败惩罚
        }

        self.reset()

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        # 随机选择一个起始状态
        if len(self.historical_data) > 0:
            random_idx = np.random.randint(0, len(self.historical_data))
            initial_state = self.historical_data.iloc[random_idx]
        else:
            # 默认初始状态
            initial_state = pd.Series({
                feature: 0.5 for feature in self.state_features
            })

        self.current_state = np.array([
            initial_state.get(feature, 0.5) for feature in self.state_features
        ], dtype=np.float32)

        self.current_step = 0
        self.episode_rewards = []

        return self.current_state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步动作"""
        action_name = self.actions[action]

        # 执行优化动作并获取结果
        next_state, reward, done, info = self._execute_optimization_action(action_name)

        self.current_state = next_state
        self.current_step += 1
        self.episode_rewards.append(reward)

        # 检查是否结束
        truncated = self.current_step >= self.max_steps
        terminated = done or truncated

        return next_state, reward, terminated, truncated, info

    def _execute_optimization_action(self, action_name: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行优化动作"""
        # 基于历史数据和动作类型模拟优化效果
        current_quality_score = np.mean(self.current_state[:6])  # 前6个是质量指标
        current_performance_score = np.mean(self.current_state[6:])  # 后3个是性能指标

        # 模拟不同优化动作的效果
        action_effects = {
            'increase_test_coverage': {
                'quality_improvement': 0.05,
                'performance_impact': -0.01,
                'resource_cost': 0.02,
                'stability_impact': 0.0
            },
            'optimize_performance': {
                'quality_improvement': 0.02,
                'performance_impact': 0.08,
                'resource_cost': 0.03,
                'stability_impact': -0.01
            },
            'reduce_error_rate': {
                'quality_improvement': 0.07,
                'performance_impact': 0.03,
                'resource_cost': 0.01,
                'stability_impact': 0.02
            },
            'improve_response_time': {
                'quality_improvement': 0.03,
                'performance_impact': 0.06,
                'resource_cost': 0.04,
                'stability_impact': 0.0
            },
            'scale_resources': {
                'quality_improvement': 0.0,
                'performance_impact': 0.04,
                'resource_cost': 0.08,
                'stability_impact': 0.01
            },
            'update_code_quality': {
                'quality_improvement': 0.06,
                'performance_impact': 0.02,
                'resource_cost': 0.02,
                'stability_impact': 0.03
            },
            'optimize_architecture': {
                'quality_improvement': 0.04,
                'performance_impact': 0.05,
                'resource_cost': 0.05,
                'stability_impact': 0.02
            },
            'enhance_monitoring': {
                'quality_improvement': 0.02,
                'performance_impact': 0.01,
                'resource_cost': 0.01,
                'stability_impact': 0.05
            },
            'no_action': {
                'quality_improvement': 0.0,
                'performance_impact': 0.0,
                'resource_cost': 0.0,
                'stability_impact': 0.0
            }
        }

        effects = action_effects.get(action_name, action_effects['no_action'])

        # 计算新的状态
        new_state = self.current_state.copy()

        # 质量指标变化 (前6个)
        for i in range(6):
            change = effects['quality_improvement'] * (0.5 + np.random.normal(0, 0.1))
            new_state[i] = np.clip(new_state[i] + change, 0, 1)

        # 性能指标变化 (后3个)
        for i in range(6, 9):
            change = effects['performance_impact'] * (0.5 + np.random.normal(0, 0.1))
            new_state[i] = np.clip(new_state[i] + change, 0, 1)

        # 计算奖励
        reward = self._calculate_reward(effects, new_state)

        # 检查是否完成优化目标
        done = self._check_optimization_goal_achieved(new_state)

        info = {
            'action': action_name,
            'effects': effects,
            'quality_improvement': effects['quality_improvement'],
            'performance_improvement': effects['performance_impact'],
            'resource_cost': effects['resource_cost']
        }

        return new_state.astype(np.float32), reward, done, info

    def _calculate_reward(self, effects: Dict, new_state: np.ndarray) -> float:
        """计算奖励"""
        reward = 0.0

        # 质量提升奖励
        quality_improvement = effects['quality_improvement']
        reward += quality_improvement * self.reward_weights['quality_improvement']

        # 性能提升奖励
        performance_improvement = effects['performance_impact']
        reward += performance_improvement * self.reward_weights['performance_gain']

        # 稳定性奖励
        stability_impact = effects['stability_impact']
        if stability_impact > 0:
            reward += stability_impact * self.reward_weights['stability_bonus']

        # 资源消耗惩罚
        resource_cost = effects['resource_cost']
        reward += resource_cost * self.reward_weights['resource_penalty']

        # 确保奖励在合理范围内
        reward = np.clip(reward, -20, 20)

        return float(reward)

    def _check_optimization_goal_achieved(self, state: np.ndarray) -> bool:
        """检查是否达到优化目标"""
        # 简单的目标检查：质量分数和性能分数都超过0.8
        quality_score = np.mean(state[:6])
        performance_score = np.mean(state[6:])

        return quality_score > 0.8 and performance_score > 0.8


class PolicyNetwork(nn.Module):
    """策略网络"""

    def __init__(self, state_dim: int, action_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ValueNetwork(nn.Module):
    """价值网络"""

    def __init__(self, state_dim: int):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPOAgent:
    """PPO强化学习智能体"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, eps_clip: float = 0.2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.value = ValueNetwork(state_dim)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()

        self.memory = []

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_logits = self.policy_old(state)
            action_probs = torch.softmax(action_logits, dim=1)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        return action.item(), action_logprob.item()

    def store_transition(self, transition: Dict):
        """存储转换"""
        self.memory.append(transition)

    def update(self):
        """更新网络"""
        if len(self.memory) == 0:
            return

        # 准备训练数据
        states = torch.FloatTensor([t['state'] for t in self.memory])
        actions = torch.LongTensor([t['action'] for t in self.memory])
        logprobs = torch.FloatTensor([t['logprob'] for t in self.memory])
        rewards = torch.FloatTensor([t['reward'] for t in self.memory])
        is_terminals = torch.FloatTensor([t['done'] for t in self.memory])

        # 计算优势函数
        advantages = self._compute_advantages(rewards, is_terminals)

        # PPO更新
        for _ in range(5):  # 多次更新
            # 策略网络更新
            action_logits = self.policy(states)
            action_probs = torch.softmax(action_logits, dim=1)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(actions)

            ratios = torch.exp(new_logprobs - logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            # 价值网络更新
            values = self.value(states).squeeze()
            targets = rewards + self.gamma * (1 - is_terminals) * values.detach()
            value_loss = self.mse_loss(values, targets)

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

        # 更新旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空记忆
        self.memory.clear()

    def _compute_advantages(self, rewards: torch.Tensor, is_terminals: torch.Tensor) -> torch.Tensor:
        """计算优势函数"""
        advantages = torch.zeros_like(rewards)

        running_add = 0
        for t in reversed(range(len(rewards))):
            if is_terminals[t]:
                running_add = 0

            running_add = rewards[t] + self.gamma * running_add
            advantages[t] = running_add

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages


class ReinforcementLearningOptimizer:
    """强化学习优化器"""

    def __init__(self, model_path: str = "models/rl_optimizer"):
        self.model_path = model_path
        self.env = None
        self.agent = None
        self.is_trained = False

        # 训练参数
        self.num_episodes = 1000
        self.max_steps_per_episode = 100
        self.update_frequency = 10

        # 历史记录
        self.training_history = []
        self.optimization_history = []

    def train_optimizer(self, historical_quality_data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练强化学习优化器

        Args:
            historical_quality_data: 历史质量数据

        Returns:
            训练结果
        """
        try:
            logger.info("开始训练强化学习优化器...")

            # 初始化环境和智能体
            self.env = QualityOptimizationEnv(historical_quality_data)
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n

            self.agent = PPOAgent(state_dim, action_dim)

            # 训练循环
            episode_rewards = []

            for episode in range(self.num_episodes):
                state, _ = self.env.reset()
                episode_reward = 0
                episode_steps = 0

                while episode_steps < self.max_steps_per_episode:
                    # 选择动作
                    action, logprob = self.agent.select_action(state)

                    # 执行动作
                    next_state, reward, terminated, truncated, info = self.env.step(action)

                    # 存储转换
                    transition = {
                        'state': state,
                        'action': action,
                        'logprob': logprob,
                        'reward': reward,
                        'done': terminated or truncated
                    }
                    self.agent.store_transition(transition)

                    state = next_state
                    episode_reward += reward
                    episode_steps += 1

                    if terminated or truncated:
                        break

                episode_rewards.append(episode_reward)

                # 定期更新网络
                if episode % self.update_frequency == 0:
                    self.agent.update()

                # 记录训练进度
                if episode % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

                # 记录训练历史
                self.training_history.append({
                    'episode': episode,
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'timestamp': datetime.now()
                })

            self.is_trained = True

            # 保存模型
            self._save_optimizer()

            # 计算训练结果
            training_metrics = self._evaluate_training_performance()

            logger.info("强化学习优化器训练完成")

            return {
                'success': True,
                'training_metrics': training_metrics,
                'total_episodes': self.num_episodes,
                'average_reward': np.mean(episode_rewards),
                'best_reward': np.max(episode_rewards)
            }

        except Exception as e:
            logger.error(f"强化学习优化器训练失败: {e}")
            return {'success': False, 'error': str(e)}

    def optimize_quality_dynamically(self, current_quality_metrics: Dict[str, Any],
                                   historical_context: pd.DataFrame) -> Dict[str, Any]:
        """
        动态优化质量

        Args:
            current_quality_metrics: 当前质量指标
            historical_context: 历史上下文

        Returns:
            优化决策
        """
        try:
            if not self.is_trained or self.agent is None:
                return {'error': '优化器未训练'}

            # 构建当前状态
            current_state = self._build_state_from_metrics(current_quality_metrics)

            # 使用智能体选择最优动作
            action, confidence = self.agent.select_action(current_state)

            action_name = self.env.actions[action] if self.env else 'unknown'

            # 预测优化效果
            predicted_reward = self._predict_optimization_effect(current_state, action)

            # 生成优化建议
            optimization_decision = {
                'recommended_action': action_name,
                'action_index': action,
                'confidence': confidence,
                'predicted_reward': predicted_reward,
                'expected_improvement': self._calculate_expected_improvement(predicted_reward),
                'implementation_guidance': self._get_action_implementation_guidance(action_name),
                'risk_assessment': self._assess_action_risks(action_name, current_quality_metrics),
                'timestamp': datetime.now()
            }

            # 记录优化历史
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'current_state': current_state.tolist(),
                'action': action,
                'action_name': action_name,
                'predicted_reward': predicted_reward,
                'confidence': confidence
            })

            return optimization_decision

        except Exception as e:
            logger.error(f"动态质量优化失败: {e}")
            return {'error': str(e)}

    def _build_state_from_metrics(self, metrics: Dict[str, Any]) -> np.ndarray:
        """从指标构建状态"""
        state_features = []

        # 质量指标 (标准化到0-1)
        quality_features = ['test_coverage', 'test_success_rate', 'code_quality_score',
                          'performance_score', 'error_rate', 'response_time']

        for feature in quality_features:
            value = metrics.get(feature, 0)
            # 标准化处理
            if feature in ['error_rate']:
                normalized_value = max(0, 1 - value)  # 错误率越高，质量越低
            else:
                normalized_value = min(1, max(0, value))

            state_features.append(normalized_value)

        # 系统指标
        system_features = ['cpu_usage', 'memory_usage', 'active_connections']
        for feature in system_features:
            value = metrics.get(feature, 0)
            normalized_value = min(1, max(0, value / 100))  # 假设是百分比
            state_features.append(normalized_value)

        return np.array(state_features, dtype=np.float32)

    def _predict_optimization_effect(self, state: np.ndarray, action: int) -> float:
        """预测优化效果"""
        try:
            # 使用价值网络预测期望奖励
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                predicted_value = self.agent.value(state_tensor).item()

            return predicted_value

        except Exception:
            return 0.0

    def _calculate_expected_improvement(self, predicted_reward: float) -> Dict[str, Any]:
        """计算期望改进"""
        # 基于预测奖励计算改进程度
        if predicted_reward > 5:
            improvement_level = 'significant'
            description = '预计显著改善系统质量和性能'
        elif predicted_reward > 2:
            improvement_level = 'moderate'
            description = '预计中等程度改善系统状态'
        elif predicted_reward > 0:
            improvement_level = 'slight'
            description = '预计轻微改善系统状态'
        else:
            improvement_level = 'minimal'
            description = '预计改善有限或可能略有负面影响'

        return {
            'level': improvement_level,
            'description': description,
            'quantitative_estimate': predicted_reward
        }

    def _get_action_implementation_guidance(self, action_name: str) -> Dict[str, Any]:
        """获取动作实施指导"""
        guidance_templates = {
            'increase_test_coverage': {
                'steps': [
                    '识别未覆盖的代码路径',
                    '编写针对性测试用例',
                    '集成到CI/CD流水线',
                    '监控覆盖率变化'
                ],
                'estimated_effort': 'medium',
                'prerequisites': ['测试框架就绪', '代码访问权限'],
                'success_criteria': ['覆盖率提升>5%', '无新缺陷引入']
            },
            'optimize_performance': {
                'steps': [
                    '进行性能分析和瓶颈识别',
                    '实施代码优化和缓存策略',
                    '数据库查询优化',
                    '资源配置调整'
                ],
                'estimated_effort': 'high',
                'prerequisites': ['性能监控工具', '代码修改权限'],
                'success_criteria': ['响应时间减少>10%', '资源利用率优化']
            },
            'reduce_error_rate': {
                'steps': [
                    '错误日志分析和根本原因识别',
                    '异常处理优化',
                    '输入验证加强',
                    '边界条件处理'
                ],
                'estimated_effort': 'medium',
                'prerequisites': ['日志系统', '错误监控'],
                'success_criteria': ['错误率降低>20%', '用户体验改善']
            },
            'scale_resources': {
                'steps': [
                    '资源使用情况评估',
                    '容量规划和资源扩展',
                    '负载均衡配置',
                    '自动伸缩设置'
                ],
                'estimated_effort': 'high',
                'prerequisites': ['基础设施访问权限', '监控系统'],
                'success_criteria': ['资源利用率均衡', '系统稳定性提升']
            }
        }

        return guidance_templates.get(action_name, {
            'steps': ['评估具体需求', '制定实施计划', '逐步执行优化'],
            'estimated_effort': 'unknown',
            'prerequisites': ['系统分析'],
            'success_criteria': ['目标达成']
        })

    def _assess_action_risks(self, action_name: str, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """评估动作风险"""
        risk_assessment = {
            'risk_level': 'low',
            'potential_issues': [],
            'mitigation_strategies': [],
            'monitoring_requirements': []
        }

        # 基于动作类型和当前状态评估风险
        if action_name == 'scale_resources':
            if current_metrics.get('cpu_usage', 0) > 80:
                risk_assessment.update({
                    'risk_level': 'medium',
                    'potential_issues': ['资源扩展可能导致成本增加', '配置变更可能引入不稳定'],
                    'mitigation_strategies': ['渐进式扩展', '充分测试', '回滚计划'],
                    'monitoring_requirements': ['资源使用率', '系统稳定性', '成本变化']
                })

        elif action_name == 'optimize_performance':
            risk_assessment.update({
                'risk_level': 'medium',
                'potential_issues': ['优化可能引入功能缺陷', '性能改善可能不明显'],
                'mitigation_strategies': ['充分测试', '渐进式部署', '性能基准对比'],
                'monitoring_requirements': ['性能指标', '功能正确性', '用户反馈']
            })

        return risk_assessment

    def _evaluate_training_performance(self) -> Dict[str, Any]:
        """评估训练性能"""
        try:
            if not self.training_history:
                return {'error': '无训练历史'}

            rewards = [h['reward'] for h in self.training_history]

            # 计算训练指标
            metrics = {
                'total_episodes': len(self.training_history),
                'average_reward': float(np.mean(rewards)),
                'best_reward': float(np.max(rewards)),
                'reward_std': float(np.std(rewards)),
                'reward_trend': 'improving' if rewards[-1] > rewards[0] else 'stable',
                'convergence_episode': self._detect_convergence_episode(rewards)
            }

            return metrics

        except Exception as e:
            logger.error(f"训练性能评估失败: {e}")
            return {'error': str(e)}

    def _detect_convergence_episode(self, rewards: List[float], window: int = 50) -> Optional[int]:
        """检测收敛回合"""
        try:
            if len(rewards) < window * 2:
                return None

            # 计算滑动平均
            recent_avg = np.mean(rewards[-window:])
            earlier_avg = np.mean(rewards[-window*2:-window])

            # 如果最近的平均奖励比之前的平均奖励高出5%，认为已收敛
            if recent_avg > earlier_avg * 1.05:
                return len(rewards) - window

            return None

        except Exception:
            return None

    def _save_optimizer(self):
        """保存优化器"""
        try:
            import os
            os.makedirs(self.model_path, exist_ok=True)

            # 保存PyTorch模型
            torch.save(self.agent.policy.state_dict(), f"{self.model_path}/policy_model.pth")
            torch.save(self.agent.value.state_dict(), f"{self.model_path}/value_model.pth")

            # 保存环境
            with open(f"{self.model_path}/environment.pkl", 'wb') as f:
                pickle.dump(self.env, f)

            # 保存训练历史
            with open(f"{self.model_path}/training_history.json", 'w') as f:
                json.dump(self.training_history, f, indent=2, default=str)

            # 保存配置
            config = {
                'is_trained': self.is_trained,
                'num_episodes': self.num_episodes,
                'max_steps_per_episode': self.max_steps_per_episode,
                'state_dim': self.env.observation_space.shape[0] if self.env else 0,
                'action_dim': self.env.action_space.n if self.env else 0
            }

            with open(f"{self.model_path}/config.json", 'w') as f:
                json.dump(config, f, indent=2)

        except Exception as e:
            logger.error(f"优化器保存失败: {e}")

    def load_optimizer(self) -> bool:
        """加载优化器"""
        try:
            # 加载配置
            with open(f"{self.model_path}/config.json", 'r') as f:
                config = json.load(f)

            state_dim = config.get('state_dim', 9)
            action_dim = 9  # 固定的动作空间

            # 初始化智能体
            self.agent = PPOAgent(state_dim, action_dim)

            # 加载模型权重
            self.agent.policy.load_state_dict(torch.load(f"{self.model_path}/policy_model.pth"))
            self.agent.value.load_state_dict(torch.load(f"{self.model_path}/value_model.pth"))
            self.agent.policy_old.load_state_dict(self.agent.policy.state_dict())

            # 加载环境
            with open(f"{self.model_path}/environment.pkl", 'rb') as f:
                self.env = pickle.load(f)

            # 加载训练历史
            with open(f"{self.model_path}/training_history.json", 'r') as f:
                self.training_history = json.load(f)

            self.is_trained = config.get('is_trained', False)

            return True

        except Exception as e:
            logger.error(f"优化器加载失败: {e}")
            return False

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        try:
            stats = {
                'total_optimizations': len(self.optimization_history),
                'is_trained': self.is_trained,
                'training_episodes': len(self.training_history) if self.training_history else 0
            }

            if self.training_history:
                rewards = [h['reward'] for h in self.training_history]
                stats.update({
                    'training_avg_reward': float(np.mean(rewards)),
                    'training_best_reward': float(np.max(rewards)),
                    'training_reward_trend': 'improving' if len(rewards) > 1 and rewards[-1] > rewards[0] else 'stable'
                })

            if self.optimization_history:
                confidences = [h['confidence'] for h in self.optimization_history]
                predicted_rewards = [h['predicted_reward'] for h in self.optimization_history]

                stats.update({
                    'optimization_avg_confidence': float(np.mean(confidences)),
                    'optimization_avg_predicted_reward': float(np.mean(predicted_rewards)),
                    'most_common_action': self._get_most_common_action()
                })

            return stats

        except Exception as e:
            logger.error(f"获取优化统计失败: {e}")
            return {'error': str(e)}

    def _get_most_common_action(self) -> str:
        """获取最常见的优化动作"""
        try:
            if not self.optimization_history:
                return 'none'

            actions = [h['action_name'] for h in self.optimization_history]
            most_common = max(set(actions), key=actions.count)

            return most_common

        except Exception:
            return 'unknown'
