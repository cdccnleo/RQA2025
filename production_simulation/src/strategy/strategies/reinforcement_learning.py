from ...interfaces.strategy_interfaces import BaseTradingStrategy
import logging
"""强化学习策略模块

实现各种强化学习算法，包括DQN、PPO、A2C等，
用于动态交易策略优化。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import secrets
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


@dataclass
class TradingState:

    """交易状态"""
    position: float  # 当前仓位
    cash: float      # 当前现金
    price: float     # 当前价格
    volume: float    # 当前成交量
    returns: float   # 当前收益率
    volatility: float  # 当前波动率
    momentum: float    # 当前动量
    trend: float       # 当前趋势


class DQNNetwork(nn.Module):

    """DQN网络"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):

        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):

    """策略网络"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):

        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))


class ValueNetwork(nn.Module):

    """价值网络"""

    def __init__(self, state_size: int, hidden_size: int = 128):

        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:

    """DQN智能体"""

    def __init__(self, state_size: int, action_size: int,


                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000):
        # 参数验证
        if state_size <= 0:
            raise ValueError("state_size must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0.0 and 1.0")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be between 0.0 and 1.0")
        if epsilon_min < 0.0 or epsilon_min > 1.0:
            raise ValueError("epsilon_min must be between 0.0 and 1.0")
        if epsilon_decay < 0.0 or epsilon_decay > 1.0:
            raise ValueError("epsilon_decay must be between 0.0 and 1.0")
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.update_target_network()

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state: np.ndarray, action: int, reward: float,


                 next_state: np.ndarray, done: bool):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """选择动作"""
        if secrets.random() <= self.epsilon:
            return secrets.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.argmax().item()

    def replay(self, batch_size: int = 32):
        """经验回放"""
        if len(self.memory) < batch_size:
            return

        batch = secrets.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PPOAgent:

    """PPO智能体"""

    def __init__(self, state_size: int, action_size: int,


                 learning_rate: float = 0.0003, gamma: float = 0.99,
                 clip_ratio: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network = PolicyNetwork(state_size, action_size).to(self.device)
        self.value_network = ValueNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy_network.parameters(), 'lr': learning_rate},
            {'params': self.value_network.parameters(), 'lr': learning_rate}
        ])

    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def update(self, states: np.ndarray, actions: np.ndarray,


               rewards: np.ndarray, old_log_probs: np.ndarray):
        """更新策略"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        # 计算折扣奖励
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.FloatTensor(returns).to(self.device)

        # 标准化奖励
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 计算价值和策略损失
        values = self.value_network(states).squeeze()
        action_probs = self.policy_network(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * returns
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * returns
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(values, returns)
        entropy_loss = -dist.entropy().mean()

        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


class A2CAgent:

    """A2C智能体"""

    def __init__(self, state_size: int, action_size: int,


                 learning_rate: float = 0.001, gamma: float = 0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network = PolicyNetwork(state_size, action_size).to(self.device)
        self.value_network = ValueNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy_network.parameters(), 'lr': learning_rate},
            {'params': self.value_network.parameters(), 'lr': learning_rate}
        ])

    def act(self, state: np.ndarray) -> Tuple[int, float]:
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def update(self, states: np.ndarray, actions: np.ndarray,


               rewards: np.ndarray, next_state: np.ndarray, done: bool):
        """更新策略"""
        states = torch.FloatTensor(states).unsqueeze(0).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(0).to(self.device)
        rewards = torch.FloatTensor([rewards]).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        # 计算优势函数
        current_value = self.value_network(states)
        next_value = self.value_network(
            next_state) if not done else torch.zeros(1, 1).to(self.device)
        advantage = rewards + self.gamma * next_value - current_value

        # 计算策略和价值损失
        action_probs = self.policy_network(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)

        policy_loss = -(log_probs * advantage.detach()).mean()
        value_loss = nn.MSELoss()(current_value, rewards + self.gamma * next_value)

        total_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


class ReinforcementLearningStrategy(BaseTradingStrategy):

    """强化学习策略基类"""

    def __init__(self, agent_type: str = 'dqn', **kwargs):

        # 创建默认配置
        from ...interfaces.strategy_interfaces import StrategyConfig
        config = StrategyConfig(
            strategy_type=f"reinforcement_learning_{agent_type}",
            strategy_params=kwargs
        )
        super().__init__(config)
        self.agent_type = agent_type
        self.agent = None
        self.state_size = 8  # 状态空间大小
        self.action_size = 3  # 动作空间大小：买入、卖出、持有
        self.kwargs = kwargs
        self.position = 0.0
        self.cash = 100000.0
        self._init_agent()

    def _init_agent(self):
        """初始化智能体"""
        if self.agent_type == 'dqn':
            self.agent = DQNAgent(self.state_size, self.action_size, **self.kwargs)
        elif self.agent_type == 'ppo':
            self.agent = PPOAgent(self.state_size, self.action_size, **self.kwargs)
        elif self.agent_type == 'a2c':
            self.agent = A2CAgent(self.state_size, self.action_size, **self.kwargs)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")

    def _get_state(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """获取当前状态"""
        if index < 20:  # 需要足够的历史数据
            return np.zeros(self.state_size)

        current_data = data.iloc[index]
        prev_data = data.iloc[index - 1]

        # 计算状态特征
        position = self.position if hasattr(self, 'position') else 0.0
        cash = self.cash if hasattr(self, 'cash') else 100000.0
        price = current_data['close']
        volume = current_data['volume']

        # 计算收益率
        returns = (price - prev_data['close']) / prev_data['close']

        # 计算波动率（20日）
        if index >= 20:
            returns_20 = data.iloc[index - 20:index]['close'].pct_change().dropna()
            volatility = returns_20.std()
        else:
            volatility = 0.0

        # 计算动量（5日）
        if index >= 5:
            momentum = (price - data.iloc[index - 5]['close']) / data.iloc[index - 5]['close']
        else:
            momentum = 0.0

        # 计算趋势（10日）
        if index >= 10:
            trend = (price - data.iloc[index - 10]['close']) / data.iloc[index - 10]['close']
        else:
            trend = 0.0

        return np.array([position, cash, price, volume, returns, volatility, momentum, trend])

    def _get_reward(self, action: int, current_price: float,


                    next_price: float, position: float) -> float:
        """计算奖励"""
        if action == 0:  # 买入
            if position < 1.0:  # 可以买入
                return (next_price - current_price) / current_price
            else:
                return -0.01  # 惩罚重复买入
        elif action == 1:  # 卖出
            if position > 0.0:  # 可以卖出
                return (current_price - next_price) / current_price
            else:
                return -0.01  # 惩罚重复卖出
        else:  # 持有
            return 0.0  # 中性奖励

    def train(self, data: pd.DataFrame, episodes: int = 1000) -> Dict[str, Any]:
        """训练策略"""
        logger.info(f"开始训练{self.agent_type.upper()}策略...")

        episode_rewards = []

        for episode in range(episodes):
            total_reward = 0
            position = 0.0
            cash = 100000.0

            for i in range(20, len(data) - 1):
                state = self._get_state(data, i)
                action_result = self.agent.act(state)

                # 处理不同的智能体返回值类型
                if isinstance(action_result, tuple):
                    action, log_prob = action_result
                else:
                    action = action_result
                    log_prob = 0.0

                current_price = data.iloc[i]['close']
                next_price = data.iloc[i + 1]['close']

                # 执行动作
                if action == 0 and cash > 0:  # 买入
                    shares = cash / current_price
                    position += shares
                    cash = 0
                elif action == 1 and position > 0:  # 卖出
                    cash = position * current_price
                    position = 0

                reward = self._get_reward(action, current_price, next_price, position)
                total_reward += reward

                next_state = self._get_state(data, i + 1)

                # 存储经验
                if hasattr(self.agent, 'remember'):
                    self.agent.remember(state, action, reward, next_state, False)

                # 更新策略
                if hasattr(self.agent, 'replay'):
                    self.agent.replay()
                elif hasattr(self.agent, 'update'):
                    # 根据智能体类型调用不同的update方法
                    if self.agent_type == 'ppo':
                        # PPO需要old_log_probs，这里我们使用0作为默认值
                        self.agent.update([state], [action], [reward], [log_prob])
                    elif self.agent_type == 'a2c':
                        self.agent.update([state], [action], [reward], next_state, False)
                    else:
                        # 默认情况
                        self.agent.update([state], [action], [reward], next_state, False)

            episode_rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode [{episode + 1}/{episodes}], Avg Reward: {avg_reward:.4f}")

        logger.info(f"{self.agent_type.upper()}策略训练完成")
        return {'episode_rewards': episode_rewards}

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """进行预测"""
        predictions = []

        for i in range(len(data)):
            state = self._get_state(data, i)
            action_result = self.agent.act(state)

            # 处理不同的智能体返回值类型
            if isinstance(action_result, tuple):
                action, _ = action_result
            else:
                action = action_result

            predictions.append(action)

        return np.array(predictions)

    def save(self, filepath: str):
        """保存模型"""
        if hasattr(self.agent, 'q_network'):
            torch.save({
                'q_network_state_dict': self.agent.q_network.state_dict(),
                'target_network_state_dict': self.agent.target_network.state_dict(),
                'agent_type': self.agent_type,
                'kwargs': self.kwargs
            }, filepath)
        elif hasattr(self.agent, 'policy_network'):
            torch.save({
                'policy_network_state_dict': self.agent.policy_network.state_dict(),
                'value_network_state_dict': self.agent.value_network.state_dict(),
                'agent_type': self.agent_type,
                'kwargs': self.kwargs
            }, filepath)
        logger.info(f"模型已保存到: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.agent_type = checkpoint['agent_type']
        self.kwargs = checkpoint['kwargs']
        self._init_agent()

        if hasattr(self.agent, 'q_network'):
            self.agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        elif hasattr(self.agent, 'policy_network'):
            self.agent.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
            self.agent.value_network.load_state_dict(checkpoint['value_network_state_dict'])

        logger.info(f"模型已从 {filepath} 加载")

    # 实现抽象方法

    def _generate_signals_impl(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        predictions = self.predict(data)

        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 'HOLD'
        signals['confidence'] = 0.5
        signals['target_price'] = data['close']
        signals['position'] = 0.0

        # 根据预测设置信号
        for i, action in enumerate(predictions):
            if action == 0:  # 买入
                signals.iloc[i]['signal'] = 'BUY'
                signals.iloc[i]['confidence'] = 0.8
                signals.iloc[i]['position'] = 1.0
            elif action == 1:  # 卖出
                signals.iloc[i]['signal'] = 'SELL'
                signals.iloc[i]['confidence'] = 0.8
                signals.iloc[i]['position'] = -1.0

        return signals

    def _get_required_columns(self) -> List[str]:
        """获取必需的列"""
        return ['open', 'high', 'low', 'close', 'volume']

    def _execute_trades(self, signals: pd.DataFrame) -> List[Dict[str, Any]]:
        """执行交易"""
        trades = []
        for i, (idx, row) in enumerate(signals.iterrows()):
            if row['signal'] != 'HOLD':
                trade = {
                    'timestamp': idx,
                    'action': row['signal'],
                    'price': row['target_price'],
                    'quantity': abs(row['position']),
                    'confidence': row['confidence']
                }
                trades.append(trade)
        return trades

    def _run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """运行回测"""
        # 简单的回测实现
        portfolio_value = 100000.0
        position = 0.0
        returns = []

        for i, (idx, row) in enumerate(signals.iterrows()):
            if row['signal'] == 'BUY' and position == 0:
                position = portfolio_value / data.loc[idx, 'close']
                portfolio_value = 0
            elif row['signal'] == 'SELL' and position > 0:
                portfolio_value = position * data.loc[idx, 'close']
                position = 0

            current_value = portfolio_value + position * data.loc[idx, 'close']
            returns.append((current_value - 100000.0) / 100000.0)

        return {
            'total_return': returns[-1] if returns else 0.0,
            'sharpe_ratio': np.mean(returns) / np.std(returns) if len(returns) > 1 else 0.0,
            'max_drawdown': min(returns) if returns else 0.0,
            'returns': returns
        }


class DQNStrategy(ReinforcementLearningStrategy):

    """DQN策略"""

    def __init__(self, learning_rate: float = 0.001, gamma: float = 0.99,


                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000):
        super().__init__('dqn', learning_rate=learning_rate, gamma=gamma,
                         epsilon=epsilon, epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay, memory_size=memory_size)


class PPOStrategy(ReinforcementLearningStrategy):

    """PPO策略"""

    def __init__(self, learning_rate: float = 0.0003, gamma: float = 0.99,


                 clip_ratio: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        super().__init__('ppo', learning_rate=learning_rate, gamma=gamma,
                         clip_ratio=clip_ratio, value_coef=value_coef,
                         entropy_coef=entropy_coef)


class A2CStrategy(ReinforcementLearningStrategy):

    """A2C策略"""

    def __init__(self, learning_rate: float = 0.001, gamma: float = 0.99):

        super().__init__('a2c', learning_rate=learning_rate, gamma=gamma)
