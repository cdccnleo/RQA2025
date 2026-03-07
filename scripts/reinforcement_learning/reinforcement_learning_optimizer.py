#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习优化器
实现强化学习优化算法
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import random


@dataclass
class RLConfig:
    """强化学习配置"""
    algorithm: str = "q_learning"
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    episodes: int = 500
    max_steps_per_episode: int = 100
    state_size: int = 10
    action_size: int = 5


@dataclass
class EnvironmentState:
    """环境状态"""
    timestamp: float
    market_volatility: float
    risk_level: float
    performance_score: float
    system_load: float
    cache_hit_rate: float
    error_rate: float
    response_time: float
    user_satisfaction: float
    optimization_potential: float


@dataclass
class Action:
    """动作"""
    action_id: int
    action_type: str
    parameters: Dict[str, float]
    confidence: float


class QLearningAgent:
    """Q学习智能体"""

    def __init__(self, config: RLConfig):
        self.config = config
        self.q_table = {}
        self.epsilon = config.epsilon
        self.learning_history = []
        self.episode_results = []

    def get_state_key(self, state: EnvironmentState) -> str:
        """获取状态键"""
        volatility_bin = int(state.market_volatility * 10)
        risk_bin = int(state.risk_level * 10)
        performance_bin = int(state.performance_score * 10)
        return f"{volatility_bin}_{risk_bin}_{performance_bin}"

    def choose_action(self, state: EnvironmentState) -> Action:
        """选择动作"""
        state_key = self.get_state_key(state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.config.action_size)

        if random.random() < self.epsilon:
            action_id = random.randint(0, self.config.action_size - 1)
        else:
            action_id = np.argmax(self.q_table[state_key])

        action_type, parameters = self._generate_action_parameters(action_id, state)

        return Action(
            action_id=action_id,
            action_type=action_type,
            parameters=parameters,
            confidence=1.0 - self.epsilon
        )

    def _generate_action_parameters(self, action_id: int, state: EnvironmentState) -> Tuple[str, Dict[str, float]]:
        """生成动作参数"""
        action_types = [
            "adjust_cache_ttl",
            "modify_risk_threshold",
            "optimize_monitoring_interval",
            "scale_system_resources",
            "update_learning_rate"
        ]

        action_type = action_types[action_id]

        if action_type == "adjust_cache_ttl":
            parameters = {"cache_ttl": max(1800, min(7200, 3600 * (1 + random.uniform(-0.5, 0.5))))}
        elif action_type == "modify_risk_threshold":
            parameters = {"risk_threshold": max(
                0.05, min(0.2, 0.1 * (1 + random.uniform(-0.5, 0.5))))}
        elif action_type == "optimize_monitoring_interval":
            parameters = {"monitoring_interval": max(
                15, min(60, 30 * (1 + random.uniform(-0.5, 0.5))))}
        elif action_type == "scale_system_resources":
            parameters = {
                "cpu_limit": max(50, min(90, 70 * (1 + random.uniform(-0.3, 0.3)))),
                "memory_limit": max(60, min(95, 80 * (1 + random.uniform(-0.3, 0.3))))
            }
        else:
            parameters = {"learning_rate": max(0.001, min(
                0.01, 0.005 * (1 + random.uniform(-0.5, 0.5))))}

        return action_type, parameters

    def update_q_value(self, state: EnvironmentState, action: Action, reward: float, next_state: EnvironmentState):
        """更新Q值"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.config.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.config.action_size)

        current_q = self.q_table[state_key][action.action_id]
        max_next_q = np.max(self.q_table[next_state_key])

        new_q = current_q + self.config.learning_rate * \
            (reward + self.config.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action.action_id] = new_q

        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.epsilon_decay)

    def get_learning_progress(self) -> Dict[str, float]:
        """获取学习进度"""
        if not self.episode_results:
            return {}

        recent_episodes = self.episode_results[-10:]
        avg_reward = np.mean([ep["total_reward"] for ep in recent_episodes])
        avg_steps = np.mean([ep["steps_taken"] for ep in recent_episodes])

        return {
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table)
        }


class Environment:
    """环境"""

    def __init__(self):
        self.current_state = None
        self.step_count = 0

    def reset(self) -> EnvironmentState:
        """重置环境"""
        self.step_count = 0

        self.current_state = EnvironmentState(
            timestamp=time.time(),
            market_volatility=random.uniform(0.01, 0.05),
            risk_level=random.uniform(0.1, 0.3),
            performance_score=random.uniform(0.6, 0.9),
            system_load=random.uniform(0.4, 0.8),
            cache_hit_rate=random.uniform(0.7, 0.95),
            error_rate=random.uniform(0.01, 0.1),
            response_time=random.uniform(20, 80),
            user_satisfaction=random.uniform(0.7, 0.95),
            optimization_potential=random.uniform(0.1, 0.4)
        )

        return self.current_state

    def step(self, action: Action) -> Tuple[EnvironmentState, float, bool]:
        """执行一步"""
        self.step_count += 1

        new_state = self._simulate_state_transition(action)
        reward = self._calculate_reward(action, new_state)
        done = self.step_count >= 100 or new_state.performance_score < 0.3

        self.current_state = new_state
        return new_state, reward, done

    def _simulate_state_transition(self, action: Action) -> EnvironmentState:
        """模拟状态转移"""
        if action.action_type == "adjust_cache_ttl":
            cache_ttl = action.parameters["cache_ttl"]
            cache_hit_rate_change = (cache_ttl - 3600) / 3600 * 0.1
            new_cache_hit_rate = max(
                0.5, min(0.98, self.current_state.cache_hit_rate + cache_hit_rate_change))
        else:
            new_cache_hit_rate = self.current_state.cache_hit_rate

        if action.action_type == "modify_risk_threshold":
            risk_threshold = action.parameters["risk_threshold"]
            risk_level_change = (risk_threshold - 0.1) / 0.1 * 0.05
            new_risk_level = max(0.05, min(0.5, self.current_state.risk_level + risk_level_change))
        else:
            new_risk_level = self.current_state.risk_level

        if action.action_type == "optimize_monitoring_interval":
            monitoring_interval = action.parameters["monitoring_interval"]
            response_time_change = (30 - monitoring_interval) / 30 * 0.1
            new_response_time = max(
                10, min(100, self.current_state.response_time + response_time_change))
        else:
            new_response_time = self.current_state.response_time

        new_state = EnvironmentState(
            timestamp=time.time(),
            market_volatility=self.current_state.market_volatility + random.uniform(-0.01, 0.01),
            risk_level=new_risk_level,
            performance_score=self.current_state.performance_score + random.uniform(-0.05, 0.05),
            system_load=self.current_state.system_load + random.uniform(-0.1, 0.1),
            cache_hit_rate=new_cache_hit_rate,
            error_rate=self.current_state.error_rate + random.uniform(-0.02, 0.02),
            response_time=new_response_time,
            user_satisfaction=self.current_state.user_satisfaction + random.uniform(-0.05, 0.05),
            optimization_potential=self.current_state.optimization_potential +
            random.uniform(-0.1, 0.1)
        )

        # 确保值在合理范围内
        new_state.market_volatility = max(0.001, min(0.1, new_state.market_volatility))
        new_state.risk_level = max(0.05, min(0.5, new_state.risk_level))
        new_state.performance_score = max(0.1, min(1.0, new_state.performance_score))
        new_state.system_load = max(0.1, min(1.0, new_state.system_load))
        new_state.cache_hit_rate = max(0.5, min(0.98, new_state.cache_hit_rate))
        new_state.error_rate = max(0.0, min(0.2, new_state.error_rate))
        new_state.response_time = max(10, min(100, new_state.response_time))
        new_state.user_satisfaction = max(0.5, min(1.0, new_state.user_satisfaction))
        new_state.optimization_potential = max(0.0, min(0.5, new_state.optimization_potential))

        return new_state

    def _calculate_reward(self, action: Action, new_state: EnvironmentState) -> float:
        """计算奖励"""
        reward = 0.0

        reward += new_state.performance_score * 10
        reward += new_state.user_satisfaction * 5
        reward -= new_state.risk_level * 10
        reward += max(0, (100 - new_state.response_time) / 100 * 3)
        reward += new_state.cache_hit_rate * 2
        reward -= new_state.error_rate * 20

        return reward


class ReinforcementLearningOptimizer:
    """强化学习优化器"""

    def __init__(self, config: RLConfig):
        self.config = config
        self.agent = QLearningAgent(config)
        self.environment = Environment()
        self.optimization_history = []

    def train(self) -> Dict[str, Any]:
        """训练强化学习智能体"""
        print("🎯 开始强化学习训练...")

        training_results = []

        for episode in range(self.config.episodes):
            if episode % 100 == 0:
                print(f"🔄 训练进度: {episode}/{self.config.episodes}")

            state = self.environment.reset()
            total_reward = 0.0
            actions_taken = []

            for step in range(self.config.max_steps_per_episode):
                action = self.agent.choose_action(state)
                next_state, reward, done = self.environment.step(action)
                self.agent.update_q_value(state, action, reward, next_state)

                actions_taken.append({
                    "step": step,
                    "action": asdict(action),
                    "reward": reward,
                    "state": asdict(state)
                })

                total_reward += reward
                state = next_state

                if done:
                    break

            episode_result = {
                "episode": episode,
                "total_reward": total_reward,
                "steps_taken": len(actions_taken),
                "final_state": asdict(state),
                "actions_taken": actions_taken,
                "learning_progress": self.agent.get_learning_progress()
            }

            self.agent.episode_results.append(episode_result)
            training_results.append(episode_result)

        print("✅ 强化学习训练完成!")

        return {
            "episodes_completed": self.config.episodes,
            "training_results": training_results,
            "final_learning_progress": self.agent.get_learning_progress()
        }

    def evaluate(self, num_episodes: int = 20) -> Dict[str, Any]:
        """评估训练结果"""
        print("📊 评估强化学习智能体...")

        evaluation_results = []

        for episode in range(num_episodes):
            state = self.environment.reset()
            total_reward = 0.0
            actions_taken = []

            for step in range(self.config.max_steps_per_episode):
                action = self.agent.choose_action(state)
                next_state, reward, done = self.environment.step(action)

                actions_taken.append({
                    "step": step,
                    "action": asdict(action),
                    "reward": reward
                })

                total_reward += reward
                state = next_state

                if done:
                    break

            evaluation_results.append({
                "episode": episode,
                "total_reward": total_reward,
                "steps_taken": len(actions_taken),
                "final_performance": state.performance_score,
                "final_risk_level": state.risk_level
            })

        rewards = [r["total_reward"] for r in evaluation_results]
        performances = [r["final_performance"] for r in evaluation_results]
        risk_levels = [r["final_risk_level"] for r in evaluation_results]

        evaluation_summary = {
            "avg_reward": np.mean(rewards),
            "avg_performance": np.mean(performances),
            "avg_risk_level": np.mean(risk_levels),
            "reward_std": np.std(rewards),
            "performance_std": np.std(performances),
            "risk_level_std": np.std(risk_levels)
        }

        return {
            "evaluation_episodes": num_episodes,
            "evaluation_results": evaluation_results,
            "evaluation_summary": evaluation_summary
        }


class RLReporter:
    """强化学习报告器"""

    def generate_training_report(self, training_result: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成训练报告"""
        report = {
            "timestamp": time.time(),
            "training_result": training_result,
            "evaluation_result": evaluation_result,
            "summary": self._generate_summary(training_result, evaluation_result),
            "recommendations": self._generate_recommendations(training_result, evaluation_result)
        }

        return report

    def _generate_summary(self, training_result: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要"""
        training_progress = training_result["final_learning_progress"]
        evaluation_summary = evaluation_result["evaluation_summary"]

        return {
            "training_status": "completed",
            "episodes_trained": training_result["episodes_completed"],
            "avg_training_reward": training_progress.get("avg_reward", 0),
            "final_epsilon": training_progress.get("epsilon", 0),
            "evaluation_episodes": evaluation_result["evaluation_episodes"],
            "avg_evaluation_reward": evaluation_summary["avg_reward"],
            "avg_evaluation_performance": evaluation_summary["avg_performance"],
            "avg_evaluation_risk": evaluation_summary["avg_risk_level"]
        }

    def _generate_recommendations(self, training_result: Dict[str, Any], evaluation_result: Dict[str, Any]) -> List[str]:
        """生成建议"""
        recommendations = []

        evaluation_summary = evaluation_result["evaluation_summary"]

        if evaluation_summary["avg_reward"] > 50:
            recommendations.append("强化学习智能体表现优秀，建议部署到生产环境")
        elif evaluation_summary["avg_reward"] > 30:
            recommendations.append("强化学习智能体表现良好，建议进一步优化后部署")
        else:
            recommendations.append("强化学习智能体表现一般，建议增加训练轮数或调整参数")

        if evaluation_summary["avg_performance"] > 0.8:
            recommendations.append("系统性能表现优秀，智能体优化效果显著")
        elif evaluation_summary["avg_performance"] > 0.6:
            recommendations.append("系统性能表现良好，仍有优化空间")
        else:
            recommendations.append("系统性能需要改进，建议检查环境设置")

        if evaluation_summary["avg_risk_level"] < 0.2:
            recommendations.append("风险控制效果良好，智能体能够有效管理风险")
        else:
            recommendations.append("风险控制需要改进，建议调整奖励函数")

        recommendations.append("建议定期重新训练智能体以适应环境变化")
        recommendations.append("建议监控智能体行为，确保符合业务要求")

        return recommendations


def main():
    """主函数"""
    print("🎯 启动强化学习优化器...")

    config = RLConfig(
        algorithm="q_learning",
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        episodes=300,
        max_steps_per_episode=100,
        state_size=10,
        action_size=5
    )

    optimizer = ReinforcementLearningOptimizer(config)

    training_result = optimizer.train()
    evaluation_result = optimizer.evaluate(num_episodes=20)

    reporter = RLReporter()
    report = reporter.generate_training_report(training_result, evaluation_result)

    print("✅ 强化学习优化完成!")

    print("\n" + "="*50)
    print("🎯 训练结果:")
    print("="*50)

    summary = report["summary"]
    print(f"训练状态: {summary['training_status']}")
    print(f"训练回合: {summary['episodes_trained']}")
    print(f"平均训练奖励: {summary['avg_training_reward']:.2f}")
    print(f"最终探索率: {summary['final_epsilon']:.3f}")

    print(f"\n📊 评估结果:")
    print(f"评估回合: {summary['evaluation_episodes']}")
    print(f"平均评估奖励: {summary['avg_evaluation_reward']:.2f}")
    print(f"平均性能得分: {summary['avg_evaluation_performance']:.3f}")
    print(f"平均风险等级: {summary['avg_evaluation_risk']:.3f}")

    print("\n💡 建议:")
    for recommendation in report["recommendations"]:
        print(f"  - {recommendation}")

    print("="*50)

    output_dir = Path("reports/reinforcement_learning/")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "reinforcement_learning_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📄 训练报告已保存: {report_file}")


if __name__ == "__main__":
    main()
