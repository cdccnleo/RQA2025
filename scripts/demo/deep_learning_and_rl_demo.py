"""深度学习模型和强化学习策略演示脚本

展示新功能扩展的深度学习模型和强化学习策略的使用方法。
"""

from src.utils.logger import get_logger
from src.trading.strategies.reinforcement_learning import (
    DQNStrategy,
    PPOStrategy,
    A2CStrategy
)
from src.models.deep_learning_models import (
    LSTMDeepLearningModel,
    CNNDeepLearningModel,
    TransformerDeepLearningModel
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


logger = get_logger(__name__)


def generate_sample_data(n_days=200):
    """生成样本数据"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

    # 生成模拟股票数据
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_days)
    }, index=dates)

    return data


def demo_deep_learning_models():
    """演示深度学习模型"""
    logger.info("=== 深度学习模型演示 ===")

    # 生成数据
    data = generate_sample_data(200)
    logger.info(f"生成样本数据: {len(data)} 天")

    # 分割训练和测试数据
    train_data = data.iloc[:150]
    test_data = data.iloc[150:]

    models = {
        'LSTM': LSTMDeepLearningModel(input_size=4, hidden_size=32, num_layers=1, output_size=1),
        'CNN': CNNDeepLearningModel(input_channels=4, num_filters=[16, 32], kernel_sizes=[3, 3], output_size=1),
        'Transformer': TransformerDeepLearningModel(input_size=4, d_model=64, nhead=4, num_layers=2, output_size=1)
    }

    results = {}

    for name, model in models.items():
        logger.info(f"\n--- 训练 {name} 模型 ---")

        try:
            # 训练模型
            train_result = model.train(
                train_data, 'close', sequence_length=20, epochs=10, batch_size=16)
            logger.info(f"{name} 模型训练完成，最终损失: {train_result['train_losses'][-1]:.6f}")

            # 预测
            predictions = model.predict(test_data, 'close', sequence_length=20)
            logger.info(f"{name} 模型预测完成，预测值: {predictions[0]:.2f}")

            # 评估
            metrics = model.evaluate(test_data, 'close', sequence_length=20)
            logger.info(f"{name} 模型评估结果:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.6f}")

            results[name] = {
                'predictions': predictions,
                'metrics': metrics,
                'train_losses': train_result['train_losses']
            }

        except Exception as e:
            logger.error(f"{name} 模型训练失败: {str(e)}")
            results[name] = None

    return results


def demo_reinforcement_learning_strategies():
    """演示强化学习策略"""
    logger.info("\n=== 强化学习策略演示 ===")

    # 生成数据
    data = generate_sample_data(200)
    logger.info(f"生成样本数据: {len(data)} 天")

    strategies = {
        'DQN': DQNStrategy(learning_rate=0.001, gamma=0.99),
        'PPO': PPOStrategy(learning_rate=0.0003, gamma=0.99),
        'A2C': A2CStrategy(learning_rate=0.001, gamma=0.99)
    }

    results = {}

    for name, strategy in strategies.items():
        logger.info(f"\n--- 训练 {name} 策略 ---")

        try:
            # 训练策略
            train_result = strategy.train(data, episodes=10)
            logger.info(f"{name} 策略训练完成")

            # 计算平均奖励
            avg_reward = np.mean(train_result['episode_rewards'])
            logger.info(f"{name} 策略平均奖励: {avg_reward:.4f}")

            # 预测
            predictions = strategy.predict(data)
            logger.info(f"{name} 策略预测完成，动作分布:")
            unique, counts = np.unique(predictions, return_counts=True)
            for action, count in zip(unique, counts):
                action_name = ['买入', '卖出', '持有'][action]
                logger.info(f"  {action_name}: {count} 次")

            results[name] = {
                'predictions': predictions,
                'episode_rewards': train_result['episode_rewards'],
                'avg_reward': avg_reward
            }

        except Exception as e:
            logger.error(f"{name} 策略训练失败: {str(e)}")
            results[name] = None

    return results


def plot_results(deep_learning_results, rl_results):
    """绘制结果图表"""
    logger.info("\n=== 生成结果图表 ===")

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('深度学习模型和强化学习策略演示结果', fontsize=16)

    # 1. 深度学习模型训练损失
    ax1 = axes[0, 0]
    for name, result in deep_learning_results.items():
        if result is not None:
            ax1.plot(result['train_losses'], label=name, alpha=0.8)
    ax1.set_title('深度学习模型训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 深度学习模型评估指标
    ax2 = axes[0, 1]
    metrics_data = {}
    for name, result in deep_learning_results.items():
        if result is not None:
            metrics_data[name] = result['metrics']

    if metrics_data:
        metric_names = list(next(iter(metrics_data.values())).keys())
        x = np.arange(len(metric_names))
        width = 0.25

        for i, (name, metrics) in enumerate(metrics_data.items()):
            values = [metrics[metric] for metric in metric_names]
            ax2.bar(x + i * width, values, width, label=name, alpha=0.8)

        ax2.set_title('深度学习模型评估指标')
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Value')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(metric_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. 强化学习策略奖励
    ax3 = axes[1, 0]
    for name, result in rl_results.items():
        if result is not None:
            ax3.plot(result['episode_rewards'], label=name, alpha=0.8)
    ax3.set_title('强化学习策略训练奖励')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 强化学习策略平均奖励
    ax4 = axes[1, 1]
    names = []
    avg_rewards = []
    for name, result in rl_results.items():
        if result is not None:
            names.append(name)
            avg_rewards.append(result['avg_reward'])

    if names:
        bars = ax4.bar(names, avg_rewards, alpha=0.8)
        ax4.set_title('强化学习策略平均奖励')
        ax4.set_ylabel('Average Reward')
        ax4.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, reward in zip(bars, avg_rewards):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                     f'{reward:.4f}', ha='center', va='bottom')

    plt.tight_layout()

    # 保存图表
    output_path = 'reports/figures/deep_learning_rl_demo_results.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"结果图表已保存到: {output_path}")

    plt.show()


def main():
    """主函数"""
    logger.info("开始深度学习模型和强化学习策略演示")

    try:
        # 演示深度学习模型
        deep_learning_results = demo_deep_learning_models()

        # 演示强化学习策略
        rl_results = demo_reinforcement_learning_strategies()

        # 绘制结果
        plot_results(deep_learning_results, rl_results)

        logger.info("\n=== 演示完成 ===")
        logger.info("功能扩展演示成功完成！")
        logger.info("新增功能包括:")
        logger.info("1. 深度学习模型: LSTM、CNN、Transformer")
        logger.info("2. 强化学习策略: DQN、PPO、A2C")
        logger.info("3. 完整的训练、预测、评估功能")
        logger.info("4. 详细的测试用例覆盖")

    except Exception as e:
        logger.error(f"演示过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
