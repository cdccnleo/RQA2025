"""
ML/RL策略工厂

用于统一创建和管理ML/RL策略实例
"""

from typing import Dict, Any, Optional
from enum import Enum
import logging
import os

from .model_driven_strategy import ModelDrivenStrategy, ModelStrategyConfig

logger = logging.getLogger(__name__)


class MLRLStrategyType(Enum):
    """ML/RL策略类型"""
    MODEL_DRIVEN = "model_driven"
    DQN = "dqn"
    PPO = "ppo"
    A2C = "a2c"


class MLRLStrategyFactory:
    """ML/RL策略工厂"""

    _strategies: Dict[str, Any] = {}
    _strategy_metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def create_strategy(cls, strategy_type: MLRLStrategyType,
                       config: Dict[str, Any]) -> Any:
        """
        创建ML/RL策略实例

        Args:
            strategy_type: 策略类型
            config: 策略配置

        Returns:
            策略实例
        """
        strategy_id = config.get('strategy_id', f"mlrl_{strategy_type.value}_{id(config)}")

        # 检查是否已存在
        if strategy_id in cls._strategies:
            logger.info(f"返回已存在的策略实例: {strategy_id}")
            return cls._strategies[strategy_id]

        try:
            if strategy_type == MLRLStrategyType.MODEL_DRIVEN:
                strategy = cls._create_model_driven_strategy(config)

            elif strategy_type == MLRLStrategyType.DQN:
                strategy = cls._create_dqn_strategy(config)

            elif strategy_type == MLRLStrategyType.PPO:
                strategy = cls._create_ppo_strategy(config)

            elif strategy_type == MLRLStrategyType.A2C:
                strategy = cls._create_a2c_strategy(config)

            else:
                raise ValueError(f"不支持的策略类型: {strategy_type}")

            # 缓存策略实例
            cls._strategies[strategy_id] = strategy
            cls._strategy_metadata[strategy_id] = {
                'type': strategy_type.value,
                'config': config,
                'created_at': __import__('time').time()
            }

            logger.info(f"创建ML/RL策略: {strategy_id}, 类型: {strategy_type.value}")
            return strategy

        except Exception as e:
            logger.error(f"创建策略失败: {strategy_id}, 错误: {e}")
            raise

    @classmethod
    def _create_model_driven_strategy(cls, config: Dict[str, Any]) -> ModelDrivenStrategy:
        """创建模型驱动策略"""
        model_config = ModelStrategyConfig(
            model_id=config.get('model_id', config.get('strategy_id')),
            prediction_threshold=config.get('prediction_threshold', 0.5),
            confidence_threshold=config.get('confidence_threshold', 0.7),
            position_sizing=config.get('position_sizing', 'equal'),
            max_position_size=config.get('max_position_size', 0.2)
        )
        return ModelDrivenStrategy(model_config)

    @classmethod
    def _create_dqn_strategy(cls, config: Dict[str, Any]):
        """创建DQN策略"""
        from .strategies.reinforcement_learning import DQNStrategy

        agent_params = config.get('agent_params', {})
        strategy = DQNStrategy(**agent_params)

        # 加载预训练模型
        model_path = config.get('model_path')
        if model_path and os.path.exists(model_path):
            strategy.load(model_path)
            logger.info(f"DQN模型已加载: {model_path}")

        return strategy

    @classmethod
    def _create_ppo_strategy(cls, config: Dict[str, Any]):
        """创建PPO策略"""
        from .strategies.reinforcement_learning import PPOStrategy

        agent_params = config.get('agent_params', {})
        strategy = PPOStrategy(**agent_params)

        # 加载预训练模型
        model_path = config.get('model_path')
        if model_path and os.path.exists(model_path):
            strategy.load(model_path)
            logger.info(f"PPO模型已加载: {model_path}")

        return strategy

    @classmethod
    def _create_a2c_strategy(cls, config: Dict[str, Any]):
        """创建A2C策略"""
        from .strategies.reinforcement_learning import A2CStrategy

        agent_params = config.get('agent_params', {})
        strategy = A2CStrategy(**agent_params)

        # 加载预训练模型
        model_path = config.get('model_path')
        if model_path and os.path.exists(model_path):
            strategy.load(model_path)
            logger.info(f"A2C模型已加载: {model_path}")

        return strategy

    @classmethod
    def get_strategy(cls, strategy_id: str) -> Optional[Any]:
        """
        获取策略实例

        Args:
            strategy_id: 策略ID

        Returns:
            策略实例或None
        """
        return cls._strategies.get(strategy_id)

    @classmethod
    def remove_strategy(cls, strategy_id: str) -> bool:
        """
        移除策略实例

        Args:
            strategy_id: 策略ID

        Returns:
            是否成功移除
        """
        if strategy_id in cls._strategies:
            # 清理资源
            strategy = cls._strategies[strategy_id]
            if hasattr(strategy, 'cleanup'):
                strategy.cleanup()

            del cls._strategies[strategy_id]
            if strategy_id in cls._strategy_metadata:
                del cls._strategy_metadata[strategy_id]

            logger.info(f"移除ML/RL策略: {strategy_id}")
            return True
        return False

    @classmethod
    def list_strategies(cls) -> Dict[str, Dict[str, Any]]:
        """
        列出所有策略

        Returns:
            策略元数据字典
        """
        return cls._strategy_metadata.copy()

    @classmethod
    def clear_all(cls):
        """清除所有策略实例"""
        for strategy_id in list(cls._strategies.keys()):
            cls.remove_strategy(strategy_id)
        logger.info("所有ML/RL策略已清除")

    @classmethod
    def get_strategy_type(cls, strategy_id: str) -> Optional[MLRLStrategyType]:
        """
        获取策略类型

        Args:
            strategy_id: 策略ID

        Returns:
            策略类型或None
        """
        metadata = cls._strategy_metadata.get(strategy_id)
        if metadata:
            return MLRLStrategyType(metadata['type'])
        return None

    @classmethod
    def update_strategy_config(cls, strategy_id: str, config_updates: Dict[str, Any]) -> bool:
        """
        更新策略配置

        Args:
            strategy_id: 策略ID
            config_updates: 配置更新

        Returns:
            是否成功更新
        """
        if strategy_id not in cls._strategies:
            logger.warning(f"策略不存在，无法更新配置: {strategy_id}")
            return False

        try:
            # 更新元数据
            if strategy_id in cls._strategy_metadata:
                cls._strategy_metadata[strategy_id]['config'].update(config_updates)

            # 更新策略实例配置
            strategy = cls._strategies[strategy_id]
            if hasattr(strategy, 'update_config'):
                strategy.update_config(config_updates)

            logger.info(f"策略配置已更新: {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"更新策略配置失败: {strategy_id}, 错误: {e}")
            return False


# 便捷函数
def create_ml_strategy(config: Dict[str, Any]) -> ModelDrivenStrategy:
    """
    创建ML策略的便捷函数

    Args:
        config: 策略配置

    Returns:
        ML策略实例
    """
    return MLRLStrategyFactory.create_strategy(MLRLStrategyType.MODEL_DRIVEN, config)


def create_rl_strategy(agent_type: str, config: Dict[str, Any]):
    """
    创建RL策略的便捷函数

    Args:
        agent_type: 智能体类型 (dqn/ppo/a2c)
        config: 策略配置

    Returns:
        RL策略实例
    """
    type_map = {
        'dqn': MLRLStrategyType.DQN,
        'ppo': MLRLStrategyType.PPO,
        'a2c': MLRLStrategyType.A2C
    }

    strategy_type = type_map.get(agent_type.lower())
    if not strategy_type:
        raise ValueError(f"不支持的RL类型: {agent_type}")

    return MLRLStrategyFactory.create_strategy(strategy_type, config)
