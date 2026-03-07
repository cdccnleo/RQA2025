"""
测试深度学习核心分布式训练器
"""

import pytest
import pandas as pd
from src.ml.deep_learning.core.distributed_trainer import DistributedTrainer, DistributedConfig


class TestDistributedConfig:
    """测试分布式配置"""

    def test_distributed_config_default_values(self):
        """测试分布式配置默认值"""
        config = DistributedConfig()
        assert config.epochs == 1
        assert config.learning_rate == 0.001

    def test_distributed_config_custom_values(self):
        """测试分布式配置自定义值"""
        config = DistributedConfig(epochs=10, learning_rate=0.01)
        assert config.epochs == 10
        assert config.learning_rate == 0.01


class TestDistributedTrainer:
    """测试分布式训练器"""

    def setup_method(self):
        """测试前准备"""
        self.config = DistributedConfig(epochs=5, learning_rate=0.005)
        self.trainer = DistributedTrainer(self.config)

    def test_distributed_trainer_init(self):
        """测试分布式训练器初始化"""
        assert self.trainer.config == self.config
        assert self.trainer.config.epochs == 5
        assert self.trainer.config.learning_rate == 0.005

    def test_distributed_trainer_run_success(self):
        """测试分布式训练器成功运行"""
        # 创建测试数据
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })

        result = self.trainer.run(data)

        # 验证结果
        assert isinstance(result, dict)
        assert result["epochs"] == 5
        assert result["learning_rate"] == 0.005
        assert "metrics" in result
        assert result["metrics"]["loss"] == 0.0

    def test_distributed_trainer_run_empty_data_raises(self):
        """测试分布式训练器空数据异常"""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="训练数据不能为空"):
            self.trainer.run(empty_data)

    def test_distributed_trainer_run_with_default_config(self):
        """测试分布式训练器使用默认配置"""
        default_config = DistributedConfig()
        default_trainer = DistributedTrainer(default_config)

        data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [0, 1, 0]
        })

        result = default_trainer.run(data)
        assert result["epochs"] == 1
        assert result["learning_rate"] == 0.001
        assert result["metrics"]["loss"] == 0.0
