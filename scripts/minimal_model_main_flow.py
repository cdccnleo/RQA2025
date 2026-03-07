#!/usr/bin/env python3
"""
RQA2025 最小化模型主流程脚本
用于测试模型层的主要功能流程
"""

import numpy as np
import pandas as pd
import sys
import os
import logging
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 检查src目录是否存在
src_path = project_root / "src"
if not src_path.exists():
    # 如果src目录不存在，尝试其他可能的路径
    possible_paths = [
        project_root,
        project_root.parent,
        Path.cwd(),
        Path.cwd().parent
    ]

    for path in possible_paths:
        if (path / "src").exists():
            project_root = path
            sys.path.insert(0, str(project_root))
            break

try:
    from src.models.model_manager import ModelManager
    from src.models.base_model import BaseModel
except ImportError as e:
    # 如果导入失败，尝试直接导入
    try:
        sys.path.insert(0, str(project_root / "src"))
        from models.model_manager import ModelManager
        from models.base_model import BaseModel
    except ImportError:
        # 最后的尝试：使用相对导入
        sys.path.insert(0, str(project_root / "src" / "models"))
        try:
            from model_manager import ModelManager
            from base_model import BaseModel
        except ImportError:
            print(f"无法导入必要的模块: {e}")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"Python路径: {sys.path}")
            sys.exit(1)


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockModel(BaseModel):
    """模拟模型类，用于测试"""

    def __init__(self, name="mock_model"):
        config = {
            "model_type": "mock",
            "model_params": {"name": name}
        }
        super().__init__(config)
        self.name = name
        self._is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """训练模型"""
        if X.empty or y.empty:
            raise ValueError("训练数据不能为空")
        if len(X) != len(y):
            raise ValueError("特征和目标变量长度不匹配")
        self._is_trained = True
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        if not self._is_trained:
            raise RuntimeError("模型未训练")
        if X.empty:
            raise ValueError("预测数据不能为空")
        return np.array([0.5] * len(X))

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """评估模型"""
        if not self._is_trained:
            raise RuntimeError("模型未训练")
        if X.empty or y.empty:
            raise ValueError("评估数据不能为空")
        return {"accuracy": 0.85, "precision": 0.83, "recall": 0.87}


def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples), name='target')

    return X, y


def main():
    """主函数"""
    try:
        logger.info("开始模型主流程测试")

        # 创建模型管理器
        model_manager = ModelManager()
        logger.info("模型管理器创建成功")

        # 创建示例数据
        X, y = create_sample_data()
        logger.info(f"创建示例数据: {X.shape}, {y.shape}")

        # 创建并添加模型
        model1 = MockModel("model_1")
        model2 = MockModel("model_2")

        model_manager.add_model("model_1", model1)
        model_manager.add_model("model_2", model2)
        logger.info("模型添加成功")

        # 训练模型
        model1.train(X, y)
        model2.train(X, y)
        logger.info("模型训练完成")

        # 进行预测
        predictions = model_manager.predict_all(X)
        logger.info(f"批量预测完成: {predictions.shape}")

        # 评估模型
        for name in ["model_1", "model_2"]:
            model = model_manager.get_model(name)
            metrics = model.evaluate(X, y)
            logger.info(f"模型 {name} 评估结果: {metrics}")

        # 保存模型信息
        model_info = model_manager.get_model_statistics()
        logger.info(f"模型统计信息: {model_info}")

        logger.info("模型主流程测试完成")
        return 0

    except Exception as e:
        logger.error(f"模型主流程测试失败: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
