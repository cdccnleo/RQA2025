"""
简单管道示例

展示如何快速运行一个简化的ML训练管道
"""

import logging
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import numpy as np

from ..core.pipeline_controller import MLPipelineController, PipelineExecutionResult
from ..core.pipeline_config import PipelineConfig, StageConfig
from ..stages.data_preparation import DataPreparationStage
from ..stages.feature_engineering import FeatureEngineeringStage
from ..stages.model_training import ModelTrainingStage
from ..stages.model_evaluation import ModelEvaluationStage


def run_simple_pipeline(
    symbols: list = None,
    model_type: str = "random_forest",
    skip_evaluation: bool = False
) -> PipelineExecutionResult:
    """
    运行简单4阶段管道
    
    阶段：数据准备 -> 特征工程 -> 模型训练 -> 模型评估
    
    Args:
        symbols: 股票代码列表
        model_type: 模型类型
        skip_evaluation: 是否跳过评估
        
    Returns:
        管道执行结果
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("simple_pipeline")
    
    logger.info("=" * 60)
    logger.info("启动简单ML训练管道")
    logger.info("=" * 60)
    
    # 默认股票代码
    if symbols is None:
        symbols = ["AAPL", "GOOGL", "MSFT"]
    
    # 创建简化配置
    stages = [
        StageConfig(
            name="data_preparation",
            enabled=True,
            config={
                "data_sources": ["market_data"],
                "date_range": "last_30_days",
                "quality_checks": True
            }
        ),
        StageConfig(
            name="feature_engineering",
            enabled=True,
            config={
                "feature_selection": "variance",
                "standardization": "zscore"
            },
            dependencies=["data_preparation"]
        ),
        StageConfig(
            name="model_training",
            enabled=True,
            config={
                "model_type": model_type,
                "target_col": "target"
            },
            dependencies=["feature_engineering"]
        )
    ]
    
    # 添加评估阶段（可选）
    if not skip_evaluation:
        stages.append(
            StageConfig(
                name="model_evaluation",
                enabled=True,
                config={
                    "metrics": ["accuracy", "f1"],
                    "min_accuracy": 0.5
                },
                dependencies=["model_training"]
            )
        )
    
    config = PipelineConfig(
        name="simple_ml_pipeline",
        version="1.0.0",
        stages=stages
    )
    
    # 创建控制器
    controller = MLPipelineController(config)
    
    # 注册阶段
    controller.register_stage(DataPreparationStage())
    controller.register_stage(FeatureEngineeringStage())
    controller.register_stage(ModelTrainingStage())
    if not skip_evaluation:
        controller.register_stage(ModelEvaluationStage())
    
    # 准备上下文
    initial_context = {
        "symbols": symbols,
        "start_date": datetime(2024, 10, 1),
        "end_date": datetime(2024, 12, 31),
        "model_dir": "models"
    }
    
    # 执行管道
    logger.info(f"执行管道: {config.name}")
    logger.info(f"模型类型: {model_type}")
    logger.info(f"股票代码: {symbols}")
    
    result = controller.execute(
        initial_context=initial_context,
        pipeline_id=f"simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # 输出结果
    logger.info("\n" + "=" * 60)
    logger.info("管道执行完成")
    logger.info("=" * 60)
    logger.info(f"状态: {result.status.name}")
    logger.info(f"成功: {result.is_success}")
    
    if result.summary:
        logger.info(f"\n摘要:")
        for key, value in result.summary.items():
            logger.info(f"  - {key}: {value}")
    
    return result


if __name__ == "__main__":
    # 运行简单管道
    result = run_simple_pipeline(
        symbols=["AAPL", "MSFT"],
        model_type="xgboost"
    )
    
    # 检查结果
    if result.is_success:
        print("\n✓ 管道执行成功!")
        print(f"模型路径: {result.state.context.get('model_path', 'N/A')}")
    else:
        print("\n✗ 管道执行失败")
        print(f"错误: {result.error}")
