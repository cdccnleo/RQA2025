#!/usr/bin/env python3
"""
RQA2025 ML业务流程示例

演示如何使用ML业务流程驱动架构进行模型训练、预测等操作
"""

import logging

from src.ml.process_orchestrator import (
    MLProcessType, ProcessPriority, create_ml_process, submit_ml_process,
    get_ml_process_status, get_ml_orchestrator_stats
)
from src.ml.process_builder import (
    get_ml_process_builder, quick_predict
)
from src.ml.step_executors import register_ml_step_executors

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_training():
    """基础模型训练示例"""
    logger.info("=== 基础模型训练示例 ===")

    # 创建流程构建器
    builder = get_ml_process_builder()

    # 从模板创建流程
    process = builder.from_template('basic_training', {
        'name': '示例模型训练流程',
        'config': {
            'data_path': 'data/sample_train.csv',
            'target_column': 'target'
        }
    }).set_priority(ProcessPriority.HIGH).build()

    # 提交流程
    process_id = submit_ml_process(process)
    logger.info(f"已提交流程: {process_id}")

    # 监控流程状态
    import time
    while True:
        status = get_ml_process_status(process_id)
        if status:
            logger.info(f"流程状态: {status['status']} (进度: {status['progress']:.1%})")
            if status['status'] in ['completed', 'failed']:
                break
        time.sleep(2)

    return process_id


def example_custom_process():
    """自定义流程示例"""
    logger.info("=== 自定义流程示例 ===")

    # 创建自定义流程
    process = create_ml_process(
        process_type=MLProcessType.MODEL_TRAINING,
        process_name="自定义ML流程",
        steps={
            'load_data': {
                'step_id': 'load_data',
                'step_name': '加载数据',
                'step_type': 'data_loading',
                'config': {
                    'data_source': 'file',
                    'data_format': 'csv',
                    'data_path': 'data/custom_data.csv'
                }
            },
            'feature_eng': {
                'step_id': 'feature_eng',
                'step_name': '特征工程',
                'step_type': 'feature_engineering',
                'dependencies': ['load_data'],
                'config': {
                    'target_column': 'label',
                    'scaling_method': 'robust',
                    'encoding_method': 'label'
                }
            },
            'train': {
                'step_id': 'train',
                'step_name': '训练模型',
                'step_type': 'model_training',
                'dependencies': ['feature_eng'],
                'config': {
                    'model_type': 'xgboost',
                    'model_config': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1
                    }
                }
            },
            'evaluate': {
                'step_id': 'evaluate',
                'step_name': '评估模型',
                'step_type': 'model_evaluation',
                'dependencies': ['train'],
                'config': {
                    'metrics': ['accuracy', 'auc', 'log_loss']
                }
            }
        },
        config={'experiment_name': 'custom_ml_experiment'},
        priority=ProcessPriority.NORMAL,
        timeout=1800  # 30分钟超时
    )

    # 提交流程
    process_id = submit_ml_process(process)
    logger.info(f"已提交流程: {process_id}")

    return process_id


def example_hyperparameter_tuning():
    """超参数调优示例"""
    logger.info("=== 超参数调优示例 ===")

    builder = get_ml_process_builder()

    process = builder.from_template('complete_ml_pipeline', {
        'name': '超参数调优流程',
        'config': {
            'data_path': 'data/tuning_data.csv',
            'target_column': 'target',
            'model_type': 'random_forest'
        }
    }).configure_step('tune_hyperparameters', {
        'param_space': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'tuning_method': 'grid',
        'cv_folds': 3,
        'max_evals': 20
    }).build()

    process_id = submit_ml_process(process)
    logger.info(f"已提交流程: {process_id}")

    return process_id


def example_batch_prediction():
    """批量预测示例"""
    logger.info("=== 批量预测示例 ===")

    # 使用快速预测函数
    process_id = quick_predict(
        data_path='data/prediction_data.csv',
        model_id='trained_model_001',
        config={
            'batch_size': 500,
            'output_format': 'dataframe'
        }
    )

    logger.info(f"快速预测流程已提交: {process_id}")
    return process_id


def monitor_processes(process_ids):
    """监控多个流程"""
    logger.info("=== 流程监控 ===")

    import time
    active_processes = set(process_ids)

    while active_processes:
        for process_id in list(active_processes):
            status = get_ml_process_status(process_id)
            if status:
                logger.info(f"流程 {process_id}: {status['status']} "
                            f"(进度: {status['progress']:.1%}, "
                            f"步骤: {status['completed_steps']}/{status['step_count']})")

                if status['status'] in ['completed', 'failed']:
                    active_processes.remove(process_id)

        time.sleep(5)

    # 显示最终统计
    stats = get_ml_orchestrator_stats()
    logger.info(f"编排器统计: 活跃流程 {stats['active_processes']}, "
                f"已完成 {stats['completed_processes']}, "
                f"失败 {stats['failed_processes']}")


def main():
    """主函数"""
    logger.info("RQA2025 ML业务流程示例开始")

    try:
        # 注册步骤执行器
        from src.ml.process_orchestrator import get_ml_process_orchestrator
        orchestrator = get_ml_process_orchestrator()
        register_ml_step_executors(orchestrator)

        # 运行各种示例
        process_ids = []

        # 1. 基础训练示例
        try:
            process_ids.append(example_basic_training())
        except Exception as e:
            logger.error(f"基础训练示例失败: {e}")

        # 2. 自定义流程示例
        try:
            process_ids.append(example_custom_process())
        except Exception as e:
            logger.error(f"自定义流程示例失败: {e}")

        # 3. 超参数调优示例
        try:
            process_ids.append(example_hyperparameter_tuning())
        except Exception as e:
            logger.error(f"超参数调优示例失败: {e}")

        # 4. 批量预测示例
        try:
            process_ids.append(example_batch_prediction())
        except Exception as e:
            logger.error(f"批量预测示例失败: {e}")

        # 监控所有流程
        if process_ids:
            monitor_processes(process_ids)

        logger.info("所有ML业务流程示例完成")

    except Exception as e:
        logger.error(f"ML业务流程示例执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
