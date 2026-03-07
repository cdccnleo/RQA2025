#!/usr/bin/env python3
"""
RQA2025 ML错误处理示例

演示ML错误处理系统的功能，包括：
1. 错误分类和处理
2. 自动恢复机制
3. 错误统计和报告
4. 自定义错误恢复策略
"""

import logging
import time
import random
from datetime import datetime

from src.ml.error_handling import (
    handle_ml_error, get_error_statistics, get_recent_errors,
    register_error_recovery_strategy, register_error_callback,
    MLErrorCategory, MLErrorSeverity, ErrorRecoveryStrategy,
    DataValidationError, ModelLoadError, TrainingError, InferenceError,
    ml_error_handler
)
from src.ml.process_orchestrator import create_ml_process, submit_ml_process

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simulate_data_errors():
    """模拟数据相关错误"""
    logger.info("=== 模拟数据错误场景 ===")

    error_scenarios = [
        ("数据文件不存在", "file_not_found.csv"),
        ("数据格式错误", "invalid_format.json"),
        ("数据缺失值过多", "missing_values.csv"),
        ("数据类型不匹配", "type_mismatch.csv")
    ]

    for error_msg, data_file in error_scenarios:
        try:
            # 模拟数据加载失败
            if random.random() < 0.7:  # 70%概率失败
                raise FileNotFoundError(f"数据文件不存在: {data_file}")

            # 模拟数据验证失败
            if random.random() < 0.5:  # 50%概率验证失败
                raise DataValidationError(
                    f"数据验证失败: {error_msg}",
                    context={'data_file': data_file, 'error_type': 'validation'}
                )

        except Exception as e:
            # 使用统一的错误处理
            context = {
                'operation': 'data_loading',
                'data_file': data_file,
                'timestamp': datetime.now().isoformat()
            }
            handle_ml_error(e, context)
            logger.warning(f"数据处理异常: {error_msg}")

        time.sleep(0.5)

    logger.info("数据错误模拟完成")


def simulate_model_errors():
    """模拟模型相关错误"""
    logger.info("=== 模拟模型错误场景 ===")

    model_operations = [
        ("模型加载", "load_model"),
        ("模型保存", "save_model"),
        ("模型训练", "train_model"),
        ("模型推理", "predict_model")
    ]

    for operation_name, operation_type in model_operations:
        try:
            # 模拟模型加载失败
            if operation_type == "load_model" and random.random() < 0.6:
                raise ModelLoadError(
                    f"{operation_name}失败: 模型文件损坏",
                    model_id=f"model_{random.randint(1, 100)}",
                    context={'operation': operation_type}
                )

            # 模拟训练失败
            if operation_type == "train_model" and random.random() < 0.4:
                raise TrainingError(
                    f"{operation_name}失败: 数据不足",
                    context={
                        'operation': operation_type,
                        'training_samples': random.randint(10, 100)
                    }
                )

            # 模拟推理失败
            if operation_type == "predict_model" and random.random() < 0.3:
                raise InferenceError(
                    f"{operation_name}失败: 模型输入格式错误",
                    context={
                        'operation': operation_type,
                        'input_shape': (random.randint(1, 100), random.randint(1, 50))
                    }
                )

            # 模拟成功操作
            logger.info(f"{operation_name}成功完成")

        except Exception as e:
            context = {
                'operation': operation_type,
                'operation_name': operation_name,
                'timestamp': datetime.now().isoformat()
            }
            handle_ml_error(e, context)

        time.sleep(0.3)

    logger.info("模型错误模拟完成")


def demonstrate_error_recovery():
    """演示错误恢复机制"""
    logger.info("=== 演示错误恢复机制 ===")

    # 注册自定义恢复策略
    def custom_inference_recovery(error):
        """自定义推理错误恢复策略"""
        logger.info("执行自定义推理恢复策略...")
        # 模拟恢复操作
        time.sleep(0.5)
        return {
            "status": "recovered",
            "action": "switched_to_backup_model",
            "recovery_time": datetime.now().isoformat()
        }

    # 注册恢复策略
    recovery_strategy = ErrorRecoveryStrategy(
        strategy_id="custom_inference_recovery",
        error_category=MLErrorCategory.INFERENCE_ERROR,
        condition=lambda e: e.severity == MLErrorSeverity.HIGH,
        recovery_action=custom_inference_recovery,
        max_attempts=2,
        cooldown_seconds=30,
        priority=10
    )

    register_error_recovery_strategy(recovery_strategy)
    logger.info("已注册自定义错误恢复策略")

    # 注册错误回调
    def error_notification_callback(error):
        """错误通知回调"""
        logger.warning(f"错误通知: [{error.category.value}] {error.message}")

    register_error_callback(MLErrorCategory.INFERENCE_ERROR, error_notification_callback)
    logger.info("已注册错误通知回调")

    # 模拟需要恢复的错误
    for i in range(3):
        try:
            if random.random() < 0.8:  # 80%概率触发错误
                raise InferenceError(
                    f"模拟推理错误 #{i+1}",
                    context={'error_sequence': i+1}
                )
            else:
                logger.info(f"推理操作 #{i+1} 成功")
        except Exception as e:
            handle_ml_error(e)

        time.sleep(1)

    logger.info("错误恢复演示完成")


@ml_error_handler(MLErrorCategory.SYSTEM_ERROR, MLErrorSeverity.MEDIUM)
def risky_ml_operation(operation_name: str, failure_rate: float = 0.3):
    """有风险的ML操作（使用错误处理装饰器）"""
    if random.random() < failure_rate:
        if random.random() < 0.5:
            raise TrainingError(f"{operation_name} 训练失败")
        else:
            raise InferenceError(f"{operation_name} 推理失败")

    logger.info(f"{operation_name} 成功完成")
    return {"status": "success", "operation": operation_name}


def demonstrate_decorator_usage():
    """演示装饰器使用"""
    logger.info("=== 演示错误处理装饰器 ===")

    operations = [
        "数据预处理",
        "特征工程",
        "模型训练",
        "模型验证",
        "模型部署"
    ]

    for operation in operations:
        try:
            result = risky_ml_operation(operation, failure_rate=0.4)
            logger.info(f"操作结果: {result}")
        except Exception as e:
            logger.error(f"操作失败，已通过装饰器处理: {e}")

        time.sleep(0.5)

    logger.info("装饰器演示完成")


def create_error_prone_process():
    """创建容易出错的ML流程"""
    logger.info("=== 创建错误易发ML流程 ===")

    process = create_ml_process(
        process_type="model_training",
        process_name="错误易发训练流程",
        steps={
            'data_load': {
                'step_id': 'data_load',
                'step_name': '数据加载',
                'step_type': 'data_loading',
                'config': {
                    'data_source': 'file',
                    'data_format': 'csv',
                    'data_path': 'nonexistent_data.csv'  # 故意设置不存在的文件
                }
            },
            'feature_eng': {
                'step_id': 'feature_eng',
                'step_name': '特征工程',
                'step_type': 'feature_engineering',
                'dependencies': ['data_load'],
                'config': {
                    'target_column': 'target',
                    'scaling_method': 'invalid_method'  # 故意设置无效的方法
                }
            }
        },
        config={'experiment_name': 'error_handling_demo'},
        timeout=60
    )

    # 提交流程
    process_id = submit_ml_process(process)
    logger.info(f"已提交流程: {process_id}")

    return process_id


def generate_error_report():
    """生成错误报告"""
    logger.info("=== 生成错误报告 ===")

    # 获取错误统计
    stats = get_error_statistics()
    logger.info("错误统计信息:")
    logger.info(f"  总错误数: {stats['total_errors']}")
    logger.info(f"  活跃错误: {stats['active_errors']}")
    logger.info(f"  已解决错误: {stats['resolved_errors']}")
    logger.info(f"  恢复成功率: {stats['recovery_success_rate']:.2%}")

    # 显示错误分布
    error_distribution = stats['error_distribution']
    logger.info("错误类别分布:")
    for category, count in error_distribution.items():
        logger.info(f"  {category}: {count}")

    # 获取最近错误
    recent_errors = get_recent_errors(5)
    logger.info(f"\n最近 {len(recent_errors)} 个错误:")
    for error in recent_errors:
        logger.info(f"  [{error['severity']}] {error['category']}: {error['message']}")

    logger.info("错误报告生成完成")


def main():
    """主函数"""
    logger.info("RQA2025 ML错误处理示例开始")

    try:
        # 1. 模拟数据错误
        simulate_data_errors()

        # 2. 模拟模型错误
        simulate_model_errors()

        # 3. 演示错误恢复机制
        demonstrate_error_recovery()

        # 4. 演示装饰器使用
        demonstrate_decorator_usage()

        # 5. 创建并监控错误易发流程
        process_id = create_error_prone_process()

        # 等待流程执行并处理错误
        time.sleep(3)

        # 6. 生成错误报告
        generate_error_report()

        logger.info("所有ML错误处理示例完成")

    except Exception as e:
        logger.error(f"ML错误处理示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
