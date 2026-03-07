#!/usr/bin/env python3
"""
RQA2025 AutoML示例

演示AutoML系统的完整功能：
1. 自动模型选择和超参数优化
2. 智能特征选择
3. 模型解释和可视化
4. 分布式训练
"""

import logging
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.ml.automl.automl_engine import (
    AutoMLConfig, AutoMLEngine, run_automl, create_automl_config
)
from src.ml.automl.feature_selector import (
    select_features_auto
)
from src.ml.automl.model_interpreter import (
    ModelInterpreter
)
from src.ml.automl.distributed_trainer import (
    DistributedConfig, train_distributed_model
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(task_type: str = 'classification', n_samples: int = 1000, n_features: int = 20):
    """创建示例数据集"""
    logger.info(f"创建{task_type}示例数据集: {n_samples}样本, {n_features}特征")

    if task_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            noise=0.1,
            random_state=42
        )

    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.Series(y, name='target')

    return X_df, y_df


def demonstrate_basic_automl():
    """演示基础AutoML功能"""
    logger.info("=== 基础AutoML演示 ===")

    # 创建示例数据
    X, y = create_sample_data('classification', n_samples=500, n_features=15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 配置AutoML
    config = create_automl_config(
        task_type='classification',
        time_limit=300,  # 5分钟
        max_models=5,
        enable_hyperparameter_tuning=True,
        max_tuning_evals=20
    )

    logger.info("开始AutoML训练...")
    start_time = datetime.now()

    # 运行AutoML
    automl_result = run_automl(X_train, y_train, **config.__dict__)

    training_time = (datetime.now() - start_time).total_seconds()

    if automl_result['success']:
        logger.info("AutoML训练完成!")
        logger.info(f"最佳模型: {automl_result['best_model_type']}")
        logger.info(f"训练时间: {training_time:.4f}秒")
        logger.info(f"评估模型数: {automl_result['total_models_evaluated']}")
        logger.info(f"最佳分数: {automl_result.get('best_score', 'N/A'):.4f}")
        # 测试最佳模型
        best_model = automl_result['best_model']
        if best_model and best_model.trained_model:
            # 进行预测
            predictions = best_model.trained_model.predict(X_test)
            test_accuracy = np.mean(predictions == y_test)

            logger.info(f"测试准确率: {test_accuracy:.4f}")
        else:
            logger.error(f"AutoML训练失败: {automl_result.get('error', '未知错误')}")

    return automl_result


def demonstrate_feature_selection():
    """演示特征选择功能"""
    logger.info("=== 特征选择演示 ===")

    # 创建示例数据
    X, y = create_sample_data('classification', n_samples=300, n_features=25)

    # 自动特征选择
    logger.info("执行自动特征选择...")
    X_selected, summary = select_features_auto(X, y, k=10)

    logger.info(f"原始特征数: {X.shape[1]}")
    logger.info(f"选择特征数: {X_selected.shape[1]}")
    logger.info(f"选择的特征: {summary.get('selected_feature_names', [])}")

    # 比较不同特征选择方法
    methods = ['univariate', 'model_based']
    for method in methods:
        try:
            X_method, summary_method = select_features_auto(X, y, method=method, k=8)
            logger.info(f"{method}方法 - 选择特征数: {X_method.shape[1]}")
        except Exception as e:
            logger.warning(f"{method}方法失败: {e}")

    return X_selected, summary


def demonstrate_model_interpretation():
    """演示模型解释功能"""
    logger.info("=== 模型解释演示 ===")

    # 创建简单数据集用于解释
    X, y = create_sample_data('classification', n_samples=200, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练一个简单的模型用于解释
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # 创建模型解释器
    interpreter = ModelInterpreter(model, X_train)

    # 解释单个预测
    test_instance = X_test.iloc[:1]
    logger.info("解释单个预测实例...")
    prediction_explanation = interpreter.explain_prediction(test_instance)

    if prediction_explanation.get('method') != 'error':
        logger.info("预测解释成功")
        if 'feature_importance' in prediction_explanation:
            top_features = sorted(
                prediction_explanation['feature_importance'].items(),
                key=lambda x: x[1], reverse=True
            )[:5]
            logger.info("最重要的特征:")
            for feature, importance in top_features:
                logger.info(f"  {feature}: {importance:.4f}")
    else:
        logger.warning("预测解释失败，使用替代方法")

    # 获取全局特征重要性
    logger.info("获取全局特征重要性...")
    feature_importance = interpreter.get_feature_importance(X_train, top_k=10)

    if feature_importance:
        logger.info("全局特征重要性:")
        for feature, importance in list(feature_importance.items())[:5]:
            logger.info(f"  {feature}: {importance:.4f}")

    # 生成解释报告
    logger.info("生成模型解释报告...")
    explanation_report = interpreter.generate_explanation_report(X_train, y_train)

    if explanation_report.get('method') != 'error':
        logger.info("解释报告生成成功")
        logger.info(f"可用解释方法: {explanation_report.get('available_methods', [])}")

    return explanation_report


def demonstrate_distributed_training():
    """演示分布式训练功能"""
    logger.info("=== 分布式训练演示 ===")

    # 创建示例数据
    X, y = create_sample_data('classification', n_samples=1000, n_features=15)

    # 配置分布式训练
    config = DistributedConfig(
        n_workers=2,  # 使用2个工作节点
        batch_size=32,
        max_epochs=5,  # 减少训练轮数用于演示
        learning_rate=0.1
    )

    logger.info("开始分布式训练...")
    start_time = datetime.now()

    # 运行分布式训练
    result = train_distributed_model(
        model_type='random_forest',
        training_data=X.values,  # 转换为numpy数组
        config=config
    )

    training_time = (datetime.now() - start_time).total_seconds()

    if result['success']:
        logger.info("分布式训练完成!")
        training_metrics = result.get('training_metrics', {})
        if training_metrics.get('global_accuracy'):
            final_accuracy = training_metrics['global_accuracy'][-1]
            logger.info(f"最终准确率: {final_accuracy:.4f}")
    else:
        logger.error(f"分布式训练失败: {result.get('error', '未知错误')}")

    return result


def demonstrate_advanced_automl():
    """演示高级AutoML功能"""
    logger.info("=== 高级AutoML演示 ===")

    # 创建示例数据
    X, y = create_sample_data('classification', n_samples=400, n_features=18)

    # 配置高级AutoML
    config = AutoMLConfig(
        task_type='classification',
        time_limit=600,  # 10分钟
        max_models=6,
        enable_hyperparameter_tuning=True,
        max_tuning_evals=30,
        enable_feature_engineering=True,
        max_features=12,
        cv_folds=3
    )

    # 创建AutoML引擎
    engine = AutoMLEngine(config)

    logger.info("开始高级AutoML训练...")
    start_time = datetime.now()

    # 执行AutoML
    result = engine.fit(X, y)

    training_time = (datetime.now() - start_time).total_seconds()

    logger.info("高级AutoML训练完成!")
    logger.info(f"训练时间: {training_time:.4f}秒")
    logger.info(f"评估模型数: {result.total_models_evaluated}")
    logger.info(f"最佳分数: {result.best_score:.4f}")
    # 显示特征工程信息
    if result.feature_engineering_summary.get('feature_engineering_applied'):
        logger.info("特征工程已应用:")
        logger.info(f"  原始特征: {result.feature_engineering_summary.get('original_features', 0)}")
        logger.info(f"  处理后特征: {result.feature_engineering_summary.get('processed_features', 0)}")

    # 显示最佳模型信息
    if result.best_model:
        logger.info(f"最佳模型: {result.best_model.model_name}")
        logger.info(f"模型分数: {result.best_score:.4f}")
        logger.info(f"训练时间: {result.best_model.training_time:.2f}秒")

        # 显示超参数（如果已优化）
        if hasattr(result.best_model, 'config') and result.best_model.config:
            logger.info("最佳超参数:")
            for param, value in list(result.best_model.config.items())[:3]:
                logger.info(f"  {param}: {value}")

    return result


def run_performance_comparison():
    """运行性能对比"""
    logger.info("=== AutoML性能对比演示 ===")

    # 创建不同规模的数据集
    test_cases = [
        ('小数据集', 200, 10),
        ('中等数据集', 500, 15),
        ('大数据集', 1000, 20)
    ]

    results = {}

    for name, n_samples, n_features in test_cases:
        logger.info(f"测试{name}: {n_samples}样本, {n_features}特征")

        # 创建数据
        X, y = create_sample_data('classification', n_samples, n_features)

        # 配置快速AutoML
        config = create_automl_config(
            task_type='classification',
            time_limit=120,  # 2分钟
            max_models=3,
            enable_hyperparameter_tuning=False  # 禁用超参数调优以加快速度
        )

        start_time = datetime.now()
        result = run_automl(X, y, **config.__dict__)
        elapsed_time = (datetime.now() - start_time).total_seconds()

        if result['success']:
            results[name] = {
                'accuracy': result.get('best_score', 0),
                'time': elapsed_time,
                'models_evaluated': result.get('total_models_evaluated', 0)
            }
            logger.info(f"AutoML完成 - 准确率: {result.get('best_score', 0):.4f}")
        else:
            logger.warning(f"{name} AutoML失败")

    # 显示对比结果
    logger.info("\n性能对比结果:")
    logger.info("-" * 50)
    for name, metrics in results.items():
        logger.info("<15")
    logger.info("-" * 50)

    return results


def main():
    """主函数"""
    logger.info("RQA2025 AutoML系统示例开始")

    try:
        # 1. 基础AutoML演示
        basic_result = demonstrate_basic_automl()

        # 2. 特征选择演示
        selected_features, feature_summary = demonstrate_feature_selection()

        # 3. 模型解释演示
        explanation_report = demonstrate_model_interpretation()

        # 4. 分布式训练演示
        distributed_result = demonstrate_distributed_training()

        # 5. 高级AutoML演示
        advanced_result = demonstrate_advanced_automl()

        # 6. 性能对比演示
        performance_results = run_performance_comparison()

        logger.info("所有AutoML示例完成!")

        # 输出总结
        logger.info("\n" + "="*60)
        logger.info("AutoML系统演示总结")
        logger.info("="*60)

        successful_demos = sum([
            1 if basic_result.get('success') else 0,
            1 if feature_summary.get('method') != 'error' else 0,
            1 if explanation_report.get('method') != 'error' else 0,
            1 if distributed_result.get('success') else 0,
            1 if advanced_result.best_score > 0 else 0,
            1 if performance_results else 0
        ])

        logger.info(f"成功演示: {successful_demos}/6")
        logger.info("✅ AutoML系统功能验证完成!")

    except Exception as e:
        logger.error(f"AutoML示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
