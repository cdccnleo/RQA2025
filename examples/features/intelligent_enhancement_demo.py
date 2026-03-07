# examples/features/intelligent_enhancement_demo.py
"""
智能化增强功能演示脚本
展示自动特征选择、智能告警和机器学习模型集成的功能
"""

from src.features.intelligent.smart_alert_system import AlertRule, AlertType, AlertLevel
from src.features.intelligent.intelligent_enhancement_manager import IntelligentEnhancementManager
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def create_sample_data(n_samples: int = 1000, n_features: int = 20) -> tuple:
    """创建示例数据"""
    print("创建示例数据...")

    # 生成特征数据
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # 生成目标变量（分类任务）
    y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]))

    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"目标变量分布: {y.value_counts().to_dict()}")

    return X, y


def demo_auto_feature_selection():
    """演示自动特征选择功能"""
    print("\n" + "="*50)
    print("演示自动特征选择功能")
    print("="*50)

    # 创建数据
    X, y = create_sample_data(n_samples=500, n_features=15)

    # 初始化增强管理器
    manager = IntelligentEnhancementManager(
        enable_auto_feature_selection=True,
        enable_smart_alerts=False,
        enable_ml_integration=False
    )

    # 执行特征增强
    X_enhanced, enhancement_info = manager.enhance_features(X, y, target_features=10)

    print(f"原始特征数量: {len(X.columns)}")
    print(f"选择后特征数量: {len(X_enhanced.columns)}")

    # 安全地获取特征选择信息
    if 'feature_selection' in enhancement_info and 'reduction_ratio' in enhancement_info['feature_selection']:
        reduction_ratio = enhancement_info['feature_selection']['reduction_ratio']
        print(f"特征减少比例: {reduction_ratio:.2%}")
    else:
        print("特征选择信息不完整")

    if 'feature_selection' in enhancement_info and 'selected_features' in enhancement_info['feature_selection']:
        selected_features = enhancement_info['feature_selection']['selected_features']
        print(f"选择的特征: {selected_features}")
    else:
        print("未获取到选择的特征")

    if 'feature_selection' in enhancement_info and 'selection_info' in enhancement_info['feature_selection']:
        selection_info = enhancement_info['feature_selection']['selection_info']
        if 'method' in selection_info:
            print(f"选择方法: {selection_info['method']}")
        else:
            print("选择方法信息不完整")


def demo_smart_alert_system():
    """演示智能告警系统功能"""
    print("\n" + "="*50)
    print("演示智能告警系统功能")
    print("="*50)

    # 初始化增强管理器
    manager = IntelligentEnhancementManager(
        enable_auto_feature_selection=False,
        enable_smart_alerts=True,
        enable_ml_integration=False
    )

    # 添加自定义告警规则
    custom_rule = AlertRule(
        name="custom_feature_count",
        alert_type=AlertType.THRESHOLD,
        metric="feature_count",
        condition=">",
        threshold=15,
        level=AlertLevel.WARNING,
        description="特征数量过多告警"
    )
    manager.add_custom_alert_rule(custom_rule)

    # 创建不同场景的数据
    scenarios = [
        ("正常数据", create_sample_data(100, 10)),
        ("高维数据", create_sample_data(100, 25)),
        ("缺失值数据", create_sample_data_with_missing(100, 10)),
        ("异常值数据", create_sample_data_with_outliers(100, 10))
    ]

    for scenario_name, (X, y) in scenarios:
        print(f"\n测试场景: {scenario_name}")
        print(f"数据形状: X={X.shape}")

        # 执行增强（主要是告警检查）
        _, enhancement_info = manager.enhance_features(X, y)

        alerts = enhancement_info.get('alerts', [])
        if alerts:
            print(f"检测到 {len(alerts)} 个告警:")
            for alert in alerts:
                print(f"  - {alert['message']} (级别: {alert['level']})")
        else:
            print("未检测到告警")


def demo_ml_model_integration():
    """演示机器学习模型集成功能"""
    print("\n" + "="*50)
    print("演示机器学习模型集成功能")
    print("="*50)

    # 创建数据
    X, y = create_sample_data(n_samples=300, n_features=12)

    # 初始化增强管理器
    manager = IntelligentEnhancementManager(
        enable_auto_feature_selection=True,
        enable_smart_alerts=False,
        enable_ml_integration=True
    )

    # 执行特征增强
    X_enhanced, enhancement_info = manager.enhance_features(X, y, target_features=8)

    if 'ml_integration' in enhancement_info and 'performance' in enhancement_info['ml_integration']:
        print("模型性能:")
        performance = enhancement_info['ml_integration']['performance']
        for model_name, metrics in performance.items():
            print(f"  {model_name}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
    else:
        print("模型性能信息不完整")

    # 进行预测
    print("\n进行预测测试...")
    try:
        test_X = X_enhanced.iloc[:10]  # 使用前10个样本进行测试
        predictions, prediction_info = manager.predict_with_enhanced_model(test_X)

        print(f"预测结果形状: {predictions.shape}")
        print(f"预测方法: {prediction_info['prediction_method']}")
        print(f"是否应用特征选择: {prediction_info['feature_selection_applied']}")
    except Exception as e:
        print(f"预测失败: {e}")


def demo_full_integration():
    """演示完整集成功能"""
    print("\n" + "="*50)
    print("演示完整集成功能")
    print("="*50)

    # 创建数据
    X, y = create_sample_data(n_samples=400, n_features=18)

    # 初始化增强管理器（启用所有功能）
    manager = IntelligentEnhancementManager(
        enable_auto_feature_selection=True,
        enable_smart_alerts=True,
        enable_ml_integration=True
    )

    # 执行完整增强
    X_enhanced, enhancement_info = manager.enhance_features(X, y, target_features=12)

    print("增强结果摘要:")
    print(f"原始特征数量: {enhancement_info['original_shape'][1]}")
    print(f"增强后特征数量: {X_enhanced.shape[1]}")
    print(f"执行的增强步骤: {enhancement_info['enhancement_steps']}")

    # 检查告警
    alerts = enhancement_info.get('alerts', [])
    print(f"检测到的告警数量: {len(alerts)}")

    # 模型性能
    if 'ml_integration' in enhancement_info:
        ml_info = enhancement_info['ml_integration']
        if 'performance' in ml_info and 'best_model' in ml_info:
            performance = ml_info['performance']
            best_model = ml_info['best_model']
            print(f"最佳模型: {best_model}")
            if best_model in performance:
                best_metrics = performance[best_model]
                print(f"最佳模型性能: {best_metrics}")

    # 获取增强摘要
    summary = manager.get_enhancement_summary()
    print(f"\n增强功能摘要:")
    print(f"当前特征: {summary['current_features']}")
    print(f"告警数量: {summary['current_alerts_count']}")
    print(f"增强历史记录数: {summary['enhancement_history_count']}")


def create_sample_data_with_missing(n_samples: int, n_features: int) -> tuple:
    """创建包含缺失值的数据"""
    X, y = create_sample_data(n_samples, n_features)

    # 添加缺失值
    missing_indices = np.random.choice(X.shape[0], size=int(X.shape[0] * 0.1), replace=False)
    missing_cols = np.random.choice(X.shape[1], size=int(X.shape[1] * 0.2), replace=False)

    for idx in missing_indices:
        for col in missing_cols:
            X.iloc[idx, col] = np.nan

    return X, y


def create_sample_data_with_outliers(n_samples: int, n_features: int) -> tuple:
    """创建包含异常值的数据"""
    X, y = create_sample_data(n_samples, n_features)

    # 添加异常值
    outlier_indices = np.random.choice(X.shape[0], size=int(X.shape[0] * 0.05), replace=False)
    outlier_cols = np.random.choice(X.shape[1], size=int(X.shape[1] * 0.3), replace=False)

    for idx in outlier_indices:
        for col in outlier_cols:
            X.iloc[idx, col] = np.random.choice([-1000, 1000])

    return X, y


def demo_state_management():
    """演示状态管理功能"""
    print("\n" + "="*50)
    print("演示状态管理功能")
    print("="*50)

    # 创建数据
    X, y = create_sample_data(n_samples=200, n_features=15)

    # 初始化增强管理器
    manager = IntelligentEnhancementManager()

    # 执行增强
    X_enhanced, enhancement_info = manager.enhance_features(X, y, target_features=10)

    # 保存状态
    state_file = Path("temp_enhancement_state.json")
    manager.save_enhancement_state(state_file)
    print(f"增强状态已保存到: {state_file}")

    # 创建新的管理器并加载状态
    new_manager = IntelligentEnhancementManager()
    new_manager.load_enhancement_state(state_file)
    print("增强状态已加载")

    # 验证状态
    summary = new_manager.get_enhancement_summary()
    print(f"加载后的特征数量: {len(summary['current_features']) if summary['current_features'] else 0}")
    print(f"加载后的告警数量: {summary['current_alerts_count']}")

    # 清理临时文件
    if state_file.exists():
        state_file.unlink()
        print("临时文件已清理")


def main():
    """主函数"""
    print("智能化增强功能演示")
    print("="*60)

    try:
        # 演示各个功能模块
        demo_auto_feature_selection()
        demo_smart_alert_system()
        demo_ml_model_integration()
        demo_full_integration()
        demo_state_management()

        print("\n" + "="*60)
        print("所有演示完成！")
        print("="*60)

    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
