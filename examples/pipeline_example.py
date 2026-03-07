"""
自动化训练管道使用示例

演示如何使用MLPipelineController运行完整的8阶段ML训练管道
"""

import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 导入管道模块
from src.pipeline import (
    MLPipelineController,
    PipelineConfig,
    load_pipeline_config,
    create_default_config
)
from src.pipeline.stages import (
    DataPreparationStage,
    FeatureEngineeringStage,
    ModelTrainingStage,
    ModelEvaluationStage,
    ModelValidationStage,
    CanaryDeploymentStage,
    FullDeploymentStage,
    MonitoringStage
)
from src.monitoring import get_alert_manager, get_metrics_collector
from src.rollback import get_rollback_manager


def run_pipeline_with_config_file(config_path: str):
    """使用配置文件运行管道"""
    print(f"加载配置文件: {config_path}")
    
    # 加载配置
    config = load_pipeline_config(config_path)
    print(f"管道名称: {config.name}")
    print(f"管道版本: {config.version}")
    print(f"阶段数量: {len(config.stages)}")
    
    # 创建控制器
    controller = MLPipelineController(config)
    
    # 注册所有阶段
    _register_stages(controller, config)
    
    # 执行管道
    print("\n开始执行管道...")
    result = controller.execute()
    
    # 输出结果
    print(f"\n管道执行结果:")
    print(f"  状态: {result.status.name}")
    print(f"  耗时: {result.duration_seconds:.2f}秒")
    print(f"  成功: {result.is_success}")
    
    if result.error:
        print(f"  错误: {result.error}")
    
    print(f"\n执行摘要:")
    print(f"  完成阶段: {result.summary.get('stages_completed', 0)}/{result.summary.get('stages_total', 0)}")
    print(f"  总耗时: {result.summary.get('total_duration_seconds', 0):.2f}秒")
    
    return result


def run_pipeline_programmatic():
    """以编程方式运行管道"""
    print("以编程方式创建和运行管道")
    
    # 创建默认配置
    config = create_default_config()
    
    # 自定义配置
    config.name = "my_quant_pipeline"
    config.global_config = {
        "project_name": "我的量化策略",
        "environment": "development"
    }
    
    # 创建控制器
    controller = MLPipelineController(config)
    
    # 注册所有阶段
    _register_stages(controller, config)
    
    # 执行管道
    print("\n开始执行管道...")
    result = controller.execute()
    
    print(f"\n管道执行完成: {result.status.name}")
    
    return result


def _register_stages(controller, config):
    """注册所有阶段到控制器"""
    stage_mapping = {
        "data_preparation": DataPreparationStage,
        "feature_engineering": FeatureEngineeringStage,
        "model_training": ModelTrainingStage,
        "model_evaluation": ModelEvaluationStage,
        "model_validation": ModelValidationStage,
        "canary_deployment": CanaryDeploymentStage,
        "full_deployment": FullDeploymentStage,
        "monitoring": MonitoringStage
    }
    
    for stage_config in config.stages:
        stage_name = stage_config.name
        stage_class = stage_mapping.get(stage_name)
        
        if stage_class:
            stage = stage_class(config=stage_config)
            controller.register_stage(stage)
            print(f"  注册阶段: {stage_name}")
        else:
            print(f"  警告: 未知的阶段类型 {stage_name}")


def setup_monitoring_and_rollback():
    """设置监控和回滚"""
    print("\n设置监控和回滚系统")
    
    # 获取监控组件
    metrics_collector = get_metrics_collector()
    alert_manager = get_alert_manager()
    rollback_manager = get_rollback_manager()
    
    # 注册告警处理器
    def alert_handler(alert):
        print(f"  收到告警: [{alert.severity.value}] {alert.message}")
        
        # 严重告警触发回滚
        if alert.severity.value == "critical":
            print("  严重告警，触发自动回滚")
            # rollback_manager.evaluate_metric("deployment_1", alert.metric, alert.value)
    
    alert_manager.register_handler(alert_handler)
    
    # 注册回滚处理器
    def rollback_handler(deployment_id: str, target_version: str) -> bool:
        print(f"  执行回滚: {deployment_id} -> {target_version}")
        return True
    
    rollback_manager.register_deployment_handler("deployment_1", rollback_handler)
    
    print("  监控和回滚系统设置完成")


def demonstrate_pipeline_features():
    """演示管道功能"""
    print("\n演示管道功能")
    
    # 1. 创建配置
    config = create_default_config()
    
    # 2. 验证配置
    errors = config.validate()
    if errors:
        print(f"配置错误: {errors}")
        return
    
    print("  配置验证通过")
    
    # 3. 保存配置
    config_path = "configs/my_pipeline_config.yaml"
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config_path)
    print(f"  配置已保存到: {config_path}")
    
    # 4. 显示阶段信息
    print(f"\n  管道阶段:")
    for i, stage in enumerate(config.stages, 1):
        deps = f" (依赖: {', '.join(stage.dependencies)})" if stage.dependencies else ""
        print(f"    {i}. {stage.name}{deps}")
    
    # 5. 显示回滚触发器
    print(f"\n  回滚触发器:")
    for trigger in config.rollback.triggers:
        print(f"    - {trigger.metric} {trigger.operator} {trigger.threshold}")
    
    # 6. 显示告警阈值
    print(f"\n  告警阈值:")
    for metric, threshold in config.monitoring.alert_thresholds.items():
        print(f"    - {metric}: {threshold}")


def main():
    """主函数"""
    print("=" * 60)
    print("自动化训练管道使用示例")
    print("=" * 60)
    
    # 演示1: 功能演示
    demonstrate_pipeline_features()
    
    # 演示2: 设置监控和回滚
    setup_monitoring_and_rollback()
    
    # 演示3: 以编程方式运行
    print("\n" + "=" * 60)
    print("演示: 以编程方式运行管道")
    print("=" * 60)
    result = run_pipeline_programmatic()
    
    # 演示4: 使用配置文件运行
    print("\n" + "=" * 60)
    print("演示: 使用配置文件运行管道")
    print("=" * 60)
    config_path = "configs/pipeline_config.yaml"
    if Path(config_path).exists():
        result = run_pipeline_with_config_file(config_path)
    else:
        print(f"配置文件不存在: {config_path}")
        print("请先运行演示3生成配置文件")
    
    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
