#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证层级导入状态脚本

快速验证各层级核心模块的导入状态
"""

import sys
import os
from pathlib import Path

# 设置路径
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"

# 添加项目根目录和src目录到路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def test_layer_import(layer_name, module_path, class_name=None):
    """
    测试层级模块导入

    Args:
        layer_name: 层级名称
        module_path: 模块路径
        class_name: 要测试的类名

    Returns:
        dict: 测试结果
    """
    try:
        # 直接使用import语句而不是__import__
        if class_name:
            if module_path == 'src.core.core_services' and class_name == 'ServiceRegistry':
                from src.core.core_services import ServiceRegistry as cls
            elif module_path == 'src.data.adapters' and class_name == 'AdapterComponent':
                from src.data.adapters.adapter_components import AdapterComponent as cls
            elif module_path == 'src.features.core' and class_name == 'FeatureEngineer':
                from src.features.core.feature_engineer import FeatureEngineer as cls
            elif module_path == 'src.ml.core' and class_name == 'MLCore':
                from src.ml.core.ml_core import MLCore as cls
            elif module_path == 'src.monitoring.monitoring_system' and class_name == 'MonitoringSystem':
                from src.monitoring.monitoring_system import MonitoringSystem as cls
            elif module_path == 'src.strategy.strategies' and class_name == 'BaseStrategy':
                from src.strategy.strategies.base_strategy import BaseStrategy as cls
            else:
                return {
                    'status': 'FAILED',
                    'layer': layer_name,
                    'module': module_path,
                    'class': class_name,
                    'error': f'Unsupported class: {class_name}'
                }

            # 尝试实例化
            try:
                if class_name in ['MonitoringSystem', 'MLCore']:
                    instance = cls()  # 这些类可以实例化
                    instance_type = str(type(instance))
                else:
                    instance_type = f'Class: {cls}'
            except Exception as inst_e:
                instance_type = f'Class (cannot instantiate: {inst_e})'

            return {
                'status': 'SUCCESS',
                'layer': layer_name,
                'module': module_path,
                'class': class_name,
                'instance': instance_type
            }
        else:
            # 只是测试模块导入
            __import__(module_path)
            return {
                'status': 'SUCCESS',
                'layer': layer_name,
                'module': module_path,
                'class': None
            }
    except Exception as e:
        return {
            'status': 'FAILED',
            'layer': layer_name,
            'module': module_path,
            'class': class_name,
            'error': str(e)
        }

def main():
    """主函数：验证各层级导入状态"""

    print("🔍 开始验证层级导入状态...")

    # 定义要验证的层级
    layers_to_validate = [
        ('核心服务层', 'src.core.core_services', 'ServiceRegistry'),
        ('数据管理层', 'src.data.adapters', 'AdapterComponent'),
        ('特征分析层', 'src.features.core', 'FeatureEngineer'),
        ('机器学习层', 'src.ml.core', 'MLCore'),
        ('监控层', 'src.monitoring.monitoring_system', 'MonitoringSystem'),
        ('策略服务层', 'src.strategy.strategies', 'BaseStrategy'),
    ]

    results = []
    successful_imports = 0

    for layer_name, module_path, class_name in layers_to_validate:
        result = test_layer_import(layer_name, module_path, class_name)
        results.append(result)

        status_icon = "✅" if result['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {layer_name}: {result['status']}")

        if result['status'] == 'SUCCESS':
            successful_imports += 1
            if 'instance' in result:
                print(f"   实例化: {result['instance']}")
        else:
            print(f"   错误: {result['error']}")

    # 生成总结报告
    summary = {
        'total_layers': len(layers_to_validate),
        'successful_imports': successful_imports,
        'failed_imports': len(layers_to_validate) - successful_imports,
        'success_rate': f"{successful_imports}/{len(layers_to_validate)} ({successful_imports/len(layers_to_validate)*100:.1f}%)",
        'overall_status': 'SUCCESS' if successful_imports == len(layers_to_validate) else 'PARTIAL_SUCCESS'
    }

    print("\n📊 导入验证总结:")
    print(f"   总层级数: {summary['total_layers']}")
    print(f"   成功导入: {summary['successful_imports']}")
    print(f"   失败导入: {summary['failed_imports']}")
    print(f"   成功率: {summary['success_rate']}")
    print(f"   总体状态: {summary['overall_status']}")

    # 保存详细结果
    import json
    from datetime import datetime

    output_dir = project_root / "test_logs"
    output_dir.mkdir(exist_ok=True)

    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'detailed_results': results
    }

    report_file = output_dir / "layer_import_validation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"   详细报告: {report_file}")

    return summary

if __name__ == "__main__":
    main()
