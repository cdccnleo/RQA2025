#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证重构后的代码功能

测试重构后的模块导入和基本功能。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_enhanced_data_integration_import():
    """测试增强数据集成模块导入"""
    print("=" * 60)
    print("测试1：增强数据集成模块导入")
    print("=" * 60)
    
    try:
        from src.data.integration.enhanced_data_integration import (
            EnhancedDataIntegration,
            IntegrationConfig,
            TaskPriority,
            create_enhanced_data_integration,
        )
        print("✅ 主模块导入成功")
        
        # 测试配置创建
        config = IntegrationConfig()
        print("✅ 配置创建成功")
        print(f"  - parallel_loading配置存在: {'max_workers' in config.parallel_loading}")
        print(f"  - cache_strategy配置存在: {'max_size' in config.cache_strategy}")
        print(f"  - quality_monitor配置存在: {'enable_alerting' in config.quality_monitor}")
        
        # 测试组件导入
        print(f"✅ TaskPriority枚举: {TaskPriority.HIGH}")
        print(f"✅ create_enhanced_data_integration函数: {callable(create_enhanced_data_integration)}")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_aligner():
    """测试数据对齐器"""
    print("\n" + "=" * 60)
    print("测试2：DataAligner 和 align_time_series 重构")
    print("=" * 60)
    
    try:
        from src.data.alignment.data_aligner import DataAligner, AlignmentMethod, FrequencyType
        import pandas as pd
        
        # 创建对齐器
        aligner = DataAligner()
        print("✅ DataAligner 实例化成功")
        
        # 创建测试数据
        df1 = pd.DataFrame(
            {'value': [1, 2, 3]},
            index=pd.date_range('2024-01-01', periods=3)
        )
        df2 = pd.DataFrame(
            {'value': [4, 5, 6]},
            index=pd.date_range('2024-01-02', periods=3)
        )
        
        # 测试对齐功能
        result = aligner.align_time_series(
            {'df1': df1, 'df2': df2},
            method=AlignmentMethod.OUTER
        )
        
        print("✅ align_time_series 方法执行成功")
        print(f"  - 输入数据框数: 2")
        print(f"  - 输出数据框数: {len(result)}")
        print(f"  - df1 形状: {result['df1'].shape}")
        print(f"  - df2 形状: {result['df2'].shape}")
        
        # 验证辅助方法存在
        helper_methods = [
            '_convert_enums_to_strings',
            '_ensure_datetime_index',
            '_determine_date_range',
            '_get_start_date_by_method',
            '_get_end_date_by_method',
            '_apply_fill_method'
        ]
        
        print("✅ 辅助方法检查:")
        for method in helper_methods:
            exists = hasattr(aligner, method)
            status = "✅" if exists else "❌"
            print(f"  {status} {method}: {exists}")
        
        return True
    except Exception as e:
        print(f"❌ DataAligner测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_structure():
    """测试模块结构"""
    print("\n" + "=" * 60)
    print("测试3：模块化结构验证")
    print("=" * 60)
    
    try:
        from src.data.integration.enhanced_data_integration_modules import (
            config,
            components,
            cache_utils,
            performance_utils,
            integration_manager,
        )
        
        modules = {
            'config': config,
            'components': components,
            'cache_utils': cache_utils,
            'performance_utils': performance_utils,
            'integration_manager': integration_manager,
        }
        
        print("✅ 所有模块导入成功:")
        for name, module in modules.items():
            print(f"  ✅ {name}: {module.__name__}")
        
        # 验证核心类和函数
        from src.data.integration.enhanced_data_integration_modules import (
            IntegrationConfig,
            EnhancedDataIntegration,
            DynamicThreadPoolManager,
            check_cache_for_symbols,
            shutdown,
        )
        
        print("\n✅ 核心类和函数:")
        print(f"  ✅ IntegrationConfig: {IntegrationConfig}")
        print(f"  ✅ EnhancedDataIntegration: {EnhancedDataIntegration}")
        print(f"  ✅ DynamicThreadPoolManager: {DynamicThreadPoolManager}")
        print(f"  ✅ check_cache_for_symbols: {callable(check_cache_for_symbols)}")
        print(f"  ✅ shutdown: {callable(shutdown)}")
        
        return True
    except Exception as e:
        print(f"❌ 模块结构测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n🚀 开始验证重构后的代码...")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("增强数据集成导入", test_enhanced_data_integration_import()))
    results.append(("数据对齐器功能", test_data_aligner()))
    results.append(("模块化结构", test_module_structure()))
    
    # 统计结果
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    print("=" * 60)
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！重构验证成功！")
        return 0
    else:
        print(f"⚠️ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit(main())

