#!/usr/bin/env python3
"""
验证优化器重构

检查重构后的optimizer是否：
1. 正确导入所有组件
2. 保持向后兼容
3. 基础功能正常
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def validate_imports():
    """验证导入"""
    print("=" * 80)
    print("1. 验证导入...")
    print("=" * 80)
    
    try:
        from src.core.business.optimizer.optimizer_refactored import (
            IntelligentBusinessProcessOptimizer,
            ProcessStage,
            ProcessContext,
            OptimizerConfig
        )
        print("✅ 主类导入成功")
        
        from src.core.business.optimizer.components import (
            PerformanceAnalyzer,
            DecisionEngine,
            ProcessExecutor,
            RecommendationGenerator,
            ProcessMonitor
        )
        print("✅ 组件导入成功")
        
        from src.core.business.optimizer.configs import OptimizerConfig
        print("✅ 配置导入成功")
        
        from src.core.business.optimizer.models import ProcessContext
        print("✅ 模型导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_initialization():
    """验证初始化"""
    print("\n" + "=" * 80)
    print("2. 验证初始化...")
    print("=" * 80)
    
    try:
        from src.core.business.optimizer.optimizer_refactored import (
            IntelligentBusinessProcessOptimizer
        )
        
        # 测试1: 无配置初始化
        print("\n测试: 无配置初始化...")
        optimizer1 = IntelligentBusinessProcessOptimizer()
        print(f"  ✅ 创建成功")
        print(f"  - 组件数量: 5")
        print(f"  - 配置对象: {optimizer1.config.__class__.__name__}")
        
        # 测试2: 字典配置初始化（向后兼容）
        print("\n测试: 字典配置初始化...")
        config_dict = {
            'max_concurrent_processes': 5,
            'decision_timeout': 20,
            'risk_threshold': 0.8
        }
        optimizer2 = IntelligentBusinessProcessOptimizer(config=config_dict)
        print(f"  ✅ 创建成功")
        print(f"  - max_concurrent: {optimizer2.max_concurrent_processes}")
        print(f"  - decision_timeout: {optimizer2.decision_timeout}")
        print(f"  - risk_threshold: {optimizer2.risk_threshold}")
        
        # 测试3: OptimizerConfig对象初始化
        print("\n测试: OptimizerConfig对象初始化...")
        from src.core.business.optimizer.configs import OptimizerConfig
        config_obj = OptimizerConfig.create_high_performance()
        optimizer3 = IntelligentBusinessProcessOptimizer(config=config_obj)
        print(f"  ✅ 创建成功")
        print(f"  - 策略: {config_obj.optimization_strategy.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_components():
    """验证组件"""
    print("\n" + "=" * 80)
    print("3. 验证组件集成...")
    print("=" * 80)
    
    try:
        from src.core.business.optimizer.optimizer_refactored import (
            IntelligentBusinessProcessOptimizer
        )
        
        optimizer = IntelligentBusinessProcessOptimizer()
        
        # 检查5个组件
        components = {
            'analyzer': optimizer.analyzer,
            'decision_engine': optimizer.decision_engine,
            'executor': optimizer.executor,
            'recommender': optimizer.recommender,
            'monitor': optimizer.monitor
        }
        
        print(f"\n组件检查:")
        for name, component in components.items():
            if component is None:
                print(f"  ❌ {name}: 未初始化")
                return False
            else:
                status = component.get_status()
                print(f"  ✅ {name}: {component.__class__.__name__}")
                print(f"      状态: {list(status.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 组件验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_backward_compatibility():
    """验证向后兼容性"""
    print("\n" + "=" * 80)
    print("4. 验证向后兼容性...")
    print("=" * 80)
    
    try:
        from src.core.business.optimizer.optimizer_refactored import (
            IntelligentBusinessProcessOptimizer
        )
        
        optimizer = IntelligentBusinessProcessOptimizer()
        
        # 检查旧属性
        old_attrs = [
            'active_processes',
            'completed_processes',
            'optimization_recommendations',
            'process_metrics',
            'max_concurrent_processes',
            'decision_timeout',
            'risk_threshold'
        ]
        
        print(f"\n旧属性检查:")
        for attr in old_attrs:
            if hasattr(optimizer, attr):
                value = getattr(optimizer, attr)
                print(f"  ✅ {attr}: {type(value).__name__}")
            else:
                print(f"  ❌ {attr}: 不存在")
                return False
        
        # 检查旧方法
        print(f"\n旧方法检查:")
        old_methods = [
            'start_optimization_engine',
            'optimize_trading_process',
            'get_optimization_status'
        ]
        
        for method in old_methods:
            if hasattr(optimizer, method):
                print(f"  ✅ {method}()")
            else:
                print(f"  ❌ {method}(): 不存在")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 向后兼容性验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_status():
    """验证状态方法"""
    print("\n" + "=" * 80)
    print("5. 验证get_optimization_status()...")
    print("=" * 80)
    
    try:
        from src.core.business.optimizer.optimizer_refactored import (
            IntelligentBusinessProcessOptimizer
        )
        
        optimizer = IntelligentBusinessProcessOptimizer()
        status = optimizer.get_optimization_status()
        
        print(f"\n状态信息:")
        print(f"  ✅ 返回类型: {type(status).__name__}")
        print(f"  ✅ 包含字段: {list(status.keys())}")
        
        # 检查必要字段
        required_fields = [
            'active_processes',
            'completed_processes',
            'components',
            'config'
        ]
        
        print(f"\n必要字段检查:")
        for field in required_fields:
            if field in status:
                print(f"  ✅ {field}")
            else:
                print(f"  ❌ {field}: 缺失")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 状态验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" " * 20 + "优化器重构验证")
    print("=" * 80)
    
    results = []
    
    # 执行验证
    results.append(("导入验证", validate_imports()))
    results.append(("初始化验证", validate_initialization()))
    results.append(("组件验证", validate_components()))
    results.append(("向后兼容性验证", validate_backward_compatibility()))
    results.append(("状态方法验证", validate_status()))
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("验证结果汇总:")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:.<50} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 80)
    total = passed + failed
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"总计: {passed}/{total} 通过 ({success_rate:.1f}%)")
    print("=" * 80)
    
    if failed == 0:
        print("\n🎉 所有验证通过！重构成功！")
        return 0
    else:
        print(f"\n⚠️ 有{failed}项验证失败，需要修复")
        return 1


if __name__ == '__main__':
    sys.exit(main())

