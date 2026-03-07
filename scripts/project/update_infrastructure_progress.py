#!/usr/bin/env python3
"""
更新基础设施层任务进度
"""

from optimization_task_tracker import OptimizationTaskTracker, TaskStatus
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """更新基础设施层任务进度"""
    tracker = OptimizationTaskTracker()

    # 更新基础设施层监控指标任务进度
    infrastructure_task_id = "infrastructure_short_1"
    if infrastructure_task_id in tracker.tasks:
        # 标记为进行中，进度20%（已创建错误处理模块，修复了导入错误）
        tracker.update_task_progress(infrastructure_task_id, 20.0, TaskStatus.IN_PROGRESS)
        print(f"✅ 已更新基础设施层监控指标任务进度: 20%")
        print(f"📋 已完成工作:")
        print(f"   - 创建了 src/infrastructure/error_handler.py")
        print(f"   - 修复了测试文件中的导入错误")
        print(f"   - 修复了 trading 模块的 StrategyOptimizer 导入问题")
        print(f"📝 下一步工作:")
        print(f"   - 运行基础设施层测试")
        print(f"   - 完善监控指标")
        print(f"   - 优化性能")
    else:
        print(f"❌ 任务 {infrastructure_task_id} 不存在")

    # 显示更新后的状态
    print("\n" + "="*60)
    tracker.print_status()


if __name__ == "__main__":
    main()
