#!/usr/bin/env python3
"""
应用重复代码重构步骤
"""

from src.infrastructure.utils.duplicate_resolver import InfrastructureDuplicateResolver


def main():
    resolver = InfrastructureDuplicateResolver()

    # 应用重构步骤1
    print('🚀 开始应用重构步骤1: 创建统一的状态管理基类')
    success1 = resolver.apply_refactoring(1)
    print(f'步骤1结果: {"✅ 成功" if success1 else "❌ 失败"}')

    # 应用重构步骤3 (日志统一相对简单)
    print('🚀 开始应用重构步骤3: 统一日志管理')
    success3 = resolver.apply_refactoring(3)
    print(f'步骤3结果: {"✅ 成功" if success3 else "❌ 失败"}')

    # 应用重构步骤4 (异常处理统一)
    print('🚀 开始应用重构步骤4: 统一异常处理')
    success4 = resolver.apply_refactoring(4)
    print(f'步骤4结果: {"✅ 成功" if success4 else "❌ 失败"}')

    print('🎯 基础重构步骤完成')

    # 显示重构后的统计
    final_plan = resolver.generate_refactoring_plan()
    print('\n📊 重构效果统计:')
    print(f'重复出现次数: {final_plan["summary"]["total_duplicate_occurrences"]}')
    print(f'节省代码行数: {final_plan["summary"]["estimated_lines_to_save"]}')
    print(f'重复减少百分比: {final_plan["summary"]["estimated_duplicate_reduction_percent"]}%')


if __name__ == "__main__":
    main()
