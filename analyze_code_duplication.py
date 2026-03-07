#!/usr/bin/env python3
"""
代码重复和接口标准化分析工具
用于Phase 7: 代码重复消除和接口标准化
"""

import json
from collections import defaultdict
from typing import Dict


def analyze_code_duplication():
    """分析代码重复和接口标准化问题"""

    print("🔍 Phase 7: 代码重复和接口标准化分析")
    print("=" * 60)

    # 加载分析结果
    with open('phase7_pre_analysis.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 分析重复模式
    analyze_duplication_patterns(data)

    # 分析接口标准化问题
    analyze_interface_standardization(data)

    print("\n🎯 Phase 7 重构策略:")
    print("1. 创建共享的异常处理基类")
    print("2. 统一配置验证接口")
    print("3. 标准化日志记录模式")
    print("4. 创建通用的数据验证器")
    print("5. 统一资源管理接口")
    print("6. 标准化错误处理流程")


def analyze_duplication_patterns(data: Dict) -> None:
    """分析重复模式"""

    print("\n📋 代码重复模式分析:")

    # 统计重复的导入模式
    import_patterns = defaultdict(int)
    for opp in data['opportunities']:
        if '重复' in opp['title'].lower() or 'duplicate' in opp['title'].lower():
            import_patterns[opp['title']] += 1

    if import_patterns:
        print("🔄 发现的重复模式:")
        for pattern, count in sorted(import_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {pattern}: {count}处")
    else:
        print("✅ 未发现明显的代码重复模式")

    # 分析常见重复结构
    analyze_common_patterns(data)


def analyze_common_patterns(data: Dict) -> None:
    """分析常见重复结构"""

    print("\n🔧 常见重复结构分析:")

    # 分析配置验证模式
    config_validation_patterns = []
    logging_patterns = []
    error_handling_patterns = []

    for opp in data['opportunities']:
        title = opp['title'].lower()
        desc = opp.get('description', '').lower()

        if 'config' in title and 'validat' in desc:
            config_validation_patterns.append(opp)
        elif 'log' in title:
            logging_patterns.append(opp)
        elif 'error' in title or 'exception' in title:
            error_handling_patterns.append(opp)

    print(f"• 配置验证模式: {len(config_validation_patterns)}处")
    print(f"• 日志记录模式: {len(logging_patterns)}处")
    print(f"• 错误处理模式: {len(error_handling_patterns)}处")

    # 分析接口不一致问题
    analyze_interface_inconsistencies(data)


def analyze_interface_standardization(data: Dict) -> None:
    """分析接口标准化问题"""

    print("\n🔗 接口标准化分析:")

    # 分析方法命名不一致
    method_patterns = defaultdict(list)
    for opp in data['opportunities']:
        if 'interface' in opp['title'].lower() or '一致性' in opp['title']:
            file_path = opp['file_path']
            method_patterns[opp['title']].append(file_path)

    if method_patterns:
        print("📝 接口不一致问题:")
        for pattern, files in method_patterns.items():
            print(f"  • {pattern}")
            print(f"    涉及文件: {', '.join(set(files))}")
    else:
        print("✅ 接口相对一致")


def analyze_interface_inconsistencies(data: Dict) -> None:
    """分析接口不一致性"""

    print("\n⚠️  潜在的接口标准化机会:")

    opportunities = []

    # 检查配置接口
    config_interfaces = [
        "validate_config", "check_config", "verify_config", "config_validation"
    ]

    # 检查日志接口
    logging_interfaces = [
        "log_info", "log_error", "log_warning", "logger.info", "logging.info"
    ]

    # 检查错误处理接口
    error_interfaces = [
        "handle_error", "process_error", "raise_error", "error_handler"
    ]

    # 检查资源管理接口
    resource_interfaces = [
        "get_resource", "acquire_resource", "release_resource", "resource_manager"
    ]

    interface_groups = {
        "配置验证接口": config_interfaces,
        "日志记录接口": logging_interfaces,
        "错误处理接口": error_interfaces,
        "资源管理接口": resource_interfaces
    }

    for category, interfaces in interface_groups.items():
        print(f"• {category}: {len(interfaces)}个相关接口")

    print("\n🎯 标准化建议:")
    print("1. 创建统一的配置验证接口")
    print("2. 标准化日志记录方法")
    print("3. 统一错误处理模式")
    print("4. 创建通用的资源管理接口")
    print("5. 建立共享的工具类库")


if __name__ == "__main__":
    analyze_code_duplication()
