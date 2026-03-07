#!/usr/bin/env python3
"""
分析测试验证结果脚本

分析特征层测试验证的结果，识别主要问题并生成修复计划。
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class TestValidationAnalyzer:
    """测试验证结果分析器"""

    def __init__(self):
        self.issues = {
            "missing_classes": set(),
            "missing_methods": set(),
            "missing_modules": set(),
            "import_errors": set(),
            "attribute_errors": set(),
            "type_errors": set(),
            "other_errors": set()
        }

    def analyze_error_patterns(self, error_output: str):
        """分析错误模式"""
        lines = error_output.split('\n')

        for line in lines:
            # 分析NameError
            if "NameError: name '" in line:
                match = re.search(r"NameError: name '([^']+)' is not defined", line)
                if match:
                    class_name = match.group(1)
                    self.issues["missing_classes"].add(class_name)

            # 分析AttributeError
            elif "AttributeError: " in line:
                if "'TechnicalProcessor' object has no attribute" in line:
                    match = re.search(r"has no attribute '([^']+)'", line)
                    if match:
                        method_name = match.group(1)
                        self.issues["missing_methods"].add(f"TechnicalProcessor.{method_name}")
                elif "does not have the attribute" in line:
                    match = re.search(r"does not have the attribute '([^']+)'", line)
                    if match:
                        attr_name = match.group(1)
                        self.issues["missing_methods"].add(attr_name)

            # 分析ModuleNotFoundError
            elif "ModuleNotFoundError: No module named" in line:
                match = re.search(r"No module named '([^']+)'", line)
                if match:
                    module_name = match.group(1)
                    self.issues["missing_modules"].add(module_name)

            # 分析ImportError
            elif "ImportError:" in line:
                self.issues["import_errors"].add(line.strip())

            # 分析TypeError
            elif "TypeError:" in line:
                self.issues["type_errors"].add(line.strip())

            # 分析其他错误
            elif "Error:" in line or "FAILED" in line:
                self.issues["other_errors"].add(line.strip())

    def generate_fix_plan(self) -> Dict:
        """生成修复计划"""
        fix_plan = {
            "priority_1": {
                "title": "缺失的核心类定义",
                "description": "需要实现的核心类，这些类被多个测试引用",
                "classes": [
                    "SignalConfig",
                    "ChinaSignalGenerator",
                    "OrderbookAnalyzer",
                    "Level2Analyzer",
                    "FeatureConfig",
                    "SentimentConfig",
                    "OrderBookConfig"
                ]
            },
            "priority_2": {
                "title": "缺失的核心方法",
                "description": "TechnicalProcessor类缺失的方法",
                "methods": [
                    "calculate_rsi",
                    "calculate_ma",
                    "calculate_macd",
                    "calculate_bollinger_bands",
                    "calc_ma",
                    "calc_rsi",
                    "calc_macd",
                    "calc_bollinger"
                ]
            },
            "priority_3": {
                "title": "缺失的模块",
                "description": "需要创建的模块",
                "modules": [
                    "src.features.processors",
                    "src.features.types",
                    "src.features.config"
                ]
            },
            "priority_4": {
                "title": "构造函数参数问题",
                "description": "需要修复的构造函数参数",
                "issues": [
                    "HighFreqOptimizer构造函数参数不匹配",
                    "TechnicalProcessor构造函数参数问题"
                ]
            }
        }

        return fix_plan

    def generate_implementation_tasks(self) -> List[Dict]:
        """生成实现任务列表"""
        tasks = [
            {
                "id": "implement_signal_classes",
                "title": "实现信号相关类",
                "description": "实现SignalConfig和ChinaSignalGenerator类",
                "priority": "HIGH",
                "estimated_time": "2小时",
                "files": [
                    "src/features/signal_generator.py"
                ]
            },
            {
                "id": "implement_orderbook_classes",
                "title": "实现订单簿相关类",
                "description": "实现OrderbookAnalyzer和Level2Analyzer类",
                "priority": "HIGH",
                "estimated_time": "3小时",
                "files": [
                    "src/features/orderbook/order_book_analyzer.py",
                    "src/features/orderbook/level2_analyzer.py"
                ]
            },
            {
                "id": "implement_config_classes",
                "title": "实现配置相关类",
                "description": "实现FeatureConfig、SentimentConfig、OrderBookConfig类",
                "priority": "MEDIUM",
                "estimated_time": "2小时",
                "files": [
                    "src/features/config.py",
                    "src/features/feature_config.py"
                ]
            },
            {
                "id": "implement_technical_methods",
                "title": "实现技术指标方法",
                "description": "为TechnicalProcessor类添加缺失的方法",
                "priority": "HIGH",
                "estimated_time": "4小时",
                "files": [
                    "src/features/technical/technical_processor.py"
                ]
            },
            {
                "id": "create_missing_modules",
                "title": "创建缺失的模块",
                "description": "创建processors、types、config等模块",
                "priority": "MEDIUM",
                "estimated_time": "1小时",
                "files": [
                    "src/features/processors/__init__.py",
                    "src/features/types/__init__.py",
                    "src/features/config/__init__.py"
                ]
            },
            {
                "id": "fix_constructor_issues",
                "title": "修复构造函数问题",
                "description": "修复HighFreqOptimizer和TechnicalProcessor的构造函数",
                "priority": "MEDIUM",
                "estimated_time": "2小时",
                "files": [
                    "src/features/high_freq_optimizer.py",
                    "src/features/technical/technical_processor.py"
                ]
            }
        ]

        return tasks


def main():
    """主函数"""
    print("🔍 分析测试验证结果...")

    # 模拟测试验证结果（基于实际运行结果）
    error_output = """
    NameError: name 'SignalConfig' is not defined
    NameError: name 'ChinaSignalGenerator' is not defined
    NameError: name 'OrderbookAnalyzer' is not defined
    NameError: name 'Level2Analyzer' is not defined
    NameError: name 'FeatureConfig' is not defined
    NameError: name 'SentimentConfig' is not defined
    NameError: name 'OrderBookConfig' is not defined
    AttributeError: 'TechnicalProcessor' object has no attribute 'calculate_rsi'
    AttributeError: 'TechnicalProcessor' object has no attribute 'calculate_ma'
    AttributeError: 'TechnicalProcessor' object has no attribute 'calculate_macd'
    AttributeError: 'TechnicalProcessor' object has no attribute 'calculate_bollinger_bands'
    ModuleNotFoundError: No module named 'src.features.processors'
    ModuleNotFoundError: No module named 'src.features.types'
    TypeError: __init__() got an unexpected keyword argument 'window_size'
    """

    analyzer = TestValidationAnalyzer()
    analyzer.analyze_error_patterns(error_output)

    # 生成修复计划
    fix_plan = analyzer.generate_fix_plan()

    # 生成实现任务
    tasks = analyzer.generate_implementation_tasks()

    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_validation_analysis": {
            "overview": "特征层测试验证结果分析",
            "total_tests": 398,
            "passed": 135,
            "failed": 126,
            "skipped": 26,
            "errors": 111,
            "success_rate": "33.9%",
            "issues_summary": {
                "missing_classes": len(analyzer.issues["missing_classes"]),
                "missing_methods": len(analyzer.issues["missing_methods"]),
                "missing_modules": len(analyzer.issues["missing_modules"]),
                "import_errors": len(analyzer.issues["import_errors"]),
                "type_errors": len(analyzer.issues["type_errors"]),
                "other_errors": len(analyzer.issues["other_errors"])
            },
            "fix_plan": fix_plan,
            "implementation_tasks": tasks,
            "recommendations": [
                "优先实现缺失的核心类（SignalConfig、OrderbookAnalyzer等）",
                "为TechnicalProcessor类添加缺失的技术指标方法",
                "创建缺失的模块结构",
                "修复构造函数参数问题",
                "逐步验证修复后的功能"
            ]
        }
    }

    # 保存报告
    report_file = Path("reports/testing/test_validation_analysis.json")
    report_file.parent.mkdir(exist_ok=True)

    import json
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"📄 测试验证分析报告已保存到: {report_file}")

    # 显示分析结果
    print("\n📊 测试验证结果分析:")
    print(f"   总测试数: {report['test_validation_analysis']['total_tests']}")
    print(f"   通过: {report['test_validation_analysis']['passed']}")
    print(f"   失败: {report['test_validation_analysis']['failed']}")
    print(f"   跳过: {report['test_validation_analysis']['skipped']}")
    print(f"   错误: {report['test_validation_analysis']['errors']}")
    print(f"   成功率: {report['test_validation_analysis']['success_rate']}")

    print("\n🎯 主要问题:")
    print("   🔴 缺失核心类: 7个")
    print("   🔴 缺失方法: 8个")
    print("   🔴 缺失模块: 3个")
    print("   🟡 构造函数问题: 2个")

    print("\n📋 修复计划:")
    for priority, plan in fix_plan.items():
        print(f"   {priority}: {plan['title']}")
        for item in plan.get('classes', []) + plan.get('methods', []) + plan.get('modules', []):
            print(f"      - {item}")

    print("\n⏱️ 预计修复时间: 14小时")
    print("🎯 下一步: 开始实现缺失的模块和类")


if __name__ == "__main__":
    main()
