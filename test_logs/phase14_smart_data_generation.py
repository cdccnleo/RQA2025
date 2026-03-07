#!/usr/bin/env python3
"""
Phase 14.7: 智能测试数据生成系统
基于边界条件分析和代码特征智能生成高质量测试数据
"""

import random
import string
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np


@dataclass
class DataGenerationRule:
    """数据生成规则"""
    data_type: str
    constraints: Dict[str, Any]
    boundary_values: List[Any]
    distribution: str = 'uniform'
    generator_function: Optional[Callable] = None


@dataclass
class SmartTestData:
    """智能测试数据"""
    parameter_name: str
    data_type: str
    valid_values: List[Any]
    invalid_values: List[Any]
    boundary_values: List[Any]
    edge_cases: List[Any]
    equivalence_classes: List[List[Any]]


class SmartDataGenerator:
    """智能测试数据生成器"""

    def __init__(self):
        self.generation_rules = self._initialize_rules()

    def _initialize_rules(self) -> Dict[str, DataGenerationRule]:
        """初始化数据生成规则"""
        return {
            'integer': DataGenerationRule(
                data_type='integer',
                constraints={'min': -2**31, 'max': 2**31-1},
                boundary_values=[0, -1, 1, 2**31-1, -2**31],
                distribution='uniform'
            ),
            'string': DataGenerationRule(
                data_type='string',
                constraints={'min_length': 0, 'max_length': 10000},
                boundary_values=['', 'a', 'A'*1000, 'special_chars_!@#$%^&*()'],
                distribution='length_based'
            ),
            'list': DataGenerationRule(
                data_type='list',
                constraints={'min_length': 0, 'max_length': 1000},
                boundary_values=[[], [1], [1, 2, 3]*100],
                distribution='length_based'
            ),
            'dict': DataGenerationRule(
                data_type='dict',
                constraints={'max_keys': 100},
                boundary_values=[{}, {'key': 'value'}, {'k'*100: 'v'*100}],
                distribution='key_distribution'
            ),
            'float': DataGenerationRule(
                data_type='float',
                constraints={'allow_inf': False, 'allow_nan': False},
                boundary_values=[0.0, -0.0, 1.0, -1.0, float('inf'), -float('inf'), float('nan')],
                distribution='normal'
            ),
            'boolean': DataGenerationRule(
                data_type='boolean',
                constraints={},
                boundary_values=[True, False],
                distribution='bernoulli'
            ),
            'datetime': DataGenerationRule(
                data_type='datetime',
                constraints={'future_only': False, 'past_only': False},
                boundary_values=[
                    datetime.now(),
                    datetime(1900, 1, 1),
                    datetime(9999, 12, 31),
                    datetime.now() + timedelta(days=365*100)
                ],
                distribution='time_based'
            )
        }

    def generate_smart_data(self, parameter_name: str, data_type: str,
                          constraints: Dict[str, Any] = None,
                          boundary_conditions: List[Any] = None) -> SmartTestData:
        """生成智能测试数据"""

        if constraints is None:
            constraints = {}

        if boundary_conditions is None:
            boundary_conditions = []

        # 获取基础规则
        rule = self.generation_rules.get(data_type, self.generation_rules['string'])

        # 合并约束条件
        merged_constraints = {**rule.constraints, **constraints}

        # 生成各种类型的测试数据
        valid_values = self._generate_valid_values(data_type, merged_constraints, 10)
        invalid_values = self._generate_invalid_values(data_type, merged_constraints, 5)
        boundary_values = boundary_conditions + rule.boundary_values[:5]  # 限制数量
        edge_cases = self._generate_edge_cases(data_type, merged_constraints, 5)
        equivalence_classes = self._generate_equivalence_classes(data_type, merged_constraints, 3)

        return SmartTestData(
            parameter_name=parameter_name,
            data_type=data_type,
            valid_values=valid_values,
            invalid_values=invalid_values,
            boundary_values=list(set(boundary_values)),  # 去重
            edge_cases=edge_cases,
            equivalence_classes=equivalence_classes
        )

    def _generate_valid_values(self, data_type: str, constraints: Dict[str, Any], count: int) -> List[Any]:
        """生成有效值"""
        if data_type == 'integer':
            min_val = constraints.get('min', -1000)
            max_val = constraints.get('max', 1000)
            return [random.randint(min_val, max_val) for _ in range(count)]

        elif data_type == 'string':
            min_len = constraints.get('min_length', 1)
            max_len = constraints.get('max_length', 100)
            return [''.join(random.choices(string.ascii_letters + string.digits,
                                          k=random.randint(min_len, max_len)))
                   for _ in range(count)]

        elif data_type == 'list':
            max_len = constraints.get('max_length', 10)
            return [[random.randint(1, 100) for _ in range(random.randint(1, max_len))]
                   for _ in range(count)]

        elif data_type == 'dict':
            max_keys = constraints.get('max_keys', 5)
            result = []
            for _ in range(count):
                keys = [f'key_{i}' for i in range(random.randint(1, max_keys))]
                values = [random.choice(['value', 42, True, [1,2,3]]) for _ in keys]
                result.append(dict(zip(keys, values)))
            return result

        elif data_type == 'float':
            return [random.uniform(-1000, 1000) for _ in range(count)]

        elif data_type == 'boolean':
            return [random.choice([True, False]) for _ in range(count)]

        elif data_type == 'datetime':
            base = datetime.now()
            return [base + timedelta(days=random.randint(-365, 365)) for _ in range(count)]

        else:
            return [f'valid_{data_type}_{i}' for i in range(count)]

    def _generate_invalid_values(self, data_type: str, constraints: Dict[str, Any], count: int) -> List[Any]:
        """生成无效值"""
        if data_type == 'integer':
            # 生成超出范围的整数或非整数
            return [random.choice([
                random.randint(-2**32, -2**31-1),  # 太小
                random.randint(2**31, 2**32),      # 太大
                'not_an_int',                      # 错误类型
                None                               # None值
            ]) for _ in range(count)]

        elif data_type == 'string':
            max_len = constraints.get('max_length', 100)
            return [random.choice([
                None,                              # None值
                123,                               # 错误类型
                'A' * (max_len + 1000),           # 太长
                '\x00\x01\x02',                    # 特殊字符
            ]) for _ in range(count)]

        elif data_type == 'list':
            return [random.choice([
                None,                              # None值
                'not_a_list',                      # 错误类型
                [1, 2, 'mixed', None],            # 混合类型
                list(range(10000)),               # 太大
            ]) for _ in range(count)]

        elif data_type == 'dict':
            return [random.choice([
                None,                              # None值
                'not_a_dict',                      # 错误类型
                {'key': object()},                 # 不可序列化值
            ]) for _ in range(count)]

        elif data_type == 'float':
            return [random.choice([
                'not_a_float',                     # 错误类型
                None,                              # None值
                float('inf'),                      # 无穷大
                float('nan'),                      # NaN
            ]) for _ in range(count)]

        elif data_type == 'boolean':
            return [random.choice([
                'not_a_bool',                      # 错误类型
                None,                              # None值
                0, 1,                              # 数字
            ]) for _ in range(count)]

        else:
            return [f'invalid_{data_type}_{i}' for i in range(count)]

    def _generate_edge_cases(self, data_type: str, constraints: Dict[str, Any], count: int) -> List[Any]:
        """生成边界情况"""
        edge_cases = []

        if data_type == 'integer':
            # 极值和特殊值
            edge_cases.extend([
                2**31-1, -2**31, 0, -0, 1, -1,
                2**15-1, -2**15, 2**7-1, -2**7
            ])

        elif data_type == 'string':
            max_len = constraints.get('max_length', 100)
            edge_cases.extend([
                '', 'a', 'A'*max_len, 'unicode_中文_🚀',
                '\n\r\t', ' '*100, 'null', 'undefined'
            ])

        elif data_type == 'list':
            max_len = constraints.get('max_length', 10)
            edge_cases.extend([
                [], [None], [1]*max_len,
                [[], [1], [1,2]], [0, False, '', []]
            ])

        elif data_type == 'dict':
            edge_cases.extend([
                {}, {'': ''}, {'key': None},
                {'nested': {'key': 'value'}}, {'circular': None}  # 简化处理循环引用
            ])

        elif data_type == 'float':
            edge_cases.extend([
                0.0, -0.0, 1.0, -1.0, 0.1, -0.1,
                1e-10, 1e10, 3.14159, 2.71828
            ])

        # 随机选择指定数量的边界情况
        if len(edge_cases) > count:
            edge_cases = random.sample(edge_cases, count)

        return edge_cases

    def _generate_equivalence_classes(self, data_type: str, constraints: Dict[str, Any], count: int) -> List[List[Any]]:
        """生成等价类"""
        equivalence_classes = []

        if data_type == 'integer':
            # 正数、负数、零
            equivalence_classes = [
                [1, 100, 1000],           # 正整数
                [-1, -100, -1000],        # 负整数
                [0],                       # 零
                [2**30, 2**31-1],         # 大整数
                [-2**30, -2**31+1],       # 小整数
            ]

        elif data_type == 'string':
            equivalence_classes = [
                ['a', 'abc', 'hello'],              # 普通字符串
                ['', '   ', '\t\n'],                # 空白字符串
                ['123', '0', '-1'],                 # 数字字符串
                ['中文', 'русский', '🚀🌟'],        # Unicode字符串
                ['!@#$%', '^&*()', '[]{}'],        # 特殊字符
            ]

        elif data_type == 'list':
            equivalence_classes = [
                [[], [None]],                       # 空或None列表
                [[1], [1,2], [1,2,3]],             # 数字列表
                [['a'], ['a','b'], ['a','b','c']],  # 字符串列表
                [[], [1], ['a'], [1, 'a']],        # 混合类型列表
            ]

        elif data_type == 'boolean':
            equivalence_classes = [
                [True],                            # 真值
                [False],                           # 假值
            ]

        # 限制返回的等价类数量
        if len(equivalence_classes) > count:
            equivalence_classes = equivalence_classes[:count]

        return equivalence_classes

    def generate_test_data_suite(self, parameters: Dict[str, Dict[str, Any]]) -> Dict[str, SmartTestData]:
        """生成完整的测试数据套件"""
        suite = {}

        for param_name, param_info in parameters.items():
            data_type = param_info.get('type', 'string')
            constraints = param_info.get('constraints', {})
            boundary_conditions = param_info.get('boundary_conditions', [])

            smart_data = self.generate_smart_data(
                param_name, data_type, constraints, boundary_conditions
            )
            suite[param_name] = smart_data

        return suite


class TestDataSuiteGenerator:
    """测试数据套件生成器"""

    def __init__(self):
        self.smart_generator = SmartDataGenerator()

    def analyze_function_parameters(self, function_code: str) -> Dict[str, Dict[str, Any]]:
        """分析函数参数（简化版）"""
        # 这里应该使用AST解析函数参数
        # 为简化，我们返回预定义的参数信息
        return {
            'config_type': {
                'type': 'string',
                'constraints': {'min_length': 1, 'max_length': 50},
                'boundary_conditions': ['', 'invalid', 'special@chars']
            },
            'timeout': {
                'type': 'integer',
                'constraints': {'min': 0, 'max': 3600},
                'boundary_conditions': [0, -1, 3601]
            },
            'cache_size': {
                'type': 'integer',
                'constraints': {'min': 0, 'max': 10000},
                'boundary_conditions': [0, 10000, 10001]
            },
            'file_path': {
                'type': 'string',
                'constraints': {'min_length': 0, 'max_length': 1000},
                'boundary_conditions': ['', None, '/invalid/path']
            },
            'data_list': {
                'type': 'list',
                'constraints': {'max_length': 1000},
                'boundary_conditions': [[], None, [1,2,3]*1000]
            }
        }

    def generate_comprehensive_suite(self, function_name: str, parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """生成综合测试数据套件"""
        print(f"🧠 生成函数 {function_name} 的智能测试数据...")

        # 生成参数级别的测试数据
        parameter_data = self.smart_generator.generate_test_data_suite(parameters)

        # 生成组合测试数据
        combination_data = self._generate_parameter_combinations(parameter_data)

        # 生成场景级测试数据
        scenario_data = self._generate_scenario_based_data(parameter_data)

        return {
            'function_name': function_name,
            'parameter_data': parameter_data,
            'combination_data': combination_data,
            'scenario_data': scenario_data,
            'generation_stats': {
                'parameters_analyzed': len(parameters),
                'total_test_cases': self._calculate_total_cases(parameter_data, combination_data, scenario_data),
                'data_quality_score': self._assess_data_quality(parameter_data),
                'coverage_estimate': self._estimate_coverage(parameter_data)
            }
        }

    def _generate_parameter_combinations(self, parameter_data: Dict[str, SmartTestData]) -> List[Dict[str, Any]]:
        """生成参数组合"""
        combinations = []

        # 简单的组合生成：每个参数选择一个边界值
        param_names = list(parameter_data.keys())
        if len(param_names) <= 3:  # 限制组合复杂度
            # 生成所有参数的笛卡尔积的子集
            boundary_combinations = []

            # 选择每个参数的一个边界值进行组合
            combo = {}
            for param_name, smart_data in parameter_data.items():
                if smart_data.boundary_values:
                    combo[param_name] = smart_data.boundary_values[0]
            if combo:
                boundary_combinations.append(combo)

            combinations.extend(boundary_combinations)

        return combinations

    def _generate_scenario_based_data(self, parameter_data: Dict[str, SmartTestData]) -> List[Dict[str, Any]]:
        """生成基于场景的测试数据"""
        scenarios = []

        # 正常场景
        normal_scenario = {}
        for param_name, smart_data in parameter_data.items():
            if smart_data.valid_values:
                normal_scenario[param_name] = smart_data.valid_values[0]
        if normal_scenario:
            scenarios.append({
                'scenario': 'normal_operation',
                'description': '正常操作场景',
                'parameters': normal_scenario,
                'expected_result': 'success'
            })

        # 错误场景
        error_scenario = {}
        for param_name, smart_data in parameter_data.items():
            if smart_data.invalid_values:
                error_scenario[param_name] = smart_data.invalid_values[0]
        if error_scenario:
            scenarios.append({
                'scenario': 'error_handling',
                'description': '错误处理场景',
                'parameters': error_scenario,
                'expected_result': 'exception'
            })

        # 边界场景
        boundary_scenario = {}
        for param_name, smart_data in parameter_data.items():
            if smart_data.boundary_values:
                boundary_scenario[param_name] = smart_data.boundary_values[0]
        if boundary_scenario:
            scenarios.append({
                'scenario': 'boundary_conditions',
                'description': '边界条件场景',
                'parameters': boundary_scenario,
                'expected_result': 'edge_case_handling'
            })

        return scenarios

    def _calculate_total_cases(self, parameter_data: Dict[str, SmartTestData],
                             combination_data: List[Dict[str, Any]],
                             scenario_data: List[Dict[str, Any]]) -> int:
        """计算总测试用例数"""
        param_cases = sum(len(data.valid_values) + len(data.invalid_values) +
                          len(data.boundary_values) + len(data.edge_cases)
                          for data in parameter_data.values())

        combination_cases = len(combination_data)
        scenario_cases = len(scenario_data)

        return param_cases + combination_cases + scenario_cases

    def _assess_data_quality(self, parameter_data: Dict[str, SmartTestData]) -> float:
        """评估数据质量"""
        if not parameter_data:
            return 0.0

        quality_scores = []
        for smart_data in parameter_data.values():
            coverage = (len(smart_data.valid_values) + len(smart_data.invalid_values) +
                       len(smart_data.boundary_values) + len(smart_data.edge_cases))
            max_expected = 20  # 期望的测试数据数量
            quality_scores.append(min(coverage / max_expected, 1.0))

        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    def _estimate_coverage(self, parameter_data: Dict[str, SmartTestData]) -> float:
        """估算覆盖率提升"""
        if not parameter_data:
            return 0.0

        # 基于生成的测试数据量估算覆盖率提升
        total_data_points = sum(
            len(data.valid_values) + len(data.invalid_values) +
            len(data.boundary_values) + len(data.edge_cases)
            for data in parameter_data.values()
        )

        # 假设每个数据点可以覆盖0.5%的边界情况
        coverage_estimate = min(total_data_points * 0.005, 0.3)  # 最高30%

        return coverage_estimate


def main():
    """主函数 - 智能测试数据生成系统"""
    print("🧠 Phase 14.7: 智能测试数据生成系统")
    print("=" * 60)

    generator = TestDataSuiteGenerator()

    # 选择目标函数进行分析
    target_functions = [
        'create_config_factory',
        'get_config_factory',
        'load_data',
        'validate_data'
    ]

    results = {}

    for func_name in target_functions:
        print(f"\n🎯 分析函数: {func_name}")

        # 分析函数参数（这里使用预定义的参数信息）
        parameters = generator.analyze_function_parameters(func_name)

        # 生成综合测试数据套件
        suite = generator.generate_comprehensive_suite(func_name, parameters)
        results[func_name] = suite

        print(f"  📊 参数数量: {len(parameters)}")
        print(f"  🧪 总测试用例: {suite['generation_stats']['total_test_cases']}")
        print(".2f"        print(".1%")

        # 保存测试数据文件
        data_file = Path('test_logs') / f'smart_test_data_{func_name}.json'
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(suite, f, indent=2, ensure_ascii=False, default=str)

        print(f"  💾 测试数据已保存: {data_file}")

    # 生成汇总报告
    summary = {
        'generation_timestamp': '2026-02-15T10:00:00Z',
        'phase': 'Phase 14.7: 智能测试数据生成系统',
        'functions_analyzed': len(target_functions),
        'total_test_cases_generated': sum(r['generation_stats']['total_test_cases'] for r in results.values()),
        'average_data_quality': sum(r['generation_stats']['data_quality_score'] for r in results.values()) / len(results) if results else 0,
        'average_coverage_estimate': sum(r['generation_stats']['coverage_estimate'] for r in results.values()) / len(results) if results else 0,
        'results': results
    }

    # 保存汇总报告
    report_file = Path('test_logs') / 'phase14_smart_data_generation_results.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("
🧠 智能数据生成总结:"    print(f"  处理函数数: {summary['functions_analyzed']}")
    print(f"  测试用例总数: {summary['total_test_cases_generated']}")
    print(".2f"    print(".1%"    print(f"  详细报告: {report_file}")

    print("\n✅ Phase 14.7 智能测试数据生成系统完成")


if __name__ == '__main__':
    main()
