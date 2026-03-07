#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化交易系统21层级测试覆盖率检查脚本
"""

import os
import subprocess
import re
import json
from datetime import datetime

def run_coverage_test(test_dirs, src_dirs, test_type_name):
    """运行覆盖率测试"""
    try:
        # 构建pytest命令
        cmd = ['python', '-m', 'pytest', '--cov-report', 'term-missing', '--tb=no', '-q']

        # 添加源码目录
        for src_dir in src_dirs:
            if os.path.exists(src_dir):
                cmd.extend(['--cov', src_dir])

        # 添加测试目录
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                cmd.append(test_dir)

        if len(cmd) <= 4:  # 只有基本参数，没有实际的测试或源码目录
            return None

        # 运行测试
        result = subprocess.run(cmd, capture_output=True, text=True,
                              encoding='utf-8', errors='ignore', timeout=600)

        output = result.stdout + result.stderr

        # 解析覆盖率结果
        cov_match = re.search(r'TOTAL.*?(\d+%)\s+(\d+)\s+(\d+)', output, re.MULTILINE)
        if cov_match:
            return {
                'coverage': cov_match.group(1),
                'total_lines': cov_match.group(2),
                'covered_lines': cov_match.group(3)
            }

        return None

    except Exception as e:
        print(f'  调试: {test_type_name}执行异常: {str(e)[:100]}')
        return None

def run_layer_coverage_check():
    print('🔍 量化交易系统21层级全面测试覆盖率检查')
    print('=' * 80)

    # 基于现有测试结构定义层级映射
    layer_mappings = {
        'infrastructure': {
            'test_dirs': ['tests/infrastructure', 'tests/unit/infrastructure'],
            'src_dirs': ['src'],
            'name': '基础设施层'
        },
        'core_services': {
            'test_dirs': ['tests/unit/core', 'tests/integration/core'],
            'src_dirs': ['src/core'],
            'name': '核心服务层'
        },
        'data_management': {
            'test_dirs': ['tests/unit/data', 'tests/integration/data', 'tests/data'],
            'src_dirs': ['src/data'],
            'name': '数据管理层'
        },
        'feature_analysis': {
            'test_dirs': ['tests/unit/features', 'tests/features'],
            'src_dirs': ['src/features'],
            'name': '特征分析层'
        },
        'machine_learning': {
            'test_dirs': ['tests/unit/ml', 'tests/test_ml_comprehensive.py'],
            'src_dirs': ['src/ml'],
            'name': '机器学习层'
        },
        'strategy_services': {
            'test_dirs': ['tests/unit/strategy', 'tests/business'],
            'src_dirs': ['src/strategy'],
            'name': '策略服务层'
        },
        'trading': {
            'test_dirs': ['tests/unit/trading', 'tests/test_trading_engine_comprehensive.py'],
            'src_dirs': ['src/trading'],
            'name': '交易层'
        },
        'risk_control': {
            'test_dirs': ['tests/unit/risk', 'tests/test_risk_compliance_comprehensive.py'],
            'src_dirs': ['src/risk'],
            'name': '风险控制层'
        },
        'monitoring': {
            'test_dirs': ['tests/unit/monitoring', 'tests/test_monitoring_optimization_comprehensive.py'],
            'src_dirs': ['src/monitoring'],
            'name': '监控层'
        },
        'stream_processing': {
            'test_dirs': ['tests/unit/stream', 'tests/test_gateway_streaming_comprehensive.py'],
            'src_dirs': ['src/stream'],
            'name': '流处理层'
        },
        'gateway': {
            'test_dirs': ['tests/gateway'],
            'src_dirs': ['src/gateway'],
            'name': '网关层'
        },
        'optimization': {
            'test_dirs': ['tests/unit/optimization', 'tests/test_automation_resilience_comprehensive.py'],
            'src_dirs': ['src/optimization'],
            'name': '优化层'
        },
        'adapter': {
            'test_dirs': ['tests/unit/adapters', 'tests/test_utils_adapters_comprehensive.py'],
            'src_dirs': ['src/adapters'],
            'name': '适配器层'
        },
        'automation': {
            'test_dirs': ['tests/unit/automation'],
            'src_dirs': ['src/automation'],
            'name': '自动化层'
        },
        'resilience': {
            'test_dirs': ['tests/unit/resilience'],
            'src_dirs': ['src/resilience'],
            'name': '弹性层'
        },
        'testing': {
            'test_dirs': ['tests/unit/testing'],
            'src_dirs': ['src/testing'],
            'name': '测试层'
        },
        'tools': {
            'test_dirs': ['tests/unit/tools'],
            'src_dirs': ['src/tools'],
            'name': '工具层'
        },
        'distributed_coordinator': {
            'test_dirs': ['tests/unit/distributed'],
            'src_dirs': ['src/distributed'],
            'name': '分布式协调器层'
        },
        'async_processor': {
            'test_dirs': ['tests/unit/async'],
            'src_dirs': ['src/async'],
            'name': '异步处理器层'
        },
        'mobile': {
            'test_dirs': ['tests/unit/mobile'],
            'src_dirs': ['src/mobile'],
            'name': '移动端层'
        },
        'business_boundary': {
            'test_dirs': ['tests/business', 'tests/unit/business'],
            'src_dirs': ['src/business'],
            'name': '业务边界层'
        }
    }

    print(f'📋 总计检查层级: {len(layer_mappings)} 个')
    print()

    results = {}
    total_lines_all = 0
    covered_lines_all = 0
    valid_layers = 0

    for layer_code, config in layer_mappings.items():
        layer_name = config['name']
        print(f'🔍 检查 {layer_name} ({layer_code})...')

        # 检查测试文件和源码是否存在
        test_files_exist = any(os.path.exists(test_dir) for test_dir in config['test_dirs'])
        src_files_exist = any(os.path.exists(src_dir) for src_dir in config['src_dirs'])

        if not test_files_exist and not src_files_exist:
            results[layer_code] = {
                'layer_name': layer_name,
                'coverage': 'N/A',
                'total_lines': 'N/A',
                'covered_lines': 'N/A',
                'status': '⚠️ 测试和源码均不存在',
                'test_type': '无'
            }
            print(f'  ⚠️ 测试和源码目录均不存在')
            continue

        # 运行不同类型的测试
        coverage_data = {'unit': None, 'integration': None, 'e2e': None}

        # 单元测试
        unit_test_dirs = [d for d in config['test_dirs'] if 'unit' in d]
        if unit_test_dirs:
            coverage_data['unit'] = run_coverage_test(unit_test_dirs, config['src_dirs'], '单元测试')

        # 集成测试
        integration_test_dirs = [d for d in config['test_dirs'] if 'integration' in d or 'e2e' in d or not 'unit' in d]
        if integration_test_dirs:
            coverage_data['integration'] = run_coverage_test(integration_test_dirs, config['src_dirs'], '集成测试')

        # 汇总覆盖率
        total_lines = 0
        covered_lines = 0
        test_types = []

        for test_type, cov_data in coverage_data.items():
            if cov_data and cov_data['total_lines'] and cov_data['total_lines'] != 'N/A':
                try:
                    total_lines += int(cov_data['total_lines'])
                    covered_lines += int(cov_data['covered_lines'])
                    test_types.append(test_type)
                except:
                    pass

        if total_lines > 0:
            coverage_pct = f'{covered_lines/total_lines*100:.1f}%'
            status = '✅ 已完成'
            test_type_str = '+'.join(test_types) + '测试' if test_types else '测试'

            results[layer_code] = {
                'layer_name': layer_name,
                'coverage': coverage_pct,
                'total_lines': str(total_lines),
                'covered_lines': str(covered_lines),
                'status': status,
                'test_type': test_type_str
            }

            print(f'  ✅ 覆盖率: {coverage_pct} (覆盖{total_lines}行中的{covered_lines}行)')
            total_lines_all += total_lines
            covered_lines_all += covered_lines
            valid_layers += 1
        else:
            results[layer_code] = {
                'layer_name': layer_name,
                'coverage': '0%',
                'total_lines': '0',
                'covered_lines': '0',
                'status': '❌ 无有效覆盖率数据',
                'test_type': '无'
            }
            print(f'  ❌ 无有效覆盖率数据')

    print()
    print('📊 各层级测试覆盖率汇总:')
    print('=' * 80)
    print('层级名称          覆盖率    总行数  覆盖行数  状态           测试类型')
    print('-' * 80)

    for layer_code, data in results.items():
        name = data['layer_name'][:18]
        coverage = data['coverage'][:8]
        total_lines = data['total_lines'][:6]
        covered_lines = data['covered_lines'][:8]
        status = data['status'][:12]
        test_type = data['test_type']

        print(f'{name:<18} {coverage:<8} {total_lines:<6} {covered_lines:<8} {status:<12} {test_type}')

    # 计算总体覆盖率
    overall_coverage = 'N/A'
    if total_lines_all > 0:
        overall_percentage = (covered_lines_all / total_lines_all) * 100
        overall_coverage = f'{overall_percentage:.1f}%'

    print('-' * 80)
    summary_text = f'总体统计: {overall_coverage} (覆盖{total_lines_all}行中的{covered_lines_all}行, {valid_layers}/{len(layer_mappings)}层有效)'
    print(summary_text)

    print()
    print('🎯 测试覆盖率评估标准:')
    print('  🟢 优秀: ≥ 80%')
    print('  ✅ 良好: 60-79%')
    print('  🟡 一般: 40-59%')
    print('  🔴 需改进: < 40%')

    print()
    print('📝 建议:')
    print('1. 对覆盖率低于60%的层级进行重点优化')
    print('2. 补充集成测试和端到端测试用例')
    print('3. 建立持续的覆盖率监控机制')
    print('4. 完善测试文档和开发规范')

    # 保存详细结果到报告
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'total_layers': len(layer_mappings),
        'valid_layers': valid_layers,
        'overall_coverage': overall_coverage,
        'total_lines': total_lines_all,
        'covered_lines': covered_lines_all,
        'layer_results': results
    }

    with open('test_logs/layer_coverage_current_status.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f'\n💾 详细报告已保存到: test_logs/layer_coverage_current_status.json')

if __name__ == "__main__":
    run_layer_coverage_check()
