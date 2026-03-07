#!/usr/bin/env python3
"""
分层测试覆盖率审计脚本

按照架构设计对各层进行分层测试覆盖率审计
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 层级定义和对应的测试目录及源代码目录
LAYERS = {
    'infrastructure': {
        'test_dir': 'tests/infrastructure',
        'src_dir': 'src/infrastructure',
        'threshold': 80
    },
    'core': {
        'test_dir': 'tests/unit/core',
        'src_dir': 'src/core',
        'threshold': 80
    },
    'data': {
        'test_dir': 'tests/unit/data',
        'src_dir': 'src/data',
        'threshold': 80
    },
    'ml': {
        'test_dir': 'tests/unit/ml',
        'src_dir': 'src/ml',
        'threshold': 80
    },
    'strategy': {
        'test_dir': 'tests/unit/strategy',
        'src_dir': 'src/strategy',
        'threshold': 80
    }
}

def run_coverage_for_layer(layer_name, layer_config):
    """为指定层级运行覆盖率测试"""
    print(f"\n🔍 正在审计 {layer_name} 层...")

    test_dir = layer_config['test_dir']
    src_dir = layer_config['src_dir']

    if not os.path.exists(test_dir):
        print(f"⚠️  测试目录不存在: {test_dir}")
        return None

    if not os.path.exists(src_dir):
        print(f"⚠️  源代码目录不存在: {src_dir}")
        return None

    # 构建pytest命令
    cmd = [
        sys.executable, '-m', 'pytest',
        test_dir,
        f'--cov={src_dir}',
        '--cov-report=term-missing',
        '--cov-report=json:coverage.json',
        '-q',
        '--tb=no'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)

        # 解析覆盖率结果
        coverage_data = None
        if os.path.exists('coverage.json'):
            with open('coverage.json', 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)

        return {
            'layer': layer_name,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'coverage_data': coverage_data
        }

    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return None

def extract_coverage_percentage(coverage_data, src_dir):
    """从覆盖率数据中提取百分比"""
    if not coverage_data or 'files' not in coverage_data:
        return 0.0

    total_lines = 0
    covered_lines = 0

    for file_path, file_data in coverage_data['files'].items():
        # 只统计指定目录下的文件
        if file_path.startswith(src_dir):
            summary = file_data.get('summary', {})
            total_lines += summary.get('num_statements', 0)
            covered_lines += summary.get('covered_statements', 0)

    if total_lines == 0:
        return 0.0

    return (covered_lines / total_lines) * 100

def main():
    """主函数"""
    print("🚀 RQA2025 分层测试覆盖率审计")
    print("=" * 50)

    results = {}
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_layers': len(LAYERS),
        'passed_layers': 0,
        'failed_layers': 0,
        'average_coverage': 0.0
    }

    total_coverage = 0.0
    valid_layers = 0

    for layer_name, layer_config in LAYERS.items():
        result = run_coverage_for_layer(layer_name, layer_config)

        if result:
            coverage_pct = extract_coverage_percentage(
                result['coverage_data'],
                layer_config['src_dir']
            )

            threshold = layer_config['threshold']
            passed = coverage_pct >= threshold

            results[layer_name] = {
                'coverage': coverage_pct,
                'threshold': threshold,
                'passed': passed,
                'returncode': result['returncode']
            }

            if passed:
                summary['passed_layers'] += 1
            else:
                summary['failed_layers'] += 1

            total_coverage += coverage_pct
            valid_layers += 1

            status = "✅" if passed else "❌"
            print(f"{status} {layer_name}: {coverage_pct:.1f}% (阈值: {threshold}%)")
        else:
            results[layer_name] = {'error': '无法执行测试'}
            summary['failed_layers'] += 1
            print(f"❌ {layer_name}: 测试执行失败")

    # 计算平均覆盖率
    if valid_layers > 0:
        summary['average_coverage'] = total_coverage / valid_layers

    # 生成报告
    report_path = project_root / 'test_logs' / 'layer_coverage_audit_summary.json'
    os.makedirs(report_path.parent, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("📊 审计总结")
    print(f"📈 平均覆盖率: {summary['average_coverage']:.1f}%")
    print(f"✅ 达标层级: {summary['passed_layers']}")
    print(f"❌ 未达标层级: {summary['failed_layers']}")

    # 投产建议
    if summary['average_coverage'] >= 80 and summary['failed_layers'] == 0:
        print("\n🎉 恭喜！所有层级均达到投产标准")
    elif summary['average_coverage'] >= 70:
        print("\n⚠️ 大部分层级接近达标，建议继续完善")
    else:
        print("\n❌ 测试覆盖率严重不足，不建议立即投产")

    print(f"\n📄 详细报告已保存至: {report_path}")

    return 0 if summary['failed_layers'] == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
