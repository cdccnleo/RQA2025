#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础设施层精确覆盖率测试脚本

使用标准pytest命令格式:
    pytest tests/unit/infrastructure/{module} --cov=src/infrastructure/{module}

输出详细的覆盖率报告和统计数据
"""

import subprocess
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

# 17个基础设施核心模块
MODULES = [
    'constants', 'interfaces', 'core', 'ops', 
    'optimization', 'distributed', 'versioning', 'error',
    'api', 'monitoring', 'cache', 'security',
    'logging', 'resource', 'health', 'config', 'utils'
]

def run_pytest_coverage(module: str) -> Tuple[float, Dict]:
    """运行pytest覆盖率测试"""
    
    test_path = f"tests/unit/infrastructure/{module}"
    src_path = f"src/infrastructure/{module}"
    
    # 检查路径
    if not Path(test_path).exists():
        return 0.0, {'error': 'no_test_path', 'test_path': test_path}
    
    if not Path(src_path).exists():
        return 0.0, {'error': 'no_src_path', 'src_path': src_path}
    
    # 清理之前的覆盖率数据文件，避免累积
    coverage_files = ['.coverage', '.coverage.*']
    for pattern in coverage_files:
        for cov_file in Path.cwd().glob(pattern):
            try:
                cov_file.unlink()
            except:
                pass
    
    # pytest命令 - 使用用户指定的格式
    cmd = [
        'python', '-m', 'pytest',
        test_path,
        f'--cov={src_path}',
        '--cov-report=term',
        '-q',
        '--tb=no',
        '--no-header',
        '--maxfail=3',
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=Path.cwd(),
            encoding='utf-8',
            errors='replace'
        )
        
        # 解析TOTAL行的覆盖率
        total_match = re.search(
            r'TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%',
            result.stdout
        )
        
        if total_match:
            stmts = int(total_match.group(1))
            miss = int(total_match.group(2))
            coverage = int(total_match.group(3))
            
            # 解析测试统计
            passed = 0
            failed = 0
            errors = 0
            
            passed_match = re.search(r'(\d+) passed', result.stdout)
            failed_match = re.search(r'(\d+) failed', result.stdout)
            error_match = re.search(r'(\d+) error', result.stdout)
            
            if passed_match:
                passed = int(passed_match.group(1))
            if failed_match:
                failed = int(failed_match.group(1))
            if error_match:
                errors = int(error_match.group(1))
            
            stats = {
                'statements': stmts,
                'missed': miss,
                'coverage': coverage,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'exit_code': result.returncode,
                'success': result.returncode == 0
            }
            
            return float(coverage), stats
        else:
            # 无法解析覆盖率
            return 0.0, {
                'error': 'parse_failed',
                'stdout_preview': result.stdout[:500]
            }
            
    except subprocess.TimeoutExpired:
        return 0.0, {'error': 'timeout'}
    except Exception as e:
        return 0.0, {'error': 'exception', 'message': str(e)}


def format_result(module: str, coverage: float, stats: Dict) -> str:
    """格式化输出结果"""
    
    if 'error' in stats:
        return f"  ⚠️  {coverage:>5}% | ERROR: {stats['error']}"
    
    status = "✅" if coverage >= 80 else "⚠️" if coverage >= 60 else "❌"
    
    stmts = stats.get('statements', 0)
    miss = stats.get('missed', 0)
    passed = stats.get('passed', 0)
    failed = stats.get('failed', 0)
    errors = stats.get('errors', 0)
    
    return (
        f"  {status}  {coverage:>5.0f}% | "
        f"语句: {stmts:>6} | 未覆盖: {miss:>6} | "
        f"测试: 通过{passed:>4} 失败{failed:>3} 错误{errors:>2}"
    )


def main():
    """主函数"""
    
    print("="*80)
    print("基础设施层精确覆盖率测试（pytest --cov 实际执行）")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试模块数: {len(MODULES)}")
    print()
    
    results = {}
    total_stmts = 0
    total_miss = 0
    
    # 逐个测试模块
    for idx, module in enumerate(MODULES, 1):
        print(f"[{idx:2d}/17] {module:12s} ", end='', flush=True)
        
        coverage, stats = run_pytest_coverage(module)
        results[module] = {
            'coverage': coverage,
            'stats': stats
        }
        
        print(format_result(module, coverage, stats))
        
        # 累计统计
        if 'error' not in stats:
            total_stmts += stats.get('statements', 0)
            total_miss += stats.get('missed', 0)
    
    # 计算总体覆盖率
    if total_stmts > 0:
        overall_coverage = ((total_stmts - total_miss) / total_stmts) * 100
    else:
        overall_coverage = 0.0
    
    print()
    print("="*80)
    print("汇总统计")
    print("="*80)
    
    # 统计各等级模块
    excellent = sum(1 for r in results.values() if r['coverage'] >= 90)
    good = sum(1 for r in results.values() if 80 <= r['coverage'] < 90)
    qualified = sum(1 for r in results.values() if 60 <= r['coverage'] < 80)
    poor = sum(1 for r in results.values() if 30 <= r['coverage'] < 60)
    critical = sum(1 for r in results.values() if r['coverage'] < 30)
    
    print(f"总语句数: {total_stmts:,}")
    print(f"已覆盖: {total_stmts - total_miss:,}")
    print(f"未覆盖: {total_miss:,}")
    print(f"整体覆盖率: {overall_coverage:.1f}%")
    print()
    print(f"模块分类:")
    print(f"  优秀 (≥90%): {excellent} 个")
    print(f"  良好 (80-90%): {good} 个")
    print(f"  及格 (60-80%): {qualified} 个")
    print(f"  不及格 (30-60%): {poor} 个")
    print(f"  严重不足 (<30%): {critical} 个")
    print()
    
    # 投产建议
    if overall_coverage >= 80:
        print("🎯 投产建议: ✅ 符合80%投产标准")
    elif overall_coverage >= 60:
        print("⚠️ 投产建议: ⚠️ 需要提升至80%后投产")
    else:
        print("🚨 投产建议: ❌ 严重不达标，强烈不建议投产")
    
    print(f"   当前覆盖率: {overall_coverage:.1f}%")
    print(f"   目标标准: 80.0%")
    print(f"   差距: {overall_coverage - 80:.1f}个百分点")
    print()
    
    # 保存详细结果
    output_data = {
        'generated_at': datetime.now().isoformat(),
        'overall_coverage': overall_coverage,
        'total_statements': total_stmts,
        'total_missed': total_miss,
        'summary': {
            'excellent': excellent,
            'good': good,
            'qualified': qualified,
            'poor': poor,
            'critical': critical
        },
        'modules': results
    }
    
    json_file = Path('test_logs/accurate_coverage_results.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果已保存至: {json_file}")
    
    # 生成markdown报告
    generate_markdown_report(output_data)
    
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0


def generate_markdown_report(data: Dict):
    """生成markdown格式报告"""
    
    report = []
    report.append("# 基础设施层精确覆盖率测试报告（pytest实测）\n")
    report.append(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    report.append(f"**测试方法**: pytest --cov 标准命令  ")
    report.append(f"**测试模块**: 17个核心模块\n")
    report.append("---\n")
    
    report.append("## 总体统计\n")
    report.append(f"- **总语句数**: {data['total_statements']:,}")
    report.append(f"- **已覆盖**: {data['total_statements'] - data['total_missed']:,}")
    report.append(f"- **未覆盖**: {data['total_missed']:,}")
    report.append(f"- **整体覆盖率**: **{data['overall_coverage']:.1f}%**")
    
    if data['overall_coverage'] >= 80:
        report.append(f"- **投产建议**: ✅ 符合80%标准\n")
    else:
        report.append(f"- **投产建议**: ❌ 不符合80%标准（差{80 - data['overall_coverage']:.1f}%）\n")
    
    report.append("## 模块分类\n")
    s = data['summary']
    report.append(f"- 优秀 (≥90%): {s['excellent']} 个")
    report.append(f"- 良好 (80-90%): {s['good']} 个")
    report.append(f"- 及格 (60-80%): {s['qualified']} 个")
    report.append(f"- 不及格 (30-60%): {s['poor']} 个")
    report.append(f"- 严重不足 (<30%): {s['critical']} 个\n")
    
    report.append("## 详细模块数据\n")
    report.append("| # | 模块 | 覆盖率 | 语句数 | 未覆盖 | 测试通过 | 状态 |")
    report.append("|---|------|--------|--------|--------|----------|------|")
    
    # 按覆盖率排序
    sorted_modules = sorted(
        data['modules'].items(),
        key=lambda x: x[1]['coverage'],
        reverse=True
    )
    
    for idx, (module, info) in enumerate(sorted_modules, 1):
        cov = info['coverage']
        stats = info['stats']
        
        if 'error' in stats:
            status = "⚠️ 错误"
            stmts = "-"
            miss = "-"
            passed = "-"
        else:
            stmts = f"{stats.get('statements', 0):,}"
            miss = f"{stats.get('missed', 0):,}"
            passed = stats.get('passed', 0)
            
            if cov >= 90:
                status = "✅ 优秀"
            elif cov >= 80:
                status = "✅ 良好"
            elif cov >= 60:
                status = "⚠️ 及格"
            elif cov >= 30:
                status = "❌ 不及格"
            else:
                status = "❌ 严重不足"
        
        report.append(
            f"| {idx} | {module} | {cov:.0f}% | {stmts} | {miss} | {passed} | {status} |"
        )
    
    report.append("\n---\n")
    report.append("*报告生成时间*: " + datetime.now().isoformat())
    
    # 保存markdown报告
    report_file = Path('test_logs/accurate_coverage_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Markdown报告已保存至: {report_file}")


if __name__ == '__main__':
    exit(main())

