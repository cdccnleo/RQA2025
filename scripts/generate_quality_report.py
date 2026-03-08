#!/usr/bin/env python3
"""
代码质量报告生成脚本
生成详细的代码质量分析报告
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime


def run_flake8_analysis():
    """运行Flake8分析"""
    try:
        result = subprocess.run(
            ['python', '-m', 'flake8', 'src', 
             '--max-line-length=100',
             '--extend-ignore=E203,W503',
             '--exclude=backups,production_simulation,docs,reports,__pycache__',
             '--count', '--statistics', '--format=json'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # 解析JSON输出
        if result.stdout:
            try:
                errors = json.loads(result.stdout)
                return errors
            except:
                return []
        return []
    except Exception as e:
        print(f"Flake8分析失败: {e}")
        return []


def analyze_error_types(errors):
    """分析错误类型分布"""
    error_types = {}
    file_errors = {}
    
    for error in errors:
        code = error.get('code', 'UNKNOWN')
        filename = error.get('filename', 'unknown')
        
        # 统计错误类型
        error_types[code] = error_types.get(code, 0) + 1
        
        # 统计文件错误
        if filename not in file_errors:
            file_errors[filename] = []
        file_errors[filename].append(error)
    
    return error_types, file_errors


def calculate_quality_score(total_errors, total_lines):
    """计算代码质量评分"""
    if total_lines == 0:
        return 10.0
    
    # 基础分数10分
    base_score = 10.0
    
    # 根据错误密度扣分
    error_rate = total_errors / total_lines
    
    # 扣分规则
    if error_rate < 0.001:  # 错误率 < 0.1%
        deduction = 0
    elif error_rate < 0.005:  # 错误率 < 0.5%
        deduction = 1
    elif error_rate < 0.01:  # 错误率 < 1%
        deduction = 2
    elif error_rate < 0.02:  # 错误率 < 2%
        deduction = 3
    elif error_rate < 0.05:  # 错误率 < 5%
        deduction = 4
    else:
        deduction = 5
    
    return max(0, base_score - deduction)


def count_lines_of_code():
    """统计代码行数"""
    total_lines = 0
    src_dir = Path("src")
    
    for py_file in src_dir.rglob("*.py"):
        if any(skip in str(py_file) for skip in ['backups', 'production_simulation', 'docs', 'reports', '__pycache__']):
            continue
        
        try:
            content = py_file.read_text(encoding='utf-8')
            lines = len(content.split('\n'))
            total_lines += lines
        except:
            pass
    
    return total_lines


def generate_report():
    """生成质量报告"""
    print("="*70)
    print("RQA2025 代码质量分析报告")
    print("="*70)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 统计代码行数
    print("📊 代码统计")
    print("-"*70)
    total_lines = count_lines_of_code()
    print(f"总代码行数: {total_lines:,}")
    print()
    
    # 运行Flake8分析
    print("🔍 正在运行代码质量分析...")
    errors = run_flake8_analysis()
    
    if not errors:
        print("✅ 没有发现代码质量问题！")
        return
    
    # 分析错误
    error_types, file_errors = analyze_error_types(errors)
    
    # 计算质量评分
    quality_score = calculate_quality_score(len(errors), total_lines)
    
    # 显示总体情况
    print("\n📈 质量评分")
    print("-"*70)
    print(f"综合评分: {quality_score:.2f}/10.0")
    print(f"总错误数: {len(errors)}")
    print(f"错误密度: {len(errors)/total_lines*100:.3f}%")
    print()
    
    # 显示错误类型分布
    print("📋 错误类型分布")
    print("-"*70)
    print(f"{'错误代码':<12} {'数量':<8} {'占比':<8} {'描述'}")
    print("-"*70)
    
    # 错误代码描述
    error_descriptions = {
        'E501': '行过长',
        'E302': '函数/类前空行不足',
        'E305': '函数/类后空行不足',
        'E401': '多行导入',
        'F401': '未使用导入',
        'F821': '未定义变量',
        'F822': '未定义导出',
        'W291': '行尾空格',
        'W293': '空行空格',
        'W391': '文件末尾空行',
    }
    
    for code, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:15]:
        percentage = count / len(errors) * 100
        description = error_descriptions.get(code, '其他')
        print(f"{code:<12} {count:<8} {percentage:>6.2f}%  {description}")
    
    print()
    
    # 显示问题最严重的文件
    print("📁 问题文件Top 10")
    print("-"*70)
    print(f"{'文件路径':<50} {'错误数'}")
    print("-"*70)
    
    sorted_files = sorted(file_errors.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for filepath, file_error_list in sorted_files:
        short_path = filepath.replace('src\\', '').replace('src/', '')
        if len(short_path) > 48:
            short_path = '...' + short_path[-45:]
        print(f"{short_path:<50} {len(file_error_list):>5}")
    
    print()
    
    # 显示改进建议
    print("💡 改进建议")
    print("-"*70)
    
    suggestions = []
    
    if 'E501' in error_types:
        suggestions.append("1. 使用Black自动格式化代码，统一行长度")
    
    if 'F401' in error_types:
        suggestions.append("2. 清理未使用的导入语句")
    
    if 'F821' in error_types:
        suggestions.append("3. 修复未定义变量错误，添加缺失的导入")
    
    if 'W291' in error_types or 'W293' in error_types:
        suggestions.append("4. 去除行尾和空行中的空格")
    
    if 'E302' in error_types or 'E305' in error_types:
        suggestions.append("5. 规范函数和类定义前后的空行")
    
    if not suggestions:
        suggestions.append("1. 继续保持良好的代码质量")
        suggestions.append("2. 考虑添加更多类型注解")
        suggestions.append("3. 增加单元测试覆盖率")
    
    for suggestion in suggestions:
        print(suggestion)
    
    print()
    print("="*70)
    
    # 保存详细报告
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'total_lines': total_lines,
        'total_errors': len(errors),
        'quality_score': quality_score,
        'error_types': error_types,
        'top_files': [{'file': f, 'errors': len(e)} for f, e in sorted_files],
        'suggestions': suggestions
    }
    
    report_file = Path('code_quality_report.json')
    report_file.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"详细报告已保存: {report_file}")


if __name__ == "__main__":
    generate_report()
