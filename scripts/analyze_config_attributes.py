"""
分析配置类属性缺失问题

收集所有AttributeError，分析缺失的配置类属性
"""

import re
from pathlib import Path
import subprocess


def run_test_and_collect_errors(test_dir):
    """运行测试并收集AttributeError"""
    
    cmd = [
        'pytest', test_dir,
        '-v', '--tb=line',
        '-n', '4',
        '--timeout=10',
        '--maxfail=50'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    output = result.stdout + result.stderr
    
    # 提取AttributeError
    attr_errors = re.findall(r"AttributeError: ['\"]?(\w+)['\"]? object has no attribute ['\"](\w+)['\"]", output)
    attr_errors += re.findall(r"AttributeError: '(\w+)' object has no attribute '(\w+)'", output)
    
    return attr_errors


def analyze_missing_attributes(test_dir):
    """分析缺失的属性"""
    
    print(f"分析 {test_dir} 的配置属性缺失...")
    print("=" * 70)
    
    errors = run_test_and_collect_errors(test_dir)
    
    # 统计
    from collections import Counter
    error_count = Counter(errors)
    
    print(f"\n发现 {len(errors)} 个AttributeError")
    print(f"涉及 {len(error_count)} 种不同的属性缺失\n")
    
    # 按频率排序
    print("缺失属性TOP 20:")
    for (class_name, attr_name), count in error_count.most_common(20):
        print(f"  {count:3d}x  {class_name}.{attr_name}")
    
    return error_count


if __name__ == '__main__':
    # 分析Config模块
    print("【分析Config模块】\n")
    config_errors = analyze_missing_attributes('tests/unit/infrastructure/config')
    
    print("\n" + "=" * 70)
    print("\n【分析Cache模块】\n")
    cache_errors = analyze_missing_attributes('tests/unit/infrastructure/cache')
    
    print("\n" + "=" * 70)
    print("\n【分析API模块】\n")
    api_errors = analyze_missing_attributes('tests/unit/infrastructure/api')


