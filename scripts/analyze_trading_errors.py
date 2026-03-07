#!/usr/bin/env python3
"""
分析Trading层测试错误，分类并生成修复计划
"""
import subprocess
import re
from collections import defaultdict

def analyze_trading_errors():
    """分析Trading层测试错误"""
    print("=" * 60)
    print("  Trading层错误分析工具")
    print("=" * 60)
    print()
    
    # 运行pytest收集错误
    print("正在运行测试并收集错误信息...")
    result = subprocess.run(
        [
            "pytest",
            "tests/unit/trading/",
            "--tb=line",
            "-q"
        ],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    output = result.stdout + result.stderr
    lines = output.split('\n')
    
    # 分类错误
    error_types = defaultdict(list)
    failed_tests = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 收集ERROR类型的错误
        if 'ERROR' in line and 'collecting' in line.lower():
            test_path = line.split('ERROR')[1].strip() if 'ERROR' in line else ''
            error_info = []
            i += 1
            
            # 收集错误详情
            while i < len(lines) and not lines[i].strip().startswith('E '):
                i += 1
            
            while i < len(lines) and lines[i].strip().startswith('E '):
                error_line = lines[i].strip()
                error_info.append(error_line)
                
                # 提取错误类型
                if 'ImportError' in error_line:
                    error_types['ImportError'].append((test_path, error_line))
                elif 'AttributeError' in error_line:
                    error_types['AttributeError'].append((test_path, error_line))
                elif 'ModuleNotFoundError' in error_line:
                    error_types['ModuleNotFoundError'].append((test_path, error_line))
                elif 'TypeError' in error_line:
                    error_types['TypeError'].append((test_path, error_line))
                elif 'SyntaxError' in error_line:
                    error_types['SyntaxError'].append((test_path, error_line))
                
                i += 1
            
            if error_info:
                error_types['AllErrors'].append((test_path, '\n'.join(error_info[:3])))
        
        # 收集FAILED类型的错误
        elif 'FAILED' in line and '::' in line:
            failed_tests.append(line.strip())
        
        i += 1
    
    # 输出分析结果
    print("\n" + "=" * 60)
    print("  📊 错误分析结果")
    print("=" * 60)
    
    # 统计
    total_errors = len(error_types.get('AllErrors', []))
    print(f"\n总错误数: {total_errors}")
    print(f"失败测试数: {len(failed_tests)}")
    
    # 按类型输出
    print("\n按错误类型分类:")
    for error_type in ['ImportError', 'AttributeError', 'ModuleNotFoundError', 'TypeError', 'SyntaxError']:
        if error_type in error_types:
            count = len(error_types[error_type])
            print(f"\n  {error_type}: {count}个")
            
            # 显示前5个示例
            for i, (test_path, error_msg) in enumerate(error_types[error_type][:5], 1):
                print(f"    {i}. {test_path}")
                print(f"       {error_msg[:100]}...")
            
            if len(error_types[error_type]) > 5:
                print(f"    ... 还有{len(error_types[error_type]) - 5}个")
    
    # 输出到文件
    with open('test_logs/TRADING_ERRORS_ANALYSIS.md', 'w', encoding='utf-8') as f:
        f.write("# Trading层错误分析报告\n\n")
        f.write(f"**总错误数**: {total_errors}\n")
        f.write(f"**失败测试数**: {len(failed_tests)}\n\n")
        
        for error_type in ['ImportError', 'AttributeError', 'ModuleNotFoundError', 'TypeError', 'SyntaxError']:
            if error_type in error_types:
                f.write(f"\n## {error_type} ({len(error_types[error_type])}个)\n\n")
                for test_path, error_msg in error_types[error_type]:
                    f.write(f"### {test_path}\n\n")
                    f.write(f"```\n{error_msg}\n```\n\n")
    
    print(f"\n详细报告已保存到: test_logs/TRADING_ERRORS_ANALYSIS.md")
    
    return error_types, failed_tests

if __name__ == "__main__":
    analyze_trading_errors()

