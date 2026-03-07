#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
识别所有Collection Errors
找出无法被pytest正确收集的测试文件
"""

import subprocess
import re
from pathlib import Path


def find_collection_errors():
    """运行pytest收集所有测试，识别errors"""
    
    print("="*80)
    print("识别Collection Errors")
    print("="*80)
    
    # 运行pytest --collect-only来收集所有测试
    cmd = ["pytest", "tests/", "--collect-only", "-q"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout + result.stderr
        
        # 提取错误信息
        error_pattern = r"ERROR collecting (tests[^\s]+)"
        errors = re.findall(error_pattern, output)
        
        print(f"\n找到 {len(errors)} 个Collection Errors:")
        print("-"*80)
        
        for i, error_file in enumerate(errors, 1):
            print(f"{i}. {error_file}")
        
        # 提取错误原因
        print("\n\n" + "="*80)
        print("错误详情")
        print("="*80)
        
        # 查找ERROR行及其后续行
        lines = output.split('\n')
        in_error = False
        current_error = []
        
        for line in lines:
            if line.startswith('ERROR collecting'):
                if current_error:
                    print('\n'.join(current_error))
                    print('-'*80)
                current_error = [line]
                in_error = True
            elif in_error:
                if line.startswith('ERROR') or line.startswith('='):
                    in_error = False
                    if current_error:
                        print('\n'.join(current_error))
                        print('-'*80)
                    current_error = []
                else:
                    current_error.append(line)
        
        if current_error:
            print('\n'.join(current_error))
        
        return errors
        
    except subprocess.TimeoutExpired:
        print("⏱️ 超时！测试收集时间过长")
        return []
    except Exception as e:
        print(f"❌ 错误: {e}")
        return []


if __name__ == "__main__":
    errors = find_collection_errors()
    
    print(f"\n\n{'='*80}")
    print(f"总结: 找到 {len(errors)} 个Collection Errors")
    print(f"{'='*80}")
    
    if errors:
        print("\n下一步：逐一修复这些errors")
    else:
        print("\n✅ 没有Collection Errors！")

