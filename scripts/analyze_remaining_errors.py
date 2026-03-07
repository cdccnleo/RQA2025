#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析剩余的错误和失败的测试
"""

import subprocess
import re
from pathlib import Path
from datetime import datetime

def run_pytest_collection():
    """运行pytest收集测试并分析错误"""
    print("=" * 80)
    print("🔍 分析剩余收集错误")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 运行pytest收集测试
    cmd = ["pytest", "tests/unit/infrastructure/utils/", "--co", "-v", "--tb=short"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        output = result.stdout + result.stderr
        
        # 查找ERROR
        error_pattern = r'ERROR collecting (.+?)\n(.+?)(?=\nERROR|\n============================|\Z)'
        errors = re.findall(error_pattern, output, re.DOTALL)
        
        if errors:
            print(f"📋 找到 {len(errors)} 个收集错误：")
            print()
            
            for i, (file_path, error_detail) in enumerate(errors, 1):
                print(f"{'='*80}")
                print(f"错误 #{i}: {file_path}")
                print(f"{'='*80}")
                
                # 提取错误类型和消息
                if "ImportError" in error_detail:
                    import_match = re.search(r'ImportError: (.+?)(?:\n|$)', error_detail)
                    if import_match:
                        print(f"  类型: ImportError")
                        print(f"  消息: {import_match.group(1)}")
                
                if "ModuleNotFoundError" in error_detail:
                    module_match = re.search(r'ModuleNotFoundError: (.+?)(?:\n|$)', error_detail)
                    if module_match:
                        print(f"  类型: ModuleNotFoundError")
                        print(f"  消息: {module_match.group(1)}")
                
                # 显示关键错误行
                lines = error_detail.split('\n')
                for line in lines[:10]:
                    if line.strip() and not line.startswith('>'):
                        print(f"  {line}")
                
                print()
        
        else:
            print("  ✅ 没有找到收集错误！")
        
        # 统计测试数量
        collected_match = re.search(r'collected (\d+) items', output)
        if collected_match:
            print(f"📊 收集的测试数: {collected_match.group(1)}")
        
        print()
        print("=" * 80)
        print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        return errors
        
    except Exception as e:
        print(f"❌ 运行pytest时出错: {e}")
        return []

if __name__ == '__main__':
    errors = run_pytest_collection()

