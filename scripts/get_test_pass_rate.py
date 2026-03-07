#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取测试通过率
"""

import subprocess
import sys
import re

def run_tests():
    """运行测试并解析结果"""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/infrastructure/monitoring",
        "-q",
        "--tb=no"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace'
        )
        
        output = result.stdout + result.stderr
        
        # 解析测试统计
        stats = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'pass_rate': 0.0
        }
        
        # 查找测试统计行
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line.lower():
                # 使用正则表达式提取
                patterns = [
                    (r'(\d+)\s+passed', 'passed'),
                    (r'(\d+)\s+failed', 'failed'),
                    (r'(\d+)\s+error', 'errors'),
                    (r'(\d+)\s+skipped', 'skipped')
                ]
                
                for pattern, key in patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        stats[key] = int(match.group(1))
        
        stats['total'] = stats['passed'] + stats['failed'] + stats['errors'] + stats['skipped']
        if stats['total'] > 0:
            stats['pass_rate'] = (stats['passed'] / stats['total']) * 100
        
        return stats, output
        
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return None, None

if __name__ == "__main__":
    stats, output = run_tests()
    
    if stats:
        print("测试通过率统计:")
        print(f"  总测试数: {stats['total']}")
        print(f"  通过: {stats['passed']}")
        print(f"  失败: {stats['failed']}")
        print(f"  错误: {stats['errors']}")
        print(f"  跳过: {stats['skipped']}")
        print(f"  通过率: {stats['pass_rate']:.2f}%")
    else:
        print("无法获取测试统计信息")
        if output:
            print("\n输出:")
            print(output[-500:])  # 显示最后500字符

