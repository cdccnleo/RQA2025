"""
策略执行监控模块测试执行脚本

批量运行所有监控模块的单元测试
"""

import subprocess
import sys
import os
from pathlib import Path

# 测试文件列表
TEST_FILES = [
    "test_execution_monitor.py",
]

def run_tests():
    """运行所有测试"""
    test_dir = Path(__file__).parent
    
    print("=" * 60)
    print("策略执行监控模块单元测试")
    print("=" * 60)
    
    all_passed = True
    
    for test_file in TEST_FILES:
        test_path = test_dir / test_file
        if not test_path.exists():
            print(f"\n⚠️  测试文件不存在: {test_file}")
            continue
            
        print(f"\n{'='*60}")
        print(f"运行测试: {test_file}")
        print('='*60)
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
                cwd=test_dir.parent.parent.parent.parent,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            if result.returncode != 0:
                all_passed = False
                print(f"❌ {test_file} 测试失败")
            else:
                print(f"✅ {test_file} 测试通过")
                
        except subprocess.TimeoutExpired:
            print(f"❌ {test_file} 测试超时")
            all_passed = False
        except Exception as e:
            print(f"❌ {test_file} 运行错误: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(run_tests())
