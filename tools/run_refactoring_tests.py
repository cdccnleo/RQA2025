#!/usr/bin/env python3
"""
重构验证测试脚本

运行所有相关测试，验证重构后的代码质量

创建时间: 2025-11-03
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_pytest(test_path: str, description: str) -> dict:
    """运行pytest测试"""
    print(f"\n🧪 运行测试: {description}")
    print("-" * 70)
    
    cmd = [
        sys.executable, '-m', 'pytest',
        test_path,
        '-v',
        '--tb=short',
        '-n', 'auto'  # 并行测试
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        return {
            'description': description,
            'path': test_path,
            'returncode': result.returncode,
            'success': result.returncode == 0,
            'output': result.stdout
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ 测试超时: {description}")
        return {
            'description': description,
            'path': test_path,
            'returncode': -1,
            'success': False,
            'output': '测试超时'
        }
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return {
            'description': description,
            'path': test_path,
            'returncode': -1,
            'success': False,
            'output': str(e)
        }


def main():
    """主函数"""
    print("=" * 70)
    print("重构验证测试套件")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_suites = [
        ('tests/unit/core/foundation/test_base_component.py', 'BaseComponent单元测试'),
        ('tests/unit/core/foundation/test_base_adapter.py', 'BaseAdapter单元测试'),
    ]
    
    results = []
    
    for test_path, description in test_suites:
        result = run_pytest(test_path, description)
        results.append(result)
    
    # 生成报告
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    
    for result in results:
        status = "✅ 通过" if result['success'] else "❌ 失败"
        print(f"{status}: {result['description']}")
    
    print("")
    print(f"总计: {len(results)} 个测试套件")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    print(f"成功率: {(passed / len(results) * 100):.1f}%")
    
    # 保存报告
    report_file = PROJECT_ROOT / 'test_logs' / 'refactoring_test_results.txt'
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"重构验证测试报告\n")
        f.write(f"执行时间: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")
        
        for result in results:
            f.write(f"测试: {result['description']}\n")
            f.write(f"路径: {result['path']}\n")
            f.write(f"状态: {'通过' if result['success'] else '失败'}\n")
            f.write("-" * 70 + "\n")
            f.write(result['output'])
            f.write("\n\n")
    
    print(f"\n📄 详细报告已保存到: {report_file.relative_to(PROJECT_ROOT)}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

