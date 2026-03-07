#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest收集阶段自动化排查脚本
自动检测pytest collecting卡死的测试文件，并分析依赖链。
"""
import sys
import time
import subprocess
from pathlib import Path
import ast
import threading

PROJECT_ROOT = Path(__file__).parent.parent
TEST_DIR = PROJECT_ROOT / 'tests' / 'unit' / 'infrastructure'
TIMEOUT = 20  # 单文件收集超时时间（秒）


def find_pytest_files(test_dir):
    """递归查找所有测试文件"""
    return sorted([str(p) for p in test_dir.rglob('test_*.py')])


def try_pytest_collect(test_file, timeout=TIMEOUT):
    """尝试pytest --collect-only收集单个文件"""
    cmd = [sys.executable, '-m', 'pytest', '--collect-only', '-v', test_file]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        timer = threading.Timer(timeout, proc.kill)
        timer.start()
        stdout, stderr = proc.communicate()
        timer.cancel()
        return proc.returncode, stdout, stderr
    except Exception as e:
        return -1, '', str(e)


def analyze_imports(pyfile):
    """分析测试文件的import依赖链"""
    try:
        with open(pyfile, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=pyfile)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.append(n.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    except Exception as e:
        return [f'分析失败: {e}']


def main():
    print(f"\n🔍 自动化排查pytest collecting卡死文件...\n")
    test_files = find_pytest_files(TEST_DIR)
    stuck_files = []
    all_results = {}
    for tf in test_files:
        print(f"[收集] {tf} ...", end='', flush=True)
        start = time.time()
        rc, out, err = try_pytest_collect(tf)
        duration = time.time() - start
        if rc == 0:
            print(f" OK ({duration:.1f}s)")
        else:
            print(f" ❌ 卡死/超时/异常 ({duration:.1f}s)")
            stuck_files.append(tf)
            all_results[tf] = {'rc': rc, 'stdout': out, 'stderr': err, 'duration': duration}

    if not stuck_files:
        print("\n✅ 未发现卡死的测试文件，pytest collecting阶段正常！")
        return

    print(f"\n⚠️  检测到 {len(stuck_files)} 个卡死/超时/异常的测试文件：")
    for f in stuck_files:
        print(f"  - {f}")

    print("\n🔬 依赖链分析与修复建议：")
    for f in stuck_files:
        print(f"\n--- {f} ---")
        imports = analyze_imports(f)
        print("依赖import链：")
        for imp in imports:
            print(f"  - {imp}")
        # 简单修复建议
        print("修复建议：")
        print("  1. 检查上述依赖模块是否在import时有阻塞/死循环/长时间等待/外部IO。")
        print("  2. 检查测试文件和依赖模块的顶层代码，避免在import时执行业务逻辑、网络请求、数据库连接等。")
        print("  3. 可用mock/patch替换外部依赖，或将耗时操作放到函数/fixture内部。")
        print("  4. 如有conftest.py，检查其中的全局fixture和hook。")
        if all_results.get(f):
            print(f"stderr片段: {all_results[f]['stderr'][:300]}")


if __name__ == "__main__":
    main()
