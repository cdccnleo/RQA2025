#!/usr/bin/env python3
"""Data层测试运行脚本"""
import argparse
import sys
import pytest
import signal
from pathlib import Path


def timeout_handler(signum, frame):
    raise TimeoutError("测试超时")


def main():
    parser = argparse.ArgumentParser(description='运行Data层测试')
    parser.add_argument(
        '--module',
        type=str,
        help='指定测试模块 (stock_loader, industry_loader, adapters等)'
    )
    parser.add_argument(
        '--test',
        type=str,
        help='指定具体测试用例'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='测试超时时间(秒)'
    )
    parser.add_argument(
        '--maxfail',
        type=int,
        default=10,
        help='最大失败数'
    )
    parser.add_argument(
        '--html-report',
        metavar='FILE',
        help='生成HTML测试报告'
    )

    args = parser.parse_args()

    # 确定测试路径
    base_dir = Path(__file__).parent.parent
    test_path = base_dir / "tests/unit/data"

    if args.module:
        test_path = test_path / f"test_{args.module}.py"
    elif args.test:
        # 如果指定了具体测试，需要找到对应的文件
        test_path = test_path / f"test_{args.test.split('_')[0]}.py"
    else:
        # 默认运行所有Data层测试
        test_path = test_path

    # 构建pytest参数
    pytest_args = [
        str(test_path),
        "-v",
        f"--maxfail={args.maxfail}",
        "--disable-warnings",
        "--tb=short"
    ]

    if args.test:
        pytest_args.append(f"::{args.test}")

    if args.html_report:
        pytest_args.extend([
            "--html", args.html_report,
            "--self-contained-html"
        ])

    # 设置超时
    if args.timeout > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)

    try:
        # 运行测试
        exit_code = pytest.main(pytest_args)
        signal.alarm(0)  # 取消超时
        sys.exit(exit_code)
    except TimeoutError:
        print(f"测试超时 ({args.timeout}秒)")
        sys.exit(1)
    except KeyboardInterrupt:
        print("测试被用户中断")
        sys.exit(1)


if __name__ == "__main__":
    main()
