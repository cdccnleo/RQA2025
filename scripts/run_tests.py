"""测试运行脚本"""
import argparse
import sys
import pytest
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='运行基础设施层测试')
    parser.add_argument(
        '--unit',
        action='store_true',
        help='仅运行单元测试'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='仅运行集成测试'
    )
    parser.add_argument(
        '--performance',
        action='store_true',
        help='仅运行性能测试'
    )
    parser.add_argument(
        '--html-report',
        metavar='FILE',
        help='生成HTML测试报告'
    )
    parser.add_argument(
        '--junit-xml',
        metavar='FILE',
        help='生成JUnit XML报告'
    )

    args = parser.parse_args()

    # 确定测试目录
    test_dirs = []
    base_dir = Path(__file__).parent.parent

    if args.unit:
        test_dirs.append(str(base_dir / "tests/infrastructure"))
    elif args.integration:
        test_dirs.append(str(base_dir / "tests/integration"))
    elif args.performance:
        test_dirs.append(str(base_dir / "tests/performance"))
    else:
        # 默认运行所有测试
        test_dirs.extend([
            str(base_dir / "tests/infrastructure"),
            str(base_dir / "tests/integration"),
            str(base_dir / "tests/performance")
        ])

    # 构建pytest参数
    pytest_args = []

    if args.html_report:
        pytest_args.extend([
            "--html", args.html_report,
            "--self-contained-html"
        ])

    if args.junit_xml:
        pytest_args.extend(["--junit-xml", args.junit_xml])

    pytest_args.extend(test_dirs)

    # 运行测试
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
