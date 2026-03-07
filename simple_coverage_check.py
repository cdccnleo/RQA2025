import json


def analyze_coverage():
    coverage_file = "reports/coverage.json"

    try:
        with open(coverage_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading coverage file: {e}")
        return

    total_statements = 0
    total_covered = 0

    for file_path, file_data in data['files'].items():
        if 'summary' in file_data:
            summary = file_data['summary']
            total_statements += summary.get('num_statements', 0)
            total_covered += summary.get('covered_lines', 0)

    if total_statements == 0:
        coverage_percent = 0.0
    else:
        coverage_percent = (total_covered / total_statements) * 100

    print("RQA2025项目90%测试覆盖率验证报告")
    print("=" * 50)
    print(f"总语句数: {total_statements}")
    print(f"覆盖语句数: {total_covered}")
    print(f"当前覆盖率: {coverage_percent:.2f}%")
    print(f"90%目标: {'✅ 达成' if coverage_percent >= 90 else '❌ 未达成'}")

    if coverage_percent < 90:
        gap = 90 - coverage_percent
        print(f"覆盖率缺口: {gap:.2f}%")

        needed_statements = int((90 * total_statements / 100) - total_covered)
        print(f"需要额外覆盖语句数: {needed_statements}")


if __name__ == "__main__":
    analyze_coverage()
