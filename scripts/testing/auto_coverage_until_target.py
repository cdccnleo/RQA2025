import time
import subprocess
import re
import sys

TARGET = 85.0  # 目标覆盖率，可根据需要修改
CHECK_INTERVAL = 60  # 1分钟
REPORT_PATH = 'reports/testing/ai_coverage_automation_report.md'
LAYERS = ["infrastructure", "data", "features", "models", "trading", "backtest"]


def parse_coverage_report(report_path):
    """
    解析覆盖率报告，返回各层级当前覆盖率的字典
    """
    coverage = {}
    pattern = re.compile(r'\| (\w+) \| ([\d.]+)% \| ([\d.]+)%')
    with open(report_path, encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                layer, current, target = m.groups()
                if layer in LAYERS:
                    coverage[layer] = float(current)
    return coverage


def all_layers_reached(coverage, target):
    return all(coverage.get(layer, 0) >= target for layer in LAYERS)


if __name__ == "__main__":
    while True:
        print("[自动化] 执行AI覆盖率自动化...")
        result = subprocess.run(
            [sys.executable, "scripts/testing/start_ai_coverage_automation.py", "once"])
        if result.returncode != 0:
            print("[自动化] 自动化执行失败，1分钟后重试...")
            time.sleep(CHECK_INTERVAL)
            continue
        try:
            coverage = parse_coverage_report(REPORT_PATH)
            print(f"[自动化] 当前覆盖率: {coverage}")
            if all_layers_reached(coverage, TARGET):
                print("🎉 所有层级覆盖率已达标，自动退出！")
                break
            else:
                print("[自动化] 覆盖率未达标，1分钟后重试...")
        except Exception as e:
            print(f"[自动化] 解析覆盖率报告失败: {e}，1分钟后重试...")
        time.sleep(CHECK_INTERVAL)
