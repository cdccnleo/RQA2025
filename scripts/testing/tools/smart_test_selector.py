"""智能测试选择器 - 根据优先级和失败历史自动选择测试"""
import json
from pathlib import Path
from datetime import datetime
import subprocess


class SmartTestSelector:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.test_history_file = self.project_root / "test_history.json"
        self.priority_config = {
            "infrastructure": {
                "high_priority": ["database", "monitoring", "cache", "error"],
                "medium_priority": ["config", "security", "health"],
                "low_priority": ["utils", "web", "versioning"]
            },
            "features": {
                "high_priority": ["technical", "sentiment", "orderbook"],
                "medium_priority": ["processors", "config"],
                "low_priority": ["utils", "enums"]
            },
            "trading": {
                "high_priority": ["execution", "portfolio", "risk"],
                "medium_priority": ["backtest", "strategy"],
                "low_priority": ["utils", "analysis"]
            }
        }

    def load_test_history(self):
        """加载测试历史"""
        if self.test_history_file.exists():
            with open(self.test_history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_test_history(self, history):
        """保存测试历史"""
        with open(self.test_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def get_failed_tests(self, layer):
        """获取最近失败的测试"""
        history = self.load_test_history()
        layer_history = history.get(layer, {})

        failed_tests = []
        for test_file, test_data in layer_history.items():
            if test_data.get('last_status') == 'failed':
                failed_tests.append(test_file)

        return failed_tests

    def get_priority_tests(self, layer, priority_level="high"):
        """获取优先级测试"""
        if layer not in self.priority_config:
            return []

        priority_modules = self.priority_config[layer].get(f"{priority_level}_priority", [])
        priority_tests = []

        layer_dir = self.project_root / f"tests/unit/{layer}"
        if layer_dir.exists():
            for module in priority_modules:
                module_dir = layer_dir / module
                if module_dir.exists():
                    for test_file in module_dir.glob("test_*.py"):
                        priority_tests.append(str(test_file.relative_to(self.project_root)))

        return priority_tests

    def select_tests(self, layer, max_tests=20, include_failed=True, include_priority=True):
        """智能选择测试用例"""
        selected_tests = []

        # 1. 优先选择失败的测试
        if include_failed:
            failed_tests = self.get_failed_tests(layer)
            selected_tests.extend(failed_tests[:max_tests//2])

        # 2. 选择高优先级测试
        if include_priority and len(selected_tests) < max_tests:
            priority_tests = self.get_priority_tests(layer, "high")
            remaining_slots = max_tests - len(selected_tests)
            selected_tests.extend(priority_tests[:remaining_slots])

        # 3. 如果还不够，选择中等优先级测试
        if len(selected_tests) < max_tests:
            medium_tests = self.get_priority_tests(layer, "medium")
            remaining_slots = max_tests - len(selected_tests)
            selected_tests.extend(medium_tests[:remaining_slots])

        return selected_tests[:max_tests]

    def run_smart_tests(self, layer, max_tests=20, timeout=300):
        """运行智能选择的测试"""
        selected_tests = self.select_tests(layer, max_tests)

        if not selected_tests:
            print(f"未找到{layer}层的测试用例")
            return 1

        print(f"智能选择{layer}层测试用例:")
        for test in selected_tests:
            print(f"  - {test}")

        # 运行选中的测试
        pytest_cmd = [
            "python", "-m", "pytest"
        ] + selected_tests + [
            "-v", "--tb=short", "--maxfail=10",
            f"--timeout={timeout}"
        ]

        print(f"\n执行命令: {' '.join(pytest_cmd)}")
        result = subprocess.run(pytest_cmd, cwd=self.project_root)

        # 更新测试历史
        self.update_test_history(layer, selected_tests, result.returncode == 0)

        return result.returncode

    def update_test_history(self, layer, test_files, success):
        """更新测试历史"""
        history = self.load_test_history()

        if layer not in history:
            history[layer] = {}

        current_time = datetime.now().isoformat()
        status = "passed" if success else "failed"

        for test_file in test_files:
            history[layer][test_file] = {
                "last_status": status,
                "last_run": current_time,
                "run_count": history[layer].get(test_file, {}).get("run_count", 0) + 1
            }

        self.save_test_history(history)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="智能测试选择器")
    parser.add_argument("--layer", required=True,
                        help="测试层 (infrastructure/features/trading/backtest/data)")
    parser.add_argument("--max-tests", type=int, default=20, help="最大测试数量")
    parser.add_argument("--timeout", type=int, default=300, help="超时时间（秒）")
    parser.add_argument("--no-failed", action="store_true", help="不包含失败的测试")
    parser.add_argument("--no-priority", action="store_true", help="不包含优先级测试")

    args = parser.parse_args()

    selector = SmartTestSelector(".")
    exit_code = selector.run_smart_tests(
        args.layer,
        args.max_tests,
        args.timeout
    )

    import sys
    sys.exit(exit_code)
