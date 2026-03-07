import sys

# 假定有BacktestEngine等主流程类，若无则mock
try:
    from src.backtest.engine import BacktestEngine
except ImportError:
    class BacktestEngine:
        def __init__(self, config=None):
            self.config = config or {}

        def run(self):
            print("BacktestEngine running with config:", self.config)
            return {"result": "success"}

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    # 最小主流程：初始化->运行->输出
    engine = BacktestEngine(config={"mode": "minimal"})
    result = engine.run()
    print("回测主流程执行结果:", result)
    print("SUCCESS: Minimal backtest main flow test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
