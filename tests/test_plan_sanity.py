# 时间感知智能调度与 parse_rate_limit 加固 - 计划实施后的快速校验
from src.gateway.web.data_collectors import parse_rate_limit

# parse_rate_limit 单元校验
for s, expect_min in [("100次/分钟", 5), ("10次/分钟", 5), ("按协议", 30), ("无限制", 5), ("1次/小时", 5), ("x", 60)]:
    v = parse_rate_limit(s)
    assert v >= 5 or (s == "x" and v == 60), f"{s} -> {v}"
print("parse_rate_limit: OK")

from src.core.orchestration.business_process.service_scheduler import DataCollectionServiceScheduler
from datetime import datetime

s = DataCollectionServiceScheduler()
# 时间乘数
mult, phase = s._get_trading_hours_interval_multiplier(datetime(2025, 1, 20, 9, 35))  # 周一 9:35 交易
assert phase == "trading" and mult == 1.3, f"trading: {phase},{mult}"
mult, phase = s._get_trading_hours_interval_multiplier(datetime(2025, 1, 20, 12, 0))  # 午休
assert phase == "off_hours" and mult == 0.75, f"off_hours: {phase},{mult}"
mult, phase = s._get_trading_hours_interval_multiplier(datetime(2025, 1, 18, 10, 0))  # 周六
assert phase == "weekend" and mult == 1.5, f"weekend: {phase},{mult}"
print("_get_trading_hours_interval_multiplier: OK")
# 间隔调整使用 MIN/MAX
iv = s._adjust_interval_intelligent(10.0, "medium", "test")
assert 30 <= iv <= 86400, f"interval {iv}"
print("_adjust_interval_intelligent MIN/MAX: OK")
print("All checks passed.")
