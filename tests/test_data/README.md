# 测试数据目录

本目录包含RQA2025量化交易系统测试所需的数据文件。

## 目录结构

```
tests/test_data/
├── market_data/          # 市场数据
│   ├── sample_stocks.csv # 示例股票价格数据
│   └── ...
├── risk_data/           # 风险数据
├── strategy_data/       # 策略数据
├── performance_data/    # 性能数据
└── README.md           # 本文件
```

## 数据说明

### market_data/sample_stocks.csv

包含示例股票价格数据的CSV文件，字段包括：
- `symbol`: 股票代码
- `date`: 交易日期
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `returns`: 收益率

涵盖股票：
- AAPL (苹果)
- GOOGL (谷歌)
- MSFT (微软)
- AMZN (亚马逊)

时间范围：2023年1月1日-5日

## 使用方法

### 在测试中使用

```python
import pandas as pd
import os

# 获取测试数据路径
test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
market_data_file = os.path.join(test_data_dir, 'market_data', 'sample_stocks.csv')

# 加载数据
market_data = pd.read_csv(market_data_file, parse_dates=['date'])

# 使用数据进行测试
def test_market_data_loading():
    assert len(market_data) > 0
    assert 'symbol' in market_data.columns
    assert 'close' in market_data.columns
```

### 添加新测试数据

1. 在相应子目录中创建新的数据文件
2. 更新本README文件，说明新数据的用途和格式
3. 在相关的测试文件中引用新数据

## 数据生成脚本

如果需要生成更多测试数据，可以使用以下脚本：

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_stock_data(symbol, start_date, days=252):
    """生成示例股票数据"""
    dates = [start_date + timedelta(days=i) for i in range(days)]

    # 生成价格数据（随机游走）
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)
    prices = 100 * np.exp(np.cumsum(returns))  # 起始价格100

    # 生成OHLC数据
    high_multipliers = 1 + np.random.uniform(0, 0.02, days)
    low_multipliers = 1 - np.random.uniform(0, 0.02, days)
    volume_base = 1000000

    data = []
    for i, date in enumerate(dates):
        open_price = prices[i] * (1 + np.random.normal(0, 0.005))
        close_price = prices[i]
        high_price = close_price * high_multipliers[i]
        low_price = close_price * low_multipliers[i]
        volume = int(volume_base * (1 + np.random.normal(0, 0.3)))

        data.append({
            'symbol': symbol,
            'date': date.strftime('%Y-%m-%d'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'returns': returns[i]
        })

    return pd.DataFrame(data)

# 生成示例数据
start_date = datetime(2023, 1, 1)
sample_data = generate_sample_stock_data('TEST', start_date, days=10)
sample_data.to_csv('tests/test_data/market_data/test_sample.csv', index=False)
```

## 注意事项

1. **数据隐私**: 确保测试数据不包含真实的用户或交易信息
2. **数据质量**: 测试数据应尽可能接近真实数据分布
3. **版本控制**: 重要的测试数据文件应纳入版本控制
4. **更新频率**: 根据业务变化及时更新测试数据

## 相关文档

- [测试框架文档](../../docs/testing/)
- [数据层架构文档](../../docs/architecture/data_layer_architecture_design.md)
- [API文档](../../docs/api/)
