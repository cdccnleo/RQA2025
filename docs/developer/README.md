# 👨‍💻 RQA2025开发者文档

## 🎯 概述

欢迎来到RQA2025量化交易系统开发者文档！本文档为开发者提供完整的开发指南，包括环境搭建、代码规范、测试策略、扩展开发等全方位开发指导。

## 🏗️ 开发环境搭建

### 1. 系统要求

#### 硬件要求
```yaml
操作系统: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
CPU: 4核以上 (推荐8核+)
内存: 8GB以上 (推荐16GB+)
存储: 50GB可用空间
网络: 稳定的互联网连接
```

#### 软件依赖
```bash
# 必需
Python 3.9+          # 主要开发语言
Docker Desktop       # 容器化开发
Git                  # 版本控制
VS Code / PyCharm    # IDE

# 推荐
PostgreSQL 13+       # 开发数据库
Redis 6+            # 缓存服务
Node.js 16+         # 前端开发 (可选)
```

### 2. 环境初始化

#### 克隆项目
```bash
# HTTPS方式
git clone https://github.com/your-org/rqa2025.git
cd rqa2025

# SSH方式 (推荐)
git clone git@github.com:your-org/rqa2025.git
cd rqa2025

# 初始化子模块 (如果有)
git submodule update --init --recursive
```

#### 创建虚拟环境
```bash
# 使用venv (推荐)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 或使用conda
conda create -n rqa2025 python=3.9
conda activate rqa2025
```

#### 安装依赖
```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt

# 安装预提交钩子
pre-commit install

# 验证安装
python -c "import rqa2025; print('安装成功!')"
```

### 3. 数据库设置

#### PostgreSQL设置
```bash
# 使用Docker运行PostgreSQL (推荐)
docker run -d \
  --name rqa-postgres \
  -e POSTGRES_DB=rqa2025 \
  -e POSTGRES_USER=rqa_user \
  -e POSTGRES_PASSWORD=dev_password \
  -p 5432:5432 \
  postgres:13-alpine

# 或安装本地PostgreSQL
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo -u postgres createuser --createdb rqa_user
sudo -u postgres psql -c "ALTER USER rqa_user PASSWORD 'dev_password';"
createdb -U rqa_user rqa2025
```

#### Redis设置
```bash
# 使用Docker运行Redis
docker run -d \
  --name rqa-redis \
  -p 6379:6379 \
  redis:6-alpine

# 或安装本地Redis
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis-server
```

### 4. 配置开发环境

#### 环境变量配置
```bash
# 创建.env文件
cat > .env << EOF
# 开发环境配置
DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production
DATABASE_URL=postgresql://rqa_user:dev_password@localhost:5432/rqa2025
REDIS_URL=redis://localhost:6379/0

# JWT配置
JWT_SECRET_KEY=dev-jwt-secret-key
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# 邮件配置 (开发环境可选)
MAIL_SERVER=localhost
MAIL_PORT=1025
MAIL_USERNAME=
MAIL_PASSWORD=

# 开发工具配置
FLASK_ENV=development
LOG_LEVEL=DEBUG
EOF
```

#### IDE配置

##### VS Code配置
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.venv": true
  }
}
```

##### PyCharm配置
```
1. File -> Settings -> Project Interpreter -> Add -> Existing environment -> 选择.venv/bin/python
2. File -> Settings -> Tools -> Python Integrated Tools -> Testing -> 选择py.test
3. File -> Settings -> Editor -> Code Style -> 选择Black
4. File -> Settings -> Tools -> File Watchers -> 添加Black和isort
```

### 5. 运行开发服务器

#### 启动后端服务
```bash
# 方式1: 使用Python直接运行
python -m src.main

# 方式2: 使用uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 方式3: 使用Docker Compose
docker-compose -f docker-compose.dev.yml up
```

#### 启动前端服务 (如果有)
```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

#### 验证服务启动
```bash
# 检查API服务
curl http://localhost:8000/health

# 检查数据库连接
python -c "from src.database import db; db.engine.execute('SELECT 1')"

# 检查Redis连接
python -c "from src.cache import redis_client; redis_client.ping()"
```

## 📝 代码规范

### 1. Python代码规范

#### Black代码格式化
```python
# 配置pyproject.toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

#### isort导入排序
```python
# 配置pyproject.toml
[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
line_length = 88
```

#### flake8代码检查
```ini
# .flake8
[flake8]
max-line-length = 88
extend-ignore = E203, E501, W503
exclude =
    .git,
    __pycache__,
    .pytest_cache,
    .venv,
    build,
    dist,
    *.egg-info
```

### 2. 命名规范

#### 类和函数命名
```python
# 类名: PascalCase
class UserManager:
    pass

class TradingStrategy:
    pass

# 函数名: snake_case
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    pass

def validate_market_data(data):
    pass

# 私有方法: 前缀单下划线
def _calculate_volatility(self, prices):
    pass

# 保护方法: 前缀双下划线 (慎用)
def __validate_input(self, input_data):
    pass
```

#### 变量命名
```python
# 常量: UPPER_SNAKE_CASE
MAX_POSITION_SIZE = 0.1
DEFAULT_COMMISSION = 0.001

# 变量: snake_case
user_balance = 10000.0
trading_signals = []
market_data = {}

# 布尔变量: 前缀is/has/can等
is_market_open = True
has_pending_orders = False
can_execute_trade = True
```

### 3. 文档规范

#### 函数文档
```python
def calculate_portfolio_return(
    positions: List[Position],
    current_prices: Dict[str, float],
    include_dividends: bool = True
) -> float:
    """
    计算投资组合收益率

    根据当前持仓和最新价格计算投资组合的总收益率。
    可以选择是否包含股息收入。

    Args:
        positions: 持仓列表
        current_prices: 当前价格字典，key为股票代码，value为价格
        include_dividends: 是否包含股息收入，默认为True

    Returns:
        投资组合收益率 (百分比)

    Raises:
        ValueError: 当持仓数据无效时抛出
        KeyError: 当缺少必要的价格数据时抛出

    Examples:
        >>> positions = [Position('AAPL', 100, 150.0)]
        >>> prices = {'AAPL': 160.0}
        >>> calculate_portfolio_return(positions, prices)
        6.67
    """
    if not positions:
        return 0.0

    total_cost = sum(pos.quantity * pos.avg_cost for pos in positions)
    total_value = sum(
        pos.quantity * current_prices.get(pos.symbol, pos.avg_cost)
        for pos in positions
    )

    if include_dividends:
        # 计算股息收入逻辑
        pass

    return ((total_value - total_cost) / total_cost) * 100
```

#### 类文档
```python
class MovingAverageStrategy(BaseStrategy):
    """
    移动平均交叉策略

    该策略基于短期和长期移动平均线的交叉来生成交易信号。
    当短期均线上穿长期均线时产生买入信号，反之产生卖出信号。

    Attributes:
        short_window: 短期均线周期
        long_window: 长期均线周期
        position_size: 每次交易的头寸大小

    Examples:
        >>> strategy = MovingAverageStrategy(
        ...     symbol='AAPL',
        ...     short_window=5,
        ...     long_window=20,
        ...     position_size=100
        ... )
        >>> signals = strategy.generate_signals(market_data)
    """

    def __init__(
        self,
        symbol: str,
        short_window: int = 5,
        long_window: int = 20,
        position_size: int = 100
    ):
        """
        初始化移动平均策略

        Args:
            symbol: 交易标的代码
            short_window: 短期均线周期，默认5
            long_window: 长期均线周期，默认20
            position_size: 每次交易的头寸大小，默认100
        """
        super().__init__(symbol)
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
```

## 🧪 测试策略

### 1. 测试框架

#### pytest配置
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --strict-config
    --disable-warnings
    --tb=short
    -v
markers =
    unit: 单元测试
    integration: 集成测试
    slow: 慢速测试
    database: 数据库相关测试
    redis: Redis相关测试
```

#### 测试目录结构
```
tests/
├── unit/                    # 单元测试
│   ├── test_strategies.py   # 策略测试
│   ├── test_orders.py       # 订单测试
│   └── test_risk.py         # 风险管理测试
├── integration/             # 集成测试
│   ├── test_trading_flow.py # 交易流程测试
│   └── test_api_endpoints.py # API端点测试
├── fixtures/                # 测试夹具
│   ├── conftest.py          # pytest配置
│   ├── market_data.json     # 市场数据样本
│   └── sample_orders.json   # 订单样本
└── performance/             # 性能测试
    ├── test_throughput.py   # 吞吐量测试
    └── test_latency.py      # 延迟测试
```

### 2. 编写测试

#### 单元测试示例
```python
import pytest
from unittest.mock import Mock, patch
from rqa2025.strategies.moving_average import MovingAverageStrategy
from rqa2025.core.market_data import MarketData


class TestMovingAverageStrategy:
    """移动平均策略单元测试"""

    @pytest.fixture
    def strategy(self):
        """测试夹具：创建策略实例"""
        return MovingAverageStrategy(
            symbol='AAPL',
            short_window=5,
            long_window=20,
            position_size=100
        )

    @pytest.fixture
    def sample_market_data(self):
        """测试夹具：创建市场数据样本"""
        return MarketData(
            symbol='AAPL',
            timestamp='2024-01-01T10:00:00Z',
            open=150.0,
            high=155.0,
            low=149.0,
            close=152.0,
            volume=1000000
        )

    def test_initialization(self, strategy):
        """测试策略初始化"""
        assert strategy.symbol == 'AAPL'
        assert strategy.short_window == 5
        assert strategy.long_window == 20
        assert strategy.position_size == 100

    def test_generate_signals_insufficient_data(self, strategy):
        """测试数据不足时的信号生成"""
        # 少于long_window的数据
        prices = [150.0, 151.0, 152.0]

        with patch.object(strategy, '_get_historical_prices', return_value=prices):
            signals = strategy.generate_signals(None)
            assert len(signals) == 0

    def test_generate_buy_signal(self, strategy):
        """测试买入信号生成"""
        # 创建金叉场景的数据
        prices = [140.0] * 20 + [145.0, 146.0, 147.0, 148.0, 149.0]  # 短期均线上穿长期均线

        with patch.object(strategy, '_get_historical_prices', return_value=prices):
            signals = strategy.generate_signals(None)

            assert len(signals) == 1
            signal = signals[0]
            assert signal.signal_type == 'BUY'
            assert signal.symbol == 'AAPL'
            assert signal.quantity == 100

    def test_generate_sell_signal(self, strategy):
        """测试卖出信号生成"""
        # 创建死叉场景的数据
        prices = [160.0] * 20 + [155.0, 154.0, 153.0, 152.0, 151.0]  # 短期均线下穿长期均线

        with patch.object(strategy, '_get_historical_prices', return_value=prices):
            signals = strategy.generate_signals(None)

            assert len(signals) == 1
            signal = signals[0]
            assert signal.signal_type == 'SELL'
            assert signal.symbol == 'AAPL'
            assert signal.quantity == 100

    @pytest.mark.parametrize("invalid_window", [-1, 0, 1000])
    def test_invalid_window_parameters(self, invalid_window):
        """测试无效的窗口参数"""
        with pytest.raises(ValueError):
            MovingAverageStrategy(
                symbol='AAPL',
                short_window=invalid_window,
                long_window=20
            )

    def test_calculate_moving_averages(self, strategy):
        """测试移动平均计算"""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]

        short_ma, long_ma = strategy._calculate_moving_averages(prices)

        # 短期均线基于最后short_window个价格
        expected_short_ma = sum(prices[-5:]) / 5
        assert short_ma == pytest.approx(expected_short_ma)

        # 长期均线基于所有价格 (不足时使用全部)
        expected_long_ma = sum(prices) / len(prices)
        assert long_ma == pytest.approx(expected_long_ma)
```

#### 集成测试示例
```python
import pytest
from httpx import AsyncClient
from rqa2025.main import app
from rqa2025.database import get_db
from rqa2025.cache import get_redis


@pytest.mark.asyncio
class TestTradingFlowIntegration:
    """交易流程集成测试"""

    @pytest.fixture
    async def client(self):
        """测试客户端"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            yield client

    @pytest.fixture
    async def authenticated_client(self, client):
        """认证后的客户端"""
        # 登录获取token
        response = await client.post(
            "/api/v1/auth/login",
            json={
                "username": "test_trader",
                "password": "test_password"
            }
        )
        token = response.json()["data"]["access_token"]

        # 设置认证头
        client.headers["Authorization"] = f"Bearer {token}"
        return client

    async def test_complete_trading_flow(self, authenticated_client):
        """测试完整的交易流程"""
        # 1. 创建策略
        strategy_response = await authenticated_client.post(
            "/api/v1/strategies",
            json={
                "name": "测试策略",
                "type": "moving_average",
                "symbol": "AAPL",
                "parameters": {
                    "short_window": 5,
                    "long_window": 20
                }
            }
        )
        assert strategy_response.status_code == 201
        strategy_id = strategy_response.json()["data"]["id"]

        # 2. 启动策略
        start_response = await authenticated_client.post(
            f"/api/v1/strategies/{strategy_id}/start",
            json={"account_id": "test_account"}
        )
        assert start_response.status_code == 200

        # 3. 检查策略状态
        status_response = await authenticated_client.get(
            f"/api/v1/strategies/{strategy_id}"
        )
        assert status_response.status_code == 200
        assert status_response.json()["data"]["status"] == "running"

        # 4. 执行回测
        backtest_response = await authenticated_client.post(
            "/api/v1/backtests",
            json={
                "strategy_id": strategy_id,
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "initial_balance": 10000.0
            }
        )
        assert backtest_response.status_code == 201

        # 5. 检查回测结果
        backtest_id = backtest_response.json()["data"]["backtest_id"]

        # 等待回测完成 (在实际测试中可能需要重试)
        result_response = await authenticated_client.get(
            f"/api/v1/backtests/{backtest_id}"
        )
        assert result_response.status_code == 200

        result = result_response.json()["data"]
        assert "performance" in result
        assert "total_return" in result["performance"]

        # 6. 停止策略
        stop_response = await authenticated_client.post(
            f"/api/v1/strategies/{strategy_id}/stop"
        )
        assert stop_response.status_code == 200

    async def test_error_handling(self, authenticated_client):
        """测试错误处理"""
        # 测试无效策略ID
        response = await authenticated_client.post(
            "/api/v1/strategies/invalid_id/start"
        )
        assert response.status_code == 404

        # 测试无效参数
        response = await authenticated_client.post(
            "/api/v1/strategies",
            json={
                "name": "",
                "type": "invalid_type"
            }
        )
        assert response.status_code == 422
```

### 3. 测试运行

#### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/unit/test_strategies.py

# 运行特定测试类
pytest tests/unit/test_strategies.py::TestMovingAverageStrategy

# 运行特定测试方法
pytest tests/unit/test_strategies.py::TestMovingAverageStrategy::test_initialization

# 运行标记的测试
pytest -m "unit and not slow"

# 生成覆盖率报告
pytest --cov=rqa2025 --cov-report=html --cov-report=term-missing
```

#### 测试配置
```python
# tests/conftest.py
import pytest
import asyncio
from rqa2025.database import get_db
from rqa2025.cache import get_redis


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db():
    """数据库连接"""
    db = get_db()
    yield db
    # 清理测试数据
    await db.execute("DELETE FROM test_data")


@pytest.fixture(scope="session")
async def redis():
    """Redis连接"""
    redis = get_redis()
    yield redis
    # 清理测试数据
    await redis.flushdb()


@pytest.fixture
def sample_market_data():
    """市场数据样本"""
    return {
        "symbol": "AAPL",
        "timestamp": "2024-01-01T10:00:00Z",
        "open": 150.0,
        "high": 155.0,
        "low": 149.0,
        "close": 152.0,
        "volume": 1000000
    }


@pytest.fixture
def sample_strategy_config():
    """策略配置样本"""
    return {
        "name": "测试策略",
        "type": "moving_average",
        "symbol": "AAPL",
        "parameters": {
            "short_window": 5,
            "long_window": 20
        }
    }
```

## 🔧 扩展开发

### 1. 创建自定义策略

#### 策略基类
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
from rqa2025.core.market_data import MarketData
from rqa2025.core.signals import Signal


@dataclass
class StrategyConfig:
    """策略配置"""
    symbol: str
    parameters: Dict[str, Any]
    risk_limits: Dict[str, Any]


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.symbol = config.symbol
        self.parameters = config.parameters
        self.risk_limits = config.risk_limits

        # 策略状态
        self.is_active = False
        self.last_signal_time = None
        self.performance_metrics = {}

    @abstractmethod
    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        生成交易信号

        Args:
            market_data: 市场数据

        Returns:
            信号列表
        """
        pass

    @abstractmethod
    def validate_parameters(self) -> bool:
        """验证策略参数"""
        pass

    def start(self) -> bool:
        """启动策略"""
        if not self.validate_parameters():
            return False

        self.is_active = True
        return True

    def stop(self) -> bool:
        """停止策略"""
        self.is_active = False
        return True

    def update_performance(self, metrics: Dict[str, Any]):
        """更新性能指标"""
        self.performance_metrics.update(metrics)

    def get_status(self) -> Dict[str, Any]:
        """获取策略状态"""
        return {
            "is_active": self.is_active,
            "symbol": self.symbol,
            "last_signal_time": self.last_signal_time,
            "performance": self.performance_metrics
        }
```

#### 自定义策略实现
```python
from rqa2025.strategies.base import BaseStrategy, StrategyConfig
from rqa2025.core.signals import Signal, SignalType
import numpy as np


class CustomStrategy(BaseStrategy):
    """自定义策略示例"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        # 策略特定参数
        self.fast_period = self.parameters.get('fast_period', 12)
        self.slow_period = self.parameters.get('slow_period', 26)
        self.signal_period = self.parameters.get('signal_period', 9)

        # 状态变量
        self.price_history = []
        self.macd_history = []
        self.signal_history = []

    def validate_parameters(self) -> bool:
        """验证参数"""
        if not (5 <= self.fast_period < self.slow_period <= 100):
            return False
        if not (5 <= self.signal_period <= 20):
            return False
        return True

    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """生成信号"""
        signals = []

        # 更新价格历史
        self.price_history.append(market_data.close)

        # 保持历史长度
        max_length = max(self.slow_period * 2, 100)
        if len(self.price_history) > max_length:
            self.price_history.pop(0)

        # 计算MACD
        if len(self.price_history) >= self.slow_period:
            macd, signal = self._calculate_macd()

            # 生成交易信号
            if self._should_buy(macd, signal):
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.BUY,
                    quantity=self._calculate_position_size(market_data),
                    price=market_data.close,
                    timestamp=market_data.timestamp,
                    metadata={
                        'strategy': 'custom_macd',
                        'macd': macd,
                        'signal': signal
                    }
                ))

            elif self._should_sell(macd, signal):
                signals.append(Signal(
                    symbol=self.symbol,
                    signal_type=SignalType.SELL,
                    quantity=self._calculate_position_size(market_data),
                    price=market_data.close,
                    timestamp=market_data.timestamp,
                    metadata={
                        'strategy': 'custom_macd',
                        'macd': macd,
                        'signal': signal
                    }
                ))

        return signals

    def _calculate_macd(self):
        """计算MACD指标"""
        prices = np.array(self.price_history)

        # 计算EMA
        fast_ema = self._calculate_ema(prices, self.fast_period)
        slow_ema = self._calculate_ema(prices, self.slow_period)

        # 计算MACD线
        macd_line = fast_ema - slow_ema

        # 计算信号线
        signal_line = self._calculate_ema(
            np.array([macd_line] if np.isscalar(macd_line) else macd_line),
            self.signal_period
        )

        return macd_line, signal_line

    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """计算指数移动平均"""
        if len(data) < period:
            return np.mean(data)

        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])

        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _should_buy(self, macd: float, signal: float) -> bool:
        """判断是否应该买入"""
        # MACD线上穿信号线
        if len(self.macd_history) >= 2 and len(self.signal_history) >= 2:
            prev_macd = self.macd_history[-2]
            prev_signal = self.signal_history[-2]

            # 金叉信号
            if prev_macd <= prev_signal and macd > signal:
                return True

        return False

    def _should_sell(self, macd: float, signal: float) -> bool:
        """判断是否应该卖出"""
        # MACD线下穿信号线
        if len(self.macd_history) >= 2 and len(self.signal_history) >= 2:
            prev_macd = self.macd_history[-2]
            prev_signal = self.signal_history[-2]

            # 死叉信号
            if prev_macd >= prev_signal and macd < signal:
                return True

        return False

    def _calculate_position_size(self, market_data: MarketData) -> int:
        """计算仓位大小"""
        # 基于风险管理的仓位计算
        account_balance = 10000.0  # 从账户获取
        risk_per_trade = self.risk_limits.get('risk_per_trade', 0.01)  # 1%风险

        # 基于波动率的止损距离
        stop_loss_pct = self.risk_limits.get('stop_loss_pct', 0.02)  # 2%止损
        risk_amount = account_balance * risk_per_trade

        # 计算最大仓位
        max_position_value = risk_amount / stop_loss_pct
        position_size = max_position_value / market_data.close

        # 应用最大仓位限制
        max_position_pct = self.risk_limits.get('max_position_pct', 0.1)  # 10%
        max_position_from_limit = (account_balance * max_position_pct) / market_data.close

        return int(min(position_size, max_position_from_limit))
```

#### 注册自定义策略
```python
from rqa2025.strategies.registry import StrategyRegistry
from rqa2025.strategies.custom_strategy import CustomStrategy

# 注册策略
registry = StrategyRegistry()
registry.register('custom_macd', CustomStrategy)

# 使用策略
config = StrategyConfig(
    symbol='TSLA',
    parameters={
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    },
    risk_limits={
        'risk_per_trade': 0.01,
        'stop_loss_pct': 0.02,
        'max_position_pct': 0.1
    }
)

strategy = registry.create('custom_macd', config)
```

### 2. 创建数据适配器

#### 数据适配器接口
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from rqa2025.core.market_data import MarketData


class DataAdapter(ABC):
    """数据适配器基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """连接数据源"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """断开连接"""
        pass

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d'
    ) -> List[MarketData]:
        """获取历史数据"""
        pass

    @abstractmethod
    async def get_real_time_data(self, symbol: str) -> Optional[MarketData]:
        """获取实时数据"""
        pass

    @abstractmethod
    async def subscribe_real_time(self, symbol: str, callback) -> bool:
        """订阅实时数据"""
        pass

    @abstractmethod
    async def unsubscribe_real_time(self, symbol: str) -> bool:
        """取消订阅实时数据"""
        pass

    async def health_check(self) -> bool:
        """健康检查"""
        return self.is_connected
```

#### Alpha Vantage适配器实现
```python
import aiohttp
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from rqa2025.core.market_data import MarketData
from rqa2025.adapters.base import DataAdapter


class AlphaVantageAdapter(DataAdapter):
    """Alpha Vantage数据适配器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key']
        self.base_url = 'https://www.alphavantage.co/query'
        self.session: Optional[aiohttp.ClientSession] = None
        self.subscriptions: Dict[str, Callable] = {}

    async def connect(self) -> bool:
        """连接到Alpha Vantage"""
        try:
            self.session = aiohttp.ClientSession()
            self.is_connected = True
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    async def disconnect(self) -> bool:
        """断开连接"""
        if self.session:
            await self.session.close()
        self.is_connected = False
        return True

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1d'
    ) -> List[MarketData]:
        """获取历史数据"""
        if not self.session:
            raise ConnectionError("未连接到数据源")

        # 转换时间间隔
        av_interval = self._convert_interval(interval)

        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full'
        }

        async with self.session.get(self.base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"API请求失败: {response.status}")

            data = await response.json()

            if 'Error Message' in data:
                raise Exception(f"API错误: {data['Error Message']}")

            # 解析数据
            time_series = data.get('Time Series (Daily)', {})
            market_data = []

            for date_str, values in time_series.items():
                date = datetime.strptime(date_str, '%Y-%m-%d')

                # 检查日期范围
                if start_date <= date <= end_date:
                    market_data.append(MarketData(
                        symbol=symbol,
                        timestamp=date.isoformat(),
                        open=float(values['1. open']),
                        high=float(values['2. high']),
                        low=float(values['3. low']),
                        close=float(values['4. close']),
                        volume=int(float(values['5. volume']))
                    ))

            # 按时间排序
            market_data.sort(key=lambda x: x.timestamp)
            return market_data

    async def get_real_time_data(self, symbol: str) -> Optional[MarketData]:
        """获取实时数据"""
        if not self.session:
            raise ConnectionError("未连接到数据源")

        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }

        async with self.session.get(self.base_url, params=params) as response:
            if response.status != 200:
                return None

            data = await response.json()

            if 'Global Quote' not in data:
                return None

            quote = data['Global Quote']

            return MarketData(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                open=float(quote.get('02. open', 0)),
                high=float(quote.get('03. high', 0)),
                low=float(quote.get('04. low', 0)),
                close=float(quote.get('05. price', 0)),
                volume=int(float(quote.get('06. volume', 0)))
            )

    async def subscribe_real_time(self, symbol: str, callback: Callable) -> bool:
        """订阅实时数据"""
        # Alpha Vantage不支持WebSocket，这里使用轮询方式模拟
        self.subscriptions[symbol] = callback

        # 启动轮询任务
        asyncio.create_task(self._poll_real_time_data(symbol))
        return True

    async def unsubscribe_real_time(self, symbol: str) -> bool:
        """取消订阅"""
        if symbol in self.subscriptions:
            del self.subscriptions[symbol]
            return True
        return False

    async def _poll_real_time_data(self, symbol: str):
        """轮询实时数据"""
        while symbol in self.subscriptions:
            try:
                data = await self.get_real_time_data(symbol)
                if data:
                    await self.subscriptions[symbol](data)
            except Exception as e:
                print(f"获取实时数据失败: {e}")

            await asyncio.sleep(60)  # 每分钟更新一次

    def _convert_interval(self, interval: str) -> str:
        """转换时间间隔"""
        mapping = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '60min',
            '1d': 'daily',
            '1w': 'weekly'
        }
        return mapping.get(interval, 'daily')
```

#### 注册数据适配器
```python
from rqa2025.adapters.registry import AdapterRegistry
from rqa2025.adapters.alpha_vantage import AlphaVantageAdapter

# 注册适配器
registry = AdapterRegistry()
registry.register('alpha_vantage', AlphaVantageAdapter)

# 配置并使用
config = {
    'api_key': 'your_alpha_vantage_api_key'
}

adapter = registry.create('alpha_vantage', config)
await adapter.connect()

# 获取数据
historical_data = await adapter.get_historical_data(
    'AAPL',
    datetime(2023, 1, 1),
    datetime(2023, 12, 31)
)
```

### 3. 贡献代码

#### 贡献流程
```bash
# 1. Fork项目到自己的GitHub账户

# 2. 克隆Fork的项目
git clone https://github.com/your-username/rqa2025.git
cd rqa2025

# 3. 创建功能分支
git checkout -b feature/your-feature-name

# 4. 安装开发依赖并运行测试
pip install -r requirements-dev.txt
pytest tests/unit/

# 5. 提交更改
git add .
git commit -m "feat: add your feature description"

# 6. 推送分支
git push origin feature/your-feature-name

# 7. 创建Pull Request
# 在GitHub上创建PR，描述你的更改和动机
```

#### 提交规范
```bash
# 格式: <type>(<scope>): <subject>

# 类型
feat: 新功能
fix: 修复bug
docs: 文档更新
style: 代码格式调整
refactor: 代码重构
test: 测试相关
chore: 构建过程或工具配置

# 示例
git commit -m "feat(strategies): add RSI divergence strategy"
git commit -m "fix(api): resolve authentication token refresh issue"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(backtest): add performance regression tests"
```

#### 代码审查 checklist
- [ ] 代码符合项目规范 (Black, isort, flake8)
- [ ] 添加了相应的单元测试
- [ ] 更新了相关文档
- [ ] 没有破坏现有功能
- [ ] 性能测试通过
- [ ] 安全性检查通过

### 4. 插件开发

#### 插件接口
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from rqa2025.core.events import Event


class Plugin(ABC):
    """插件基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.enabled = True

    @abstractmethod
    def get_hooks(self) -> Dict[str, callable]:
        """定义插件钩子"""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """插件初始化"""
        pass

    @abstractmethod
    async def cleanup(self) -> bool:
        """插件清理"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return {
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
            'description': self.get_description()
        }

    def get_description(self) -> str:
        """获取插件描述"""
        return "插件描述"
```

#### 监控插件示例
```python
from rqa2025.plugins.base import Plugin
from rqa2025.monitoring.metrics import MetricsCollector
import psutil
import time


class SystemMonitorPlugin(Plugin):
    """系统监控插件"""

    def __init__(self, config):
        super().__init__(config)
        self.collect_interval = config.get('collect_interval', 60)
        self.metrics_collector = MetricsCollector()

    def get_hooks(self):
        return {
            'app_startup': self.on_app_startup,
            'app_shutdown': self.on_app_shutdown,
            'before_request': self.on_before_request,
            'after_request': self.on_after_request
        }

    async def initialize(self) -> bool:
        """初始化插件"""
        # 启动监控任务
        asyncio.create_task(self.collect_system_metrics())
        return True

    async def cleanup(self) -> bool:
        """清理插件"""
        # 停止监控任务
        return True

    def get_description(self) -> str:
        return "系统性能监控插件，收集CPU、内存、磁盘等指标"

    async def on_app_startup(self, app):
        """应用启动钩子"""
        await self.metrics_collector.record_event('app_startup', {
            'timestamp': time.time(),
            'version': app.version
        })

    async def on_app_shutdown(self, app):
        """应用关闭钩子"""
        await self.metrics_collector.record_event('app_shutdown', {
            'timestamp': time.time(),
            'uptime': time.time() - app.start_time
        })

    async def on_before_request(self, request):
        """请求前钩子"""
        request.start_time = time.time()

    async def on_after_request(self, request, response):
        """请求后钩子"""
        duration = time.time() - request.start_time

        await self.metrics_collector.record_request(
            method=request.method,
            url=str(request.url),
            status=response.status_code,
            duration=duration
        )

    async def collect_system_metrics(self):
        """收集系统指标"""
        while self.enabled:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)

                # 内存使用
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used = memory.used

                # 磁盘使用
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent

                # 网络IO
                network = psutil.net_io_counters()
                bytes_sent = network.bytes_sent
                bytes_recv = network.bytes_recv

                # 记录指标
                await self.metrics_collector.record_gauge('system.cpu_percent', cpu_percent)
                await self.metrics_collector.record_gauge('system.memory_percent', memory_percent)
                await self.metrics_collector.record_gauge('system.memory_used', memory_used)
                await self.metrics_collector.record_gauge('system.disk_percent', disk_percent)
                await self.metrics_collector.record_counter('system.network_bytes_sent', bytes_sent)
                await self.metrics_collector.record_counter('system.network_bytes_recv', bytes_recv)

            except Exception as e:
                print(f"收集系统指标失败: {e}")

            await asyncio.sleep(self.collect_interval)
```

#### 插件管理系统
```python
class PluginManager:
    """插件管理器"""

    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_dir = Path("plugins")

    async def load_plugins(self):
        """加载所有插件"""
        if not self.plugin_dir.exists():
            return

        for plugin_file in self.plugin_dir.glob("*.py"):
            try:
                # 动态导入插件
                module_name = f"plugins.{plugin_file.stem}"
                module = importlib.import_module(module_name)

                # 查找插件类
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and
                        issubclass(attr, Plugin) and
                        attr != Plugin):
                        # 实例化插件
                        plugin_config = self.load_plugin_config(plugin_file.stem)
                        plugin = attr(plugin_config)

                        # 初始化插件
                        if await plugin.initialize():
                            self.plugins[plugin.name] = plugin
                            print(f"插件 {plugin.name} 加载成功")
                        else:
                            print(f"插件 {plugin.name} 初始化失败")

            except Exception as e:
                print(f"加载插件 {plugin_file.name} 失败: {e}")

    async def execute_hook(self, hook_name: str, *args, **kwargs):
        """执行插件钩子"""
        results = []
        for plugin in self.plugins.values():
            if plugin.enabled and hook_name in plugin.get_hooks():
                try:
                    hook_func = plugin.get_hooks()[hook_name]
                    result = await hook_func(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"执行插件 {plugin.name} 钩子 {hook_name} 失败: {e}")

        return results

    def load_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """加载插件配置"""
        config_file = self.plugin_dir / f"{plugin_name}.yaml"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """获取所有插件信息"""
        return [plugin.get_info() for plugin in self.plugins.values()]

    async def enable_plugin(self, plugin_name: str) -> bool:
        """启用插件"""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            plugin.enabled = True
            return True
        return False

    async def disable_plugin(self, plugin_name: str) -> bool:
        """禁用插件"""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            plugin.enabled = False
            await plugin.cleanup()
            return True
        return False
```

---

**🚀 RQA2025 开发者文档 - 让贡献变得简单而规范！**

**💡 提示**: 优秀的代码不仅需要技术，更需要良好的文档和规范。让我们一起构建更好的量化交易系统！

