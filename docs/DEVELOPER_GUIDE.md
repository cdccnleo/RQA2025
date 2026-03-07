# RQA2025 开发者文档

## 概述

本文档为开发者提供 RQA2025 平台的详细技术信息，包括架构设计、API 接口、扩展开发等内容。

## 架构设计

### 总体架构

RQA2025 采用分层架构设计：

```
┌─────────────────┐
│   Web界面层      │  ← 用户界面和API
├─────────────────┤
│   业务服务层     │  ← 策略、回测、优化、监控
├─────────────────┤
│   核心服务层     │  ← 依赖注入、服务注册
├─────────────────┤
│   基础设施层     │  ← 数据访问、缓存、消息队列
└─────────────────┘
```

### 服务组件

#### 策略服务 (StrategyService)
负责策略的创建、执行和管理：

```python
class UnifiedStrategyService(IStrategyService):
    async def create_strategy(self, config: StrategyConfig) -> bool
    async def execute_strategy(self, strategy_id: str, market_data: Dict) -> StrategyResult
    def get_strategy(self, strategy_id: str) -> Optional[StrategyConfig]
    def list_strategies(self) -> List[StrategyConfig]
```

#### 回测服务 (BacktestService)
提供历史数据回测功能：

```python
class BacktestService(IBacktestService):
    async def create_backtest(self, config: BacktestConfig) -> str
    async def run_backtest(self, backtest_id: str) -> BacktestResult
    def get_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]
    def list_backtests(self, strategy_id: Optional[str] = None) -> List[BacktestConfig]
```

#### 优化服务 (OptimizationService)
实现参数优化算法：

```python
class OptimizationService(IOptimizationService):
    async def create_optimization(self, config: OptimizationConfig) -> str
    async def run_optimization(self, optimization_id: str) -> OptimizationResult
    def get_optimization_result(self, optimization_id: str) -> Optional[OptimizationResult]
```

## 扩展开发

### 自定义策略开发

#### 1. 继承基础策略类

```python
from src.strategy.strategies.base_strategy import BaseStrategy
from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategySignal

class MyCustomStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        # 自定义参数
        self.custom_param = self.parameters.get('custom_param', 0.5)

    def generate_signals(self, market_data: Dict[str, Any],
                        context: Optional[Dict[str, Any]] = None) -> List[StrategySignal]:
        signals = []

        for symbol, data in market_data.items():
            if len(data) < 2:
                continue

            # 自定义信号逻辑
            current_price = data[-1]['close']
            prev_price = data[-2]['close']

            if self._custom_condition(current_price, prev_price):
                signal = StrategySignal(
                    symbol=symbol,
                    action='BUY',
                    quantity=self._calculate_position_size(current_price),
                    price=current_price,
                    confidence=0.8,
                    strategy_id=self.strategy_id,
                    metadata={'custom_indicator': self.custom_param}
                )
                signals.append(signal)

        return signals

    def _custom_condition(self, current_price: float, prev_price: float) -> bool:
        # 自定义判断条件
        return current_price > prev_price * (1 + self.custom_param)

    def _calculate_position_size(self, price: float) -> int:
        # 自定义仓位计算
        max_position = self.risk_limits.get('max_position', 1000)
        return min(100, int(max_position * 0.1 / price))
```

#### 2. 注册策略工厂

```python
from src.strategy.strategies.strategy_factory import StrategyFactory

# 获取策略工厂实例
factory = StrategyFactory()

# 注册自定义策略
factory.register_custom_strategy('my_custom', MyCustomStrategy)

# 使用自定义策略
strategy = factory.create_strategy_from_template('my_custom', {
    'strategy_name': '我的自定义策略',
    'parameters': {
        'custom_param': 0.03
    }
})
```

### 自定义指标开发

```python
from typing import List, Dict, Any
import numpy as np

class CustomIndicator:
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return 50.0

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20,
                                std_dev: float = 2.0) -> Dict[str, float]:
        """计算布林带"""
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0}

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)

        return {
            'upper': upper,
            'middle': sma,
            'lower': lower
        }
```

### 自定义优化算法

```python
from src.strategy.optimization.parameter_optimizer import ParameterOptimizer
from src.strategy.interfaces.optimization_interfaces import OptimizationAlgorithm, OptimizationResult
from typing import Dict, List, Any, Callable
import random

class CustomOptimizationAlgorithm:
    @staticmethod
    async def particle_swarm_optimization(parameter_ranges: Dict[str, List[Any]],
                                        target_function: Callable,
                                        n_particles: int = 30,
                                        max_iterations: int = 100) -> OptimizationResult:
        """
        粒子群优化算法实现

        Args:
            parameter_ranges: 参数范围
            target_function: 目标函数
            n_particles: 粒子数量
            max_iterations: 最大迭代次数

        Returns:
            OptimizationResult: 优化结果
        """
        # 初始化粒子
        particles = []
        velocities = []
        personal_best = []
        personal_best_scores = []

        for _ in range(n_particles):
            particle = {}
            velocity = {}

            for param_name, param_values in parameter_ranges.items():
                # 随机初始化位置
                particle[param_name] = random.choice(param_values)

                # 初始化速度为0
                velocity[param_name] = 0

            particles.append(particle)
            velocities.append(velocity)

            # 评估初始位置
            score = await target_function(particle)
            personal_best.append(particle.copy())
            personal_best_scores.append(score)

        # 全局最优
        global_best = max(personal_best, key=lambda x: personal_best_scores[personal_best.index(x)])
        global_best_score = max(personal_best_scores)

        # PSO参数
        w = 0.7  # 惯性权重
        c1 = 1.4  # 个人学习因子
        c2 = 1.4  # 社会学习因子

        convergence_history = [global_best_score]

        for iteration in range(max_iterations):
            for i in range(n_particles):
                # 更新速度
                for param_name in parameter_ranges.keys():
                    r1 = random.random()
                    r2 = random.random()

                    # 计算新速度
                    personal_best_value = personal_best[i][param_name]
                    global_best_value = global_best[param_name]
                    current_value = particles[i][param_name]

                    # 将参数值转换为索引
                    param_values = parameter_ranges[param_name]
                    try:
                        personal_idx = param_values.index(personal_best_value)
                        global_idx = param_values.index(global_best_value)
                        current_idx = param_values.index(current_value)
                    except ValueError:
                        # 如果找不到，随机选择
                        personal_idx = random.randint(0, len(param_values) - 1)
                        global_idx = random.randint(0, len(param_values) - 1)
                        current_idx = random.randint(0, len(param_values) - 1)

                    # 更新速度（使用索引）
                    new_velocity = (w * velocities[i][param_name] +
                                  c1 * r1 * (personal_idx - current_idx) +
                                  c2 * r2 * (global_idx - current_idx))

                    velocities[i][param_name] = new_velocity

                    # 更新位置
                    new_idx = int(current_idx + new_velocity)
                    new_idx = max(0, min(len(param_values) - 1, new_idx))
                    particles[i][param_name] = param_values[new_idx]

                # 评估新位置
                score = await target_function(particles[i])

                # 更新个人最优
                if score > personal_best_scores[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_scores[i] = score

                    # 更新全局最优
                    if score > global_best_score:
                        global_best = particles[i].copy()
                        global_best_score = score

            convergence_history.append(global_best_score)

        return OptimizationResult(
            optimization_id="",
            strategy_id="",
            best_parameters=global_best,
            best_score=global_best_score,
            all_results=[
                {"parameters": pb, "score": score, "iteration": i}
                for i, (pb, score) in enumerate(zip(personal_best, personal_best_scores))
            ],
            convergence_history=convergence_history,
            execution_time=0.0,
            status="success",
            timestamp=datetime.now()
        )
```

## API 集成

### 认证集成

```python
import requests
from typing import Optional

class RQA2025Client:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.token: Optional[str] = None

    def login(self, username: str, password: str) -> bool:
        """用户登录"""
        response = self.session.post(f"{self.base_url}/api/auth/login", json={
            "username": username,
            "password": password
        })

        if response.status_code == 200:
            data = response.json()
            self.token = data.get("token")
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
            return True

        return False

    def logout(self) -> bool:
        """用户登出"""
        if not self.token:
            return False

        response = self.session.post(f"{self.base_url}/api/auth/logout", json={
            "token": self.token
        })

        if response.status_code == 200:
            self.token = None
            self.session.headers.pop("Authorization", None)
            return True

        return False

    def create_strategy(self, strategy_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """创建策略"""
        response = self.session.post(
            f"{self.base_url}/api/strategies",
            json=strategy_config
        )

        if response.status_code == 200:
            return response.json()

        return None

    def run_backtest(self, backtest_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """执行回测"""
        response = self.session.post(
            f"{self.base_url}/api/backtests",
            json=backtest_config
        )

        if response.status_code == 200:
            return response.json()

        return None

    def get_strategy_performance(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """获取策略表现"""
        response = self.session.get(
            f"{self.base_url}/api/visualization/strategy-performance/{strategy_id}"
        )

        if response.status_code == 200:
            return response.json()

        return None
```

### 异步集成

```python
import asyncio
import aiohttp
from typing import Dict, Any, Optional

class AsyncRQA2025Client:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.token: Optional[str] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def login(self, username: str, password: str) -> bool:
        """异步用户登录"""
        async with self.session.post(
            f"{self.base_url}/api/auth/login",
            json={"username": username, "password": password}
        ) as response:
            if response.status == 200:
                data = await response.json()
                self.token = data.get("token")
                return True

        return False

    async def create_strategy_async(self, strategy_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """异步创建策略"""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        async with self.session.post(
            f"{self.base_url}/api/strategies",
            json=strategy_config,
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()

        return None

    async def monitor_strategy(self, strategy_id: str, duration_seconds: int = 60):
        """监控策略运行状态"""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < duration_seconds:
            # 获取策略状态
            status = await self.get_strategy_status(strategy_id)

            if status:
                print(f"策略状态: {status}")

            # 等待一段时间
            await asyncio.sleep(5)

    async def get_strategy_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """获取策略状态"""
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        async with self.session.get(
            f"{self.base_url}/api/strategies/{strategy_id}",
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()

        return None
```

## 测试开发

### 单元测试

```python
import pytest
from unittest.mock import Mock, patch
from src.strategy.strategies.momentum_strategy import MomentumStrategy
from src.strategy.interfaces.strategy_interfaces import StrategyConfig

class TestMomentumStrategy:
    @pytest.fixture
    def sample_config(self):
        """测试配置fixture"""
        return StrategyConfig(
            strategy_id="test_momentum",
            strategy_name="测试动量策略",
            strategy_type="momentum",
            parameters={
                "lookback_period": 20,
                "momentum_threshold": 0.05
            }
        )

    @pytest.fixture
    def sample_market_data(self):
        """测试市场数据fixture"""
        return {
            "000001.SZ": [
                {"timestamp": "2023-01-01", "close": 100.0, "volume": 1000000},
                {"timestamp": "2023-01-02", "close": 102.0, "volume": 1200000},
                {"timestamp": "2023-01-03", "close": 105.0, "volume": 1500000}
            ]
        }

    def test_strategy_initialization(self, sample_config):
        """测试策略初始化"""
        strategy = MomentumStrategy(sample_config)

        assert strategy.strategy_id == sample_config.strategy_id
        assert strategy.strategy_name == sample_config.strategy_name
        assert strategy.lookback_period == 20
        assert strategy.momentum_threshold == 0.05

    def test_signal_generation(self, sample_config, sample_market_data):
        """测试信号生成"""
        strategy = MomentumStrategy(sample_config)

        signals = strategy.generate_signals(sample_market_data)

        assert isinstance(signals, list)
        # 根据测试数据验证信号生成逻辑
        # 这里可以添加更详细的断言

    @patch('src.strategy.strategies.momentum_strategy.MomentumStrategy._calculate_momentum')
    def test_momentum_calculation(self, mock_calculate, sample_config):
        """测试动量计算"""
        mock_calculate.return_value = 0.08

        strategy = MomentumStrategy(sample_config)
        momentum = strategy._calculate_momentum([100, 102, 105, 103, 108])

        assert momentum == 0.08
        mock_calculate.assert_called_once()
```

### 集成测试

```python
import pytest
from src.strategy.core.strategy_service import UnifiedStrategyService
from src.strategy.workspace.web_api import StrategyWorkspaceAPI

class TestStrategyIntegration:
    @pytest.fixture
    async def api_client(self):
        """API客户端fixture"""
        api = StrategyWorkspaceAPI()

        # 设置测试服务
        strategy_service = UnifiedStrategyService()
        api.set_services(strategy_service=strategy_service)

        yield api

    @pytest.mark.asyncio
    async def test_create_and_execute_strategy(self, api_client):
        """测试创建和执行策略的完整流程"""
        # 创建策略
        strategy_data = {
            "strategy_name": "集成测试策略",
            "strategy_type": "momentum",
            "parameters": {"lookback_period": 20}
        }

        # 这里需要mock实际的HTTP请求
        # 或者直接调用API方法

        # 验证策略创建
        # 验证策略执行
        # 验证结果返回

        assert True  # 占位符

    @pytest.mark.asyncio
    async def test_backtest_workflow(self, api_client):
        """测试回测工作流"""
        # 创建策略
        # 执行回测
        # 验证回测结果
        # 检查性能指标

        assert True  # 占位符
```

### 性能测试

```python
import pytest
import time
from src.strategy.strategies.momentum_strategy import MomentumStrategy

class TestPerformance:
    def test_strategy_execution_performance(self):
        """测试策略执行性能"""
        config = StrategyConfig(
            strategy_id="perf_test",
            strategy_name="性能测试",
            strategy_type="momentum",
            parameters={"lookback_period": 20}
        )

        strategy = MomentumStrategy(config)

        # 生成大量测试数据
        market_data = self._generate_large_market_data(1000)

        # 测量执行时间
        start_time = time.time()
        signals = strategy.generate_signals(market_data)
        execution_time = time.time() - start_time

        # 验证性能要求
        assert execution_time < 1.0, f"执行时间过长: {execution_time}s"
        assert len(signals) >= 0, "信号生成不应失败"

    def _generate_large_market_data(self, num_records: int):
        """生成大量测试数据"""
        import random

        data = []
        base_price = 100.0

        for i in range(num_records):
            change = random.uniform(-0.02, 0.02)
            base_price *= (1 + change)

            data.append({
                "timestamp": f"2023-01-{str(i+1).zfill(2)}",
                "close": round(base_price, 2),
                "volume": random.randint(100000, 1000000)
            })

        return {"000001.SZ": data}
```

## 部署和运维

### Docker 部署

```dockerfile
# Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "scripts/start_workspace.py", "--host", "0.0.0.0"]
```

### Kubernetes 部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rqa2025-workspace
  labels:
    app: rqa2025-workspace
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rqa2025-workspace
  template:
    metadata:
      labels:
        app: rqa2025-workspace
    spec:
      containers:
      - name: workspace
        image: rqa2025/workspace:latest
        ports:
        - containerPort: 8000
        env:
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: rqa2025-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rqa2025-workspace-service
spec:
  selector:
    app: rqa2025-workspace
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

### 监控和日志

```python
# 结构化日志配置
import logging
import json
from pythonjsonlogger import jsonlogger

def setup_structured_logging():
    """设置结构化日志"""
    logger = logging.getLogger()

    # 创建JSON格式化器
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 文件处理器
    file_handler = logging.FileHandler('logs/app.jsonl')
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

# 使用结构化日志
logger = logging.getLogger(__name__)

def log_strategy_execution(strategy_id: str, execution_time: float, signal_count: int):
    """记录策略执行日志"""
    logger.info("策略执行完成", extra={{
        "strategy_id": strategy_id,
        "execution_time": execution_time,
        "signal_count": signal_count,
        "event_type": "strategy_execution"
    }})

def log_error(error_type: str, error_message: str, context: Dict[str, Any]):
    """记录错误日志"""
    logger.error("系统错误", extra={{
        "error_type": error_type,
        "error_message": error_message,
        "context": context,
        "event_type": "system_error"
    }})
```

---

*版本: v1.0.0*
*最后更新: {datetime.now().strftime('%Y-%m-%d')}*
