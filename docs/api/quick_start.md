# RQA2025 API 快速开始指南

## 简介

本指南将帮助您在5分钟内快速上手 RQA2025 API，完成从认证到执行第一个策略的完整流程。

## 前提条件

- Python 3.8+ 或 Node.js 14+
- 有效的 API 访问凭证
- 网络连接

## 第一步：获取访问令牌

### Python

```python
import requests

# API基础URL
BASE_URL = "https://api.rqa2025.com/v1"

# 登录获取Token
response = requests.post(
    f"{BASE_URL}/auth/login",
    json={
        "username": "your_username",
        "password": "your_password"
    }
)

if response.status_code == 200:
    data = response.json()["data"]
    access_token = data["access_token"]
    print(f"✅ 登录成功！Token有效期: {data['expires_in']}秒")
else:
    print(f"❌ 登录失败: {response.json()}")
```

### JavaScript

```javascript
const axios = require('axios');

const BASE_URL = 'https://api.rqa2025.com/v1';

async function login() {
  try {
    const response = await axios.post(`${BASE_URL}/auth/login`, {
      username: 'your_username',
      password: 'your_password'
    });
    
    const { access_token, expires_in } = response.data.data;
    console.log(`✅ 登录成功！Token有效期: ${expires_in}秒`);
    return access_token;
  } catch (error) {
    console.error('❌ 登录失败:', error.response?.data || error.message);
    throw error;
  }
}
```

## 第二步：创建您的第一个策略

### Python

```python
import requests

# 使用获取到的Token
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

# 创建双均线策略
strategy_data = {
    "name": "我的第一个策略",
    "description": "简单的双均线交叉策略",
    "type": "technical",
    "parameters": {
        "short_period": 5,
        "long_period": 20,
        "symbol": "000001.SZ"
    }
}

response = requests.post(
    f"{BASE_URL}/strategies",
    headers=headers,
    json=strategy_data
)

if response.status_code == 200:
    strategy = response.json()["data"]
    strategy_id = strategy["id"]
    print(f"✅ 策略创建成功！ID: {strategy_id}")
else:
    print(f"❌ 策略创建失败: {response.json()}")
```

### JavaScript

```javascript
async function createStrategy(token) {
  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  };
  
  const strategyData = {
    name: '我的第一个策略',
    description: '简单的双均线交叉策略',
    type: 'technical',
    parameters: {
      short_period: 5,
      long_period: 20,
      symbol: '000001.SZ'
    }
  };
  
  try {
    const response = await axios.post(
      `${BASE_URL}/strategies`,
      strategyData,
      { headers }
    );
    
    const strategy = response.data.data;
    console.log(`✅ 策略创建成功！ID: ${strategy.id}`);
    return strategy.id;
  } catch (error) {
    console.error('❌ 策略创建失败:', error.response?.data || error.message);
    throw error;
  }
}
```

## 第三步：执行策略回测

### Python

```python
# 执行回测
backtest_data = {
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "initial_capital": 100000,
    "commission": 0.001
}

response = requests.post(
    f"{BASE_URL}/strategies/{strategy_id}/backtest",
    headers=headers,
    json=backtest_data
)

if response.status_code == 200:
    backtest = response.json()["data"]
    backtest_id = backtest["backtest_id"]
    print(f"✅ 回测启动成功！ID: {backtest_id}")
    print(f"⏳ 状态: {backtest['status']}")
else:
    print(f"❌ 回测启动失败: {response.json()}")
```

### JavaScript

```javascript
async function runBacktest(token, strategyId) {
  const headers = {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  };
  
  const backtestData = {
    start_date: '2025-01-01',
    end_date: '2025-12-31',
    initial_capital: 100000,
    commission: 0.001
  };
  
  try {
    const response = await axios.post(
      `${BASE_URL}/strategies/${strategyId}/backtest`,
      backtestData,
      { headers }
    );
    
    const backtest = response.data.data;
    console.log(`✅ 回测启动成功！ID: ${backtest.backtest_id}`);
    console.log(`⏳ 状态: ${backtest.status}`);
    return backtest.backtest_id;
  } catch (error) {
    console.error('❌ 回测启动失败:', error.response?.data || error.message);
    throw error;
  }
}
```

## 第四步：查询回测结果

### Python

```python
import time

# 轮询回测结果
def wait_for_backtest(backtest_id, timeout=300):
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = requests.get(
            f"{BASE_URL}/backtests/{backtest_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]
            
            if status == "completed":
                print("✅ 回测完成！")
                print(f"📊 总收益率: {result['performance']['total_return']}%")
                print(f"📊 夏普比率: {result['performance']['sharpe_ratio']}")
                print(f"📊 最大回撤: {result['performance']['max_drawdown']}%")
                return result
            elif status == "failed":
                print(f"❌ 回测失败: {result.get('error', '未知错误')}")
                return None
            else:
                print(f"⏳ 回测进行中... {result.get('progress', 0)}%")
        
        time.sleep(5)
    
    print("⏰ 等待超时")
    return None

# 执行等待
result = wait_for_backtest(backtest_id)
```

## 完整示例代码

### Python完整示例

```python
import requests
import time

class RQA2025Client:
    def __init__(self, base_url="https://api.rqa2025.com/v1"):
        self.base_url = base_url
        self.token = None
    
    def login(self, username, password):
        """登录获取Token"""
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password}
        )
        
        if response.status_code == 200:
            self.token = response.json()["data"]["access_token"]
            return True
        return False
    
    def get_headers(self):
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def create_strategy(self, name, description, strategy_type, parameters):
        """创建策略"""
        response = requests.post(
            f"{self.base_url}/strategies",
            headers=self.get_headers(),
            json={
                "name": name,
                "description": description,
                "type": strategy_type,
                "parameters": parameters
            }
        )
        return response.json()["data"] if response.status_code == 200 else None
    
    def run_backtest(self, strategy_id, start_date, end_date, initial_capital):
        """执行回测"""
        response = requests.post(
            f"{self.base_url}/strategies/{strategy_id}/backtest",
            headers=self.get_headers(),
            json={
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital
            }
        )
        return response.json()["data"] if response.status_code == 200 else None
    
    def get_backtest_result(self, backtest_id):
        """获取回测结果"""
        response = requests.get(
            f"{self.base_url}/backtests/{backtest_id}",
            headers=self.get_headers()
        )
        return response.json()["data"] if response.status_code == 200 else None

# 使用示例
def main():
    # 初始化客户端
    client = RQA2025Client()
    
    # 登录
    if not client.login("your_username", "your_password"):
        print("❌ 登录失败")
        return
    
    print("✅ 登录成功")
    
    # 创建策略
    strategy = client.create_strategy(
        name="双均线策略",
        description="基于5日和20日均线的交叉策略",
        strategy_type="technical",
        parameters={
            "short_period": 5,
            "long_period": 20,
            "symbol": "000001.SZ"
        }
    )
    
    if not strategy:
        print("❌ 策略创建失败")
        return
    
    print(f"✅ 策略创建成功: {strategy['id']}")
    
    # 执行回测
    backtest = client.run_backtest(
        strategy_id=strategy['id'],
        start_date="2025-01-01",
        end_date="2025-12-31",
        initial_capital=100000
    )
    
    if not backtest:
        print("❌ 回测启动失败")
        return
    
    print(f"✅ 回测启动成功: {backtest['backtest_id']}")
    
    # 等待回测完成
    print("⏳ 等待回测完成...")
    while True:
        result = client.get_backtest_result(backtest['backtest_id'])
        if not result:
            print("❌ 获取回测结果失败")
            return
        
        if result['status'] == 'completed':
            print("✅ 回测完成！")
            print(f"📊 总收益率: {result['performance']['total_return']}%")
            print(f"📊 夏普比率: {result['performance']['sharpe_ratio']}")
            print(f"📊 最大回撤: {result['performance']['max_drawdown']}%")
            break
        elif result['status'] == 'failed':
            print(f"❌ 回测失败: {result.get('error', '未知错误')}")
            break
        else:
            print(f"⏳ 回测进行中... {result.get('progress', 0)}%")
            time.sleep(5)

if __name__ == "__main__":
    main()
```

### JavaScript完整示例

```javascript
const axios = require('axios');

class RQA2025Client {
  constructor(baseUrl = 'https://api.rqa2025.com/v1') {
    this.baseUrl = baseUrl;
    this.token = null;
  }
  
  async login(username, password) {
    try {
      const response = await axios.post(`${this.baseUrl}/auth/login`, {
        username,
        password
      });
      this.token = response.data.data.access_token;
      return true;
    } catch (error) {
      console.error('登录失败:', error.message);
      return false;
    }
  }
  
  getHeaders() {
    return {
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json'
    };
  }
  
  async createStrategy(name, description, type, parameters) {
    try {
      const response = await axios.post(
        `${this.baseUrl}/strategies`,
        { name, description, type, parameters },
        { headers: this.getHeaders() }
      );
      return response.data.data;
    } catch (error) {
      console.error('创建策略失败:', error.message);
      return null;
    }
  }
  
  async runBacktest(strategyId, startDate, endDate, initialCapital) {
    try {
      const response = await axios.post(
        `${this.baseUrl}/strategies/${strategyId}/backtest`,
        { start_date: startDate, end_date: endDate, initial_capital: initialCapital },
        { headers: this.getHeaders() }
      );
      return response.data.data;
    } catch (error) {
      console.error('启动回测失败:', error.message);
      return null;
    }
  }
  
  async getBacktestResult(backtestId) {
    try {
      const response = await axios.get(
        `${this.baseUrl}/backtests/${backtestId}`,
        { headers: this.getHeaders() }
      );
      return response.data.data;
    } catch (error) {
      console.error('获取回测结果失败:', error.message);
      return null;
    }
  }
}

// 使用示例
async function main() {
  const client = new RQA2025Client();
  
  // 登录
  if (!await client.login('your_username', 'your_password')) {
    console.log('❌ 登录失败');
    return;
  }
  
  console.log('✅ 登录成功');
  
  // 创建策略
  const strategy = await client.createStrategy(
    '双均线策略',
    '基于5日和20日均线的交叉策略',
    'technical',
    {
      short_period: 5,
      long_period: 20,
      symbol: '000001.SZ'
    }
  );
  
  if (!strategy) {
    console.log('❌ 策略创建失败');
    return;
  }
  
  console.log(`✅ 策略创建成功: ${strategy.id}`);
  
  // 执行回测
  const backtest = await client.runBacktest(
    strategy.id,
    '2025-01-01',
    '2025-12-31',
    100000
  );
  
  if (!backtest) {
    console.log('❌ 回测启动失败');
    return;
  }
  
  console.log(`✅ 回测启动成功: ${backtest.backtest_id}`);
  
  // 等待回测完成
  console.log('⏳ 等待回测完成...');
  while (true) {
    const result = await client.getBacktestResult(backtest.backtest_id);
    if (!result) {
      console.log('❌ 获取回测结果失败');
      return;
    }
    
    if (result.status === 'completed') {
      console.log('✅ 回测完成！');
      console.log(`📊 总收益率: ${result.performance.total_return}%`);
      console.log(`📊 夏普比率: ${result.performance.sharpe_ratio}`);
      console.log(`📊 最大回撤: ${result.performance.max_drawdown}%`);
      break;
    } else if (result.status === 'failed') {
      console.log(`❌ 回测失败: ${result.error || '未知错误'}`);
      break;
    } else {
      console.log(`⏳ 回测进行中... ${result.progress || 0}%`);
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
}

main().catch(console.error);
```

## 常见问题

### Q: 如何处理Token过期？

**A**: Token过期后需要重新登录获取新的Token。建议实现自动刷新机制：

```python
import time

class TokenManager:
    def __init__(self, client):
        self.client = client
        self.token = None
        self.expires_at = 0
    
    def get_valid_token(self):
        # 如果Token即将过期（提前5分钟），则刷新
        if time.time() > self.expires_at - 300:
            self.refresh_token()
        return self.token
    
    def refresh_token(self):
        # 实现Token刷新逻辑
        pass
```

### Q: 如何处理API限流？

**A**: 当遇到429错误时，应实现指数退避重试：

```python
import time
import random

def retry_with_backoff(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait_time = (2 ** i) + random.uniform(0, 1)
                print(f"⏳ 遇到限流，等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("达到最大重试次数")
```

### Q: 如何调试API请求？

**A**: 建议开启详细日志记录：

```python
import logging
import http.client as http_client

# 启用HTTP调试日志
http_client.HTTPConnection.debuglevel = 1
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True
```

## 下一步

- 📚 阅读完整的 [API参考文档](api_reference.md)
- 🔍 查看 [示例代码](../../examples/api_examples.py)
- 💬 加入开发者社区获取支持

---

**快速开始完成！** 🎉

您已经成功完成了RQA2025 API的快速开始指南。现在可以开始构建您的量化交易策略了！
